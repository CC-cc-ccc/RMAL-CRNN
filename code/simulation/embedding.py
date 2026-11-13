import math, random, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
warnings.filterwarnings('ignore')
try:
    import umap
except ImportError:
    raise ImportError('请先安装 UMAP:  pip install umap-learn')
SEED = 2026
NUM_CLASS = 10
N_PER_CLASS = 90
T_LEN = 64
FULL_DIM = 192
D_VIEW = 32
VIEWS = [(0, 32), (32, 64), (64, 96), (96, 128), (128, 160), (160, 192)]
FEATURE_NAMES = ['HHT', 'LOFAR', 'FFT', 'WPT', 'HHSA', 'MFCC']
V_NUM = len(VIEWS)
PROJ_OUT = 64
N_HEADS = 4
BETA_INIT = 0.5
DROPOUT = 0.5
LRR_RANK = 8
LRR_LAM_INIT, LRR_ALPHA_INIT = (0.1, 0.3)
LRR_LAM_NUC, LRR_LAM_SPS = (0.001, 0.001)
EPOCHS = 90
BATCH_SIZE = 32
LR_MAIN = 0.001
WEIGHT_DECAY = 0.0001
LABEL_SMOOTHING = 0.1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT_PREFIX = 'umap_three_scenarios_full'

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def make_clean_dataset(rng):
    k_latent = 16
    protos = rng.standard_normal((NUM_CLASS, k_latent))
    protos /= np.linalg.norm(protos, axis=1, keepdims=True)
    protos *= 3.0
    block_W = [rng.standard_normal((k_latent, D_VIEW)) * 0.6 for _ in range(V_NUM)]
    block_b = [rng.standard_normal(D_VIEW) * 0.3 for _ in range(V_NUM)]
    N = NUM_CLASS * N_PER_CLASS
    X = np.zeros((N, T_LEN, FULL_DIM), dtype=np.float32)
    y = np.zeros(N, dtype=np.int64)
    idx = 0
    for c in range(NUM_CLASS):
        for _ in range(N_PER_CLASS):
            z0 = protos[c] + rng.standard_normal(k_latent) * 0.35
            phase = rng.uniform(0, 2 * np.pi, size=k_latent)
            freq = rng.uniform(0.5, 2.0, size=k_latent)
            tgrid = np.linspace(0, 1, T_LEN)[:, None]
            z_t = z0[None, :] + 0.25 * np.sin(2 * np.pi * freq[None, :] * tgrid + phase[None, :])
            for b, (s, e) in enumerate(VIEWS):
                raw = z_t @ block_W[b] + block_b[b]
                feat = np.tanh(raw) + 0.05 * rng.standard_normal(raw.shape)
                X[idx, :, s:e] = feat.astype(np.float32)
            y[idx] = c
            idx += 1
    return (X, y)

def corrupt_heavy_tail(x, rng, frac=0.08, scale=6.0):
    out = x.copy()
    mask = rng.random(out.shape) < frac
    spikes = rng.standard_t(df=2, size=out.shape) * scale
    out[mask] += spikes[mask].astype(np.float32)
    return out

def corrupt_lowrank_sparse(x, rng, rank=2, lr_amp=2.5, sp_frac=0.04, sp_amp=8.0):
    out = x.copy()
    T, D = (out.shape[1], out.shape[2])
    for i in range(out.shape[0]):
        U = rng.standard_normal((T, rank))
        V = rng.standard_normal((rank, D))
        low = U @ V / math.sqrt(rank) * lr_amp
        out[i] += low.astype(np.float32)
    sp_mask = rng.random(out.shape) < sp_frac
    sp_val = rng.standard_normal(out.shape) * sp_amp
    out[sp_mask] += sp_val[sp_mask].astype(np.float32)
    return out

def corrupt_warp_missing(x, rng, drop_frac=0.2, warp=0.5):
    out = x.copy()
    D = out.shape[2]
    for s, e in VIEWS:
        A = rng.standard_normal((D_VIEW, D_VIEW))
        Q, _ = np.linalg.qr(A)
        R = (1 - warp) * np.eye(D_VIEW) + warp * Q
        out[:, :, s:e] = out[:, :, s:e] @ R.astype(np.float32)
    drop = rng.random(out.shape) < drop_frac
    out[drop] = 0.0
    out += (rng.standard_normal(out.shape) * 0.4).astype(np.float32)
    return out

def make_combined_complex(X_clean, y, rng):
    N = X_clean.shape[0]
    perm = rng.permutation(N)
    g1, g2 = (perm[:N // 3], perm[N // 3:2 * N // 3])
    g3 = perm[2 * N // 3:]
    X_cx = X_clean.copy()
    X_cx[g1] = corrupt_heavy_tail(X_clean[g1], rng)
    X_cx[g2] = corrupt_lowrank_sparse(X_clean[g2], rng)
    X_cx[g3] = corrupt_warp_missing(X_clean[g3], rng)
    scenario = np.zeros(N, dtype=np.int64)
    scenario[g1], scenario[g2], scenario[g3] = (0, 1, 2)
    return (X_cx, scenario)

class StudentTRobustGate(nn.Module):

    def __init__(self, dim, eps=1e-06, nu_init=100.0):
        super().__init__()
        self.dim, self.eps = (dim, eps)
        self.center = nn.Parameter(torch.zeros(dim))
        self.log_scale = nn.Parameter(torch.zeros(dim))
        self.log_inv_nu = nn.Parameter(torch.full((dim,), math.log(1.0 / float(nu_init))))

    def _nu(self):
        inv_nu = torch.exp(self.log_inv_nu).clamp(min=0.0001, max=1.0)
        return (1.0 / inv_nu).clamp(min=1.0, max=10000.0)

    def forward(self, z):
        s = torch.exp(self.log_scale).clamp(min=self.eps)
        nu = self._nu()
        r = (z - self.center) / s
        return z * ((nu + 1.0) / (nu + r.pow(2)))

class SphericalTangentEmbedding(nn.Module):

    def __init__(self, dim=D_VIEW, eps=1e-06):
        super().__init__()
        self.dim, self.eps = (dim, eps)
        self.p = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, z):
        D = z.shape[-1]
        z_flat = z.reshape(-1, D)
        p_unit = self.p / self.p.norm().clamp(min=self.eps)
        parallel = (z_flat * p_unit).sum(dim=-1, keepdim=True)
        tangent = z_flat - parallel * p_unit
        t_norm = tangent.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        out = math.sqrt(D) * tangent / t_norm + self.bias
        return out.reshape(z.shape)

class QRInitLinear(nn.Module):

    def __init__(self, d_in, d_out):
        super().__init__()
        W = torch.randn(d_in, d_out)
        Q, _ = torch.linalg.qr(W, mode='reduced')
        self.weight = nn.Parameter(Q)

    def forward(self, x):
        return x @ self.weight

class LogMapTangentAttention(nn.Module):

    def __init__(self, d_model=D_VIEW, n_heads=N_HEADS, V=V_NUM, eps=1e-06):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model, self.n_heads = (d_model, n_heads)
        self.d_head, self.V, self.eps = (d_model // n_heads, V, eps)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        bp = torch.randn(n_heads, self.d_head) * 0.3
        bp[:, 0] = bp[:, 0] + 1.0
        self.base_point = nn.Parameter(bp)
        self.log_beta = nn.Parameter(torch.full((n_heads,), math.log(BETA_INIT)))

    def _log_map_unit(self, x_unit, p_unit):
        inner = (x_unit * p_unit).sum(dim=-1, keepdim=True)
        return x_unit - inner * p_unit

    def _log_map_general(self, x, p_unit):
        inner = (x * p_unit).sum(dim=-1, keepdim=True)
        return x - inner * p_unit

    def forward(self, tokens):
        B, T, V, D = tokens.shape
        x = tokens.reshape(B * T, V, D)
        qkv = self.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = [t.reshape(B * T, V, self.n_heads, self.d_head).transpose(1, 2) for t in qkv]
        p_unit = F.normalize(self.base_point, dim=-1, eps=self.eps)
        p_b = p_unit.view(1, self.n_heads, 1, self.d_head)
        q_n = F.normalize(q, dim=-1, eps=self.eps)
        k_n = F.normalize(k, dim=-1, eps=self.eps)
        xi_q = self._log_map_unit(q_n, p_b)
        xi_k = self._log_map_unit(k_n, p_b)
        xi_v = self._log_map_general(v, p_b)
        nq = (xi_q * xi_q).sum(dim=-1, keepdim=True)
        nk_t = (xi_k * xi_k).sum(dim=-1, keepdim=True).transpose(-1, -2)
        cross = xi_q @ xi_k.transpose(-2, -1)
        d2 = (nq + nk_t - 2.0 * cross).clamp(min=0.0)
        beta = torch.exp(self.log_beta).view(1, self.n_heads, 1, 1)
        attn = F.softmax(-beta * d2, dim=-1)
        out_tan = attn @ xi_v
        out = out_tan.transpose(1, 2).reshape(B * T, V, D)
        return self.norm(self.out_proj(out) + x).reshape(B, T, V, D)

class DiffNSLLR(nn.Module):

    def __init__(self, d_in=PROJ_OUT, rank=LRR_RANK, lam_init=LRR_LAM_INIT, alpha_init=LRR_ALPHA_INIT, eps=1e-06):
        super().__init__()
        self.d_in = d_in
        self.rank = rank
        self.eps = eps
        self.proj_u = nn.Linear(d_in, rank, bias=False)
        self.proj_v = nn.Linear(d_in, rank, bias=False)
        nn.init.orthogonal_(self.proj_u.weight)
        nn.init.orthogonal_(self.proj_v.weight)
        self.log_lam = nn.Parameter(torch.tensor(math.log(lam_init)))
        self.alpha_logit = nn.Parameter(torch.tensor(math.log(alpha_init / (1.0 - alpha_init))))
        self.refined_norm = nn.LayerNorm(d_in)
        self._cache = {}

    def _alpha(self):
        return torch.sigmoid(self.alpha_logit)

    def forward(self, feat):
        U = self.proj_u(feat)
        V = self.proj_v(feat)
        Z_raw = torch.einsum('btr,bsr->bts', U, V) / math.sqrt(self.rank)
        Z = torch.clamp(Z_raw - self.log_lam.exp(), min=0.0)
        F_ref = torch.einsum('bts,bsd->btd', Z, feat)
        F_ref = self.refined_norm(F_ref)
        self._cache = {'Z': Z, 'U': U, 'V': V}
        return feat + self._alpha() * (F_ref - feat)

    def regularization(self, lam_nuc=LRR_LAM_NUC, lam_sps=LRR_LAM_SPS):
        if not self._cache:
            return torch.zeros((), device=self.log_lam.device)
        Z, U, V = (self._cache['Z'], self._cache['U'], self._cache['V'])
        nuc_bm = 0.5 * (U.pow(2).mean() + V.pow(2).mean())
        return lam_nuc * nuc_bm + lam_sps * Z.abs().mean()

class RMALCRNN(nn.Module):

    def __init__(self, views=VIEWS, num_class=NUM_CLASS):
        super().__init__()
        self.views = views
        self.V = len(views)
        D = views[0][1] - views[0][0]
        self.d_view = D
        self.gates = nn.ModuleList([StudentTRobustGate(D) for _ in views])
        self.norms = nn.ModuleList([SphericalTangentEmbedding(D) for _ in views])
        self.view_emb = nn.Parameter(torch.zeros(self.V, D))
        nn.init.normal_(self.view_emb, std=0.02)
        self.cross_view_attn = LogMapTangentAttention(D, N_HEADS, self.V)
        self.fusion_proj = QRInitLinear(self.V * D, PROJ_OUT)
        self.lrr_block = DiffNSLLR(d_in=PROJ_OUT)
        self.cnn = nn.Sequential(nn.Conv1d(PROJ_OUT, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2), nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2))
        self.rnn = nn.LSTM(128, 128, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(p=DROPOUT)
        self.fc = nn.Linear(256, num_class)

    def forward(self, x, return_embedding=False):
        B, T, _ = x.shape
        chunks = []
        for i, (s, e) in enumerate(self.views):
            z = x[:, :, s:e]
            z = self.gates[i](z)
            z = self.norms[i](z)
            z = z + self.view_emb[i].view(1, 1, -1)
            chunks.append(z)
        tokens = torch.stack(chunks, dim=2)
        tokens = self.cross_view_attn(tokens)
        feat = tokens.reshape(B, T, self.V * self.d_view)
        feat = self.fusion_proj(feat)
        feat = self.lrr_block(feat)
        h = self.cnn(feat.permute(0, 2, 1)).permute(0, 2, 1)
        self.rnn.flatten_parameters()
        out, _ = self.rnn(h)
        out = self.drop(out)
        emb = out.mean(dim=1)
        logits = self.fc(emb)
        if return_embedding:
            return (logits, emb)
        return logits

    def lrr_regularization(self):
        return self.lrr_block.regularization()

class SeqDataset(Dataset):

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return (self.X[i], self.y[i])

def train_full(X, y):
    set_seed(SEED)
    loader = DataLoader(SeqDataset(X, y), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    net = RMALCRNN().to(DEVICE)
    opt = optim.AdamW(net.parameters(), lr=LR_MAIN, weight_decay=WEIGHT_DECAY)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    for ep in range(EPOCHS):
        net.train()
        ce_sum = 0.0
        correct = 0
        total = 0
        for bx, by in loader:
            bx, by = (bx.to(DEVICE), by.to(DEVICE))
            opt.zero_grad()
            logits = net(bx)
            ce = F.cross_entropy(logits, by, label_smoothing=LABEL_SMOOTHING)
            loss = ce + net.lrr_regularization()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            opt.step()
            ce_sum += float(ce)
            correct += (logits.argmax(1) == by).sum().item()
            total += by.size(0)
        sched.step()
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f'  [train] epoch {ep + 1:3d}/{EPOCHS}  CE={ce_sum / len(loader):.4f}  train_acc={correct / total:.4f}')
    return net

@torch.no_grad()
def extract_embeddings(net, X):
    net.eval()
    loader = DataLoader(SeqDataset(X, np.zeros(len(X))), batch_size=128, shuffle=False)
    embs = []
    for bx, _ in loader:
        _, emb = net(bx.to(DEVICE), return_embedding=True)
        embs.append(emb.cpu().numpy())
    return np.concatenate(embs, axis=0)

def run_umap(feat, seed=SEED, n_neighbors=15, min_dist=0.1):
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, metric='euclidean', random_state=seed)
    return reducer.fit_transform(feat)

def set_journal_style():
    mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], 'font.size': 12, 'axes.linewidth': 1.1, 'axes.edgecolor': '#2b2b2b', 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'figure.dpi': 120})
PALETTE = ['#4E79A7', '#F28E2B', '#59A14F', '#E15759', '#B07AA1', '#76B7B2', '#EDC948', '#FF9DA7', '#9C755F', '#499894']

def styled_scatter(ax, emb, labels, title, panel):
    for c in range(NUM_CLASS):
        m = labels == c
        ax.scatter(emb[m, 0], emb[m, 1], s=16, c=PALETTE[c], alpha=0.85, linewidths=0.25, edgecolors='white', rasterized=True)
    ax.set_title(title, fontsize=13, pad=10, fontweight='medium')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('UMAP-1', fontsize=10.5, labelpad=4)
    ax.set_ylabel('UMAP-2', fontsize=10.5, labelpad=4)
    for s in ax.spines.values():
        s.set_color('#2b2b2b')
        s.set_linewidth(1.1)
    ax.text(-0.08, 1.06, panel, transform=ax.transAxes, fontsize=17, fontweight='bold', va='top', ha='left')
    ax.set_aspect('equal', adjustable='datalim')

def main():
    set_seed(SEED)
    set_journal_style()
    rng = np.random.default_rng(SEED)
    print('=' * 70)
    print(f'  RMAL-CRNN full  三场景 UMAP 可视化   (seed={SEED}, device={DEVICE})')
    print('=' * 70)
    print('[1/5] 生成干净数据集 ...')
    X_clean, y = make_clean_dataset(rng)
    print(f'      X_clean={X_clean.shape}, 类别数={NUM_CLASS}')
    print('[2/5] 构造三种复杂场景并合并 ...')
    X_cx, scenario = make_combined_complex(X_clean, y, rng)
    print(f'      X_complex={X_cx.shape}  (S1/S2/S3 各 ~1/3)')
    print('[3/5] 在复杂数据上训练 full 算法 ...')
    net = train_full(X_cx, y)
    print('[4/5] 提取算法后嵌入 ...')
    emb_after = extract_embeddings(net, X_cx)
    feat_clean = X_clean.mean(axis=1)
    feat_cx = X_cx.mean(axis=1)
    print('[5/5] UMAP 降维 ...')
    u_a = run_umap(feat_clean)
    u_b = run_umap(feat_cx)
    u_c = run_umap(emb_after)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4))
    styled_scatter(axes[0], u_a, y, 'Clean features (six views)', '(a)')
    styled_scatter(axes[1], u_b, y, 'Complex scenarios (S1+S2+S3)', '(b)')
    styled_scatter(axes[2], u_c, y, 'After RMAL-CRNN (full)', '(c)')
    handles = [Line2D([0], [0], marker='o', linestyle='', markerfacecolor=PALETTE[c], markeredgecolor='white', markeredgewidth=0.4, markersize=8, label=f'C{c:02d}') for c in range(NUM_CLASS)]
    fig.legend(handles=handles, loc='lower center', ncol=NUM_CLASS, frameon=False, fontsize=10.5, bbox_to_anchor=(0.5, -0.02), columnspacing=1.2, handletextpad=0.3)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.9, bottom=0.14, wspace=0.18)
    fig.savefig(f'{OUT_PREFIX}.pdf', bbox_inches='tight')
    fig.savefig(f'{OUT_PREFIX}.png', dpi=300, bbox_inches='tight')
    print(f'\n已保存:  {OUT_PREFIX}.pdf  /  {OUT_PREFIX}.png')
    plt.show()
if __name__ == '__main__':
    main()
