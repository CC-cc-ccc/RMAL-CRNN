import os, math, json, copy, random, time, warnings, csv
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from scipy import stats as sstats
warnings.filterwarnings('ignore')
DATA_ROOTS = {'0dB': 'data/features/0dB'}
SAVE_DIR = 'ablation'
os.makedirs(SAVE_DIR, exist_ok=True)
CLASSES = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09']
NUM_CLASS = len(CLASSES)
FULL_DIM = 192
FEATURE_BLOCKS = {'HHT': (0, 32), 'LOFAR': (32, 64), 'FFT': (64, 96), 'WPT': (96, 128), 'HHSA': (128, 160), 'MFCC': (160, 192)}
FEATURE_NAMES = list(FEATURE_BLOCKS.keys())
VIEWS = list(FEATURE_BLOCKS.values())
D_VIEW = 32
V_NUM = len(VIEWS)
SEEDS = [44, 46, 55, 234, 666, 989, 1919, 2002, 2008, 2026]
DATA_SPLIT_SEED = 42
TOTAL_EPOCHS = 120
SWA_EPOCHS = 30
SWA_START = TOTAL_EPOCHS - SWA_EPOCHS
BEST_TRACK_START = 30
VAL_SMOOTH = 5
BATCH_SIZE = 32
LR_MAIN = 0.001
WEIGHT_DECAY = 0.0001
DROPOUT = 0.5
LABEL_SMOOTHING = 0.1
PROJ_OUT = 64
N_HEADS = 4
BETA_INIT = 0.5
LRR_RANK = 8
LRR_LAM_INIT = 0.1
LRR_ALPHA_INIT = 0.3
LRR_LAM_NUC = 0.001
LRR_LAM_SPS = 0.001
LOG_PER_EPOCH = True
SAVE_PREDICTIONS = True
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@dataclass
class VariantConfig:
    tag: str
    use_gst: bool = True
    use_attn: bool = True
    use_nsllr: bool = True
VARIANTS = [VariantConfig('M0_full', use_gst=True, use_attn=True, use_nsllr=True), VariantConfig('M1_no_gst', use_gst=False, use_attn=True, use_nsllr=True), VariantConfig('M2_no_LMA', use_gst=True, use_attn=False, use_nsllr=True), VariantConfig('M3_no_DNSLR', use_gst=True, use_attn=True, use_nsllr=False), VariantConfig('M4_baseline', use_gst=False, use_attn=False, use_nsllr=False)]
REFERENCE_TAG = 'M0_full'

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_all_data(snr_root):
    if not os.path.exists(snr_root):
        raise RuntimeError(f'Data folder not found: {snr_root}')
    X_list, y_list = ([], [])
    for label_idx, label_name in enumerate(CLASSES):
        folder = os.path.join(snr_root, label_name)
        if not os.path.exists(folder):
            continue
        for f in sorted(os.listdir(folder)):
            if not f.endswith('.npy'):
                continue
            data = np.load(os.path.join(folder, f))
            if data.ndim != 2 or data.shape[1] != FULL_DIM:
                continue
            X_list.append(data)
            y_list.append(label_idx)
    if not X_list:
        raise RuntimeError(f'No .npy files under {snr_root}')
    max_len = max((x.shape[0] for x in X_list))
    X = np.zeros((len(X_list), max_len, FULL_DIM), dtype='float32')
    for i, x in enumerate(X_list):
        X[i, :x.shape[0], :] = x
    return (X, np.array(y_list))

def split_data(X, y, seed=42):
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=seed, stratify=y_tmp)
    return ((X_tr, y_tr), (X_va, y_va), (X_te, y_te))

class SeqDataset(Dataset):

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return (self.X[i], self.y[i])

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

    @torch.no_grad()
    def diagnostics(self):
        nu = self._nu()
        return {'stt_nu_median': float(nu.median()), 'stt_nu_min': float(nu.min()), 'stt_scale_median': float(torch.exp(self.log_scale).median())}

class IdentityGate(nn.Module):

    def __init__(self, dim):
        super().__init__()

    def forward(self, z):
        return z

    @torch.no_grad()
    def diagnostics(self):
        return {}

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

    @torch.no_grad()
    def diagnostics(self):
        return {'ste_basepoint_l2': float(self.p.norm())}

class LayerNormFallback(nn.Module):

    def __init__(self, dim, eps=1e-06):
        super().__init__()
        self.ln = nn.LayerNorm(dim, eps=eps)

    def forward(self, z):
        return self.ln(z)

    @torch.no_grad()
    def diagnostics(self):
        return {}

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

    @torch.no_grad()
    def diagnostics(self):
        beta = torch.exp(self.log_beta)
        p_norm = self.base_point.norm(dim=-1)
        return {'attn_beta_median': float(beta.median()), 'attn_base_l2_mean': float(p_norm.mean()), 'attn_base_l2_std': float(p_norm.std()) if p_norm.numel() > 1 else 0.0}

class IdentityAttention(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, tokens):
        return tokens

    @torch.no_grad()
    def diagnostics(self):
        return {}

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

    @torch.no_grad()
    def diagnostics(self):
        d = {'lrr_alpha': float(self._alpha())}
        if self._cache:
            d['lrr_Z_density'] = float((self._cache['Z'] > 1e-06).float().mean())
        return d

class IdentityNSLLR(nn.Module):

    def __init__(self):
        super().__init__()
        self.register_buffer('_dummy', torch.zeros(1))

    def forward(self, feat):
        return feat

    def regularization(self, *args, **kwargs):
        return torch.zeros((), device=self._dummy.device)

    @torch.no_grad()
    def diagnostics(self):
        return {}

def _make_gate(use, dim):
    return StudentTRobustGate(dim) if use else IdentityGate(dim)

def _make_norm(use, dim):
    return SphericalTangentEmbedding(dim) if use else LayerNormFallback(dim)

def _make_attention(use, d_model, n_heads, V):
    return LogMapTangentAttention(d_model, n_heads, V) if use else IdentityAttention()

def _make_fusion(d_in, d_out):
    return QRInitLinear(d_in, d_out)

def _make_nsllr(use, d_in):
    return DiffNSLLR(d_in=d_in) if use else IdentityNSLLR()

class RMALCRNN(nn.Module):

    def __init__(self, cfg: VariantConfig, views=VIEWS, num_class=NUM_CLASS):
        super().__init__()
        self.cfg = cfg
        self.views = views
        self.V = len(views)
        D = views[0][1] - views[0][0]
        self.d_view = D
        self.gates = nn.ModuleList([_make_gate(cfg.use_gst, D) for _ in views])
        self.norms = nn.ModuleList([_make_norm(cfg.use_gst, D) for _ in views])
        if cfg.use_attn:
            self.view_emb = nn.Parameter(torch.zeros(self.V, D))
            nn.init.normal_(self.view_emb, std=0.02)
        else:
            self.register_buffer('view_emb', torch.zeros(self.V, D))
        self.cross_view_attn = _make_attention(cfg.use_attn, D, N_HEADS, self.V)
        self.fusion_proj = _make_fusion(self.V * D, PROJ_OUT)
        self.lrr_block = _make_nsllr(cfg.use_nsllr, PROJ_OUT)
        self.cnn = nn.Sequential(nn.Conv1d(PROJ_OUT, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2), nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2))
        self.rnn = nn.LSTM(128, 128, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(p=DROPOUT)
        self.fc = nn.Linear(256, num_class)

    def forward(self, x):
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
        return self.fc(emb)

    def lrr_regularization(self):
        return self.lrr_block.regularization()

    @torch.no_grad()
    def diagnostics(self):
        d = {}
        for i, name in enumerate(FEATURE_NAMES):
            for k, v in self.gates[i].diagnostics().items():
                d[f'g_{name}_{k}'] = v
            for k, v in self.norms[i].diagnostics().items():
                d[f'n_{name}_{k}'] = v
        for k, v in self.cross_view_attn.diagnostics().items():
            d[k] = v
        for k, v in self.lrr_block.diagnostics().items():
            d[k] = v
        return d

class BaselineCRNN(nn.Module):

    def __init__(self, num_class=NUM_CLASS):
        super().__init__()
        self.global_ln = nn.LayerNorm(FULL_DIM)
        self.input_proj = nn.Linear(FULL_DIM, PROJ_OUT)
        self.cnn = nn.Sequential(nn.Conv1d(PROJ_OUT, 64, kernel_size=7, padding=3), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2), nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2))
        self.rnn = nn.LSTM(128, 128, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(p=DROPOUT)
        self.fc = nn.Linear(256, num_class)
        self.register_buffer('_dummy', torch.zeros(1))

    def forward(self, x):
        x = self.global_ln(x)
        feat = self.input_proj(x)
        h = self.cnn(feat.permute(0, 2, 1)).permute(0, 2, 1)
        self.rnn.flatten_parameters()
        out, _ = self.rnn(h)
        out = self.drop(out)
        emb = out.mean(dim=1)
        return self.fc(emb)

    def lrr_regularization(self):
        return torch.zeros((), device=self._dummy.device)

    @torch.no_grad()
    def diagnostics(self):
        return {}

class SWACollector:

    def __init__(self, model):
        self.swa = {n: torch.zeros_like(p.data) for n, p in model.named_parameters() if p.requires_grad}
        self.n = 0

    @torch.no_grad()
    def accumulate(self, model):
        self.n += 1
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.swa:
                self.swa[n] += (p.data - self.swa[n]) / self.n

    @torch.no_grad()
    def apply_to(self, model):
        if self.n == 0:
            return
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.swa:
                p.data.copy_(self.swa[n])

def update_bn(model, loader, device):
    has_bn = any((isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)) for m in model.modules()))
    if not has_bn:
        return
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.reset_running_stats()
            m.momentum = None
    model.train()
    with torch.no_grad():
        for x, _ in loader:
            model(x.to(device))
    model.eval()

def evaluate(model, loader):
    model.eval()
    preds, trues = ([], [])
    with torch.no_grad():
        for bx, by in loader:
            logits = model(bx.to(DEVICE))
            preds.extend(logits.argmax(1).cpu().numpy())
            trues.extend(by.numpy())
    return (np.array(trues), np.array(preds))

def train_one(splits, seed, cfg, verbose=False):
    set_seed(seed)
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = splits
    tr_loader = DataLoader(SeqDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    va_loader = DataLoader(SeqDataset(X_va, y_va), batch_size=BATCH_SIZE, shuffle=False)
    te_loader = DataLoader(SeqDataset(X_te, y_te), batch_size=BATCH_SIZE, shuffle=False)
    if cfg.tag == 'M4_baseline':
        net = BaselineCRNN().to(DEVICE)
    else:
        net = RMALCRNN(cfg=cfg).to(DEVICE)
    n_params = sum((p.numel() for p in net.parameters()))
    n_train = sum((p.numel() for p in net.parameters() if p.requires_grad))
    opt = optim.AdamW(net.parameters(), lr=LR_MAIN, weight_decay=WEIGHT_DECAY)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=TOTAL_EPOCHS)
    swa = SWACollector(net)
    val_acc_hist, train_acc_hist, train_ce_hist, train_reg_hist = ([], [], [], [])
    best_smooth_va = 0.0
    best_wts = None
    best_epoch = -1
    t_train_start = time.time()
    for epoch in range(TOTAL_EPOCHS):
        net.train()
        ce_sum = 0.0
        reg_sum = 0.0
        nb = 0
        correct = 0
        total = 0
        for bx, by in tr_loader:
            bx, by = (bx.to(DEVICE), by.to(DEVICE))
            opt.zero_grad()
            logits = net(bx)
            ce = F.cross_entropy(logits, by, label_smoothing=LABEL_SMOOTHING)
            reg = net.lrr_regularization()
            loss = ce + reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            opt.step()
            ce_sum += float(ce)
            reg_sum += float(reg)
            nb += 1
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                correct += (preds == by).sum().item()
                total += by.size(0)
        sched.step()
        if epoch >= SWA_START:
            swa.accumulate(net)
        vt, vp = evaluate(net, va_loader)
        va_acc = accuracy_score(vt, vp)
        val_acc_hist.append(float(va_acc))
        train_acc_hist.append(float(correct / max(total, 1)))
        train_ce_hist.append(float(ce_sum / max(nb, 1)))
        train_reg_hist.append(float(reg_sum / max(nb, 1)))
        if epoch >= BEST_TRACK_START and len(val_acc_hist) >= VAL_SMOOTH:
            smooth_va = float(np.mean(val_acc_hist[-VAL_SMOOTH:]))
            if smooth_va > best_smooth_va:
                best_smooth_va = smooth_va
                best_wts = copy.deepcopy(net.state_dict())
                best_epoch = epoch + 1
    swa.apply_to(net)
    update_bn(net, tr_loader, DEVICE)
    swa_state = copy.deepcopy(net.state_dict())
    vt, vp = evaluate(net, va_loader)
    swa_va = accuracy_score(vt, vp)
    final_va = swa_va
    selector = 'swa'
    if best_wts is not None:
        net.load_state_dict(best_wts)
        update_bn(net, tr_loader, DEVICE)
        vt, vp = evaluate(net, va_loader)
        bw_va = accuracy_score(vt, vp)
        if bw_va >= swa_va:
            final_va = bw_va
            selector = 'best_wts'
        else:
            net.load_state_dict(swa_state)
            update_bn(net, tr_loader, DEVICE)
            best_epoch = -1
    train_time_s = time.time() - t_train_start
    net.eval()
    with torch.no_grad():
        for bx, _ in va_loader:
            net(bx.to(DEVICE))
            break
    tt, tp = evaluate(net, te_loader)
    test_acc = accuracy_score(tt, tp)
    macro_f1 = f1_score(tt, tp, average='macro', zero_division=0)
    per_class_f1 = f1_score(tt, tp, average=None, labels=list(range(NUM_CLASS)), zero_division=0).tolist()
    cm = confusion_matrix(tt, tp, labels=list(range(NUM_CLASS)))
    result = {'seed': int(seed), 'variant': cfg.tag, 'acc': float(test_acc), 'val_acc': float(final_va), 'selector': selector, 'f1': float(macro_f1), 'per_class_f1': per_class_f1, 'diagnostics': net.diagnostics(), 'n_params': int(n_params), 'n_trainable': int(n_train), 'train_time_s': float(train_time_s), 'best_epoch': int(best_epoch), 'final_train_acc': float(train_acc_hist[-1]) if train_acc_hist else float('nan')}
    if LOG_PER_EPOCH:
        result['training_curves'] = {'val_acc': val_acc_hist, 'train_acc': train_acc_hist, 'train_ce': train_ce_hist, 'train_reg': train_reg_hist}
    if SAVE_PREDICTIONS:
        result['predictions'] = {'y_true': tt.astype(int).tolist(), 'y_pred': tp.astype(int).tolist(), 'cm': cm.astype(int).tolist()}
    return result

def paired_analysis(ref_acc, var_acc):
    diff = np.asarray(var_acc) - np.asarray(ref_acc)
    n = len(diff)
    mean_d = float(diff.mean())
    sd_d = float(diff.std(ddof=1)) if n > 1 else 0.0
    se_d = sd_d / math.sqrt(n) if n > 1 else 0.0
    if n > 1:
        t_stat, t_p = sstats.ttest_rel(var_acc, ref_acc)
        t_stat, t_p = (float(t_stat), float(t_p))
    else:
        t_stat, t_p = (float('nan'), float('nan'))
    try:
        if np.allclose(diff, 0):
            w_p = 1.0
        else:
            _, w_p = sstats.wilcoxon(var_acc, ref_acc, zero_method='wilcox')
            w_p = float(w_p)
    except Exception:
        w_p = float('nan')
    d_z = mean_d / sd_d if sd_d > 0 else 0.0
    wins = int((diff > 0).sum())
    losses = int((diff < 0).sum())
    ties = int((diff == 0).sum())
    if n > 1 and sd_d > 0:
        ci_low, ci_high = sstats.t.interval(0.95, n - 1, loc=mean_d, scale=se_d)
        ci_low, ci_high = (float(ci_low), float(ci_high))
    else:
        ci_low, ci_high = (mean_d, mean_d)
    return {'n': int(n), 'mean_diff': mean_d, 'sd_diff': sd_d, 'se_diff': se_d, 'ci_95_low': ci_low, 'ci_95_high': ci_high, 't_stat': t_stat, 't_p': t_p, 'w_p': w_p, 'cohen_dz': float(d_z), 'wins': wins, 'losses': losses, 'ties': ties}

def holm_bonferroni(pvals):
    n = len(pvals)
    if n == 0:
        return np.array([])
    order = np.argsort(pvals)
    sorted_p = np.array(pvals, dtype=float)[order]
    adj_sorted = np.array([min(1.0, p * (n - i)) for i, p in enumerate(sorted_p)])
    for i in range(1, n):
        adj_sorted[i] = max(adj_sorted[i], adj_sorted[i - 1])
    out = np.empty(n)
    out[order] = adj_sorted
    return out

def variant_summary(accs):
    accs = np.asarray(accs)
    n = len(accs)
    if n == 0:
        return None
    mean = float(accs.mean())
    sd = float(accs.std(ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else 0.0
    if n > 1 and sd > 0:
        ci_low, ci_high = sstats.t.interval(0.95, n - 1, loc=mean, scale=se)
        ci_low, ci_high = (float(ci_low), float(ci_high))
    else:
        ci_low, ci_high = (mean, mean)
    return {'n': int(n), 'mean': mean, 'std': sd, 'se': se, 'ci_95_low': ci_low, 'ci_95_high': ci_high, 'min': float(accs.min()), 'max': float(accs.max()), 'median': float(np.median(accs)), 'cv_percent': float(sd / mean * 100) if mean > 0 else 0.0}

def _log(line, log_lines):
    print(line)
    log_lines.append(line)

def save_csv(rows, path):
    if not rows:
        return
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

def _strip_heavy(result):
    return {k: v for k, v in result.items() if k not in ('predictions', 'training_curves')}

def save_results_json(all_results, out_dir, elapsed_s):
    stripped = {}
    for tag, seeds in all_results.items():
        stripped[tag] = {int(s): _strip_heavy(r) for s, r in seeds.items()}
    payload = {'method': 'RMAL-CRNN v2 - Module Ablation (GST + attention + NSLLR; fusion projection kept)', 'modules': {'gst': 'Gated Spherical Tangent (Student-t robust gate + spherical tangent embedding)', 'attention': 'Manifold-aware cross-view fusion (view-PE + log-map cross-view attention; fusion projection kept fixed)', 'fusion_projection': 'QR-initialized fusion projection is always used in RMALCRNN variants and is not ablated', 'nsllr': 'Differentiable non-negative sparse low-rank refinement', 'baseline': 'M4_baseline = BaselineCRNN from code 2 (global LayerNorm + linear projection + CRNN)'}, 'data_root': DATA_ROOTS['0dB'], 'reference_tag': REFERENCE_TAG, 'seeds': SEEDS, 'data_split_seed': DATA_SPLIT_SEED, 'epochs': TOTAL_EPOCHS, 'wall_time_s': elapsed_s, 'variants': [asdict(v) for v in VARIANTS], 'results': stripped}
    with open(os.path.join(out_dir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

def save_per_seed_csv(all_results, out_dir):
    rows = []
    for v in VARIANTS:
        tag = v.tag
        if tag not in all_results:
            continue
        for seed, r in all_results[tag].items():
            d = r['diagnostics']
            rows.append({'variant': tag, 'seed': int(seed), 'use_gst': v.use_gst, 'use_attn': v.use_attn, 'use_nsllr': v.use_nsllr, 'test_acc': r['acc'], 'val_acc': r['val_acc'], 'macro_f1': r['f1'], 'selector': r['selector'], 'n_params': r['n_params'], 'n_trainable': r['n_trainable'], 'train_time_s': r.get('train_time_s', float('nan')), 'best_epoch': r.get('best_epoch', -1), 'final_train_acc': r.get('final_train_acc', float('nan')), 'train_val_gap': r.get('final_train_acc', float('nan')) - r['val_acc'], 'stt_nu_median': d.get('g_HHT_stt_nu_median', float('nan')), 'stt_scale_med': d.get('g_HHT_stt_scale_median', float('nan')), 'ste_basepoint_l2': d.get('n_HHT_ste_basepoint_l2', float('nan')), 'attn_beta_med': d.get('attn_beta_median', float('nan')), 'attn_base_l2': d.get('attn_base_l2_mean', float('nan')), 'lrr_alpha': d.get('lrr_alpha', float('nan')), 'lrr_Z_density': d.get('lrr_Z_density', float('nan'))})
    save_csv(rows, os.path.join(out_dir, 'per_seed.csv'))

def save_for_visualization(all_results, out_dir):
    viz_dir = os.path.join(out_dir, 'viz')
    os.makedirs(viz_dir, exist_ok=True)
    cm_dir = os.path.join(viz_dir, 'confusion_matrices')
    os.makedirs(cm_dir, exist_ok=True)
    boxplot_rows, per_class_rows, curve_rows, diag_rows, timing_rows = ([], [], [], [], [])
    for v in VARIANTS:
        tag = v.tag
        for seed in SEEDS:
            if seed not in all_results.get(tag, {}):
                continue
            r = all_results[tag][seed]
            boxplot_rows.append({'variant': tag, 'seed': int(seed), 'test_acc': float(r['acc']), 'val_acc': float(r['val_acc']), 'macro_f1': float(r['f1']), 'n_trainable': int(r['n_trainable'])})
            timing_rows.append({'variant': tag, 'seed': int(seed), 'train_time_s': float(r.get('train_time_s', float('nan'))), 'best_epoch': int(r.get('best_epoch', -1)), 'n_trainable': int(r['n_trainable']), 'selector': r['selector']})
            if 'per_class_f1' in r:
                for ci, f1v in enumerate(r['per_class_f1']):
                    per_class_rows.append({'variant': tag, 'seed': int(seed), 'class': CLASSES[ci], 'class_idx': ci, 'f1': float(f1v)})
            if 'training_curves' in r:
                tc = r['training_curves']
                va = tc.get('val_acc', [])
                ta = tc.get('train_acc', [])
                ce = tc.get('train_ce', [])
                rg = tc.get('train_reg', [])
                for ep in range(len(va)):
                    curve_rows.append({'variant': tag, 'seed': int(seed), 'epoch': ep + 1, 'val_acc': float(va[ep]) if ep < len(va) else float('nan'), 'train_acc': float(ta[ep]) if ep < len(ta) else float('nan'), 'train_ce': float(ce[ep]) if ep < len(ce) else float('nan'), 'train_reg': float(rg[ep]) if ep < len(rg) else float('nan')})
            d_row = {'variant': tag, 'seed': int(seed)}
            for k, val in r.get('diagnostics', {}).items():
                if isinstance(val, (int, float, bool)):
                    d_row[k] = float(val)
            diag_rows.append(d_row)
            if 'predictions' in r:
                p = r['predictions']
                np.savez_compressed(os.path.join(cm_dir, f'cm_{tag}_{seed}.npz'), y_true=np.array(p['y_true'], dtype=int), y_pred=np.array(p['y_pred'], dtype=int), cm=np.array(p['cm'], dtype=int), labels=np.array(CLASSES))
    paired_rows = []
    if REFERENCE_TAG in all_results:
        ref_data = all_results[REFERENCE_TAG]
        for v in VARIANTS:
            if v.tag == REFERENCE_TAG:
                continue
            for seed in SEEDS:
                if seed not in all_results.get(v.tag, {}) or seed not in ref_data:
                    continue
                va = all_results[v.tag][seed]['acc']
                ra = ref_data[seed]['acc']
                paired_rows.append({'variant': v.tag, 'seed': int(seed), 'variant_acc': float(va), 'reference_acc': float(ra), 'diff_acc': float(va - ra)})
    for v in VARIANTS:
        tag = v.tag
        if tag not in all_results:
            continue
        cms = []
        for seed in SEEDS:
            if seed not in all_results[tag]:
                continue
            r = all_results[tag][seed]
            if 'predictions' in r:
                cms.append(np.array(r['predictions']['cm'], dtype=int))
        if cms:
            cm_agg = np.sum(cms, axis=0).astype(int)
            np.savez_compressed(os.path.join(viz_dir, f'cm_aggregate_{tag}.npz'), cm=cm_agg, labels=np.array(CLASSES), n_seeds=len(cms))
    summary_rows = []
    complexity_rows = []
    ref_accs = [all_results[REFERENCE_TAG][s]['acc'] for s in SEEDS if s in all_results.get(REFERENCE_TAG, {})] if REFERENCE_TAG in all_results else []
    ref_arr = np.array(ref_accs) if ref_accs else None
    for v in VARIANTS:
        tag = v.tag
        accs = [all_results[tag][s]['acc'] for s in SEEDS if s in all_results.get(tag, {})]
        f1s = [all_results[tag][s]['f1'] for s in SEEDS if s in all_results.get(tag, {})]
        times = [all_results[tag][s].get('train_time_s', float('nan')) for s in SEEDS if s in all_results.get(tag, {})]
        bepoch = [all_results[tag][s].get('best_epoch', -1) for s in SEEDS if s in all_results.get(tag, {})]
        nparam = all_results[tag][SEEDS[0]]['n_trainable'] if SEEDS[0] in all_results.get(tag, {}) else 0
        if not accs:
            continue
        sa = variant_summary(accs)
        sf = variant_summary(f1s)
        d_mean = d_se = d_p_t = d_p_w = d_dz = float('nan')
        if tag != REFERENCE_TAG and ref_arr is not None:
            common = [s for s in SEEDS if s in all_results.get(tag, {}) and s in all_results.get(REFERENCE_TAG, {})]
            if common:
                v_arr = np.array([all_results[tag][s]['acc'] for s in common])
                r_arr = np.array([all_results[REFERENCE_TAG][s]['acc'] for s in common])
                st = paired_analysis(r_arr, v_arr)
                d_mean = st['mean_diff']
                d_se = st['se_diff']
                d_p_t = st['t_p']
                d_p_w = st['w_p']
                d_dz = st['cohen_dz']
        summary_rows.append({'variant': tag, 'is_reference': tag == REFERENCE_TAG, 'n': sa['n'], 'acc_mean': sa['mean'], 'acc_std': sa['std'], 'acc_se': sa['se'], 'acc_ci95_low': sa['ci_95_low'], 'acc_ci95_high': sa['ci_95_high'], 'acc_min': sa['min'], 'acc_max': sa['max'], 'acc_median': sa['median'], 'cv_percent': sa['cv_percent'], 'f1_mean': sf['mean'], 'f1_std': sf['std'], 'diff_vs_ref': d_mean, 'diff_se': d_se, 'paired_t_p': d_p_t, 'wilcoxon_p': d_p_w, 'cohen_dz': d_dz, 'n_trainable': int(nparam), 'train_time_mean_s': float(np.nanmean(times)), 'train_time_std_s': float(np.nanstd(times, ddof=1)) if len(times) > 1 else 0.0, 'best_epoch_median': float(np.median([e for e in bepoch if e > 0])) if any((e > 0 for e in bepoch)) else -1.0})
        complexity_rows.append({'variant': tag, 'n_trainable': int(nparam), 'acc_mean': sa['mean'], 'acc_ci95_low': sa['ci_95_low'], 'acc_ci95_high': sa['ci_95_high'], 'train_time_mean_s': float(np.nanmean(times))})
    pivot_rows = []
    for seed in SEEDS:
        row = {'seed': int(seed)}
        for v in VARIANTS:
            tag = v.tag
            if seed in all_results.get(tag, {}):
                row[f'{tag}_acc'] = float(all_results[tag][seed]['acc'])
                row[f'{tag}_f1'] = float(all_results[tag][seed]['f1'])
            else:
                row[f'{tag}_acc'] = float('nan')
                row[f'{tag}_f1'] = float('nan')
        pivot_rows.append(row)
    if boxplot_rows:
        save_csv(boxplot_rows, os.path.join(viz_dir, 'boxplot.csv'))
    if paired_rows:
        save_csv(paired_rows, os.path.join(viz_dir, 'paired_diff.csv'))
    if per_class_rows:
        save_csv(per_class_rows, os.path.join(viz_dir, 'per_class_f1.csv'))
    if curve_rows:
        save_csv(curve_rows, os.path.join(viz_dir, 'training_curves.csv'))
    if timing_rows:
        save_csv(timing_rows, os.path.join(viz_dir, 'timing.csv'))
    if summary_rows:
        save_csv(summary_rows, os.path.join(viz_dir, 'variant_summary.csv'))
    if complexity_rows:
        save_csv(complexity_rows, os.path.join(viz_dir, 'complexity_perf.csv'))
    if pivot_rows:
        save_csv(pivot_rows, os.path.join(viz_dir, 'per_seed_pivot.csv'))
    if diag_rows:
        all_keys = sorted(set().union(*[r.keys() for r in diag_rows]))
        normalized = [{k: r.get(k, float('nan')) for k in all_keys} for r in diag_rows]
        save_csv(normalized, os.path.join(viz_dir, 'diagnostics.csv'))

def run_analysis_and_stats(all_results, out_dir):
    log = []
    _log('=' * 110, log)
    _log('  MODULE ABLATION SUMMARY  (GST + attention + NSLLR; fusion projection kept)', log)
    _log('=' * 110, log)
    _log(f"  {'variant':<14}{'n':>4}{'mean':>10}{'std':>10}{'se':>10}{'95% CI':>22}{'median':>10}{'min':>10}{'max':>10}", log)
    _log('  ' + '-' * 100, log)
    summaries = {}
    for v in VARIANTS:
        tag = v.tag
        accs = [all_results[tag][s]['acc'] for s in SEEDS if s in all_results.get(tag, {})]
        s = variant_summary(accs)
        summaries[tag] = (accs, s)
        if s is None:
            _log(f'  {tag:<14}  (no successful runs)', log)
            continue
        marker = '  <-- ref' if tag == REFERENCE_TAG else '  baseline' if tag == 'M4_baseline' else ''
        ci = f"[{s['ci_95_low']:.4f}, {s['ci_95_high']:.4f}]"
        _log(f"  {tag:<14}{s['n']:>4}{s['mean']:>10.4f}{s['std']:>10.4f}{s['se']:>10.4f}{ci:>22}{s['median']:>10.4f}{s['min']:>10.4f}{s['max']:>10.4f}{marker}", log)
    ranking = sorted([(tag, s['mean']) for tag, (_, s) in summaries.items() if s is not None], key=lambda x: -x[1])
    _log('\n  Ranking by mean test accuracy:', log)
    for i, (tag, m) in enumerate(ranking, 1):
        mark = ' (PROPOSED)' if tag == REFERENCE_TAG else ' (BASELINE)' if tag == 'M4_baseline' else ''
        _log(f'    {i:>2}. {tag:<14}  {m:.4f}{mark}', log)
    stat_records = {}
    if REFERENCE_TAG not in all_results or not all_results[REFERENCE_TAG]:
        _log('\n  [!] No reference results.', log)
    else:
        _log('\n' + '=' * 110, log)
        _log(f'  PAIRED STATISTICS  (variant - {REFERENCE_TAG})', log)
        _log('=' * 110, log)
        _log(f"  {'variant':<14}{'n':>4}{'D_acc(%)':>10}{'SE(%)':>9}{'95% CI(%)':>22}{'paired t p':>14}{'wilc. p':>12}{'d_z':>8}{'wins':>10}", log)
        _log('  ' + '-' * 100, log)
        compare_tags = [v.tag for v in VARIANTS if v.tag != REFERENCE_TAG]
        raw_pt, raw_pw, rows = ([], [], [])
        for tag in compare_tags:
            if tag not in all_results or not all_results[tag]:
                rows.append((tag, None))
                raw_pt.append(1.0)
                raw_pw.append(1.0)
                continue
            common = sorted(set(all_results[tag].keys()) & set(all_results[REFERENCE_TAG].keys()))
            if not common:
                rows.append((tag, None))
                raw_pt.append(1.0)
                raw_pw.append(1.0)
                continue
            r_arr = np.array([all_results[REFERENCE_TAG][s]['acc'] for s in common])
            v_arr = np.array([all_results[tag][s]['acc'] for s in common])
            st = paired_analysis(r_arr, v_arr)
            rows.append((tag, st))
            raw_pt.append(st['t_p'] if not math.isnan(st['t_p']) else 1.0)
            raw_pw.append(st['w_p'] if not math.isnan(st['w_p']) else 1.0)
        adj_t = holm_bonferroni(raw_pt)
        adj_w = holm_bonferroni(raw_pw)
        for tag, st in rows:
            if st is None:
                _log(f'  {tag:<14}  (no results)', log)
                continue
            ci = f"[{st['ci_95_low'] * 100:+.2f}, {st['ci_95_high'] * 100:+.2f}]"
            wl = f"{st['wins']}/{st['wins'] + st['losses'] + st['ties']}"
            _log(f"  {tag:<14}{st['n']:>4}{st['mean_diff'] * 100:>+10.3f}{st['se_diff'] * 100:>9.3f}{ci:>22}{st['t_p']:>14.4g}{st['w_p']:>12.4g}{st['cohen_dz']:>+8.3f}{wl:>10}", log)
        _log('\n  ' + '-' * 80, log)
        _log(f'  Holm-Bonferroni adjusted p-values ({len(compare_tags)} comparisons vs reference):', log)
        _log(f"  {'variant':<14}{'raw t_p':>12}{'adj t_p':>12}{'raw w_p':>12}{'adj w_p':>12}{'survives a=0.05':>20}", log)
        for (tag, st), rt, rw, at, aw in zip(rows, raw_pt, raw_pw, adj_t, adj_w):
            if st is None:
                continue
            survives = 'BOTH' if at < 0.05 and aw < 0.05 else 't-only' if at < 0.05 else 'w-only' if aw < 0.05 else 'no'
            _log(f'  {tag:<14}{rt:>12.4g}{at:>12.4g}{rw:>12.4g}{aw:>12.4g}{survives:>20}', log)
            stat_records[tag] = {**st, 'raw_t_p': float(rt), 'raw_w_p': float(rw), 'adj_t_p': float(at), 'adj_w_p': float(aw), 'survives': survives}
        _log('\n  Interpretation:', log)
        _log('    M1_no_gst     contribution of 门控球切 (robust gate + spherical tangent embedding)', log)
        _log('    M2_no_attn    contribution of 注意力 (view-PE + log-map attention; QR fusion kept)', log)
        _log('    M3_no_nsllr   contribution of NSLLR', log)
        _log('    M4_baseline   total gain of all three modules over the CRNN baseline', log)
    _log('\n' + '=' * 110, log)
    _log('  COMPLEXITY & TIMING SUMMARY', log)
    _log('=' * 110, log)
    _log(f"  {'variant':<14}{'params (k)':>14}{'time/run (s)':>20}{'best epoch (med)':>20}", log)
    _log('  ' + '-' * 72, log)
    for v in VARIANTS:
        tag = v.tag
        if tag not in all_results or not all_results[tag]:
            continue
        times = [r.get('train_time_s', float('nan')) for r in all_results[tag].values()]
        bepoch = [r.get('best_epoch', -1) for r in all_results[tag].values()]
        nparam = next(iter(all_results[tag].values()))['n_trainable']
        bm = float(np.median([e for e in bepoch if e > 0])) if any((e > 0 for e in bepoch)) else float('nan')
        time_mean = np.nanmean(times)
        time_std = np.nanstd(times, ddof=1) if len(times) > 1 else 0.0
        _log(f'  {tag:<14}{nparam / 1000.0:>13.2f}k{time_mean:>15.1f} ± {time_std:>4.1f}{bm:>20.0f}', log)
    with open(os.path.join(out_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(log))
    stat_payload = {'reference_tag': REFERENCE_TAG, 'architecture': '门控球切 (GST) + 注意力 (view-PE + LMA) + QR fusion projection (kept) + NSLLR + CRNN', 'variant_summary': {tag: {'n': s['n'], 'mean': s['mean'], 'std': s['std'], 'se': s['se'], 'ci_95_low': s['ci_95_low'], 'ci_95_high': s['ci_95_high'], 'median': s['median'], 'min': s['min'], 'max': s['max'], 'cv_percent': s['cv_percent']} if s else None for tag, (_, s) in summaries.items()}, 'paired_vs_reference': stat_records, 'holm_bonferroni_applied': True, 'n_seeds_per_variant': len(SEEDS)}
    with open(os.path.join(out_dir, 'statistical_analysis.json'), 'w', encoding='utf-8') as f:
        json.dump(stat_payload, f, indent=2, ensure_ascii=False)

def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(SAVE_DIR, f'run_{timestamp}')
    os.makedirs(out_dir, exist_ok=True)
    print('=' * 110)
    print('  RMAL-CRNN v2  --  Module Ablation (GST + Attention + NSLLR; Fusion Projection Kept)')
    print('=' * 110)
    print(f'  Variants : {len(VARIANTS)}   Seeds : {len(SEEDS)}   Epochs : {TOTAL_EPOCHS}   Device : {DEVICE}')
    print(f'  Reference: {REFERENCE_TAG}')
    print(f'  Output   : {out_dir}')
    print('-' * 110)
    for v in VARIANTS:
        flags = f'gst={v.use_gst}, attn={v.use_attn}, nsllr={v.use_nsllr}'
        marker = ' (proposed)' if v.tag == REFERENCE_TAG else ' (BASELINE)' if v.tag == 'M4_baseline' else ''
        print(f'    {v.tag:<14}  {flags}{marker}')
    print('=' * 110)
    print('  M1_no_gst      : robust gate + spherical tangent embedding -> LayerNorm')
    print('  M2_no_attn     : view-PE + cross-view attention -> zero view_emb + IdentityAttention; QR fusion kept')
    print('  M3_no_nsllr    : NSLLR -> Identity')
    print('  M4_baseline    : BaselineCRNN from code 2: global LayerNorm + linear projection + CRNN')
    print('=' * 110)
    print('\n[data] loading 0 dB ...')
    root = DATA_ROOTS['0dB']
    if not os.path.exists(root):
        raise RuntimeError(f'Missing data: {root}')
    X, y = load_all_data(root)
    splits = split_data(X, y, seed=DATA_SPLIT_SEED)
    print(f'  shape={X.shape},  train/val/test = {len(splits[0][1])}/{len(splits[1][1])}/{len(splits[2][1])}')
    all_results = {v.tag: {} for v in VARIANTS}
    t_global = time.time()
    total_runs = len(VARIANTS) * len(SEEDS)
    run_idx = 0
    for vidx, cfg in enumerate(VARIANTS, 1):
        print(f"\n{'#' * 110}")
        print(f'  VARIANT [{vidx:02d}/{len(VARIANTS)}]  {cfg.tag}    gst={cfg.use_gst}  attn={cfg.use_attn}  nsllr={cfg.use_nsllr}')
        print(f"{'#' * 110}")
        for seed in SEEDS:
            run_idx += 1
            t0 = time.time()
            try:
                r = train_one(splits, seed=seed, cfg=cfg, verbose=False)
                dt = time.time() - t0
                all_results[cfg.tag][seed] = r
                d = r['diagnostics']
                nu = d.get('g_HHT_stt_nu_median', float('nan'))
                bt = d.get('attn_beta_median', float('nan'))
                bp = d.get('attn_base_l2_mean', float('nan'))
                la = d.get('lrr_alpha', float('nan'))
                gap = r.get('final_train_acc', float('nan')) - r['val_acc']
                if cfg.tag == 'M4_baseline':
                    extra = f'  BaselineCRNN  gap={gap:+.3f}'
                else:
                    bp_str = f'||p||={bp:.2f}' if not math.isnan(bp) else '||p||=N/A'
                    extra = f'  nu={nu:.0f}  beta={bt:.3f}  {bp_str}  lrr_a={la:.3f}  gap={gap:+.3f}'
                print(f"  [{run_idx:03d}/{total_runs}]  {cfg.tag:<14}  seed={seed:<5}  test={r['acc']:.4f}  val={r['val_acc']:.4f}  f1={r['f1']:.4f}  sel={r['selector']:<9}  ep*={r.get('best_epoch', -1):>3}  params={r['n_trainable'] / 1000.0:.1f}k{extra}  ({dt:.0f}s)")
            except Exception as e:
                print(f'  [{run_idx:03d}/{total_runs}]  {cfg.tag:<14}  seed={seed:<5}  FAILED: {type(e).__name__}: {e}')
                import traceback
                traceback.print_exc()
            try:
                save_results_json(all_results, out_dir, time.time() - t_global)
                save_per_seed_csv(all_results, out_dir)
                save_for_visualization(all_results, out_dir)
            except Exception as e:
                print(f'  [warn] partial save failed: {e}')
    elapsed = time.time() - t_global
    print(f"\n{'=' * 110}")
    print(f'  ALL DONE.  wall time: {elapsed / 60:.1f} min  ({elapsed / 3600:.2f} h)')
    print('=' * 110)
    run_analysis_and_stats(all_results, out_dir)
    save_per_seed_csv(all_results, out_dir)
    save_results_json(all_results, out_dir, elapsed)
    save_for_visualization(all_results, out_dir)
    print(f'\n  Results dir: {out_dir}')
if __name__ == '__main__':
    main()
