import os, math, copy, random, time, csv, argparse, warnings
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
warnings.filterwarnings('ignore')
DATA_ROOT_DEFAULT = 'data/features/0dB'
OUT_DIR_DEFAULT = 'hp_sweep_results'
CLASSES = [f'{i:02d}' for i in range(10)]
NUM_CLASS = len(CLASSES)
FULL_DIM = 192
FEATURE_BLOCKS = {'HHT': (0, 32), 'LOFAR': (32, 64), 'FFT': (64, 96), 'WPT': (96, 128), 'HHSA': (128, 160), 'MFCC': (160, 192)}
VIEWS = list(FEATURE_BLOCKS.values())
D_VIEW = 32
V_NUM = len(VIEWS)
EPOCHS_SWEEP = 60
SWA_EPOCHS_SW = 15
SEEDS_SWEEP = [1919, 2002, 2008, 2026]
DATA_SPLIT_SEED = 42
PROJ_OUT_DEF = 64
N_HEADS_DEF = 4
LRR_RANK_DEF = 8
DROPOUT_DEF = 0.5
LR_DEFAULT = 0.001
WD_DEFAULT = 0.0001
BATCH_DEFAULT = 32
LABEL_SMOOTHING = 0.1
BETA_INIT = 0.5
LRR_LAM_INIT = 0.1
LRR_ALPHA_INIT = 0.3
LRR_LAM_NUC = 0.001
LRR_LAM_SPS = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_all_data(root):
    Xs, ys = ([], [])
    for li, cls in enumerate(CLASSES):
        d = os.path.join(root, cls)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith('.npy'):
                continue
            x = np.load(os.path.join(d, f))
            if x.ndim != 2 or x.shape[1] != FULL_DIM:
                continue
            Xs.append(x)
            ys.append(li)
    if not Xs:
        raise RuntimeError(f'No .npy files under {root}')
    T = max((x.shape[0] for x in Xs))
    X = np.zeros((len(Xs), T, FULL_DIM), dtype='float32')
    for i, x in enumerate(Xs):
        X[i, :x.shape[0]] = x
    return (X, np.array(ys))

def split_data(X, y, seed=DATA_SPLIT_SEED):
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)
    Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=seed, stratify=ytmp)
    return ((Xtr, ytr), (Xva, yva), (Xte, yte))

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
        self.eps = eps
        self.center = nn.Parameter(torch.zeros(dim))
        self.log_scale = nn.Parameter(torch.zeros(dim))
        self.log_inv_nu = nn.Parameter(torch.full((dim,), math.log(1.0 / nu_init)))

    def _nu(self):
        return (1.0 / torch.exp(self.log_inv_nu).clamp(0.0001, 1.0)).clamp(1.0, 10000.0)

    def forward(self, z):
        s = torch.exp(self.log_scale).clamp(min=self.eps)
        nu = self._nu()
        r = (z - self.center) / s
        return z * ((nu + 1.0) / (nu + r.pow(2)))

class SphericalTangentEmbedding(nn.Module):

    def __init__(self, dim=D_VIEW, eps=1e-06):
        super().__init__()
        self.eps = eps
        self.p = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, z):
        D = z.shape[-1]
        zf = z.reshape(-1, D)
        p_unit = self.p / self.p.norm().clamp(min=self.eps)
        parallel = (zf * p_unit).sum(-1, keepdim=True)
        tangent = zf - parallel * p_unit
        n = tangent.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        out = math.sqrt(D) * tangent / n + self.bias
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

    def __init__(self, d_model, n_heads, V, eps=1e-06):
        super().__init__()
        assert d_model % n_heads == 0, f'd_model ({d_model}) must be divisible by n_heads ({n_heads})'
        self.n_heads, self.d_head, self.eps = (n_heads, d_model // n_heads, eps)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        bp = torch.randn(n_heads, self.d_head) * 0.3
        bp[:, 0] += 1.0
        self.base_point = nn.Parameter(bp)
        self.log_beta = nn.Parameter(torch.full((n_heads,), math.log(BETA_INIT)))

    def _logmap(self, x, p):
        inner = (x * p).sum(-1, keepdim=True)
        return x - inner * p

    def forward(self, tokens):
        B, T, V, D = tokens.shape
        x = tokens.reshape(B * T, V, D)
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = [t.reshape(B * T, V, self.n_heads, self.d_head).transpose(1, 2) for t in (q, k, v)]
        p_unit = F.normalize(self.base_point, dim=-1, eps=self.eps)
        p_b = p_unit.view(1, self.n_heads, 1, self.d_head)
        q_n = F.normalize(q, dim=-1, eps=self.eps)
        k_n = F.normalize(k, dim=-1, eps=self.eps)
        xi_q = self._logmap(q_n, p_b)
        xi_k = self._logmap(k_n, p_b)
        xi_v = self._logmap(v, p_b)
        nq = (xi_q * xi_q).sum(-1, keepdim=True)
        nk_t = (xi_k * xi_k).sum(-1, keepdim=True).transpose(-1, -2)
        cross = xi_q @ xi_k.transpose(-2, -1)
        d2 = (nq + nk_t - 2.0 * cross).clamp(min=0.0)
        beta = torch.exp(self.log_beta).view(1, self.n_heads, 1, 1)
        attn = F.softmax(-beta * d2, dim=-1)
        out = (attn @ xi_v).transpose(1, 2).reshape(B * T, V, D)
        return self.norm(self.out_proj(out) + x).reshape(B, T, V, D)

class DiffNSLLR(nn.Module):

    def __init__(self, d_in, rank, eps=1e-06):
        super().__init__()
        self.rank = rank
        self.eps = eps
        self.proj_u = nn.Linear(d_in, rank, bias=False)
        self.proj_v = nn.Linear(d_in, rank, bias=False)
        nn.init.orthogonal_(self.proj_u.weight)
        nn.init.orthogonal_(self.proj_v.weight)
        self.log_lam = nn.Parameter(torch.tensor(math.log(LRR_LAM_INIT)))
        self.alpha_logit = nn.Parameter(torch.tensor(math.log(LRR_ALPHA_INIT / (1 - LRR_ALPHA_INIT))))
        self.refined_norm = nn.LayerNorm(d_in)
        self._cache = {}

    def forward(self, feat):
        U = self.proj_u(feat)
        V = self.proj_v(feat)
        Z_raw = torch.einsum('btr,bsr->bts', U, V) / math.sqrt(self.rank)
        Z = torch.clamp(Z_raw - self.log_lam.exp(), min=0.0)
        F_ref = self.refined_norm(torch.einsum('bts,bsd->btd', Z, feat))
        self._cache = {'Z': Z, 'U': U, 'V': V}
        a = torch.sigmoid(self.alpha_logit)
        return feat + a * (F_ref - feat)

    def regularization(self):
        if not self._cache:
            return torch.zeros((), device=self.log_lam.device)
        Z, U, V = (self._cache['Z'], self._cache['U'], self._cache['V'])
        return LRR_LAM_NUC * 0.5 * (U.pow(2).mean() + V.pow(2).mean()) + LRR_LAM_SPS * Z.abs().mean()

@dataclass
class HPConfig:
    proj_out: int = PROJ_OUT_DEF
    n_heads: int = N_HEADS_DEF
    lrr_rank: int = LRR_RANK_DEF
    dropout: float = DROPOUT_DEF
    lr: float = LR_DEFAULT
    batch: int = BATCH_DEFAULT
    wd: float = WD_DEFAULT

class RMALCRNN(nn.Module):

    def __init__(self, hp: HPConfig, views=VIEWS, num_class=NUM_CLASS):
        super().__init__()
        self.views = views
        self.V = len(views)
        D = views[0][1] - views[0][0]
        self.d_view = D
        self.gates = nn.ModuleList([StudentTRobustGate(D) for _ in views])
        self.norms = nn.ModuleList([SphericalTangentEmbedding(D) for _ in views])
        self.view_emb = nn.Parameter(torch.zeros(self.V, D))
        nn.init.normal_(self.view_emb, std=0.02)
        self.cross_view_attn = LogMapTangentAttention(D, hp.n_heads, self.V)
        self.fusion_proj = QRInitLinear(self.V * D, hp.proj_out)
        self.lrr_block = DiffNSLLR(hp.proj_out, hp.lrr_rank)
        self.cnn = nn.Sequential(nn.Conv1d(hp.proj_out, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2), nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2))
        self.rnn = nn.LSTM(128, 128, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(p=hp.dropout)
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
        emb = self.drop(out).mean(dim=1)
        return self.fc(emb)

    def lrr_regularization(self):
        return self.lrr_block.regularization()

class SWA:

    def __init__(self, model):
        self.swa = {n: torch.zeros_like(p.data) for n, p in model.named_parameters() if p.requires_grad}
        self.n = 0

    @torch.no_grad()
    def accumulate(self, model):
        self.n += 1
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.swa[n] += (p.data - self.swa[n]) / self.n

    @torch.no_grad()
    def apply(self, model):
        if self.n == 0:
            return
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.swa[n])

def update_bn(model, loader):
    has = any((isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)) for m in model.modules()))
    if not has:
        return
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.reset_running_stats()
            m.momentum = None
    model.train()
    with torch.no_grad():
        for x, _ in loader:
            model(x.to(DEVICE))
    model.eval()

def evaluate(model, loader):
    model.eval()
    pr, tr = ([], [])
    with torch.no_grad():
        for bx, by in loader:
            logits = model(bx.to(DEVICE))
            pr.extend(logits.argmax(1).cpu().numpy())
            tr.extend(by.numpy())
    return (np.array(tr), np.array(pr))

def train_one(splits, hp: HPConfig, seed, epochs):
    set_seed(seed)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = splits
    tl = DataLoader(SeqDataset(Xtr, ytr), batch_size=hp.batch, shuffle=True, drop_last=True)
    el = DataLoader(SeqDataset(Xte, yte), batch_size=hp.batch, shuffle=False)
    net = RMALCRNN(hp).to(DEVICE)
    opt = optim.AdamW(net.parameters(), lr=hp.lr, weight_decay=hp.wd)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    swa = SWA(net)
    swa_start = max(epochs - SWA_EPOCHS_SW, 1)
    for ep in range(epochs):
        net.train()
        for bx, by in tl:
            bx, by = (bx.to(DEVICE), by.to(DEVICE))
            opt.zero_grad()
            ce = F.cross_entropy(net(bx), by, label_smoothing=LABEL_SMOOTHING)
            reg = net.lrr_regularization()
            (ce + reg).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            opt.step()
        sch.step()
        if ep >= swa_start:
            swa.accumulate(net)
    swa.apply(net)
    update_bn(net, tl)
    t, p = evaluate(net, el)
    return (float(accuracy_score(t, p)), float(f1_score(t, p, average='macro', zero_division=0)))

def write_rows(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

def sweep_1d(splits, out_csv, hp_attr, values, base_hp=None, seeds=None, epochs=EPOCHS_SWEEP):
    base = base_hp or HPConfig()
    seeds = seeds or SEEDS_SWEEP
    rows = []
    t_global = time.time()
    print(f'\n[sweep] {hp_attr} over {values}  (seeds={seeds}, epochs={epochs})')
    for v in values:
        for sd in seeds:
            hp = copy.deepcopy(base)
            setattr(hp, hp_attr, v)
            t0 = time.time()
            try:
                acc, f1 = train_one(splits, hp, sd, epochs=epochs)
                dt = time.time() - t0
                print(f'    {hp_attr}={v!s:<10}  seed={sd:<5}  acc={acc:.4f}  f1={f1:.4f}  ({dt:.0f}s)')
                rows.append({hp_attr: v, 'seed': sd, 'acc': acc * 100, 'f1': f1 * 100})
            except Exception as e:
                print(f'    {hp_attr}={v!s:<10}  seed={sd:<5}  FAILED: {type(e).__name__}: {e}')
            write_rows(out_csv, rows)
    print(f'[sweep] {hp_attr} done in {(time.time() - t_global) / 60:.1f} min -> {out_csv}')

def sweep_lr_bs(splits, out_csv, lrs, bss, seeds=None, epochs=EPOCHS_SWEEP):
    seeds = seeds or SEEDS_SWEEP
    rows = []
    t_global = time.time()
    print(f'\n[sweep] lr x batch grid  {len(lrs)} x {len(bss)}  (seeds={seeds}, epochs={epochs})')
    for lr in lrs:
        for bs in bss:
            for sd in seeds:
                hp = HPConfig(lr=lr, batch=bs)
                t0 = time.time()
                try:
                    acc, f1 = train_one(splits, hp, sd, epochs=epochs)
                    dt = time.time() - t0
                    print(f'    lr={lr:.0e}  bs={bs:<4}  seed={sd:<5}  acc={acc:.4f}  f1={f1:.4f}  ({dt:.0f}s)')
                    rows.append({'lr': lr, 'batch': bs, 'seed': sd, 'acc': acc * 100, 'f1': f1 * 100})
                except Exception as e:
                    print(f'    lr={lr:.0e}  bs={bs:<4}  seed={sd:<5}  FAILED: {type(e).__name__}: {e}')
                write_rows(out_csv, rows)
    print(f'[sweep] lr x batch done in {(time.time() - t_global) / 60:.1f} min -> {out_csv}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default=DATA_ROOT_DEFAULT, help='dir of .npy class subfolders')
    parser.add_argument('--out-dir', default=OUT_DIR_DEFAULT)
    parser.add_argument('--epochs', type=int, default=EPOCHS_SWEEP)
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS_SWEEP)
    parser.add_argument('--only', nargs='+', default=None, choices=['proj_out', 'n_heads', 'lrr_rank', 'lr_bs'])
    parser.add_argument('--smoke', action='store_true', help='Tiny sanity-check sweep: 2 seeds x 15 epochs x small grids')
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.smoke:
        args.seeds = [42, 7]
        args.epochs = 15
    print('=' * 80)
    print('  HP sweeps on RMAL-CRNN v2 (M0_full)')
    print('=' * 80)
    print(f'  data root : {args.data_root}')
    print(f'  out  dir  : {args.out_dir}')
    print(f'  device    : {DEVICE}')
    print(f'  seeds     : {args.seeds}')
    print(f'  epochs    : {args.epochs}')
    print(f"  only      : {args.only or 'all'}")
    print('-' * 80)
    print('\n[data] loading ...')
    X, y = load_all_data(args.data_root)
    splits = split_data(X, y)
    print(f'  shape={X.shape}  split={len(splits[0][1])}/{len(splits[1][1])}/{len(splits[2][1])}')
    todo = args.only or ['proj_out', 'n_heads', 'lrr_rank', 'lr_bs']
    grid_proj = [32, 48, 64, 80, 96, 128]
    grid_heads = [1, 2, 4, 8]
    grid_rank = [2, 4, 8, 16, 32]
    grid_lr = [0.0001, 0.0003, 0.001, 0.003, 0.01]
    grid_bs = [16, 32, 64, 128]
    if args.smoke:
        grid_proj = [32, 64, 128]
        grid_heads = [1, 4]
        grid_rank = [4, 8, 16]
        grid_lr = [0.0001, 0.001, 0.01]
        grid_bs = [32, 64]
    if 'proj_out' in todo:
        sweep_1d(splits, os.path.join(args.out_dir, 'sweep_proj_out.csv'), 'proj_out', grid_proj, seeds=args.seeds, epochs=args.epochs)
    if 'n_heads' in todo:
        sweep_1d(splits, os.path.join(args.out_dir, 'sweep_n_heads.csv'), 'n_heads', grid_heads, seeds=args.seeds, epochs=args.epochs)
    if 'lrr_rank' in todo:
        sweep_1d(splits, os.path.join(args.out_dir, 'sweep_lrr_rank.csv'), 'lrr_rank', grid_rank, seeds=args.seeds, epochs=args.epochs)
    if 'lr_bs' in todo:
        sweep_lr_bs(splits, os.path.join(args.out_dir, 'sweep_lr_bs.csv'), lrs=grid_lr, bss=grid_bs, seeds=args.seeds, epochs=args.epochs)
    print('\n[done]  all sweeps finished.')
    print(f'        CSVs are in: {args.out_dir}')
    print(f'        Plot them with:  python hp_plots.py --data-dir {args.out_dir}')
if __name__ == '__main__':
    main()
