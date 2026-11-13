import os, math, json, copy, random, time, warnings, csv
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
warnings.filterwarnings('ignore')
DATA_ROOTS = {'-5dB': 'data/features/-5dB', '0dB': 'data/features/0dB', '5dB': 'data/features/5dB', '10dB': 'data/features/10dB'}
SNR_ORDER = ['-5dB', '0dB', '5dB', '10dB']
SAVE_DIR = 'baselines_comparison'
os.makedirs(SAVE_DIR, exist_ok=True)
CLASSES = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09']
NUM_CLASS = len(CLASSES)
FULL_DIM = 192
SEEDS = [1919, 2002, 2008, 2026]
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
GRAD_CLIP = 2.0
STANDARDIZE = True
EIGH_FLOAT64 = True
COV_EPS = 0.001
COV_SHRINK = 0.1
TOK_SHRINK = 0.3
EIG_EPS = 0.001
MODELS_TO_RUN = ['MAtt', 'SPDNet', 'TCN', 'BiLSTM']
QUICK_TEST = False
if QUICK_TEST:
    TOTAL_EPOCHS, SWA_EPOCHS, SWA_START, BEST_TRACK_START = (8, 3, 5, 3)
    SEEDS = SEEDS[:2]
LOG_PER_EPOCH = True
SAVE_PREDICTIONS = True
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def standardize_splits(splits):
    if not STANDARDIZE:
        return splits
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = splits
    mu = X_tr.reshape(-1, FULL_DIM).mean(axis=0, keepdims=True)
    sd = X_tr.reshape(-1, FULL_DIM).std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-06, 1.0, sd).astype('float32')
    mu = mu.astype('float32')

    def f(X):
        Xn = (X.reshape(-1, FULL_DIM) - mu) / sd
        return Xn.reshape(X.shape).astype('float32')
    return ((f(X_tr), y_tr), (f(X_va), y_va), (f(X_te), y_te))

class SeqDataset(Dataset):

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return (self.X[i], self.y[i])

def _sym(x):
    return 0.5 * (x + x.transpose(-1, -2))

def _cov_over_time(z, eps=COV_EPS, shrink=COV_SHRINK):
    z = z - z.mean(dim=1, keepdim=True)
    T = z.shape[1]
    c = z.transpose(1, 2) @ z / max(T - 1, 1)
    d = c.shape[-1]
    eye = torch.eye(d, device=z.device, dtype=z.dtype)
    c = _sym(c)
    avg_var = c.diagonal(dim1=-2, dim2=-1).mean(dim=-1)[..., None, None]
    c = (1.0 - shrink) * c + shrink * avg_var * eye
    return c + eps * eye

def _eigh_safe(S):
    S = _sym(S)
    d = S.shape[-1]
    eye = torch.eye(d, device=S.device, dtype=S.dtype)
    S = S + 1e-05 * eye
    work = S.double() if EIGH_FLOAT64 else S
    try:
        w, V = torch.linalg.eigh(work)
    except Exception:
        w, V = torch.linalg.eigh(work.cpu())
        w, V = (w.to(S.device), V.to(S.device))
    return (w.to(S.dtype), V.to(S.dtype))

def _reeig(S, eps_eig=EIG_EPS):
    w, V = _eigh_safe(S)
    w = torch.clamp(w, min=eps_eig)
    return _sym(V @ torch.diag_embed(w) @ V.transpose(-1, -2))

def _logeig(S, eps_eig=EIG_EPS):
    w, V = _eigh_safe(S)
    w = torch.clamp(w, min=eps_eig)
    return _sym(V @ torch.diag_embed(torch.log(w)) @ V.transpose(-1, -2))

def _expeig(S):
    w, V = _eigh_safe(S)
    w = torch.clamp(w, min=-30.0, max=30.0)
    return _sym(V @ torch.diag_embed(torch.exp(w)) @ V.transpose(-1, -2))

def _vech(S):
    d = S.shape[-1]
    iu = torch.triu_indices(d, d, offset=0, device=S.device)
    diag_mask = (iu[0] == iu[1]).to(S.dtype)
    scale = torch.where(diag_mask.bool(), torch.ones_like(diag_mask), math.sqrt(2.0) * torch.ones_like(diag_mask))
    flat = S[..., iu[0], iu[1]]
    return flat * scale

class _BaseModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.register_buffer('_dummy', torch.zeros(1))

    def lrr_regularization(self):
        return torch.zeros((), device=self._dummy.device)

    @torch.no_grad()
    def diagnostics(self):
        return {}

class BiMap(nn.Module):

    def __init__(self, d_in, d_out):
        super().__init__()
        assert d_in >= d_out, 'BiMap reduces (or keeps) dimension: need d_in >= d_out'
        W = torch.empty(d_in, d_out)
        nn.init.orthogonal_(W)
        self.weight = nn.Parameter(W)
        self.d_out = d_out

    def forward(self, S):
        Q, _ = torch.linalg.qr(self.weight)
        Q = Q[:, :self.d_out]
        return _sym(Q.transpose(0, 1) @ S @ Q)

class ReEig(nn.Module):

    def __init__(self, eps_eig=EIG_EPS):
        super().__init__()
        self.eps_eig = eps_eig

    def forward(self, S):
        return _reeig(S, self.eps_eig)

class LogEig(nn.Module):

    def forward(self, S):
        return _logeig(S)

class SPDNet(_BaseModel):

    def __init__(self, in_dim=FULL_DIM, proj=24, dims=(24, 16, 8), n_class=NUM_CLASS):
        super().__init__()
        self.proj = nn.Linear(in_dim, proj)
        layers = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers += [BiMap(d_in, d_out), ReEig()]
        self.spdnet = nn.Sequential(*layers)
        self.logeig = LogEig()
        d_last = dims[-1]
        feat = d_last * (d_last + 1) // 2
        self.classifier = nn.Sequential(nn.BatchNorm1d(feat), nn.Linear(feat, n_class))

    def forward(self, x):
        z = self.proj(x)
        S = _cov_over_time(z)
        S = self.spdnet(S)
        S = self.logeig(S)
        v = _vech(S)
        return self.classifier(v)

class _ManifoldAttention(nn.Module):

    def __init__(self, d):
        super().__init__()
        self.q = BiMap(d, d)
        self.k = BiMap(d, d)
        self.v = BiMap(d, d)
        self.tau = nn.Parameter(torch.tensor(1.0 / math.sqrt(d)))

    def forward(self, S):
        B, n, d, _ = S.shape
        flat = S.reshape(B * n, d, d)
        Q = self.q(flat)
        K = self.k(flat)
        V = self.v(flat)
        logQ = _logeig(Q).reshape(B, n, d, d)
        logK = _logeig(K).reshape(B, n, d, d)
        logV = _logeig(V).reshape(B, n, d, d)
        diff = logQ.unsqueeze(2) - logK.unsqueeze(1)
        dist = torch.sqrt((diff ** 2).sum(dim=(-1, -2)) + 1e-08)
        attn = torch.softmax(-dist * torch.abs(self.tau), dim=-1)
        mixed = torch.einsum('bij,bjde->bide', attn, logV)
        out = _expeig(mixed.reshape(B * n, d, d)).reshape(B, n, d, d)
        return out

class MAtt(_BaseModel):

    def __init__(self, in_dim=FULL_DIM, feat=16, n_windows=4, n_class=NUM_CLASS):
        super().__init__()
        self.front = nn.Sequential(nn.Conv1d(in_dim, 40, kernel_size=7, padding=3), nn.BatchNorm1d(40), nn.ELU(), nn.Conv1d(40, feat, kernel_size=7, padding=3), nn.BatchNorm1d(feat), nn.ELU())
        self.n_windows = n_windows
        self.feat = feat
        self.attn = _ManifoldAttention(feat)
        self.logeig = LogEig()
        vlen = feat * (feat + 1) // 2
        self.classifier = nn.Sequential(nn.BatchNorm1d(vlen * n_windows), nn.Linear(vlen * n_windows, n_class))

    def _to_spd_tokens(self, h):
        B, C, T = h.shape
        n = self.n_windows
        pad = (n - T % n) % n
        if pad:
            h = F.pad(h, (0, pad))
            T = T + pad
        w = T // n
        chunks = h.reshape(B, C, n, w).permute(0, 2, 3, 1).contiguous()
        toks = _cov_over_time(chunks.reshape(B * n, w, C), shrink=TOK_SHRINK)
        return toks.reshape(B, n, C, C)

    def forward(self, x):
        h = self.front(x.transpose(1, 2))
        S = self._to_spd_tokens(h)
        S = self.attn(S)
        B, n, d, _ = S.shape
        L = self.logeig(S.reshape(B * n, d, d))
        v = _vech(L).reshape(B, -1)
        return self.classifier(v)
from torch.nn.utils import weight_norm as _wn

class _Chomp1d(nn.Module):

    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class _TemporalBlock(nn.Module):

    def __init__(self, n_in, n_out, k, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = _wn(nn.Conv1d(n_in, n_out, k, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = _Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = _wn(nn.Conv1d(n_out, n_out, k, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = _Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.drop1, self.conv2, self.chomp2, self.relu2, self.drop2)
        self.downsample = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else None
        self.relu = nn.ReLU()
        self._init()

    def _init(self):
        for c in (self.conv1, self.conv2):
            try:
                c.weight.data.normal_(0, 0.01)
            except Exception:
                pass
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(_BaseModel):

    def __init__(self, in_dim=FULL_DIM, channels=(64, 64, 64, 64), k=7, dropout=0.2, n_class=NUM_CLASS):
        super().__init__()
        layers = []
        for i in range(len(channels)):
            dil = 2 ** i
            n_in = in_dim if i == 0 else channels[i - 1]
            layers += [_TemporalBlock(n_in, channels[i], k, 1, dil, (k - 1) * dil, dropout)]
        self.network = nn.Sequential(*layers)
        self.classifier = nn.Linear(channels[-1], n_class)

    def forward(self, x):
        y = self.network(x.transpose(1, 2))
        return self.classifier(y.mean(dim=-1))

class _InceptionModule(nn.Module):

    def __init__(self, in_ch, nb_filters=32, bottleneck=32, ks=(40, 20, 10), use_bottleneck=True):
        super().__init__()
        self.use_bottleneck = use_bottleneck and in_ch > 1
        b_ch = bottleneck if self.use_bottleneck else in_ch
        if self.use_bottleneck:
            self.bottleneck = nn.Conv1d(in_ch, bottleneck, 1, padding='same', bias=False)
        self.convs = nn.ModuleList([nn.Conv1d(b_ch, nb_filters, k, padding='same', bias=False) for k in ks])
        self.maxpool = nn.MaxPool1d(3, stride=1, padding=1)
        self.conv_pool = nn.Conv1d(in_ch, nb_filters, 1, padding='same', bias=False)
        self.bn = nn.BatchNorm1d(nb_filters * (len(ks) + 1))
        self.act = nn.ReLU()

    def forward(self, x):
        inp = self.bottleneck(x) if self.use_bottleneck else x
        outs = [c(inp) for c in self.convs]
        outs.append(self.conv_pool(self.maxpool(x)))
        return self.act(self.bn(torch.cat(outs, dim=1)))

class InceptionTime(_BaseModel):

    def __init__(self, in_dim=FULL_DIM, nb_filters=32, depth=6, n_class=NUM_CLASS):
        super().__init__()
        self.depth = depth
        out_ch = nb_filters * 4
        self.inc = nn.ModuleList()
        self.shortcuts = nn.ModuleList()
        cur, res_in = (in_dim, in_dim)
        for d in range(depth):
            self.inc.append(_InceptionModule(cur, nb_filters))
            cur = out_ch
            if d % 3 == 2:
                self.shortcuts.append(nn.Sequential(nn.Conv1d(res_in, out_ch, 1, padding='same', bias=False), nn.BatchNorm1d(out_ch)))
                res_in = out_ch
        self.act = nn.ReLU()
        self.classifier = nn.Linear(out_ch, n_class)

    def forward(self, x):
        x = x.transpose(1, 2)
        res, out, si = (x, x, 0)
        for d in range(self.depth):
            out = self.inc[d](out)
            if d % 3 == 2:
                out = self.act(out + self.shortcuts[si](res))
                si += 1
                res = out
        return self.classifier(out.mean(dim=-1))

class BiLSTM(_BaseModel):

    def __init__(self, in_dim=FULL_DIM, hidden=128, layers=2, dropout=0.5, n_class=NUM_CLASS):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=layers, batch_first=True, bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        self.classifier = nn.Sequential(nn.LayerNorm(hidden * 2), nn.Dropout(dropout), nn.Linear(hidden * 2, n_class))

    def forward(self, x):
        h = torch.relu(self.proj(x))
        out, _ = self.lstm(h)
        return self.classifier(out.mean(dim=1))

class _SE1d(nn.Module):

    def __init__(self, ch, r=16):
        super().__init__()
        h = max(ch // r, 4)
        self.fc = nn.Sequential(nn.Linear(ch, h), nn.ReLU(inplace=True), nn.Linear(h, ch), nn.Sigmoid())

    def forward(self, x):
        s = self.fc(x.mean(dim=-1)).unsqueeze(-1)
        return x * s

class _SEResBlock(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1, r=16):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 7, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.se = _SE1d(out_ch, r)
        self.act = nn.ReLU(inplace=True)
        self.down = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False), nn.BatchNorm1d(out_ch))

    def forward(self, x):
        idt = x if self.down is None else self.down(x)
        o = self.act(self.bn1(self.conv1(x)))
        o = self.se(self.bn2(self.conv2(o)))
        return self.act(o + idt)

class SEResNet1D(_BaseModel):

    def __init__(self, in_dim=FULL_DIM, channels=(64, 128, 128), n_class=NUM_CLASS):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(in_dim, channels[0], 7, padding=3, bias=False), nn.BatchNorm1d(channels[0]), nn.ReLU(inplace=True))
        blocks, cur = ([], channels[0])
        for i, c in enumerate(channels):
            blocks.append(_SEResBlock(cur, c, stride=1 if i == 0 else 2))
            cur = c
        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Linear(cur, n_class)

    def forward(self, x):
        h = self.blocks(self.stem(x.transpose(1, 2)))
        return self.classifier(h.mean(dim=-1))

class _PosEnc(nn.Module):

    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class CNNTransformer(_BaseModel):

    def __init__(self, in_dim=FULL_DIM, d_model=128, nhead=8, layers=3, ff=256, dropout=0.3, n_class=NUM_CLASS):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(in_dim, d_model, 7, stride=2, padding=3), nn.BatchNorm1d(d_model), nn.GELU(), nn.Conv1d(d_model, d_model, 5, stride=2, padding=2), nn.BatchNorm1d(d_model), nn.GELU())
        self.pos = _PosEnc(d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout, batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.classifier = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_class))

    def forward(self, x):
        h = self.stem(x.transpose(1, 2)).transpose(1, 2)
        h = self.encoder(self.pos(h))
        return self.classifier(h.mean(dim=1))
MODEL_FACTORY = {'SPDNet': SPDNet, 'MAtt': MAtt, 'TCN': TCN, 'InceptionTime': InceptionTime, 'BiLSTM': BiLSTM, 'SEResNet1D': SEResNet1D, 'CNNTransformer': CNNTransformer}

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
    if not any((isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)) for m in model.modules())):
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

def train_one(splits, seed, model_name):
    set_seed(seed)
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = splits
    tr = DataLoader(SeqDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    va = DataLoader(SeqDataset(X_va, y_va), batch_size=BATCH_SIZE, shuffle=False)
    te = DataLoader(SeqDataset(X_te, y_te), batch_size=BATCH_SIZE, shuffle=False)
    net = MODEL_FACTORY[model_name]().to(DEVICE)
    n_params = sum((p.numel() for p in net.parameters()))
    n_train = sum((p.numel() for p in net.parameters() if p.requires_grad))
    opt = optim.AdamW(net.parameters(), lr=LR_MAIN, weight_decay=WEIGHT_DECAY)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=TOTAL_EPOCHS)
    swa = SWACollector(net)
    val_hist, tracc_hist, ce_hist = ([], [], [])
    best_smooth = 0.0
    best_wts = None
    best_epoch = -1
    t0 = time.time()
    for epoch in range(TOTAL_EPOCHS):
        net.train()
        ce_sum = 0.0
        nb = 0
        correct = 0
        total = 0
        for bx, by in tr:
            bx, by = (bx.to(DEVICE), by.to(DEVICE))
            opt.zero_grad()
            logits = net(bx)
            ce = F.cross_entropy(logits, by, label_smoothing=LABEL_SMOOTHING)
            loss = ce + net.lrr_regularization()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
            opt.step()
            ce_sum += float(ce)
            nb += 1
            with torch.no_grad():
                correct += (logits.argmax(1) == by).sum().item()
                total += by.size(0)
        sched.step()
        if epoch >= SWA_START:
            swa.accumulate(net)
        vt, vp = evaluate(net, va)
        va_acc = accuracy_score(vt, vp)
        val_hist.append(float(va_acc))
        tracc_hist.append(float(correct / max(total, 1)))
        ce_hist.append(float(ce_sum / max(nb, 1)))
        if epoch >= BEST_TRACK_START and len(val_hist) >= VAL_SMOOTH:
            sm = float(np.mean(val_hist[-VAL_SMOOTH:]))
            if sm > best_smooth:
                best_smooth = sm
                best_wts = copy.deepcopy(net.state_dict())
                best_epoch = epoch + 1
    swa.apply_to(net)
    update_bn(net, tr, DEVICE)
    swa_state = copy.deepcopy(net.state_dict())
    vt, vp = evaluate(net, va)
    swa_va = accuracy_score(vt, vp)
    final_va, selector = (swa_va, 'swa')
    if best_wts is not None:
        net.load_state_dict(best_wts)
        update_bn(net, tr, DEVICE)
        vt, vp = evaluate(net, va)
        bw_va = accuracy_score(vt, vp)
        if bw_va >= swa_va:
            final_va, selector = (bw_va, 'best_wts')
        else:
            net.load_state_dict(swa_state)
            update_bn(net, tr, DEVICE)
            best_epoch = -1
    train_time = time.time() - t0
    tt, tp = evaluate(net, te)
    test_acc = accuracy_score(tt, tp)
    macro_f1 = f1_score(tt, tp, average='macro', zero_division=0)
    per_class_f1 = f1_score(tt, tp, average=None, labels=list(range(NUM_CLASS)), zero_division=0).tolist()
    cm = confusion_matrix(tt, tp, labels=list(range(NUM_CLASS)))
    res = {'model': model_name, 'seed': int(seed), 'acc': float(test_acc), 'val_acc': float(final_va), 'selector': selector, 'f1': float(macro_f1), 'per_class_f1': per_class_f1, 'n_params': int(n_params), 'n_trainable': int(n_train), 'train_time_s': float(train_time), 'best_epoch': int(best_epoch), 'final_train_acc': float(tracc_hist[-1]) if tracc_hist else float('nan')}
    if LOG_PER_EPOCH:
        res['training_curves'] = {'val_acc': val_hist, 'train_acc': tracc_hist, 'train_ce': ce_hist}
    if SAVE_PREDICTIONS:
        res['predictions'] = {'y_true': tt.astype(int).tolist(), 'y_pred': tp.astype(int).tolist(), 'cm': cm.astype(int).tolist()}
    return res

def _save_csv(rows, path):
    if not rows:
        return
    keys = sorted(set().union(*[r.keys() for r in rows]))
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in keys})

def _strip(r):
    return {k: v for k, v in r.items() if k not in ('predictions', 'training_curves')}

def save_all(all_results, out_dir, elapsed):
    viz = os.path.join(out_dir, 'viz')
    os.makedirs(viz, exist_ok=True)
    cmdir = os.path.join(viz, 'confusion_matrices')
    os.makedirs(cmdir, exist_ok=True)
    stripped = {snr: {m: {int(s): _strip(r) for s, r in seeds.items()} for m, seeds in models.items()} for snr, models in all_results.items()}
    with open(os.path.join(out_dir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump({'snr_order': SNR_ORDER, 'seeds': SEEDS, 'epochs': TOTAL_EPOCHS, 'data_split_seed': DATA_SPLIT_SEED, 'wall_time_s': elapsed, 'results': stripped}, f, indent=2, ensure_ascii=False)
    per_seed, curves, per_class = ([], [], [])
    for snr in SNR_ORDER:
        for m, seeds in all_results.get(snr, {}).items():
            for s, r in seeds.items():
                row = {'snr': snr, 'model': m, 'seed': int(s), 'test_acc': r['acc'], 'val_acc': r['val_acc'], 'macro_f1': r['f1'], 'selector': r['selector'], 'n_trainable': r['n_trainable'], 'n_params': r['n_params'], 'train_time_s': r.get('train_time_s', float('nan')), 'best_epoch': r.get('best_epoch', -1)}
                for i, v in enumerate(r.get('per_class_f1', [])):
                    row[f'pcf1_{i:02d}'] = float(v)
                per_seed.append(row)
                for ci, v in enumerate(r.get('per_class_f1', [])):
                    per_class.append({'snr': snr, 'model': m, 'seed': int(s), 'class': CLASSES[ci], 'class_idx': ci, 'f1': float(v)})
                tc = r.get('training_curves', {})
                for ep, va in enumerate(tc.get('val_acc', [])):
                    curves.append({'snr': snr, 'model': m, 'seed': int(s), 'epoch': ep + 1, 'val_acc': float(va), 'train_acc': float(tc['train_acc'][ep]), 'train_ce': float(tc['train_ce'][ep])})
                if 'predictions' in r:
                    p = r['predictions']
                    np.savez_compressed(os.path.join(cmdir, f'cm_{m}_{snr}_{s}.npz'), y_true=np.array(p['y_true']), y_pred=np.array(p['y_pred']), cm=np.array(p['cm']), labels=np.array(CLASSES))
    _save_csv(per_seed, os.path.join(viz, 'per_seed.csv'))
    _save_csv(per_seed, os.path.join(out_dir, 'per_seed.csv'))
    _save_csv(curves, os.path.join(viz, 'training_curves.csv'))
    _save_csv(per_class, os.path.join(viz, 'per_class_f1.csv'))
    for snr in SNR_ORDER:
        for m, seeds in all_results.get(snr, {}).items():
            cms = [np.array(r['predictions']['cm']) for r in seeds.values() if 'predictions' in r]
            if cms:
                np.savez_compressed(os.path.join(viz, f'cm_aggregate_{m}_{snr}.npz'), cm=np.sum(cms, axis=0).astype(int), labels=np.array(CLASSES), n_seeds=len(cms))
    summ = []
    for snr in SNR_ORDER:
        for m, seeds in all_results.get(snr, {}).items():
            accs = np.array([r['acc'] for r in seeds.values()])
            f1s = np.array([r['f1'] for r in seeds.values()])
            if len(accs) == 0:
                continue
            summ.append({'snr': snr, 'model': m, 'n': len(accs), 'acc_mean': float(accs.mean()), 'acc_std': float(accs.std(ddof=1)) if len(accs) > 1 else 0.0, 'acc_min': float(accs.min()), 'acc_max': float(accs.max()), 'f1_mean': float(f1s.mean()), 'f1_std': float(f1s.std(ddof=1)) if len(f1s) > 1 else 0.0, 'n_trainable': int(next(iter(seeds.values()))['n_trainable'])})
    _save_csv(summ, os.path.join(viz, 'summary.csv'))
    _save_csv(summ, os.path.join(out_dir, 'summary.csv'))

def main():
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(SAVE_DIR, f'run_{ts}')
    os.makedirs(out_dir, exist_ok=True)
    models = list(MODELS_TO_RUN)
    print('=' * 100)
    print('  Baseline comparison  --  train/test/save  (OPTIMIZED)')
    print('=' * 100)
    print(f'  Models : {models}')
    print(f'  SNRs   : {SNR_ORDER}    Seeds : {SEEDS}    Epochs : {TOTAL_EPOCHS}    Device : {DEVICE}')
    print(f'  Standardize={STANDARDIZE}  Eigh_f64={EIGH_FLOAT64}  cov_shrink={COV_SHRINK}/{TOK_SHRINK}')
    print(f'  Output : {out_dir}')
    print('=' * 100)
    splits_by_snr = {}
    for snr in SNR_ORDER:
        root = DATA_ROOTS[snr]
        if not os.path.exists(root):
            print(f'  [skip] {snr}: data folder missing ({root})')
            continue
        X, y = load_all_data(root)
        splits = split_data(X, y, seed=DATA_SPLIT_SEED)
        splits = standardize_splits(splits)
        splits_by_snr[snr] = splits
        print(f'  [data] {snr:>5}  X={X.shape}  train/val/test = {len(splits[0][1])}/{len(splits[1][1])}/{len(splits[2][1])}')
    all_results = {snr: {m: {} for m in models} for snr in splits_by_snr}
    total = len(splits_by_snr) * len(models) * len(SEEDS)
    idx = 0
    t_global = time.time()
    for snr, splits in splits_by_snr.items():
        for m in models:
            print(f"\n{'#' * 100}\n  SNR={snr}   MODEL={m}\n{'#' * 100}")
            for seed in SEEDS:
                idx += 1
                t0 = time.time()
                try:
                    r = train_one(splits, seed, m)
                    all_results[snr][m][seed] = r
                    print(f"  [{idx:03d}/{total}]  {snr:>5}  {m:<14}  seed={seed:<5}  test={r['acc']:.4f}  val={r['val_acc']:.4f}  f1={r['f1']:.4f}  sel={r['selector']:<9}  params={r['n_trainable'] / 1000.0:.1f}k  ({time.time() - t0:.0f}s)")
                except Exception as e:
                    import traceback
                    print(f'  [{idx:03d}/{total}]  {snr:>5}  {m:<14}  seed={seed:<5}  FAILED: {type(e).__name__}: {e}')
                    traceback.print_exc()
                try:
                    save_all(all_results, out_dir, time.time() - t_global)
                except Exception as e:
                    print(f'  [warn] partial save failed: {e}')
    elapsed = time.time() - t_global
    save_all(all_results, out_dir, elapsed)
    print(f"\n{'=' * 100}\n  DONE.  wall time {elapsed / 60:.1f} min  ({elapsed / 3600:.2f} h)")
    print(f"  Results dir: {out_dir}\n{'=' * 100}")
if __name__ == '__main__':
    main()
