import numpy as np
from scipy.signal import hilbert, get_window
from scipy.fft import rfft, dct
FS = 4000.0
DUR = 2.0
N_SAMP = int(FS * DUR)
F_B = 900.0
BETA_DEC = 30.0
D_VIEW = 32
V_NUM = 6
D_SIGMA = D_VIEW * V_NUM
PROJ_OUT = 64
LRR_RANK = 8
FEATURE_NAMES = ['HHT', 'LOFAR', 'FFT', 'WPT', 'HHSA', 'MFCC']
CLASS_PARAMS = {0: (100, 8, 3), 1: (120, 10, 4), 2: (140, 12, 5), 3: (160, 14, 3), 4: (180, 16, 4), 5: (200, 8, 5), 6: (220, 10, 3), 7: (240, 12, 4), 8: (260, 14, 5), 9: (280, 16, 3)}
NUM_CLASS = len(CLASS_PARAMS)

def _lowpass_white(n, fc, fs, rng):
    x = rng.standard_normal(n)
    X = rfft(x)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    X[f > fc] = 0.0
    y = np.fft.irfft(X, n=n)
    return y / (y.std() + 1e-12)

def clean_source(cls, rng, fs=FS, dur=DUR):
    fc, fm, fr = CLASS_PARAMS[cls]
    t = np.arange(int(fs * dur)) / fs
    rotor = np.sin(2 * np.pi * fc * t) + 0.5 * np.sin(2 * np.pi * 2 * fc * t)
    w = _lowpass_white(len(t), fc=1000.0, fs=fs, rng=rng)
    aero = 0.4 * w * (1.0 + 0.8 * np.sin(2 * np.pi * fm * t))
    env = np.exp(-BETA_DEC * (t % 1.0 / fr))
    transient = 0.8 * np.sin(2 * np.pi * F_B * t) * env
    return (t, rotor + aero + transient)

def degrade(s, rng, fs=FS, snr_db=0.0, sir_db=5.0, taus=(0, 3, 7), alphas=(1.0, 0.5, 0.3), turb_strength=0.3, doppler_dev=5.0, doppler_rate=0.5, sweep_lo=50.0, sweep_hi=350.0, return_parts=False):
    n = len(s)
    t = np.arange(n) / fs
    mp = np.zeros_like(s)
    for a, tau in zip(alphas, taus):
        mp += a * np.roll(s, int(tau))
    g = _lowpass_white(n, fc=5.0, fs=fs, rng=rng)
    A_twrb = np.exp(turb_strength * g)
    inst = 2 * np.pi * doppler_dev * np.sin(2 * np.pi * doppler_rate * t)
    phi = np.cumsum(inst) / fs
    analytic = hilbert(mp * A_twrb) * np.exp(1j * phi)
    core = np.real(analytic)
    Psig = core.var() + 1e-12
    f_inst = sweep_lo + (sweep_hi - sweep_lo) * (t / t[-1])
    J = np.sin(2 * np.pi * np.cumsum(f_inst) / fs)
    J *= np.sqrt(Psig / (J.var() + 1e-12) / 10 ** (sir_db / 10.0))
    v = rng.standard_normal(n)
    v *= np.sqrt(Psig / 10 ** (snr_db / 10.0))
    y = core + J + v
    if return_parts:
        return (t, y, dict(core=core, multipath=mp, A_twrb=A_twrb, phi=phi, J=J, v=v))
    return (t, y)

def _frame(x, frame_len, hop):
    n_frames = 1 + (len(x) - frame_len) // hop
    idx = np.arange(frame_len)[None, :] + hop * np.arange(n_frames)[:, None]
    return x[idx]

def _resize_bins(mat, d):
    T, B = mat.shape
    edges = np.linspace(0, B, d + 1).astype(int)
    out = np.empty((T, d), dtype=mat.dtype)
    for i in range(d):
        a, b = (edges[i], max(edges[i] + 1, edges[i + 1]))
        out[:, i] = mat[:, a:b].mean(axis=1)
    return out

def feat_FFT(x, fs=FS, d=D_VIEW, frame_len=256, hop=128):
    fr = _frame(x, frame_len, hop) * get_window('hann', frame_len)[None, :]
    mag = np.abs(rfft(fr, axis=1))
    return _resize_bins(mag, d)

def feat_LOFAR(x, fs=FS, d=D_VIEW, frame_len=256, hop=128):
    fr = _frame(x, frame_len, hop) * get_window('hann', frame_len)[None, :]
    p = np.abs(rfft(fr, axis=1)) ** 2
    lofar = 10.0 * np.log10(p + 1e-08)
    return _resize_bins(lofar, d)

def feat_HHT(x, fs=FS, d=D_VIEW, frame_len=256, hop=128):
    env = np.abs(hilbert(x))
    fr = _frame(env, frame_len, hop)
    return _resize_bins(np.abs(rfft(fr, axis=1)), d)

def feat_HHSA(x, fs=FS, d=D_VIEW, frame_len=256, hop=128):
    a = hilbert(x)
    inst_phase = np.unwrap(np.angle(a))
    inst_freq = np.diff(inst_phase) * fs / (2 * np.pi)
    inst_freq = np.concatenate([inst_freq, inst_freq[-1:]])
    fr = _frame(inst_freq, frame_len, hop)
    return _resize_bins(np.abs(rfft(fr, axis=1)), d)

def feat_WPT(x, fs=FS, d=D_VIEW, frame_len=256, hop=128, levels=5):
    fr = _frame(x, frame_len, hop)
    bands = [fr]
    for _ in range(levels):
        nxt = []
        for b in bands:
            lo = (b[:, ::2] + b[:, 1::2]) / np.sqrt(2.0)
            hi = (b[:, ::2] - b[:, 1::2]) / np.sqrt(2.0)
            nxt += [lo, hi]
        bands = nxt
    energy = np.stack([np.mean(b ** 2, axis=1) for b in bands], axis=1)
    return _resize_bins(energy, d)

def feat_MFCC(x, fs=FS, d=D_VIEW, frame_len=256, hop=128, n_mel=40):
    fr = _frame(x, frame_len, hop) * get_window('hann', frame_len)[None, :]
    p = np.abs(rfft(fr, axis=1)) ** 2
    n_fft_bins = p.shape[1]
    f_hz = np.linspace(0, fs / 2, n_fft_bins)
    mel = 2595.0 * np.log10(1 + f_hz / 700.0)
    mel_edges = np.linspace(mel.min(), mel.max(), n_mel + 2)
    fb = np.zeros((n_mel, n_fft_bins))
    for m in range(1, n_mel + 1):
        l, c, r = (mel_edges[m - 1], mel_edges[m], mel_edges[m + 1])
        left = (mel >= l) & (mel <= c)
        right = (mel >= c) & (mel <= r)
        fb[m - 1, left] = (mel[left] - l) / (c - l + 1e-09)
        fb[m - 1, right] = (r - mel[right]) / (r - c + 1e-09)
    mel_e = np.log(p @ fb.T + 1e-08)
    cc = dct(mel_e, type=2, norm='ortho', axis=1)
    return _resize_bins(cc, d)
FEATURE_FUNCS = {'HHT': feat_HHT, 'LOFAR': feat_LOFAR, 'FFT': feat_FFT, 'WPT': feat_WPT, 'HHSA': feat_HHSA, 'MFCC': feat_MFCC}

def extract_views(x):
    views = {name: FEATURE_FUNCS[name](x) for name in FEATURE_NAMES}
    T = min((v.shape[0] for v in views.values()))
    views = {k: v[:T] for k, v in views.items()}
    return (views, T)

def student_t_gate_weight(rho, nu):
    return (nu + 1.0) / (nu + rho ** 2)

def spherical_tangent_embed(Z, p=None, d=None, bias=0.0, eps=1e-09):
    Z = np.atleast_2d(Z)
    D = Z.shape[1]
    d = D if d is None else d
    if p is None:
        p = np.ones(D)
    p = p / (np.linalg.norm(p) + eps)
    par = (Z @ p)[:, None]
    tang = Z - par * p
    tn = np.linalg.norm(tang, axis=1, keepdims=True) + eps
    return np.sqrt(d) * tang / tn + bias

def _log_map(x_unit, p_unit):
    inner = (x_unit * p_unit).sum(axis=-1, keepdims=True)
    return x_unit - inner * p_unit

def logmap_tangent_attention(tokens, base_point, beta=0.5, eps=1e-09, return_logits=False):
    X = np.atleast_2d(tokens)
    p = base_point / (np.linalg.norm(base_point) + eps)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)
    xi = _log_map(Xn, p)
    nrm = (xi * xi).sum(axis=1)
    d2 = nrm[:, None] + nrm[None, :] - 2.0 * (xi @ xi.T)
    d2 = np.clip(d2, 0.0, None)
    logits = -beta * d2
    A = np.exp(logits - logits.max(axis=1, keepdims=True))
    A = A / A.sum(axis=1, keepdims=True)
    if return_logits:
        return (A, logits, d2)
    return A

def standard_dot_attention(tokens, eps=1e-09, return_logits=False):
    X = np.atleast_2d(tokens)
    D = X.shape[1]
    logits = X @ X.T / np.sqrt(D)
    A = np.exp(logits - logits.max(axis=1, keepdims=True))
    A = A / A.sum(axis=1, keepdims=True)
    if return_logits:
        return (A, logits, None)
    return A

def diff_nsllr_Z(feat, rank=8, lam=0.1, rng=None, U=None, Vv=None, return_factors=False):
    feat = np.atleast_2d(feat)
    T, dft = feat.shape
    if U is None or Vv is None:
        rng = np.random.default_rng(0) if rng is None else rng
        Wu, _ = np.linalg.qr(rng.standard_normal((dft, rank)))
        Wv, _ = np.linalg.qr(rng.standard_normal((dft, rank)))
        U = feat @ Wu
        Vv = feat @ Wv
    Z_raw = U @ Vv.T / np.sqrt(rank)
    Z = np.clip(Z_raw - lam, 0.0, None)
    if return_factors:
        return (Z, Z_raw, U, Vv)
    return Z

def layernorm(x, eps=1e-06):
    x = np.atleast_2d(x)
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    return (x - mu) / (sd + eps)

def build_view_bank(n_per_class=12, snr_db=0.0, sir_db=5.0, seed=2026, classes=None, frame_index=None):
    rng = np.random.default_rng(seed)
    classes = list(CLASS_PARAMS) if classes is None else classes
    per_view = {name: [] for name in FEATURE_NAMES}
    labels = []
    Tref = None
    for cls in classes:
        for _ in range(n_per_class):
            _, s = clean_source(cls, rng)
            _, y = degrade(s, rng, snr_db=snr_db, sir_db=sir_db)
            views, T = extract_views(y)
            if Tref is None:
                Tref = T
            for name in FEATURE_NAMES:
                per_view[name].append(views[name][:Tref])
            labels.append(cls)
    Xv = {name: np.asarray(v) for name, v in per_view.items()}
    return (Xv, np.asarray(labels), Tref)

def apply_style():
    import matplotlib as mpl
    mpl.rcParams.update({'figure.dpi': 130, 'savefig.dpi': 300, 'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'mathtext.fontset': 'dejavuserif', 'axes.linewidth': 1.1, 'axes.edgecolor': '#333333', 'axes.labelsize': 12, 'axes.titlesize': 12, 'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9, 'legend.frameon': True, 'legend.edgecolor': '#cccccc', 'grid.color': '#dddddd', 'grid.linewidth': 0.7})
C_INK = '#1f3b57'
C_OURS = '#2f7d4f'
C_STD = '#9aa3ad'
C_WARN = '#c0392b'
C_AMBER = '#e0a526'
C_TEAL = '#3a86a8'
VIEW_COLORS = ['#1f3b57', '#3a86a8', '#5fa8c7', '#2f7d4f', '#7bbf8f', '#e0a526']
