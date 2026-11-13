from __future__ import annotations
import re
import warnings
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
DATA_DIR = Path('data/audio')
OUT_DIR = Path('./figures_s32')
SR = 16000
FRAME_LEN_MS = 40
HOP_MS = 20
N_MELS_PER_VIEW = 16
K_VIEWS = 6
TOKEN_DIM = N_MELS_PER_VIEW
N_FILES_PER_CLASS = 8
N_FRAMES_PER_FILE = 80
H_HEADS = 4
BETA = 1.0
RNG = np.random.default_rng(42)
mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'], 'mathtext.fontset': 'stixsans', 'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 11.5, 'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5, 'legend.fontsize': 9.5, 'figure.dpi': 150, 'savefig.dpi': 600, 'axes.linewidth': 0.8, 'xtick.major.width': 0.8, 'ytick.major.width': 0.8, 'xtick.major.size': 3.0, 'ytick.major.size': 3.0, 'xtick.direction': 'out', 'ytick.direction': 'out', 'axes.axisbelow': True, 'axes.grid': False, 'pdf.fonttype': 42, 'ps.fonttype': 42})
FNAME_RE = re.compile('^(?P<cls>.+?)_(?P<date>\\d{6})_(?P<idx>\\d{4})\\.wav$', re.IGNORECASE)

def discover_files(root: Path):
    files = list(root.rglob('*.wav')) + list(root.rglob('*.WAV'))
    cls = {}
    for f in files:
        m = FNAME_RE.match(f.name)
        key = m.group('cls') if m else f.stem
        cls.setdefault(key, []).append(f)
    return cls

def load_audio(path: Path) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    y = y - y.mean()
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    return y.astype(np.float32)

def extract_logmel_views(y: np.ndarray):
    n_fft = int(SR * FRAME_LEN_MS / 1000)
    hop = int(SR * HOP_MS / 1000)
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop, center=True)) ** 2
    n_mels_total = K_VIEWS * N_MELS_PER_VIEW
    mel = librosa.feature.melspectrogram(S=S, sr=SR, n_mels=n_mels_total, power=2.0)
    logmel = np.log(mel + 1e-09).T
    if logmel.shape[0] > N_FRAMES_PER_FILE:
        start = (logmel.shape[0] - N_FRAMES_PER_FILE) // 2
        logmel = logmel[start:start + N_FRAMES_PER_FILE]
    return [logmel[:, k * TOKEN_DIM:(k + 1) * TOKEN_DIM] for k in range(K_VIEWS)]

def synth_logmel_views(seed: int):
    rng = np.random.default_rng(seed)
    out_views = []
    base = np.linspace(0, 1, TOKEN_DIM)
    templates = [0.4 + 1.4 * np.exp(-(base - 0.15) ** 2 / 0.02), 0.4 + 1.4 * np.exp(-(base - 0.5) ** 2 / 0.025), 0.4 + 1.4 * np.exp(-(base - 0.85) ** 2 / 0.02)]
    sigma = 0.5
    for k in range(K_VIEWS):
        template = templates[k // 2]
        frames = template[None, :] + sigma * stats.t.rvs(df=4.0, size=(N_FRAMES_PER_FILE, TOKEN_DIM), random_state=rng)
        out_views.append(np.log(np.maximum(frames, 0.001)))
    return out_views

def fit_gate_params(x: np.ndarray, eps: float=1e-06):
    c = np.median(x, axis=0, keepdims=True)
    s = stats.median_abs_deviation(x, axis=0, scale='normal')[None, :] + eps
    return (c, s)

def student_t_gate(x, c, s, nu=4.0):
    rho = (x - c) / s
    return (nu + 1.0) / (nu + rho * rho) * x

def tangent_embed(x, p, eps=1e-06):
    p_hat = p / (np.linalg.norm(p) + eps)
    xi = x - np.outer(x @ p_hat, p_hat)
    d = x.shape[1]
    return np.sqrt(d) * xi / (np.linalg.norm(xi, axis=1, keepdims=True) + eps)

def tangent_distance_logits(q, k, p_hat, eps=1e-09):
    qn = q / (np.linalg.norm(q, axis=1, keepdims=True) + eps)
    kn = k / (np.linalg.norm(k, axis=1, keepdims=True) + eps)
    p_hat = p_hat / (np.linalg.norm(p_hat) + eps)
    qp = qn - np.outer(qn @ p_hat, p_hat)
    kp = kn - np.outer(kn @ p_hat, p_hat)
    nq = (qp * qp).sum(axis=1, keepdims=True)
    nk = (kp * kp).sum(axis=1, keepdims=True).T
    cross = qp @ kp.T
    d2 = np.clip(nq + nk - 2.0 * cross, 0.0, None)
    return -d2

def dotprod_logits(q, k, eps=1e-09):
    d = q.shape[1]
    return q @ k.T / np.sqrt(d)

def softmax_rows(M):
    M = M - M.max(axis=1, keepdims=True)
    eM = np.exp(M)
    return eM / eM.sum(axis=1, keepdims=True)

def collect_token_frames():
    views_per_clip = []
    if HAS_LIBROSA and DATA_DIR.exists():
        cls = discover_files(DATA_DIR)
        cls = {k: v for k, v in cls.items() if v}
        if cls:
            n_clips = 0
            for name, files in cls.items():
                for f in files[:N_FILES_PER_CLASS]:
                    try:
                        views = extract_logmel_views(load_audio(f))
                        views_per_clip.append(views)
                        n_clips += 1
                    except Exception as e:
                        warnings.warn(f'skip {f.name}: {e}')
            print(f'[real data] {n_clips} clips read from {DATA_DIR}')
    if not views_per_clip:
        print('[fallback] synthesising 9 conditions x several clips of drone-like log-mel sub-bands (set DATA_DIR for real-data figures)')
        for ci in range(9):
            for ji in range(N_FILES_PER_CLASS):
                views_per_clip.append(synth_logmel_views(seed=100 * ci + ji))
    n_clips = len(views_per_clip)
    per_view_frames = []
    for k in range(K_VIEWS):
        stack = np.vstack([vc[k] for vc in views_per_clip])
        per_view_frames.append(stack)
    n_frames_per_view = per_view_frames[0].shape[0]
    Z_per_view = []
    for k, X in enumerate(per_view_frames):
        c, s = fit_gate_params(X)
        p_hat_init = X.mean(axis=0)
        Z_per_view.append(tangent_embed(student_t_gate(X, c, s), p_hat_init))
    Z_all = np.stack(Z_per_view, axis=1)
    return Z_all

def figure4(out_stem: Path) -> None:
    Z_all = collect_token_frames()
    N_frames, K, d = Z_all.shape
    print(f'Z_all shape: {Z_all.shape}  (N_frames, K, d) ')
    rng = np.random.default_rng(123)
    Z_flat = Z_all.reshape(-1, d)
    _, _, Vt = np.linalg.svd(Z_flat - Z_flat.mean(0), full_matrices=False)
    head_p_hats = [Vt[i] for i in range(H_HEADS)]
    rng_b = np.random.default_rng(7)
    W_q_b = rng_b.standard_normal((d, d)) / np.sqrt(d)
    W_k_b = rng_b.standard_normal((d, d)) / np.sqrt(d)
    n_sample = min(800, N_frames)
    sample_idx = rng_b.choice(N_frames, size=n_sample, replace=False)
    Z_sample = Z_all[sample_idx]
    ours_logits, std_logits_all = ([], [])
    ours_max, std_max = ([], [])
    ours_H, std_H = ([], [])
    D_max_list = []

    def _entropy(p, axis=-1):
        p = np.clip(p, 1e-12, 1.0)
        return -(p * np.log(p)).sum(axis=axis)
    for f in range(n_sample):
        Z_f = Z_sample[f]
        Qf = Z_f @ W_q_b
        Kf = Z_f @ W_k_b
        std_logits_f = dotprod_logits(Qf, Kf)
        std_logits_all.extend(std_logits_f.ravel())
        A_std = softmax_rows(std_logits_f)
        std_max.extend(A_std.max(axis=1))
        std_H.extend(_entropy(A_std))
        for ph in head_p_hats:
            raw = tangent_distance_logits(Qf, Kf, ph)
            ours_logits.extend((BETA * raw).ravel())
            A_h = softmax_rows(BETA * raw)
            ours_max.extend(A_h.max(axis=1))
            ours_H.extend(_entropy(A_h))
            D_max_list.append(-raw.min())
    ours_logits = np.array(ours_logits)
    std_logits = np.array(std_logits_all)
    ours_max = np.array(ours_max)
    std_max = np.array(std_max)
    ours_H = np.array(ours_H)
    std_H = np.array(std_H)
    D_max = float(np.percentile(D_max_list, 95))
    logit_lo_bound = -BETA * D_max
    nc_bound = 1.0 / (1.0 + (K - 1) * np.exp(-BETA * D_max))
    H_floor = float(np.log(1.0 + (K - 1) * np.exp(-BETA * D_max)))
    H_max = float(np.log(K))
    pct_std_logit_violation = 100.0 * (std_logits > 0).mean()
    pct_std_collapse = 100.0 * (std_max > nc_bound).mean()
    pct_std_H_violation = 100.0 * (std_H < H_floor).mean()
    pct_ours_logit_violation = 100.0 * (ours_logits > 0).mean()
    pct_ours_collapse = 100.0 * (ours_max > nc_bound).mean()
    pct_ours_H_violation = 100.0 * (ours_H < H_floor).mean()
    C_OURS = '#2e8b57'
    C_OURS_E = '#1d6b40'
    C_STD = '#9aa5b1'
    C_STD_E = '#6b7785'
    C_BOUND = '#c0392b'
    C_DANGER_F = '#f9e4e1'
    C_SAFE_F = '#eaf3ec'
    GRID_C = '#eef0f2'

    def _style_panel(ax, letter, title):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#444')
        ax.spines['bottom'].set_color('#444')
        ax.tick_params(colors='#444')
        ax.yaxis.grid(True, color=GRID_C, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.text(-0.135, 1.09, letter, transform=ax.transAxes, fontsize=15, fontweight='bold', va='top', color='#111')
        ax.text(0.0, 1.09, title, transform=ax.transAxes, fontsize=10.5, va='top', color='#333', fontweight='semibold')

    def _bound_tick(ax, x, label):
        ax.annotate(label, xy=(x, 0), xycoords=('data', 'axes fraction'), xytext=(0, -15.5), textcoords='offset points', ha='center', va='top', color=C_BOUND, fontsize=9, fontweight='bold')

    def _hist(ax, data, bins, color, edge, label):
        return ax.hist(data, bins=bins, density=True, alpha=0.9, color=color, edgecolor=edge, linewidth=0.4, label=label, zorder=3)
    fig = plt.figure(figsize=(15.0, 4.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.0], wspace=0.255, left=0.052, right=0.99, top=0.83, bottom=0.155)
    ax = fig.add_subplot(gs[0])
    bin_lo = min(ours_logits.min(), std_logits.min())
    bin_hi = max(0.5, std_logits.max() + 0.3)
    bins = np.linspace(bin_lo - 0.2, bin_hi + 0.2, 64)
    ax.axvspan(0.0, bin_hi + 0.3, color=C_DANGER_F, lw=0, zorder=0)
    ax.axvspan(logit_lo_bound, 0.0, color=C_SAFE_F, lw=0, zorder=0)
    h_std = _hist(ax, std_logits, bins, C_STD, C_STD_E, 'standard')
    h_our = _hist(ax, ours_logits, bins, C_OURS, C_OURS_E, 'ours')
    ax.axvline(0.0, color=C_BOUND, linewidth=1.8, linestyle=(0, (5, 3)), zorder=4)
    ax.axvline(logit_lo_bound, color=C_BOUND, linewidth=1.1, linestyle=(0, (1.6, 1.8)), zorder=4)
    ax.set_xlabel('attention logit value')
    ax.set_ylabel('density')
    ax.set_xlim(bin_lo - 0.1, bin_hi + 0.3)
    _bound_tick(ax, logit_lo_bound, f'{logit_lo_bound:.2f}')
    _bound_tick(ax, 0.0, '0')
    _style_panel(ax, 'a', 'Bounded logits')
    ax = fig.add_subplot(gs[1])
    bins = np.linspace(0, 1.0, 56)
    ax.axvspan(nc_bound, 1.0, color=C_DANGER_F, lw=0, zorder=0)
    ax.axvspan(0.0, nc_bound, color=C_SAFE_F, lw=0, zorder=0)
    _hist(ax, std_max, bins, C_STD, C_STD_E, 'standard')
    _hist(ax, ours_max, bins, C_OURS, C_OURS_E, 'ours')
    ax.axvline(nc_bound, color=C_BOUND, linewidth=1.8, linestyle=(0, (5, 3)), zorder=4)
    ax.axvline(1.0, color=C_BOUND, linewidth=1.0, linestyle=(0, (1.6, 1.8)), zorder=4)
    ax.set_xlabel('max attention weight  $\\max_j A_{ij}$')
    ax.set_ylabel('density')
    ax.set_xlim(0, 1.0)
    _bound_tick(ax, nc_bound, f'{nc_bound:.2f}')
    _style_panel(ax, 'b', 'Non-collapse')
    ax = fig.add_subplot(gs[2])
    ours_support = np.exp(ours_H)
    std_support = np.exp(std_H)
    support_floor = float(np.exp(H_floor))
    bins = np.linspace(1.0, K, 56)
    ax.axvspan(1.0, support_floor, color=C_DANGER_F, lw=0, zorder=0)
    ax.axvspan(support_floor, K, color=C_SAFE_F, lw=0, zorder=0)
    _hist(ax, std_support, bins, C_STD, C_STD_E, 'standard')
    _hist(ax, ours_support, bins, C_OURS, C_OURS_E, 'ours')
    ax.axvline(support_floor, color=C_BOUND, linewidth=1.8, linestyle=(0, (5, 3)), zorder=4)
    ax.axvline(K, color=C_BOUND, linewidth=1.0, linestyle=(0, (1.6, 1.8)), zorder=4)
    ax.set_xlabel('effective attention support  $\\exp \\mathcal{H}(A_{i,\\cdot})$')
    ax.set_ylabel('density')
    ax.set_xlim(1.0, K)
    _bound_tick(ax, support_floor, f'{support_floor:.2f}')
    _style_panel(ax, 'c', 'Effective support')
    fig.legend(handles=[h_our[2][0], h_std[2][0]], labels=['ours', 'standard'], loc='upper center', ncol=2, frameon=False, handlelength=1.5, columnspacing=1.8, fontsize=10.5, bbox_to_anchor=(0.5, 1.005))
    plt.savefig(out_stem.with_suffix('.pdf'), bbox_inches='tight')
    plt.savefig(out_stem.with_suffix('.png'), bbox_inches='tight', dpi=400)
    plt.close()
    print(f'\n--- summary stats for caption ---')
    print(f'D_max (95th %ile across heads) = {D_max:.3f}, beta = {BETA}, K = {K}')
    print(f'Theoretical bounds:')
    print(f'   logit lower bound  -beta D_max  = {logit_lo_bound:.3f}')
    print(f'   non-collapse bound              = {nc_bound:.3f}')
    print(f'   entropy floor                   = {H_floor:.3f}')
    print(f'   log K (max entropy)             = {H_max:.3f}')
    print(f'Bound-violation rates:')
    print(f'   logit > 0     : ours = {pct_ours_logit_violation:.2f}%, standard = {pct_std_logit_violation:.2f}%')
    print(f'   max > nc_bound: ours = {pct_ours_collapse:.2f}%, standard = {pct_std_collapse:.2f}%')
    print(f'   entropy<floor : ours = {pct_ours_H_violation:.2f}%, standard = {pct_std_H_violation:.2f}%')
    print(f'Panel (c) effective attention support  exp(H)  (K = {K}):')
    print(f'   support floor (= exp(H_floor))  = {float(np.exp(H_floor)):.3f}')
    print(f'   mean      : ours = {float(ours_support.mean()):.3f}, standard = {float(std_support.mean()):.3f}')
    print(f'   median    : ours = {float(np.median(ours_support)):.3f}, standard = {float(np.median(std_support)):.3f}')
    print(f'   5th pctile: ours = {float(np.percentile(ours_support, 5)):.3f}, standard = {float(np.percentile(std_support, 5)):.3f}')
    print(f'   1st pctile: ours = {float(np.percentile(ours_support, 1)):.3f}, standard = {float(np.percentile(std_support, 1)):.3f}')

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print('rendering Figure 4 ...')
    figure4(OUT_DIR / 'figure4')
    print(f'\nwritten to {OUT_DIR.resolve()}/figure4.{{pdf,png}}')
if __name__ == '__main__':
    main()
