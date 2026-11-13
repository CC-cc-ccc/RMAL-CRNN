from __future__ import annotations
import re
import warnings
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
DATA_DIR = Path('data/audio')
OUT_DIR = Path('./figures_s33')
SR = 16000
FRAME_LEN_MS = 40
HOP_MS = 20
N_MELS_PER_VIEW = 16
K_VIEWS = 6
TOKEN_DIM = N_MELS_PER_VIEW
T_FRAMES = 80
PROJ_OUT = 64
LRR_RANK = 8
LRR_LAM = 0.1
LRR_ALPHA = 0.3
RNG = np.random.default_rng(42)
mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'], 'mathtext.fontset': 'stixsans', 'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 11.5, 'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5, 'legend.fontsize': 9.5, 'figure.dpi': 150, 'savefig.dpi': 600, 'axes.linewidth': 0.8, 'xtick.major.width': 0.8, 'ytick.major.width': 0.8, 'xtick.major.size': 3.0, 'ytick.major.size': 3.0, 'xtick.direction': 'out', 'ytick.direction': 'out', 'axes.axisbelow': True, 'pdf.fonttype': 42, 'ps.fonttype': 42})
FNAME_RE = re.compile('^(?P<cls>.+?)_(?P<date>\\d{6})_(?P<idx>\\d{4})\\.wav$', re.IGNORECASE)

def discover_files(root: Path):
    files = list(root.rglob('*.wav')) + list(root.rglob('*.WAV'))
    out = []
    for f in files:
        out.append(f)
    return out

def load_audio(path: Path) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    y = y - y.mean()
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    return y.astype(np.float32)

def extract_logmel_seq(y: np.ndarray) -> np.ndarray:
    n_fft = int(SR * FRAME_LEN_MS / 1000)
    hop = int(SR * HOP_MS / 1000)
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop, center=True)) ** 2
    mel = librosa.feature.melspectrogram(S=S, sr=SR, n_mels=K_VIEWS * TOKEN_DIM, power=2.0)
    logmel = np.log(mel + 1e-09).T
    if logmel.shape[0] >= T_FRAMES:
        start = (logmel.shape[0] - T_FRAMES) // 2
        logmel = logmel[start:start + T_FRAMES]
    else:
        pad = T_FRAMES - logmel.shape[0]
        logmel = np.pad(logmel, ((0, pad), (0, 0)), mode='edge')
    return logmel

def synth_structured_seq(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    d = K_VIEWS * TOKEN_DIM
    n_seg = 4
    seg_bounds = np.linspace(0, T_FRAMES, n_seg + 1).astype(int)
    F = np.zeros((T_FRAMES, d))
    for s in range(n_seg):
        lo, hi = (seg_bounds[s], seg_bounds[s + 1])
        basis = rng.standard_normal((2, d))
        coeff = rng.standard_normal((hi - lo, 2))
        seg = coeff @ basis
        seg = seg + 0.25 * stats.t.rvs(df=4.0, size=(hi - lo, d), random_state=rng)
        F[lo:hi] = seg
    return F

def fit_gate_params(x, eps=1e-06):
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

def phi_k_pipeline(F_raw):
    T = F_raw.shape[0]
    outs = []
    for k in range(K_VIEWS):
        block = F_raw[:, k * TOKEN_DIM:(k + 1) * TOKEN_DIM]
        c, s = fit_gate_params(block)
        p0 = block.mean(axis=0)
        outs.append(tangent_embed(student_t_gate(block, c, s), p0))
    return np.concatenate(outs, axis=1)

def layernorm_rows(X, eps=1e-06):
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True) + eps
    return (X - mu) / sd

def nsllr_forward(F, r=LRR_RANK, lam=LRR_LAM, alpha=LRR_ALPHA, seed=0):
    rng = np.random.default_rng(seed)
    d = F.shape[1]
    Wu = np.linalg.qr(rng.standard_normal((d, r)))[0].T
    Wv = np.linalg.qr(rng.standard_normal((d, r)))[0].T
    U = F @ Wu.T
    V = F @ Wv.T
    Z_raw = U @ V.T / np.sqrt(r)
    Z = np.clip(Z_raw - lam, 0.0, None)
    F_ref = layernorm_rows(Z @ F)
    F_out = (1 - alpha) * F + alpha * F_ref
    return dict(Z_raw=Z_raw, Z=Z, F_ref=F_ref, F_out=F_out, d=d, alpha=alpha, r=r, lam=lam)

def collect_sequences():
    rng_f = np.random.default_rng(2024)
    Wf = np.linalg.qr(rng_f.standard_normal((K_VIEWS * TOKEN_DIM, PROJ_OUT)))[0]
    seqs = []
    if HAS_LIBROSA and DATA_DIR.exists():
        files = discover_files(DATA_DIR)
        if files:
            idx = RNG.permutation(len(files))[:40]
            for j in idx:
                try:
                    raw = extract_logmel_seq(load_audio(files[j]))
                    tok = phi_k_pipeline(raw)
                    seqs.append(tok @ Wf)
                except Exception as e:
                    warnings.warn(f'skip {files[j].name}: {e}')
            print(f'[real data] {len(seqs)} clip sequences from {DATA_DIR}')
            if seqs:
                return seqs
    print('[fallback] synthesising structured sequences (set DATA_DIR for real-data figures)')
    for s in range(40):
        tok = phi_k_pipeline(synth_structured_seq(seed=s))
        seqs.append(tok @ Wf)
    return seqs

def figure5(out_stem: Path) -> None:
    seqs = collect_sequences()
    rep = seqs[len(seqs) // 2]
    res = nsllr_forward(rep, seed=7)
    d = res['d']
    alpha = res['alpha']
    r = res['r']
    T = rep.shape[0]
    sqrt_d = np.sqrt(d)
    sv_raw_all, sv_full_all = ([], [])
    for F in seqs:
        rr = nsllr_forward(F, seed=7)
        s_raw = np.linalg.svd(rr['Z_raw'], compute_uv=False)
        Z_full = F @ F.T / np.sqrt(d)
        s_full = np.linalg.svd(Z_full, compute_uv=False)
        sv_raw_all.append(s_raw / (s_raw[0] + 1e-12))
        sv_full_all.append(s_full / (s_full[0] + 1e-12))
    L = min(min((len(s) for s in sv_full_all)), 48)
    sv_raw = np.mean([s[:L] for s in sv_raw_all], axis=0)
    sv_full = np.mean([s[:L] for s in sv_full_all], axis=0)
    in_norms, out_norms, out_noln = ([], [], [])
    for F in seqs:
        rng = np.random.default_rng(7)
        Wu = np.linalg.qr(rng.standard_normal((F.shape[1], r)))[0].T
        Wv = np.linalg.qr(rng.standard_normal((F.shape[1], r)))[0].T
        U = F @ Wu.T
        V = F @ Wv.T
        Z = np.clip(U @ V.T / np.sqrt(r) - LRR_LAM, 0.0, None)
        ZF = Z @ F
        F_ref_ln = layernorm_rows(ZF)
        F_out_ln = (1 - alpha) * F + alpha * F_ref_ln
        F_out_no = (1 - alpha) * F + alpha * ZF
        in_norms.extend(np.linalg.norm(F, axis=1))
        out_norms.extend(np.linalg.norm(F_out_ln, axis=1))
        out_noln.extend(np.linalg.norm(F_out_no, axis=1))
    in_norms = np.array(in_norms)
    out_norms = np.array(out_norms)
    out_noln = np.array(out_noln)
    med_noln = float(np.median(out_noln))
    med_ours = float(np.median(out_norms))
    T_grid = np.array([16, 32, 64, 128, 256, 512, 1024], dtype=float)
    n_iter_lrr = 100.0
    flops_lrr = n_iter_lrr * T_grid ** 3
    flops_ours = T_grid ** 2 * d
    speedup_at_T = float(n_iter_lrr * T ** 3 / (T ** 2 * d))
    C_OURS = '#2e8b57'
    C_BASE = '#9aa5b1'
    C_BOUND = '#c0392b'
    fig = plt.figure(figsize=(15.0, 4.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.0], wspace=0.28, left=0.055, right=0.985, top=0.85, bottom=0.16)

    def _panel_letter(ax, letter, title, dx=-0.17):
        ax.text(dx, 1.09, letter, transform=ax.transAxes, fontsize=15, fontweight='bold', va='top', color='#111')
        ax.text(dx + 0.105, 1.09, title, transform=ax.transAxes, fontsize=10.5, va='top', color='#333', fontweight='semibold')
    ax = fig.add_subplot(gs[0])
    ax.fill_between(T_grid, flops_ours, flops_lrr, color=C_OURS, alpha=0.1, zorder=0)
    ax.loglog(T_grid, flops_lrr, color=C_BASE, lw=2.0, marker='o', ms=4, label='nuclear-norm LRR  $\\mathcal{O}(n_{\\rm it}T^3)$', zorder=3)
    ax.loglog(T_grid, flops_ours, color=C_OURS, lw=2.2, marker='s', ms=4, label='NSLLR (ours)  $\\mathcal{O}(T^2 d)$', zorder=4)
    Tplot = T_grid[np.argmin(np.abs(T_grid - T))]
    ax.annotate(f'$\\approx{speedup_at_T:.0f}\\times$ fewer FLOPs at $T={T}$', xy=(Tplot, Tplot ** 2 * d), xytext=(Tplot * 0.55, Tplot ** 2 * d * 18), fontsize=9.3, color=C_OURS, fontweight='semibold', va='center', ha='center', arrowprops=dict(arrowstyle='->', color=C_OURS, lw=1.1, connectionstyle='arc3,rad=0.2'))
    ax.set_xlabel('sequence length  $T$')
    ax.set_ylabel('FLOPs  (relative)')
    ax.legend(loc='upper left', frameon=False, handlelength=1.8, fontsize=9.0, labelspacing=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, which='major', color='#eef0f2', lw=0.7)
    _panel_letter(ax, 'a', 'Computational cost')
    ax = fig.add_subplot(gs[1])
    ks = np.arange(1, L + 1)
    ax.fill_between(ks, 0.0001, sv_raw, color=C_OURS, alpha=0.12, zorder=1)
    ax.semilogy(ks, sv_full, color=C_BASE, lw=2.0, marker='o', ms=3.2, label='full self-expression  $FF^\\top$', zorder=3)
    ax.semilogy(ks, sv_raw, color=C_OURS, lw=2.2, marker='s', ms=3.4, label='low-rank  $Z_{\\rm raw}=UV^\\top/\\sqrt{r}$', zorder=4)
    ax.axvline(r + 0.5, color=C_BOUND, lw=1.6, linestyle=(0, (5, 3)), zorder=5)
    ax.annotate(f'rank $\\leq r={r}$\n(vs full rank $d$)', xy=(r + 0.5, 0.012), xytext=(r + 4.0, 0.04), fontsize=9.3, color=C_BOUND, fontweight='semibold', va='center', ha='left', arrowprops=dict(arrowstyle='->', color=C_BOUND, lw=1.1, connectionstyle='arc3,rad=-0.2'))
    ax.set_xlabel('singular-value index')
    ax.set_ylabel('normalised singular value')
    ax.set_xlim(1, L)
    ax.set_ylim(0.001, 1.4)
    ax.legend(loc='upper right', frameon=False, handlelength=1.8, fontsize=9.0, labelspacing=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, color='#eef0f2', lw=0.7)
    _panel_letter(ax, 'b', 'Rank budget')
    ax = fig.add_subplot(gs[2])
    xmax = max(in_norms.max(), sqrt_d) * 1.05
    xs = np.linspace(0, xmax, 200)
    envelope = (1 - alpha) * xs + alpha * sqrt_d
    ax.fill_between(xs, 0, envelope, color=C_OURS, alpha=0.08, zorder=0)
    ax.scatter(in_norms, out_noln, s=8, color=C_BASE, alpha=0.4, edgecolor='none', zorder=2, rasterized=True, label='no LayerNorm (ablated)')
    ax.scatter(in_norms, out_norms, s=8, color=C_OURS, alpha=0.55, edgecolor='none', zorder=3, rasterized=True, label='ours (LayerNorm)')
    ax.plot(xs, envelope, color=C_BOUND, lw=1.9, linestyle=(0, (5, 3)), zorder=4, label='envelope $(1-\\alpha)\\|F_t\\|+\\alpha\\sqrt{d}$')
    ax.axhline(sqrt_d, color='#888', lw=0.8, alpha=0.6, zorder=1)
    ax.text(xmax * 0.02, sqrt_d + 0.4, f'$\\sqrt{{d}}={sqrt_d:.1f}$', fontsize=8.8, color='#666', va='bottom', ha='left')
    y_top = float(np.percentile(out_noln, 96))
    ax.set_xlabel('input frame norm  $\\|F_t\\|_2$')
    ax.set_ylabel('output frame norm  $\\|F_t^{\\rm out}\\|_2$')
    ax.set_xlim(0, xmax)
    y_lim = max(y_top, envelope.max() * 1.1)
    ax.set_ylim(0, y_lim)
    ax.annotate(f'no-LN norm $\\approx{med_noln / sqrt_d:.1f}\\sqrt{{d}}$\n(explodes off-axis)', xy=(in_norms.mean(), y_lim * 0.93), xytext=(xmax * 0.1, y_lim * 0.7), fontsize=8.8, color='#6b7785', fontweight='semibold', va='center', ha='left', arrowprops=dict(arrowstyle='->', color='#6b7785', lw=1.0, connectionstyle='arc3,rad=-0.2'))
    ax.legend(loc='lower right', frameon=False, handlelength=1.6, fontsize=8.6, labelspacing=0.35, borderaxespad=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, color='#eef0f2', lw=0.7)
    _panel_letter(ax, 'c', 'Energy stability')
    plt.savefig(out_stem.with_suffix('.pdf'), bbox_inches='tight')
    plt.savefig(out_stem.with_suffix('.png'), bbox_inches='tight', dpi=400)
    plt.close()
    pct_ours_below = float((out_norms <= (1 - alpha) * in_norms + alpha * sqrt_d + 1e-06).mean()) * 100
    print('\n--- Figure 5 caption stats ---')
    print(f'T={T}, d={d}, r={r}, lambda={LRR_LAM}, alpha={alpha}')
    print(f'Panel a: speedup at T={T}: {speedup_at_T:.0f}x fewer FLOPs (LRR n_iter={n_iter_lrr:.0f})')
    print(f"Panel b: low-rank Z_raw sv at index r+1: {(sv_raw[r] if len(sv_raw) > r else float('nan')):.2e} (full self-expression at same index: {(sv_full[r] if len(sv_full) > r else float('nan')):.3f})")
    print(f'Panel c: ours below envelope = {pct_ours_below:.2f}%; median output norm  ours = {med_ours:.2f}  vs  no-LN = {med_noln:.2f}  (sqrt_d = {sqrt_d:.2f})')

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print('rendering Figure 5 ...')
    figure5(OUT_DIR / 'figure5')
    print(f'\nwritten to {OUT_DIR.resolve()}/figure5.{{pdf,png}}')
if __name__ == '__main__':
    main()
