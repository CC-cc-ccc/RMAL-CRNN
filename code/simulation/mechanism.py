from __future__ import annotations
import re
import warnings
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from scipy import stats
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
DATA_DIR = Path('data/audio')
OUT_DIR = Path('./figures_s31')
SR = 16000
FRAME_LEN_MS = 40
HOP_MS = 20
N_MELS = 64
N_FRAMES_PER_FILE = 80
N_FILES_PER_CLASS = 25
RNG = np.random.default_rng(42)
mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'], 'mathtext.fontset': 'stixsans', 'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 11.5, 'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5, 'legend.fontsize': 9, 'figure.dpi': 150, 'savefig.dpi': 600, 'axes.linewidth': 0.8, 'xtick.major.width': 0.8, 'ytick.major.width': 0.8, 'xtick.major.size': 3.2, 'ytick.major.size': 3.2, 'xtick.direction': 'out', 'ytick.direction': 'out', 'pdf.fonttype': 42, 'ps.fonttype': 42})
DIST_COLORS = {'5m': ['#9ecae1', '#4292c6', '#08519c'], '8m': ['#a1d99b', '#41ab5d', '#006d2c'], '10m': ['#fdae6b', '#e6550d', '#a63603']}
HEIGHT_INDEX = {'0m': 0, '5m': 1, '8m': 2}

def parse_dist_height(label: str):
    parts = label.split('_')
    m_parts = [p for p in parts if p.endswith('m') and p[:-1].isdigit()]
    if len(m_parts) >= 2:
        return (m_parts[-2], m_parts[-1])
    return (None, None)

def shorten_label(label: str) -> str:
    dist, height = parse_dist_height(label)
    if dist and height:
        return f'{dist} / {height}'
    return label

def class_color(label: str, fallback_palette, fallback_idx: int) -> str:
    dist, height = parse_dist_height(label)
    if dist in DIST_COLORS and height in HEIGHT_INDEX:
        return DIST_COLORS[dist][HEIGHT_INDEX[height]]
    return fallback_palette[fallback_idx % len(fallback_palette)]

class Arrow3D(FancyArrowPatch):

    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = (xs, ys, zs)

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return float(np.min(zs))
FNAME_RE = re.compile('^(?P<cls>.+?)_(?P<date>\\d{6})_(?P<idx>\\d{4})\\.wav$', re.IGNORECASE)

def discover_files(root: Path):
    files = list(root.rglob('*.wav')) + list(root.rglob('*.WAV'))
    cls: dict[str, list[Path]] = {}
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

def synth_drone_signal(duration_s=4.0, n_harmonics=6, f0=200.0, noise_dof=4.0, snr_db=6.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration_s * SR)) / SR
    f0_jit = f0 * (1 + 0.03 * rng.standard_normal(t.size).cumsum() / np.sqrt(t.size))
    sig = np.zeros_like(t)
    for h in range(1, n_harmonics + 1):
        amp = 1.0 / h ** 0.7
        phase = 2 * np.pi * np.cumsum(h * f0_jit) / SR
        sig += amp * np.sin(phase + rng.uniform(0, 2 * np.pi))
    sig /= np.max(np.abs(sig)) + 1e-09
    noise = stats.t.rvs(df=noise_dof, size=t.size, random_state=rng)
    noise = noise / np.std(noise)
    sig = sig + 10 ** (-snr_db / 20) * noise
    sig /= np.max(np.abs(sig)) + 1e-09
    return sig.astype(np.float32)

def extract_logmel(y: np.ndarray) -> np.ndarray:
    n_fft = int(SR * FRAME_LEN_MS / 1000)
    hop = int(SR * HOP_MS / 1000)
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop, center=True)) ** 2
    mel = librosa.feature.melspectrogram(S=S, sr=SR, n_mels=N_MELS, power=2.0)
    logmel = np.log(mel + 1e-09).T
    if logmel.shape[0] > N_FRAMES_PER_FILE:
        start = (logmel.shape[0] - N_FRAMES_PER_FILE) // 2
        logmel = logmel[start:start + N_FRAMES_PER_FILE]
    return logmel

def collect_frames_by_class():
    if HAS_LIBROSA and DATA_DIR.exists():
        cls = discover_files(DATA_DIR)
        cls = {k: v for k, v in cls.items() if v}
        if cls:
            print(f'[real data] {sum((len(v) for v in cls.values()))} files across {len(cls)} classes under {DATA_DIR}')
            out = {}
            for name, files in cls.items():
                idx = RNG.permutation(len(files))[:N_FILES_PER_CLASS]
                stacks = []
                for j in idx:
                    try:
                        stacks.append(extract_logmel(load_audio(files[j])))
                    except Exception as e:
                        warnings.warn(f'skip {files[j].name}: {e}')
                if stacks:
                    out[name] = np.vstack(stacks)
            return out
    print('[fallback] synthesising 9 drone-like (distance x height) conditions; set DATA_DIR for real-data figures')
    out = {}
    for di, d in enumerate(['5m', '8m', '10m']):
        for hi, h in enumerate(['0m', '5m', '8m']):
            snr_db = 12 - di * 4 - hi * 2
            amp = {'5m': 1.0, '8m': 0.55, '10m': 0.35}[d]
            stacks = []
            for j in range(N_FILES_PER_CLASS):
                y = synth_drone_signal(snr_db=snr_db, seed=100 * (3 * di + hi) + j) * amp
                stacks.append(extract_logmel(y) if HAS_LIBROSA else _synth_logmel_direct(snr_db, amp, seed=100 * (3 * di + hi) + j))
            name = f'x4_d4_mavicair2_{d}_{h}'
            out[name] = np.vstack(stacks)
    return out

def _synth_logmel_direct(snr_db, amp, seed):
    rng = np.random.default_rng(seed)
    base = np.linspace(0, 1, N_MELS)
    template = 0.5 + amp * np.exp(-(base - 0.2) ** 2 / 0.02) + amp * 0.6 * np.exp(-(base - 0.4) ** 2 / 0.03)
    sigma = 0.6 * 10 ** (-snr_db / 20)
    frames = template[None, :] + sigma * stats.t.rvs(df=4.0, size=(N_FRAMES_PER_FILE, N_MELS), random_state=rng)
    return np.log(np.maximum(frames, 0.001))

def fit_gate_params(x: np.ndarray, eps: float=1e-06):
    c = np.median(x, axis=0, keepdims=True)
    s = stats.median_abs_deviation(x, axis=0, scale='normal')[None, :] + eps
    return (c, s)

def student_t_gate(x: np.ndarray, c: np.ndarray, s: np.ndarray, nu: float=4.0) -> np.ndarray:
    rho = (x - c) / s
    g = (nu + 1.0) / (nu + rho * rho)
    return g * x

def tangent_embed(x: np.ndarray, p: np.ndarray, eps: float=1e-06) -> np.ndarray:
    p_hat = p / (np.linalg.norm(p) + eps)
    xi = x - np.outer(x @ p_hat, p_hat)
    d = x.shape[1]
    return np.sqrt(d) * xi / (np.linalg.norm(xi, axis=1, keepdims=True) + eps)

def figure3_combined(frames_by_class, out_stem: Path) -> None:
    classes = sorted(frames_by_class.keys())
    all_x = np.vstack([frames_by_class[c] for c in classes])
    c_gate, s_gate = fit_gate_params(all_x)
    p_hat = all_x.mean(axis=0)
    d = all_x.shape[1]
    sqrt_d = np.sqrt(d)
    norms_before = {cn: np.linalg.norm(X, axis=1) for cn, X in frames_by_class.items()}
    norms_after = {cn: np.linalg.norm(tangent_embed(student_t_gate(X, c_gate, s_gate), p_hat), axis=1) for cn, X in frames_by_class.items()}
    fig = plt.figure(figsize=(13.0, 9.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.42], height_ratios=[1.05, 1.0], hspace=0.42, wspace=0.06, left=0.06, right=0.985, top=0.965, bottom=0.075)
    ax1 = fig.add_subplot(gs[0, 0])
    rho = np.linspace(-5, 5, 1200)
    palette_nu = ['#08306b', '#2171b5', '#88bbe0']
    for nu, col in zip([1, 10, 100], palette_nu):
        ax1.plot(rho, (nu + 1) / (nu + rho ** 2), color=col, linewidth=2.3, label=f'$\\nu={nu}$', solid_capstyle='round')
    ax1.axhline(1.0, color='#c8553d', linestyle=(0, (5, 3)), linewidth=1.9, label='$\\nu\\!\\to\\!\\infty$', dash_capstyle='round')
    ax1.axvspan(-5, -2, alpha=0.045, color='#7b241c', linewidth=0)
    ax1.axvspan(2, 5, alpha=0.045, color='#7b241c', linewidth=0)
    ax1.text(-3.5, 0.3, 'outlier zone', ha='center', va='center', fontsize=9, color='#7b241c', style='italic')
    ax1.text(3.5, 0.3, 'outlier zone', ha='center', va='center', fontsize=9, color='#7b241c', style='italic')
    ax1.set_xlabel('Standardised residual  $\\rho$')
    ax1.set_ylabel('Gate weight  $g(\\rho)$')
    ax1.set_xlim(-5, 5)
    ax1.set_ylim(0, 2.15)
    ax1.set_xticks([-4, -2, 0, 2, 4])
    ax1.set_yticks([0, 0.5, 1.0, 1.5, 2.0])
    ax1.legend(loc='upper right', frameon=False, handlelength=2.0, labelspacing=0.4, borderpad=0.2, bbox_to_anchor=(0.995, 0.97))
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.text(-0.16, 1.07, 'a', transform=ax1.transAxes, fontsize=15, fontweight='bold', va='top')
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    pos = ax2.get_position()
    ax2.set_position([pos.x0 - 0.025, pos.y0 - 0.045, pos.width + 0.055, pos.height + 0.09])
    u = np.linspace(0, 2 * np.pi, 80)
    v = np.linspace(0, np.pi, 40)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax2.plot_surface(xs, ys, zs, color='#e9ecef', alpha=0.22, linewidth=0, antialiased=True)
    ax2.plot_wireframe(xs, ys, zs, color='#9aa0a6', alpha=0.18, linewidth=0.35, rstride=8, cstride=8)
    p_hat_sphere = np.array([0.4, 0.0, 0.92])
    p_hat_sphere = p_hat_sphere / np.linalg.norm(p_hat_sphere)
    e1 = np.array([1.0, 0.0, 0.0])
    e1 = e1 - e1 @ p_hat_sphere * p_hat_sphere
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(p_hat_sphere, e1)
    hu, hv = (1.25, 0.65)
    S, T = np.meshgrid(np.linspace(-hu, hu, 12), np.linspace(-hv, hv, 10))
    plane = p_hat_sphere[:, None, None] + S[None, :, :] * e1[:, None, None] + T[None, :, :] * e2[:, None, None]
    ax2.plot_surface(plane[0], plane[1], plane[2], color='#f4a261', alpha=0.26, linewidth=0, antialiased=True)
    c_par = 0.3
    x_tilde = c_par * p_hat_sphere + 1.3 * e1 + 0.45 * e2
    xi = x_tilde - c_par * p_hat_sphere
    xi_tip = p_hat_sphere + xi
    ax2.add_artist(Arrow3D([x_tilde[0], xi_tip[0]], [x_tilde[1], xi_tip[1]], [x_tilde[2], xi_tip[2]], arrowstyle='-|>', mutation_scale=14, lw=1.8, color='#3b4757', linestyle=(0, (4, 2.5)), zorder=9))
    ra = 0.16
    drop_dir = (xi_tip - x_tilde) / (np.linalg.norm(xi_tip - x_tilde) + 1e-09)
    plane_dir = -xi / (np.linalg.norm(xi) + 1e-09)
    p_along = xi_tip - ra * drop_dir
    corner = xi_tip - ra * drop_dir + ra * plane_dir
    p_inplane = xi_tip + ra * plane_dir
    ax2.plot([p_along[0], corner[0], p_inplane[0]], [p_along[1], corner[1], p_inplane[1]], [p_along[2], corner[2], p_inplane[2]], color='#3b4757', linewidth=1.6, zorder=10)
    ax2.add_artist(Arrow3D([p_hat_sphere[0], xi_tip[0]], [p_hat_sphere[1], xi_tip[1]], [p_hat_sphere[2], xi_tip[2]], arrowstyle='-|>', mutation_scale=20, lw=3.0, color='#0d6e36', zorder=11))
    ax2.scatter(*p_hat_sphere, color='#c0392b', s=85, depthshade=False, zorder=12, edgecolor='white', linewidth=1.2)
    ax2.scatter(*x_tilde, color='#111', s=58, depthshade=False, zorder=12, edgecolor='white', linewidth=0.9)
    ax2.set_xlim(-1.0, 1.55)
    ax2.set_ylim(-1.0, 1.55)
    ax2.set_zlim(-1.0, 1.55)
    ax2.set_box_aspect([1, 1, 1])
    ax2.view_init(elev=15, azim=40)
    ax2.set_axis_off()
    labels = [('Base point  $\\hat{p}_k$', (0.4, 0.92), '#c0392b', 11.5, 'bold', 'normal'), ('Gated feature  $\\tilde{x}_t^{\\,k}$', (0.79, 0.36), '#111', 11.5, 'normal', 'normal'), ('Tangent vector  $\\xi_t^{\\,k}$', (0.42, 0.16), '#0d6e36', 12, 'bold', 'normal'), ('Tangent space  $T_{\\hat{p}_k}\\mathcal{S}^{d-1}$', (0.66, 0.82), '#a05a2c', 10.5, 'normal', 'italic'), ('project', (0.74, 0.58), '#3b4757', 10.5, 'semibold', 'italic'), ('$\\mathcal{S}^{d-1}$', (0.1, 0.32), '#7d7d7d', 13.5, 'normal', 'italic')]
    for txt, (xx, yy), col, sz, weight, style in labels:
        ax2.text2D(xx, yy, txt, transform=ax2.transAxes, color=col, fontsize=sz, fontweight=weight, style=style, va='center')
    ax2.text2D(-0.04, 1.07, 'b', transform=ax2.transAxes, fontsize=15, fontweight='bold', va='top')
    ax2.text2D(0.5, 0.04, 'Component along $\\hat{p}_k$ discarded; tangential direction kept', transform=ax2.transAxes, fontsize=10.5, color='#2d2d2d', ha='center', va='bottom', style='italic')
    fallback_palette = ['#9ecae1', '#4292c6', '#08519c', '#a1d99b', '#41ab5d', '#006d2c', '#fdae6b', '#e6550d', '#a63603']
    ax3 = fig.add_subplot(gs[1, 0])
    pooled_before = np.concatenate(list(norms_before.values()))
    bins_before = np.linspace(pooled_before.min() * 0.98, pooled_before.max() * 1.02, 38)
    for i, cn in enumerate(classes):
        col = class_color(cn, fallback_palette, i)
        ax3.hist(norms_before[cn], bins=bins_before, alpha=0.55, color=col, label=shorten_label(cn), density=True, edgecolor='white', linewidth=0.3)
    medians_before = np.array([np.median(v) for v in norms_before.values()])
    cv_before = medians_before.std() / medians_before.mean() * 100
    ax3.text(0.96, 0.95, f'CV of class medians\n= {cv_before:.1f}%', transform=ax3.transAxes, fontsize=10, color='#444', va='top', ha='right', bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='#bbb', lw=0.5))
    ax3.set_xlabel('$\\|x\\|_2$')
    ax3.set_ylabel('density')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.text(-0.16, 1.07, 'c', transform=ax3.transAxes, fontsize=15, fontweight='bold', va='top')
    ax4 = fig.add_subplot(gs[1, 1])
    pooled_after = np.concatenate(list(norms_after.values()))
    half = max(pooled_after.std() * 4, sqrt_d * 0.015)
    bins_after = np.linspace(sqrt_d - half, sqrt_d + half, 30)
    for i, cn in enumerate(classes):
        col = class_color(cn, fallback_palette, i)
        ax4.hist(norms_after[cn], bins=bins_after, alpha=0.55, color=col, density=True, edgecolor='white', linewidth=0.3)
    ax4.axvline(sqrt_d, color='#222', linewidth=1.7, linestyle=(0, (5, 3)))
    ax4.annotate(f'$\\sqrt{{d}} = {sqrt_d:.2f}$', xy=(sqrt_d, 0.45), xycoords=('data', 'axes fraction'), xytext=(0.65, 0.45), textcoords='axes fraction', fontsize=11.5, color='#222', va='center', ha='left', arrowprops=dict(arrowstyle='-', color='#888', lw=0.8))
    ax4.annotate('all 9 conditions\ncollapse to this spike', xy=(sqrt_d, 0.9), xycoords=('data', 'axes fraction'), xytext=(0.55, 0.72), textcoords='axes fraction', fontsize=10, color='#0d6e36', va='center', ha='left', style='italic', fontweight='semibold', arrowprops=dict(arrowstyle='->', color='#0d6e36', lw=1.2, connectionstyle='arc3,rad=-0.22'))
    medians_after = np.array([np.median(v) for v in norms_after.values()])
    cv_after = medians_after.std() / (medians_after.mean() + 1e-12) * 100
    ax4.text(0.04, 0.95, f'CV of class medians\n= {cv_after:.2f}%', transform=ax4.transAxes, fontsize=10, color='#444', va='top', ha='left', bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='#bbb', lw=0.5))
    ax4.set_xlabel('$\\|\\Phi_k(x)\\|_2$')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.text(-0.16, 1.07, 'd', transform=ax4.transAxes, fontsize=15, fontweight='bold', va='top')

    def sort_key(cn):
        dist, hgt = parse_dist_height(cn)
        di = {'5m': 0, '8m': 1, '10m': 2}.get(dist, 99)
        hi = HEIGHT_INDEX.get(hgt, 99)
        return (di, hi)
    legend_order = sorted(classes, key=sort_key)
    handles = []
    labels_short = []
    for i, cn in enumerate(legend_order):
        col = class_color(cn, fallback_palette, i)
        handles.append(plt.Rectangle((0, 0), 1, 1, color=col, alpha=0.7))
        labels_short.append(shorten_label(cn))
    leg = fig.legend(handles, labels_short, loc='lower center', ncol=len(legend_order), bbox_to_anchor=(0.5, -0.005), frameon=False, handlelength=1.2, columnspacing=1.0, labelspacing=0.3, fontsize=9.5)
    plt.savefig(out_stem.with_suffix('.pdf'), bbox_inches='tight')
    plt.savefig(out_stem.with_suffix('.png'), bbox_inches='tight', dpi=400)
    plt.close()
    print(f'  CV(class-median ||x||)  before = {cv_before:.2f}%')
    print(f'  CV(class-median ||z||)  after  = {cv_after:.2f}%')
    print(f'  reduction factor = {cv_before / (cv_after + 1e-12):.1f}x')

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames_by_class = collect_frames_by_class()
    print(f"  classes ({len(frames_by_class)}): {', '.join(list(frames_by_class.keys())[:4])} ...")
    print(f'  frame matrix dim per class: {list(frames_by_class.values())[0].shape}')
    print('\nrendering Figure 3 (combined) ...')
    figure3_combined(frames_by_class, OUT_DIR / 'figure3')
    print(f'\nfigure written to {OUT_DIR.resolve()}/figure3.{{pdf,png}}')
if __name__ == '__main__':
    main()
