import argparse
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
try:
    from scipy.interpolate import RectBivariateSpline
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False
warnings.filterwarnings('ignore')
HIGHLIGHT_RED = '#C0392B'
SPINE_COLOR = '#202020'
GRID_COLOR = '#D9D9D9'
TEXT_COLOR = '#111111'
DPI = 600

def _synth_proj_out():
    rng = np.random.default_rng(11)
    vals = [32, 48, 64, 80, 96, 128]
    rows = []
    for v in vals:
        peak = 1.0 - 0.55 * (np.log2(v) - np.log2(64)) ** 2
        for s in range(10):
            acc = 93.6 + 1.55 * peak + rng.normal(0, 0.22)
            rows.append({'proj_out': v, 'seed': s, 'acc': acc, 'f1': acc - 0.1})
    return pd.DataFrame(rows)

def _synth_n_heads():
    rng = np.random.default_rng(22)
    vals = [1, 2, 4, 8]
    peak_at = {1: 0.3, 2: 0.78, 4: 1.0, 8: 0.62}
    rows = []
    for v in vals:
        for s in range(10):
            acc = 93.5 + 1.55 * peak_at[v] + rng.normal(0, 0.24)
            f1 = acc - 0.2 + rng.normal(0, 0.1)
            rows.append({'n_heads': v, 'seed': s, 'acc': acc, 'f1': f1})
    return pd.DataFrame(rows)

def _synth_lrr_rank():
    rng = np.random.default_rng(33)
    vals = [2, 4, 8, 16, 32]
    peak_at = {2: 0.3, 4: 0.66, 8: 1.0, 16: 0.74, 32: 0.5}
    rows = []
    for v in vals:
        sd = 0.36 - 0.16 * peak_at[v]
        for s in range(10):
            acc = 93.6 + 1.6 * peak_at[v] + rng.normal(0, sd)
            rows.append({'lrr_rank': v, 'seed': s, 'acc': acc, 'f1': acc - 0.1})
    return pd.DataFrame(rows)

def _synth_lr_bs():
    rng = np.random.default_rng(44)
    lrs = [0.0001, 0.0003, 0.001, 0.003, 0.01]
    bss = [16, 32, 64, 128]
    rows = []
    for lr in lrs:
        for bs in bss:
            d_lr = -0.7 * (np.log10(lr) - np.log10(0.001)) ** 2
            d_bs = -0.45 * (np.log2(bs) - np.log2(32)) ** 2
            peak = np.exp(d_lr + d_bs)
            for s in range(5):
                acc = 93.4 + 1.7 * peak + rng.normal(0, 0.18)
                rows.append({'lr': lr, 'batch': bs, 'seed': s, 'acc': acc, 'f1': acc - 0.1})
    return pd.DataFrame(rows)

def _load_or_synth(data_dir, fname, synth_fn):
    if data_dir is not None:
        path = os.path.join(data_dir, fname)
        if os.path.isfile(path):
            return pd.read_csv(path)
    print(f'  [info] {fname} not found; using synthetic preview data.')
    return synth_fn()

def set_journal_style(font='serif'):
    plt.rcdefaults()
    if font == 'sans':
        family = 'sans-serif'
        family_list = ['Helvetica', 'Arial', 'DejaVu Sans']
        family_key = 'font.sans-serif'
    else:
        family = 'serif'
        family_list = ['Times New Roman', 'Times', 'DejaVu Serif']
        family_key = 'font.serif'
    plt.rcParams.update({'font.family': family, family_key: family_list, 'mathtext.fontset': 'cm', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'axes.facecolor': 'white', 'axes.edgecolor': SPINE_COLOR, 'axes.labelcolor': TEXT_COLOR, 'axes.linewidth': 1.05, 'axes.labelsize': 13, 'axes.titlesize': 13, 'xtick.labelsize': 11.5, 'ytick.labelsize': 11.5, 'legend.fontsize': 10.5, 'axes.spines.top': True, 'axes.spines.right': True, 'axes.spines.left': True, 'axes.spines.bottom': True, 'xtick.top': True, 'ytick.right': True, 'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.color': TEXT_COLOR, 'ytick.color': TEXT_COLOR, 'lines.linewidth': 1.8, 'patch.linewidth': 1.0})

def apply_full_box(ax, grid_axis='y', enable_minor=True):
    for side in ['left', 'right', 'top', 'bottom']:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.05)
        ax.spines[side].set_color(SPINE_COLOR)
    ax.tick_params(axis='both', which='major', direction='in', length=4.5, width=0.9, top=True, right=True, bottom=True, left=True, pad=4)
    if enable_minor:
        ax.minorticks_on()
        ax.tick_params(axis='both', which='minor', direction='in', length=2.5, width=0.7, top=True, right=True, bottom=True, left=True)
    if grid_axis is not None:
        ax.grid(True, axis=grid_axis, linestyle='-', linewidth=0.55, color=GRID_COLOR, alpha=0.85)
        ax.set_axisbelow(True)

def save_figure(fig, out_prefix):
    fig.savefig(out_prefix + '.pdf', bbox_inches='tight', pad_inches=0.025)
    fig.savefig(out_prefix + '.svg', bbox_inches='tight', pad_inches=0.025)
    fig.savefig(out_prefix + '.png', dpi=DPI, bbox_inches='tight', pad_inches=0.025)

def plot_proj_out(df, out_dir):
    set_journal_style('serif')
    g = df.groupby('proj_out')['acc'].agg(['mean', 'std']).sort_index()
    x = g.index.values.astype(float)
    m = g['mean'].values
    s = g['std'].fillna(0.0).values
    fig, ax = plt.subplots(figsize=(6.8, 4.35))
    line_c = '#1F77B4'
    band_c = '#BBD7EE'
    ax.fill_between(x, m - s, m + s, color=band_c, alpha=0.45, linewidth=0, label='Mean $\\pm$ 1 SD', zorder=1)
    ax.plot(x, m, color=line_c, marker='o', markersize=5.8, markerfacecolor='white', markeredgecolor=line_c, markeredgewidth=1.25, linewidth=1.9, label='Accuracy', zorder=3)
    i_best = int(np.argmax(m))
    ax.scatter([x[i_best]], [m[i_best]], s=360, facecolors='none', edgecolors=HIGHLIGHT_RED, linewidths=1.6, zorder=5, clip_on=False)
    ax.set_xlabel('QR-fusion embedding dimension $d_{\\mathrm{proj}}$')
    ax.set_ylabel('Classification accuracy (%)')
    ax.set_xticks(x)
    y_min = float(np.min(m - s))
    y_max = float(np.max(m + s))
    y_pad = max(0.45, 0.12 * (y_max - y_min))
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    apply_full_box(ax, grid_axis='y')
    leg = ax.legend(loc='lower right', frameon=True, fancybox=False, framealpha=1.0, borderpad=0.35, handlelength=1.4, handletextpad=0.45)
    leg.get_frame().set_edgecolor('#7F7F7F')
    leg.get_frame().set_linewidth(0.75)
    fig.tight_layout()
    out = os.path.join(out_dir, 'fig_hp_proj_out')
    save_figure(fig, out)
    plt.close(fig)
    print(f'  saved: {out}.pdf / .svg / .png')

def plot_n_heads(df, out_dir):
    set_journal_style('serif')
    g_acc = df.groupby('n_heads')['acc'].agg(['mean', 'std']).sort_index()
    g_f1 = df.groupby('n_heads')['f1'].agg(['mean', 'std']).sort_index()
    x = g_acc.index.values
    xpos = np.arange(len(x))
    width = 0.34
    fig, ax = plt.subplots(figsize=(6.8, 4.35))
    c_acc = '#4C78A8'
    c_f1 = '#F2BE5C'
    ax.bar(xpos - width / 2, g_acc['mean'].values, width, yerr=g_acc['std'].fillna(0.0).values, color=c_acc, edgecolor='#1F3A5A', linewidth=0.9, error_kw=dict(ecolor='#202020', capsize=3.5, capthick=0.9, lw=0.9), label='Accuracy', zorder=3)
    ax.bar(xpos + width / 2, g_f1['mean'].values, width, yerr=g_f1['std'].fillna(0.0).values, color=c_f1, edgecolor='#9B6B10', linewidth=0.9, error_kw=dict(ecolor='#202020', capsize=3.5, capthick=0.9, lw=0.9), label='Macro-F1', zorder=3)
    i_best = int(np.argmax(g_acc['mean'].values))
    ax.set_xticks(xpos)
    tick_texts = [f'  {v}  ' if i == i_best else str(v) for i, v in enumerate(x)]
    ax.set_xticklabels(tick_texts)
    ax.set_xlabel('Number of LogMap-Tangent Attention heads $h$')
    ax.set_ylabel('Score (%)')
    fig.canvas.draw()
    labels = ax.get_xticklabels()
    for lab in labels:
        lab.set_clip_on(False)
    labels[i_best].set_bbox(dict(boxstyle='square,pad=0.30', edgecolor=HIGHLIGHT_RED, facecolor='none', linewidth=1.55))
    labels[i_best].set_color(HIGHLIGHT_RED)
    labels[i_best].set_fontweight('bold')
    y_low = min((g_acc['mean'] - g_acc['std'].fillna(0.0)).min(), (g_f1['mean'] - g_f1['std'].fillna(0.0)).min())
    y_high = max((g_acc['mean'] + g_acc['std'].fillna(0.0)).max(), (g_f1['mean'] + g_f1['std'].fillna(0.0)).max())
    ax.set_ylim(float(y_low) - 0.55, float(y_high) + 0.65)
    apply_full_box(ax, grid_axis='y')
    leg = ax.legend(loc='upper right', frameon=True, fancybox=False, framealpha=1.0, borderpad=0.35, handlelength=1.3, handletextpad=0.45)
    leg.get_frame().set_edgecolor('#7F7F7F')
    leg.get_frame().set_linewidth(0.75)
    fig.tight_layout()
    out = os.path.join(out_dir, 'fig_hp_n_heads')
    save_figure(fig, out)
    plt.close(fig)
    print(f'  saved: {out}.pdf / .svg / .png')

def plot_lrr_rank(df, out_dir):
    set_journal_style('serif')
    df = df.sort_values('lrr_rank')
    vals = sorted(df['lrr_rank'].unique())
    data = [df.loc[df['lrr_rank'] == v, 'acc'].values for v in vals]
    positions = np.arange(1, len(vals) + 1)
    fig, ax = plt.subplots(figsize=(6.8, 4.35))
    box_face = '#F6D7CC'
    box_edge = '#9E4B3E'
    median_c = '#14213D'
    ax.boxplot(data, positions=positions, widths=0.46, patch_artist=True, showfliers=False, medianprops=dict(color=median_c, linewidth=1.85), whiskerprops=dict(color='#555555', linewidth=1.0), capprops=dict(color='#555555', linewidth=1.0), boxprops=dict(facecolor=box_face, edgecolor=box_edge, linewidth=1.15))
    means = [float(np.mean(d)) for d in data]
    ax.plot(positions, means, marker='o', markersize=5.2, markerfacecolor='white', markeredgecolor=box_edge, markeredgewidth=1.25, linestyle='None', zorder=5)
    i_best = int(np.argmax(means))
    ax.set_xticks(positions)
    tick_texts = [f'  {v}  ' if i == i_best else str(v) for i, v in enumerate(vals)]
    ax.set_xticklabels(tick_texts)
    ax.set_xlabel('Rank $r$ of DiffNSLLR')
    ax.set_ylabel('Classification accuracy (%)')
    fig.canvas.draw()
    labels = ax.get_xticklabels()
    for lab in labels:
        lab.set_clip_on(False)
    labels[i_best].set_bbox(dict(boxstyle='square,pad=0.30', edgecolor=HIGHLIGHT_RED, facecolor='none', linewidth=1.55))
    labels[i_best].set_color(HIGHLIGHT_RED)
    labels[i_best].set_fontweight('bold')
    y_low = min((float(np.min(d)) for d in data))
    y_high = max((float(np.max(d)) for d in data))
    ax.set_ylim(y_low - 0.45, y_high + 0.55)
    apply_full_box(ax, grid_axis='y')
    fig.tight_layout()
    out = os.path.join(out_dir, 'fig_hp_lrr_rank')
    save_figure(fig, out)
    plt.close(fig)
    print(f'  saved: {out}.pdf / .svg / .png')

def plot_lr_bs(df, out_dir):
    set_journal_style('serif')
    lrs = sorted(df['lr'].unique())
    bss = sorted(df['batch'].unique())
    Z = np.zeros((len(bss), len(lrs)), dtype=float)
    for i, b in enumerate(bss):
        for j, lr in enumerate(lrs):
            sub = df.loc[(df['lr'] == lr) & (df['batch'] == b), 'acc']
            Z[i, j] = sub.mean() if len(sub) > 0 else np.nan
    X_log = np.log10(np.asarray(lrs, dtype=float))
    Y_log = np.log2(np.asarray(bss, dtype=float))
    if HAS_SCIPY and len(lrs) >= 2 and (len(bss) >= 2):
        kx = min(3, len(bss) - 1)
        ky = min(3, len(lrs) - 1)
        spl = RectBivariateSpline(Y_log, X_log, Z, kx=kx, ky=ky)
        Xd = np.linspace(X_log.min(), X_log.max(), 90)
        Yd = np.linspace(Y_log.min(), Y_log.max(), 90)
        Xgd, Ygd = np.meshgrid(Xd, Yd)
        Zd = spl(Yd, Xd)
    else:
        Xgd, Ygd = np.meshgrid(X_log, Y_log)
        Zd = Z
    cmap = LinearSegmentedColormap.from_list('soft_cool_warm', ['#1D3557', '#457B9D', '#A8DADC', '#F1FAEE', '#E9C46A', '#F4A261', '#E76F51'], N=256)
    fig = plt.figure(figsize=(7.6, 5.45))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(Xgd, Ygd, Zd, cmap=cmap, edgecolor='none', alpha=0.94, rstride=1, cstride=1, antialiased=True)
    ax.plot_wireframe(Xgd, Ygd, Zd, color='white', linewidth=0.16, alpha=0.3, rstride=7, cstride=7)
    i_best, j_best = np.unravel_index(np.nanargmax(Z), Z.shape)
    x_best = X_log[j_best]
    y_best = Y_log[i_best]
    z_best = float(Z[i_best, j_best])
    z_floor = float(np.nanmin(Z) - 0.4)
    ax.set_zlim(z_floor, float(np.nanmax(Z)) + 0.35)
    dx = (X_log[1] - X_log[0]) * 0.22 if len(X_log) > 1 else 0.15
    dy = (Y_log[1] - Y_log[0]) * 0.26 if len(Y_log) > 1 else 0.2
    tri_verts = [[(x_best, y_best + dy, z_floor), (x_best - dx, y_best - dy * 0.58, z_floor), (x_best + dx, y_best - dy * 0.58, z_floor)]]
    tri = Poly3DCollection(tri_verts, facecolors=(0.753, 0.224, 0.169, 0.35), edgecolors=HIGHLIGHT_RED, linewidths=1.55)
    ax.add_collection3d(tri)
    ax.plot([x_best, x_best], [y_best, y_best], [z_floor, z_best], color=HIGHLIGHT_RED, linewidth=1.0, linestyle='--', alpha=0.78, zorder=9)
    ax.set_xticks(X_log)
    ax.set_xticklabels([f'{lr:.0e}' for lr in lrs])
    ax.set_yticks(Y_log)
    ax.set_yticklabels([str(b) for b in bss])
    ax.set_xlabel('Learning rate $\\eta$', labelpad=9)
    ax.set_ylabel('Batch size $B$', labelpad=8)
    ax.set_zlabel('Accuracy (%)', labelpad=6)
    ax.view_init(elev=24, azim=-58)
    try:
        ax.set_box_aspect((1.25, 1.0, 0.68))
    except Exception:
        pass
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_alpha(0.08)
        axis.pane.set_edgecolor('#AAAAAA')
        try:
            axis._axinfo['grid']['linewidth'] = 0.45
            axis._axinfo['grid']['linestyle'] = ':'
            axis._axinfo['grid']['color'] = (0.72, 0.72, 0.72, 0.65)
            axis._axinfo['axisline']['linewidth'] = 0.9
            axis._axinfo['axisline']['color'] = SPINE_COLOR
        except Exception:
            pass
    ax.tick_params(axis='both', which='major', direction='in', pad=2)
    cbar = fig.colorbar(surf, ax=ax, shrink=0.62, aspect=15, pad=0.1)
    cbar.set_label('Accuracy (%)', rotation=270, labelpad=14)
    cbar.outline.set_edgecolor('#666666')
    cbar.outline.set_linewidth(0.75)
    fig.tight_layout()
    out = os.path.join(out_dir, 'fig_hp_lr_bs')
    save_figure(fig, out)
    plt.close(fig)
    print(f'  saved: {out}.pdf / .svg / .png')

def plot_two_domain_curve(out_dir):
    set_journal_style('serif')
    x = np.array([64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144], dtype=float)
    xjtu_mean = np.array([94.5, 94.45, 94.85, 95.25, 95.15, 95.2, 94.85, 94.9, 94.6, 94.62, 93.85])
    xjtu_std = np.array([0.8, 0.45, 0.72, 0.85, 0.78, 0.7, 0.55, 0.5, 0.58, 0.95, 0.9])
    seu_mean = np.array([97.15, 97.35, 97.55, 97.42, 97.6, 97.8, 97.82, 97.72, 97.76, 97.75, 97.62])
    seu_std = np.array([0.55, 0.45, 0.18, 0.35, 0.3, 0.28, 0.33, 0.32, 0.35, 0.12, 0.25])
    fig, ax = plt.subplots(figsize=(5.6, 3.75))
    ax.errorbar(x, xjtu_mean, yerr=xjtu_std, fmt='-o', color='#1E88E5', ecolor='#1E88E5', elinewidth=1.05, capsize=2.8, capthick=1.05, markersize=4.7, markerfacecolor='#1E88E5', markeredgecolor='#1E88E5', linewidth=1.55, label='XJTU', zorder=3)
    ax.errorbar(x, seu_mean, yerr=seu_std, fmt='-o', color='#FDBE5B', ecolor='#FDBE5B', elinewidth=1.05, capsize=2.8, capthick=1.05, markersize=4.7, markerfacecolor='#FDBE5B', markeredgecolor='#FDBE5B', linewidth=1.55, label='SEU', zorder=3)
    i_best_xjtu = int(np.argmax(xjtu_mean))
    i_best_seu = int(np.argmax(seu_mean))
    ax.scatter([x[i_best_xjtu]], [xjtu_mean[i_best_xjtu]], s=330, facecolors='none', edgecolors=HIGHLIGHT_RED, linewidths=1.45, clip_on=False, zorder=6)
    ax.scatter([x[i_best_seu]], [seu_mean[i_best_seu]], s=330, facecolors='none', edgecolors=HIGHLIGHT_RED, linewidths=1.45, clip_on=False, zorder=6)
    ax.set_xlabel('Embedding dimension $d$')
    ax.set_ylabel('Classification Accuracy')
    ax.set_xlim(64, 144)
    ax.set_xticks(x)
    ax.set_ylim(92, 100)
    ax.set_yticks(np.arange(92, 101, 2))
    apply_full_box(ax, grid_axis='y')
    leg = ax.legend(loc='upper right', ncol=2, frameon=True, fancybox=False, framealpha=1.0, borderpad=0.25, handlelength=1.25, handletextpad=0.35, columnspacing=0.75)
    leg.get_frame().set_edgecolor('#7F7F7F')
    leg.get_frame().set_linewidth(0.75)
    fig.tight_layout()
    out = os.path.join(out_dir, 'fig_two_domain_curve')
    save_figure(fig, out)
    plt.close(fig)
    print(f'  saved: {out}.pdf / .svg / .png')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default=None, help='Directory containing sweep_*.csv files. If missing, synthetic preview data is used.')
    parser.add_argument('--out-dir', type=str, default='hp_figures_journal', help='Output directory.')
    parser.add_argument('--with-two-domain-demo', action='store_true', help='Also render a screenshot-style two-domain line plot template.')
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    print('Rendering publication-quality full-box figures...')
    plot_proj_out(_load_or_synth(args.data_dir, 'sweep_proj_out.csv', _synth_proj_out), args.out_dir)
    plot_n_heads(_load_or_synth(args.data_dir, 'sweep_n_heads.csv', _synth_n_heads), args.out_dir)
    plot_lrr_rank(_load_or_synth(args.data_dir, 'sweep_lrr_rank.csv', _synth_lrr_rank), args.out_dir)
    plot_lr_bs(_load_or_synth(args.data_dir, 'sweep_lr_bs.csv', _synth_lr_bs), args.out_dir)
    if args.with_two_domain_demo:
        plot_two_domain_curve(args.out_dir)
    print(f'\nDone. Figures saved to: {args.out_dir}/')
if __name__ == '__main__':
    main()
