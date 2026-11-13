import argparse
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
SNRS = np.array([-5, 0, 5, 10])
RAW_ACC = {'MAtt': {-5: [0.691, 0.695, 0.712, 0.688], 0: [0.804, 0.802, 0.786, 0.821], 5: [0.849, 0.863, 0.831, 0.837], 10: [0.898, 0.895, 0.875, 0.878]}, 'InceptionTime': {-5: [0.785, 0.797, 0.78, 0.794], 0: [0.858, 0.866, 0.867, 0.861], 5: [0.913, 0.903, 0.9, 0.914], 10: [0.953, 0.947, 0.941, 0.945]}, 'CNNTransformer': {-5: [0.804, 0.782, 0.801, 0.795], 0: [0.87, 0.863, 0.866, 0.874], 5: [0.919, 0.909, 0.917, 0.904], 10: [0.94, 0.942, 0.932, 0.94]}, 'CRNN (BL)': {-5: [0.8, 0.814, 0.816, 0.802], 0: [0.874, 0.867, 0.874, 0.871], 5: [0.923, 0.917, 0.916, 0.916], 10: [0.953, 0.944, 0.939, 0.951]}, 'SEResNet1D': {-5: [0.827, 0.818, 0.816, 0.818], 0: [0.893, 0.891, 0.894, 0.893], 5: [0.932, 0.925, 0.921, 0.931], 10: [0.952, 0.961, 0.962, 0.956]}, 'TCN': {-5: [0.845, 0.844, 0.84, 0.819], 0: [0.903, 0.906, 0.914, 0.911], 5: [0.93, 0.934, 0.932, 0.93], 10: [0.966, 0.961, 0.966, 0.958]}, 'MAS-CRNN (Ours)': {-5: [0.861, 0.866, 0.865, 0.86], 0: [0.934, 0.938, 0.942, 0.941], 5: [0.957, 0.962, 0.965, 0.959], 10: [0.978, 0.969, 0.98, 0.972]}}
STYLE = {'MAtt': ('#79C7C5', '--', 'D', 1.5, 6), 'InceptionTime': ('#86B0DC', '--', 'D', 1.5, 6), 'CNNTransformer': ('#B59ED1', '--', 'D', 1.5, 6), 'CRNN (BL)': ('#3A7CA5', '--', 'D', 1.6, 6), 'SEResNet1D': ('#7EA85E', '-', 'D', 1.7, 6), 'TCN': ('#E89A3C', '-', 'D', 1.7, 6), 'MAS-CRNN (Ours)': ('#C0392B', '-', 'D', 2.6, 8)}

def _load(csv_path=None):
    if csv_path is not None and os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
        out = {}
        for name in df['model'].unique():
            sub = df[df['model'] == name]
            means = [sub[sub['snr'] == s]['acc'].mean() for s in SNRS]
            stds = [sub[sub['snr'] == s]['acc'].std(ddof=1) for s in SNRS]
            scale = 100.0 if max(means) < 2.0 else 1.0
            out[name] = (np.array(means) * scale, np.array(stds) * scale)
        return out
    out = {}
    for name, snr_dict in RAW_ACC.items():
        means, stds = ([], [])
        for s in SNRS:
            arr = np.array(snr_dict[int(s)])
            means.append(arr.mean() * 100)
            stds.append(arr.std(ddof=1) * 100)
        out[name] = (np.array(means), np.array(stds))
    return out

def _set_journal_style():
    plt.rcdefaults()
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'], 'mathtext.fontset': 'cm', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.labelsize': 14, 'axes.titlesize': 14, 'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 10.5, 'axes.linewidth': 1.1})

def _draw_curve(ax, name, mean, std, marker_every=True):
    color, ls, marker, lw, ms = STYLE[name]
    eb = ax.errorbar(SNRS, mean, yerr=std, color=color, linestyle=ls, linewidth=lw, marker=marker, markersize=ms, markerfacecolor=color, markeredgecolor='white', markeredgewidth=0.7, capsize=3.4, capthick=1.1, elinewidth=1.0, label=name, zorder=10 if 'Ours' in name else 5)
    return eb

def plot_multi_snr(data, out_dir):
    _set_journal_style()
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    for name in STYLE:
        mean, std = data[name]
        _draw_curve(ax, name, mean, std)
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Classification accuracy (%)')
    ax.set_xticks(SNRS)
    ax.set_xlim(SNRS[0] - 1.2, SNRS[-1] + 1.2)
    ax.set_ylim(50, 100)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(1.1)
        s.set_color('#222222')
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', direction='in', length=5.0, width=0.9, top=True, right=True, color='#222222')
    ax.tick_params(axis='both', which='minor', direction='in', length=2.6, width=0.7, top=True, right=True, color='#222222')
    ax.yaxis.grid(True, which='major', linestyle='-', linewidth=0.45, color='#E1E1E1', alpha=1.0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    leg = ax.legend(loc='lower left', frameon=True, framealpha=0.97, edgecolor='#444444', fancybox=False, borderpad=0.45, handlelength=2.0, handletextpad=0.5, labelspacing=0.3, ncol=2, columnspacing=1.1, fontsize=9.8, bbox_to_anchor=(0.012, 0.012))
    leg.get_frame().set_linewidth(0.8)
    leg.set_zorder(50)
    for txt in leg.get_texts():
        if 'Ours' in txt.get_text():
            txt.set_fontweight('bold')
            txt.set_color('#C0392B')
    axins = ax.inset_axes([0.69, 0.05, 0.28, 0.46])
    for name in STYLE:
        mean, std = data[name]
        color, ls, marker, lw, ms = STYLE[name]
        axins.errorbar(SNRS, mean, yerr=std, color=color, linestyle=ls, linewidth=lw * 0.9, marker=marker, markersize=ms * 0.85, markerfacecolor=color, markeredgecolor='white', markeredgewidth=0.6, capsize=2.8, capthick=1.0, elinewidth=0.9, zorder=10 if 'Ours' in name else 5)
    axins.set_xlim(-5.6, 0.6)
    axins.set_ylim(76.5, 95.0)
    axins.set_xticks([-5, 0])
    axins.set_yticks([80, 85, 90, 95])
    axins.minorticks_on()
    axins.tick_params(axis='both', which='major', direction='in', length=3.6, width=0.8, top=True, right=True, color='#222222', labelsize=10)
    axins.tick_params(axis='both', which='minor', direction='in', length=1.9, width=0.6, top=True, right=True, color='#222222')
    axins.yaxis.grid(True, which='major', linestyle='-', linewidth=0.4, color='#E1E1E1', alpha=1.0)
    axins.set_axisbelow(True)
    for spine in axins.spines.values():
        spine.set_visible(True)
        spine.set_color('#222222')
        spine.set_linewidth(1.0)
    plt.tight_layout()
    out = os.path.join(out_dir, 'fig_multi_snr_curves')
    plt.savefig(out + '.pdf', dpi=400, bbox_inches='tight')
    plt.savefig(out + '.svg', bbox_inches='tight')
    plt.savefig(out + '.png', dpi=400, bbox_inches='tight')
    plt.close()
    print(f'  saved: {out}.pdf / .svg / .png')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', type=str, default=None, help='Optional CSV with columns: model, snr, seed, acc')
    p.add_argument('--out-dir', type=str, default='hp_figures')
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    data = _load(args.csv)
    plot_multi_snr(data, args.out_dir)
    print('Done.')
if __name__ == '__main__':
    main()
