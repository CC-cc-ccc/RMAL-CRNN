import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
import synthesis as ms
import common as mc
ms.apply_style()
OUT = 'fig_dissociation'
N_TRIALS = 160
M, row_names, col_names = mc.dissociation_matrix(n_trials=N_TRIALS, seed=2026)
Mc = np.clip(M, 0.0, None)
row_max = Mc.max(axis=1, keepdims=True) + 1e-12
Mn = Mc / row_max
row_metric = ['GST\n(scale-bias\nreduction)', 'LMA\n(fusion-fidelity\ngain)', 'NSLLR\n(low-rank SNR\ngain, dB)']
col_lab = ['Impulsive\ninterference', 'Co-channel\ninterference (SIR)', 'Additive\nnoise (SNR)']
raw_units = [['', '', ''], ['', '', ''], ['', '', '']]
fig, ax = plt.subplots(figsize=(6.6, 5.4))
greens = LinearSegmentedColormap.from_list('ours', ['#ffffff', '#cfe6d8', '#7bbf8f', mc.C_OURS, '#16512f'])
im = ax.imshow(Mn, cmap=greens, vmin=0, vmax=1, aspect='auto')
ax.set_xticks(range(3))
ax.set_xticklabels(col_lab, fontsize=10)
ax.set_yticks(range(3))
ax.set_yticklabels(row_metric, fontsize=10)
ax.set_xlabel('complex degradation', fontsize=11, labelpad=8)
ax.set_ylabel('innovation module  (signature metric)', fontsize=11, labelpad=8)
ax.set_title('Triple dissociation: each module matches one degradation', fontweight='bold', fontsize=12, pad=12)
for i in range(3):
    for j in range(3):
        val = M[i, j]
        is_diag = i == j
        txt = f'{val:+.2f}'
        col = 'white' if Mn[i, j] > 0.55 else mc.C_INK
        ax.text(j, i, txt, ha='center', va='center', fontsize=12.5 if is_diag else 11, fontweight='bold' if is_diag else 'normal', color=col)
        if is_diag:
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=mc.C_WARN, lw=2.8, zorder=5))
        ax.text(j, i + 0.3, f'(row-norm {Mn[i, j]:.2f})', ha='center', va='center', fontsize=7.5, color=col, alpha=0.8)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('row-normalised signature gain', fontsize=9.5)
cbar.ax.tick_params(labelsize=8)
diag_ok = all((M[i].argmax() == i for i in range(3)))
note = 'row-wise argmax on the diagonal: ' + ('YES (matching is optimal)' if diag_ok else 'NO')
fig.text(0.5, -0.02, note, ha='center', fontsize=9, color=mc.C_INK)
fig.tight_layout()
fig.savefig(OUT + '.pdf', bbox_inches='tight')
fig.savefig(OUT + '.png', bbox_inches='tight', dpi=200)
print('saved', OUT + '.pdf /.png')
print('  columns =', col_names)
for i, r in enumerate(row_names):
    print(f'  {r:6s}: raw {np.round(M[i], 4)}  argmax={col_names[M[i].argmax()]}')
print('  diagonal dominance:', 'PASS' if diag_ok else 'FAIL')
