import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import synthesis as M
M.apply_style()
C_OURS = '#e63946'
C_STD = '#5c677d'
SNRS = np.array([20, 15, 10, 5, 0, -5, -10, -15, -20], dtype=float)
BETA = 0.5

def sweep(seed=7, n_rep=80):
    rng = np.random.default_rng(seed)
    out = {k: {'sup': [], 'sup_sd': [], 'mx': [], 'mx_sd': []} for k in ('ours', 'std')}
    for snr in SNRS:
        acc = {k: {'sup': [], 'mx': []} for k in ('ours', 'std')}
        for _ in range(n_rep):
            _, s = M.clean_source(int(rng.integers(10)), rng)
            _, y = M.degrade(s, rng, snr_db=snr, sir_db=40)
            v = M.extract_views(y)[0]
            toks = np.stack([v[k].mean(0) for k in M.FEATURE_NAMES])
            E = M.spherical_tangent_embed(toks)
            Ao = M.logmap_tangent_attention(E, np.ones(E.shape[1]), beta=BETA)
            As = M.standard_dot_attention(E)
            for A, k in [(Ao, 'ours'), (As, 'std')]:
                acc[k]['sup'].append(np.exp(-(A * np.log(A + 1e-12)).sum(1)).mean())
                acc[k]['mx'].append(A.max(1).mean())
        for k in ('ours', 'std'):
            out[k]['sup'].append(np.mean(acc[k]['sup']))
            out[k]['sup_sd'].append(np.std(acc[k]['sup']))
            out[k]['mx'].append(np.mean(acc[k]['mx']))
            out[k]['mx_sd'].append(np.std(acc[k]['mx']))
    for k in out:
        for kk in out[k]:
            out[k][kk] = np.array(out[k][kk])
    return out

def main():
    d = sweep()
    fig = plt.figure(figsize=(13.2, 5.4))
    gs = gridspec.GridSpec(1, 2, wspace=0.26)
    ax = fig.add_subplot(gs[0, 0])
    ax.fill_between(SNRS, d['std']['sup'], d['ours']['sup'], color=C_OURS, alpha=0.08, zorder=0)
    for k, c, lab, mk in [('ours', C_OURS, 'ours (log-map tangent)', 'o'), ('std', C_STD, 'standard dot-product', 's')]:
        ax.plot(SNRS, d[k]['sup'], color=c, lw=2.6, marker=mk, ms=6, label=lab)
        ax.fill_between(SNRS, d[k]['sup'] - d[k]['sup_sd'], d[k]['sup'] + d[k]['sup_sd'], color=c, alpha=0.15)
    ax.axhline(6, ls=':', color='#aaa', lw=1.1)
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('effective support  $\\exp\\,H(A)$')
    ax.set_title('(a)  cross-domain support', loc='left')
    ax.invert_xaxis()
    ax.set_ylim(1, 6.4)
    ax.grid(True, alpha=0.4)
    ax.legend(loc='center left', fontsize=9)
    ax = fig.add_subplot(gs[0, 1])
    for k, c, lab, mk in [('std', C_STD, 'standard dot-product', 's'), ('ours', C_OURS, 'ours (log-map tangent)', 'o')]:
        ax.plot(SNRS, d[k]['mx'], color=c, lw=2.6, marker=mk, ms=6, label=lab)
        ax.fill_between(SNRS, d[k]['mx'] - d[k]['mx_sd'], d[k]['mx'] + d[k]['mx_sd'], color=c, alpha=0.15)
    ax.axhline(1 / 6, ls=':', color='#aaa', lw=1.1)
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('mean peak weight  $\\max_j A_{ij}$')
    ax.set_title('(b)  resistance to domination', loc='left')
    ax.invert_xaxis()
    ax.set_ylim(0, 0.72)
    ax.grid(True, alpha=0.4)
    ax.legend(loc='center left', fontsize=9)
    fig.tight_layout()
    out = 'figII_2_noncollapse_curves.png'
    fig.savefig(out, bbox_inches='tight', dpi=200)
    print('saved', out)
if __name__ == '__main__':
    main()
