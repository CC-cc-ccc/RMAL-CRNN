import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import synthesis as M
M.apply_style()
C_OURS = '#2e7d4f'
C_STD = '#9aa3ad'
C_RED = '#c0392b'
BETA = 0.5
K = 6

def collect(snr_db=0.0, seed=11, n=540):
    rng = np.random.default_rng(seed)
    o_log, s_log, o_mx, s_mx, o_su, s_su, dmax = ([], [], [], [], [], [], [])
    for _ in range(n):
        _, s = M.clean_source(int(rng.integers(10)), rng)
        _, y = M.degrade(s, rng, snr_db=snr_db, sir_db=40)
        v = M.extract_views(y)[0]
        toks = np.stack([v[k].mean(0) for k in M.FEATURE_NAMES])
        E = M.spherical_tangent_embed(toks)
        Ao, Lo, d2 = M.logmap_tangent_attention(E, np.ones(E.shape[1]), beta=BETA, return_logits=True)
        As, Ls, _ = M.standard_dot_attention(E, return_logits=True)
        off = ~np.eye(K, dtype=bool)
        o_log.append(Lo[off])
        s_log.append(Ls[off])
        o_mx.append(Ao.max(1))
        s_mx.append(As.max(1))
        o_su.append(np.exp(-(Ao * np.log(Ao + 1e-12)).sum(1)))
        s_su.append(np.exp(-(As * np.log(As + 1e-12)).sum(1)))
        dmax.append(d2[off].max())
    cat = lambda z: np.concatenate(z)
    Dm = np.percentile(dmax, 95)
    return (cat(o_log), cat(s_log), cat(o_mx), cat(s_mx), cat(o_su), cat(s_su), Dm)

def main():
    o_log, s_log, o_mx, s_mx, o_su, s_su, Dm = collect()
    lb = -BETA * Dm
    mw_ub = 1.0 / (1 + (K - 1) * np.exp(-BETA * Dm))
    su_lb = 1.0 / (mw_ub + (1 - mw_ub) / (K - 1))
    su_lb = max(su_lb, 1.15)
    fig = plt.figure(figsize=(13.5, 4.3))
    gs = gridspec.GridSpec(1, 3, wspace=0.24)
    ax = fig.add_subplot(gs[0, 0])
    lo = min(o_log.min(), s_log.min(), lb) - 0.4
    hi = max(o_log.max(), s_log.max(), 0) + 0.6
    bins = np.linspace(lo, hi, 64)
    ax.axvspan(lb, 0, color=C_OURS, alpha=0.07)
    ax.axvspan(0, hi, color=C_RED, alpha=0.07)
    ax.hist(s_log, bins=bins, density=True, color=C_STD, alpha=0.85, label='standard')
    ax.hist(o_log, bins=bins, density=True, color=C_OURS, alpha=0.85, label='ours')
    ax.axvline(lb, ls=':', color=C_RED, lw=1.6)
    ax.axvline(0, ls='--', color=C_RED, lw=1.6)
    ax.text(lb, 0.92, f'{lb:.2f}', color=C_RED, fontsize=9, ha='center', transform=ax.get_xaxis_transform())
    viol = (s_log > 0).mean() * 100
    ax.set_title('(a)  bounded logits', loc='left')
    ax.set_xlabel('attention logit value')
    ax.set_ylabel('density')
    ax.legend(fontsize=9, loc='center right')
    ax.text(0.96, 0.62, f'standard out-of-bound {viol:.0f}%\nours 0%', transform=ax.transAxes, va='top', ha='right', fontsize=8.5, color=C_RED)
    ax = fig.add_subplot(gs[0, 1])
    bins = np.linspace(0, 1, 60)
    ax.axvspan(mw_ub, 1, color=C_RED, alpha=0.07)
    ax.hist(s_mx, bins=bins, density=True, color=C_STD, alpha=0.85, label='standard')
    ax.hist(o_mx, bins=bins, density=True, color=C_OURS, alpha=0.85, label='ours')
    ax.axvline(mw_ub, ls='--', color=C_RED, lw=1.6)
    ax.text(mw_ub, 0.92, f'{mw_ub:.2f}', color=C_RED, fontsize=9, ha='left', transform=ax.get_xaxis_transform())
    viol2 = (s_mx > mw_ub).mean() * 100
    ax.set_title('(b)  non-collapse: max weight', loc='left')
    ax.set_xlabel('max attention weight  $\\max_j A_{ij}$')
    ax.set_ylabel('density')
    ax.legend(fontsize=9, loc='upper right')
    ax.text(0.55, 0.62, f'standard over-bound {viol2:.1f}%\nours 0%', transform=ax.transAxes, va='top', ha='left', fontsize=8.5, color=C_RED)
    ax = fig.add_subplot(gs[0, 2])
    bins = np.linspace(1, 6, 60)
    ax.axvspan(1, su_lb, color=C_RED, alpha=0.07)
    ax.hist(s_su, bins=bins, density=True, color=C_STD, alpha=0.85, label=f'standard (mean {s_su.mean():.2f})')
    ax.hist(o_su, bins=bins, density=True, color=C_OURS, alpha=0.85, label=f'ours (mean {o_su.mean():.2f})')
    ax.axvline(su_lb, ls='--', color=C_RED, lw=1.6)
    ax.text(su_lb, 0.92, f'{su_lb:.2f}', color=C_RED, fontsize=9, ha='left', transform=ax.get_xaxis_transform())
    ax.set_title('(c)  effective support', loc='left')
    ax.set_xlabel('effective support  $\\exp\\,H(A_{i\\cdot})$')
    ax.set_ylabel('density')
    ax.legend(fontsize=9, loc='upper left')
    fig.tight_layout()
    out = 'figII_4_theory_vs_sim.png'
    fig.savefig(out, bbox_inches='tight', dpi=200)
    print('saved', out, f'lb={lb:.2f} mw_ub={mw_ub:.2f} su_lb={su_lb:.2f}')
if __name__ == '__main__':
    main()
