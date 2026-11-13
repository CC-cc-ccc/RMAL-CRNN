import numpy as np
import matplotlib.pyplot as plt
import synthesis as M
M.apply_style()
DOM_COLORS = {'HHT': '#1f3b57', 'LOFAR': '#3a86a8', 'FFT': '#c0392b', 'WPT': '#e0a526', 'HHSA': '#7d4f9e', 'MFCC': '#2f7d4f'}
DOM_MARK = {'HHT': 'o', 'LOFAR': 's', 'FFT': '^', 'WPT': 'D', 'HHSA': 'v', 'MFCC': 'P'}
KAPPAS = np.array([0, 0.5, 1, 2, 4, 8], dtype=float)

def impulsive_crosstalk(y, kappa, rng, fs=M.FS):
    n = len(y)
    spikes = (rng.random(n) < 0.003).astype(float) * rng.standard_normal(n)
    k = np.exp(-np.arange(40) / 6.0) * np.cos(2 * np.pi * M.F_B * np.arange(40) / fs)
    burst = np.convolve(spikes, k, mode='same')
    return y + kappa * burst * y.std() / (burst.std() + 1e-09)

def robust_residual(x):
    c = np.median(x, axis=0, keepdims=True)
    s = 1.4826 * np.median(np.abs(x - c), axis=0, keepdims=True) + 1e-06
    return (x - c) / s

def main():
    rng = np.random.default_rng(2026)
    frac = {n: [] for n in M.FEATURE_NAMES}
    fsd = {n: [] for n in M.FEATURE_NAMES}
    for kap in KAPPAS:
        tmp = {n: [] for n in M.FEATURE_NAMES}
        for _ in range(30):
            _, s = M.clean_source(3, rng)
            _, y = M.degrade(s, rng, snr_db=10, sir_db=15)
            y = impulsive_crosstalk(y, kap, rng)
            v = M.extract_views(y)[0]
            for n in M.FEATURE_NAMES:
                tmp[n].append((np.abs(robust_residual(v[n])) > 3).mean())
        for n in M.FEATURE_NAMES:
            frac[n].append(np.mean(tmp[n]))
            fsd[n].append(np.std(tmp[n]))
    fig, ax = plt.subplots(figsize=(8.8, 5.1))
    for n in M.FEATURE_NAMES:
        f = np.array(frac[n]) * 100
        sd = np.array(fsd[n]) * 100
        ax.plot(KAPPAS, f, color=DOM_COLORS[n], marker=DOM_MARK[n], ms=6, lw=2.0, label=n)
        ax.fill_between(KAPPAS, f - sd, f + sd, color=DOM_COLORS[n], alpha=0.08)
    ax.set_xlabel('crosstalk strength  $\\kappa$')
    ax.set_ylabel('outlier coordinates  ($|\\rho|>3$, %)')
    ax.grid(True, alpha=0.4)
    ax.margins(y=0.04)
    ax.legend(ncol=2, fontsize=9.5, loc='upper left', title='feature domain', title_fontsize=9)
    axin = ax.inset_axes([0.56, 0.22, 0.4, 0.25])
    _, s = M.clean_source(3, rng)
    _, y0 = M.degrade(s, rng, snr_db=10, sir_db=15)
    yk = impulsive_crosstalk(y0.copy(), 4.0, rng)
    f0 = M.extract_views(y0)[0]['FFT'][10]
    fk = M.extract_views(yk)[0]['FFT'][10]
    axin.plot(f0, color=M.C_INK, lw=1.4, label='clean')
    axin.plot(fk, color=DOM_COLORS['FFT'], lw=1.5, label='$\\kappa{=}4$')
    axin.set_title('FFT feature domain (single frame)', fontsize=8.5)
    axin.set_xticks([])
    axin.set_yticks([])
    axin.legend(fontsize=7.5, loc='upper right', handlelength=1.2)
    for sp in axin.spines.values():
        sp.set_edgecolor('black')
        sp.set_linewidth(1.3)
    fig.tight_layout()
    out = 'figI_A_domain_corruption.png'
    fig.savefig(out, bbox_inches='tight')
    print('saved', out)
if __name__ == '__main__':
    main()
