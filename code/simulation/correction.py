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

def gate_apply(x, nu=5.0):
    rho = robust_residual(x)
    return (x * M.student_t_gate_weight(rho, nu), rho)

def panel_gate(ax, rng, n_rep=30):
    gw = {n: [] for n in M.FEATURE_NAMES}
    for kap in KAPPAS:
        tmp = {n: [] for n in M.FEATURE_NAMES}
        for _ in range(n_rep):
            _, s = M.clean_source(3, rng)
            _, y = M.degrade(s, rng, snr_db=10, sir_db=15)
            y = impulsive_crosstalk(y, kap, rng)
            v = M.extract_views(y)[0]
            for n in M.FEATURE_NAMES:
                rho = robust_residual(v[n])
                w = M.student_t_gate_weight(rho, 5.0)
                thr = np.quantile(np.abs(rho), 0.9)
                tmp[n].append(w[np.abs(rho) >= thr].mean())
        for n in M.FEATURE_NAMES:
            gw[n].append(np.mean(tmp[n]))
    for n in M.FEATURE_NAMES:
        ax.plot(KAPPAS, gw[n], color=DOM_COLORS[n], marker=DOM_MARK[n], ms=6, lw=2.0, label=n)
    ax.set_xlabel('crosstalk strength  $\\kappa$')
    ax.set_ylabel('gate weight on corrupted decile')
    ax.set_title('(a)  the gate suppresses the corruption', loc='left')
    ax.grid(True, alpha=0.4)
    ax.legend(ncol=2, fontsize=8.5, loc='center', bbox_to_anchor=(0.62, 0.4), title='feature domain', title_fontsize=8.5)

def panel_correction(ax, rng):

    def rep_ln(y):
        v = M.extract_views(y)[0]
        return np.concatenate([(v[k].mean(0) - v[k].mean()) / (v[k].std() + 1e-06) for k in M.FEATURE_NAMES])

    def rep_gst(y, nu=5.0):
        v = M.extract_views(y)[0]
        out = []
        for k in M.FEATURE_NAMES:
            g, _ = gate_apply(v[k], nu)
            out.append(M.spherical_tangent_embed(g).mean(0))
        return np.concatenate(out)
    dl_m, dl_s, dg_m, dg_s = ([], [], [], [])
    for kap in KAPPAS:
        dl, dg = ([], [])
        for cls in [0, 3, 6, 9]:
            for _ in range(12):
                _, s = M.clean_source(cls, rng)
                _, y0 = M.degrade(s, rng, snr_db=10, sir_db=15)
                y = impulsive_crosstalk(y0.copy(), kap, rng)
                l0, lk = (rep_ln(y0), rep_ln(y))
                g0, gk = (rep_gst(y0), rep_gst(y))
                dl.append(np.linalg.norm(lk - l0) / (np.linalg.norm(l0) + 1e-09))
                dg.append(np.linalg.norm(gk - g0) / (np.linalg.norm(g0) + 1e-09))
        dl_m.append(np.mean(dl))
        dl_s.append(np.std(dl))
        dg_m.append(np.mean(dg))
        dg_s.append(np.std(dg))
    dl_m, dl_s = (np.array(dl_m), np.array(dl_s))
    dg_m, dg_s = (np.array(dg_m), np.array(dg_s))
    ax.plot(KAPPAS, dl_m, color='#9aa3ad', marker='s', ms=6, lw=2.4, label='LayerNorm  (M1: no GST)')
    ax.fill_between(KAPPAS, dl_m - dl_s, dl_m + dl_s, color='#9aa3ad', alpha=0.18)
    ax.plot(KAPPAS, dg_m, color='#b5275a', marker='o', ms=6, lw=2.6, label='GST (ours)')
    ax.fill_between(KAPPAS, dg_m - dg_s, dg_m + dg_s, color='#b5275a', alpha=0.16)
    ax.set_xlabel('crosstalk strength  $\\kappa$')
    ax.set_ylabel('fused-representation drift')
    ax.set_title('(b)  net correction vs the baseline', loc='left')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.4)

def main():
    rng = np.random.default_rng(2026)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    panel_gate(axes[0], rng)
    panel_correction(axes[1], rng)
    fig.tight_layout()
    out = 'figI_B_gst_correction.png'
    fig.savefig(out, bbox_inches='tight')
    print('saved', out)
if __name__ == '__main__':
    main()
