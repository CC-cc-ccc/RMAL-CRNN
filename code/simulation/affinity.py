import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
import synthesis as M
M.apply_style()
C_NS = '#0d7d7d'
C_RAW = '#d98c00'
C_CLEAN = '#9aa3ad'
RANK = 8
AFF_CMAP = LinearSegmentedColormap.from_list('aff', ['#ffffff', '#0d7d7d', '#06403f'])

def feat_matrix(snr, rng, n=100):
    X = []
    for _ in range(n):
        c = int(rng.integers(10))
        _, s = M.clean_source(c, rng)
        _, y = M.degrade(s, rng, snr_db=snr, sir_db=40)
        v = M.extract_views(y)[0]
        X.append(np.concatenate([v[k].mean(0) for k in M.FEATURE_NAMES]))
    return np.array(X)

def cum_energy(A, k=24):
    s = np.linalg.svd(A, compute_uv=False) ** 2
    return (np.cumsum(s) / s.sum())[:k]

def topr_energy(A, r=RANK):
    s = np.linalg.svd(A, compute_uv=False) ** 2
    return np.cumsum(s)[r - 1] / s.sum()

def main():
    SNRS = np.array([20, 15, 10, 5, 0, -5, -10, -15, -20], float)
    raw_m, ns_m, raw_s, ns_s = ([], [], [], [])
    for snr in SNRS:
        rr, nn = ([], [])
        for t in range(5):
            rng = np.random.default_rng(100 + t)
            X = feat_matrix(snr, rng)
            Xc = X - X.mean(0)
            rr.append(topr_energy(Xc @ Xc.T))
            nn.append(topr_energy(M.diff_nsllr_Z(Xc, rank=RANK, lam=0.1, rng=rng)))
        raw_m.append(np.mean(rr))
        ns_m.append(np.mean(nn))
        raw_s.append(np.std(rr))
        ns_s.append(np.std(nn))
    raw_m, ns_m = (np.array(raw_m), np.array(ns_m))
    raw_s, ns_s = (np.array(raw_s), np.array(ns_s))
    rng = np.random.default_rng(7)
    Xn = feat_matrix(-10, rng)
    Xnc = Xn - Xn.mean(0)
    Xcl = feat_matrix(20, np.random.default_rng(7))
    Xclc = Xcl - Xcl.mean(0)
    Araw = Xnc @ Xnc.T
    Z = M.diff_nsllr_Z(Xnc, rank=RANK, lam=0.1, rng=rng)
    Acl = Xclc @ Xclc.T
    e_raw, e_ns, e_cl = (cum_energy(Araw), cum_energy(Z), cum_energy(Acl))
    spars = (Z == 0).mean()
    fig = plt.figure(figsize=(13.0, 5.2))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.05, 1.0], wspace=0.26)
    ax = fig.add_subplot(gs[0, 0])
    xk = np.arange(1, len(e_raw) + 1)
    ax.plot(xk, e_cl, color=C_CLEAN, lw=2.0, ls='--', label='clean reference (SNR 20 dB)')
    ax.plot(xk, e_ns, color=C_NS, lw=2.8, marker='o', ms=5, label='NSLLR (ours)')
    ax.plot(xk, e_raw, color=C_RAW, lw=2.6, marker='s', ms=5, label='raw representation')
    ax.axvline(RANK, color='#444', ls=':', lw=1.3)
    ax.text(RANK, 1.04, f'rank budget r = {RANK}', fontsize=8.5, color='#444', ha='center', va='bottom')
    ax.set_xlabel('number of components')
    ax.set_ylabel('cumulative spectral energy')
    ax.set_title('(a)  energy compaction at SNR = −10 dB', loc='left')
    ax.set_xlim(1, len(e_raw))
    ax.set_ylim(0, 1.08)
    ax.grid(True, alpha=0.4)
    ax.legend(loc='center right', fontsize=9)
    ax_in1 = ax.inset_axes([0.4, 0.06, 0.26, 0.26])
    ax_in2 = ax.inset_axes([0.7, 0.06, 0.26, 0.26])
    nshow = 40
    rawn = np.abs(Araw[:nshow, :nshow])
    rawn /= rawn.max()
    zn = Z[:nshow, :nshow]
    zn = zn / (zn.max() + 1e-09)
    ax_in1.imshow(rawn, cmap=AFF_CMAP, vmin=0, vmax=1)
    ax_in1.set_title('raw\n(dense)', fontsize=7.5, color=C_RAW, pad=2)
    ax_in2.imshow(zn, cmap=AFF_CMAP, vmin=0, vmax=1)
    ax_in2.set_title(f'NSLLR\n({spars * 100:.0f}% sparse, ≥0)', fontsize=7.5, color=C_NS, pad=2)
    for a in (ax_in1, ax_in2):
        a.set_xticks([])
        a.set_yticks([])
    ax = fig.add_subplot(gs[0, 1])
    ax.fill_between(SNRS, raw_m, ns_m, where=ns_m >= raw_m, color=C_NS, alpha=0.18, interpolate=True, label='compaction gain')
    ax.plot(SNRS, ns_m, color=C_NS, lw=2.8, marker='o', ms=6, label='NSLLR (ours)')
    ax.fill_between(SNRS, ns_m - ns_s, ns_m + ns_s, color=C_NS, alpha=0.15)
    ax.plot(SNRS, raw_m, color=C_RAW, lw=2.6, marker='s', ms=6, label='raw representation')
    ax.fill_between(SNRS, raw_m - raw_s, raw_m + raw_s, color=C_RAW, alpha=0.15)
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel(f'top-{RANK} spectral energy fraction')
    ax.set_title('(b)  low-rank retention across the SNR envelope', loc='left')
    ax.invert_xaxis()
    ax.set_ylim(0.6, 1.02)
    ax.grid(True, alpha=0.4)
    ax.legend(loc='lower left', fontsize=9)
    fig.tight_layout()
    out = 'figIII_nsllr.png'
    fig.savefig(out, bbox_inches='tight', dpi=200)
    print('saved', out)
if __name__ == '__main__':
    main()
