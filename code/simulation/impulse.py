import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import synthesis as ms
import common as mc
ms.apply_style()
OUT = 'fig_gst_impulse'

def psi_student(rho, nu):
    return (nu + 1.0) * rho / (nu + rho ** 2)

def influence_mean(A):
    return A

def influence_mest(A, nu, c):
    return psi_student(A, nu) / c

def _Epsi_prime(nu, n=200000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    num = (nu + 1.0) * (nu - x ** 2)
    den = (nu + x ** 2) ** 2
    return float(np.mean(num / den))

def sensitivity_curve_sim(A_grid, nu, n=120, n_rep=400, seed=1):
    sc_mean = np.zeros_like(A_grid)
    sc_mest = np.zeros_like(A_grid)
    for rep in range(n_rep):
        rng = np.random.default_rng(seed + rep)
        base = rng.standard_normal(n - 1)
        mean_base = base.mean()
        mu0, _ = mc.irls_location_scale(base, nu=nu)
        for i, A in enumerate(A_grid):
            xs = np.append(base, A)
            sc_mean[i] += n * (xs.mean() - mean_base)
            mu, _ = mc.irls_location_scale(xs, nu=nu)
            sc_mest[i] += n * (mu - mu0)
    return (sc_mean / n_rep, sc_mest / n_rep)
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
ax = axes[0]
rho = np.linspace(-8, 8, 800)
ax.plot(rho, mc.psi_ls(rho), color=mc.C_STD, lw=2.2, ls='--', label='LayerNorm / LS:  $\\psi(\\rho)=\\rho$  (unbounded)')
nus = [2.0, 5.0, 30.0]
greens = ['#7bbf8f', mc.C_OURS, '#15512f']
for nu, col in zip(nus, greens):
    ax.plot(rho, psi_student(rho, nu), color=col, lw=2.4, label=f'Student-$t$ gate, $\\nu={nu:g}$')
    gstar = mc.gross_error_sensitivity(nu)
    ax.plot([np.sqrt(nu)], [gstar], 'o', color=col, ms=5, zorder=5)
    ax.plot([-np.sqrt(nu)], [-gstar], 'o', color=col, ms=5, zorder=5)
nu_a = 5.0
g_a = mc.gross_error_sensitivity(nu_a)
ax.axhline(g_a, color=mc.C_WARN, lw=1.0, ls=':')
ax.axhline(-g_a, color=mc.C_WARN, lw=1.0, ls=':')
ax.annotate(f'$\\gamma^*=\\dfrac{{\\nu+1}}{{2\\sqrt{{\\nu}}}}={g_a:.2f}$' + '\n' + f'at $\\rho=\\sqrt{{\\nu}}$', xy=(np.sqrt(nu_a), g_a), xytext=(3.2, 2.15), fontsize=9, color=mc.C_WARN, arrowprops=dict(arrowstyle='->', color=mc.C_WARN, lw=1.0))
ax.set_xlabel('standardised residual  $\\rho=(z-\\mu)/s$')
ax.set_ylabel('influence  $\\psi(\\rho)$')
ax.set_title('(a)  Bounded vs unbounded influence', loc='left', fontweight='bold')
ax.set_ylim(-4.2, 4.2)
ax.set_xlim(-8, 8)
ax.axhline(0, color='#888', lw=0.6)
ax.axvline(0, color='#888', lw=0.6)
ax.legend(loc='lower right', fontsize=8.2)
ax.grid(alpha=0.5)
ax = axes[1]
nu = 4.0
A = np.linspace(0, 12, 25)
sc_mean, sc_mest = sensitivity_curve_sim(A, nu=nu, n=120, n_rep=500, seed=7)
c = _Epsi_prime(nu)
A_dense = np.linspace(0, 12, 400)
ax.plot(A_dense, influence_mean(A_dense), color=mc.C_STD, lw=2.0, ls='--', label='mean: theory  IF$(A)=A$')
ax.plot(A_dense, influence_mest(A_dense, nu, c), color=mc.C_OURS, lw=2.2, label=f'Student-$t$ gate: theory  IF$(A)=\\psi(A)/\\mathbb{{E}}[\\psi]$')
ax.plot(A, sc_mean, 's', color=mc.C_STD, ms=5, mfc='white', mec=mc.C_STD, label='mean: simulation  SC$_n(A)$')
ax.plot(A, sc_mest, 'o', color=mc.C_OURS, ms=5, label='Student-$t$ gate: simulation  SC$_n(A)$')
A_peak = np.sqrt(nu)
ax.axvline(A_peak, color=mc.C_WARN, lw=1.0, ls=':')
peak_val = influence_mest(A_peak, nu, c)
ax.annotate(f'peak at $A=\\sqrt{{\\nu}}={A_peak:.1f}$,' + '\n' + 'then redescends', xy=(A_peak, peak_val), xytext=(4.6, peak_val + 2.7), fontsize=9, color=mc.C_WARN, arrowprops=dict(arrowstyle='->', color=mc.C_WARN, lw=1.0))
ax.set_xlabel('gross-outlier value  $A$  (units of scale $s$)')
ax.set_ylabel('sensitivity of location estimate  SC$_n(A)$')
ax.set_title('(b)  One spike: bounded perturbation', loc='left', fontweight='bold')
ax.set_xlim(0, 12)
ax.set_ylim(-0.5, 12.5)
ax.legend(loc='upper left', fontsize=8.2)
ax.grid(alpha=0.5)
fig.tight_layout()
fig.savefig(OUT + '.pdf', bbox_inches='tight')
fig.savefig(OUT + '.png', bbox_inches='tight', dpi=200)
print('saved', OUT + '.pdf /.png')
print('  gamma*(nu=4) =', round(mc.gross_error_sensitivity(4.0), 3), " E[psi'] =", round(c, 4))
print('  SC_mean(A=12) = %.2f   SC_mest(A=12) = %.2f' % (sc_mean[-1], sc_mest[-1]))
