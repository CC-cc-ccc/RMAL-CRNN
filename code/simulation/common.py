import numpy as np
import synthesis as ms
_PROJ_RNG = np.random.default_rng(7)
_W_FUSE, _ = np.linalg.qr(_PROJ_RNG.standard_normal((ms.D_SIGMA, ms.PROJ_OUT)))
BETA_INIT = 0.5
C_INK = ms.C_INK
C_OURS = ms.C_OURS
C_STD = ms.C_STD
C_WARN = ms.C_WARN
C_AMBER = ms.C_AMBER
C_TEAL = ms.C_TEAL
VIEW_COLORS = ms.VIEW_COLORS

def fused_feature(sig):
    views, T = ms.extract_views(sig)
    cat = np.concatenate([views[n] for n in ms.FEATURE_NAMES], axis=1)
    return cat @ _W_FUSE

def view_tokens(sig):
    views, T = ms.extract_views(sig)
    return np.stack([views[n].mean(axis=0) for n in ms.FEATURE_NAMES], axis=0)

def add_impulse(sig, rng, eps=0.06, amp=8.0):
    n = len(sig)
    sigma = sig.std() + 1e-12
    mask = rng.random(n) < eps
    mag = amp * sigma * (0.5 + np.abs(rng.standard_normal(n)))
    sgn = rng.choice([-1.0, 1.0], size=n)
    out = sig.copy()
    out[mask] += (mag * sgn)[mask]
    return out

def psi_student(rho, nu):
    return (nu + 1.0) * rho / (nu + rho ** 2)

def psi_ls(rho):
    return rho

def gross_error_sensitivity(nu):
    return (nu + 1.0) / (2.0 * np.sqrt(nu))

def irls_location_scale(x, nu=4.0, n_iter=50, eps=1e-09):
    x = np.asarray(x, dtype=float)
    mu = np.median(x)
    s = 1.4826 * np.median(np.abs(x - mu)) + eps
    for _ in range(n_iter):
        rho = (x - mu) / s
        w = (nu + 1.0) / (nu + rho ** 2)
        mu = np.sum(w * x) / np.sum(w)
    return (mu, s)

def robust_gate_pipeline(Z, nu=4.0):
    mu = np.median(Z, axis=0, keepdims=True)
    s = 1.4826 * np.median(np.abs(Z - mu), axis=0, keepdims=True) + 1e-09
    rho = (Z - mu) / s
    g = (nu + 1.0) / (nu + rho ** 2)
    Zg = Z * g
    return ms.spherical_tangent_embed(Zg)

def layernorm_pipeline(Z):
    return ms.layernorm(Z)

def attention_entropy(A):
    A = np.clip(A, 1e-12, 1.0)
    H = -(A * np.log(A)).sum(axis=1)
    return float(H.mean())

def effective_support(A):
    return float(np.exp(attention_entropy(A)))

def max_attention(A):
    return float(A.max(axis=1).mean())

def lma_support_floor(beta, V):
    Delta = 4.0 * beta
    w = np.exp(np.array([0.0] + [-Delta] * (V - 1)))
    w = w / w.sum()
    H = -(w * np.log(w)).sum()
    return float(np.exp(H))

def lma_maxweight_cap(beta, V):
    Delta = 4.0 * beta
    return 1.0 / (1.0 + (V - 1) * np.exp(-Delta))

def fused_repr(tokens, A):
    return (A @ tokens).reshape(-1)

def cosine(a, b, eps=1e-12):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + eps))

def feature_snr_db(F0, Fhat):
    num = (F0 ** 2).sum()
    den = ((Fhat - F0) ** 2).sum() + 1e-12
    return float(10.0 * np.log10(num / den + 1e-12))

def effective_rank(M):
    s = np.linalg.svd(M, compute_uv=False)
    p = s / (s.sum() + 1e-12)
    p = p[p > 1e-12]
    return float(np.exp(-(p * np.log(p)).sum()))

def tsvd_denoise(F, r):
    U, s, Vt = np.linalg.svd(F, full_matrices=False)
    s2 = s.copy()
    s2[r:] = 0.0
    return U * s2 @ Vt

def period_matrix(sig, cls):
    fc = ms.CLASS_PARAMS[cls][0]
    L = int(round(ms.FS / fc))
    nseg = len(sig) // L
    return sig[:nseg * L].reshape(nseg, L)

def gavish_donoho_threshold(M, sigma=None):
    m, n = M.shape
    b = min(m, n) / max(m, n)
    lam = np.sqrt(2.0 * (b + 1.0) + 8.0 * b / (b + 1.0 + np.sqrt(b ** 2 + 14.0 * b + 1.0)))
    s = np.linalg.svd(M, compute_uv=False)
    if sigma is not None:
        return (float(lam * np.sqrt(max(m, n)) * sigma), s)
    y_med = np.median(s)
    omega = lam / np.sqrt(_mp_median(b))
    return (float(omega * y_med), s)

def _mp_median(b):
    lo = (1.0 - np.sqrt(b)) ** 2
    hi = (1.0 + np.sqrt(b)) ** 2
    xs = np.linspace(lo, hi, 20001)
    dens = np.sqrt(np.clip((hi - xs) * (xs - lo), 0, None)) / (2.0 * np.pi * b * xs)
    cdf = np.cumsum(dens) * (xs[1] - xs[0])
    cdf /= cdf[-1]
    return float(xs[np.searchsorted(cdf, 0.5)])

def nsllr_denoise(F, rank=ms.LRR_RANK, lam=0.1, alpha=0.3, seed=0):
    Z, _, U, Vv = ms.diff_nsllr_Z(F, rank=rank, lam=lam, rng=np.random.default_rng(seed), return_factors=True)
    row = Z.sum(axis=1, keepdims=True) + 1e-09
    F_ref = Z / row @ F
    return F + alpha * (F_ref - F)

def oracle_lowrank_gain_db(D, r):
    return float(10.0 * np.log10(D / r))
SIR_OFF = 40.0
SNR_OFF = 40.0

def make_pair(cls, rng, kind, level):
    _, s = ms.clean_source(cls, rng)
    if kind == 'snr':
        _, y, parts = ms.degrade(s, rng, snr_db=level, sir_db=SIR_OFF, return_parts=True)
        return (parts['core'], y)
    if kind == 'sir':
        _, y, parts = ms.degrade(s, rng, snr_db=SNR_OFF, sir_db=level, return_parts=True)
        return (parts['core'], y)
    if kind == 'impulse':
        eps, amp = level
        _, y, parts = ms.degrade(s, rng, snr_db=SNR_OFF, sir_db=SIR_OFF, return_parts=True)
        core = parts['core']
        return (core, add_impulse(core, rng, eps=eps, amp=amp))
    raise ValueError(kind)

def gain_gst(clean, deg, nu=4.0):
    cv, _ = ms.extract_views(clean)
    dv, _ = ms.extract_views(deg)
    C = np.concatenate([cv[n] for n in ms.FEATURE_NAMES], axis=1)
    Dg = np.concatenate([dv[n] for n in ms.FEATURE_NAMES], axis=1)
    n = min(len(C), len(Dg))
    C, Dg = (C[:n], Dg[:n])
    truth = C.std(axis=0) + 1e-09
    s_l2 = Dg.std(axis=0)
    s_rob = 1.4826 * np.median(np.abs(Dg - np.median(Dg, axis=0)), axis=0)
    bias_off = np.abs(s_l2 - truth) / truth
    bias_on = np.abs(s_rob - truth) / truth
    return float(np.mean(bias_off - bias_on))

def gain_lma(clean, deg, beta=BETA_INIT):
    tc, td = (view_tokens(clean), view_tokens(deg))
    bp = np.ones(tc.shape[1])
    A_on_c = ms.logmap_tangent_attention(tc, bp, beta=beta)
    A_on_d = ms.logmap_tangent_attention(td, bp, beta=beta)
    A_of_c = ms.standard_dot_attention(tc)
    A_of_d = ms.standard_dot_attention(td)
    fid_on = cosine(fused_repr(td, A_on_d), fused_repr(tc, A_on_c))
    fid_of = cosine(fused_repr(td, A_of_d), fused_repr(tc, A_of_c))
    return float(fid_on - fid_of)

def gain_nsllr(clean, deg, cls, rank=ms.LRR_RANK):
    P0 = period_matrix(clean, cls)
    P = period_matrix(deg, cls)
    n = min(len(P0), len(P))
    P0, P = (P0[:n], P[:n])
    snr_in = feature_snr_db(P0, P)
    snr_out = feature_snr_db(P0, tsvd_denoise(P, rank))
    return float(snr_out - snr_in)

def dissociation_matrix(n_trials=120, seed=2026, impulse=(0.07, 9.0), sir_db=-3.0, snr_db=-6.0):
    rng = np.random.default_rng(seed)
    conds = [('impulse', impulse), ('sir', sir_db), ('snr', snr_db)]
    rows = ['GST', 'LMA', 'NSLLR']
    M = np.zeros((3, 3))
    for j, (kind, level) in enumerate(conds):
        g_gst, g_lma, g_nsllr = ([], [], [])
        for _ in range(n_trials):
            cls = int(rng.integers(ms.NUM_CLASS))
            clean, deg = make_pair(cls, rng, kind, level)
            g_gst.append(gain_gst(clean, deg))
            g_lma.append(gain_lma(clean, deg))
            g_nsllr.append(gain_nsllr(clean, deg, cls))
        M[0, j] = np.mean(g_gst)
        M[1, j] = np.mean(g_lma)
        M[2, j] = np.mean(g_nsllr)
    return (M, rows, ['Impulse', 'SIR', 'SNR'])
