"""
metrics.py — Estimators for the Brazilian stock-bond correlation study.

Everything that has a closed form and can be unit-tested lives here rather than in
notebook strings, so the notebooks stay presentation code.

Contents
--------
  Inference        : fisher_ci, corr_with_ci, bootstrap_corr_diff
  Bias correction  : forbes_rigobon
  Copulas          : gaussian / student-t / clayton / gumbel log-densities + fitters
  DCC-GARCH        : fit_dcc (with standard errors), dcc_rho_path
  Portfolio        : diversification_ratio, effective_num_bets, pc1_share, portfolio_var
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize, minimize_scalar
from scipy.special import gammaln
from scipy.stats import rankdata, t as t_dist

TRADING_DAYS = 252


# ═════════════════════════════════════════════════════════════════════════════
# Inference
# ═════════════════════════════════════════════════════════════════════════════
def fisher_ci(rho, n, alpha=0.05):
    """Fisher z confidence interval for a Pearson correlation."""
    if not np.isfinite(rho) or n < 4:
        return (np.nan, np.nan)
    rho = np.clip(rho, -0.999999, 0.999999)
    z, se = np.arctanh(rho), 1.0 / np.sqrt(n - 3)
    k = stats.norm.ppf(1 - alpha / 2)
    return float(np.tanh(z - k * se)), float(np.tanh(z + k * se))


def corr_with_ci(x, y, alpha=0.05):
    """Pearson rho with Fisher CI and a two-sided p-value against rho = 0."""
    p = pd.concat([pd.Series(x), pd.Series(y)], axis=1).dropna()
    n = len(p)
    if n < 4:
        return dict(rho=np.nan, lo=np.nan, hi=np.nan, p=np.nan, n=n, sig=False)
    r, pv = stats.pearsonr(p.iloc[:, 0], p.iloc[:, 1])
    lo, hi = fisher_ci(r, n, alpha)
    return dict(rho=float(r), lo=lo, hi=hi, p=float(pv), n=n, sig=bool(lo * hi > 0))


def bootstrap_did(pre_a, post_a, pre_b, post_b, n_boot=2000, block=6, seed=0):
    """
    Difference-in-differences of correlations, with a block-bootstrap p-value.

        DiD = [rho(post_a) - rho(pre_a)] - [rho(post_b) - rho(pre_b)]

    Each argument is a 2-column array-like (x, y) whose correlation is taken.
    Used to test convergence: `a` is the country under test, `b` the benchmark.
    DiD > 0 means a's correlation rose by more than the benchmark's, which is
    what "moving toward the benchmark" means when a starts below it.

    Testing the *gap* rather than each level separately matters because both
    correlations are estimated with error; comparing two point estimates across
    two periods without a joint test invites reading noise as convergence.
    """
    rng = np.random.default_rng(seed)

    def prep(pair):
        return pd.concat([pd.Series(pair[0]), pd.Series(pair[1])], axis=1).dropna().values

    A0, A1 = prep(pre_a), prep(post_a)
    B0, B1 = prep(pre_b), prep(post_b)
    if min(len(A0), len(A1), len(B0), len(B1)) < block * 3:
        return dict(did=np.nan, p=np.nan, boot_se=np.nan)

    def rho(arr):
        return np.corrcoef(arr[:, 0], arr[:, 1])[0, 1]

    def boot_rho(arr):
        n = len(arr)
        idx = []
        while len(idx) < n:
            s = rng.integers(0, n)
            idx.extend(range(s, min(s + block, n)))
        return rho(arr[np.array(idx[:n])])

    obs = (rho(A1) - rho(A0)) - (rho(B1) - rho(B0))
    draws = np.array([(boot_rho(A1) - boot_rho(A0)) - (boot_rho(B1) - boot_rho(B0))
                      for _ in range(n_boot)])
    draws = draws[np.isfinite(draws)]
    p = float(np.mean(np.abs(draws - draws.mean()) >= abs(obs)))
    return dict(did=float(obs), p=p, boot_se=float(draws.std()))


def bootstrap_corr_diff(x1, y1, x2, y2, n_boot=2000, block=21, seed=0):
    """
    Stationary-block-bootstrap p-value for H0: rho(sample 1) == rho(sample 2).

    Blocks preserve the serial dependence in daily returns, so the p-value is not
    inflated by autocorrelation the way an iid bootstrap would be.
    """
    rng = np.random.default_rng(seed)
    a = pd.concat([pd.Series(x1), pd.Series(y1)], axis=1).dropna().values
    b = pd.concat([pd.Series(x2), pd.Series(y2)], axis=1).dropna().values
    if len(a) < block * 3 or len(b) < block * 3:
        return dict(diff=np.nan, p=np.nan)

    def boot_rho(arr):
        n = len(arr)
        idx = []
        while len(idx) < n:
            s = rng.integers(0, n)
            idx.extend(range(s, min(s + block, n)))
        idx = np.array(idx[:n])
        s = arr[idx]
        return np.corrcoef(s[:, 0], s[:, 1])[0, 1]

    obs = np.corrcoef(a[:, 0], a[:, 1])[0, 1] - np.corrcoef(b[:, 0], b[:, 1])[0, 1]
    diffs = np.array([boot_rho(a) - boot_rho(b) for _ in range(n_boot)])
    diffs = diffs[np.isfinite(diffs)]
    # p-value for the null of no difference: centre the bootstrap on zero
    p = float(np.mean(np.abs(diffs - diffs.mean()) >= abs(obs)))
    return dict(diff=float(obs), p=p, boot_se=float(diffs.std()))


# ═════════════════════════════════════════════════════════════════════════════
# Forbes-Rigobon heteroskedasticity correction
# ═════════════════════════════════════════════════════════════════════════════
def forbes_rigobon(x_calm, y_calm, x_crisis, y_crisis):
    """
    Forbes & Rigobon (2002) volatility-adjusted correlation.

    A rise in the variance of the *source* market x mechanically inflates the
    measured correlation even with no change in the underlying propagation
    mechanism.  The adjusted estimate removes that bias:

        rho* = rho_c / sqrt(1 + delta * (1 - rho_c^2)),
        delta = var(x_crisis)/var(x_calm) - 1

    "Contagion" means rho* still exceeds the calm-period rho; otherwise the
    apparent crisis correlation surge is only "interdependence".
    """
    c = pd.concat([pd.Series(x_calm), pd.Series(y_calm)], axis=1).dropna()
    k = pd.concat([pd.Series(x_crisis), pd.Series(y_crisis)], axis=1).dropna()
    if len(c) < 30 or len(k) < 10:
        return dict(rho_calm=np.nan, rho_crisis=np.nan, rho_adj=np.nan,
                    delta=np.nan, contagion=None, n_calm=len(c), n_crisis=len(k))
    rho_calm   = c.iloc[:, 0].corr(c.iloc[:, 1])
    rho_crisis = k.iloc[:, 0].corr(k.iloc[:, 1])
    delta = k.iloc[:, 0].var() / c.iloc[:, 0].var() - 1.0
    if delta <= -1:
        rho_adj = np.nan
    else:
        rho_adj = rho_crisis / np.sqrt(1 + delta * (1 - rho_crisis ** 2))
    return dict(rho_calm=float(rho_calm), rho_crisis=float(rho_crisis),
                rho_adj=float(rho_adj), delta=float(delta),
                contagion=bool(rho_adj > rho_calm) if np.isfinite(rho_adj) else None,
                n_calm=len(c), n_crisis=len(k))


# ═════════════════════════════════════════════════════════════════════════════
# Copulas
# ═════════════════════════════════════════════════════════════════════════════
def pseudo_obs(df):
    """Rank-transform returns to uniform pseudo-observations in (0, 1)."""
    n = len(df)
    return pd.DataFrame({c: rankdata(df[c]) / (n + 1) for c in df.columns},
                        index=df.index)


def _clip_u(u, eps=1e-10):
    return np.clip(np.asarray(u, dtype=float), eps, 1 - eps)


def gaussian_logpdf(rho, u, v):
    """log c(u,v) for the Gaussian copula."""
    if abs(rho) >= 1:
        return np.full_like(np.asarray(u, dtype=float), -np.inf)
    x, y = stats.norm.ppf(_clip_u(u)), stats.norm.ppf(_clip_u(v))
    return (-0.5 * np.log(1 - rho ** 2)
            - rho ** 2 * (x ** 2 + y ** 2) / (2 * (1 - rho ** 2))
            + rho * x * y / (1 - rho ** 2))


def student_t_logpdf(rho, nu, u, v):
    """
    log c(u,v) for the Student-t copula.

    The bivariate *t* density over the *product of marginal t densities* — using a
    bivariate normal density here (a common slip) forces nu to its upper bound and
    silently collapses the fit onto the Gaussian.
    """
    if abs(rho) >= 1 or nu <= 2:
        return np.full_like(np.asarray(u, dtype=float), -np.inf)
    x = t_dist.ppf(_clip_u(u), df=nu)
    y = t_dist.ppf(_clip_u(v), df=nu)
    q = (x ** 2 - 2 * rho * x * y + y ** 2) / (1 - rho ** 2)
    # log bivariate t density
    log_biv = (gammaln((nu + 2) / 2) - gammaln(nu / 2)
               - np.log(nu * np.pi) - 0.5 * np.log(1 - rho ** 2)
               - (nu + 2) / 2 * np.log1p(q / nu))
    return log_biv - t_dist.logpdf(x, df=nu) - t_dist.logpdf(y, df=nu)


def clayton_logpdf(theta, u, v):
    """log c(u,v) for the Clayton copula (lower-tail dependence)."""
    if theta <= 0:
        return np.full_like(np.asarray(u, dtype=float), -np.inf)
    u, v = _clip_u(u), _clip_u(v)
    return (np.log1p(theta)
            - (1 + theta) * (np.log(u) + np.log(v))
            - (1 / theta + 2) * np.log(u ** (-theta) + v ** (-theta) - 1))


def gumbel_logpdf(theta, u, v):
    """
    log c(u,v) for the Gumbel copula (upper-tail dependence).

        c = exp(-A)/(uv) * (lu*lv)^(t-1) * A^(1-2t) * (A + t - 1),
        A = (lu^t + lv^t)^(1/t),  lu = -ln u,  lv = -ln v

    At theta = 1 this must collapse to the independence copula, c == 1
    (see test_metrics.py). Exponents of A^(2-2t) or a bracket of (A^t + t - 1)
    give a function that is not a density and whose likelihood is unbounded.
    """
    if theta < 1:
        return np.full_like(np.asarray(u, dtype=float), -np.inf)
    u, v = _clip_u(u), _clip_u(v)
    lu, lv = -np.log(u), -np.log(v)
    A = (lu ** theta + lv ** theta) ** (1.0 / theta)
    return (-A - np.log(u) - np.log(v)
            + (theta - 1) * (np.log(lu) + np.log(lv))
            + (1 - 2 * theta) * np.log(A)
            + np.log(A + theta - 1))


def fit_gaussian(u, v):
    r = minimize_scalar(lambda p: -np.sum(gaussian_logpdf(p, u, v)),
                        bounds=(-0.99, 0.99), method="bounded")
    return dict(family="Gaussian", rho=float(r.x), ll=float(-r.fun), k=1,
                lambda_L=0.0, lambda_U=0.0, param=f"rho={r.x:.3f}")


def fit_student_t(u, v):
    best = None
    for nu0 in (4.0, 8.0, 20.0):
        r = minimize(lambda p: -np.sum(student_t_logpdf(p[0], p[1], u, v)),
                     x0=[np.corrcoef(stats.norm.ppf(_clip_u(u)),
                                     stats.norm.ppf(_clip_u(v)))[0, 1], nu0],
                     method="L-BFGS-B", bounds=[(-0.99, 0.99), (2.1, 200.0)])
        if best is None or r.fun < best.fun:
            best = r
    rho, nu = float(best.x[0]), float(best.x[1])
    lam = 2 * t_dist.cdf(-np.sqrt((nu + 1) * (1 - rho) / (1 + rho)), df=nu + 1)
    return dict(family="Student-t", rho=rho, nu=nu, ll=float(-best.fun), k=2,
                lambda_L=float(lam), lambda_U=float(lam),
                param=f"rho={rho:.3f}, nu={nu:.1f}")


def fit_clayton(u, v):
    r = minimize_scalar(lambda p: -np.sum(clayton_logpdf(p, u, v)),
                        bounds=(1e-4, 30), method="bounded")
    th = float(r.x)
    return dict(family="Clayton", theta=th, ll=float(-r.fun), k=1,
                lambda_L=float(2 ** (-1 / th)), lambda_U=0.0, param=f"theta={th:.3f}")


def fit_gumbel(u, v):
    r = minimize_scalar(lambda p: -np.sum(gumbel_logpdf(p, u, v)),
                        bounds=(1.0, 20), method="bounded")
    th = float(r.x)
    return dict(family="Gumbel", theta=th, ll=float(-r.fun), k=1,
                lambda_L=0.0, lambda_U=float(2 - 2 ** (1 / th)), param=f"theta={th:.3f}")


def fit_all_copulas(u, v):
    """Fit the four families and rank them by AIC (lower is better)."""
    n = len(u)
    fits = [fit_gaussian(u, v), fit_student_t(u, v), fit_clayton(u, v), fit_gumbel(u, v)]
    for f in fits:
        f["AIC"] = 2 * f["k"] - 2 * f["ll"]
        f["BIC"] = f["k"] * np.log(n) - 2 * f["ll"]
    return sorted(fits, key=lambda f: f["AIC"])


def tail_dependence_empirical(u, v, q=0.05):
    """
    Non-parametric lower/upper tail dependence: P(V<q | U<q) and P(V>1-q | U>1-q).
    Independence implies both equal q; the ratio to q is the excess co-crash rate.
    """
    u, v = np.asarray(u), np.asarray(v)
    lo = (u < q)
    hi = (u > 1 - q)
    lam_L = float(np.mean(v[lo] < q)) if lo.sum() else np.nan
    lam_U = float(np.mean(v[hi] > 1 - q)) if hi.sum() else np.nan
    return dict(q=q, lambda_L=lam_L, lambda_U=lam_U,
                n_lower=int(lo.sum()), n_co_lower=int(((u < q) & (v < q)).sum()),
                expected_indep=float(len(u) * q * q))


# ═════════════════════════════════════════════════════════════════════════════
# DCC-GARCH
# ═════════════════════════════════════════════════════════════════════════════
def dcc_negloglik(params, eps, Qbar):
    """Negative concentrated DCC log-likelihood (Engle 2002, stage 2)."""
    a, b = params
    if a <= 0 or b <= 0 or a + b >= 0.9999:
        return 1e10
    n = len(eps)
    Q, ll = Qbar.copy(), 0.0
    for t in range(1, n):
        e = eps[t - 1]
        Q = (1 - a - b) * Qbar + a * np.outer(e, e) + b * Q
        d = np.sqrt(np.diag(Q))
        R = Q / np.outer(d, d)
        sign, logdet = np.linalg.slogdet(R)
        if sign <= 0:
            return 1e10
        et = eps[t]
        try:
            rinv = np.linalg.inv(R)
        except np.linalg.LinAlgError:
            return 1e10
        ll += -0.5 * (logdet + et @ rinv @ et - et @ et)
    return -ll


def dcc_rho_path(eps, a, b, Qbar):
    """Conditional correlation path. Q_1 = Qbar, so there is no look-ahead at t=0."""
    Q, rho = Qbar.copy(), np.empty(len(eps))
    for t in range(len(eps)):
        if t > 0:
            e = eps[t - 1]
            Q = (1 - a - b) * Qbar + a * np.outer(e, e) + b * Q
        d = np.sqrt(np.diag(Q))
        R = Q / np.outer(d, d)
        rho[t] = R[0, 1]
    return rho


def _num_hessian(f, x, h=1e-4):
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            xpp, xpm, xmp, xmm = x.copy(), x.copy(), x.copy(), x.copy()
            xpp[i] += h; xpp[j] += h
            xpm[i] += h; xpm[j] -= h
            xmp[i] -= h; xmp[j] += h
            xmm[i] -= h; xmm[j] -= h
            H[i, j] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * h * h)
    return H


def fit_dcc(std_resid_a, std_resid_b, multistart=True):
    """
    Stage-2 DCC(1,1) fit with standard errors from the numerical Hessian.

    Returns rho_t plus (a, b), their standard errors and t-statistics, so
    "correlations are time-varying" (a > 0) becomes a testable claim rather than
    an assertion. Multiple starts avoid the boundary optimum that a single start
    at (0.05, 0.90) tends to fall into.
    """
    sa, sb = std_resid_a.dropna(), std_resid_b.dropna()
    common = sa.index.intersection(sb.index)
    eps = np.column_stack([sa.loc[common].values, sb.loc[common].values])
    eps = eps / eps.std(axis=0)
    Qbar = eps.T @ eps / len(eps)

    starts = [(0.02, 0.95), (0.05, 0.90), (0.10, 0.80), (0.01, 0.50)] if multistart \
        else [(0.05, 0.90)]
    best = None
    for s in starts:
        r = minimize(dcc_negloglik, list(s), args=(eps, Qbar), method="L-BFGS-B",
                     bounds=[(1e-5, 0.5), (1e-3, 0.999)], options={"maxiter": 500})
        if best is None or r.fun < best.fun:
            best = r

    a, b = float(best.x[0]), float(best.x[1])
    try:
        H = _num_hessian(lambda p: dcc_negloglik(p, eps, Qbar), np.array([a, b]))
        cov = np.linalg.inv(H)
        se = np.sqrt(np.abs(np.diag(cov)))
    except Exception:
        se = np.array([np.nan, np.nan])

    rho = pd.Series(dcc_rho_path(eps, a, b, Qbar), index=common, name="dcc_rho")

    def _t(v, s):
        # A vanishing SE means the Hessian is singular (flat likelihood), not
        # infinite precision -- report NaN rather than an absurd t-statistic.
        return float(v / s) if (np.isfinite(s) and s > 1e-8) else np.nan

    # DCC is not identified when the conditional correlation path is degenerate
    identified = bool(a > 1e-5 and rho.std() > 1e-6)
    return dict(rho=rho, a=a, b=b, se_a=float(se[0]), se_b=float(se[1]),
                t_a=_t(a, se[0]), t_b=_t(b, se[1]),
                persistence=a + b, ll=float(-best.fun), n=len(eps),
                identified=identified)


# ═════════════════════════════════════════════════════════════════════════════
# Portfolio metrics
# ═════════════════════════════════════════════════════════════════════════════
def diversification_ratio(weights, cov):
    """DR = (w'sigma) / sqrt(w'Sigma w). 1 = no benefit."""
    w = np.asarray(weights, dtype=float)
    vols = np.sqrt(np.diag(cov))
    pv = np.sqrt(w @ cov @ w)
    return float((w @ vols) / pv) if pv > 1e-12 else np.nan


def effective_num_bets(weights, cov):
    """Meucci (2009) ENB: exponential Shannon entropy of PC risk contributions."""
    w = np.asarray(weights, dtype=float)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 0)
    p = (eigvecs.T @ w) ** 2 * eigvals
    s = p.sum()
    if s < 1e-12:
        return np.nan
    p = p / s
    p = p[p > 1e-12]
    return float(np.exp(-np.sum(p * np.log(p))))


def pc1_share(returns_window):
    """Share of variance explained by PC1 of the correlation matrix."""
    X = returns_window.dropna()
    if len(X) < X.shape[1] + 2:
        return np.nan
    C = np.corrcoef(X.values, rowvar=False)
    ev = np.linalg.eigvalsh(C)[::-1]
    return float(ev[0] / ev.sum())


def portfolio_var(weights, cov_daily, alpha=0.99, horizon_days=TRADING_DAYS):
    """
    Gaussian VaR over an explicit horizon, as a positive percentage loss.

    `cov_daily` must be a *daily* covariance matrix; the horizon is applied here so
    the scaling is visible at the call site. Square-root-of-time scaling of a
    crisis-window covariance to one year assumes crisis volatility persists for a
    full year and will produce losses above 100% of capital — quote a horizon that
    matches the estimation window, or label the result explicitly as a
    counterfactual.
    """
    cols  = [c for c in cov_daily.columns if c in weights]
    cov_s = cov_daily.loc[cols, cols].values
    w = np.array([weights[c] for c in cols], dtype=float)
    w = w / w.sum()
    vol = np.sqrt(w @ cov_s @ w) * np.sqrt(horizon_days)
    return float(-stats.norm.ppf(1 - alpha) * vol * 100)


def shrink_to_equicorr(cov, alpha):
    """Shrink the correlation matrix toward all-ones (rho -> 1), keeping vols fixed."""
    vols = np.sqrt(np.diag(cov))
    corr = cov / np.outer(vols, vols)
    s_corr = (1 - alpha) * corr + alpha * np.ones_like(corr)
    np.fill_diagonal(s_corr, 1.0)
    ev = np.linalg.eigvalsh(s_corr)
    if ev.min() < 0:
        s_corr = s_corr + (-ev.min() + 1e-8) * np.eye(len(s_corr))
    return np.outer(vols, vols) * s_corr
