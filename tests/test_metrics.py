"""
Unit tests for src/metrics.py.

Run:  python -m pytest tests/ -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import metrics as M  # noqa: E402


# ═════════════════════════════════════════════════════════════════════════════
# Copula densities
# ═════════════════════════════════════════════════════════════════════════════
def test_gumbel_collapses_to_independence_at_theta_one():
    """theta = 1 is the independence copula: c(u,v) == 1 everywhere."""
    rng = np.random.default_rng(0)
    u, v = rng.uniform(0.01, 0.99, 500), rng.uniform(0.01, 0.99, 500)
    assert np.allclose(M.gumbel_logpdf(1.0, u, v), 0.0, atol=1e-9)


def test_clayton_collapses_to_independence_as_theta_to_zero():
    rng = np.random.default_rng(1)
    u, v = rng.uniform(0.01, 0.99, 500), rng.uniform(0.01, 0.99, 500)
    assert np.allclose(M.clayton_logpdf(1e-8, u, v), 0.0, atol=1e-5)


def test_gaussian_collapses_to_independence_at_rho_zero():
    rng = np.random.default_rng(2)
    u, v = rng.uniform(0.01, 0.99, 500), rng.uniform(0.01, 0.99, 500)
    assert np.allclose(M.gaussian_logpdf(0.0, u, v), 0.0, atol=1e-12)


@pytest.mark.parametrize("theta", [1.2, 2.0, 4.0])
def test_gumbel_density_integrates_to_one(theta):
    """A valid copula density integrates to 1 over the unit square."""
    g = (np.arange(400) + 0.5) / 400
    U, V = np.meshgrid(g, g)
    total = np.exp(M.gumbel_logpdf(theta, U.ravel(), V.ravel())).sum() / 400 ** 2
    assert total == pytest.approx(1.0, rel=0.02)


@pytest.mark.parametrize("theta", [0.5, 2.0])
def test_clayton_density_integrates_to_one(theta):
    g = (np.arange(400) + 0.5) / 400
    U, V = np.meshgrid(g, g)
    total = np.exp(M.clayton_logpdf(theta, U.ravel(), V.ravel())).sum() / 400 ** 2
    assert total == pytest.approx(1.0, rel=0.02)


def test_student_t_converges_to_gaussian_for_large_nu():
    rng = np.random.default_rng(3)
    u, v = rng.uniform(0.02, 0.98, 4000), rng.uniform(0.02, 0.98, 4000)
    rho = 0.4
    ll_t = M.student_t_logpdf(rho, 5000.0, u, v).sum()
    ll_g = M.gaussian_logpdf(rho, u, v).sum()
    assert abs(ll_t - ll_g) / abs(ll_g) < 0.02


def test_student_t_density_integrates_to_one():
    g = (np.arange(300) + 0.5) / 300
    U, V = np.meshgrid(g, g)
    total = np.exp(M.student_t_logpdf(0.3, 6.0, U.ravel(), V.ravel())).sum() / 300 ** 2
    assert total == pytest.approx(1.0, rel=0.03)


def test_copula_fit_recovers_gaussian_dependence():
    """Data simulated from a Gaussian copula should be best fit by Gaussian or t."""
    rng = np.random.default_rng(4)
    rho = 0.5
    z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], 6000)
    u, v = stats.norm.cdf(z[:, 0]), stats.norm.cdf(z[:, 1])
    fits = M.fit_all_copulas(u, v)
    assert fits[0]["family"] in ("Gaussian", "Student-t")
    gauss = next(f for f in fits if f["family"] == "Gaussian")
    assert gauss["rho"] == pytest.approx(rho, abs=0.05)


def test_copula_fit_recovers_clayton_lower_tail():
    """Clayton-simulated data must be picked out as Clayton, not Gumbel."""
    rng = np.random.default_rng(5)
    theta = 2.0
    u = rng.uniform(size=8000)
    w = rng.uniform(size=8000)
    v = (u ** (-theta) * (w ** (-theta / (1 + theta)) - 1) + 1) ** (-1 / theta)
    fits = M.fit_all_copulas(u, v)
    assert fits[0]["family"] == "Clayton"
    assert fits[0]["theta"] == pytest.approx(theta, rel=0.15)


def test_gumbel_likelihood_is_bounded_on_independent_data():
    """
    Regression test for the mis-specified density: on independent data the fitted
    Gumbel log-likelihood must sit near 0, not run away to thousands of nats.
    """
    rng = np.random.default_rng(6)
    u, v = rng.uniform(size=5000), rng.uniform(size=5000)
    fit = M.fit_gumbel(u, v)
    assert abs(fit["ll"]) < 25
    assert fit["theta"] == pytest.approx(1.0, abs=0.05)


# ═════════════════════════════════════════════════════════════════════════════
# DCC
# ═════════════════════════════════════════════════════════════════════════════
def _sim_dcc(n, a, b, rho_bar, seed):
    rng = np.random.default_rng(seed)
    Qbar = np.array([[1.0, rho_bar], [rho_bar, 1.0]])
    Q = Qbar.copy()
    eps = np.zeros((n, 2))
    e = np.zeros(2)
    for t in range(n):
        Q = (1 - a - b) * Qbar + a * np.outer(e, e) + b * Q
        d = np.sqrt(np.diag(Q))
        R = Q / np.outer(d, d)
        e = rng.multivariate_normal([0, 0], R)
        eps[t] = e
    return eps


def test_dcc_recovers_time_varying_correlation():
    import pandas as pd
    eps = _sim_dcc(4000, a=0.05, b=0.90, rho_bar=0.3, seed=7)
    idx = pd.RangeIndex(len(eps))
    fit = M.fit_dcc(pd.Series(eps[:, 0], index=idx), pd.Series(eps[:, 1], index=idx))
    assert fit["a"] == pytest.approx(0.05, abs=0.04)
    assert fit["persistence"] > 0.7
    assert fit["t_a"] > 2.0          # a > 0 detected as significant


def test_dcc_reports_near_zero_a_on_constant_correlation():
    import pandas as pd
    rng = np.random.default_rng(8)
    z = rng.multivariate_normal([0, 0], [[1, 0.3], [0.3, 1]], 4000)
    idx = pd.RangeIndex(len(z))
    fit = M.fit_dcc(pd.Series(z[:, 0], index=idx), pd.Series(z[:, 1], index=idx))
    assert fit["a"] < 0.02
    rho = fit["rho"]
    assert rho.std() < 0.06          # essentially a flat correlation path


def test_dcc_rho_path_has_no_lookahead():
    """rho at t=0 must depend only on Qbar, not on eps[0]."""
    eps = _sim_dcc(200, a=0.05, b=0.9, rho_bar=0.4, seed=9)
    Qbar = eps.T @ eps / len(eps)
    r1 = M.dcc_rho_path(eps, 0.05, 0.9, Qbar)
    eps2 = eps.copy()
    eps2[0] = [5.0, -5.0]            # perturb only the first observation
    r2 = M.dcc_rho_path(eps2, 0.05, 0.9, Qbar)
    assert r1[0] == pytest.approx(r2[0])


# ═════════════════════════════════════════════════════════════════════════════
# Inference and bias correction
# ═════════════════════════════════════════════════════════════════════════════
def test_fisher_ci_covers_true_rho():
    rng = np.random.default_rng(10)
    rho, hits = 0.25, 0
    for k in range(200):
        z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], 400)
        r = np.corrcoef(z[:, 0], z[:, 1])[0, 1]
        lo, hi = M.fisher_ci(r, 400)
        hits += lo <= rho <= hi
    assert 0.90 <= hits / 200 <= 0.99


def test_forbes_rigobon_finds_no_contagion_when_only_volatility_rises():
    """
    Same data-generating correlation, higher variance in the source market.
    Raw correlation rises; the adjusted estimate should not.
    """
    rng = np.random.default_rng(11)
    beta = 0.3
    x_c = rng.normal(0, 1.0, 4000)
    y_c = beta * x_c + rng.normal(0, 1.0, 4000)
    x_k = rng.normal(0, 3.0, 500)          # crisis: 9x the variance
    y_k = beta * x_k + rng.normal(0, 1.0, 500)
    out = M.forbes_rigobon(x_c, y_c, x_k, y_k)
    assert out["rho_crisis"] > out["rho_calm"] + 0.15     # raw correlation "surges"
    # The adjustment must remove essentially all of that spurious surge. The residual
    # gap is sampling noise centred on zero, so assert on magnitude, not on sign --
    # the `contagion` flag is a knife-edge comparison and flips seed to seed here.
    spurious = out["rho_crisis"] - out["rho_calm"]
    residual = abs(out["rho_adj"] - out["rho_calm"])
    assert residual < 0.06
    assert residual < 0.25 * spurious


def test_forbes_rigobon_detects_genuine_contagion():
    rng = np.random.default_rng(12)
    x_c = rng.normal(0, 1.0, 4000)
    y_c = 0.2 * x_c + rng.normal(0, 1.0, 4000)
    x_k = rng.normal(0, 2.0, 600)
    y_k = 1.5 * x_k + rng.normal(0, 1.0, 600)   # propagation genuinely stronger
    out = M.forbes_rigobon(x_c, y_c, x_k, y_k)
    assert out["contagion"] is True


def test_bootstrap_corr_diff_detects_a_real_difference():
    rng = np.random.default_rng(13)
    a = rng.multivariate_normal([0, 0], [[1, 0.0], [0.0, 1]], 1500)
    b = rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 1500)
    out = M.bootstrap_corr_diff(a[:, 0], a[:, 1], b[:, 0], b[:, 1], n_boot=400)
    assert out["p"] < 0.05


def test_bootstrap_did_detects_convergence():
    """
    Country A's correlation rises from -0.4 to +0.1; the benchmark B stays at +0.4.
    The difference-in-differences must be positive and significant.
    """
    rng = np.random.default_rng(20)

    def draw(rho, n):
        return rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], n)

    a0, a1 = draw(-0.4, 180), draw(0.1, 180)
    b0, b1 = draw(0.4, 180), draw(0.4, 180)
    out = M.bootstrap_did((a0[:, 0], a0[:, 1]), (a1[:, 0], a1[:, 1]),
                          (b0[:, 0], b0[:, 1]), (b1[:, 0], b1[:, 1]),
                          n_boot=600, block=6)
    assert out["did"] > 0.3
    assert out["p"] < 0.05


def test_bootstrap_did_null_is_not_rejected():
    """Both countries shift by the same amount — DiD is zero, no convergence."""
    rng = np.random.default_rng(21)

    def draw(rho, n):
        return rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], n)

    a0, a1 = draw(-0.3, 200), draw(0.0, 200)
    b0, b1 = draw(0.2, 200), draw(0.5, 200)      # same +0.3 shift
    out = M.bootstrap_did((a0[:, 0], a0[:, 1]), (a1[:, 0], a1[:, 1]),
                          (b0[:, 0], b0[:, 1]), (b1[:, 0], b1[:, 1]),
                          n_boot=600, block=6)
    assert abs(out["did"]) < 0.2
    assert out["p"] > 0.05


def test_bootstrap_did_is_antisymmetric():
    """Swapping the country and the benchmark must flip the sign of the DiD."""
    rng = np.random.default_rng(22)

    def draw(rho, n):
        return rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], n)

    a0, a1, b0, b1 = draw(-0.4, 150), draw(0.1, 150), draw(0.4, 150), draw(0.4, 150)
    fwd = M.bootstrap_did((a0[:, 0], a0[:, 1]), (a1[:, 0], a1[:, 1]),
                          (b0[:, 0], b0[:, 1]), (b1[:, 0], b1[:, 1]), n_boot=200)
    rev = M.bootstrap_did((b0[:, 0], b0[:, 1]), (b1[:, 0], b1[:, 1]),
                          (a0[:, 0], a0[:, 1]), (a1[:, 0], a1[:, 1]), n_boot=200)
    assert fwd["did"] == pytest.approx(-rev["did"], abs=1e-9)


def test_bootstrap_corr_diff_null_is_not_rejected():
    rng = np.random.default_rng(14)
    a = rng.multivariate_normal([0, 0], [[1, 0.2], [0.2, 1]], 1500)
    b = rng.multivariate_normal([0, 0], [[1, 0.2], [0.2, 1]], 1500)
    out = M.bootstrap_corr_diff(a[:, 0], a[:, 1], b[:, 0], b[:, 1], n_boot=400)
    assert out["p"] > 0.05


# ═════════════════════════════════════════════════════════════════════════════
# Portfolio metrics
# ═════════════════════════════════════════════════════════════════════════════
def test_diversification_ratio_bounds():
    perfect = np.array([[1.0, 1.0], [1.0, 1.0]])
    assert M.diversification_ratio([0.5, 0.5], perfect) == pytest.approx(1.0)
    indep = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert M.diversification_ratio([0.5, 0.5], indep) == pytest.approx(np.sqrt(2), rel=1e-6)


def test_effective_num_bets_bounds():
    indep = np.eye(2)
    assert M.effective_num_bets([0.5, 0.5], indep) == pytest.approx(2.0, rel=1e-6)
    perfect = np.array([[1.0, 0.999999], [0.999999, 1.0]])
    assert M.effective_num_bets([0.5, 0.5], perfect) < 1.05


def test_pc1_share_bounds():
    import pandas as pd
    rng = np.random.default_rng(15)
    indep = pd.DataFrame(rng.normal(size=(2000, 4)))
    assert M.pc1_share(indep) == pytest.approx(0.25, abs=0.05)
    f = rng.normal(size=2000)
    coll = pd.DataFrame({c: f + rng.normal(0, 0.01, 2000) for c in range(4)})
    assert M.pc1_share(coll) > 0.95


def test_portfolio_var_matches_closed_form_and_respects_horizon():
    import pandas as pd
    cov = pd.DataFrame([[0.0004, 0.0], [0.0, 0.0004]], columns=["a", "b"], index=["a", "b"])
    w = {"a": 0.5, "b": 0.5}
    v10 = M.portfolio_var(w, cov, alpha=0.99, horizon_days=10)
    expected = stats.norm.ppf(0.99) * np.sqrt(0.5 ** 2 * 0.0004 * 2) * np.sqrt(10) * 100
    assert v10 == pytest.approx(expected, rel=1e-9)
    # scaling is sqrt-of-time in the horizon argument only
    assert M.portfolio_var(w, cov, horizon_days=40) == pytest.approx(v10 * 2, rel=1e-9)


def test_portfolio_var_stays_below_capital_at_sane_horizons():
    """A long-only unlevered portfolio cannot lose more than 100% of capital."""
    import pandas as pd
    # COVID-scale daily vol: 6% a day for equities
    cov = pd.DataFrame([[0.0036, 0.0006], [0.0006, 0.0004]],
                       columns=["eq", "bd"], index=["eq", "bd"])
    v = M.portfolio_var({"eq": 0.6, "bd": 0.4}, cov, alpha=0.99, horizon_days=10)
    assert 0 < v < 100


def test_shrink_to_equicorr_endpoints():
    cov = np.array([[0.04, 0.002], [0.002, 0.01]])
    assert np.allclose(M.shrink_to_equicorr(cov, 0.0), cov)
    full = M.shrink_to_equicorr(cov, 1.0)
    vols = np.sqrt(np.diag(full))
    assert full[0, 1] / (vols[0] * vols[1]) == pytest.approx(1.0, abs=1e-6)


def test_tail_dependence_empirical_on_independent_data():
    rng = np.random.default_rng(16)
    u, v = rng.uniform(size=20000), rng.uniform(size=20000)
    out = M.tail_dependence_empirical(u, v, q=0.05)
    assert out["lambda_L"] == pytest.approx(0.05, abs=0.02)
    assert out["lambda_U"] == pytest.approx(0.05, abs=0.02)
