"""
Guards every number quoted in docs/04_final_paper.md and README.md against the
tables that scripts/run_analysis.py actually produces.

The failure mode this exists to prevent is a paper that drifts away from its own
code: a table is regenerated, the prose is not updated, and the two disagree
silently. Run after any change to the data pipeline or the estimators.

Skipped when outputs/ has not been generated:
    python3 src/fetch.py && python3 scripts/run_analysis.py && python3 -m pytest tests/ -q
"""

import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

BASE = Path(__file__).parent.parent
OUT = BASE / "outputs"

pytestmark = pytest.mark.skipif(
    not (OUT / "tbl_full_sample_correlations.csv").exists(),
    reason="outputs/ not generated — run scripts/run_analysis.py first",
)


def load(name, **kw):
    return pd.read_csv(OUT / name, **kw)


# ── Section 4.1: full-sample correlations ────────────────────────────────────
@pytest.mark.parametrize("pair,claimed", [
    ("Ibovespa x NTN-B 5y",  0.122),
    ("Ibovespa x LTN 2y",    0.134),
    ("Ibovespa x NTN-F 10y", 0.147),
    ("Ibovespa x LFT 1y",    0.005),
])
def test_full_sample_correlations(pair, claimed):
    t = load("tbl_full_sample_correlations.csv", index_col=0)
    assert t.loc[pair, "rho"] == pytest.approx(claimed, abs=0.0015)


def test_lft_correlation_is_not_significant():
    """The paper states the LFT correlation is indistinguishable from zero."""
    t = load("tbl_full_sample_correlations.csv", index_col=0)
    assert t.loc["Ibovespa x LFT 1y", "sig 5%"] == "no"
    assert (t.drop("Ibovespa x LFT 1y")["sig 5%"] == "yes").all()


# ── Section 4.3: the horizon result ──────────────────────────────────────────
@pytest.mark.parametrize("freq,bond,claimed", [
    ("daily",     "NTN-B 5y",  0.122),
    ("weekly",    "NTN-B 5y",  0.335),
    ("monthly",   "NTN-B 5y",  0.405),
    ("quarterly", "NTN-B 5y",  0.421),
    ("monthly",   "NTN-F 10y", 0.473),
    ("monthly",   "LTN 2y",    0.318),
])
def test_frequency_robustness(freq, bond, claimed):
    t = load("tbl_frequency_robustness.csv")
    got = t[(t["Frequency"] == freq) & (t["Bond"] == bond)]["rho"].iloc[0]
    assert got == pytest.approx(claimed, abs=0.0015)


def test_correlation_rises_with_horizon():
    """Abstract: 'the correlation more than triples from daily to monthly'."""
    t = load("tbl_frequency_robustness.csv")
    d = t[(t["Frequency"] == "daily") & (t["Bond"] == "NTN-B 5y")]["rho"].iloc[0]
    m = t[(t["Frequency"] == "monthly") & (t["Bond"] == "NTN-B 5y")]["rho"].iloc[0]
    assert m > 3 * d


# ── Section 4.2: regimes ─────────────────────────────────────────────────────
@pytest.mark.parametrize("regime,claimed", [
    ("Lula Boom",           0.037),
    ("GFC & Recovery",      0.053),
    ("Dilma Deterioration", 0.114),
    ("Reform Era",          0.199),
    ("COVID & Post-COVID",  0.227),
    ("Current Cycle",       0.109),
])
def test_regime_correlations(regime, claimed):
    t = load("tbl_regime_correlations.csv", index_col=0)
    assert t.loc[regime, "NTN-B 5y"] == pytest.approx(claimed, abs=0.0015)


def test_no_regime_correlation_is_negative():
    """Finding 3: there is a floor — no regime shows a negative correlation."""
    t = load("tbl_regime_correlations.csv", index_col=0)
    assert (t["NTN-B 5y"] >= 0).all()


def test_no_regime_is_statistically_distinguishable():
    """Finding 3: all bootstrap p-values exceed 0.05."""
    t = load("tbl_regime_difference_tests.csv", index_col=0)
    assert (t["differs 5%"] == "no").all()
    assert (t["p"] > 0.05).all()


# ── Section 5: tails ─────────────────────────────────────────────────────────
@pytest.mark.parametrize("bond,claimed", [
    ("NTN-B 5y", 0.279), ("LTN 2y", 0.211), ("NTN-F 10y", 0.294),
])
def test_conditional_tail_correlations(bond, claimed):
    t = load("tbl_conditional_correlations.csv", index_col=0)
    assert t.loc[bond, "rho|Q10"] == pytest.approx(claimed, abs=0.0015)


def test_lower_tail_dependence_exceeds_independence_and_upper_tail():
    """Finding 6: lambda_L = 0.143 vs 0.050 independence, and exceeds lambda_U."""
    t = load("tbl_tail_dependence.csv", index_col=[0, 1])
    row = t.loc[("NTN-B 5y", 0.05)]
    assert row["lambda_L"] == pytest.approx(0.143, abs=0.0015)
    assert row["lambda_U"] == pytest.approx(0.094, abs=0.0015)
    assert row["lambda_L"] > row["lambda_U"]
    assert row["lambda_L"] > 2.5 * 0.05
    assert row["co-crash obs"] == 38


# ── Section 6: Forbes-Rigobon ────────────────────────────────────────────────
@pytest.mark.parametrize("crisis,raw,adj", [
    ("GFC",        0.097, 0.033),
    ("Dilma",      0.107, 0.095),
    ("Joesley",    0.843, 0.580),
    ("COVID",      0.370, 0.124),
    ("Americanas", 0.022, 0.027),
    ("Fiscal24",   0.055, 0.062),
])
def test_forbes_rigobon(crisis, raw, adj):
    t = load("tbl_forbes_rigobon.csv", index_col=[0, 1])
    row = t.loc[(crisis, "NTN-B 5y")]
    assert row["rho crisis (raw)"] == pytest.approx(raw, abs=0.0015)
    assert row["rho adjusted"] == pytest.approx(adj, abs=0.0015)


def test_only_joesley_shows_large_contagion():
    """Finding 4: Joesley is the one unambiguous contagion episode."""
    t = load("tbl_forbes_rigobon.csv", index_col=[0, 1]).xs("NTN-B 5y", level=1)
    calm = t["rho calm"].iloc[0]
    big = t[t["rho adjusted"] > 3 * calm]
    assert list(big.index) == ["Joesley"]


def test_covid_spike_is_mostly_a_volatility_artefact():
    t = load("tbl_forbes_rigobon.csv", index_col=[0, 1])
    row = t.loc[("COVID", "NTN-B 5y")]
    shrinkage = 1 - (row["rho adjusted"] - row["rho calm"]) / (row["rho crisis (raw)"] - row["rho calm"])
    assert shrinkage > 0.8          # >80% of the apparent spike removed


# ── Section 7: DCC ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("pair,claimed_t", [
    ("Ibovespa x NTN-B 5y",  1.94),
    ("Ibovespa x LTN 2y",    2.24),
    ("Ibovespa x NTN-F 10y", 1.14),
])
def test_dcc_t_statistics(pair, claimed_t):
    t = load("tbl_dcc_parameters.csv", index_col=0)
    assert t.loc[pair, "t(a)"] == pytest.approx(claimed_t, abs=0.02)


def test_dcc_time_variation_is_marginal():
    """Finding 5: only the LTN pair rejects a = 0 at 5%; LFT is not identified."""
    t = load("tbl_dcc_parameters.csv", index_col=0)
    yes = t[t["time-varying (a>0, 5%)"] == "yes"]
    assert list(yes.index) == ["Ibovespa x LTN 2y"]
    assert t.loc["Ibovespa x LFT 1y", "time-varying (a>0, 5%)"] == "not identified"


# ── Section 8: portfolio consequences ────────────────────────────────────────
def test_var_never_exceeds_capital():
    """A long-only unlevered portfolio cannot lose more than 100% of capital."""
    t = load("tbl_stressed_var.csv", index_col=0)
    assert t.max().max() < 100


@pytest.mark.parametrize("col,claimed", [("Calm (10d)", 7.5), ("COVID (10d)", 20.9)])
def test_stressed_var(col, claimed):
    t = load("tbl_stressed_var.csv", index_col=0)
    assert t.loc["60/40 (Ibov+NTN-B)", col] == pytest.approx(claimed, abs=0.06)


def test_joesley_var_is_excluded_for_insufficient_observations():
    t = load("tbl_stressed_var.csv", index_col=0)
    assert t["Joesley (10d)"].isna().all()


def test_bonds_cushioned_four_of_six_crises():
    """Finding 7: NTN-B beat CDI in 4 of 6 episodes, failing only in domestic shocks."""
    t = load("tbl_crisis_asset_returns.csv", index_col=0)
    excess = t["NTN-B 5y"] - t["CDI"]
    assert (excess > 0).sum() == 4
    assert set(excess[excess < 0].index) == {"Joesley", "Fiscal24"}
    assert excess["GFC"] == pytest.approx(5.8, abs=0.06)
    assert excess["COVID"] == pytest.approx(1.3, abs=0.06)


def test_lft_only_portfolio_is_flat_in_excess_terms():
    """Holding Selic-linked cash IS the benchmark, so its excess return is ~0."""
    t = load("tbl_scenario_pnl_excess_cdi.csv", index_col=0)
    assert t.loc["LFT only (cash)"].abs().max() < 0.2


def test_sixty_forty_excess_loss_in_gfc():
    t = load("tbl_scenario_pnl_excess_cdi.csv", index_col=0)
    assert t.loc["60/40 (Ibov+NTN-B)", "GFC"] == pytest.approx(-20.0, abs=0.06)


# ── Section 4.1 / 8.1: summary and portfolio metrics ─────────────────────────
def test_ibovespa_sharpe_over_cdi_is_negative():
    """Finding 9 — a claim strong enough that it needs its own guard."""
    t = load("tbl_summary_stats.csv", index_col=0)
    assert t.loc["Ibovespa", "Sharpe (over CDI)"] < 0
    assert t.loc["Ibovespa", "Sharpe (over CDI)"] == pytest.approx(-0.07, abs=0.006)


@pytest.mark.parametrize("asset,col,claimed", [
    ("Ibovespa",  "Ann vol%", 26.32),
    ("Ibovespa",  "Max DD%", -59.96),
    ("NTN-B 5y",  "Max DD%",  -8.37),
    ("LFT 1y",    "Ann vol%",  0.23),
])
def test_summary_stats(asset, col, claimed):
    t = load("tbl_summary_stats.csv", index_col=0)
    assert t.loc[asset, col] == pytest.approx(claimed, abs=0.02)


def test_rolling_portfolio_metrics():
    t = load("tbl_rolling_portfolio_metrics.csv", index_col=0)
    assert t["pc1_all"].max() == pytest.approx(0.668, abs=0.002)
    assert t["DR 60/40"].mean() == pytest.approx(1.122, abs=0.002)
    assert t["ENB 60/40"].mean() == pytest.approx(1.110, abs=0.002)
    # ENB for a 2-asset portfolio is bounded above by 2 -- stated in section 8.1
    assert t["ENB 60/40"].max() <= 2.0
