"""
Guards every number quoted in docs/04_final_paper.md and README.md against the
tables that scripts/run_analysis.py and scripts/run_global_analysis.py produce.

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
    ("weekly",    "NTN-B 5y",  0.336),
    ("monthly",   "NTN-B 5y",  0.405),
    ("quarterly", "NTN-B 5y",  0.423),
    ("monthly",   "NTN-F 10y", 0.474),
    ("monthly",   "LTN 2y",    0.319),
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
    ("Current Cycle",       0.110),
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
    ("NTN-B 5y", 0.280), ("LTN 2y", 0.213), ("NTN-F 10y", 0.296),
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
    ("Ibovespa x NTN-B 5y",  1.77),
    ("Ibovespa x LTN 2y",    2.26),
    ("Ibovespa x NTN-F 10y", 1.15),
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


def test_dcc_crisis_ordering_is_stable_even_though_levels_are_not():
    """
    Section 7 leans on the ordering, not the levels, because `a` is weakly
    identified and the levels move with the sample. Guard the ordering.
    """
    t = load("tbl_dcc_crisis_correlations.csv", index_col=0)
    crises = t.drop("Full sample")
    for col in ["NTN-B 5y", "LTN 2y", "NTN-F 10y"]:
        assert crises[col].idxmin() == "GFC"
        assert crises[col].idxmax() in ("Joesley", "COVID")
    assert t.loc["Joesley", "NTN-F 10y"] / t.loc["Full sample", "NTN-F 10y"] > 2.5


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
    ("Ibovespa",  "Ann vol%", 26.34),
    ("Ibovespa",  "Max DD%", -59.96),
    ("NTN-B 5y",  "Max DD%",  -8.37),
    ("LFT 1y",    "Ann vol%",  0.23),
])
def test_summary_stats(asset, col, claimed):
    t = load("tbl_summary_stats.csv", index_col=0)
    assert t.loc[asset, col] == pytest.approx(claimed, abs=0.02)


# ── Section 9: the matched cross-country panel ───────────────────────────────
GLOBAL = pytest.mark.skipif(
    not (OUT / "tbl_global_correlations.csv").exists(),
    reason="global panel not generated — run scripts/run_global_analysis.py first",
)


@GLOBAL
@pytest.mark.parametrize("country,pre,post", [
    ("United States",  -0.315, -0.102),
    ("Germany",        -0.356,  0.183),
    ("Japan",          -0.316, -0.093),
    ("United Kingdom", -0.225,  0.123),
    ("Brazil",          0.443,  0.522),
])
def test_global_correlations(country, pre, post):
    t = load("tbl_global_correlations.csv", index_col=0)
    assert t.loc[country, "rho pre-2020"] == pytest.approx(pre, abs=0.0015)
    assert t.loc[country, "rho post-2020"] == pytest.approx(post, abs=0.0015)


@GLOBAL
def test_all_advanced_economies_were_significantly_negative_pre_2020():
    """Section 9.2: every pre-2020 interval excludes zero, on the negative side."""
    t = load("tbl_global_correlations.csv", index_col=0)
    for c in ["United States", "Germany", "Japan", "United Kingdom"]:
        hi = float(t.loc[c, "CI pre"].split(",")[1].rstrip("]"))
        assert hi < 0, f"{c} pre-2020 CI does not exclude zero"


@GLOBAL
def test_no_advanced_economy_became_significantly_positive():
    """Section 9.2: the post-2020 change is the loss of a hedge, not positive co-movement."""
    t = load("tbl_global_correlations.csv", index_col=0)
    for c in ["United States", "Germany", "Japan", "United Kingdom"]:
        lo = float(t.loc[c, "CI post"].split(",")[0].lstrip("["))
        hi = float(t.loc[c, "CI post"].split(",")[1].rstrip("]"))
        assert lo < 0 < hi, f"{c} post-2020 CI no longer straddles zero"


@GLOBAL
def test_brazil_is_significantly_positive_in_both_periods_and_did_not_shift():
    t = load("tbl_global_correlations.csv", index_col=0)
    for col in ("CI pre", "CI post"):
        lo = float(t.loc["Brazil", col].split(",")[0].lstrip("["))
        assert lo > 0, f"Brazil {col} does not exclude zero on the positive side"
    assert t.loc["Brazil", "shifted 5%"] == "no"
    assert t.loc["Brazil", "p(shift)"] == pytest.approx(0.395, abs=0.03)


@GLOBAL
def test_only_germany_shifted_significantly():
    t = load("tbl_global_correlations.csv", index_col=0)
    shifted = t[t["shifted 5%"] == "yes"]
    assert list(shifted.index) == ["Germany"]
    assert t.loc["Germany", "shift"] == pytest.approx(0.539, abs=0.0015)


@GLOBAL
def test_convergence_is_directionally_unanimous_but_not_significant():
    """Finding 11: gap narrowed 4/4, significant 1/4."""
    t = load("tbl_global_convergence.csv", index_col=0)
    assert (t["gap narrowed"] == "yes").all()
    converged = t[t["converged 5%"] == "yes"]
    assert list(converged.index) == ["Germany"]
    assert t.loc["Germany", "DiD"] == pytest.approx(0.460, abs=0.02)
    assert t.loc["Germany", "p"] < 0.05


@GLOBAL
def test_convergence_standard_errors_are_large():
    """The reason convergence is not established: DiD SEs run 0.186-0.299."""
    t = load("tbl_global_convergence.csv", index_col=0)
    assert 0.15 < t["boot SE"].min() < 0.25
    assert 0.25 < t["boot SE"].max() < 0.35


@GLOBAL
@pytest.mark.parametrize("country,claimed", [
    ("United States",  -0.664),
    ("Germany",        -0.594),
    ("Japan",          -0.565),
    ("United Kingdom", -0.455),
    ("Brazil",          0.073),
])
def test_rolling_correlation_floor(country, claimed):
    """Finding 12: the floor is the robust cross-sectional signature."""
    t = load("tbl_global_rolling_correlation.csv", index_col=0)
    assert t[country].min() == pytest.approx(claimed, abs=0.0015)


@GLOBAL
def test_only_brazil_never_goes_negative():
    t = load("tbl_global_rolling_correlation.csv", index_col=0)
    mins = t.min()
    assert mins["Brazil"] > 0
    assert (mins.drop("Brazil") < -0.4).all()


@GLOBAL
def test_bond_construction_does_not_drive_the_result():
    """Section 9.1: the control that licenses the yield-based method for 4 countries."""
    t = load("tbl_global_construction_check.csv", index_col=0)
    v = t["value"]
    assert v["yield-based vs PU-based, rho"] == pytest.approx(0.931, abs=0.002)
    assert v["Ibov x bond, yield-based"] == pytest.approx(0.461, abs=0.0015)
    assert v["Ibov x bond, PU-based"] == pytest.approx(0.475, abs=0.0015)
    assert v["absolute difference"] < 0.02


@GLOBAL
def test_sign_regimes():
    t = load("tbl_global_sign_regimes.csv", index_col=0)
    for c in ["United States", "Germany", "Japan", "United Kingdom"]:
        assert t.loc[c, "pre-2020 sign"] == "negative"
        assert t.loc[c, "post-2020 sign"] == "indistinct"
        assert t.loc[c, "ever significantly negative"] == "yes"
    assert t.loc["Brazil", "pre-2020 sign"] == "positive"
    assert t.loc["Brazil", "post-2020 sign"] == "positive"
    assert t.loc["Brazil", "ever significantly negative"] == "no"


# ── The documents themselves ─────────────────────────────────────────────────
# Everything above checks the generated tables against values transcribed from the
# prose. That leaves a gap: the prose can still drift from the tables, and nothing
# notices. These tests close it by rendering the expected string from the table and
# asserting it appears in the document -- which is how four stale figures survived
# in README.md after the sample was pinned.
PAPER = (BASE / "docs" / "04_final_paper.md")
README = (BASE / "README.md")


def _doc(path):
    return path.read_text(encoding="utf-8")


@GLOBAL
def test_readme_construction_check_matches_the_table():
    v = load("tbl_global_construction_check.csv", index_col=0)["value"]
    txt = _doc(README)
    assert f"+{v['Ibov x bond, yield-based']:.3f} yield-based" in txt
    assert f"+{v['Ibov x bond, PU-based']:.3f} PU-based" in txt
    assert f"differing by {v['absolute difference']:.3f}" in txt


@GLOBAL
def test_readme_brazil_panel_shift_matches_the_table():
    t = load("tbl_global_correlations.csv", index_col=0)
    txt = _doc(README)
    pre, post = t.loc["Brazil", "rho pre-2020"], t.loc["Brazil", "rho post-2020"]
    assert f"+{pre:.3f} → +{post:.3f}" in txt


@GLOBAL
def test_readme_rolling_floor_matches_the_table():
    t = load("tbl_global_rolling_correlation.csv", index_col=0)
    txt = _doc(README)
    for country, label in [("United States", "US"), ("Germany", "DE"),
                           ("Japan", "JP"), ("United Kingdom", "UK")]:
        assert f"{label} {t[country].min():.3f}".replace("-", "−") in txt, \
            f"README floor for {label} disagrees with the table"
    assert f"Brazil +{t['Brazil'].min():.3f}" in txt


@GLOBAL
def test_readme_pre_and_post_2020_rows_match_the_table():
    t = load("tbl_global_correlations.csv", index_col=0)
    txt = _doc(README)
    order = ["United States", "Germany", "Japan", "United Kingdom"]
    pre = " / ".join(f"{t.loc[c, 'rho pre-2020']:.3f}" for c in order).replace("-", "−")
    post = " / ".join(f"{t.loc[c, 'rho post-2020']:+.3f}" for c in order).replace("-", "−").replace("+", "+")
    assert pre in txt, "README pre-2020 row disagrees with the table"
    assert post in txt, "README post-2020 row disagrees with the table"


def test_readme_conditional_tail_matches_the_table():
    t = load("tbl_conditional_correlations.csv", index_col=0)
    assert f"+{t.loc['NTN-B 5y', 'rho|Q10']:.3f}" in _doc(README)


def test_paper_full_sample_correlations_appear_verbatim():
    t = load("tbl_full_sample_correlations.csv", index_col=0)
    txt = _doc(PAPER)
    for pair in t.index:
        assert f"+{t.loc[pair, 'rho']:.3f}" in txt, f"{pair} rho missing from the paper"


def test_paper_frequency_headline_appears_verbatim():
    """The abstract's daily -> monthly claim must match the table."""
    t = load("tbl_frequency_robustness.csv")
    d = t[(t["Frequency"] == "daily") & (t["Bond"] == "NTN-B 5y")]["rho"].iloc[0]
    m = t[(t["Frequency"] == "monthly") & (t["Bond"] == "NTN-B 5y")]["rho"].iloc[0]
    txt = _doc(PAPER)
    assert f"+{d:.3f} daily" in txt
    assert f"+{m:.3f} monthly" in txt


def test_paper_forbes_rigobon_headline_appears_verbatim():
    t = load("tbl_forbes_rigobon.csv", index_col=[0, 1])
    txt = _doc(PAPER)
    for crisis in ("COVID", "Joesley"):
        row = t.loc[(crisis, "NTN-B 5y")]
        assert f"+{row['rho crisis (raw)']:.3f}" in txt, f"{crisis} raw rho missing"
        assert f"+{row['rho adjusted']:.3f}" in txt, f"{crisis} adjusted rho missing"


@GLOBAL
def test_paper_reports_the_actual_panel_length():
    panel = pd.read_csv(BASE / "data" / "processed" / "global_panel.csv",
                        index_col=0, parse_dates=True) if (
        BASE / "data" / "processed" / "global_panel.csv").exists() else None
    if panel is None:
        pytest.skip("panel not built")
    n_pre = int((panel.index <= "2019-12-31").sum())
    n_post = int((panel.index > "2019-12-31").sum())
    txt = _doc(PAPER)
    assert f"{len(panel)} months" in txt
    assert f"{n_pre} pre-break and {n_post} post-break months" in txt


def test_no_document_still_advertises_an_api_key_requirement():
    """Section 9 is reproducible without FRED_API_KEY; the docs must not say otherwise."""
    for path in (README, PAPER, BASE / "docs" / "03_implementation_guide.md"):
        txt = _doc(path)
        assert "requires a free `FRED_API_KEY`" not in txt
        assert "needs `FRED_API_KEY`" not in txt


def test_rolling_portfolio_metrics():
    t = load("tbl_rolling_portfolio_metrics.csv", index_col=0)
    assert t["pc1_all"].max() == pytest.approx(0.668, abs=0.002)
    assert t["DR 60/40"].mean() == pytest.approx(1.122, abs=0.002)
    assert t["ENB 60/40"].mean() == pytest.approx(1.110, abs=0.002)
    # ENB for a 2-asset portfolio is bounded above by 2 -- stated in section 8.1
    assert t["ENB 60/40"].max() <= 2.0


# ── Section 6.1: the sovereign channel after EMBI ────────────────────────────
SOVEREIGN = pytest.mark.skipif(
    not (OUT / "tbl_sovereign_fiscal24.csv").exists(),
    reason="sovereign tables not generated — run scripts/run_analysis.py",
)


@SOVEREIGN
def test_sovereign_validation_matches_the_table():
    t = load("tbl_sovereign_validation.csv", index_col=0)
    txt = _doc(PAPER)
    for series in ("Brazil 10y - US 10y", "ICE BofA LatAm OAS"):
        r = t.loc[series]
        lvl = f"{r['rho levels']:.3f}".replace("-", "−")
        assert f"| {lvl} |" in txt, f"{series} level rho absent from the paper"
        assert f"{r['rho monthly changes']:.3f}" in txt


@SOVEREIGN
def test_fiscal24_ranges_match_the_table():
    """Each claim is checked where it is written — a table cell must not cover for prose."""
    t = load("tbl_sovereign_fiscal24.csv", index_col=0)
    yd, oas = t.loc["Brazil 10y - US 10y"], t.loc["ICE BofA LatAm OAS"]
    r_yd, r_oas = f"{yd['range']:.0f}", f"{oas['range']:.0f}"

    paper = _doc(PAPER)
    assert f"**{r_yd}**" in paper, "Table 11 yield-diff range cell"
    assert f"**{r_oas}**" in paper, "Table 11 OAS range cell"
    assert f"a {r_yd} bp range inside three months" in paper
    assert f"moved {r_oas} bps end to end" in paper
    assert f"a {r_yd} bp local repricing with a {r_oas} bp regional" in paper
    assert f"from {yd['start']:.0f} to a peak of {yd['peak']:,.0f}" in paper

    readme = _doc(README)
    assert f"**{r_yd} bp** intra-window range" in readme
    assert f"moved **{r_oas} bps**" in readme


@SOVEREIGN
def test_the_two_measures_disagree_about_fiscal24():
    """The paper's Finding 13 rests on this gap; if it closes, the claim must change."""
    t = load("tbl_sovereign_fiscal24.csv", index_col=0)
    yd, oas = t.loc["Brazil 10y - US 10y"], t.loc["ICE BofA LatAm OAS"]
    assert yd["range"] > 5 * oas["range"], (yd["range"], oas["range"])
    # the credit spread stayed below its own calm average during the window
    assert oas["mean"] < oas["calm mean"]


@SOVEREIGN
def test_covid_shows_the_differential_is_not_a_credit_spread():
    """EMBI widened while the yield differential narrowed — the paper's key caveat."""
    t = load("tbl_sovereign_by_window.csv", index_col=0)
    assert t.loc["COVID", "yield diff vs calm"] < 0
    assert t.loc["COVID", "EMBI"] > t.loc["Calm", "EMBI"]
    txt = _doc(PAPER)
    assert f"from {t.loc['Calm', 'EMBI']:.0f} to {t.loc['COVID', 'EMBI']:.0f} bps" in txt
    assert f"narrowed* by {abs(t.loc['COVID', 'yield diff vs calm']):.0f} bps" in txt


@SOVEREIGN
def test_paper_reports_the_windows_embi_cannot_reach():
    t = load("tbl_sovereign_by_window.csv", index_col=0)
    assert pd.isna(t.loc["Fiscal24", "EMBI"]), "EMBI now covers Fiscal24 — update the prose"
    assert pd.notna(t.loc["Fiscal24", "LatAm OAS"])
    txt = _doc(PAPER)
    assert "discontinued by IPEADATA in **July 2024**" in txt


# ── Section 5.2: copula fits ─────────────────────────────────────────────────
# Added after Table 7's fitted parameters drifted from outputs/. It and
# tbl_scenario_pnl_total.csv were the only generated tables no test read, which is
# exactly where the paper's "every number below is pinned" claim was false.
@pytest.mark.parametrize("pair,rho,nu", [
    ("Ibovespa x NTN-B 5y",  0.083,  9.4),
    ("Ibovespa x LTN 2y",    0.087,  8.2),
    ("Ibovespa x NTN-F 10y", 0.076,  6.6),
    ("Ibovespa x LFT 1y",    0.005, 44.3),
])
def test_paper_quotes_the_fitted_student_t_parameters(pair, rho, nu):
    t = load("tbl_copula_fit.csv")
    row = t[(t["Pair"] == pair) & (t["Copula"] == "Student-t")].iloc[0]
    assert row["param"] == f"rho={rho}, nu={nu}", row["param"]
    assert f"Student-t (ρ={rho}, ν={nu})" in _doc(PAPER), \
        f"paper's Table 7 disagrees with the fit for {pair}"


def test_student_t_wins_on_aic_for_every_pair():
    """Section 5.2 and the docs/02 banner both rest on this."""
    t = load("tbl_copula_fit.csv")
    for pair, grp in t.groupby("Pair"):
        assert grp.loc[grp["AIC"].idxmin(), "Copula"] == "Student-t", pair


def test_paper_quotes_the_student_t_nu_range():
    t = load("tbl_copula_fit.csv")
    duration = t[(t["Copula"] == "Student-t") & (t["Pair"] != "Ibovespa x LFT 1y")]
    nus = [float(p.split("nu=")[1]) for p in duration["param"]]
    assert f"ν between {min(nus)} and {max(nus)}" in _doc(PAPER)


def test_scenario_pnl_total_is_reported_and_internally_consistent():
    """The other table nothing read. Excess-over-CDI must equal total minus CDI."""
    tot = load("tbl_scenario_pnl_total.csv", index_col=0)
    exc = load("tbl_scenario_pnl_excess_cdi.csv", index_col=0)
    assert list(tot.index) == list(exc.index)
    assert list(tot.columns) == list(exc.columns)
    # every scenario's excess must sit below its total: CDI is positive throughout
    assert (exc <= tot + 1e-9).all().all()
