"""
Unit tests for the sovereign-risk layer in src/fetch.py.

EMBI+ Brazil was discontinued in July 2024, and the pipeline used to forward-fill it
to the sample end — 501 trading days frozen at 228 bps, covering the whole Fiscal24
window. A reader cannot tell a frozen series from a flat one, so most of these tests
exist to make that failure mode loud rather than silent.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import fetch as F  # noqa: E402


def _embi_frame(tail_value):
    """500 real EMBI observations, then 200 days of `tail_value` (np.nan = no fill)."""
    idx = pd.bdate_range("2022-01-03", periods=700)
    rng = np.random.default_rng(0)
    live = 250 + np.cumsum(rng.normal(0, 3, 500))
    return pd.DataFrame({"embi": np.concatenate([live, np.full(200, tail_value)])},
                        index=idx)


# ═════════════════════════════════════════════════════════════════════════════
# The stale forward-fill guard
# ═════════════════════════════════════════════════════════════════════════════
def test_validate_rejects_embi_frozen_at_a_constant():
    """A long flat tail on EMBI means the ffill guard regressed. It must fail loudly."""
    fails = F.validate_master(_embi_frame(228.0), verbose=False)
    assert any("stale ffill" in f for f in fails), fails


def test_validate_accepts_embi_that_simply_ends():
    """The same frame is fine when the discontinued tail is NaN rather than filled."""
    fails = F.validate_master(_embi_frame(np.nan), verbose=False)
    assert not any("stale ffill" in f for f in fails), fails


def test_validate_rejects_an_fx_rate_posing_as_a_spread():
    """SGS 21619 (EUR/BRL) was used as an EMBI proxy in an earlier draft of the study."""
    idx = pd.bdate_range("2022-01-03", periods=700)
    fx = pd.DataFrame({"embi": np.linspace(5.0, 5.5, 700)}, index=idx)
    fails = F.validate_master(fx, verbose=False)
    assert any("not a bps spread" in f for f in fails), fails


# ═════════════════════════════════════════════════════════════════════════════
# FRED helper
# ═════════════════════════════════════════════════════════════════════════════
def test_fetch_fred_csv_rejects_an_html_error_page(monkeypatch):
    """FRED answers an unknown series id with an HTML page, not a 404."""
    class FakeResponse:
        text = "<!DOCTYPE html><html><body>Not found</body></html>"

    monkeypatch.setattr(F, "get_with_retry", lambda *a, **k: FakeResponse())
    with pytest.raises(ValueError, match="does not exist"):
        F.fetch_fred_csv("NOT_A_SERIES")


def test_fetch_fred_csv_parses_a_csv_and_drops_missing(monkeypatch):
    class FakeResponse:
        text = "observation_date,DGS10\n2026-01-02,4.10\n2026-01-05,.\n2026-01-06,4.20\n"

    monkeypatch.setattr(F, "get_with_retry", lambda *a, **k: FakeResponse())
    s = F.fetch_fred_csv("DGS10")
    assert len(s) == 2                       # the "." row is dropped, not coerced to 0
    assert s.iloc[0] == pytest.approx(4.10)
    assert isinstance(s.index, pd.DatetimeIndex)


def test_sovereign_series_are_the_two_documented_ids():
    """Changing either id changes what the paper's sovereign section measures."""
    assert F.SOVEREIGN_SERIES == {"us10y": "DGS10",
                                  "lat_oas": "BAMLEMRLCRPILAOAS"}


# ═════════════════════════════════════════════════════════════════════════════
# The built dataset
# ═════════════════════════════════════════════════════════════════════════════
MASTER = F.PROC_DIR / "master_returns.csv"
needs_master = pytest.mark.skipif(not MASTER.exists(),
                                  reason="master not built — run python3 src/fetch.py")


@pytest.fixture(scope="module")
def master():
    return pd.read_csv(MASTER, index_col=0, parse_dates=True)


@needs_master
def test_embi_stops_at_its_last_published_observation(master):
    """IPEADATA discontinued the series in July 2024; nothing may appear after that."""
    e = master["embi"].dropna()
    assert e.index[-1] <= pd.Timestamp("2024-07-31")
    assert e.index[-1] < master.index[-1], "EMBI reaches the sample end — ffill is back"


@needs_master
def test_embi_does_not_end_in_a_run_of_identical_values(master):
    e = master["embi"].dropna()
    assert e.tail(60).nunique() > 1


@needs_master
def test_yield_differential_is_brazil_minus_us_in_bps(master):
    d = master[["yld_diff", "ntnf_yield", "us10y"]].dropna()
    expected = d["ntnf_yield"] * 10000 - d["us10y"]
    assert np.allclose(d["yld_diff"], expected)


@needs_master
def test_sovereign_series_are_in_basis_points(master):
    """A units slip here would silently rescale every number in the section."""
    oas = master["sov_oas"].dropna()
    assert 50 < oas.median() < 1500, oas.median()
    assert oas.index[0] >= pd.Timestamp("2023-08-01"), "OAS history starts Aug 2023"

    diff = master["yld_diff"].dropna()
    assert 0 < diff.median() < 3000, diff.median()


@needs_master
def test_the_oas_covers_fiscal24_and_embi_does_not(master):
    """This is the whole reason the series was added."""
    window = master.loc["2024-11-01":"2025-01-31"]
    assert window["sov_oas"].notna().all()
    assert window["embi"].isna().all()


@needs_master
def test_built_master_passes_validation(master):
    assert F.validate_master(master, verbose=False) == []
