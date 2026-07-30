"""
Unit tests for src/global_data.py — the constant-maturity bond return construction
used to build the matched cross-country panel.

The panel's whole purpose is comparability, so the construction has to be right for
every country including the awkward ones (Japan at ~0% yields, Germany briefly
negative). These tests pin the properties that matter.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import global_data as G  # noqa: E402


# ═════════════════════════════════════════════════════════════════════════════
# Duration and convexity
# ═════════════════════════════════════════════════════════════════════════════
def test_par_bond_duration_is_below_maturity():
    """A coupon bond's duration is always shorter than its maturity."""
    for y in (0.01, 0.03, 0.05, 0.10, 0.15):
        D, _ = G.par_bond_duration_convexity(np.array([y]), T=10.0)
        assert 0 < D[0] < 10.0


def test_duration_falls_as_yield_rises():
    ys = np.array([0.01, 0.03, 0.06, 0.12])
    D, _ = G.par_bond_duration_convexity(ys, T=10.0)
    assert np.all(np.diff(D) < 0)


def test_duration_approaches_maturity_as_yield_approaches_zero():
    """As y -> 0 a par bond's modified duration tends to its maturity."""
    D, _ = G.par_bond_duration_convexity(np.array([1e-5]), T=10.0)
    assert D[0] == pytest.approx(10.0, rel=1e-3)


def test_duration_matches_numerical_derivative():
    """D_mod = -(1/P) dP/dy, checked against a finite difference on the par bond."""
    y, T, h = 0.04, 10.0, 1e-6

    def price(yy, coupon):
        t = np.arange(1, int(T) + 1)
        return np.sum(coupon / (1 + yy) ** t) + 1.0 / (1 + yy) ** T

    c = y                                        # par bond: coupon == yield
    dP = (price(y + h, c) - price(y - h, c)) / (2 * h)
    D_num = -dP / price(y, c)
    D_an, _ = G.par_bond_duration_convexity(np.array([y]), T)
    assert D_an[0] == pytest.approx(D_num, rel=1e-4)


def test_convexity_matches_numerical_second_derivative():
    y, T, h = 0.04, 10.0, 1e-4

    def price(yy, coupon):
        t = np.arange(1, int(T) + 1)
        return np.sum(coupon / (1 + yy) ** t) + 1.0 / (1 + yy) ** T

    c = y
    d2P = (price(y + h, c) - 2 * price(y, c) + price(y - h, c)) / h ** 2
    C_num = d2P / price(y, c)
    _, C_an = G.par_bond_duration_convexity(np.array([y]), T)
    assert C_an[0] == pytest.approx(C_num, rel=1e-3)


# ═════════════════════════════════════════════════════════════════════════════
# Bond total return
# ═════════════════════════════════════════════════════════════════════════════
def _yields(vals):
    return pd.Series(vals, index=pd.date_range("2020-01-31", periods=len(vals), freq="ME"))


def test_flat_yields_give_carry_only():
    """With an unchanged yield the return is exactly the monthly carry."""
    r = G.constant_maturity_bond_return(_yields([4.0] * 12), T=10.0, periods_per_year=12)
    assert np.allclose(r.values, 0.04 / 12)


def test_rising_yields_produce_losses():
    r = G.constant_maturity_bond_return(_yields([2.0, 3.0, 4.0, 5.0]), T=10.0)
    assert (r < 0).all()


def test_falling_yields_produce_gains():
    r = G.constant_maturity_bond_return(_yields([5.0, 4.0, 3.0, 2.0]), T=10.0)
    assert (r > 0).all()


def test_longer_maturity_is_more_rate_sensitive():
    y = _yields([4.0, 5.0])
    r10 = G.constant_maturity_bond_return(y, T=10.0).iloc[0]
    r2 = G.constant_maturity_bond_return(y, T=2.0).iloc[0]
    assert r10 < r2 < 0


def test_return_magnitude_is_approximately_duration_times_yield_change():
    """A 100bp rise on a ~8y-duration 10y par bond costs roughly 8%."""
    r = G.constant_maturity_bond_return(_yields([4.0, 5.0]), T=10.0).iloc[0]
    D, _ = G.par_bond_duration_convexity(np.array([0.04]), 10.0)
    assert r == pytest.approx(0.04 / 12 - D[0] * 0.01, abs=0.005)


def test_handles_zero_and_negative_yields():
    """Japan and Germany both spent time at or below zero — must not blow up."""
    r = G.constant_maturity_bond_return(_yields([0.5, 0.0, -0.2, 0.1, 0.4]), T=10.0)
    assert len(r) == 4
    assert np.isfinite(r).all()
    assert r.abs().max() < 0.5          # no absurd magnitudes from the y -> 0 singularity


def test_convexity_makes_gains_exceed_losses_symmetrically():
    """Positive convexity: a 100bp fall gains more than a 100bp rise loses."""
    down = G.constant_maturity_bond_return(_yields([4.0, 3.0]), T=10.0).iloc[0]
    up = G.constant_maturity_bond_return(_yields([4.0, 5.0]), T=10.0).iloc[0]
    carry = 0.04 / 12
    assert (down - carry) > abs(up - carry)


# ═════════════════════════════════════════════════════════════════════════════
# Panel wiring
# ═════════════════════════════════════════════════════════════════════════════
def test_retry_helper_retries_5xx_then_succeeds(monkeypatch):
    """A transient 5xx must not abort a multi-minute rebuild."""
    import fetch as F

    calls = {"n": 0}

    class Resp:
        def __init__(self, code):
            self.status_code = code
            self.text = "ok"

        def raise_for_status(self):
            raise F.requests.HTTPError(f"{self.status_code}")

    def fake_get(url, timeout=None, **kw):
        calls["n"] += 1
        return Resp(503 if calls["n"] < 3 else 200)

    monkeypatch.setattr(F.requests, "get", fake_get)
    monkeypatch.setattr(F.time, "sleep", lambda s: None)
    r = F.get_with_retry("http://example.invalid", tries=5)
    assert r.status_code == 200
    assert calls["n"] == 3


def test_retry_helper_does_not_retry_4xx(monkeypatch):
    """A 404 is a real error — retrying wastes minutes and still fails."""
    import fetch as F

    calls = {"n": 0}

    class Resp:
        status_code = 404
        text = ""

        def raise_for_status(self):
            raise F.requests.HTTPError("404")

    def fake_get(url, timeout=None, **kw):
        calls["n"] += 1
        return Resp()

    monkeypatch.setattr(F.requests, "get", fake_get)
    monkeypatch.setattr(F.time, "sleep", lambda s: None)
    with pytest.raises(F.requests.HTTPError):
        F.get_with_retry("http://example.invalid", tries=5)
    assert calls["n"] == 1


def test_country_specs_are_well_formed():
    for cc, spec in G.COUNTRIES.items():
        assert len(cc) == 2
        assert spec["equity"].startswith("SPASTT01")
        assert spec["yield"].startswith("IRLTLT01")
        assert cc in spec["equity"] and cc in spec["yield"]


def test_brazil_is_in_all_countries_but_not_the_fred_set():
    """Brazil has no OECD long-term yield series; it comes from the domestic pipeline."""
    assert "BR" not in G.COUNTRIES
    assert "BR" in G.ALL_COUNTRIES


@pytest.mark.skipif(not (G.PROC_DIR / "global_panel.csv").exists(),
                    reason="panel not built — run scripts/run_global_analysis.py")
def test_built_panel_passes_validation():
    panel = pd.read_csv(G.PROC_DIR / "global_panel.csv", index_col=0, parse_dates=True)
    assert G.validate_panel(panel, verbose=False) == []
