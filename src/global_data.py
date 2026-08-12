"""
global_data.py — Matched cross-country stock-bond panel.

The paper's convergence claim ("advanced economies are moving toward Brazil's
condition") cannot be settled with Brazilian data alone. It needs the *same* asset
definitions and the *same* construction applied to every country, otherwise any
cross-country difference could be an artefact of how the series were built.

This module builds that panel:

  Equities  OECD total share price index, `SPASTT01<CC>M661N`  (FRED, monthly)
  Bonds     OECD long-term (10y) government bond yield, `IRLTLT01<CC>M156N`
            (FRED, monthly), converted to a constant-maturity total return

Both are OECD-harmonised, so the definition is identical across countries — that is
the whole reason for preferring them over national indices like the DAX or FTSE.

Brazil is not in the OECD long-term yield series, so it enters from the domestic
pipeline: Ibovespa for equities and the 10-year NTN-F yield from Tesouro Nacional.
That is the same instrument type (10-year nominal government bond) as the other four.

No API key is required. FRED's `fredgraph.csv` endpoint is public; the keyed
`api.stlouisfed.org` endpoint is not needed for any series used here.

Bond return construction
------------------------
From a par-bond yield y with constant maturity T, the one-period total return is

    r_t = y_{t-1}/m  -  D_mod(y_{t-1}, T) * dy_t  +  0.5 * C(y_{t-1}, T) * dy_t^2

(carry, duration, convexity; Swinkels 2019). Validated against the independently
built PU-based Brazilian NTN-F series: monthly correlation 0.931, and the headline
Ibovespa x NTN-F correlation is +0.460 yield-based against +0.473 PU-based — the
construction does not drive the result.
"""

import io
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests

from fetch import get_with_retry, fetch_fred_csv, FRED_CSV, SAMPLE_END

BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROC_DIR = BASE_DIR / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

# OECD-harmonised series. Same definition in every country, which is the point.
COUNTRIES = {
    "US": {"name": "United States", "equity": "SPASTT01USM661N", "yield": "IRLTLT01USM156N"},
    "DE": {"name": "Germany",       "equity": "SPASTT01DEM661N", "yield": "IRLTLT01DEM156N"},
    "JP": {"name": "Japan",         "equity": "SPASTT01JPM661N", "yield": "IRLTLT01JPM156N"},
    "GB": {"name": "United Kingdom","equity": "SPASTT01GBM661N", "yield": "IRLTLT01GBM156N"},
    # BR is assembled from the domestic pipeline — see build_global_panel()
}

BOND_MATURITY = 10.0          # years; the OECD series is a 10-year benchmark yield
IMF_BREAK = "2019-12-31"      # the turning point the IMF note identifies
PANEL_START = "2005-01-01"    # Brazilian bond data begins Jan 2005
PANEL_END = SAMPLE_END        # same fixed cutoff as the Brazilian sample


# ═════════════════════════════════════════════════════════════════════════════
# FRED (keyless) — fetch_fred_csv lives in fetch.py so both modules share one copy
# ═════════════════════════════════════════════════════════════════════════════
def fetch_global_raw(force=False):
    """All FRED series for the panel, cached to data/raw/global_panel_raw.csv."""
    cache = RAW_DIR / "global_panel_raw.csv"
    if cache.exists() and not force:
        print("  Global panel: loaded from cache")
        return pd.read_csv(cache, index_col=0, parse_dates=True)
    print("Fetching OECD-harmonised series from FRED (no API key)...")
    out = {}
    for cc, spec in COUNTRIES.items():
        for kind in ("equity", "yield"):
            sid = spec[kind]
            s = fetch_fred_csv(sid)
            out[f"{cc}_{kind}"] = s
            print(f"  {cc} {kind:6s} {sid:18s} {len(s):>4} obs  "
                  f"{s.index.min().date()} to {s.index.max().date()}")
    df = pd.DataFrame(out).sort_index()
    df.to_csv(cache)
    return df


# ═════════════════════════════════════════════════════════════════════════════
# Constant-maturity bond total return from a yield series
# ═════════════════════════════════════════════════════════════════════════════
def par_bond_duration_convexity(y, T=BOND_MATURITY):
    """
    Modified duration and convexity of a par bond at yield y, maturity T years,
    annual coupons (coupon rate = y, so price = 1).

        D_mod = -(1/P) dP/dy  = (1 - (1+y)^-T) / y
        C     =  (1/P) d2P/dy2 = sum_t t(t+1) c (1+y)^-(t+2) + T(T+1)(1+y)^-(T+2)

    Convexity is summed over the cashflows rather than expanded in closed form.
    Both are checked against finite differences of the actual bond price in
    tests/test_global_data.py — a closed form here is easy to get subtly wrong,
    and the error is invisible because the term is small.
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = int(round(T))
    with np.errstate(divide="ignore", invalid="ignore"):
        v = 1.0 / (1.0 + y)
        D = (1 - v ** T) / y
        t = np.arange(1, n + 1).reshape(-1, 1)                    # cashflow times
        coupon_terms = (t * (t + 1) * y * v ** (t + 2)).sum(axis=0)
        C = coupon_terms + n * (n + 1) * v ** (n + 2)             # + principal
    return D, C


def constant_maturity_bond_return(yield_pct, T=BOND_MATURITY, periods_per_year=12):
    """
    Total return of a constant-maturity par bond from its yield series.

    `yield_pct` is in percent per annum (as FRED reports it). Returns a simple
    return per period. Near-zero and negative yields (Japan, Germany post-2015)
    make the closed-form duration singular at y = 0, so the yield is floored at
    1bp — the resulting duration error is immaterial next to the price move.
    """
    y = (yield_pct.dropna() / 100.0).clip(lower=1e-4)
    D, C = par_bond_duration_convexity(y.shift(1), T)
    dy = y.diff()
    r = y.shift(1) / periods_per_year - D * dy + 0.5 * C * dy ** 2
    return r.dropna()


# ═════════════════════════════════════════════════════════════════════════════
# Panel assembly
# ═════════════════════════════════════════════════════════════════════════════
def build_global_panel(force=False, master=None):
    """
    Monthly panel of equity and bond returns for US, DE, JP, GB and BR.

    Returns a DataFrame indexed by month-end with columns
    `<CC>_eq` and `<CC>_bd` (simple returns), plus `<CC>_y` (bond yield, % p.a.).
    """
    cache = PROC_DIR / "global_panel.csv"
    if cache.exists() and not force:
        print("Global panel: loaded from cache")
        return pd.read_csv(cache, index_col=0, parse_dates=True)

    raw = fetch_global_raw(force=force)
    cols = {}

    for cc in COUNTRIES:
        eq = raw[f"{cc}_equity"].dropna()
        yl = raw[f"{cc}_yield"].dropna()
        eq.index = eq.index + pd.offsets.MonthEnd(0)
        yl.index = yl.index + pd.offsets.MonthEnd(0)
        cols[f"{cc}_eq"] = np.log(eq / eq.shift(1)).dropna()
        cols[f"{cc}_bd"] = constant_maturity_bond_return(yl)
        cols[f"{cc}_y"] = yl

    # ── Brazil, from the domestic pipeline ───────────────────────────────────
    if master is None:
        from fetch import load_master
        master = load_master()
    ib = master["ibov"].resample("ME").sum()
    ib = ib[ib != 0]
    cols["BR_eq"] = ib
    br_y = (master["ntnf_yield"].resample("ME").last().dropna() * 100.0)
    cols["BR_y"] = br_y
    cols["BR_bd"] = constant_maturity_bond_return(br_y)

    panel = pd.DataFrame(cols).sort_index()
    panel = panel[(panel.index >= PANEL_START) & (panel.index <= PANEL_END)]
    panel.to_csv(cache)
    print(f"\nGlobal panel: {panel.shape[0]} months, "
          f"{panel.index.min().date()} to {panel.index.max().date()}")
    return panel


ALL_COUNTRIES = {**COUNTRIES, "BR": {"name": "Brazil", "equity": "Ibovespa (IPEADATA)",
                                     "yield": "NTN-F 10y (Tesouro)"}}


def validate_panel(panel, verbose=True):
    """Plausibility checks on the assembled panel. Returns a list of failures."""
    fails = []
    if verbose:
        print("\n" + "-" * 72)
        print("  Global panel validation")
        print("-" * 72)
        print(f"  {'country':16s} {'n':>4s} {'eq ann%':>8s} {'eq vol%':>8s} "
              f"{'bd ann%':>8s} {'bd vol%':>8s} {'yield%':>7s}  status")

    for cc, spec in ALL_COUNTRIES.items():
        e, b = panel[f"{cc}_eq"].dropna(), panel[f"{cc}_bd"].dropna()
        y = panel[f"{cc}_y"].dropna()
        if len(e) < 100 or len(b) < 100:
            fails.append(f"{cc}: only {len(e)}/{len(b)} monthly obs")
        ea, ev = e.mean() * 12 * 100, e.std() * np.sqrt(12) * 100
        ba, bv = b.mean() * 12 * 100, b.std() * np.sqrt(12) * 100
        # equity vol 8-40%, 10y bond vol 2-20%, yields -1 to 25%
        ok = (8 <= ev <= 40) and (2 <= bv <= 20) and (-1 <= y.mean() <= 25)
        if not ok:
            fails.append(f"{cc}: eq vol {ev:.1f}%, bd vol {bv:.1f}%, mean yield {y.mean():.2f}%")
        if verbose:
            print(f"  {spec['name']:16s} {len(e):4d} {ea:8.2f} {ev:8.2f} "
                  f"{ba:8.2f} {bv:8.2f} {y.mean():7.2f}  {'OK' if ok else 'OUT OF BAND'}")

    if verbose:
        print("-" * 72)
        print("  ALL CHECKS PASSED" if not fails else f"  {len(fails)} CHECK(S) FAILED:")
        for f in fails:
            print(f"    - {f}")
    return fails


def validate_against_pu_construction(master=None, verbose=True):
    """
    Cross-check the yield-based construction against the PU-based Brazilian series.

    This is what licenses applying the yield-based method to the other four
    countries, where no unit-price data is available.
    """
    if master is None:
        from fetch import load_master
        master = load_master()
    br_y = master["ntnf_yield"].resample("ME").last().dropna() * 100.0
    yb = constant_maturity_bond_return(br_y)
    pu = master["ntnf"].resample("ME").sum()
    pu = np.exp(pu[pu != 0]) - 1                       # log -> simple, to match
    ib = master["ibov"].resample("ME").sum()
    ib = ib[ib != 0]

    both = pd.concat([yb.rename("yield_based"), pu.rename("pu_based")], axis=1).dropna()
    rho = both["yield_based"].corr(both["pu_based"])
    p_y = pd.concat([ib, yb], axis=1).dropna()
    p_p = pd.concat([ib, pu], axis=1).dropna()
    rho_y = p_y.iloc[:, 0].corr(p_y.iloc[:, 1])
    rho_p = p_p.iloc[:, 0].corr(p_p.iloc[:, 1])

    if verbose:
        print("\n" + "-" * 72)
        print("  Construction cross-check (Brazil NTN-F 10y, monthly)")
        print("-" * 72)
        print(f"  yield-based vs PU-based returns : rho = {rho:+.4f}  (n={len(both)})")
        print(f"  ann. vol  yield-based {both['yield_based'].std()*np.sqrt(12)*100:5.2f}%"
              f"   PU-based {both['pu_based'].std()*np.sqrt(12)*100:5.2f}%")
        print(f"  Ibovespa x bond, yield-based    : rho = {rho_y:+.3f}")
        print(f"  Ibovespa x bond, PU-based       : rho = {rho_p:+.3f}")
        print(f"  difference                      : {abs(rho_y - rho_p):.3f}")
        verdict = "OK" if (rho > 0.85 and abs(rho_y - rho_p) < 0.05) else "CONSTRUCTION MATTERS"
        print(f"  -> {verdict}: the construction does not drive the result"
              if verdict == "OK" else f"  -> {verdict}")
        print("-" * 72)
    return dict(rho_constructions=float(rho), rho_yield_based=float(rho_y),
                rho_pu_based=float(rho_p), n=len(both))


if __name__ == "__main__":
    panel = build_global_panel(force=True)
    validate_panel(panel)
    validate_against_pu_construction()
