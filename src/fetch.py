"""
fetch.py — Data ingestion for the Brazilian stock-bond correlation study.

Sources (all free, all public):
  - BCB SGS REST        : CDI, Selic target, IPCA, IGP-M, PTAX          (direct HTTP)
  - IPEADATA OData      : Ibovespa daily close, EMBI+ Brazil sovereign spread
  - Tesouro Transparente: NTN-B / LTN / NTN-F / LFT daily PU
                          -> constant-maturity total-return series
  - FRED (keyless CSV)  : US 10y Treasury, ICE BofA Latin America OAS
                          -> the two sovereign-risk series that outlive EMBI

Bond construction
-----------------
For each bond family we hold, on every date, the outstanding maturity closest to a
target tenor (5y NTN-B, 2y LTN, 10y NTN-F, 1y LFT).  The daily return is the return
of the bond *selected on the previous day*, so a change in the selected maturity
(a "roll") never injects a spurious price jump.  For coupon-bearing families the
coupon is added back on payment dates: the face/VNA is recovered from the bond's own
quoted yield via a standard du/252 cashflow pricer, so no external VNA series is
needed.  (Validated against NTN-F, whose face is known to be R$1,000: the pricer
recovers 1001.6, a 0.16% error.)
"""

import warnings
warnings.filterwarnings("ignore")

import io
import time
import numpy as np
import pandas as pd
import requests
from pathlib import Path


def get_with_retry(url, tries=5, timeout=180, backoff=3.0, **kwargs):
    """
    GET with exponential backoff on transient failures.

    IPEADATA and Tesouro Transparente both return sporadic 5xx under load. Without
    retries a single blip aborts the whole rebuild after several minutes of work,
    which is a poor trade for a request that usually succeeds on the next attempt.
    Client errors (4xx) are not retried — those are real and retrying won't help.
    """
    last = None
    for attempt in range(tries):
        # try/except/else, not try/except: raising the 4xx inside the `try` would be
        # caught by the handler below and retried, which is the opposite of intended.
        try:
            r = requests.get(url, timeout=timeout, **kwargs)
        except (requests.Timeout, requests.ConnectionError) as e:
            last = e
        else:
            if r.status_code < 400:
                return r
            if r.status_code < 500:
                r.raise_for_status()            # 4xx: a real error, don't retry
            last = requests.HTTPError(f"{r.status_code} for {url}", response=r)
        if attempt < tries - 1:
            wait = backoff * (2 ** attempt)
            print(f"    retry {attempt + 1}/{tries - 1} in {wait:.0f}s ({type(last).__name__})")
            time.sleep(wait)
    raise last


FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def fetch_fred_csv(series_id, timeout=90):
    """Fetch one FRED series via the public CSV endpoint. No API key required."""
    r = get_with_retry(FRED_CSV, timeout=timeout, params={"id": series_id})
    text = r.text
    if text.lstrip().startswith("<"):
        raise ValueError(f"FRED returned HTML for {series_id} — series does not exist")
    df = pd.read_csv(io.StringIO(text))
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    s = pd.to_numeric(df[df.columns[1]], errors="coerce")
    s.index = df[date_col]
    s.name = series_id
    return s.dropna()

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
RAW_DIR  = BASE_DIR / "data" / "raw"
PROC_DIR = BASE_DIR / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

START = "2004-01-01"
END   = "2026-07-31"          # how far to *request* from each API

# Hard cutoff for the analysis sample. Without this the dataset grows every day the
# sources publish, so the paper's numbers drift and "reproducible" is not true --
# a run next month would silently disagree with every figure in the text.
# June month-end also aligns the daily Brazilian data with the monthly FRED panel
# and removes a partial final month from every monthly resampling.
SAMPLE_END = "2026-06-30"

# ── Event maps (shared across all notebooks) ─────────────────────────────────
# NOTE: the usable sample starts 2005-01-03 (first Tesouro Transparente PU is
# 2004-12-31), so "Lula Boom" is 2005-2007 in-sample despite the 2003 label.
CRISES = {
    "GFC":         ("2008-09-01", "2009-03-31"),
    "Dilma":       ("2015-01-01", "2016-08-31"),
    "Joesley":     ("2017-05-17", "2017-05-31"),
    "COVID":       ("2020-02-20", "2020-06-30"),
    "Americanas":  ("2023-01-11", "2023-06-30"),
    "Fiscal24":    ("2024-11-01", "2025-01-31"),
}

REGIMES = {
    "Lula Boom":           ("2003-01-01", "2007-12-31"),
    "GFC & Recovery":      ("2008-01-01", "2012-12-31"),
    "Dilma Deterioration": ("2013-01-01", "2016-08-31"),
    "Reform Era":          ("2016-09-01", "2019-12-31"),
    "COVID & Post-COVID":  ("2020-01-01", "2022-12-31"),
    "Current Cycle":       ("2023-01-01", "2026-06-30"),
}

# ═════════════════════════════════════════════════════════════════════════════
# BCB SGS
# ═════════════════════════════════════════════════════════════════════════════
def _bcb_chunk(code, start_dt, end_dt):
    fmt = "%d/%m/%Y"
    url = (
        f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados"
        f"?formato=json"
        f"&dataInicial={start_dt.strftime(fmt)}"
        f"&dataFinal={end_dt.strftime(fmt)}"
    )
    try:
        r = get_with_retry(url, timeout=60)
        if r.status_code != 200 or not r.text.strip():
            return pd.Series(dtype=float)
        data = r.json()
        if not data:
            return pd.Series(dtype=float)
        df = pd.DataFrame(data)
        df["data"]  = pd.to_datetime(df["data"], dayfirst=True)
        df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
        return df.set_index("data")["valor"]
    except Exception:
        return pd.Series(dtype=float)


def fetch_bcb_series(code, name, start=START, end=END, yrs=5):
    s, e, chunks, cur = pd.Timestamp(start), pd.Timestamp(end), [], pd.Timestamp(start)
    while cur < e:
        nxt = min(cur + pd.DateOffset(years=yrs), e)
        c = _bcb_chunk(code, cur, nxt)
        if len(c):
            chunks.append(c)
        cur = nxt + pd.Timedelta(days=1)
    if not chunks:
        print(f"  WARNING: no data for {name} ({code})")
        return pd.Series(dtype=float, name=name)
    full = pd.concat(chunks)
    full = full[~full.index.duplicated(keep="first")].sort_index()
    full.name = name
    return full


# SGS 12 is the CDI rate expressed as **percent per day** (e.g. 0.041957 = 10.65% p.a.),
# not per annum.  Compounding it as if it were annual understates the accrual ~252x.
BCB_SERIES = {
    "cdi_daily_pct": 12,    # % per day
    "selic":         432,   # % p.a. (Copom target)
    "ipca":          433,   # % per month
    "brl_usd":       1,     # PTAX USD sell
    "igpm":          189,   # % per month
}


def fetch_bcb(force=False):
    cache = RAW_DIR / "bcb_macro.csv"
    if cache.exists() and not force:
        print("  BCB: loaded from cache")
        return pd.read_csv(cache, index_col=0, parse_dates=True)
    print("Fetching BCB SGS macro series...")
    series_list = []
    for name, code in BCB_SERIES.items():
        s = fetch_bcb_series(code, name)
        print(f"  {name:16s} ({code:>3}): {len(s):>5} obs")
        series_list.append(s)
    bcb = pd.concat(series_list, axis=1)
    bcb = bcb[~bcb.index.duplicated(keep="first")].sort_index()
    for col in ["ipca", "igpm", "selic"]:
        bcb[col] = bcb[col].ffill()
    bcb.to_csv(cache)
    print(f"  BCB total: {len(bcb)} rows, {bcb.index[0].date()} to {bcb.index[-1].date()}")
    return bcb


# ═════════════════════════════════════════════════════════════════════════════
# IPEADATA — Ibovespa and EMBI+ Brazil
# ═════════════════════════════════════════════════════════════════════════════
IPEA_SERIES = {
    "ibov": "GM366_IBVSP366",   # Ibovespa daily close, 1993+
    "embi": "JPM366_EMBI366",   # EMBI+ Brazil sovereign spread in bps, 1994 - Jul 2024
}


def _ipeadata(sercodigo):
    url = f"http://www.ipeadata.gov.br/api/odata4/ValoresSerie(SERCODIGO='{sercodigo}')"
    # IPEADATA is the flakiest of the three sources and 503s regularly under load,
    # so it gets a longer retry budget than the default.
    r = get_with_retry(url, tries=8, timeout=180, backoff=5.0)
    v = pd.DataFrame(r.json()["value"])
    # VALDATA carries a local UTC offset; take the date component directly to avoid
    # DST-transition errors on historical Brazilian dates.
    v["d"] = pd.to_datetime(v["VALDATA"].str.slice(0, 10))
    s = v.set_index("d")["VALVALOR"].astype(float).dropna().sort_index()
    return s[~s.index.duplicated(keep="last")]


# ═════════════════════════════════════════════════════════════════════════════
# Sovereign risk after EMBI
# ═════════════════════════════════════════════════════════════════════════════
# EMBI+ Brazil is the paper's measure of the sovereign-credit channel and IPEADATA
# discontinued it in July 2024 — its metadata catalogue lists exactly one
# sovereign-risk series and marks it INATIVA, so there is no drop-in replacement.
# Two free series carry the channel past that date, and neither is EMBI:
#
#   sov_oas   ICE BofA Latin America corporate/sovereign OAS. A genuine
#             option-adjusted hard-currency credit spread, so it measures the same
#             thing EMBI did, but it is regional rather than Brazil-specific and
#             its history starts 2023-08-14.
#   yld_diff  Brazil 10y (NTN-F) minus US 10y. Spans the whole sample but is NOT a
#             credit spread: it bundles credit with the currency and monetary-policy
#             differential. COVID shows why they are not interchangeable — EMBI
#             widened while this narrowed, because Brazil cut Selic to 2%.
#
# Both are reported; neither is presented as a continuation of EMBI.
SOVEREIGN_SERIES = {
    "us10y":   "DGS10",                 # US 10y constant maturity, % p.a.
    "lat_oas": "BAMLEMRLCRPILAOAS",     # ICE BofA Latin America OAS, %
}


def fetch_sovereign(force=False):
    cache = RAW_DIR / "sovereign.csv"
    if cache.exists() and not force:
        print("  Sovereign risk: loaded from cache")
        return pd.read_csv(cache, index_col=0, parse_dates=True)
    print("Fetching sovereign-risk series from FRED (no API key)...")
    out = {}
    for name, sid in SOVEREIGN_SERIES.items():
        s = fetch_fred_csv(sid)
        out[name] = s
        print(f"  {name:8s} ({sid}): {len(s):>5} obs  "
              f"{s.index.min().date()} to {s.index.max().date()}")
    df = pd.DataFrame(out).sort_index()
    df.to_csv(cache)
    return df


def fetch_ipeadata(force=False):
    cache = RAW_DIR / "ipeadata.csv"
    if cache.exists() and not force:
        print("  IPEADATA: loaded from cache")
        return pd.read_csv(cache, index_col=0, parse_dates=True)
    print("Fetching IPEADATA series...")
    out = {}
    for name, code in IPEA_SERIES.items():
        s = _ipeadata(code)
        out[name] = s
        print(f"  {name:6s} ({code}): {len(s):>5} obs  "
              f"{s.index.min().date()} to {s.index.max().date()}")
    df = pd.DataFrame(out).sort_index()
    df.to_csv(cache)
    return df


# ═════════════════════════════════════════════════════════════════════════════
# Tesouro Transparente — constant-maturity bond total returns
# ═════════════════════════════════════════════════════════════════════════════
TESOURO_URL = (
    "https://www.tesourotransparente.gov.br/ckan/dataset/"
    "df56aa42-484a-4a59-8184-7676580c81e3/resource/"
    "796d2059-14e9-44e3-80c9-2d9e30b405c1/download/PrecoTaxaTesouroDireto.csv"
)

# name: (Tesouro Direto title, target tenor in years, annual coupon rate)
BOND_SPECS = {
    "ntnb":     ("Tesouro IPCA+ com Juros Semestrais",      5.0,  0.06),
    "ltn":      ("Tesouro Prefixado",                       2.0,  0.00),
    "ntnf":     ("Tesouro Prefixado com Juros Semestrais",  10.0, 0.10),
    "lft":      ("Tesouro Selic",                           1.0,  0.00),
    "lft_long": ("Tesouro Selic",                           99.0, 0.00),  # longest outstanding
}


def fetch_tesouro(force=False):
    """Download the full Tesouro Direto price/yield history (~14 MB)."""
    cache = RAW_DIR / "tesouro_pu.csv"
    if cache.exists() and not force:
        print("  Tesouro: loaded from cache")
        df = pd.read_csv(cache, parse_dates=["base", "venc"])
        return df
    print("Fetching Tesouro Transparente PU history (~14 MB)...")
    r = get_with_retry(TESOURO_URL, timeout=600)
    df = pd.read_csv(io.BytesIO(r.content), sep=";", decimal=",", encoding="latin-1")
    df.columns = [c.strip() for c in df.columns]
    df["base"] = pd.to_datetime(df["Data Base"], dayfirst=True)
    df["venc"] = pd.to_datetime(df["Data Vencimento"], dayfirst=True)
    df["pu"]   = pd.to_numeric(df["PU Base Manha"], errors="coerce")
    df["y"]    = pd.to_numeric(df["Taxa Venda Manha"], errors="coerce") / 100.0
    df["ttm"]  = (df["venc"] - df["base"]).dt.days / 365.25
    df = df[["Tipo Titulo", "base", "venc", "pu", "y", "ttm"]].dropna(subset=["pu"])
    df.to_csv(cache, index=False)
    print(f"  Tesouro: {len(df):,} rows, {df['base'].min().date()} to {df['base'].max().date()}")
    return df


def _coupon_dates(maturity, after):
    """Semiannual coupon dates: maturity day/month, stepping back 6 months."""
    d, out = maturity, []
    while d > after:
        out.append(d)
        d -= pd.DateOffset(months=6)
    return sorted(out)


def _price_per_face(base, maturity, y, cpn_rate):
    """PU / VNA for a semiannual coupon bond on the Brazilian du/252 convention."""
    c = (1 + cpn_rate) ** 0.5 - 1
    cds = _coupon_dates(maturity, base)
    if not cds:
        return np.nan
    du = np.array([(d - base).days for d in cds]) * (252.0 / 365.0)
    dfac = (1 + y) ** (-du / 252.0)
    return float(c * dfac.sum() + dfac[-1])


def _build_one(df, name):
    """Constant-maturity total-return series for one bond family."""
    title, target, cpn = BOND_SPECS[name]
    s = df[(df["Tipo Titulo"] == title) & (df["ttm"] > 0.05)].copy()
    if s.empty:
        return pd.DataFrame()

    s["gap"] = (s["ttm"] - target).abs()
    sel = (s.loc[s.groupby("base")["gap"].idxmin(), ["base", "venc", "ttm", "y"]]
             .set_index("base").sort_index())
    pu  = s.pivot_table(index="base", columns="venc", values="pu", aggfunc="first").sort_index()
    yld = s.pivot_table(index="base", columns="venc", values="y",  aggfunc="first").sort_index()

    dates = pu.index
    # bond bought at t-1 and valued at t: rolls never create a price jump
    held  = sel["venc"].reindex(dates).ffill().shift(1)
    cpn_amt_factor = (1 + cpn) ** 0.5 - 1

    rows = []
    for i in range(1, len(dates)):
        t, tm1, b = dates[i], dates[i - 1], held.iloc[i]
        if pd.isna(b) or b not in pu.columns:
            continue
        p1, p0 = pu.at[t, b], pu.at[tm1, b]
        if not (np.isfinite(p1) and np.isfinite(p0)) or p0 <= 0:
            continue
        c_amt = 0.0
        if cpn > 0:
            paid = [d for d in _coupon_dates(b, tm1 - pd.Timedelta(days=1)) if tm1 < d <= t]
            if paid:
                yv = yld.at[tm1, b]
                if np.isfinite(yv):
                    ppf = _price_per_face(tm1, b, yv, cpn)
                    if np.isfinite(ppf) and ppf > 0:
                        c_amt = (p0 / ppf) * cpn_amt_factor * len(paid)
        rows.append((t,
                     np.log((p1 + c_amt) / p0),
                     sel["ttm"].get(tm1, np.nan),
                     sel["y"].get(tm1, np.nan),
                     float(held.iloc[i] != held.iloc[i - 1])))

    out = pd.DataFrame(rows, columns=["date", name, f"{name}_ttm",
                                      f"{name}_yield", f"{name}_roll"])
    return out.set_index("date")


def build_bond_returns(tesouro, force=False):
    cache = RAW_DIR / "bond_returns.csv"
    if cache.exists() and not force:
        print("  Bond returns: loaded from cache")
        return pd.read_csv(cache, index_col=0, parse_dates=True)
    print("Building constant-maturity bond total returns...")
    parts = []
    for name in BOND_SPECS:
        r = _build_one(tesouro, name)
        if r.empty:
            print(f"  {name}: NO DATA")
            continue
        print(f"  {name:9s} n={len(r):5d}  {r.index.min().date()}->{r.index.max().date()}  "
              f"ann.ret={r[name].mean()*252*100:6.2f}%  "
              f"ann.vol={r[name].std()*np.sqrt(252)*100:6.2f}%  "
              f"mean|ttm-target|={ (r[f'{name}_ttm']-BOND_SPECS[name][1]).abs().mean():.2f}y")
        parts.append(r)
    bonds = pd.concat(parts, axis=1).sort_index()
    bonds.to_csv(cache)
    return bonds


# ═════════════════════════════════════════════════════════════════════════════
# Master assembler
# ═════════════════════════════════════════════════════════════════════════════
def build_master_returns(start=START, force_rebuild=False):
    cache = PROC_DIR / "master_returns.csv"
    if cache.exists() and not force_rebuild:
        print("Master returns: loaded from cache")
        return pd.read_csv(cache, index_col=0, parse_dates=True)

    print("\n" + "=" * 60)
    print("  Building master returns dataset")
    print("=" * 60)

    bcb     = fetch_bcb(force=force_rebuild)
    ipea    = fetch_ipeadata(force=force_rebuild)
    tesouro = fetch_tesouro(force=force_rebuild)
    bonds   = build_bond_returns(tesouro, force=force_rebuild)
    sov     = fetch_sovereign(force=force_rebuild)

    ibov     = ipea["ibov"].dropna()
    ibov_ret = np.log(ibov / ibov.shift(1)).dropna().rename("ibov")
    ptax_ret = np.log(bcb["brl_usd"] / bcb["brl_usd"].shift(1)).dropna().rename("ptax")

    # CDI: SGS 12 is % per DAY. Daily log return = log(1 + v/100).
    cdi_daily = bcb["cdi_daily_pct"].dropna() / 100.0
    cdi_ret   = np.log1p(cdi_daily).rename("cdi_ret")
    # Annualised CDI level (% p.a.) — the correct risk-free rate for Sharpe ratios.
    cdi_level = (((1 + cdi_daily) ** 252 - 1) * 100).rename("cdi_level")

    ret = pd.concat([ibov_ret, bonds, cdi_ret, ptax_ret], axis=1)

    levels = pd.concat([
        ipea["embi"].rename("embi"),                   # bps, real EMBI+ Brazil
        cdi_level,
        bcb["selic"].rename("selic"),
        bcb["ipca"].rename("ipca"),
        bcb["brl_usd"].rename("brl_usd"),
        (sov["lat_oas"] * 100).rename("sov_oas"),      # bps, ICE BofA Latin America
        (sov["us10y"] * 100).rename("us10y"),          # bps
    ], axis=1)

    master = ret.join(levels, how="left")
    master = master[(master.index >= start) & (master.index <= SAMPLE_END)].sort_index()
    # keep only days on which the equity market and at least one bond traded
    master = master.dropna(subset=["ibov"], how="all")
    master[["cdi_level", "selic", "ipca", "brl_usd", "sov_oas", "us10y"]] = \
        master[["cdi_level", "selic", "ipca", "brl_usd", "sov_oas", "us10y"]].ffill()

    # EMBI is forward-filled only WITHIN its published span. Filling past the July
    # 2024 discontinuation froze it at 228 bps for 501 trading days — a flat line
    # across the whole Fiscal24 window, where a reader would take it for data.
    embi_end = master["embi"].last_valid_index()
    master["embi"] = master["embi"].ffill()
    if embi_end is not None:
        master.loc[master.index > embi_end, "embi"] = np.nan

    # Brazil 10y minus US 10y, in bps. Not a credit spread — see SOVEREIGN_SERIES.
    master["yld_diff"] = master["ntnf_yield"] * 10000 - master["us10y"]

    # "Calm", not "None": pandas.read_csv treats the literal string "None" as NaN by
    # default, which silently empties any `crisis == "None"` mask on reload.
    master["crisis"] = "Calm"
    for name, (s, e) in CRISES.items():
        master.loc[(master.index >= s) & (master.index <= e), "crisis"] = name

    master["regime"] = "Unknown"
    for name, (s, e) in REGIMES.items():
        master.loc[(master.index >= s) & (master.index <= e), "regime"] = name

    master.to_csv(cache)

    print(f"\nMaster dataset: {master.shape}")
    print(f"Dates : {master.index[0].date()} to {master.index[-1].date()}"
          f"  (sample fixed at {SAMPLE_END})")
    print(f"Saved -> {cache}")
    validate_master(master)
    return master


def load_master():
    cache = PROC_DIR / "master_returns.csv"
    if not cache.exists():
        return build_master_returns()
    return pd.read_csv(cache, index_col=0, parse_dates=True)


# ═════════════════════════════════════════════════════════════════════════════
# Validation
# ═════════════════════════════════════════════════════════════════════════════
# (annualised return %, annualised vol %) plausibility bands per series
_BANDS = {
    "ibov": ((0, 20), (18, 35)),
    "ntnb": ((5, 18), (3, 10)),
    "ltn":  ((5, 18), (1.5, 7)),
    "ntnf": ((5, 20), (6, 18)),
    "lft":  ((7, 16), (0.05, 1.5)),
}


def validate_master(master, verbose=True):
    """Range and coverage checks. Returns a list of failure strings (empty = OK)."""
    fails = []
    if verbose:
        print("\n" + "-" * 60)
        print("  Validation")
        print("-" * 60)
        print(f"  {'series':10s} {'n':>6s} {'ann.ret%':>9s} {'ann.vol%':>9s}   status")

    for col, ((r_lo, r_hi), (v_lo, v_hi)) in _BANDS.items():
        if col not in master.columns:
            fails.append(f"{col}: MISSING")
            continue
        s = master[col].dropna()
        if len(s) < 4000:
            fails.append(f"{col}: only {len(s)} obs (expected >4000)")
        ar = s.mean() * 252 * 100
        av = s.std() * np.sqrt(252) * 100
        ok = (r_lo <= ar <= r_hi) and (v_lo <= av <= v_hi)
        if not ok:
            fails.append(f"{col}: ann.ret={ar:.2f}% vol={av:.2f}% outside "
                         f"[{r_lo},{r_hi}]/[{v_lo},{v_hi}]")
        if verbose:
            print(f"  {col:10s} {len(s):6d} {ar:9.2f} {av:9.2f}   {'OK' if ok else 'OUT OF BAND'}")

    # CDI level must look like a Brazilian policy rate
    if "cdi_level" in master:
        c = master["cdi_level"].dropna()
        ok = 1.5 <= c.min() and c.max() <= 30
        if verbose:
            print(f"  {'cdi_level':10s} {len(c):6d} {'':9s} {'':9s}   "
                  f"range {c.min():.2f}-{c.max():.2f}% p.a. {'OK' if ok else 'OUT OF BAND'}")
        if not ok:
            fails.append(f"cdi_level range {c.min():.2f}-{c.max():.2f}% implausible")

    # EMBI must be a spread in bps, not an FX rate
    if "embi" in master:
        e = master["embi"].dropna()
        ok = len(e) > 3000 and e.median() > 100
        if verbose:
            print(f"  {'embi':10s} {len(e):6d} {'':9s} {'':9s}   "
                  f"median {e.median():.0f} bps, ends {e.index[-1].date()} "
                  f"{'OK' if ok else 'NOT A SPREAD'}")
        if not ok:
            fails.append(f"embi median {e.median():.2f} — not a bps spread")

        # The whole point of dropping the ffill: EMBI must not run to the sample end
        # with a frozen value. A long flat tail means the guard has regressed.
        tail = e.tail(60)
        if tail.nunique() <= 1:
            fails.append(f"embi ends in {len(tail)} identical values — stale ffill is back")

    # The two series that carry the sovereign channel past EMBI
    if "sov_oas" in master:
        o = master["sov_oas"].dropna()
        ok = len(o) > 200 and 50 < o.median() < 1500
        if verbose:
            print(f"  {'sov_oas':10s} {len(o):6d} {'':9s} {'':9s}   "
                  f"median {o.median():.0f} bps, from {o.index[0].date()} "
                  f"{'OK' if ok else 'OUT OF BAND'}")
        if not ok:
            fails.append(f"sov_oas median {o.median():.0f} bps implausible")

    if "yld_diff" in master:
        d = master["yld_diff"].dropna()
        ok = len(d) > 3000 and 0 < d.median() < 3000
        if verbose:
            print(f"  {'yld_diff':10s} {len(d):6d} {'':9s} {'':9s}   "
                  f"median {d.median():.0f} bps {'OK' if ok else 'OUT OF BAND'}")
        if not ok:
            fails.append(f"yld_diff median {d.median():.0f} bps implausible")

    # LFT must track CDI
    if {"lft", "cdi_ret"} <= set(master.columns):
        p = master[["lft", "cdi_ret"]].dropna()
        rho = p["lft"].corr(p["cdi_ret"])
        gap = (p["lft"].mean() - p["cdi_ret"].mean()) * 252 * 100
        ok = rho > 0.85 and abs(gap) < 2.0
        if verbose:
            print(f"  {'lft~cdi':10s} {len(p):6d} {'':9s} {'':9s}   "
                  f"rho={rho:.3f}, gap={gap:+.2f}pp {'OK' if ok else 'DRIFT'}")
        if not ok:
            fails.append(f"lft vs cdi: rho={rho:.3f} gap={gap:+.2f}pp")

    if verbose:
        print("-" * 60)
        print("  ALL CHECKS PASSED" if not fails else f"  {len(fails)} CHECK(S) FAILED:")
        for f in fails:
            print(f"    - {f}")
    return fails


if __name__ == "__main__":
    build_master_returns(force_rebuild=True)
