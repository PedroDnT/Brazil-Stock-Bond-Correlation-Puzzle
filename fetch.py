"""
fetch.py — Data ingestion for the Brazilian stock-bond correlation study.

Sources:
  - BCB REST API    : CDI, Selic, IPCA, PTAX, EMBI proxy   (direct HTTP)
  - yfinance        : Ibovespa (^BVSP) + IMA ETF proxies
  - Tesouro CSV     : NTN-B, LTN, LFT daily PU → synthetic total-return series
"""

import warnings
warnings.filterwarnings("ignore")
import io, requests
import numpy as np
import pandas as pd
from pathlib import Path
import yfinance as yf

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
RAW_DIR  = BASE_DIR / "data" / "raw"
PROC_DIR = BASE_DIR / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

# ── Event maps (shared across all notebooks) ─────────────────────────────────
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
    "Current Cycle":       ("2023-01-01", "2026-03-13"),
}

# ── BCB REST API ─────────────────────────────────────────────────────────────
def _bcb_chunk(code, start_dt, end_dt):
    fmt = "%d/%m/%Y"
    url = (
        f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados"
        f"?formato=json"
        f"&dataInicial={start_dt.strftime(fmt)}"
        f"&dataFinal={end_dt.strftime(fmt)}"
    )
    try:
        r = requests.get(url, timeout=30)
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


def fetch_bcb_series(code, name, start="2004-01-01", end="2026-03-15", yrs=5):
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


def fetch_bcb(start="2004-01-01"):
    cache = RAW_DIR / "bcb_macro.parquet"
    if cache.exists():
        print("  BCB: loaded from cache")
        return pd.read_parquet(cache)
    print("Fetching BCB macro series...")
    BCB_SERIES = {
        "cdi":          11,
        "selic_target": 432,
        "ipca":         433,
        "ptax":         1,
        "embi":         21619,
        "igpm":         189,
    }
    series_list = []
    for name, code in BCB_SERIES.items():
        s = fetch_bcb_series(code, name, start=start)
        print(f"  {name}: {len(s)} obs")
        series_list.append(s)
    bcb = pd.concat(series_list, axis=1)
    bcb = bcb[~bcb.index.duplicated(keep="first")].sort_index()
    for col in ["ipca", "igpm", "selic_target"]:
        if col in bcb.columns:
            bcb[col] = bcb[col].ffill()
    bcb.to_parquet(cache)
    print(f"  BCB total: {len(bcb)} rows, {bcb.index[0].date()} to {bcb.index[-1].date()}")
    return bcb


# ── Ibovespa ─────────────────────────────────────────────────────────────────
def fetch_ibovespa(start="2004-01-01"):
    cache = RAW_DIR / "ibovespa.parquet"
    if cache.exists():
        print("  Ibovespa: loaded from cache")
        return pd.read_parquet(cache)["ibov"]
    print("Fetching Ibovespa (^BVSP)...")
    raw  = yf.download("^BVSP", start=start, progress=False, auto_adjust=True)
    # yfinance may return MultiIndex columns — flatten
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    ibov = raw["Close"].squeeze().rename("ibov").dropna()
    ibov.index = pd.DatetimeIndex(ibov.index).tz_localize(None)
    ibov.to_frame().to_parquet(cache)
    print(f"  Ibovespa: {len(ibov)} rows, {ibov.index[0].date()} to {ibov.index[-1].date()}")
    return ibov


# ── IMA ETF proxies (2019+) ──────────────────────────────────────────────────
IMA_ETFS = {
    "imab_etf":   "IMAB11.SA",
    "imab5p_etf": "IB5M11.SA",
    "irfm_etf":   "IRFM11.SA",
}

def fetch_ima_etfs(start="2019-01-01"):
    cache = RAW_DIR / "ima_etfs.parquet"
    if cache.exists():
        print("  IMA ETFs: loaded from cache")
        return pd.read_parquet(cache)
    print("Fetching IMA ETF proxies...")
    frames = {}
    for name, ticker in IMA_ETFS.items():
        raw = yf.download(ticker, start=start, progress=False, auto_adjust=True)
        if not raw.empty:
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            s = raw["Close"].squeeze().rename(name).dropna()
            s.index = pd.DatetimeIndex(s.index).tz_localize(None)
            frames[name] = s
            print(f"  {name} ({ticker}): {len(s)} rows from {s.index[0].date()}")
        else:
            print(f"  {name} ({ticker}): no data")
    etfs = pd.DataFrame(frames)
    etfs.to_parquet(cache)
    return etfs


# ── Tesouro Transparente ─────────────────────────────────────────────────────
TESOURO_URL = (
    "https://www.tesourotransparente.gov.br/ckan/dataset/"
    "df56aa42-484a-4a59-8184-7676580c81e3/resource/"
    "796d2059-14e9-44e3-80c9-2d9e30b405c1/download/"
    "precotaxatesourodireto.csv"
)

BOND_TYPE_MAP = {
    "Tesouro IPCA+":                          "ntnb",
    "Tesouro IPCA+ com Juros Semestrais":     "ntnb_coupon",
    "Tesouro Prefixado":                      "ltn",
    "Tesouro Prefixado com Juros Semestrais": "ntnf",
    "Tesouro Selic":                          "lft",
}

BENCHMARK_TARGETS = {
    "ntnb": 365 * 5,
    "ltn":  365 * 2,
    "ntnf": 365 * 10,
    "lft":  365 * 1,
}

def fetch_tesouro():
    cache = RAW_DIR / "tesouro_raw.parquet"
    if cache.exists():
        print("  Tesouro: loaded from cache")
        return pd.read_parquet(cache)
    print("Fetching Tesouro Transparente CSV...")
    r = requests.get(TESOURO_URL, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(
        io.StringIO(r.content.decode("latin-1")),
        sep=";", decimal=",",
        parse_dates=["Data Base", "Data Vencimento"],
        dayfirst=True,
    )
    df.columns = ["bond_type", "maturity", "date",
                  "buy_rate", "sell_rate", "buy_pu", "sell_pu", "base_pu"]
    df = df.dropna(subset=["date", "base_pu"])
    df["bond_class"] = df["bond_type"].map(BOND_TYPE_MAP)
    df.to_parquet(cache)
    print(f"  Tesouro: {len(df)} rows, {df['date'].min().date()} to {df['date'].max().date()}")
    return df


def build_synthetic_returns(tesouro, start="2004-01-01"):
    print("Building synthetic bond return series...")
    synth = {}
    for bond_class, target_days in BENCHMARK_TARGETS.items():
        sub = tesouro[tesouro["bond_class"] == bond_class].copy()
        sub = sub[sub["date"] >= pd.Timestamp(start)]
        if sub.empty:
            print(f"  {bond_class}: no data — skipping")
            continue
        sub["days_to_mat"] = (sub["maturity"] - sub["date"]).dt.days
        sub = sub[sub["days_to_mat"] > 180].copy()

        def pick_pu(grp):
            diff = (grp["days_to_mat"] - target_days).abs()
            return grp.loc[diff.idxmin(), "base_pu"]

        pu = (sub.groupby("date")
                 .apply(pick_pu, include_groups=False)
                 .rename(bond_class)
                 .sort_index())
        log_ret = np.log(pu / pu.shift(1))
        log_ret[log_ret.abs() > 0.03] = np.nan  # mask roll artifacts
        synth[bond_class] = log_ret
        print(f"  {bond_class}: {log_ret.dropna().shape[0]} obs, "
              f"{pu.index[0].date()} to {pu.index[-1].date()}")

    result = pd.DataFrame(synth).dropna(how="all")
    result.to_parquet(RAW_DIR / "synthetic_bond_returns.parquet")
    return result


def build_cdi_index(bcb, start="2004-01-01"):
    cdi_ann = bcb["cdi"].dropna() / 100
    cdi_ann = cdi_ann[cdi_ann.index >= start]
    cdi_index = ((1 + cdi_ann) ** (1/252)).cumprod()
    log_ret = np.log(cdi_index / cdi_index.shift(1)).dropna()
    log_ret.name = "lft_proxy"
    return log_ret


# ── Master assembler ─────────────────────────────────────────────────────────
def build_master_returns(start="2004-01-01", force_rebuild=False):
    """
    Assemble all series into a single aligned daily log-returns DataFrame.

    Return columns  : ibov, ntnb, ltn, ntnf, lft_proxy, ptax
    Level columns   : embi, cdi_level, selic, ipca, brl_usd
    Label columns   : crisis, regime
    ETF cross-check : imab_etf_ret, imab5p_etf_ret, irfm_etf_ret  (2019+)
    """
    cache = PROC_DIR / "master_returns.parquet"
    if cache.exists() and not force_rebuild:
        print("Master returns: loaded from cache")
        return pd.read_parquet(cache)

    print("\n" + "="*55)
    print("  Building master returns dataset")
    print("="*55)

    bcb      = fetch_bcb(start)
    ibov     = fetch_ibovespa(start)
    tesouro  = fetch_tesouro()
    etfs     = fetch_ima_etfs()

    ibov_ret = np.log(ibov / ibov.shift(1)).dropna().rename("ibov")
    bond_ret = build_synthetic_returns(tesouro, start)
    lft_ret  = build_cdi_index(bcb, start)
    ptax_ret = np.log(bcb["ptax"] / bcb["ptax"].shift(1)).dropna().rename("ptax")
    etf_ret  = np.log(etfs / etfs.shift(1)).dropna()
    etf_ret.columns = [c + "_ret" for c in etf_ret.columns]

    ret = pd.concat([
        ibov_ret,
        bond_ret[["ntnb", "ltn", "ntnf"]],
        lft_ret,
        ptax_ret,
    ], axis=1).dropna(how="all")

    levels = bcb[["embi", "cdi", "selic_target", "ipca", "ptax"]].copy()
    levels.columns = ["embi", "cdi_level", "selic", "ipca", "brl_usd"]

    master = (ret.join(levels, how="left")
                 .join(etf_ret, how="left"))
    master = master[master.index >= start].sort_index()

    master["crisis"] = "None"
    for name, (s, e) in CRISES.items():
        master.loc[(master.index >= s) & (master.index <= e), "crisis"] = name

    master["regime"] = "Unknown"
    for name, (s, e) in REGIMES.items():
        master.loc[(master.index >= s) & (master.index <= e), "regime"] = name

    master.to_parquet(cache)

    print(f"\nMaster dataset: {master.shape}")
    print(f"Dates : {master.index[0].date()} to {master.index[-1].date()}")
    print(f"Cols  : {master.columns.tolist()}")
    print(f"\nNaN counts:\n{master.isnull().sum().to_string()}")
    print(f"\nSaved → {cache}")
    return master


def load_master():
    cache = PROC_DIR / "master_returns.parquet"
    if not cache.exists():
        return build_master_returns()
    return pd.read_parquet(cache)

def get_crisis_windows(df): return {n:(df[(df.index>=s)&(df.index<=e)]) for n,(s,e) in CRISES.items()}
def get_regime_windows(df): return {n:(df[(df.index>=s)&(df.index<=e)]) for n,(s,e) in REGIMES.items()}

if __name__ == "__main__":
    master = build_master_returns(force_rebuild=True)
