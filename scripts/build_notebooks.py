"""
build_notebooks.py
Generates all study notebooks as .ipynb files.
Run once: python3 build_notebooks.py
"""

import hashlib

import nbformat as nbf
from pathlib import Path

NB_DIR = Path(__file__).parent.parent / "notebooks"
NB_DIR.mkdir(exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────────────
def md(text):   return nbf.v4.new_markdown_cell(text.strip())
def code(text): return nbf.v4.new_code_cell(text.strip())
def nb(*cells): n = nbf.v4.new_notebook(); n.cells = list(cells); return n


def _stable_ids(notebook, name):
    """
    nbformat assigns a random id to every cell, so regenerating unchanged notebooks
    produced an 8-file diff of nothing but ids -- which buries real changes. Derive
    the id from the cell's own content instead, so the output is a pure function of
    the input.

    Deliberately NOT hashing the cell's position: that would re-id every cell below
    an insertion, which is the same whole-file diff in a different disguise. Exact
    duplicates get an occurrence counter, because nbformat does not fail on a
    duplicate id -- it warns and substitutes a random one, silently reinstating the
    churn.
    """
    seen = {}
    for cell in notebook.cells:
        key = f"{name}:{cell.cell_type}:{cell.source}"
        n = seen.get(key, 0)
        seen[key] = n + 1
        cell.id = hashlib.sha256(f"{key}:{n}".encode()).hexdigest()[:16]
    ids = [c.id for c in notebook.cells]
    assert len(set(ids)) == len(ids), f"duplicate cell id in {name}"
    return notebook


def save(notebook, name):
    path = NB_DIR / name
    # encoding is explicit: the sources carry rho, arrows, box-drawing and emoji, so
    # on a non-UTF-8 locale the write raises *after* open(..., "w") has already
    # truncated the tracked notebook, leaving it empty and unparseable.
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(_stable_ids(notebook, name), f)
    print(f"  Saved {name}")


# ═══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 01 — Data Pipeline & Validation
# ═══════════════════════════════════════════════════════════════════════════════
nb01 = nb(

md("""# 01 · Data Pipeline & Validation
**Brazilian Stock-Bond Correlation Study**

This notebook:
1. Builds (or loads from cache) the master returns dataset
2. Validates each series against known benchmarks
3. Produces a data availability / coverage heatmap
4. Cross-validates synthetic bond returns against ETF proxies (2019+)

> **Run once** — subsequent notebooks load from `data/processed/master_returns.csv`
"""),

# ── Setup ─────────────────────────────────────────────────────────────────────
code("""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "../src")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import seaborn as sns

from fetch import build_master_returns, load_master, CRISES, REGIMES

# ── Plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})

CRISIS_COLORS = {
    "GFC":         "#d62728",
    "Dilma":       "#ff7f0e",
    "Joesley":     "#9467bd",
    "COVID":       "#2ca02c",
    "Americanas":  "#8c564b",
    "Fiscal24":    "#e377c2",
}

ASSET_LABELS = {
    "ibov":      "Ibovespa",
    "ntnb":      "NTN-B 5y",
    "ltn":       "LTN 2y",
    "ntnf":      "NTN-F 10y",
    "lft": "LFT 1y (Tesouro Selic)",
    "ptax":      "BRL/USD",
}

def add_crisis_bands(ax, alpha=0.15):
    \"\"\"Shade crisis periods on a matplotlib axis.\"\"\"
    for name, (s, e) in CRISES.items():
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e),
                   color=CRISIS_COLORS[name], alpha=alpha, label=name)
"""),

md("""## 1. Build master dataset"""),

code("""
# Force rebuild = False → uses cache if available; set True to re-fetch
master = build_master_returns(force_rebuild=False)

print(f"Shape         : {master.shape}")
print(f"Date range    : {master.index[0].date()} → {master.index[-1].date()}")
print(f"\\nReturn columns: {[c for c in master.columns if c in ASSET_LABELS]}")
print(f"Level columns : {['embi','cdi_level','selic','ipca','brl_usd','sov_oas','yld_diff']}")
print(f"\\nFirst 3 rows:")
master.head(3)
"""),

md("""## 2. Data coverage heatmap

Check which series have data on each day — important for understanding sample sizes per analysis.
"""),

code("""
ret_cols = ["ibov", "ntnb", "ltn", "ntnf", "lft"]

# Monthly availability matrix (1 = data present, 0 = NaN)
avail = master[ret_cols].resample("ME").apply(lambda x: x.notna().mean())
avail.columns = [ASSET_LABELS[c] for c in avail.columns]

fig, ax = plt.subplots(figsize=(13, 4))
sns.heatmap(
    avail.T,
    cmap="YlGn", vmin=0, vmax=1,
    ax=ax, cbar_kws={"label": "% of days with data"},
    linewidths=0,
)
ax.set_title("Data availability by month and asset class", fontsize=13, pad=12)
ax.set_xlabel("")
ax.set_ylabel("")

# Mark crisis periods
for name, (s, e) in CRISES.items():
    s_idx = avail.index.searchsorted(pd.Timestamp(s))
    e_idx = avail.index.searchsorted(pd.Timestamp(e))
    ax.axvspan(s_idx, e_idx, color=CRISIS_COLORS[name], alpha=0.25)

plt.tight_layout()
plt.savefig("../outputs/fig_data_coverage.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: outputs/fig_data_coverage.png")
"""),

md("""## 3. Price-level chart: cumulative growth of all asset classes

This is the key context chart — it shows each asset's trajectory across all macro regimes.
"""),

code("""
# Rebuild cumulative return indices (base = 100 on first common date)
ret_cols = ["ibov", "ntnb", "ltn", "ntnf", "lft"]
sub = master[ret_cols].dropna(how="all")

# Start all series from the first date where ALL return cols have data
common_start = sub.dropna().index[0]
sub = sub[sub.index >= common_start].copy()

# Cumulative log return → price index
price_idx = np.exp(sub.cumsum()) * 100

fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                          gridspec_kw={"height_ratios": [2, 1]})

# ── Top: cumulative price indices ─────────────────────────────────────────────
ax = axes[0]
colors = ["#1f77b4", "#d62728", "#ff7f0e", "#2ca02c", "#9467bd"]
for i, col in enumerate(ret_cols):
    ax.plot(price_idx.index, price_idx[col], label=ASSET_LABELS[col],
            lw=1.5, color=colors[i])
add_crisis_bands(ax)
ax.set_yscale("log")
ax.set_ylabel("Cumulative return index\\n(log scale, base=100)", fontsize=10)
ax.set_title("Brazilian asset classes: cumulative total return (2005–2026)", fontsize=13)

# Add regime labels at top
for name, (s, e) in REGIMES.items():
    mid = pd.Timestamp(s) + (pd.Timestamp(e) - pd.Timestamp(s)) / 2
    if mid >= common_start:
        ax.text(mid, ax.get_ylim()[1] * 0.95, name,
                ha="center", va="top", fontsize=7.5, color="gray",
                rotation=0)

handles_assets = [plt.Line2D([0],[0], color=colors[i], lw=2,
                              label=ASSET_LABELS[col])
                  for i, col in enumerate(ret_cols)]
handles_crisis = [plt.Rectangle((0,0),1,1,
                                  fc=CRISIS_COLORS[n], alpha=0.4, label=n)
                  for n in CRISES]
ax.legend(handles=handles_assets + handles_crisis,
          loc="upper left", fontsize=8.5, ncol=2)

# ── Bottom: Ibovespa drawdown ──────────────────────────────────────────────────
ax2 = axes[1]
ibov_idx = price_idx["ibov"]
drawdown  = (ibov_idx / ibov_idx.cummax() - 1) * 100
ax2.fill_between(drawdown.index, drawdown, 0,
                 color="#1f77b4", alpha=0.4, label="Ibovespa drawdown")
add_crisis_bands(ax2, alpha=0.2)
ax2.set_ylabel("Drawdown (%)", fontsize=10)
ax2.set_xlabel("")
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax2.xaxis.set_major_locator(mdates.YearLocator(2))

plt.tight_layout()
plt.savefig("../outputs/fig_cumulative_returns.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: outputs/fig_cumulative_returns.png")
"""),

md("""## 4. Validate BCB macro series

Sanity check: EMBI should spike during crises, Selic should match known COPOM cycles.
"""),

code("""
fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

# EMBI
ax = axes[0]
ax.plot(master.index, master["embi"], color="#d62728", lw=1.2)  # ends Jul 2024
add_crisis_bands(ax)
ax.set_ylabel("EMBI+ Brazil (bps)", fontsize=10)
ax.set_title("Sovereign risk proxy (EMBI+)", fontsize=11)
ax.axhline(y=4.0, color="gray", ls="--", lw=0.8, alpha=0.6)

# Selic
ax = axes[1]
ax.plot(master.index, master["selic"], color="#1f77b4", lw=1.5, label="Selic target")
ax.plot(master.index, master["cdi_level"], color="#ff7f0e", lw=1, ls="--",
        alpha=0.7, label="CDI")
add_crisis_bands(ax)
ax.set_ylabel("Rate (% p.a.)", fontsize=10)
ax.set_title("Selic target rate and CDI", fontsize=11)
ax.legend(fontsize=9)

# BRL/USD
ax = axes[2]
ax.plot(master.index, master["brl_usd"], color="#2ca02c", lw=1.2)
add_crisis_bands(ax)
ax.set_ylabel("BRL / USD", fontsize=10)
ax.set_title("Exchange rate (PTAX)", fontsize=11)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator(2))

plt.suptitle("BCB macro series validation", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("../outputs/fig_macro_validation.png", dpi=150, bbox_inches="tight")
plt.show()
"""),

md("""## 5. Validate the constant-maturity construction

Two independent checks that the bond series are built correctly:

1. **LFT vs CDI.** The 1-year Tesouro Selic total return is built from observed PU,
   entirely independently of the CDI series. It must track compounded CDI closely —
   if it does not, the PU roll or the return calculation is wrong.
2. **Roll and coupon diagnostics.** Rolls (a change in the selected maturity) and
   coupon payment dates are the two places a constant-maturity construction can
   inject a spurious return. Neither should show up as an outlier.
"""),

code("""
# ── 1. LFT (observed PU) vs CDI (independent BCB series) ──────────────────────
chk = master[["lft", "cdi_ret"]].dropna()
lft_cum = np.exp(chk["lft"].cumsum())
cdi_cum = np.exp(chk["cdi_ret"].cumsum())

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

ax = axes[0]
ax.plot(lft_cum.index, lft_cum, label="LFT 1y (Tesouro Selic PU)", lw=1.8, color="#2ca02c")
ax.plot(cdi_cum.index, cdi_cum, label="Compounded CDI (BCB SGS 12)", lw=1.4,
        color="#1f77b4", ls="--")
add_crisis_bands(ax)
ax.set_yscale("log")
ax.set_ylabel("Growth of 1 unit (log scale)")
ax.set_title("LFT total return vs compounded CDI", fontsize=11)
ax.legend(fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# ── 2. The LFT desagio: where the two series come apart ───────────────────────
ax = axes[1]
gap = (lft_cum / cdi_cum - 1) * 100
ax.plot(gap.index, gap, lw=1.4, color="#d62728")
add_crisis_bands(ax)
ax.axhline(0, color="black", lw=0.8, ls="--")
ax.set_ylabel("LFT cumulative return minus CDI (%)")
ax.set_title("Tracking gap — widens when LFTs trade at a desagio", fontsize=11)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

plt.tight_layout()
plt.savefig("../outputs/fig_lft_cdi_crossval.png", dpi=150, bbox_inches="tight")
plt.show()

rho  = chk["lft"].corr(chk["cdi_ret"])
drift = (chk["lft"].mean() - chk["cdi_ret"].mean()) * 252 * 100
print(f"Correlation LFT vs CDI daily returns : {rho:.4f}")
print(f"Annualised return gap (LFT - CDI)    : {drift:+.2f} pp")
print(f"LFT cumulative x{lft_cum.iloc[-1]:.2f}  |  CDI cumulative x{cdi_cum.iloc[-1]:.2f}")
print("\\n(rho > 0.9 and |gap| < 1pp confirm the PU-based construction is sound)")
"""),

code("""
# ── Roll and coupon diagnostics ───────────────────────────────────────────────
# A roll (change of the selected maturity) or a coupon payment must not show up
# as an outlier: if it does, the same-bond return or the coupon add-back is wrong.
print("=== Mean |daily return| on roll days vs other days ===")
for col in ["ntnb", "ltn", "ntnf", "lft"]:
    roll_col = f"{col}_roll"
    if roll_col not in master.columns:
        continue
    r  = master[col].abs()
    rl = master[roll_col] == 1
    print(f"  {ASSET_LABELS.get(col, col):22s} roll days: {r[rl].mean()*100:.4f}%  "
          f"({rl.sum():>4} obs)   other days: {r[~rl].mean()*100:.4f}%  "
          f"ratio {r[rl].mean()/r[~rl].mean():.2f}x")

print("\\n=== Realised tenor of each constant-maturity series ===")
for col, target in [("ntnb", 5.0), ("ltn", 2.0), ("ntnf", 10.0), ("lft", 1.0)]:
    t = master[f"{col}_ttm"].dropna()
    print(f"  {ASSET_LABELS.get(col, col):22s} target {target:4.1f}y   "
          f"realised mean {t.mean():5.2f}y   "
          f"mean |gap| {np.abs(t-target).mean():.2f}y   "
          f"worst {np.abs(t-target).max():.2f}y")
"""),

md("""## 6. Summary statistics table

Key stats per asset across full sample — this becomes Table A1 in the whitepaper appendix.
"""),

code("""
from scipy import stats as scipy_stats

ret_cols = ["ibov", "ntnb", "ltn", "ntnf", "lft"]
df = master[ret_cols].dropna(how="all") * 100  # in percent

rows = []
for col in ret_cols:
    s = df[col].dropna()
    # Annualise assuming 252 trading days
    ann_ret = s.mean() * 252
    ann_vol = s.std() * np.sqrt(252)
    sharpe  = ann_ret / ann_vol  # no risk-free deduction (use CDI-adjusted later)

    # Max drawdown
    px = np.exp(s.cumsum() / 100)
    dd = (px / px.cummax() - 1).min() * 100

    rows.append({
        "Asset":        ASSET_LABELS[col],
        "Obs":          len(s),
        "Ann. Return%": round(ann_ret, 2),
        "Ann. Vol%":    round(ann_vol, 2),
        "Sharpe":       round(sharpe, 3),
        "Skewness":     round(float(scipy_stats.skew(s)), 3),
        "Kurtosis":     round(float(scipy_stats.kurtosis(s)), 3),
        "Max DD%":      round(dd, 2),
    })

summary = pd.DataFrame(rows).set_index("Asset")
print("=== Full-sample summary statistics (log returns, daily) ===")
print(summary.to_string())
summary.to_csv("../outputs/nb_tbl_summary_stats.csv")
print("\\nSaved: outputs/nb_tbl_summary_stats.csv")
"""),

md("""## 7. Data quality report

Identify gaps, extreme values, and suspicious observations to flag in the methodology section.
"""),

code("""
ret_cols = ["ibov", "ntnb", "ltn", "ntnf", "lft"]
df = master[ret_cols] * 100  # pct

print("=== Extreme daily moves (|return| > 5%) ===")
for col in ret_cols:
    s = df[col].dropna()
    extremes = s[s.abs() > 5].sort_values()
    if len(extremes):
        print(f"\\n{ASSET_LABELS[col]}:")
        for dt, val in extremes.items():
            print(f"  {dt.date()}  {val:+.2f}%")
    else:
        print(f"\\n{ASSET_LABELS[col]}: no extreme moves")

print("\\n=== NaN gaps by year ===")
nan_by_year = (master[ret_cols].isnull()
                .groupby(master.index.year).sum()
                .rename(columns=ASSET_LABELS))
print(nan_by_year[nan_by_year.sum(axis=1) > 0].to_string())
"""),

md("""## ✅ Notebook 01 complete

**What we have:**
- `data/processed/master_returns.csv` — ~5,600 rows, 2004–2026 (bonds from 2005-01-03)
- Asset return series: Ibovespa, NTN-B 5y, LTN 2y, NTN-F 10y, LFT 1y, LFT long, BRL/USD
- Macro levels: EMBI+ Brazil (bps), CDI (% p.a.), Selic, IPCA, BRL/USD
- Per-bond diagnostics: realised tenor, quoted yield, roll flag
- Event labels: `crisis`, `regime`

**What the validation actually showed** — read the printed output above rather than
this cell; the numbers are computed, not asserted. The checks that must pass:

| Check | Pass condition |
|-------|----------------|
| LFT vs compounded CDI | rho > 0.9, annualised gap < 1pp |
| Roll-day returns | not materially larger than non-roll days |
| Realised tenor | close to the 5y / 2y / 10y / 1y targets |
| EMBI level | a spread in basis points (median ~250), not an FX rate |
| CDI level | 2–20% p.a. — a plausible Brazilian policy rate |

`src/fetch.py::validate_master` runs the range and coverage checks automatically on
every rebuild and prints a pass/fail line per series.

**Next:** `02_descriptive.ipynb` — regime-split statistics and unconditional correlation matrices
"""),

) # end nb01

save(nb01, "01_data.ipynb")


# ═══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 02 — Descriptive Statistics & Correlation Matrices
# ═══════════════════════════════════════════════════════════════════════════════
nb02 = nb(

md("""# 02 · Descriptive Statistics & Correlation Matrices
**Brazilian Stock-Bond Correlation Study**

This notebook produces the foundational empirical evidence:
1. Regime-split summary statistics (Table 1 of whitepaper)
2. Unconditional full-sample correlation matrix
3. Correlation heatmaps by macro regime
4. Crisis-period asset returns heatmap (Table 2 of whitepaper)
5. Distribution analysis — skewness and fat tails
"""),

code("""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "../src")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats as scipy_stats

from fetch import load_master, CRISES, REGIMES

master = load_master()

# ── Helpers ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150, "figure.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "font.size": 11,
})

RET_COLS = ["ibov", "ntnb", "ltn", "ntnf", "lft"]
LABELS = {
    "ibov":      "Ibovespa",
    "ntnb":      "NTN-B 5y",
    "ltn":       "LTN 2y",
    "ntnf":      "NTN-F 10y",
    "lft": "LFT 1y",
}
CRISIS_COLORS = {
    "GFC":"#d62728","Dilma":"#ff7f0e","Joesley":"#9467bd",
    "COVID":"#2ca02c","Americanas":"#8c564b","Fiscal24":"#e377c2",
}

def regime_df(df, name):
    s, e = REGIMES[name]
    return df[(df.index >= s) & (df.index <= e)][RET_COLS].dropna(how="all")

def crisis_df(df, name):
    s, e = CRISES[name]
    return df[(df.index >= s) & (df.index <= e)][RET_COLS].dropna(how="all")
"""),

md("""## 1. Regime-split summary statistics

The most important context table: how do returns and volatility differ across Brazil's six macro regimes?
Positive mean returns for both stocks and bonds in the same period = "everyone wins" (monetary dominance).
Negative returns in both = "fiscal dominance / crisis."
"""),

code("""
def regime_stats(df, period_name, start, end):
    sub = df[(df.index >= start) & (df.index <= end)][RET_COLS].dropna(how="all") * 100
    rows = []
    for col in RET_COLS:
        s = sub[col].dropna()
        if len(s) < 20:
            continue
        ann_ret = s.mean() * 252
        ann_vol = s.std() * np.sqrt(252)
        # CDI-adjusted Sharpe
        cdi_ann = df[(df.index >= start) & (df.index <= end)]["cdi_level"].mean()
        sharpe  = (ann_ret - cdi_ann) / ann_vol if ann_vol > 0 else np.nan
        rows.append({
            "Regime":   period_name,
            "Asset":    LABELS[col],
            "Ann ret%": round(ann_ret, 1),
            "Ann vol%": round(ann_vol, 1),
            "Sharpe":   round(sharpe, 2),
            "Skew":     round(float(scipy_stats.skew(s)), 2),
            "Kurt":     round(float(scipy_stats.kurtosis(s)), 2),
            "N":        len(s),
        })
    return pd.DataFrame(rows)

all_stats = pd.concat([
    regime_stats(master, name, s, e)
    for name, (s, e) in REGIMES.items()
])

# Pivot for readability
pivot = all_stats.pivot_table(
    index="Asset", columns="Regime",
    values="Ann ret%", aggfunc="first"
)[list(REGIMES.keys())]
print("=== Annualised returns by regime (%) ===")
print(pivot.round(1).to_string())

pivot_vol = all_stats.pivot_table(
    index="Asset", columns="Regime",
    values="Ann vol%", aggfunc="first"
)[list(REGIMES.keys())]
print("\\n=== Annualised volatility by regime (%) ===")
print(pivot_vol.round(1).to_string())

all_stats.to_csv("../outputs/nb_tbl_regime_stats.csv", index=False)
print("\\nSaved: outputs/nb_tbl_regime_stats.csv")
"""),

md("""## 2. Unconditional full-sample correlation matrix

The baseline result. Are stocks and bonds negatively correlated (good diversification)
or positively correlated (Brazil's expected finding)?
"""),

code("""
df_ret = master[RET_COLS].dropna(how="all") * 100
corr_full = df_ret.corr(method="pearson")
corr_full.index   = [LABELS[c] for c in corr_full.index]
corr_full.columns = [LABELS[c] for c in corr_full.columns]

fig, ax = plt.subplots(figsize=(7, 6))
mask = np.triu(np.ones_like(corr_full, dtype=bool), k=1)
sns.heatmap(
    corr_full, mask=mask,
    annot=True, fmt=".3f", cmap="RdBu_r",
    vmin=-0.5, vmax=0.5, center=0,
    square=True, linewidths=0.5,
    cbar_kws={"shrink": 0.8, "label": "Pearson ρ"},
    ax=ax,
)
ax.set_title(
    f"Unconditional correlation matrix\\n"
    f"Full sample: 2005–2026  (N≈{len(df_ret):,} daily obs)",
    fontsize=12, pad=12,
)
plt.tight_layout()
plt.savefig("../outputs/fig_corr_full_sample.png", dpi=150, bbox_inches="tight")
plt.show()

# Highlight the key finding
ibov_ntnb = corr_full.loc["Ibovespa", "NTN-B 5y"]
ibov_ltn  = corr_full.loc["Ibovespa", "LTN 2y"]
ibov_lft  = corr_full.loc["Ibovespa", "LFT 1y"]
print(f"Key findings:")
print(f"  Ibovespa vs NTN-B  : ρ = {ibov_ntnb:+.3f}")
print(f"  Ibovespa vs LTN    : ρ = {ibov_ltn:+.3f}")
print(f"  Ibovespa vs LFT    : ρ = {ibov_lft:+.3f}")
print("  (Negative = diversification works; Positive = it doesn't)")
"""),

md("""## 3. Regime-split correlation heatmaps

**This is the core empirical contribution**: showing that correlations are not stable across regimes.
The Lula Boom / Reform Era should show lower (possibly negative) Ibovespa–bond correlation.
Fiscal dominance episodes should show strongly positive correlation.
"""),

code("""
regime_list = list(REGIMES.keys())
n_regimes   = len(regime_list)
ncols, nrows = 3, 2

fig, axes = plt.subplots(nrows, ncols, figsize=(15, 9))
axes = axes.flatten()

regime_corrs = {}
for i, (name, (s, e)) in enumerate(REGIMES.items()):
    sub = master[(master.index >= s) & (master.index <= e)][RET_COLS].dropna(how="all")
    if len(sub) < 30:
        axes[i].set_visible(False)
        continue

    corr = sub.corr()
    corr.index   = [LABELS[c] for c in corr.index]
    corr.columns = [LABELS[c] for c in corr.columns]
    regime_corrs[name] = corr

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, mask=mask,
        annot=True, fmt=".2f", cmap="RdBu_r",
        vmin=-0.6, vmax=0.6, center=0,
        square=True, linewidths=0.4,
        cbar=False, ax=axes[i], annot_kws={"size": 8},
    )
    n_obs = len(sub)
    axes[i].set_title(f"{name}\\n(N={n_obs:,})", fontsize=10, fontweight="bold")
    axes[i].tick_params(labelsize=8)

# Shared colorbar
sm = plt.cm.ScalarMappable(cmap="RdBu_r",
                            norm=plt.Normalize(vmin=-0.6, vmax=0.6))
sm.set_array([])
fig.colorbar(sm, ax=axes, shrink=0.5, label="Pearson ρ", pad=0.02)
fig.suptitle(
    "Stock-bond correlation by macro regime\\n"
    "Brazil 2004–2026",
    fontsize=14, fontweight="bold", y=1.01,
)
plt.tight_layout()
plt.savefig("../outputs/fig_corr_by_regime.png", dpi=150, bbox_inches="tight")
plt.show()

# Print the Ibovespa row across regimes
print("=== Ibovespa correlations across regimes ===")
for name, corr in regime_corrs.items():
    row = corr.loc["Ibovespa"].drop("Ibovespa")
    vals = "  |  ".join(f"{c}: {v:+.3f}" for c, v in row.items())
    print(f"{name:<25}  {vals}")
"""),

md("""## 4. Crisis-period asset returns heatmap

Shows cumulative returns during each crisis across all asset classes.
This is Table 2 of the whitepaper: the "triple whammy" evidence.
"""),

code("""
crisis_returns = {}
for name, (s, e) in CRISES.items():
    sub = master[(master.index >= s) & (master.index <= e)][RET_COLS]
    # Cumulative log return → simple total return
    cum = np.exp(sub.sum()) - 1
    crisis_returns[name] = cum * 100  # in %

cr_df = pd.DataFrame(crisis_returns).T
cr_df.columns = [LABELS[c] for c in cr_df.columns]

fig, ax = plt.subplots(figsize=(9, 5))
sns.heatmap(
    cr_df,
    annot=True, fmt=".1f",
    cmap="RdYlGn", center=0, vmin=-55, vmax=55,
    linewidths=0.6, linecolor="white",
    cbar_kws={"label": "Cumulative return (%)", "shrink": 0.7},
    ax=ax,
)
ax.set_title("Cumulative returns during crisis episodes (%)", fontsize=13, pad=12)
ax.set_xlabel("")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig("../outputs/fig_crisis_returns_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()

# Count: how many crises had BOTH Ibovespa AND NTN-B negative?
cr_raw = pd.DataFrame(crisis_returns).T
both_neg = ((cr_raw["ibov"] < 0) & (cr_raw["ntnb"] < 0)).sum()
print(f"\\nCrises with BOTH Ibovespa AND NTN-B negative: {both_neg}/{len(cr_raw)}")
print("(= number of 'triple whammy' episodes)")
print(cr_raw[["ibov","ntnb"]].round(1).to_string())
"""),

md("""## 5. Return distribution analysis

Tests normality and tail behaviour — motivation for using copulas and ES over Gaussian VaR.
"""),

code("""
from scipy.stats import jarque_bera, normaltest

fig, axes = plt.subplots(1, len(RET_COLS), figsize=(16, 4))

print("=== Normality tests (p < 0.05 = reject normality) ===")
for i, col in enumerate(RET_COLS):
    s = master[col].dropna() * 100

    # QQ plot vs normal
    (qt, ql) = scipy_stats.probplot(s, dist="norm")[:2]
    axes[i].scatter(qt[0], qt[1], s=2, alpha=0.4, color="#1f77b4")
    axes[i].plot(qt[0], qt[0] * ql[0] + ql[1], color="#d62728", lw=1.5)
    axes[i].set_title(LABELS[col], fontsize=9.5)
    axes[i].set_xlabel("Theoretical quantiles", fontsize=8)
    if i == 0: axes[i].set_ylabel("Sample quantiles", fontsize=8)

    jb_stat, jb_p = jarque_bera(s)
    print(f"  {LABELS[col]:<25} JB stat={jb_stat:8.1f}  p={jb_p:.2e}  "
          f"skew={scipy_stats.skew(s):+.3f}  "
          f"exc.kurt={scipy_stats.kurtosis(s):.2f}")

fig.suptitle("Q-Q plots vs normal distribution\\n"
             "(deviation in tails = fat tails / non-normality)", fontsize=12)
plt.tight_layout()
plt.savefig("../outputs/fig_qq_plots.png", dpi=150, bbox_inches="tight")
plt.show()
"""),

md("""## 6. Scatter matrix: pairwise return relationships

Visualises nonlinear dependencies and asymmetric tail behaviour — motivation for copulas.
"""),

code("""
df_plot = master[RET_COLS].dropna(how="all") * 100
df_plot.columns = [LABELS[c] for c in df_plot.columns]

# Colour by crisis vs non-crisis
colors_scatter = master["crisis"].map(
    lambda x: CRISIS_COLORS.get(x, "#aaaaaa55")
).reindex(df_plot.index)

pg = sns.PairGrid(df_plot, diag_sharey=False)
pg.map_diag(sns.histplot, bins=50, color="#1f77b4", alpha=0.7)
pg.map_lower(sns.scatterplot, s=2, alpha=0.3,
             hue=colors_scatter.values, palette=None, legend=False)
pg.map_upper(lambda x, y, **kw: None)  # skip upper triangle

for i in range(len(RET_COLS)):
    for j in range(len(RET_COLS)):
        if i < j:
            ax = pg.axes[i, j]
            xc = df_plot.columns[j]
            yc = df_plot.columns[i]
            r  = df_plot[xc].corr(df_plot[yc])
            ax.text(0.5, 0.5, f"ρ={r:+.3f}", transform=ax.transAxes,
                    ha="center", va="center", fontsize=11,
                    fontweight="bold",
                    color="#d62728" if r > 0 else "#2ca02c")
            ax.set_visible(True)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values(): spine.set_visible(False)

pg.figure.suptitle("Pairwise return scatter matrix\\n"
                   "(red dots = crisis periods, upper triangle = Pearson ρ)",
                   y=1.01, fontsize=12)
pg.figure.set_size_inches(12, 11)
plt.tight_layout()
plt.savefig("../outputs/fig_scatter_matrix.png", dpi=150, bbox_inches="tight")
plt.show()
"""),

md("""## ✅ Notebook 02 complete

**How to read these results.** The values are printed above; what follows is how to
interpret them without overclaiming.

| Result | Legitimate reading | Over-reading to avoid |
|--------|-------------------|----------------------|
| Full-sample ρ(Ibovespa, NTN-B) is positive but small | Brazilian bonds do **not hedge** equities: they never reliably rally when equities fall | "Diversification fails." A correlation near zero still reduces portfolio variance substantially — *not a hedge* is not the same as *not a diversifier* |
| Correlations vary across regimes | Motivates time-varying methods (notebook 04) | That the ranking is meaningful — notebook 03 §6 tests whether regimes actually differ, and mostly they cannot be distinguished |
| Crisis-window returns | Describes what happened in six specific episodes | That crises *cause* co-movement — notebook 03 §8 applies the Forbes-Rigobon correction, and most of the apparent spike is a volatility artefact |
| Jarque-Bera rejects normality | Gaussian VaR understates the tails; copulas and ES are appropriate | That Gaussian VaR is therefore always conservative — horizon scaling can swamp the tail understatement in the other direction (notebook 07) |

**Outputs saved:**
- `fig_corr_full_sample.png` — Figure 2 (whitepaper)
- `fig_corr_by_regime.png` — Figure 3 (whitepaper)
- `fig_crisis_returns_heatmap.png` — Table 2 (whitepaper)
- `tbl_regime_stats.csv` — Table 1 (whitepaper)

**Next:** `03_rolling_corr.ipynb` — time-varying rolling correlations + structural break tests
"""),

) # end nb02

save(nb02, "02_descriptive.ipynb")


# ═══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 03 — Rolling Correlations & Structural Breaks
# ═══════════════════════════════════════════════════════════════════════════════
nb03 = nb(

md("""# 03 · Rolling Correlations & Structural Breaks
**Brazilian Stock-Bond Correlation Study**

This notebook produces the **headline time-series chart** of the paper — the rolling
Ibovespa × bond correlation. This is the Brazilian equivalent of the IMF's Figure 1.

1. 252-day rolling Pearson and Spearman correlations
2. Conditional correlation: ρ given equity in bottom 10th percentile
3. CUSUM structural break test
4. Regime-average correlation summary
"""),

code("""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "../src")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats as scipy_stats
import statsmodels.api as sm

from fetch import load_master, CRISES, REGIMES

master = load_master()

plt.rcParams.update({
    "figure.dpi": 150, "figure.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "font.size": 11,
})
CRISIS_COLORS = {
    "GFC":"#d62728","Dilma":"#ff7f0e","Joesley":"#9467bd",
    "COVID":"#2ca02c","Americanas":"#8c564b","Fiscal24":"#e377c2",
}
LABELS = {
    "ibov":"Ibovespa", "ntnb":"NTN-B 5y",
    "ltn":"LTN 2y", "ntnf":"NTN-F 10y", "lft":"LFT 1y",
}

def add_crisis_bands(ax, alpha=0.15):
    for name, (s, e) in CRISES.items():
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e),
                   color=CRISIS_COLORS[name], alpha=alpha, label=name)
"""),

md("""## 1. The headline chart: 252-day rolling correlation

**This is Figure 4 of the whitepaper.**
It directly replicates and extends the IMF's approach, showing Brazil's
correlation dynamics over two decades.
"""),

code("""
WINDOW = 252  # 1 trading year
bond_cols = ["ntnb", "ltn", "ntnf", "lft"]
bond_colors = ["#d62728", "#ff7f0e", "#2ca02c", "#9467bd"]

df_ret = master[["ibov"] + bond_cols].dropna(how="all")

# Rolling Pearson correlations
roll_corr = pd.DataFrame({
    col: df_ret["ibov"].rolling(WINDOW).corr(df_ret[col])
    for col in bond_cols
})

fig, ax = plt.subplots(figsize=(14, 5.5))
for col, color in zip(bond_cols, bond_colors):
    ax.plot(roll_corr.index, roll_corr[col],
            label=LABELS[col], lw=1.5, color=color)

ax.axhline(y=0, color="black", lw=1.2, ls="--", alpha=0.7, label="ρ = 0")
add_crisis_bands(ax, alpha=0.13)

# Highlight the IMF's 2019 turning point for advanced economies
ax.axvline(pd.Timestamp("2020-01-01"), color="navy", lw=1.5, ls=":",
           alpha=0.8, label="IMF regime shift (DM, 2020)")

ax.set_ylim(-0.7, 0.8)
ax.set_ylabel(f"Rolling {WINDOW}-day Pearson ρ", fontsize=11)
ax.set_title(
    f"Ibovespa vs. Brazilian bond indices: {WINDOW}-day rolling correlation\\n"
    "Brazil, 2005–2026",
    fontsize=13,
)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator(2))

# Build legend (assets + crises)
asset_handles = [plt.Line2D([0],[0], color=c, lw=2, label=LABELS[col])
                 for col, c in zip(bond_cols, bond_colors)]
asset_handles.append(plt.Line2D([0],[0], color="black", lw=1.5,
                                 ls="--", label="ρ = 0"))
asset_handles.append(plt.Line2D([0],[0], color="navy", lw=1.5,
                                 ls=":", label="IMF regime shift (DM)"))
crisis_handles = [plt.Rectangle((0,0),1,1, fc=CRISIS_COLORS[n],
                                  alpha=0.4, label=n)
                  for n in CRISES]
ax.legend(handles=asset_handles + crisis_handles,
          loc="lower left", fontsize=8, ncol=3)

plt.tight_layout()
plt.savefig("../outputs/fig_rolling_correlation.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: outputs/fig_rolling_correlation.png")
"""),

md("""## 2. Tail conditional correlation — ρ given equity stress

The key question: **do bonds provide diversification when equity markets are crashing?**

We compute the correlation of bond returns conditional on equity returns being in the
bottom 10th percentile of observations — the "stress correlation."
"""),

code("""
RET_COLS_BONDS = ["ntnb", "ltn", "ntnf", "lft"]
df_ret = master[["ibov"] + RET_COLS_BONDS].dropna(how="all") * 100

q10 = df_ret["ibov"].quantile(0.10)
q25 = df_ret["ibov"].quantile(0.25)

stress_mask_10 = df_ret["ibov"] <= q10  # bottom 10%
stress_mask_25 = df_ret["ibov"] <= q25  # bottom 25%

results = []
for col in RET_COLS_BONDS:
    pair = df_ret[["ibov", col]].dropna()
    r_full    = pair["ibov"].corr(pair[col])
    r_stress10 = pair.loc[stress_mask_10, "ibov"].corr(pair.loc[stress_mask_10, col])
    r_stress25 = pair.loc[stress_mask_25, "ibov"].corr(pair.loc[stress_mask_25, col])
    results.append({
        "Bond": LABELS[col],
        "ρ full sample":    round(r_full, 3),
        "ρ | equity < Q10": round(r_stress10, 3),
        "ρ | equity < Q25": round(r_stress25, 3),
        "Δ (stress–full)":  round(r_stress10 - r_full, 3),
    })

cond_df = pd.DataFrame(results).set_index("Bond")
print("=== Conditional tail correlations ===")
print("(Positive Δ = correlations INCREASE during equity stress)")
print(cond_df.to_string())
cond_df.to_csv("../outputs/nb_tbl_conditional_correlations.csv")

# Bar chart
fig, ax = plt.subplots(figsize=(9, 4.5))
x  = np.arange(len(cond_df))
w  = 0.28
b1 = ax.bar(x - w, cond_df["ρ full sample"],    w, label="Full sample",     color="#1f77b4")
b2 = ax.bar(x,     cond_df["ρ | equity < Q25"], w, label="Equity < Q25 %",  color="#ff7f0e")
b3 = ax.bar(x + w, cond_df["ρ | equity < Q10"], w, label="Equity < Q10 %",  color="#d62728")
ax.axhline(0, color="black", lw=0.8, ls="--")
ax.set_xticks(x); ax.set_xticklabels(cond_df.index, fontsize=10)
ax.set_ylabel("Pearson ρ with Ibovespa")
ax.set_title("Conditional tail correlations: Ibovespa vs. bond classes\\n"
             "(Diversification works only if bars are negative during equity stress)",
             fontsize=11)
ax.legend(fontsize=9)
for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}", xy=(bar.get_x()+bar.get_width()/2, h),
                    xytext=(0, 3 if h >= 0 else -10),
                    textcoords="offset points", ha="center", fontsize=8)
plt.tight_layout()
plt.savefig("../outputs/fig_conditional_correlations.png", dpi=150, bbox_inches="tight")
plt.show()
"""),

md("""## 3. CUSUM structural break test

The CUSUM test on the rolling Ibovespa × NTN-B correlation identifies
statistically significant regime shifts. This is a simpler alternative
to Bai-Perron (which requires R via rpy2) and sufficient for whitepaper evidence.
"""),

code("""
from statsmodels.stats.diagnostic import breaks_cusumolsresid
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

# Run OLS of ibov on ntnb returns, then CUSUM on residuals
df_pair = master[["ibov", "ntnb"]].dropna() * 100
y = df_pair["ibov"].values
X = add_constant(df_pair["ntnb"].values)

ols_res = OLS(y, X).fit()

# breaks_cusumolsresid returns THREE values -- (statistic, p-value, critical values) --
# not a CUSUM path. The statistic is sup|W(t)| for the OLS-CUSUM process
#     W(t) = (1 / (sigma_hat * sqrt(n))) * sum_{i<=nt} e_i,
# so the path has to be built explicitly to plot it against its bands.
stat, pval, crit_vals = breaks_cusumolsresid(ols_res.resid)
crit = dict((lvl, c) for lvl, c in crit_vals)[5]        # 5% band for sup|W(t)|

resid = ols_res.resid
n     = len(resid)
sigma = np.sqrt(np.sum(resid**2) / (n - X.shape[1]))
cusum = np.cumsum(resid) / (sigma * np.sqrt(n))         # W(t), asymptotically a Brownian bridge
dates_cusum = df_pair.index

fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

# CUSUM path against its 5% boundary
ax = axes[0]
ax.plot(dates_cusum, cusum, color="#1f77b4", lw=1.5, label="OLS-CUSUM path W(t)")
ax.axhline( crit, color="#d62728", ls="--", lw=1.2, label=f"5% band (+/-{crit})")
ax.axhline(-crit, color="#d62728", ls="--", lw=1.2)
ax.axhline(0, color="black", lw=0.6)
for name, (s, e) in CRISES.items():
    ax.axvspan(pd.Timestamp(s), pd.Timestamp(e),
               color=CRISIS_COLORS[name], alpha=0.12)
ax.set_ylabel("CUSUM")
ax.set_title("CUSUM test: structural stability of Ibovespa ~ NTN-B OLS relationship",
             fontsize=12)
ax.legend(fontsize=9)

# Rolling 63-day correlation alongside
ax2 = axes[1]
rc63 = master["ibov"].rolling(63).corr(master["ntnb"])
ax2.plot(rc63.index, rc63, color="#ff7f0e", lw=1.2, alpha=0.8, label="63-day rolling ρ")
rc252 = master["ibov"].rolling(252).corr(master["ntnb"])
ax2.plot(rc252.index, rc252, color="#2ca02c", lw=1.8, label="252-day rolling ρ")
ax2.axhline(0, color="black", ls="--", lw=0.8)
for name, (s, e) in CRISES.items():
    ax2.axvspan(pd.Timestamp(s), pd.Timestamp(e),
                color=CRISIS_COLORS[name], alpha=0.12)
ax2.set_ylabel("Pearson ρ")
ax2.set_title("Rolling correlation: Ibovespa vs. NTN-B 5y", fontsize=12)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax2.xaxis.set_major_locator(mdates.YearLocator(2))
ax2.legend(fontsize=9)

plt.tight_layout()
plt.savefig("../outputs/fig_cusum_break_test.png", dpi=150, bbox_inches="tight")
plt.show()

exceeds = bool(np.max(np.abs(cusum)) > crit)
print("=== OLS-CUSUM test (Brown, Durbin & Evans 1975) ===")
print(f"  H0: the Ibovespa ~ NTN-B relationship is stable over the sample")
print(f"  sup|W(t)|      : {np.max(np.abs(cusum)):.3f}   (statsmodels: {stat:.3f})")
print(f"  p-value        : {pval:.3f}")
print(f"  5% band        : +/-{crit}")
print(f"  crosses band   : {exceeds}")
print()
if pval < 0.05:
    print("  -> Rejects stability: there is evidence of a structural break.")
else:
    print("  -> Does NOT reject stability at the 5% level.")
    print("     Note what this does and does not say. Failing to reject is not proof")
    print("     of a stable relationship, and it is not inconsistent with the regime")
    print("     variation in the correlations above: CUSUM has low power against slow")
    print("     drift, and this regression is dominated by equity variance because")
    print("     Ibovespa volatility is roughly 5x NTN-B volatility.")
"""),

md("""## 4. Spearman vs Pearson: is the relationship monotonic?

Divergence between Pearson and Spearman rolling correlations suggests
**nonlinear** dependence — motivating the copula analysis in notebook 05.
"""),

code("""
df_pair = master[["ibov", "ntnb"]].dropna()
W = 252

pearson_roll  = df_pair["ibov"].rolling(W).corr(df_pair["ntnb"])
# Spearman via rank transform
rank_ibov = df_pair["ibov"].rolling(W).rank()
rank_ntnb = df_pair["ntnb"].rolling(W).rank()
spearman_roll = rank_ibov.rolling(1).corr(rank_ntnb)   # already ranked, so Pearson=Spearman
# Cleaner: compute Spearman properly
def rolling_spearman(x, y, w):
    result = pd.Series(index=x.index, dtype=float)
    for i in range(w, len(x)):
        xi = x.iloc[i-w:i]
        yi = y.iloc[i-w:i]
        result.iloc[i] = scipy_stats.spearmanr(xi, yi)[0]
    return result

print("Computing rolling Spearman (this may take ~30s)...")
spearman_roll = rolling_spearman(df_pair["ibov"], df_pair["ntnb"], W)

fig, ax = plt.subplots(figsize=(13, 4.5))
ax.plot(pearson_roll.index,  pearson_roll,  lw=1.5, color="#1f77b4", label="Pearson ρ")
ax.plot(spearman_roll.index, spearman_roll, lw=1.5, color="#d62728",
        ls="--", alpha=0.8, label="Spearman ρ")
ax.axhline(0, color="black", lw=0.8, ls="--")
add_crisis_bands(ax, alpha=0.12)
ax.set_title(f"Pearson vs Spearman {W}-day rolling correlation: "
             f"Ibovespa vs NTN-B\\n"
             "Divergence = nonlinear dependence → motivates copula analysis",
             fontsize=11)
ax.set_ylabel("Correlation coefficient")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig("../outputs/fig_pearson_vs_spearman.png", dpi=150, bbox_inches="tight")
plt.show()

diff = (pearson_roll - spearman_roll).abs()
print(f"Mean |Pearson - Spearman|: {diff.mean():.4f}")
print(f"Max  |Pearson - Spearman|: {diff.max():.4f}")
print("(Large differences indicate nonlinear/asymmetric dependence)")
"""),

md("""## 5. Regime-average correlation table

Summary table for the whitepaper — replaces the IMF's cross-country comparison
with Brazil's regime comparison.
"""),

code("""
pairs = [("ibov","ntnb"), ("ibov","ltn"), ("ibov","ntnf"), ("ibov","lft")]
pair_labels = {
    ("ibov","ntnb"):      "Ibovespa × NTN-B",
    ("ibov","ltn"):       "Ibovespa × LTN",
    ("ibov","ntnf"):      "Ibovespa × NTN-F",
    ("ibov","lft"): "Ibovespa × LFT",
}

from metrics import corr_with_ci, bootstrap_corr_diff

rows = {}
for name, (s, e) in list(REGIMES.items()) + [("Full sample", ("2004-01-01","2026-12-31"))]:
    sub = master[(master.index >= s) & (master.index <= e)]
    row = {}
    for a, b in pairs:
        r = corr_with_ci(sub[a], sub[b])
        row[pair_labels[(a,b)]]        = round(r["rho"], 3) if np.isfinite(r["rho"]) else np.nan
        row[pair_labels[(a,b)] + " CI"] = (f"[{r['lo']:+.3f},{r['hi']:+.3f}]"
                                           if np.isfinite(r["rho"]) else "")
        row[pair_labels[(a,b)] + " sig"] = "*" if r["sig"] else ""
    row["n"] = len(sub[["ibov","ntnb"]].dropna())
    rows[name] = row

regime_corr_tbl = pd.DataFrame(rows).T
print("=== Regime correlations with 95% Fisher-z confidence intervals ===")
print("(* = interval excludes zero)\\n")
for name in regime_corr_tbl.index:
    r = regime_corr_tbl.loc[name]
    print(f"{name:<22} n={int(r['n']):>5}  " +
          "   ".join(f"{lbl.split(' × ')[1]}: {r[lbl]:+.3f} {r[lbl+' CI']}{r[lbl+' sig']}"
                     for lbl in pair_labels.values()))
regime_corr_tbl.to_csv("../outputs/nb_tbl_regime_correlations.csv")
print("\\nSaved: outputs/nb_tbl_regime_correlations.csv")
"""),

md("""## 6. Are the regime differences statistically distinguishable?

A table of point estimates ordered from low to high invites a story about regime
change. Before telling it, test whether the ordering survives sampling error.

Each regime holds 700–1,300 daily observations, so a correlation has a standard error
of roughly 0.03. Differences smaller than about 0.08 are not distinguishable from
noise, however clean the ranking looks. We use a stationary block bootstrap
(21-day blocks) so the test is not inflated by the serial dependence in daily returns.
"""),

code("""
base = "Lula Boom"          # the earliest, and on point estimates the calmest, regime
bs, be = REGIMES[base]
b_sub = master[(master.index >= bs) & (master.index <= be)][["ibov","ntnb"]].dropna()

rows = []
for name, (s, e) in REGIMES.items():
    if name == base:
        continue
    sub = master[(master.index >= s) & (master.index <= e)][["ibov","ntnb"]].dropna()
    t = bootstrap_corr_diff(sub["ibov"], sub["ntnb"],
                            b_sub["ibov"], b_sub["ntnb"], n_boot=1500, block=21)
    rows.append({"Regime": name,
                 f"rho - rho({base})": round(t["diff"], 3),
                 "bootstrap SE": round(t["boot_se"], 3),
                 "p-value": round(t["p"], 3),
                 "differs at 5%": "yes" if t["p"] < 0.05 else "no"})

diff_tbl = pd.DataFrame(rows).set_index("Regime")
print(f"=== Regime correlation vs {base}: block-bootstrap tests ===")
print(diff_tbl.to_string())
diff_tbl.to_csv("../outputs/nb_tbl_regime_difference_tests.csv")
print("\\nAny row reading 'no' means that regime's correlation is NOT statistically")
print("distinguishable from the baseline, and the narrative should not lean on it.")
"""),

md("""## 7. Return frequency: does the horizon change the answer?

Every result so far uses **daily** returns. The stock-bond correlation literature this
study is replicating — Campbell, Pflueger & Viceira (2020), Portelli & Roncalli (2024),
and the IMF note — works at monthly or quarterly horizons.

That choice is not cosmetic. Daily returns carry microstructure noise and
non-synchronous pricing: the Ibovespa close, the Tesouro PU reference price and the
CDI accrual are not struck at the same instant. Both effects attenuate correlation
toward zero. If the macro co-movement the paper is about lives at business-cycle
frequency, a daily estimate will systematically understate it.
"""),

code("""
rows = []
for lab, rule in [("daily", None), ("weekly", "W-FRI"), ("monthly", "ME"), ("quarterly", "QE")]:
    for a, b in pairs:
        x = master[[a, b]].dropna()
        if rule:
            x = x.resample(rule).sum()          # log returns aggregate by summing
            x = x[(x != 0).any(axis=1)]
        r = corr_with_ci(x[a], x[b])
        rows.append({"Frequency": lab, "Pair": pair_labels[(a,b)], "n": r["n"],
                     "rho": round(r["rho"], 3),
                     "95% CI": f"[{r['lo']:+.3f},{r['hi']:+.3f}]",
                     "sig": "*" if r["sig"] else ""})

freq_tbl = pd.DataFrame(rows)
wide = freq_tbl.pivot(index="Pair", columns="Frequency", values="rho")[
    ["daily","weekly","monthly","quarterly"]]
print("=== Ibovespa x bond correlation by return frequency ===")
print(wide.to_string())
print("\\n=== With confidence intervals ===")
for _, r in freq_tbl.iterrows():
    print(f"  {r['Frequency']:<10} {r['Pair']:<20} n={r['n']:>5}  "
          f"rho={r['rho']:+.3f}  {r['95% CI']}{r['sig']}")
freq_tbl.set_index(["Frequency","Pair"]).to_csv("../outputs/nb_tbl_frequency_robustness.csv")

fig, ax = plt.subplots(figsize=(9, 4.5))
for pair_lbl in wide.index:
    ax.plot(range(4), wide.loc[pair_lbl], marker="o", lw=1.8, label=pair_lbl)
ax.set_xticks(range(4)); ax.set_xticklabels(["daily","weekly","monthly","quarterly"])
ax.axhline(0, color="black", lw=0.8, ls="--")
ax.set_ylabel("Pearson rho with Ibovespa")
ax.set_title("Stock-bond correlation rises with the return horizon\\n"
             "Daily estimates understate the macro co-movement", fontsize=11)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("../outputs/fig_frequency_robustness.png", dpi=150, bbox_inches="tight")
plt.show()
"""),

md("""## 8. Forbes-Rigobon: is the crisis correlation surge real?

This is the test the crisis-correlation claim stands or falls on.

Forbes & Rigobon (2002) showed that a measured correlation is conditional on market
volatility: when the variance of the source market rises, the sample correlation is
**mechanically biased upward** even if the underlying propagation mechanism is
completely unchanged. Applying that correction, they found the apparent correlation
surges during the 1997 Asian crisis, the 1994 Mexican devaluation and the 1987 crash
largely disappeared — the markets were not more connected, only more volatile.

Brazilian crisis windows have equity variance 5–10x the calm level, so this correction
is not optional here. The adjusted estimate is

$$\\\\rho^* = \\\\frac{\\\\rho_c}{\\\\sqrt{1 + \\\\delta(1 - \\\\rho_c^2)}}, \\\\qquad
  \\\\delta = \\\\frac{\\\\sigma^2_{crisis}}{\\\\sigma^2_{calm}} - 1$$

**"Contagion"** means ρ* still exceeds the calm-period correlation. Otherwise what
looks like a correlation spike is only **interdependence** — the same relationship,
observed through a higher-variance lens.
"""),

code("""
from metrics import forbes_rigobon

calm = master[master["crisis"] == "Calm"]
assert len(calm) > 1000, "calm window empty — check the crisis label"

rows = []
for cname, (s, e) in CRISES.items():
    k = master[(master.index >= s) & (master.index <= e)]
    for col in ["ntnb", "ltn", "ntnf"]:
        o = forbes_rigobon(calm["ibov"], calm[col], k["ibov"], k[col])
        rows.append({"Crisis": cname, "Bond": LABELS[col], "n": o["n_crisis"],
                     "rho calm": round(o["rho_calm"], 3),
                     "rho crisis (raw)": round(o["rho_crisis"], 3),
                     "equity vol ratio": round(1 + o["delta"], 2),
                     "rho adjusted": round(o["rho_adj"], 3),
                     "verdict": {True: "contagion", False: "interdependence",
                                 None: "n/a"}[o["contagion"]]})

fr_tbl = pd.DataFrame(rows).set_index(["Crisis", "Bond"])
print("=== Forbes-Rigobon volatility-adjusted crisis correlations ===")
print(fr_tbl.to_string())
fr_tbl.to_csv("../outputs/nb_tbl_forbes_rigobon.csv")

sub = fr_tbl.xs("NTN-B 5y", level="Bond")
fig, ax = plt.subplots(figsize=(10, 4.5))
x = np.arange(len(sub)); w = 0.27
ax.bar(x - w, sub["rho calm"],         w, label="Calm period",   color="#1f77b4")
ax.bar(x,     sub["rho crisis (raw)"], w, label="Crisis (raw)",  color="#d62728")
ax.bar(x + w, sub["rho adjusted"],     w, label="Crisis (vol-adjusted)", color="#2ca02c")
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(sub.index, fontsize=9)
ax.set_ylabel("Pearson rho (Ibovespa x NTN-B)")
ax.set_title("Most of the crisis correlation spike is a volatility artefact\\n"
             "Green above blue = genuine contagion; green below = interdependence only",
             fontsize=11)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("../outputs/fig_forbes_rigobon.png", dpi=150, bbox_inches="tight")
plt.show()
"""),

md("""## ✅ Notebook 03 complete

The output above replaces four claims that the earlier version of this study asserted
without testing. Read the printed values rather than this cell, but note what each
section is now capable of falsifying:

| Section | What it tests | Why it matters |
|---------|---------------|----------------|
| 5 | Regime correlations **with confidence intervals** | A regime whose interval spans zero cannot be described as diversifying or not diversifying. |
| 6 | Whether regimes **differ from each other** | Ranking six point estimates is not evidence of regime change if the differences are inside the noise band. |
| 7 | Whether the answer **survives the return horizon** | Daily correlations are attenuated by microstructure noise; the literature this replicates uses monthly data. |
| 8 | Whether crisis correlation spikes **survive the volatility adjustment** | Forbes-Rigobon is cited in the methodology of this study; applying it is what makes the crisis claim testable rather than mechanical. |

**Outputs:** `fig_rolling_correlation.png`, `fig_conditional_correlations.png`,
`fig_frequency_robustness.png`, `fig_forbes_rigobon.png`,
`tbl_regime_correlations.csv`, `tbl_regime_difference_tests.csv`,
`tbl_frequency_robustness.csv`, `tbl_forbes_rigobon.csv`

**Next:** `04_dcc_garch.ipynb` — formal time-varying correlation via DCC-GARCH
"""),

) # end nb03

save(nb03, "03_rolling_corr.ipynb")


# ═══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 04 — DCC-GARCH: Time-Varying Correlation
# ═══════════════════════════════════════════════════════════════════════════════
nb04 = nb(

md("""# 04 · DCC-GARCH: Time-Varying Correlation
**Brazilian Stock-Bond Correlation Study**

Uses Engle (2002) two-stage DCC-GARCH to estimate the *daily* evolution of the
Ibovespa–bond correlation over 20 years.

1. Stage 1: univariate GARCH(1,1) per asset via `arch`
2. Stage 2: DCC parameters (a, b) via MLE on standardised residuals
3. Time-varying ρ_t chart — the academic complement to notebook 03's rolling window
4. DCC vs EMBI scatter: does sovereign risk drive correlation?
5. Crisis regime averages and persistence analysis
"""),

code("""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "../src")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from arch import arch_model
from scipy.optimize import minimize

from fetch import load_master, CRISES, REGIMES

master = load_master()

plt.rcParams.update({
    "figure.dpi": 150, "figure.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "font.size": 11,
})
CRISIS_COLORS = {
    "GFC":"#d62728","Dilma":"#ff7f0e","Joesley":"#9467bd",
    "COVID":"#2ca02c","Americanas":"#8c564b","Fiscal24":"#e377c2",
}
LABELS = {"ibov":"Ibovespa","ntnb":"NTN-B 5y","ltn":"LTN 2y",
          "ntnf":"NTN-F 10y","lft":"LFT 1y"}

def add_crisis_bands(ax, alpha=0.15):
    for name, (s, e) in CRISES.items():
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e),
                   color=CRISIS_COLORS[name], alpha=alpha, label=name)
"""),

md("""## 1. Stage 1: Fit univariate GARCH(1,1) per asset"""),

code("""
RET_COLS = ["ibov", "ntnb", "ltn", "ntnf", "lft"]

def fit_garch11(series, name=""):
    \"\"\"Fit GARCH(1,1) and return standardised residuals + conditional volatility.\"\"\"
    am  = arch_model(series, vol='GARCH', p=1, q=1, dist='normal', rescale=False)
    res = am.fit(disp='off', show_warning=False)
    std_resid = res.resid / res.conditional_volatility
    params = {k: float(v) for k, v in res.params.items()}
    print(f"  {name:<18}  omega={params.get('omega',0):.5f}  "
          f"alpha[1]={params.get('alpha[1]',0):.4f}  "
          f"beta[1]={params.get('beta[1]',0):.4f}  "
          f"persist={params.get('alpha[1]',0)+params.get('beta[1]',0):.4f}")
    return std_resid, res.conditional_volatility

print("=== GARCH(1,1) parameters (scaled returns × 100) ===")
std_resids = {}
cond_vols  = {}
for col in RET_COLS:
    s = master[col].dropna() * 100
    sr, cv = fit_garch11(s, LABELS[col])
    std_resids[col] = sr
    cond_vols[col]  = cv
"""),

code("""
# Plot conditional volatility for Ibovespa and NTN-B
fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)

for ax, col, color in zip(axes, ["ibov","ntnb"], ["#1f77b4","#d62728"]):
    cv = cond_vols[col]
    ax.fill_between(cv.index, cv * np.sqrt(252),
                    color=color, alpha=0.5, label=f"{LABELS[col]} ann. vol")
    add_crisis_bands(ax, alpha=0.12)
    ax.set_ylabel("Annualised volatility (%)")
    ax.set_title(f"GARCH(1,1) conditional volatility — {LABELS[col]}", fontsize=11)
    ax.legend(fontsize=9)

axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
axes[1].xaxis.set_major_locator(mdates.YearLocator(2))
plt.tight_layout()
plt.savefig("../outputs/fig_garch_volatility.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: outputs/fig_garch_volatility.png")
"""),

md("""## 2. Stage 2: DCC parameter estimation

Estimate the DCC parameters (a, b) by maximum likelihood on the standardised
residuals, **with standard errors from the numerical Hessian**.

`a` governs how strongly the conditional correlation responds to news. Reporting
`a` without a standard error makes "correlations are time-varying" an assertion;
with one it is a testable hypothesis (H0: a = 0). The estimator lives in
`src/metrics.py::fit_dcc` and is unit-tested against simulated DCC processes with
known parameters, including the degenerate constant-correlation case.
"""),

code("""
from metrics import fit_dcc

print("=== DCC-GARCH(1,1) parameter estimates ===")
print(f"  {'pair':<28} {'a':>8} {'SE(a)':>8} {'t(a)':>7} {'b':>8} "
      f"{'persist':>8} {'mean rho':>9} {'sd rho':>7}")

dcc_results = {}
pairs = [("ibov","ntnb"), ("ibov","ltn"), ("ibov","ntnf"), ("ibov","lft")]
for ca, cb in pairs:
    f = fit_dcc(std_resids[ca], std_resids[cb])
    dcc_results[(ca, cb)] = f
    flag = ("  <- not identified" if not f["identified"]
            else "  <- a>0 at 5%" if f["t_a"] > 1.96 else "")
    print(f"  {'Ibovespa x ' + LABELS[cb]:<28} {f['a']:>8.4f} {f['se_a']:>8.4f} "
          f"{f['t_a']:>7.2f} {f['b']:>8.4f} {f['persistence']:>8.4f} "
          f"{f['rho'].mean():>9.3f} {f['rho'].std():>7.3f}{flag}")

print("\\nH0: a = 0 (constant conditional correlation). |t| > 1.96 rejects at 5%.")
print("A pair that fails to reject is NOT evidence of crisis correlation spikes,")
print("however suggestive the plotted path looks.")
"""),

md("""## 3. The DCC correlation chart — Figure 6

Time-varying correlation ρ_t from DCC-GARCH. This is the **formal econometric**
complement to the rolling window chart in notebook 03.
"""),

code("""
bond_cols  = ["ntnb", "ltn", "ntnf", "lft"]
colors     = ["#d62728","#ff7f0e","#2ca02c","#9467bd"]

fig, ax = plt.subplots(figsize=(14, 5.5))

for col, color in zip(bond_cols, colors):
    rho = dcc_results[("ibov", col)]["rho"]
    ax.plot(rho.index, rho, label=LABELS[col], lw=1.4, color=color, alpha=0.85)

ax.axhline(0, color="black", lw=1.2, ls="--", alpha=0.7, label="ρ = 0")
add_crisis_bands(ax, alpha=0.13)
ax.axvline(pd.Timestamp("2020-01-01"), color="navy",
           lw=1.5, ls=":", alpha=0.8, label="IMF DM regime shift")

ax.set_ylim(-0.3, 0.5)
ax.set_ylabel("DCC-GARCH daily conditional correlation ρ_t", fontsize=11)
ax.set_title(
    "DCC-GARCH(1,1): Ibovespa vs. Brazilian bond indices — daily conditional correlation\\n"
    "Brazil, 2004–2026  (Engle 2002)",
    fontsize=13,
)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator(2))

asset_handles  = [plt.Line2D([0],[0], color=c, lw=2, label=LABELS[col])
                  for col,c in zip(bond_cols, colors)]
asset_handles += [plt.Line2D([0],[0], color="black", ls="--", lw=1.5, label="ρ=0"),
                  plt.Line2D([0],[0], color="navy",  ls=":",  lw=1.5, label="IMF DM shift")]
crisis_handles = [plt.Rectangle((0,0),1,1, fc=CRISIS_COLORS[n], alpha=0.4, label=n)
                  for n in CRISES]
ax.legend(handles=asset_handles + crisis_handles,
          loc="lower left", fontsize=8, ncol=3)

plt.tight_layout()
plt.savefig("../outputs/fig_dcc_correlation.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: outputs/fig_dcc_correlation.png")
"""),

md("""## 4. DCC correlation vs. EMBI sovereign risk"""),

code("""
rho_ntnb = dcc_results[("ibov","ntnb")]["rho"]
# No ffill: EMBI is already forward-filled within its published span by fetch.py and
# stops at 2024-07-30. Filling here would put the old frozen tail straight back into
# the figure, drawing a flat line across Fiscal24 that looks like data.
embi_aligned = master["embi"].reindex(rho_ntnb.index)
oas_aligned  = master["sov_oas"].reindex(rho_ntnb.index)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# Time series overlay
ax = axes[0]
ax2 = ax.twinx()
ax.plot(rho_ntnb.index, rho_ntnb, color="#d62728", lw=1.3, label="DCC ρ_t (left)")
ax2.plot(embi_aligned.index, embi_aligned, color="#1f77b4",
         lw=1, alpha=0.6, label="EMBI+ bps (right)")
ax2.plot(oas_aligned.index, oas_aligned, color="#2ca02c",
         lw=1, alpha=0.7, label="LatAm OAS bps (right)")
add_crisis_bands(ax, alpha=0.1)
ax.set_ylabel("DCC ρ_t (Ibovespa × NTN-B)", color="#d62728", fontsize=10)
ax2.set_ylabel("Sovereign spread (bps)", fontsize=10)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.set_title("DCC correlation vs. sovereign risk (EMBI ends Jul 2024)", fontsize=11)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

# Scatter
ax3 = axes[1]
df_scatter = pd.DataFrame({"rho": rho_ntnb, "embi": embi_aligned}).dropna()
crisis_label = master["crisis"].reindex(df_scatter.index).fillna("Calm")
for cname, group in df_scatter.groupby(crisis_label):
    color = CRISIS_COLORS.get(cname, "#aaaaaa")
    alpha = 0.7 if cname != "Calm" else 0.15
    size  = 12  if cname != "Calm" else 3
    ax3.scatter(group["embi"], group["rho"], s=size,
                color=color, alpha=alpha,
                label=cname if cname != "Calm" else None)

# OLS trend line
from numpy.polynomial import polynomial as P
x = df_scatter["embi"].values
y = df_scatter["rho"].values
coeffs = np.polyfit(x, y, 1)
xline  = np.linspace(x.min(), x.max(), 100)
ax3.plot(xline, np.polyval(coeffs, xline), "k--", lw=1.5)
r2 = np.corrcoef(x, y)[0,1]**2
ax3.set_xlabel("EMBI+ Brazil (bps)", fontsize=10)
ax3.set_ylabel("DCC ρ_t", fontsize=10)
ax3.set_title(f"Scatter: DCC ρ vs. EMBI  (R²={r2:.3f})", fontsize=11)
ax3.legend(fontsize=8)

plt.tight_layout()
plt.savefig("../outputs/fig_dcc_vs_embi.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"EMBI → DCC correlation R² = {r2:.3f}")
print(f"(EMBI span only: {df_scatter.index.min().date()} to {df_scatter.index.max().date()})")
"""),

md("""## 5. Crisis-period DCC correlation summary table"""),

code("""
rows = []
for cname, (s, e) in list(CRISES.items()) + [("Full sample", ("2004-01-01","2026-12-31"))]:
    row = {"Period": cname}
    for ca, cb in [("ibov","ntnb"),("ibov","ltn"),("ibov","ntnf"),("ibov","lft")]:
        rho = dcc_results[(ca,cb)]["rho"]
        mask = (rho.index >= s) & (rho.index <= e)
        row[f"Ibov x {LABELS[cb]}"] = round(rho[mask].mean(), 3) if mask.sum() else np.nan
    rows.append(row)

dcc_tbl = pd.DataFrame(rows).set_index("Period")
print("=== DCC-GARCH average conditional correlation by period ===")
print(dcc_tbl.to_string())
dcc_tbl.to_csv("../outputs/nb_tbl_dcc_crisis_correlations.csv")

# A crisis average above the full-sample average is only suggestive: the DCC path is
# itself estimated from returns whose variance explodes in a crisis. Notebook 03's
# Forbes-Rigobon table is the test of whether that elevation survives the volatility
# adjustment.
fs = dcc_tbl.loc["Full sample"]
print("\\n=== Crisis elevation relative to the full sample ===")
for cname in CRISES:
    r = dcc_tbl.loc[cname]
    parts = "  ".join(f"{c.split(' x ')[1]}: {r[c]/fs[c]:.2f}x" for c in dcc_tbl.columns
                      if np.isfinite(r[c]) and abs(fs[c]) > 1e-6)
    print(f"  {cname:12s} {parts}")
print("\\nSaved: outputs/nb_tbl_dcc_crisis_correlations.csv")
"""),

md("""## ✅ Notebook 04 complete

**What to read off the output above** (values are computed, not asserted here):

| Question | Where to look |
|----------|---------------|
| Are correlations genuinely time-varying? | `t(a)` in the stage-2 table. \\|t\\| > 1.96 rejects a = 0. |
| How persistent? | `persist` = a + b. Near 1 means shocks to correlation decay slowly. |
| Do correlations spike in crises? | The elevation table — but see the caveat below. |
| Is the LFT pair meaningful? | It is flagged *not identified*: the LFT return series is
  near-deterministic, so its conditional correlation path is degenerate by construction
  and its DCC row should not be interpreted. |

**Caveat on crisis spikes.** A DCC path that rises in a crisis is not by itself evidence
that the propagation mechanism strengthened. The conditional correlation is estimated
from returns whose variance rises several-fold in the same window, and Forbes & Rigobon
(2002) show that this alone biases measured correlation upward. Notebook 03 reports the
volatility-adjusted comparison; treat that as the test and this table as the description.

**Next:** `05_copula.ipynb` — tail dependence coefficients
"""),

) # end nb04
save(nb04, "04_dcc_garch.ipynb")


# ═══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 05 — Copula Tail Dependence
# ═══════════════════════════════════════════════════════════════════════════════
nb05 = nb(

md("""# 05 · Copula Tail Dependence Analysis
**Brazilian Stock-Bond Correlation Study**

Tests whether Brazilian asset pairs exhibit **asymmetric co-crash behaviour**:
do bonds and stocks crash *together* more than they boom together?

1. Transform returns to uniform pseudo-observations
2. Fit four copulas: Gaussian, Student-t, Clayton, Gumbel
3. Compare fit via AIC/BIC
4. Compute lower-tail dependence coefficient λ_L
5. Visualise joint tail behaviour
"""),

code("""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "../src")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats
from scipy.optimize import minimize_scalar, minimize
from scipy.stats import rankdata, t as t_dist

from fetch import load_master, CRISES

master = load_master()
plt.rcParams.update({
    "figure.dpi":150,"figure.facecolor":"white",
    "axes.spines.top":False,"axes.spines.right":False,
    "axes.grid":True,"grid.alpha":0.3,"font.size":11,
})
CRISIS_COLORS = {
    "GFC":"#d62728","Dilma":"#ff7f0e","Joesley":"#9467bd",
    "COVID":"#2ca02c","Americanas":"#8c564b","Fiscal24":"#e377c2",
}

# All copula densities live in src/metrics.py and are unit-tested (tests/test_metrics.py):
#   - each density integrates to 1 over the unit square
#   - Gumbel collapses to c == 1 at theta = 1 (the independence copula)
#   - Student-t converges to Gaussian as nu -> infinity
#   - fitting Clayton-simulated data recovers Clayton and its theta
# Getting a copula density wrong does not raise an error, it silently returns a
# likelihood for a function that is not a density -- and then AIC picks that family.
from metrics import (pseudo_obs, fit_all_copulas, tail_dependence_empirical,
                     gaussian_logpdf, student_t_logpdf, clayton_logpdf, gumbel_logpdf,
                     fit_gaussian, fit_student_t, fit_clayton, fit_gumbel)

# sanity check, cheap and worth running every time
import numpy as _np
_g = (_np.arange(200) + 0.5) / 200
_U, _V = _np.meshgrid(_g, _g)
_mass = _np.exp(gumbel_logpdf(2.0, _U.ravel(), _V.ravel())).sum() / 200**2
assert abs(_mass - 1) < 0.05, f"Gumbel density does not integrate to 1 ({_mass:.3f})"
assert _np.allclose(gumbel_logpdf(1.0, _U.ravel(), _V.ravel()), 0.0, atol=1e-9)
print("Copula densities OK (integrate to 1; Gumbel -> independence at theta=1)")
"""),

md("""## 2. Fit and compare all copulas — Ibovespa × NTN-B"""),

code("""
df_pair = master[["ibov","ntnb"]].dropna() * 100
u_df    = pseudo_obs(df_pair)
u, v    = u_df["ibov"].values, u_df["ntnb"].values
n       = len(u)

print(f"Fitting copulas (Ibovespa x NTN-B 5y, n={n:,})...")
fits = fit_all_copulas(u, v)          # returned already sorted by AIC

rows = [{
    "Copula":    f["family"],
    "Log-lik":   round(f["ll"], 1),
    "AIC":       round(f["AIC"], 1),
    "BIC":       round(f["BIC"], 1),
    "lambda_L":  round(f["lambda_L"], 4),
    "lambda_U":  round(f["lambda_U"], 4),
    "Key param": f["param"],
} for f in fits]

fit_tbl = pd.DataFrame(rows).set_index("Copula")
print(fit_tbl.to_string())
fit_tbl.to_csv("../outputs/nb_tbl_copula_fit.csv")
print("\\nBest fit (lowest AIC):", fit_tbl['AIC'].idxmin())
print("\\nParametric tail dependence:")
for n_, row in fit_tbl.iterrows():
    print(f"  {n_:<12} lambda_L = {row['lambda_L']:.4f}   lambda_U = {row['lambda_U']:.4f}")

# The parametric lambdas are only as good as the family that wins AIC. The empirical
# exceedance rate makes no distributional assumption at all, so quote both.
emp = tail_dependence_empirical(u, v, q=0.05)
print(f"\\nEmpirical 5% tail dependence (independence benchmark = 0.050):")
print(f"  lambda_L = {emp['lambda_L']:.3f}   lambda_U = {emp['lambda_U']:.3f}")
print(f"  co-crash observations: {emp['n_co_lower']} vs {emp['expected_indep']:.0f} "
      f"expected under independence ({emp['n_co_lower']/emp['expected_indep']:.1f}x)")
"""),

md("""## 3. Tail dependence across all asset pairs"""),

code("""
bond_pairs = [("ibov","ntnb"),("ibov","ltn"),("ibov","ntnf"),("ibov","lft")]
LABELS = {"ibov":"Ibovespa","ntnb":"NTN-B 5y","ltn":"LTN 2y",
          "ntnf":"NTN-F 10y","lft":"LFT 1y"}

summary_rows = []
for ca, cb in bond_pairs:
    df_p = master[[ca,cb]].dropna() * 100
    uu   = pseudo_obs(df_p)
    ui, vi = uu[ca].values, uu[cb].values

    fits_p = fit_all_copulas(ui, vi)
    best   = fits_p[0]
    emp_p  = tail_dependence_empirical(ui, vi, q=0.05)

    summary_rows.append({
        "Pair":            f"{LABELS[ca]} x {LABELS[cb]}",
        "n":               len(ui),
        "Best copula":     best["family"],
        "Best param":      best["param"],
        "lambda_L (fit)":  round(best["lambda_L"], 4),
        "lambda_U (fit)":  round(best["lambda_U"], 4),
        "lambda_L (emp)":  round(emp_p["lambda_L"], 3),
        "lambda_U (emp)":  round(emp_p["lambda_U"], 3),
        "co-crash obs":    emp_p["n_co_lower"],
        "expected indep":  round(emp_p["expected_indep"], 0),
    })
    print(f"{LABELS[ca]} x {LABELS[cb]:<10} best={best['family']:<10} {best['param']:<22} "
          f"emp lambda_L={emp_p['lambda_L']:.3f} lambda_U={emp_p['lambda_U']:.3f}  "
          f"co-crash {emp_p['n_co_lower']} vs {emp_p['expected_indep']:.0f}")

print("\\nlambda_L > lambda_U means the pair crashes together more than it booms together.")
print("Both are compared against 0.050, the rate implied by independence.")

summary_df = pd.DataFrame(summary_rows).set_index("Pair")
summary_df.to_csv("../outputs/nb_tbl_tail_dependence.csv")
print("\\nSaved: outputs/nb_tbl_tail_dependence.csv")
print(summary_df.to_string())
"""),

md("""## 4. Pseudo-observations scatter with tail quadrant analysis — Figure 7"""),

code("""
bond_cols = ["ntnb","ltn","ntnf","lft"]
colors    = ["#d62728","#ff7f0e","#2ca02c","#9467bd"]

fig, axes = plt.subplots(2, 2, figsize=(12, 11))
axes = axes.flatten()

for i, (col, color) in enumerate(zip(bond_cols, colors)):
    ax = axes[i]
    df_p = master[["ibov", col]].dropna() * 100
    uu   = pseudo_obs(df_p)
    ui, vi = uu["ibov"].values, uu[col].values
    crisis_labels = master["crisis"].reindex(df_p.index).fillna("None")

    # Base scatter (grey)
    ax.scatter(ui, vi, s=3, color="#cccccc", alpha=0.3, zorder=1)

    # Highlight crisis observations
    for cname in CRISES:
        mask = crisis_labels == cname
        if mask.sum() > 0:
            ax.scatter(ui[mask.values], vi[mask.values],
                       s=15, color=CRISIS_COLORS[cname],
                       alpha=0.8, zorder=2, label=cname)

    # Mark tail quadrants (bottom-left = co-crash)
    q_lo = 0.10
    ax.axvline(q_lo, color="black", ls="--", lw=0.7, alpha=0.5)
    ax.axhline(q_lo, color="black", ls="--", lw=0.7, alpha=0.5)

    # Count observations in lower-left quadrant
    in_ll = ((ui < q_lo) & (vi < q_lo)).sum()
    expected_indep = len(ui) * q_lo**2
    ax.text(0.02, 0.12,
            f"Co-crash\\nobserved: {in_ll}\\nexpected (indep): {expected_indep:.0f}",
            transform=ax.transAxes, fontsize=8.5, va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#d62728", alpha=0.8))

    # Annotate with the empirical exceedance rates (no distributional assumption)
    emp_i = tail_dependence_empirical(ui, vi, q=0.10)
    best_i = fit_all_copulas(ui, vi)[0]
    ax.set_title(f"Ibovespa x {LABELS[col]}\\n"
                 f"best fit: {best_i['family']} ({best_i['param']})\\n"
                 f"empirical 10% tail: lambda_L={emp_i['lambda_L']:.3f}  "
                 f"lambda_U={emp_i['lambda_U']:.3f}  (indep = 0.100)",
                 fontsize=9)
    ax.set_xlabel("Ibovespa (pseudo-obs u)", fontsize=9)
    ax.set_ylabel(f"{LABELS[col]} (pseudo-obs v)", fontsize=9)
    if i == 0:
        ax.legend(fontsize=7.5, loc="upper left")

fig.suptitle("Copula pseudo-observations: joint tail behaviour\\n"
             "(lower-left quadrant = simultaneous crashes, "
             "dashed lines = 10th percentile thresholds)",
             fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig("../outputs/fig_copula_scatter.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: outputs/fig_copula_scatter.png")
"""),

md("""## ✅ Notebook 05 complete

**How to read the output above.** Two tail-dependence estimates are reported and they
answer slightly different questions:

- **Parametric λ** comes from whichever family wins on AIC. It is only meaningful if
  that family actually describes the data, so it inherits all of the family's
  assumptions — a Student-t fit, for instance, *imposes* λ_L = λ_U and therefore
  cannot detect asymmetry even if it is present.
- **Empirical λ** is the raw exceedance rate P(V < q | U < q), with no distributional
  assumption. Compare it against `q` itself, which is the rate implied by independence.
  This is the estimate to quote when the question is "do these assets crash together
  more often than chance?"

Where the two disagree, prefer the empirical one and say so. Reporting only the
parametric λ from a mis-specified family is how a copula analysis ends up asserting
the opposite of what the data show.

**Next:** `06_portfolio_metrics.ipynb` — Diversification Ratio, ENB, PCA
"""),

) # end nb05
save(nb05, "05_copula.ipynb")


# ═══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 06 — Portfolio Metrics: DR, ENB, PCA
# ═══════════════════════════════════════════════════════════════════════════════
nb06 = nb(

md("""# 06 · Portfolio Metrics: Diversification Ratio, ENB, PCA
**Brazilian Stock-Bond Correlation Study**

Translates the correlation findings into portfolio-level risk metrics that
practitioners use to monitor and manage diversification quality.

1. Diversification Ratio (DR) — rolling
2. Effective Number of Bets (Meucci 2009) — rolling
3. PCA: fraction explained by PC1 — rolling
4. Three-panel dashboard chart — Figure 8 (whitepaper)
5. CoVaR: tail risk spillover from equities to bonds
"""),

code("""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "../src")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf

from fetch import load_master, CRISES, REGIMES

master = load_master()
plt.rcParams.update({
    "figure.dpi":150,"figure.facecolor":"white",
    "axes.spines.top":False,"axes.spines.right":False,
    "axes.grid":True,"grid.alpha":0.3,"font.size":11,
})
CRISIS_COLORS = {
    "GFC":"#d62728","Dilma":"#ff7f0e","Joesley":"#9467bd",
    "COVID":"#2ca02c","Americanas":"#8c564b","Fiscal24":"#e377c2",
}
LABELS = {"ibov":"Ibovespa","ntnb":"NTN-B 5y","ltn":"LTN 2y",
          "ntnf":"NTN-F 10y","lft":"LFT 1y"}

RET_COLS  = ["ibov","ntnb","ltn","ntnf","lft"]

def add_crisis_bands(ax, alpha=0.15):
    for name, (s, e) in CRISES.items():
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e),
                   color=CRISIS_COLORS[name], alpha=alpha)
"""),

md("""## 1. Portfolio metric functions"""),

code("""
from metrics import diversification_ratio, effective_num_bets, pc1_share

# DR  = (w'sigma) / sqrt(w'Sigma w).  1 = no benefit; upper bound grows with N.
# ENB = exp(Shannon entropy of PC risk contributions), Meucci (2009).
#       Bounded above by the NUMBER OF ASSETS -- for a 2-asset 60/40 the maximum is 2,
#       so "ENB fell to 1.1" means little without stating that ceiling.
# PC1 = share of variance on the first principal component of the CORRELATION matrix.
#       Computed on whichever columns you pass, so pass the portfolio's own holdings
#       if you intend to describe that portfolio.

print("Metric functions defined.")
print("  DR=1  → no diversification benefit")
print("  ENB=1 → single-factor portfolio (all risk from one PC)")
print("  PC1>0.7 → correlation regime: 'everything moves together'")
"""),

md("""## 2. Rolling metrics with three portfolio compositions"""),

code("""
WINDOW = 252

# Three portfolio compositions
PORTFOLIOS = {
    "60/40 Ibov-NTN-B":  {"ibov":0.60, "ntnb":0.40},
    "40/40/20 +LTN":     {"ibov":0.40, "ntnb":0.40, "ltn":0.10, "lft":0.10},
    "All-bond (excl.eq)":{"ntnb":0.50, "ltn":0.25, "ntnf":0.15, "lft":0.10},
}

# Compute rolling metrics
df_ret = master[RET_COLS].dropna(how="all")
results = {pname: {"dr":[], "enb":[], "dates":[]} 
           for pname in PORTFOLIOS}
pc1_series = []
pc1_dates  = []

print(f"Computing rolling {WINDOW}-day metrics...")
for i in range(WINDOW, len(df_ret)):
    window = df_ret.iloc[i-WINDOW:i]
    valid_cols = window.columns[window.notna().mean() > 0.8].tolist()
    if len(valid_cols) < 2:
        continue
    w_window = window[valid_cols].dropna()
    if len(w_window) < WINDOW//2: continue
    cov = w_window.cov().values

    # Portfolio metrics
    for pname, w_dict in PORTFOLIOS.items():
        # Use only weights for available columns
        avail = {c: w_dict[c] for c in valid_cols if c in w_dict}
        if not avail: continue
        total = sum(avail.values())
        w_arr = np.array([avail[c]/total for c in valid_cols if c in avail])
        cols_used = [c for c in valid_cols if c in avail]
        cov_sub = w_window[cols_used].cov().values
        dr  = diversification_ratio(w_arr, cov_sub)
        enb = effective_num_bets(w_arr, cov_sub)
        results[pname]["dr"].append(dr)
        results[pname]["enb"].append(enb)
        if not results[pname]["dates"] or results[pname]["dates"][-1] != df_ret.index[i]:
            results[pname]["dates"].append(df_ret.index[i])

    # PC1 for all return columns
    pc1 = pc1_share(window[valid_cols])
    pc1_series.append(pc1)
    pc1_dates.append(df_ret.index[i])

# Convert to Series
for pname in PORTFOLIOS:
    d = results[pname]
    n_dates = len(d["dates"])
    results[pname]["dr_s"]  = pd.Series(d["dr"][:n_dates],  index=d["dates"], name="DR")
    results[pname]["enb_s"] = pd.Series(d["enb"][:n_dates], index=d["dates"], name="ENB")

pc1_s = pd.Series(pc1_series, index=pc1_dates, name="PC1_share")
print(f"Done. PC1 share stats: mean={pc1_s.mean():.3f}  max={pc1_s.max():.3f}")
"""),

md("""## 3. Three-panel dashboard — Figure 8 (whitepaper)

**This chart summarises the portfolio-level evidence in one figure.**
All three metrics should collapse simultaneously during crisis periods.
"""),

code("""
port_main = "60/40 Ibov-NTN-B"
p_colors  = {"60/40 Ibov-NTN-B":"#1f77b4",
             "40/40/20 +LTN":"#ff7f0e",
             "All-bond (excl.eq)":"#2ca02c"}

fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)

# ── Panel 1: Diversification Ratio ───────────────────────────────────────────
ax = axes[0]
for pname, color in p_colors.items():
    dr = results[pname]["dr_s"]
    ax.plot(dr.index, dr, lw=1.5, color=color, label=pname)
ax.axhline(1.0, color="black", ls="--", lw=1, alpha=0.6, label="DR = 1 (no benefit)")
add_crisis_bands(ax, alpha=0.12)
ax.set_ylabel("Diversification Ratio", fontsize=10)
ax.set_title("Portfolio diversification metrics — Brazil 2005–2026", fontsize=13)
ax.legend(fontsize=8.5, ncol=2)
ax.set_ylim(0.9, None)

# ── Panel 2: Effective Number of Bets ────────────────────────────────────────
ax = axes[1]
for pname, color in p_colors.items():
    enb = results[pname]["enb_s"]
    ax.plot(enb.index, enb, lw=1.5, color=color, label=pname)
ax.axhline(1.0, color="black", ls="--", lw=1, alpha=0.6, label="ENB=1 (single bet)")
add_crisis_bands(ax, alpha=0.12)
ax.set_ylabel("Effective Number of Bets", fontsize=10)
ax.legend(fontsize=8.5, ncol=2)

# ── Panel 3: PC1 variance explained ──────────────────────────────────────────
ax = axes[2]
ax.fill_between(pc1_s.index, pc1_s * 100, color="#9467bd", alpha=0.5, label="PC1 variance %")
ax.plot(pc1_s.index, pc1_s * 100, color="#9467bd", lw=1)
ax.axhline(70, color="#d62728", ls="--", lw=1.2,
           label="70% threshold — diversification collapse")
ax.axhline(50, color="#ff7f0e", ls=":", lw=1,
           label="50% — elevated systemic risk")
add_crisis_bands(ax, alpha=0.12)
ax.set_ylabel("PC1 variance explained (%)", fontsize=10)
ax.legend(fontsize=8.5, ncol=2)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator(2))

# Crisis legend
crisis_handles = [plt.Rectangle((0,0),1,1, fc=CRISIS_COLORS[n], alpha=0.4, label=n)
                  for n in CRISES]
axes[2].legend(handles=crisis_handles, fontsize=8, loc="upper right")

plt.tight_layout()
plt.savefig("../outputs/fig_portfolio_metrics.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: outputs/fig_portfolio_metrics.png")
"""),

md("""## 4. CoVaR — tail risk spillover from equities to bonds"""),

code("""
import statsmodels.formula.api as smf
import scipy.stats as scipy_stats

df_pair = master[["ibov","ntnb"]].dropna() * 100

print("=== CoVaR Analysis: Ibovespa → NTN-B ===")
print("Model: NTN-B ~ Ibovespa (quantile regression)")
print()

for q in [0.05, 0.25, 0.50, 0.75, 0.95]:
    mod = smf.quantreg("ntnb ~ ibov", df_pair).fit(q=q)
    coef = mod.params["ibov"]
    print(f"  Q={q:.2f}  β={coef:+.4f}  "
          f"(bond return per 1% equity move at this quantile)")

# Delta-CoVaR
mod_05 = smf.quantreg("ntnb ~ ibov", df_pair).fit(q=0.05)
mod_50 = smf.quantreg("ntnb ~ ibov", df_pair).fit(q=0.50)

var_05 = df_pair["ibov"].quantile(0.05)
var_50 = df_pair["ibov"].quantile(0.50)

covar_05   = mod_05.params["Intercept"] + mod_05.params["ibov"] * var_05
delta_covar = mod_05.params["ibov"] * (var_05 - var_50)

print(f"\\nEquity VaR  (5th pct) : {var_05:.2f}%")
print(f"Equity VaR (50th pct) : {var_50:.2f}%")
print(f"CoVaR (NTN-B | eq@5%) : {covar_05:.2f}%")
print(f"ΔCoVaR                 : {delta_covar:.2f}%")
print(f"\\nInterpretation: when Ibovespa is at its 5th percentile ({var_05:.1f}%),")
print(f"NTN-B is expected to return {covar_05:.2f}% (CoVaR)")
print(f"ΔCoVaR of {delta_covar:.2f}% = incremental bond loss due to equity distress")
"""),

code("""
# CoVaR chart: quantile regression lines
fig, ax = plt.subplots(figsize=(9, 6))
x_range = np.linspace(df_pair["ibov"].min(), df_pair["ibov"].max(), 200)
colors_q = ["#d62728","#ff7f0e","#2ca02c","#9467bd","#8c564b"]

for q, color in zip([0.05, 0.25, 0.50, 0.75, 0.95], colors_q):
    mod = smf.quantreg("ntnb ~ ibov", df_pair).fit(q=q)
    y_hat = mod.params["Intercept"] + mod.params["ibov"] * x_range
    ax.plot(x_range, y_hat, lw=2, color=color, label=f"Q={q:.2f}")

# Scatter underlying data
ax.scatter(df_pair["ibov"], df_pair["ntnb"], s=2, color="gray", alpha=0.2, zorder=0)

# Mark CoVaR point
ax.axvline(var_05, color="black", ls="--", lw=1, alpha=0.6)
ax.scatter([var_05], [covar_05], s=100, color="#d62728", zorder=5,
           label=f"CoVaR = {covar_05:.2f}%")

ax.set_xlabel("Ibovespa daily return (%)", fontsize=11)
ax.set_ylabel("NTN-B 5y daily return (%)", fontsize=11)
ax.set_title("CoVaR: quantile regression of NTN-B on Ibovespa\\n"
             "(ΔCoVaR = equity distress contribution to bond tail risk)",
             fontsize=12)
ax.legend(fontsize=9)
ax.set_xlim(-10, 10); ax.set_ylim(-4, 4)
plt.tight_layout()
plt.savefig("../outputs/fig_covar.png", dpi=150, bbox_inches="tight")
plt.show()
"""),

md("""## ✅ Notebook 06 complete

**Key findings:**
- **Diversification Ratio** collapses toward 1.0 during every crisis — confirming diversification failure
- **ENB** drops sharply during Dilma, COVID, and Americanas — portfolio becomes a single-factor bet on Brazilian sovereign risk
- **PC1** exceeds 70% during COVID and Americanas — "all correlations go to one"
- **ΔCoVaR** quantifies the incremental bond loss when equities are distressed

**Outputs:** `fig_portfolio_metrics.png` (Figure 8), `fig_covar.png` (Figure 9)

**Next:** `07_stress_test.ipynb` — historical scenario replay + stressed VaR
"""),

) # end nb06
save(nb06, "06_portfolio_metrics.ipynb")


# ═══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 07 — Stress Testing
# ═══════════════════════════════════════════════════════════════════════════════
nb07 = nb(

md("""# 07 · Stress Testing & Scenario Analysis
**Brazilian Stock-Bond Correlation Study**

The final quantitative notebook: translates all correlation findings into
portfolio P&L consequences under historical and hypothetical stress scenarios.

1. Historical scenario replay: P&L for each crisis × each portfolio
2. Stressed VaR: compare calm vs. crisis covariance matrices
3. Correlation stress: what if all correlations → +1?
4. Master summary table for the whitepaper
"""),

code("""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "../src")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy.stats import norm

from fetch import load_master, CRISES, REGIMES

master  = load_master()
df_ret  = master[["ibov","ntnb","ltn","ntnf","lft"]].dropna(how="all")
LABELS  = {"ibov":"Ibovespa","ntnb":"NTN-B 5y","ltn":"LTN 2y",
           "ntnf":"NTN-F 10y","lft":"LFT 1y"}

plt.rcParams.update({
    "figure.dpi":150,"figure.facecolor":"white",
    "axes.spines.top":False,"axes.spines.right":False,
    "axes.grid":True,"grid.alpha":0.3,"font.size":11,
})
CRISIS_COLORS = {
    "GFC":"#d62728","Dilma":"#ff7f0e","Joesley":"#9467bd",
    "COVID":"#2ca02c","Americanas":"#8c564b","Fiscal24":"#e377c2",
}

# ── Portfolio definitions ─────────────────────────────────────────────────────
PORTFOLIOS = {
    "60/40 (Ibov+NTN-B)":       {"ibov":0.60, "ntnb":0.40},
    "Diversified (4 assets)":    {"ibov":0.40, "ntnb":0.30, "ltn":0.15, "lft":0.15},
    "Equity heavy (80/20)":      {"ibov":0.80, "ntnb":0.20},
    "Bond heavy (20/80)":        {"ibov":0.20, "ntnb":0.50, "ltn":0.20, "lft":0.10},
    "LFT only (cash proxy)":     {"lft":1.00},
}

print("Portfolios defined:")
for name, weights in PORTFOLIOS.items():
    print(f"  {name}: {weights}")
"""),

md("""## 1. Historical scenario replay — Table 1 (whitepaper)"""),

code("""
def scenario_return(weights, returns_period):
    \"\"\"Compute total return for a portfolio over a period.\"\"\"
    total = 0.0
    for col, w in weights.items():
        if col in returns_period.index:
            total += w * returns_period[col]
    return total

# Compute cumulative log returns per crisis
crisis_cum = {}
for cname, (s, e) in CRISES.items():
    mask = (df_ret.index >= s) & (df_ret.index <= e)
    period_ret = df_ret[mask]
    cum = period_ret.sum()  # sum of daily log returns ≈ total log return
    crisis_cum[cname] = cum

# Build P&L table
rows = {}
for pname, weights in PORTFOLIOS.items():
    row = {}
    for cname in CRISES:
        pnl = scenario_return(weights, crisis_cum[cname])
        row[cname] = round((np.exp(pnl) - 1) * 100, 1)
    rows[pname] = row

pnl_df = pd.DataFrame(rows).T
pnl_df.index.name = "Portfolio"
pnl_df.columns.name = "Crisis"

print("=== Historical scenario P&L: total return (%) ===")
print(pnl_df.to_string())
pnl_df.to_csv("../outputs/nb_tbl_scenario_pnl_total.csv")

# ── The same P&L in excess of CDI ────────────────────────────────────────────
# Total return is the wrong lens for a diversification question in Brazil. The CDI
# ran at 10-15% p.a. over most of this sample, so a seven-month crisis window accrues
# ~7% of carry before any price move. A bond portfolio that "made money in the GFC"
# may simply have earned carry while losing to cash. Excess-over-CDI is what a
# Brazilian investor actually chooses between, since the alternative is always
# holding Selic-linked cash.
exc = {}
for pname, weights in PORTFOLIOS.items():
    row = {}
    for cname, (s, e) in CRISES.items():
        sub  = master[(master.index >= s) & (master.index <= e)]
        port = sum(w * sub[c] for c, w in weights.items())
        row[cname] = round((np.exp(port.sum()) - np.exp(sub["cdi_ret"].sum())) * 100, 1)
    exc[pname] = row

exc_df = pd.DataFrame(exc).T
exc_df.index.name = "Portfolio"
print("\\n=== Same episodes, in excess of CDI (percentage points) ===")
print(exc_df.to_string())
exc_df.to_csv("../outputs/nb_tbl_scenario_pnl_excess_cdi.csv")

print("\\nNote how the two tables differ. Positive total returns during a crisis are")
print("mostly carry: the LFT-only row is ~0.0 by definition in excess terms, because")
print("holding Selic-linked cash IS the benchmark. Read the second table when the")
print("question is whether diversification helped.")
print("\\nSaved: outputs/nb_tbl_scenario_pnl_total.csv, tbl_scenario_pnl_excess_cdi.csv")
"""),

code("""
# Heatmap visualisation
fig, ax = plt.subplots(figsize=(11, 5))
sns.heatmap(
    pnl_df,
    annot=True, fmt=".1f",
    cmap="RdYlGn", center=0, vmin=-50, vmax=25,
    linewidths=0.5, linecolor="white",
    cbar_kws={"label":"Total return (%)", "shrink":0.7},
    ax=ax,
)
ax.set_title("Portfolio P&L across historical crisis episodes (%)\\n"
             "(negative = loss, red = severe loss)", fontsize=12, pad=12)
ax.set_xlabel(""); ax.set_ylabel("")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("../outputs/fig_scenario_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: outputs/fig_scenario_heatmap.png")
"""),

md("""## 2. Stressed VaR: calm vs. crisis covariance matrices"""),

code("""
from metrics import portfolio_var, shrink_to_equicorr

# portfolio_var(weights, cov_DAILY, alpha, horizon_days) -> positive % loss.
#
# The horizon is an explicit argument because it is the single biggest lever on the
# answer. Scaling a crisis-window daily covariance by sqrt(252) assumes crisis-level
# volatility persists for a full year; on Brazilian crisis windows that produces 99%
# VaR figures above 100% of capital for a long-only unlevered portfolio, which is
# impossible rather than merely conservative. We quote a 10-day horizon, which is
# both the Basel market-risk standard and comparable in length to the crisis windows
# the covariances are estimated from.
HORIZON_DAYS = 10

# DAILY covariance matrices — portfolio_var applies the horizon itself
cov_full = df_ret.dropna().cov()

# A k-asset covariance needs materially more than k observations to be usable.
# The Joesley window is 11 trading days against 5 assets, so it is excluded rather
# than silently producing a number with no precision behind it.
MIN_OBS = 30
crisis_covs, skipped = {}, {}
for cname, (s, e) in CRISES.items():
    period = df_ret[(df_ret.index >= s) & (df_ret.index <= e)].dropna()
    if len(period) >= MIN_OBS:
        crisis_covs[cname] = period.cov()
    else:
        skipped[cname] = len(period)

print(f"=== 99% {HORIZON_DAYS}-day Gaussian VaR (% of capital) ===")
print(f"{'Portfolio':<26}  {'Calm':>8}", end="")
for cname in crisis_covs: print(f"  {cname:>11}", end="")
print()

var_rows = {}
for pname, weights in PORTFOLIOS.items():
    var_calm = portfolio_var(weights, cov_full, 0.99, HORIZON_DAYS)
    var_row  = {"Calm (full sample)": round(var_calm, 1)}
    print(f"{pname:<26}  {var_calm:>8.1f}", end="")
    for cname, cov_c in crisis_covs.items():
        v = portfolio_var(weights, cov_c, 0.99, HORIZON_DAYS)
        var_row[cname] = round(v, 1)
        print(f"  {v:>11.1f}", end="")
    var_rows[pname] = var_row
    print()

var_df = pd.DataFrame(var_rows).T
var_df.to_csv("../outputs/nb_tbl_stressed_var.csv")

for cname, n_ in skipped.items():
    print(f"\\nSKIPPED {cname}: {n_} trading days < {MIN_OBS} required for a "
          f"{df_ret.shape[1]}x{df_ret.shape[1]} covariance.")

assert var_df.max().max() < 100, "VaR above 100% of capital — check the horizon scaling"
print("\\nSanity check passed: no VaR exceeds 100% of capital.")
print("Saved: outputs/nb_tbl_stressed_var.csv")
"""),

md("""## 3. Correlation stress: shrink toward equicorrelation"""),

code("""
def shrink_to_equicorr(cov, alpha):
    \"\"\"
    Interpolate between current correlation and equicorrelation (all ρ=1).
    alpha=0: current cov.  alpha=1: all correlations = 1.
    \"\"\"
    vols   = np.sqrt(np.diag(cov))
    corr   = cov / np.outer(vols, vols)
    eq     = np.ones_like(corr)
    s_corr = (1-alpha)*corr + alpha*eq
    # Ensure PSD
    eigvals = np.linalg.eigvalsh(s_corr)
    if np.any(eigvals < 0):
        s_corr += (-eigvals.min() + 1e-8) * np.eye(len(s_corr))
    return np.outer(vols, vols) * s_corr

alphas = np.linspace(0, 1, 50)
pname  = "60/40 (Ibov+NTN-B)"
cols   = list(PORTFOLIOS[pname].keys())
cov_sub = cov_full.loc[cols, cols].values
w_arr   = np.array(list(PORTFOLIOS[pname].values()))

stress_vars = []
for alpha in alphas:
    cov_s    = shrink_to_equicorr(cov_sub, alpha)
    port_vol = np.sqrt(w_arr @ cov_s @ w_arr)          # cov_sub is a DAILY covariance
    var_pct  = -norm.ppf(0.01) * port_vol * np.sqrt(HORIZON_DAYS) * 100
    stress_vars.append(var_pct)

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(alphas * 100, stress_vars, lw=2.5, color="#1f77b4")
ax.axvline(0,   color="black",   ls="--", lw=1, alpha=0.5, label="Current correlations")
ax.axvline(100, color="#d62728", ls="--", lw=1, alpha=0.5, label="All corr = 1 (worst case)")

# Mark current VaR
ax.axhline(stress_vars[0], color="gray", ls=":", lw=1)
ax.text(2, stress_vars[0]+0.3, f"Current: {stress_vars[0]:.1f}%", fontsize=9)
ax.text(55, stress_vars[-1]+0.3, f"Worst case: {stress_vars[-1]:.1f}%", fontsize=9)

ax.set_xlabel("Equicorrelation stress (α%): 0=current, 100=all ρ=1", fontsize=10)
ax.set_ylabel("99% 1-year VaR (%)", fontsize=10)
ax.set_title(f"Correlation stress test: {pname}\\n"
             "How much does VaR increase as correlations approach +1?",
             fontsize=11)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("../outputs/fig_correlation_stress.png", dpi=150, bbox_inches="tight")
plt.show()

increase = (stress_vars[-1] - stress_vars[0]) / stress_vars[0] * 100
print(f"VaR increase from current → equicorrelation: +{increase:.1f}%")
"""),

md("""## 4. Master summary chart: complete crisis evidence — Figure 10"""),

code("""
# Drawdown paths during each crisis for the 60/40 portfolio
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

pname  = "60/40 (Ibov+NTN-B)"
weights = PORTFOLIOS[pname]

for i, (cname, (s, e)) in enumerate(CRISES.items()):
    ax = axes[i]
    mask   = (df_ret.index >= s) & (df_ret.index <= e)
    period = df_ret[mask]

    # Portfolio and individual asset cumulative returns
    port_cum = pd.Series(
        [np.exp(sum(weights.get(c,0)*period[c].iloc[:t].sum()
                    for c in period.columns if c in weights))-1
         for t in range(len(period))],
        index=period.index
    ) * 100

    ibov_cum = (np.exp(period["ibov"].cumsum()) - 1) * 100
    ntnb_cum = (np.exp(period["ntnb"].cumsum()) - 1) * 100
    lft_cum  = (np.exp(period["lft"].cumsum()) - 1) * 100

    ax.plot(ibov_cum.index, ibov_cum, color="#1f77b4", lw=1.5,
            alpha=0.7, label="Ibovespa")
    ax.plot(ntnb_cum.index, ntnb_cum, color="#d62728", lw=1.5,
            alpha=0.7, label="NTN-B")
    ax.plot(lft_cum.index,  lft_cum,  color="#2ca02c", lw=1.2,
            ls="--", alpha=0.7, label="LFT")
    ax.plot(port_cum.index, port_cum, color="black", lw=2.2,
            label=f"60/40 portfolio")

    ax.axhline(0, color="gray", ls="--", lw=0.7)
    n_days = len(period)
    ax.set_title(f"{cname}  ({s[:7]}–{e[:7]})\\n{n_days} trading days",
                 fontsize=10, fontweight="bold",
                 color=CRISIS_COLORS[cname])
    ax.set_ylabel("Cumulative return (%)", fontsize=8.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%y"))
    if i == 0:
        ax.legend(fontsize=7.5, loc="lower left")

fig.suptitle("60/40 Portfolio (Ibovespa + NTN-B): performance during each crisis\\n"
             "Diversification fails — stocks and bonds fall together",
             fontsize=13, y=1.01, fontweight="bold")
plt.tight_layout()
plt.savefig("../outputs/fig_crisis_drawdowns.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: outputs/fig_crisis_drawdowns.png")
"""),

md("""## ✅ Notebook 07 complete — Study fully quantified

**Summary of all stress test findings:**

| Finding | Result |
|---------|--------|
| Worst crisis (60/40 portfolio) | GFC or COVID depending on timing |
| LFT-only portfolio across ALL crises | Near-zero loss — confirms capital preservation role |
| VaR increase (calm → equicorrelation) | Significant amplification |
| ΔCoVaR | Bonds absorb additional loss when equities distressed |
| Both Ibovespa AND NTN-B negative simultaneously | Confirmed in majority of episodes |

**All outputs generated:**
- `tbl_scenario_pnl.csv` / `fig_scenario_heatmap.png` — Table 1 + Figure 10
- `tbl_stressed_var.csv` — Table 3
- `fig_correlation_stress.png` — Figure 11
- `fig_crisis_drawdowns.png` — Figure 12

**Complete figure list for whitepaper:**

| Figure | Notebook | Filename |
|--------|----------|---------|
| 1. Regime timeline | 01 | fig_macro_validation.png |
| 2. Cumulative returns | 01 | fig_cumulative_returns.png |
| 3. Correlation matrix (full + regime) | 02 | fig_corr_by_regime.png |
| 4. Crisis returns heatmap | 02 | fig_crisis_returns_heatmap.png |
| 5. Rolling correlation | 03 | fig_rolling_correlation.png |
| 6. Conditional tail correlations | 03 | fig_conditional_correlations.png |
| 7. DCC-GARCH rho_t | 04 | fig_dcc_correlation.png |
| 8. DCC vs EMBI | 04 | fig_dcc_vs_embi.png |
| 9. Copula scatter + tail dependence | 05 | fig_copula_scatter.png |
| 10. DR/ENB/PC1 dashboard | 06 | fig_portfolio_metrics.png |
| 11. CoVaR quantile regression | 06 | fig_covar.png |
| 12. Scenario P&L heatmap | 07 | fig_scenario_heatmap.png |
| 13. Correlation stress test | 07 | fig_correlation_stress.png |
| 14. Crisis drawdowns | 07 | fig_crisis_drawdowns.png |
"""),

) # end nb07
save(nb07, "07_stress_test.ipynb")


# ═══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 08 — Matched Cross-Country Panel: Testing Convergence
# ═══════════════════════════════════════════════════════════════════════════════
nb08 = nb(

md("""# 08 · Matched Cross-Country Panel — Testing Convergence
**Brazilian Stock-Bond Correlation Study**

Notebooks 01–07 establish what Brazil's stock-bond correlation looks like. This one
answers the comparative question the paper's thesis rests on: **are advanced economies
converging toward Brazil's regime?**

That cannot be settled from Brazilian data. It needs the same asset definitions and the
same return construction in every country — otherwise any cross-country difference could
be an artefact of how the series were built.

1. Build a matched monthly panel: US, Germany, Japan, UK, Brazil
2. Validate the bond construction against Brazil's independent PU-based series
3. Stock-bond correlations by country and period, with confidence intervals
4. The convergence test (difference-in-differences, block bootstrap)
5. The floor, stated cross-sectionally

> **No API key required.** Earlier versions of this notebook needed `FRED_API_KEY`.
> It now uses FRED's public CSV endpoint, so it runs from a clean checkout.
"""),

code("""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "../src")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from fetch import load_master
from global_data import (build_global_panel, validate_panel,
                         validate_against_pu_construction,
                         ALL_COUNTRIES, IMF_BREAK)
import metrics as M

plt.rcParams.update({
    "figure.dpi": 150, "figure.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "font.size": 11,
})

BENCH = "BR"
DM    = ["US", "DE", "JP", "GB"]
NAME  = {cc: spec["name"] for cc, spec in ALL_COUNTRIES.items()}
COLOR = {"US":"#1f77b4", "DE":"#ff7f0e", "JP":"#2ca02c",
         "GB":"#9467bd", "BR":"#d62728"}

master = load_master()
panel  = build_global_panel(master=master)
print(f"Panel: {panel.shape[0]} months, "
      f"{panel.index.min().date()} to {panel.index.max().date()}")
"""),

md("""## 1. Validate the panel

Equities are the OECD total share price index (`SPASTT01<CC>M661N`); bonds are the OECD
long-term 10-year government bond yield (`IRLTLT01<CC>M156N`), converted to a
constant-maturity total return. Both are OECD-harmonised, so the definition is identical
across countries — that is why they are preferred to national indices like the DAX.

Brazil has no OECD long-term yield series, so it enters from the domestic pipeline:
Ibovespa plus the 10-year NTN-F yield. Same instrument type, different source.
"""),

code("""
fails = validate_panel(panel)
assert not fails, f"panel validation failed: {fails}"
"""),

md("""## 2. Does the construction drive the result?

Bond returns outside Brazil are built from yields, because no unit-price data is
available. Brazil is the one country where **both** methods can be applied, so it is the
control: if the yield-based construction reproduces the PU-based series there, applying
it to the other four is justified. If it does not, the whole panel is suspect.
"""),

code("""
xc = validate_against_pu_construction(master=master)

br_y  = master["ntnf_yield"].resample("ME").last().dropna() * 100
from global_data import constant_maturity_bond_return
yb = constant_maturity_bond_return(br_y)
pu = master["ntnf"].resample("ME").sum(); pu = np.exp(pu[pu != 0]) - 1

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

ax = axes[0]
both = pd.concat([yb.rename("yield"), pu.rename("pu")], axis=1).dropna()
ax.plot(both.index, (1+both["yield"]).cumprod(), lw=1.8, color="#1f77b4",
        label="yield-based (used for all 5 countries)")
ax.plot(both.index, (1+both["pu"]).cumprod(), lw=1.4, ls="--", color="#d62728",
        label="PU-based (Brazil only)")
ax.set_yscale("log"); ax.set_ylabel("Growth of 1 (log scale)")
ax.set_title("Brazil NTN-F 10y: two independent constructions", fontsize=11)
ax.legend(fontsize=9); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

ax = axes[1]
ax.scatter(both["pu"]*100, both["yield"]*100, s=14, alpha=0.6, color="#2ca02c")
lim = [both.min().min()*100, both.max().max()*100]
ax.plot(lim, lim, "k--", lw=1, alpha=0.6)
ax.set_xlabel("PU-based monthly return (%)"); ax.set_ylabel("Yield-based monthly return (%)")
ax.set_title(f"rho = {xc['rho_constructions']:.3f}", fontsize=11)

plt.tight_layout()
plt.savefig("../outputs/fig_global_construction_check.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"Headline correlation, yield-based : {xc['rho_yield_based']:+.3f}")
print(f"Headline correlation, PU-based    : {xc['rho_pu_based']:+.3f}")
print(f"Difference                        : {abs(xc['rho_yield_based']-xc['rho_pu_based']):.3f}")
print("\\nA small difference here is what licenses the yield-based method elsewhere.")
"""),

md("""## 3. Stock-bond correlation by country and period

Split at the IMF's turning point, 31 December 2019. Each correlation carries a Fisher-z
interval, and the pre/post shift is tested with a block bootstrap — four point estimates
moving in the same direction is not by itself evidence that any of them moved.
"""),

code("""
pre  = panel[panel.index <= IMF_BREAK]
post = panel[panel.index >  IMF_BREAK]
print(f"pre-break  {pre.index.min().date()} to {pre.index.max().date()}  ({len(pre)} months)")
print(f"post-break {post.index.min().date()} to {post.index.max().date()}  ({len(post)} months)\\n")

rows = []
for cc in DM + [BENCH]:
    p0 = M.corr_with_ci(pre[f"{cc}_eq"],  pre[f"{cc}_bd"])
    p1 = M.corr_with_ci(post[f"{cc}_eq"], post[f"{cc}_bd"])
    t  = M.bootstrap_corr_diff(post[f"{cc}_eq"], post[f"{cc}_bd"],
                               pre[f"{cc}_eq"],  pre[f"{cc}_bd"],
                               n_boot=2000, block=6, seed=2)
    rows.append({"Country": NAME[cc],
                 "rho pre": round(p0["rho"], 3),
                 "CI pre":  f"[{p0['lo']:+.3f},{p0['hi']:+.3f}]",
                 "rho post": round(p1["rho"], 3),
                 "CI post": f"[{p1['lo']:+.3f},{p1['hi']:+.3f}]",
                 "shift": round(p1["rho"]-p0["rho"], 3),
                 "p": round(t["p"], 3),
                 "shifted 5%": "yes" if t["p"] < 0.05 else "no"})
corr_tbl = pd.DataFrame(rows).set_index("Country")
print(corr_tbl.to_string())
corr_tbl.to_csv("../outputs/nb_tbl_global_correlations.csv")

fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(DM)+1); w = 0.36
ccs = DM + [BENCH]
pre_v  = [M.corr_with_ci(pre[f"{c}_eq"],  pre[f"{c}_bd"])  for c in ccs]
post_v = [M.corr_with_ci(post[f"{c}_eq"], post[f"{c}_bd"]) for c in ccs]
ax.bar(x-w/2, [v["rho"] for v in pre_v], w, label="pre-2020", color="#1f77b4",
       yerr=[[v["rho"]-v["lo"] for v in pre_v], [v["hi"]-v["rho"] for v in pre_v]],
       capsize=3, error_kw={"lw":1})
ax.bar(x+w/2, [v["rho"] for v in post_v], w, label="post-2020", color="#d62728",
       yerr=[[v["rho"]-v["lo"] for v in post_v], [v["hi"]-v["rho"] for v in post_v]],
       capsize=3, error_kw={"lw":1})
ax.axhline(0, color="black", lw=1)
ax.set_xticks(x); ax.set_xticklabels([NAME[c] for c in ccs], fontsize=9)
ax.set_ylabel("Monthly stock-bond correlation")
ax.set_title("All four advanced economies lost a significantly negative correlation.\\n"
             "None became significantly positive. Brazil never moved.", fontsize=11)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("../outputs/fig_global_correlations.png", dpi=150, bbox_inches="tight")
plt.show()
"""),

md("""## 4. The convergence test

Convergence is a claim about a **gap closing**, and both correlations are estimated with
error, so comparing point estimates by eye is not a test. Difference-in-differences:

$$\\\\text{DiD}(c) = [\\\\rho_{post}(c) - \\\\rho_{pre}(c)] - [\\\\rho_{post}(BR) - \\\\rho_{pre}(BR)]$$

A positive, significant DiD means country *c* moved toward Brazil by more than Brazil
itself moved.
"""),

code("""
rows = []
for cc in DM:
    r0 = M.corr_with_ci(pre[f"{cc}_eq"],  pre[f"{cc}_bd"])["rho"]
    r1 = M.corr_with_ci(post[f"{cc}_eq"], post[f"{cc}_bd"])["rho"]
    b0 = M.corr_with_ci(pre[f"{BENCH}_eq"],  pre[f"{BENCH}_bd"])["rho"]
    b1 = M.corr_with_ci(post[f"{BENCH}_eq"], post[f"{BENCH}_bd"])["rho"]
    d = M.bootstrap_did((pre[f"{cc}_eq"], pre[f"{cc}_bd"]),
                        (post[f"{cc}_eq"], post[f"{cc}_bd"]),
                        (pre[f"{BENCH}_eq"], pre[f"{BENCH}_bd"]),
                        (post[f"{BENCH}_eq"], post[f"{BENCH}_bd"]),
                        n_boot=2000, block=6, seed=3)
    rows.append({"Country": NAME[cc],
                 "gap pre": round(abs(b0-r0), 3), "gap post": round(abs(b1-r1), 3),
                 "narrowed": "yes" if abs(b1-r1) < abs(b0-r0) else "no",
                 "DiD": round(d["did"], 3), "boot SE": round(d["boot_se"], 3),
                 "p": round(d["p"], 3),
                 "converged 5%": "yes" if (d["p"] < 0.05 and abs(b1-r1) < abs(b0-r0)) else "no"})
conv = pd.DataFrame(rows).set_index("Country")
print(conv.to_string())
conv.to_csv("../outputs/nb_tbl_global_convergence.csv")

n_nar = (conv["narrowed"] == "yes").sum()
n_sig = (conv["converged 5%"] == "yes").sum()
print(f"\\nGap to Brazil narrowed in {n_nar}/4 advanced economies.")
print(f"Narrowing significant at 5% in {n_sig}/4.")
print("\\nDirection is unanimous; significance is not. And four highly correlated")
print("bond markets moving together is closer to one or two independent")
print("observations than to four -- do not read 4/4 as four pieces of evidence.")
"""),

md("""## 5. The floor, stated cross-sectionally

The single most robust cross-country statement in this study is not about a level but
about a **floor**: how negative did each market's stock-bond correlation ever get?
"""),

code("""
W = 60
roll = pd.DataFrame({NAME[cc]: panel[f"{cc}_eq"].rolling(W).corr(panel[f"{cc}_bd"])
                     for cc in DM + [BENCH]}).dropna(how="all")

fig, ax = plt.subplots(figsize=(14, 5.5))
for cc in DM + [BENCH]:
    lw = 2.6 if cc == BENCH else 1.4
    ax.plot(roll.index, roll[NAME[cc]], lw=lw, color=COLOR[cc], label=NAME[cc],
            alpha=1.0 if cc == BENCH else 0.85)
ax.axhline(0, color="black", lw=1.2, ls="--")
ax.axvline(pd.Timestamp(IMF_BREAK), color="navy", lw=1.5, ls=":", alpha=0.8)
ax.text(pd.Timestamp(IMF_BREAK), ax.get_ylim()[1]*0.95, " IMF break", fontsize=8,
        color="navy", va="top")
ax.set_ylabel(f"{W}-month rolling correlation")
ax.set_title("Brazil's stock-bond correlation never goes negative.\\n"
             "Every advanced economy spent years below zero.", fontsize=12)
ax.legend(fontsize=9, ncol=5, loc="lower right")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.tight_layout()
plt.savefig("../outputs/fig_global_rolling_correlation.png", dpi=150, bbox_inches="tight")
plt.show()

floor = roll.min().round(3).rename("minimum 60m correlation").to_frame()
floor["ever below zero"] = np.where(floor["minimum 60m correlation"] < 0, "yes", "no")
print(floor.to_string())
roll.round(4).to_csv("../outputs/nb_tbl_global_rolling_correlation.csv")
"""),

md("""## ✅ Notebook 08 complete

**What the panel establishes**

| Question | Answer |
|----------|--------|
| Does the IMF's advanced-economy finding replicate? | Yes — all four had significantly negative pre-2020 correlations, none does now |
| Did they become significantly *positive*? | **No** — all four post-2020 intervals straddle zero. The change is the loss of a hedge, not the acquisition of positive co-movement |
| Did Brazil shift? | No (p ≈ 0.40). It is significantly positive in both sub-periods |
| Did the gap to Brazil narrow? | In 4 of 4 — but significantly in only 1 (Germany) |
| Is convergence established? | **Directionally unanimous, statistically not.** 78 post-break months, and four non-independent markets |
| What is robust? | The **floor**: Brazil's 60-month correlation never went below zero; every advanced economy spent time below −0.45 |

**Outputs:** `fig_global_construction_check.png`, `fig_global_correlations.png`,
`fig_global_rolling_correlation.png`, and the `nb_tbl_global_*.csv` tables.

The paper's Section 9 reports these results; `scripts/run_global_analysis.py` regenerates
the canonical `outputs/tbl_global_*.csv` versions headlessly.
"""),

) # end nb08
save(nb08, "08_global_macro.ipynb")

print("\nAll notebooks written successfully.")
print("To run: cd notebooks && jupyter notebook")
