# Implementation Guide: Brazilian Stock-Bond Correlation Study

**Format:** Python notebooks → whitepaper  
**Timeline:** 3–4 weeks  
**Data:** Free BCB/ANBIMA public + ANBIMA Feed API (optional, gated)

---

## Architecture overview

```
project/
├── data/
│   ├── raw/          # BCB SGS, IPEADATA, Tesouro CSV
│   └── processed/    # master_returns.csv — aligned returns + levels + event labels
├── notebooks/
│   ├── 01_data.ipynb            # pipeline + construction validation
│   ├── 02_descriptive.ipynb
│   ├── 03_rolling_corr.ipynb    # + frequency robustness, Forbes-Rigobon
│   ├── 04_dcc_garch.ipynb
│   ├── 05_copula.ipynb
│   ├── 06_portfolio_metrics.ipynb
│   ├── 07_stress_test.ipynb
│   └── 08_global_macro.ipynb    # hand-maintained; needs FRED_API_KEY
├── src/
│   ├── fetch.py      # All data ingestion + validate_master()
│   └── metrics.py    # Inference, Forbes-Rigobon, copulas, DCC, DR/ENB/PC1, VaR
├── tests/
│   ├── test_metrics.py          # 28 unit tests for the estimators
│   └── test_paper_consistency.py # 49 tests: paper numbers vs outputs/
├── scripts/
│   ├── run_analysis.py          # regenerates every table in the paper
│   └── build_notebooks.py       # generates notebooks 01-07
├── config/plot_style.py         # shared matplotlib rcParams (apply_style())
└── outputs/          # All charts/tables for the paper
```

**Why estimators live in `src/metrics.py` rather than in the notebooks.** A copula
density with a wrong exponent does not raise an error — it returns a likelihood for a
function that is not a density, and AIC then selects it over the correct families. The
same applies to a DCC likelihood, a VaR horizon, or a heteroskedasticity correction:
these fail silently and produce plausible numbers. Keeping them in one importable
module makes them unit-testable, and `tests/test_metrics.py` asserts the properties
that catch exactly this class of error — densities integrating to 1, copulas
collapsing to independence at their independence parameter, the DCC recovering known
simulated parameters, and VaR never exceeding 100% of capital for a long-only
unlevered portfolio.

---

## Week 1 — Data + Descriptive Foundation

### Day 1–2: Data pipeline

**Install dependencies:**

```bash
pip install -r requirements.txt
```

**Core series actually used** (implemented in `src/fetch.py`; verified against the
live APIs, not copied from a catalogue):

| Series | Source | Code | Units — read this column carefully |
|--------|--------|------|------------------------------------|
| Ibovespa daily close | IPEADATA | `GM366_IBVSP366` | index points, 1993– |
| EMBI+ Brazil | IPEADATA | `JPM366_EMBI366` | **basis points**, ends Jul 2024 |
| CDI | BCB SGS | 12 | **percent per DAY** (0.0420 ≈ 10.7% p.a.) |
| Selic target | BCB SGS | 432 | percent per annum |
| IPCA | BCB SGS | 433 | percent per month |
| IGP-M | BCB SGS | 189 | percent per month |
| PTAX BRL/USD | BCB SGS | 1 | BRL per USD |
| NTN-B / LTN / NTN-F / LFT PU + yields | Tesouro Transparente | `PrecoTaxaTesouroDireto.csv` | R$ per unit, Dec 2004– |

Three traps worth stating explicitly, because each one silently produces plausible
numbers rather than an error:

- **SGS 12 is a daily rate, not annual.** Compounding it as annual and taking a
  252nd root understates the accrual by roughly two orders of magnitude, and
  propagates into any Sharpe ratio computed against it.
- **EMBI is not in the SGS system at all.** SGS code 21619, which reads plausibly as
  a spread, is the EUR/BRL exchange rate. Use IPEADATA.
- **Ibovespa is not in SGS as a daily series either.** IPEADATA has it back to 1993.

Validate any series you add against a known historical value before using it: the
Ibovespa should print 73,517 on 2008-05-20 and EMBI+ should print ~2,443bp at the
September 2002 peak.

**Fetch pattern:**

```python
import requests, pandas as pd

# BCB SGS paginates at ~10-year windows; src/fetch.py chunks at 5 years.
url = ("https://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados"
       "?formato=json&dataInicial=01/01/2004&dataFinal=31/12/2008")
df = pd.DataFrame(requests.get(url, timeout=60).json())

# IPEADATA OData. VALDATA carries a local UTC offset, so slice the date component
# rather than converting timezones -- historical Brazilian DST transitions raise.
url = "http://www.ipeadata.gov.br/api/odata4/ValoresSerie(SERCODIGO='GM366_IBVSP366')"
v = pd.DataFrame(requests.get(url, timeout=180).json()["value"])
v["d"] = pd.to_datetime(v["VALDATA"].str.slice(0, 10))
```

**Bond total returns.** The Tesouro file gives unit prices, not returns. Two things
must be handled or the series is wrong in ways that survive inspection:

- **Rolls.** Take the return of the bond selected on the *previous* day. Differencing
  across a change of maturity prices one bond against a different one.
- **Coupons.** An untreated NTN-B coupon looks like a one-day loss of ~2.9% of VNA,
  about 42 times over the sample. `src/fetch.py` recovers the face/VNA from the
  bond's own quoted yield through a du/252 cashflow pricer, so no external VNA
  series is needed. Validate the pricer on NTN-F, whose face is known to be 1,000.

**LFT.** Build it from observed Tesouro Selic prices. Defining the LFT return as
compounded CDI makes "the LFT never loses money" true by construction and therefore
untestable — and it hides that long LFTs drew down 1.33% in October 2020.

**🔒 ANBIMA-gated (buy later):** IDA-DI, IDA-IPCA, IDA-Geral for a debenture total
return index. Note that Yahoo Finance (`IMAB11.SA`, `IB5M11.SA`) is unreachable from
some environments and is not used by the current pipeline.

---

### Day 3–4: Clean + align data

**Key decisions:**

- **Frequency:** Daily for DCC-GARCH and copula; monthly for rolling correlations and stress tables
- **Returns:** Log returns for modeling; simple returns for scenario P&L
- **Sample:** Full 2004–2025 for main analysis; sub-samples per regime for robustness

```python
import numpy as np

prices = raw[['ibov', 'imab', 'imab5p', 'irfm', 'imas']].copy()
log_ret = np.log(prices / prices.shift(1)).dropna()

# Define crisis event flags
crises = {
    'GFC':        ('2008-09-01', '2009-03-31'),
    'Dilma':      ('2015-01-01', '2016-08-31'),
    'Joesley':    ('2017-05-17', '2017-05-31'),
    'COVID':      ('2020-02-20', '2020-06-30'),
    'Americanas': ('2023-01-11', '2023-06-30'),
    'Fiscal24':   ('2024-11-01', '2025-01-31'),
}
```

---

### Day 5: Descriptive statistics (notebook 02)

**Output tables for whitepaper:**

1. Summary stats: mean return, std dev, Sharpe (vs CDI), max drawdown, skewness, kurtosis — per asset, per regime
2. Unconditional correlation matrix — full sample vs. each regime
3. Crisis period returns heatmap: all assets × all crises

**Key chart (Figure 1):** Regime timeline — Selic rate + EMBI + shaded crisis bands on a single axis.

---

## Week 2 — Core Correlation Analysis

### Notebook 03: Rolling correlations (2–3 days)

```python
window = 252
# Rolling Pearson
roll_corr = log_ret['ibov'].rolling(window).corr(log_ret['imab'])

# Conditional correlation (bottom 10th percentile of equity)
q10 = log_ret['ibov'].quantile(0.10)
stress_corr = log_ret[log_ret['ibov'] <= q10].corr()
```

**Structural breaks:** Use CUSUM test from `statsmodels` — visually identify breaks at GFC, 2015, 2020. Full Bai-Perron can be deferred to thesis phase.

**Key chart (Figure 2):** Rolling 252-day Ibovespa×IMA-B correlation with crisis shading and ρ=0 reference line — the Brazilian equivalent of the IMF's Figure 1.

---

### Notebook 04: DCC-GARCH (2–3 days)

**Two-step approach:**

```python
from arch import arch_model

def fit_garch(series):
    m = arch_model(series * 100, vol='GARCH', p=1, q=1, dist='skewt')
    res = m.fit(disp='off')
    return res.resid / res.conditional_volatility  # standardized residuals

std_resids = {col: fit_garch(log_ret[col])
              for col in ['ibov', 'imab', 'irfm']}

# Step 2: DCC via mvgarch package
# pip install mvgarch
```

**Key outputs:**

- Time series of ρ_t(Ibovespa, IMA-B)
- Scatter: DCC ρ_t vs. EMBI level (expect strong positive relationship)
- Crisis average ρ_t vs. calm average ρ_t

**Key chart (Figure 3):** DCC ρ_t over time with crisis markers. Expected spikes to +0.6/+0.8 during COVID and Americanas.

---

## Week 3 — Advanced Metrics + Portfolio Layer

### Notebook 05: Copula tail dependence (2 days)

**Clayton copula** is the headline result for Brazil (captures lower-tail co-crashes):

```python
from scipy.stats import rankdata
from scipy.optimize import minimize_scalar

def pseudo_obs(data):
    n = len(data)
    return pd.DataFrame({col: rankdata(data[col]) / (n + 1)
                         for col in data.columns})

u = pseudo_obs(log_ret[['ibov', 'imab']])

def clayton_ll(theta, u1, u2):
    if theta <= 0: return 1e10
    log_c = (-(1/theta + 1)) * (np.log(u1) + np.log(u2)) \
            - (1/theta + 2) * np.log(u1**(-theta) + u2**(-theta) - 1) \
            + np.log(1 + theta)
    return -log_c.sum()

res = minimize_scalar(lambda t: clayton_ll(t, u['ibov'], u['imab']),
                      bounds=(0.01, 20), method='bounded')
theta_hat = res.x
lambda_L = 2 ** (-1 / theta_hat)  # Lower tail dependence
```

Compare Gaussian (λ_L=0), Student-t, Clayton, Gumbel via AIC/BIC.

**Key chart (Figure 4):** Scatter of pseudo-observations with Clayton copula density overlay — shows lower-left clustering (co-crashes).

---

### Notebook 06: Portfolio metrics (2 days)

```python
def diversification_ratio(weights, cov):
    w = np.array(weights)
    vols = np.sqrt(np.diag(cov))
    return (w @ vols) / np.sqrt(w @ cov @ w)

def enb(weights, cov):
    w = np.array(weights)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    p = (eigenvectors.T @ w) ** 2 * eigenvalues
    p = p / p.sum()
    return np.exp(-np.sum(p * np.log(p + 1e-10)))

# Rolling both metrics over 252-day windows
```

**PCA:**

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaled = StandardScaler().fit_transform(
    log_ret[['ibov', 'imab', 'irfm', 'imas']])
pca = PCA().fit(scaled)
# PC1 > 70% = diversification has collapsed
```

**Key chart (Figure 5):** Three-panel: rolling DR, rolling ENB, rolling PC1 share. All collapse simultaneously during crises.

---

### Notebook 07: Stress testing (1–2 days)

```python
weights = {'ibov': 0.40, 'imab': 0.40, 'irfm': 0.10, 'imas': 0.10}

# Historical scenario replay
for name, (start, end) in crises.items():
    mask = (log_ret.index >= start) & (log_ret.index <= end)
    rets = log_ret[mask].sum()
    loss = sum(w * rets[a] for a, w in weights.items())
    print(f"{name}: {loss:.2%}")

# Stressed VaR using COVID correlation window
covid_mask = (log_ret.index >= '2020-02-20') & (log_ret.index <= '2020-06-30')
cov_stressed = log_ret[covid_mask][list(weights.keys())].cov() * 252

# Correlation stress → all correlations = 1
def shrink_to_equicorr(cov, alpha):
    vols = np.sqrt(np.diag(cov))
    corr = cov / np.outer(vols, vols)
    stressed_corr = (1 - alpha) * corr + alpha * np.ones_like(corr)
    return np.outer(vols, vols) * stressed_corr
```

**Key table (Table 1):** Portfolio P&L across 6 crisis scenarios × 3 portfolio compositions.

---

## Week 4 — CoVaR + Polish + Whitepaper Prep

### CoVaR (1 day)

```python
import statsmodels.formula.api as smf

df = log_ret[['ibov', 'imab']].copy()
model_05 = smf.quantreg('imab ~ ibov', df).fit(q=0.05)
model_50 = smf.quantreg('imab ~ ibov', df).fit(q=0.50)

var_05 = df['ibov'].quantile(0.05)
var_50 = df['ibov'].quantile(0.50)
delta_covar = model_05.params['ibov'] * (var_05 - var_50)
```

### Robustness checks (1 day)

- Repeat on monthly returns
- Drop 2020 — do results hold without COVID?
- Pre-2015 vs. post-2015 subsamples
- Pearson vs. Spearman divergence test

---

## Priority queue

| Priority | Task                                | Days | Impact                    |
| -------- | ----------------------------------- | ---- | ------------------------- |
| 🔴 P0    | Data pipeline + descriptive stats   | 3    | Unblocks everything       |
| 🔴 P0    | Rolling correlations + crisis chart | 2    | Core visual               |
| 🟡 P1    | DCC-GARCH time-varying ρ            | 2    | Academic credibility      |
| 🟡 P1    | DR + ENB + PCA panel                | 2    | Practitioner contribution |
| 🟡 P1    | Stress test table                   | 1    | Actionable output         |
| 🟢 P2    | Clayton copula + λ_L                | 2    | Methodological depth      |
| 🟢 P2    | CoVaR                               | 1    | Contagion quantification  |
| ⚪ P3    | Bai-Perron breaks                   | 1    | Thesis rigor              |
| ⚪ P3    | IDA debenture layer                 | —    | **Needs ANBIMA data**     |

---

## ANBIMA data decision point

The paper stands without it as a strong **government bond** diversification study. Adding debentures makes it the definitive **multi-asset** study.

**Buy trigger:** If ANBIMA Feed API costs < R$500–1000/month and whitepaper becomes a submission target, buy it for the debenture extension in Week 4.

---

## Whitepaper section map

| Section                            | Notebook | Key output                         |
| ---------------------------------- | -------- | ---------------------------------- |
| 1. Introduction + IMF context      | —        | Manual                             |
| 2. Brazilian market idiosyncrasies | 01, 02   | Regime timeline (Fig 1)            |
| 3. Data and methodology            | 01       | Data table                         |
| 4. Unconditional correlations      | 02       | Correlation heatmaps               |
| 5. Time-varying correlations       | 03, 04   | Rolling corr + DCC (Fig 2, 3)      |
| 6. Tail dependence                 | 05       | Copula scatter + λ_L table (Fig 4) |
| 7. Portfolio impact                | 06       | DR/ENB/PC1 panel (Fig 5)           |
| 8. Stress testing                  | 07       | Crisis scenario table (Table 1)    |
| 9. Debenture extension             | 05, 07   | Needs ANBIMA data                  |
| 10. Conclusions                    | —        | Manual                             |
