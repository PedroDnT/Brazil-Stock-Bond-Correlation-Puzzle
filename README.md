# Brazil Stock-Bond Correlation Study

Replication and extension of the IMF's February 2026 study on stock-bond
diversification breakdown, applied to the Brazilian market (2004–2026).

## Setup

```bash
pip install python-bcb yfinance arch statsmodels scikit-learn scipy pandas matplotlib seaborn pyarrow pyield nbformat nbconvert vaderSentiment python-dotenv
```

## Run

```bash
# 1. Build the master dataset (fetches all data, ~2 min)
python3 src/fetch.py

# 2. Generate all notebooks
python3 build_notebooks.py

# 3. Execute notebooks in order
cd notebooks
jupyter nbconvert --to notebook --execute 01_data.ipynb --output 01_data_executed.ipynb
jupyter nbconvert --to notebook --execute 02_descriptive.ipynb --output 02_descriptive_executed.ipynb
jupyter nbconvert --to notebook --execute 03_rolling_corr.ipynb --output 03_rolling_corr_executed.ipynb
jupyter nbconvert --to notebook --execute 04_dcc_garch.ipynb --output 04_dcc_garch_executed.ipynb --ExecutePreprocessor.timeout=600
jupyter nbconvert --to notebook --execute 05_copula.ipynb --output 05_copula_executed.ipynb --ExecutePreprocessor.timeout=600
jupyter nbconvert --to notebook --execute 06_portfolio_metrics.ipynb --output 06_portfolio_metrics_executed.ipynb --ExecutePreprocessor.timeout=900
jupyter nbconvert --to notebook --execute 07_stress_test.ipynb --output 07_stress_test_executed.ipynb
jupyter nbconvert --to notebook --execute 08_global_macro.ipynb --output 08_global_macro_executed.ipynb

# Or just launch Jupyter and run interactively
jupyter notebook
```

## Data sources (all free)

| Series | Source |
|--------|--------|
| Ibovespa | Yahoo Finance `^BVSP` |
| CDI, Selic, IPCA, PTAX, EMBI proxy | BCB REST API (SGS) |
| NTN-B, LTN, NTN-F, LFT prices | Tesouro Transparente CSV |
| IMA-B, IRF-M ETFs (2019+) | Yahoo Finance |
| Fed, ECB, BoE, RBI, BoJ, BoC rates | FRED API |
| CPI inflation (25 countries) | World Bank API |
| Monetary policy headlines | NewsAPI + VADER |

For debenture (IDA) data: requires ANBIMA Feed API subscription.

**API keys** (optional, for notebook 08): set `FRED_API_KEY` and `NEWS_API_KEY` in `.env`.

## Key results

| Metric | Value |
|--------|-------|
| Ibovespa × NTN-B (full sample) | ρ = +0.073 |
| Ibovespa × LFT (full sample) | ρ = +0.006 |
| Lula Boom correlation | ρ = −0.018 (only negative regime) |
| COVID cycle correlation | ρ = +0.156 (highest) |
| 60/40 VaR calm → COVID | 37.5% → 101.3% (+2.7×) |
| LFT max loss across all crises | ≈ 0% |
| PC1 max variance explained | 68.7% |

## Outputs

All figures and tables are in `outputs/`. See `outputs/whitepaper.md` for the full paper.

## Project structure

```
brazil_study/
├── src/
│   └── fetch.py           # All data ingestion
├── notebooks/
│   ├── 01_data.ipynb      # Data pipeline & validation
│   ├── 02_descriptive.ipynb   # Regime stats & correlation matrices
│   ├── 03_rolling_corr.ipynb  # Rolling correlations & CUSUM
│   ├── 04_dcc_garch.ipynb     # DCC-GARCH time-varying correlation
│   ├── 05_copula.ipynb        # Copula tail dependence
│   ├── 06_portfolio_metrics.ipynb  # DR, ENB, PCA, CoVaR
│   ├── 07_stress_test.ipynb   # Historical scenarios & stressed VaR
│   └── 08_global_macro.ipynb  # Global rates, CPI, sentiment, IMF framework
├── data/
│   ├── raw/               # Cached raw fetches
│   └── processed/         # master_returns.parquet
├── outputs/               # All figures (.png) and tables (.csv)
│   └── whitepaper.md      # Full research paper
└── build_notebooks.py     # Generates all .ipynb files
```
