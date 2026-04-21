# Brazil Stock-Bond Correlation Study

Replication and extension of the IMF's February 2026 study on stock-bond
diversification breakdown, applied to the Brazilian market (2004-2026).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
# 1. Build the master dataset (fetches all data, ~2 min)
python3 src/fetch.py

# 2. Generate all notebooks
python3 scripts/build_notebooks.py

# 3. Execute notebooks in order
cd notebooks
for nb in 01_data 02_descriptive 03_rolling_corr 04_dcc_garch 05_copula 06_portfolio_metrics 07_stress_test 08_global_macro; do
  jupyter nbconvert --to notebook --execute ${nb}.ipynb --output ${nb}_executed.ipynb --ExecutePreprocessor.timeout=900
done

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
| Ibovespa x NTN-B (full sample) | rho = +0.073 |
| Ibovespa x LFT (full sample) | rho = +0.006 |
| Lula Boom correlation | rho = -0.018 (only negative regime) |
| COVID cycle correlation | rho = +0.156 (highest) |
| 60/40 VaR calm -> COVID | 37.5% -> 101.3% (+2.7x) |
| LFT max loss across all crises | ~0% |
| PC1 max variance explained | 68.7% |

## Outputs

All figures and tables are generated in `outputs/`.

## Documentation series

| # | Document | Description |
|---|----------|-------------|
| 1 | [Stock-Bond Diversification](docs/01_stock_bond_diversification.md) | Why Brazil's correlation dynamics diverge from developed markets |
| 2 | [Quantifying Hidden Correlation Risk](docs/02_quantifying_hidden_correlation_risk.md) | Methods to detect, measure, and monitor hidden risk |
| 3 | [Implementation Guide](docs/03_implementation_guide.md) | Python notebook architecture and data pipeline |
| 4 | [Final Paper](docs/04_final_paper.md) | Full research paper |

## Project structure

```
src/
  fetch.py                # All data ingestion (BCB, Yahoo, Tesouro)
notebooks/
  01_data.ipynb           # Data pipeline & validation
  02_descriptive.ipynb    # Regime stats & correlation matrices
  03_rolling_corr.ipynb   # Rolling correlations & CUSUM
  04_dcc_garch.ipynb      # DCC-GARCH time-varying correlation
  05_copula.ipynb         # Copula tail dependence
  06_portfolio_metrics.ipynb  # DR, ENB, PCA, CoVaR
  07_stress_test.ipynb    # Historical scenarios & stressed VaR
  08_global_macro.ipynb   # Global rates, CPI, sentiment, IMF framework
scripts/
  build_notebooks.py      # Generates all .ipynb files
docs/                     # Research series (see above)
config/
  plot_style.py           # Matplotlib rcParams
data/                     # Cached raw & processed data (gitignored)
outputs/                  # Figures (.png) and tables (.csv)
```
