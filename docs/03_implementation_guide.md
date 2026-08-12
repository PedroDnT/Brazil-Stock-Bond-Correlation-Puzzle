# Implementation Guide

How the study is put together, what runs where, and the traps that produce
plausible-looking wrong numbers rather than errors.

This document describes the pipeline **as it exists**. For what the study found, read
[the paper](04_final_paper.md); for open work, read [Known limitations and open
work](#known-limitations-and-open-work) at the end of this file.

---

## Architecture

```
project/
├── data/
│   ├── raw/          # BCB SGS, IPEADATA, Tesouro, FRED — cached downloads
│   └── processed/    # master_returns.csv, global_panel.csv
├── src/
│   ├── fetch.py       # Brazilian ingestion + FRED helper + validate_master()
│   ├── global_data.py # Matched cross-country panel (FRED, keyless)
│   └── metrics.py     # Inference, Forbes-Rigobon, copulas, DCC, DR/ENB/PC1, VaR
├── scripts/
│   ├── run_analysis.py         # Sections 4-8 + 6.1 -> outputs/tbl_*.csv
│   ├── run_global_analysis.py  # Section 9          -> outputs/tbl_global_*.csv
│   └── build_notebooks.py      # generates notebooks 01-08
├── notebooks/         # generated; figures + outputs/nb_tbl_*.csv
├── tests/
│   ├── test_metrics.py           # 31 tests: the estimators
│   ├── test_global_data.py       # 17 tests: cross-country bond construction
│   ├── test_sovereign.py         # 12 tests: sovereign series + the ffill guard
│   ├── test_docs.py              # 18 tests: docs match the pipeline that exists
│   └── test_paper_consistency.py # 91 tests: paper + README numbers vs outputs/
├── .github/workflows/tests.yml   # CI: the 78 network-free tests
├── site/              # static web version of the paper
└── outputs/           # figures (.png) and tables (.csv), gitignored
```

`data/` and `outputs/` are gitignored and regenerate from source.

## Running it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python3 src/fetch.py                     # ~3 min, prints validation
python3 scripts/run_analysis.py
python3 scripts/run_global_analysis.py
python3 -m pytest tests/ -q              # 169 tests
```

No API key is required at any stage.

## Two design decisions worth knowing

**Estimators live in `src/metrics.py`, not in the notebooks.** A copula density with a
wrong exponent does not raise an error — it returns a likelihood for a function that is
not a density, and AIC then selects it over the correct families. The same applies to a
DCC likelihood, a VaR horizon, or a heteroskedasticity correction: these fail silently
and produce plausible numbers. One importable module makes them unit-testable, and
`tests/test_metrics.py` asserts the properties that catch this class of error —
densities integrating to 1, copulas collapsing to independence at their independence
parameter, the DCC recovering known simulated parameters, and VaR never exceeding 100%
of capital for a long-only unlevered portfolio.

**Each output file has exactly one writer.** `scripts/run_analysis.py` writes
`outputs/tbl_*.csv`; the notebooks write `outputs/nb_tbl_*.csv`. They previously wrote
the same filenames with different schemas, so whichever ran last won.

## Data sources

Verified against the live APIs, not copied from a catalogue.

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
| ICE BofA Latin America OAS | FRED | `BAMLEMRLCRPILAOAS` | percent, Aug 2023– |
| US 10y Treasury | FRED | `DGS10` | percent per annum |
| Equity index, 4 advanced economies | FRED (OECD) | `SPASTT01<CC>M661N` | index, monthly |
| 10y government yield, 4 advanced economies | FRED (OECD) | `IRLTLT01<CC>M156N` | percent, monthly |

### Traps

Each of these silently produces plausible numbers rather than an error. All were live
defects in this repo at some point.

- **SGS 12 is a daily rate, not annual.** Compounding it as annual and taking a 252nd
  root understates the accrual by roughly two orders of magnitude, and propagates into
  any Sharpe ratio computed against it.
- **EMBI is not in the SGS system at all.** SGS code 21619, which reads plausibly as a
  spread, is the EUR/BRL exchange rate. Use IPEADATA.
- **EMBI stops in July 2024 and must not be forward-filled past that.** Filling it held
  the series at a constant 228 bps for 501 trading days, across the whole Fiscal24
  window. `validate_master` now fails on a constant tail; see Section 6.1 of the paper.
- **The crisis sentinel is `"Calm"`, not `"None"`.** `pandas.read_csv` reads the literal
  string `"None"` as NaN, which silently empties any `crisis == "None"` mask on reload.
- **The sample end is pinned** at `SAMPLE_END = 2026-06-30` in `src/fetch.py`. Every
  source publishes continuously, so without a fixed cutoff the paper's numbers drift
  daily and the reproducibility claim is false.
- **Ibovespa is not in SGS as a daily series either.** IPEADATA has it back to 1993.

Validate any series you add against a known historical value before using it: the
Ibovespa should print 73,517 on 2008-05-20 and EMBI+ ~2,443 bps at the September 2002
peak. `validate_master()` runs range and coverage checks on every column and is called
at the end of `src/fetch.py`.

## Bond construction

The Tesouro file gives unit prices, not returns. Two things must be handled or the
series is wrong in ways that survive inspection:

- **Rolls.** Take the return of the bond selected on the *previous* day. Differencing
  across a change of maturity prices one bond against a different one.
- **Coupons.** An untreated NTN-B coupon looks like a one-day loss of ~2.9% of VNA,
  about 42 times over the sample. `src/fetch.py` recovers the face/VNA from the bond's
  own quoted yield through a du/252 cashflow pricer, so no external VNA series is
  needed. The pricer is validated on NTN-F, whose face is known to be 1,000 — it
  recovers 1,001.6.

Target tenors: NTN-B 5y, LTN 2y, NTN-F 10y, LFT 1y and longest-outstanding.

**LFT.** Built from observed Tesouro Selic prices. Defining the LFT return as compounded
CDI makes "the LFT never loses money" true by construction and therefore untestable —
and it hides that long LFTs drew down 1.33% in October 2020.

**Cross-country bonds.** For the four advanced economies, total returns come from yields
via carry, duration and convexity, applied identically everywhere. Brazil is the control:
it is the one country where a unit-price construction also exists, and the two agree at
ρ = 0.931 with the headline correlation differing by 0.014. That is what licenses
applying the yield-based method to the others.

## Notebooks

Generated by `scripts/build_notebooks.py` — edit the generator, not the `.ipynb`. All
eight execute without an API key and produce 27 figures.

| # | Notebook | Produces |
|---|----------|----------|
| 01 | `01_data` | pipeline and construction validation |
| 02 | `02_descriptive` | regime stats, correlation matrices |
| 03 | `03_rolling_corr` | rolling correlations, CUSUM, frequency, Forbes-Rigobon |
| 04 | `04_dcc_garch` | DCC-GARCH with standard errors; DCC vs sovereign spread |
| 05 | `05_copula` | copula fits and empirical tail dependence |
| 06 | `06_portfolio_metrics` | DR, ENB, PCA, CoVaR |
| 07 | `07_stress_test` | historical scenarios, stressed VaR |
| 08 | `08_global_macro` | matched cross-country panel; the convergence test |

## How the paper is kept honest

`tests/test_paper_consistency.py` reads `docs/04_final_paper.md` and `README.md` and
asserts that every number quoted in the prose matches the regenerated table it came
from. The failure mode it exists to prevent is a paper that drifts away from its own
code: a table is regenerated, the prose is not updated, and the two disagree silently.
That is not hypothetical — it caught four stale README figures when the sample was
pinned.

New consistency tests should be **mutation-checked**: change the number in the document,
confirm exactly one test fails, change it back. A test asserting `A in text or B in
text` can pass on B while A rots, which is how one of these was written wrong the first
time.

---

## Known limitations and open work

Ordered by how much they would change the paper's claims. The paper's own Limitations
paragraph (Section 10) covers the same ground from the reader's side; this list is the
maintainer's view, including repo-level items that do not belong in a paper.

### Substantive — these bound what the study can claim

**The sovereign-credit channel has no single measure spanning the sample.** EMBI+ Brazil
ends July 2024; the ICE BofA hard-currency spread that replaces it begins August 2023 and
is regional rather than Brazil-specific; the yield differential that does span the sample
is not a credit spread (Section 6.1). A Brazil 5-year CDS series would close this, and is
not available free. Until then the channel is measured in two pieces with a documented
seam.

**Regime boundaries are imposed, not estimated.** The six macro regimes are drawn from
political economy. A Markov-switching estimate is the natural extension — and given that
no regime is statistically distinguishable from any other (Table 4), it might well find
no distinct regimes, which would itself be a result.

**The cross-country break date is imposed.** 2019-12-31 comes from the IMF note rather
than from the data. Estimating it would make the convergence test self-contained.

**The convergence result is directionally unanimous and not established.** Four of four
advanced economies narrowed toward Brazil; only Germany is significant. Two things bind
here and the paper lists both: the post-break window is 78 months, which is short for a
correlation estimate, and four correlated bond markets are closer to one or two
independent observations than to four. Waiting lengthens the window; only more countries
fixes the second.

**Bond series come from Tesouro Direto retail reference prices**, not institutional
secondary-market quotes. Realised tenor deviates from target by 0.74 years (NTN-B) and
1.11 years (NTN-F). ANBIMA secondary quotes would be the upgrade; they are gated.

**Three claims from an earlier draft remain unreproduced** and are deliberately not
reported: Brazil's real policy rate ranked against the G7, a pre/post-2020 real-rate
convergence, and correlations between news sentiment and rate changes. They rested on a
notebook that had never been executed. Section 9.5 says so explicitly.

### Methodological caveats already handled in the prose

- `a` in the DCC is weakly identified (t = 1.77 for NTN-B), so the conditional
  correlation *levels* are sample-sensitive. Section 7 reports the ordering as the
  stable result and lets Forbes-Rigobon carry the crisis claim.
- The Joesley window is 11 trading days. Any statistic from it is imprecise, and it is
  excluded from the 5×5 stressed-VaR table for that reason.

### Repo-level

**The 84 consistency tests do not run in CI.** They read `outputs/`, which is gitignored
and regenerates from live BCB, IPEADATA, Tesouro and FRED calls — a CI job that goes red
whenever IPEADATA returns a 503. Closing this properly means committing a pinned data
snapshot and running the full suite against it. Until then the guard against prose drift
runs only when someone runs it locally.

**Documents 01 and 02 predate the empirical work.** They are literature and method
surveys whose stated expectations were partly falsified by the results; each now carries
a header saying so. They have not been rewritten, because their value is the sourcing and
the method exposition, not their predictions.
