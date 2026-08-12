# Stock-bond diversification in Brazil: a different beast entirely

> **Status: background, written before the empirical work.** This is a literature and
> data-source survey that motivated the study. Its sourcing and framing stand; several of
> its *expectations* were tested and revised by the results, and it has been left as
> written rather than retrofitted. Where it and [the paper](04_final_paper.md) disagree,
> the paper is the finding. Specifically:
>
> | This document expected | The study found |
> |---|---|
> | Stocks and bonds "overwhelmingly sell off together during crises" | Most crisis correlation spikes are volatility artefacts; after the Forbes-Rigobon adjustment only Joesley Day shows contagion (§6) |
> | Bonds "almost always fall with stocks during crises" | Bonds cushioned four of six episodes — NTN-B returned **+5.8pp** over CDI during the GFC (§8.2) |
> | Sustained negative correlation in 2003–2007 and 2016–2019 | **No regime** has a negative correlation; the lowest is Lula Boom at +0.037, and no regime is statistically distinguishable from any other (§4.2) |
> | The IMF finding represents convergence toward Brazil | Directionally unanimous (4 of 4) but significant in only one country; "defensible as a direction and premature as a finding" (§9.3) |

Brazil has lived with persistently positive stock-bond correlations for decades — driven by fiscal dominance, sovereign credit risk, and the unique structure of its domestic debt market. While the IMF's February 2026 study documents diversification breakdown as a post-2020 phenomenon for the US, Germany, Japan, and the UK, this report replicates and extends the IMF framework for Brazil, identifies data sources for quantitative implementation, documents regime-specific correlation dynamics, and explains why the country's floating-rate LFT bonds, commodity-heavy equity index, and recurring fiscal crises create a fundamentally different diversification landscape.

---

## The IMF's finding arrived late for emerging markets

The IMF blog post by Tobias Adrian, Johannes Kramer, and Sheheryar Malik (February 18, 2026) identifies **end-2019 as the turning point** when the historically negative stock-bond correlation in advanced economies flipped positive. Their analysis traces standardized expected returns for stocks and bonds against the VIX across the US, Germany, Japan, and the UK, showing that from 2000–2019, rising market stress pushed equity expected returns up (prices down) while simultaneously pushing bond expected returns down (prices up — the classic safe-haven effect). After 2020, both asset classes began selling off concurrently during stress episodes.

The IMF attributes this regime shift to three structural forces: pandemic-era supply shocks fueling inflation, expanding fiscal deficits increasing bond supply, and central bank balance-sheet runoff shifting absorption to price-sensitive private investors. The blog recommends fiscal discipline, credible monetary policy, and incorporating correlation breakdown scenarios into stress tests.

Critically, the IMF study is a policy-oriented blog post, not a formal working paper. It does not specify exact econometric methods — no DCC-GARCH parameters, copula estimates, or quantile regression coefficients are reported. The underlying methodology appears descriptive and visual rather than formally econometric, though the authors' broader research (particularly Malik's work on quantile regressions and Adrian's financial conditions indices) suggests more sophisticated methods may underlie the analysis. The October 2025 GFSR chapter "Shifting Ground Beneath the Calm" contains supporting material on bond supply dynamics.

For Brazil, this finding is unremarkable. **Academic evidence consistently shows that emerging market stock-bond correlations are predominantly positive** — Dimic et al. (2016) found that all ten emerging markets in their sample (including Brazil) exhibited significantly positive unconditional stock-bond correlations. Portelli and Roncalli (2024) explicitly note: "When looking at emerging markets, we typically expect to see a positive correlation between stocks and bonds, as credit risk affects both equity and debt markets. This pattern is evident in Brazil, South Africa, and Turkey."

---

## Why Brazil's correlation dynamics diverge from developed markets

### Fiscal dominance: the correlation's master switch

The single most important driver of positive stock-bond correlation in Brazil is **fiscal dominance** — the condition where fiscal stress overwhelms monetary policy transmission. Blanchard's seminal 2004 NBER paper formalized this for Brazil: when debt is high, higher interest rates increase default probability rather than attracting capital, causing the currency to depreciate (the opposite of textbook predictions) and both stocks and bonds to sell off simultaneously.

Research using Markov-Switching VAR models identifies distinct regimes: **monetary dominance prevailed from 2003–2013** (negative stock-bond correlation, effective monetary policy), followed by **fiscal dominance from 2013 onward** (with a brief return to monetary dominance around 2016–2019). The current cycle, with Selic at **15.00%** as of January 2026 — the highest since July 2006 — yet BRL weakening and long-end yields elevated, strongly suggests fiscal dominance has returned.

Under monetary dominance, rate hikes attract capital, strengthen the currency, and cool the economy — stocks and bonds diverge (negative correlation). Under fiscal dominance, rate hikes raise debt service costs (~43% of federal debt is Selic-indexed via LFTs), increase default risk, weaken the currency, and push up inflation expectations — stocks and bonds move in lockstep (positive correlation). This regime-switching behavior is **the single most important distinction** between Brazil and the advanced economies studied by the IMF.

### The LFT puzzle: a bond that doesn't behave like a bond

Brazil's domestic debt market contains an instrument with no developed-market equivalent: the **LFT (Letra Financeira do Tesouro)**, now branded Tesouro Selic. This floating-rate bond accrues the daily Selic overnight rate and has **near-zero duration** — its mark-to-market price barely moves regardless of interest rate changes. LFTs and Selic-indexed repo agreements constitute roughly **43% of total federal public debt**.

This fundamentally breaks the standard stock-bond diversification model. In developed markets, government bonds rally during equity selloffs because falling growth expectations drive yields lower. LFTs cannot rally — they simply accrue the policy rate. During stress, LFTs provide **liquidity preservation** (no capital loss) but no **capital appreciation** (no "flight to quality" price gain). The IMA-S index (tracking LFTs) exhibits near-zero volatility and near-zero correlation with Ibovespa — offering genuine diversification through non-correlation rather than negative correlation.

LFTs can, however, trade at small discounts ("deságio") during extreme stress when sovereign creditworthiness itself is questioned, as occurred briefly in 2002. The existence of LFTs also creates a perverse monetary transmission channel: raising Selic to fight inflation directly increases government debt service on 43% of outstanding debt, potentially worsening fiscal dynamics and reinforcing positive stock-bond correlation.

### Commodity exposure and the Ibovespa's dual identity

Ibovespa's composition creates a **structural divergence** between equity and bond behavior during commodity cycles. Vale (iron ore) and Petrobras (oil) together account for **25–28%** of the index, with the broader materials and energy sectors representing 30–35%. Financials add another 25–30% (Itaú, Bradesco, Banco do Brasil). This commodity tilt means Ibovespa can rally on global commodity booms even while domestic fiscal conditions deteriorate and bond yields rise — producing temporary negative correlation. Conversely, commodity busts reduce both corporate earnings and government commodity royalties, creating fiscal pressure that hits bonds simultaneously — reinforcing positive correlation.

### The triple whammy: when everything falls at once

In advanced economies, government bonds act as a shock absorber during equity selloffs. In Brazil, global risk-off episodes trigger simultaneous selling of equities, bonds, _and_ the currency — the "triple whammy." This occurs because foreign investors (roughly 40% of B3 trading volume) exit all Brazilian assets simultaneously, and BRL depreciation feeds directly into fiscal stress through dollar-linked liabilities. Every major crisis episode since 1999 has exhibited this pattern to some degree.

---

## Crisis-by-crisis correlation evidence

The following table summarizes correlation behavior across Brazil's major financial stress episodes, documenting the near-universal pattern of positive stock-bond co-movement during crises:

| Episode                  | Ibovespa                                                           | Bond yields/spreads                    | BRL/USD                | Correlation sign                                                  | Primary driver                                  |
| ------------------------ | ------------------------------------------------------------------ | -------------------------------------- | ---------------------- | ----------------------------------------------------------------- | ----------------------------------------------- |
| **1999 devaluation**     | Initially −15% in one day (Oct 1997), then rallied +59% post-float | Selic peaked at 42%; spreads surged    | −66% in 45 days        | Mixed: positive on entry, then negative                           | Balance of payments / currency peg collapse     |
| **2002 Lula election**   | −33% (Apr–Oct)                                                     | Benchmark bonds fell to 49¢ on dollar  | −40% in 6 months       | **Strong positive**                                               | Sovereign default fear / political risk         |
| **2008 GFC**             | −60% peak-to-trough                                                | Spreads widened sharply                | −43% (Sep–Dec)         | **Strong positive**                                               | Global risk-off / capital flight                |
| **2013 taper tantrum**   | −14%                                                               | Yields +250 bps                        | −13.5%                 | **Positive**                                                      | US monetary policy spillover / Fragile Five     |
| **2015–16 Dilma crisis** | Fell to 37,000 (Jan 2016)                                          | Junk downgrade; long yields elevated   | Significant weakness   | **Positive** (downturn); then negative (recovery on reform hopes) | Fiscal credibility collapse / impeachment       |
| **2017 Joesley Day**     | −8.8% single day                                                   | CDS +15% single day                    | −7.5% single day       | **Strong positive**                                               | Domestic political shock                        |
| **2020 COVID**           | −47% peak-to-trough                                                | Yields initially spiked                | Breached 5.00/USD      | **Strong positive**                                               | Pandemic / global risk-off                      |
| **2022–24 fiscal**       | Under pressure                                                     | Long-end steepened; Selic hiked to 15% | Weakened progressively | **Positive**                                                      | Fiscal framework uncertainty / PEC da Transição |

The only period of sustained negative stock-bond correlation (diversification working as intended) was **2003–2007 and 2016–2019** — both characterized by reform optimism, falling Selic, improving fiscal metrics, and monetary dominance. The Lula boom (2003–2007) saw Ibovespa surge from ~10,000 to ~73,000 while bond yields compressed dramatically. The Temer reform era (2016–2019) produced a similar dynamic after the spending ceiling constitutional amendment.

---

## Proposed methodology: adapting the IMF approach for Brazil

### Asset class mapping

The IMF analyzes stock-bond correlations using broad equity and government bond indices. For Brazil, the analysis should decompose bonds by type, since each instrument class carries fundamentally different risk characteristics:

- **Equities**: Ibovespa total return index (BCB SGS code 7 for price index; compute returns from daily closes)
- **Prefixed nominal bonds**: IRF-M index (LTN + NTN-F total return) — closest equivalent to developed-market government bonds
- **Inflation-linked bonds**: IMA-B index (NTN-B total return) — comparable to US TIPS; split into IMA-B 5 (short) and IMA-B 5+ (long)
- **Floating-rate bonds**: IMA-S index (LFT total return) — uniquely Brazilian, near-zero duration
- **Broad government bonds**: IMA-Geral (all government bonds weighted by market value)

Running the analysis separately for each bond type reveals whether diversification failure is universal or concentrated in specific instruments.

### Statistical methods to implement

**Rolling correlations.** Compute 12-month and 36-month rolling Pearson and Spearman rank correlations between monthly returns on Ibovespa and each IMA sub-index. Molenaar et al. (2024) use 36-month rolling Spearman correlations as their baseline; replicating this permits direct comparison with their international results.

**Conditional tail analysis.** Calculate the average bond return conditional on equity returns falling in the bottom 10th and 20th percentiles. This directly tests whether bonds provide diversification "when it matters most" — during equity selloffs. Compute this separately for each bond type.

**DCC-GARCH estimation.** Fit a Dynamic Conditional Correlation GARCH(1,1) model to daily returns of Ibovespa and each bond index. This captures time-varying correlations while accounting for volatility clustering — essential given Brazil's GARCH-style volatility dynamics. The `arch` Python package or R's `rmgarch` support this.

**Regime-switching models.** Estimate a bivariate Markov-Switching model (following Chen 2009 and the FGV fiscal dominance literature) to formally identify correlation regimes. Define states as: (i) monetary dominance / negative correlation, (ii) fiscal dominance / positive correlation, (iii) crisis / high positive correlation. Map these to economic regimes.

**Standardized returns vs. VIX replication.** Directly replicate the IMF's Chart 3 by plotting standardized expected returns (or realized returns) for Ibovespa and IRF-M/IMA-B against VIX levels, comparing pre-2020 vs. post-2020 patterns. For Brazil, also condition on domestic stress indicators (EMBI+ Brazil spread, CDS spread, implied volatility from Ibovespa options).

**Copula-based tail dependence.** Fit Student's t and Gumbel copulas to the joint distribution of stock-bond returns to estimate upper and lower tail dependence coefficients. Righi and Ceretta (2012) demonstrate that Student's t copulas fit Brazilian data better than Gaussian, indicating elevated tail dependence.

### Regime definitions

Seven distinct regimes should be analyzed separately:

1. **Pre-Real Plan hyperinflation** (pre-July 1994): Not meaningful for correlation analysis due to price distortions
2. **Post-Real stabilization / high-rate era** (1995–2002): Currency crises, Selic above 20%, fiscal stress
3. **Lula boom / monetary dominance** (2003–2007): Reforms, commodity super-cycle, falling rates
4. **GFC and aftermath** (2008–2012): External shock, V-shaped recovery, gradual rate normalization
5. **Dilma fiscal deterioration** (2013–2016): Fiscal dominance return, recession, impeachment
6. **Reform era** (2016–2019): Spending ceiling, pension reform expectations, rate cuts
7. **COVID and post-COVID cycle** (2020–present): Pandemic shock, emergency easing, inflation spike, aggressive tightening to 15%

---

## Brazil vs. developed markets: expected comparison

Based on the literature synthesis, the following comparison table summarizes the key divergences:

| Dimension                                            | IMF finding (US, Germany, Japan, UK)               | Expected Brazil finding                                                                     |
| ---------------------------------------------------- | -------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| **Unconditional stock-bond correlation (2000–2019)** | Negative (≈ −0.2 to −0.4)                          | Predominantly positive (≈ +0.1 to +0.3), with brief negative episodes during reform periods |
| **Post-2020 correlation**                            | Turned positive (new regime)                       | Remained positive (no regime change — already positive)                                     |
| **Tail dependence during equity selloffs**           | Bonds rallied pre-2020; bonds now fall with stocks | Bonds almost always fall with stocks during crises (triple whammy)                          |
| **Primary correlation driver**                       | Inflation vs. growth shock type                    | Fiscal dominance vs. monetary dominance regime                                              |
| **Role of inflation**                                | High inflation → positive correlation              | High inflation → positive correlation, but fiscal channel dominates                         |
| **Explanatory power of macro variables**             | High R² for inflation/real rates                   | Low R² (< 0.20 per Molenaar et al. 2024); Brazil-specific factors dominate                  |
| **Bond types that diversify**                        | Government bonds (pre-2020 only)                   | LFTs provide non-correlation; nominal/inflation-linked bonds fail during crises             |
| **Alternative safe havens**                          | Gold, Swiss franc                                  | USD cash, LFTs, offshore assets                                                             |
| **Structural vulnerability**                         | Post-2020 fiscal expansion, QT                     | Permanent structural feature since at least the 1990s                                       |

The critical insight is that the IMF's finding represents a **convergence** of advanced economies toward the correlation regime that Brazil (and other EMs) have long experienced. What is new for the G4 is not new for Brazil.

---

## Complete data access guide for Python implementation

> **Correction.** The snippets in this section were drafted from source catalogues
> and several of the identifiers do not resolve against the live APIs. Ibovespa and
> EMBI+ Brazil are not available as daily BCB SGS series (both are free from
> IPEADATA), the IMA index codes do not resolve, and SGS 11/12 are percent-per-day
> rather than annual rates. The verified source list is in
> `docs/03_implementation_guide.md`; the working implementation is `src/fetch.py`,
> which validates each series against known historical values on every rebuild.

### Core data retrieval using python-bcb

```python
from bcb import sgs
import pandas as pd

# Fetch all core series
data = sgs.get({
    'Ibovespa': 7,           # Daily index level (from 1968)
    'Selic_daily': 11,       # Daily Selic rate (from 1986)
    'Selic_target': 432,     # COPOM target rate
    'IPCA': 433,             # Monthly inflation (from 1980)
    'USDBRL': 1,             # Daily BRL/USD exchange rate (from 1984)
    'CDI': 12,               # Daily CDI rate
    'IMA_B': 12466,          # IMA-B index (NTN-B total return, from ~2003)
    'IMA_B5': 12467,         # IMA-B 5 (short NTN-B)
    'IMA_B5_plus': 12468,    # IMA-B 5+ (long NTN-B)
}, start='2003-01-01')
```

### Additional IMA indices (IRF-M and IMA-S)

IRF-M (prefixed bonds) and IMA-S (LFT/Selic-linked) SGS codes should be searched at `https://www3.bcb.gov.br/sgspub/` — search for "IMA" to locate all available codes. Alternatively, the `anbimapi` Python package scrapes these directly:

```python
from anbimapi import get_ima_index
irfm = get_ima_index("IRF-M")      # Prefixed bonds total return
ima_s = get_ima_index("IMA-S")     # Selic-linked bonds total return
ima_geral = get_ima_index("IMA-GERAL")  # Broad government bond index
```

### Individual bond yields from Tesouro Transparente

Daily yields and prices for all government bond types (LTN, NTN-F, NTN-B, LFT) from December 2004 onward:

```python
url = ("https://www.tesourotransparente.gov.br/ckan/dataset/"
       "df56aa42-484a-4a59-8184-7676580c81e3/resource/"
       "796d2059-14e9-44e3-80c9-2d9e30b405c1/download/"
       "precotaxatesourodireto.csv")
bonds = pd.read_csv(url, sep=";", decimal=",")
```

### DI futures (yield curve)

```python
from pyield import futures
di1 = futures.data("DI1", "23-08-2024")  # DI1 settlement prices
```

### Key BCB SGS API pattern

```
https://api.bcb.gov.br/dados/serie/bcdata.sgs.{CODE}/dados?formato=json
  &dataInicial={DD/MM/YYYY}&dataFinal={DD/MM/YYYY}
```

Note: Since March 2025, queries are limited to 10-year windows; use multiple paginated requests for full history.

### Summary of data availability

| Series                  | Earliest date | Best source          | SGS code    |
| ----------------------- | ------------- | -------------------- | ----------- |
| Ibovespa                | 1968          | BCB SGS              | 7           |
| Selic (daily)           | 1986          | BCB SGS              | 11          |
| IPCA                    | 1980          | BCB SGS              | 433         |
| BRL/USD                 | 1984          | BCB SGS              | 1           |
| IMA-B (NTN-B index)     | ~2003         | BCB SGS              | 12466       |
| IMA-B 5 / IMA-B 5+      | ~2003         | BCB SGS              | 12467/12468 |
| IRF-M / IMA-S           | ~2003         | ANBIMA / anbimapi    | —           |
| Individual bond yields  | Dec 2004      | Tesouro Transparente | CSV         |
| DI futures              | 2000          | B3 / PYield          | —           |
| IDA (debenture spreads) | ~2017         | ANBIMA Feed          | —           |

---

## Investment implications for Brazilian portfolios

### The LFT is Brazil's true diversifier — but not for the reasons you'd expect

In developed markets, the diversification argument for bonds rests on **negative correlation**: bonds rally when stocks fall, cushioning portfolio losses. In Brazil, fixed-rate bonds (IRF-M) and inflation-linked bonds (IMA-B) fail this test during every major crisis — they sell off alongside equities. The LFT, paradoxically, provides diversification through **non-correlation** rather than negative correlation. Its near-zero duration means it neither gains nor loses during stress; it simply accrues the policy rate. In a market where "not losing money" during a crisis is the realistic benchmark for diversification, the LFT is the most reliable domestic instrument.

However, the LFT's diversification benefit is **bounded**: it cannot produce the portfolio-saving positive returns during equity crashes that US Treasuries historically delivered (pre-2020). For a Brazilian investor seeking genuine negative correlation with Ibovespa, the options are limited to **USD-denominated assets** (the BRL weakens during equity selloffs, so dollar exposure benefits) and **gold**.

### Regime awareness should drive allocation

The most actionable finding is that **the fiscal dominance regime determines whether any bond class diversifies**. During monetary dominance (2003–2007, 2016–2019), longer-duration bonds (IMA-B 5+, IRF-M) can provide genuine diversification as rates fall and equities rally on reform optimism — but these are periods of positive returns for both asset classes (the "everyone wins" regime), not stress diversification. During fiscal dominance, even inflation-linked bonds fail: NTN-B yields rise (prices fall) alongside equities because fiscal stress raises real rates and default risk simultaneously.

A practical allocation framework for Brazilian portfolios should incorporate a fiscal dominance indicator (debt-to-GDP trajectory, primary balance, EMBI+ spread, CDS spread) to dynamically adjust between: (a) longer-duration bonds during monetary dominance, and (b) LFTs plus USD exposure during fiscal dominance.

### NTN-B behavior: not equivalent to US TIPS

Inflation-linked bonds (NTN-B/IMA-B) in Brazil behave differently from US TIPS during equity stress. In the US, TIPS provide moderate diversification because real yields tend to fall during recessions. In Brazil, real yields on NTN-Bs **rise** during fiscal crises because the sovereign risk premium embedded in real yields increases — even though inflation expectations may simultaneously rise. The IMA-B 5+ (long-dated NTN-B) has exhibited some of the largest drawdowns of any government bond index during fiscal stress episodes, making it a poor crisis diversifier despite its inflation protection during normal times.

---

## Conclusion: what this means and where to go from here

Brazil's stock-bond correlation story is not the story of a post-2020 regime change — it is the story of a market where **sovereign credit risk permeates every asset class**, and where the fiscal-monetary policy interaction determines whether any diversification is possible. The IMF's finding that advanced-economy bonds have lost their hedging properties is, for Brazilian investors, a description of normalcy rather than novelty.

Three novel insights emerge from this analysis. First, the **fiscal dominance regime** is the master variable — not inflation or growth shocks as in developed markets. Molenaar et al. (2024) confirm this quantitatively: standard macro variables explain less than 20% of Brazil's stock-bond correlation variance, compared to much higher explanatory power in G7 countries. Second, the **LFT's near-zero duration** creates a unique diversification instrument that has no developed-market equivalent — it offers capital preservation but not capital appreciation during stress, fundamentally altering the portfolio construction problem. Third, the **commodity composition of Ibovespa** introduces an additional dimension of correlation complexity, as external commodity shocks and domestic fiscal shocks operate through different channels with potentially opposite effects on stock-bond co-movement.

For implementation, the recommended analytical sample runs from 2003 to present (when IMA indices become available and Brazil enters a more stable institutional framework post-Lula's orthodox pivot). The complete Python data pipeline using `python-bcb`, `anbimapi`, and Tesouro Transparente CSVs provides daily data for all required asset classes. The regime-switching DCC-GARCH framework, complemented by copula-based tail dependence analysis, will most effectively capture Brazil's time-varying, regime-dependent correlation structure — and demonstrate convincingly that what is new for the IMF's advanced economies has been Brazil's persistent reality.
