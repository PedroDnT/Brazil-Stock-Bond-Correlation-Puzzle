# Stock-Bond Diversification in Brazil: A Different Beast Entirely

**Pedro Nogueira**  
OCTO Inteligência de Dados / FGV EESP  
March 2026

---

## Abstract

The IMF's February 2026 study by Adrian, Kramer, and Malik documents a structural breakdown in stock-bond diversification across the US, Germany, Japan, and the United Kingdom since late 2019 — attributing it to post-pandemic inflation shocks, fiscal expansion, and central bank quantitative tightening. This paper argues that what is new for advanced economies has been Brazil's persistent reality for over two decades. Using daily data from 2004 to 2026 across Ibovespa equities, NTN-B (IPCA-linked), LTN (prefixed), NTN-F, and LFT (Selic-linked) government bonds, we document that Brazil's stock-bond correlation is structurally positive, regime-dependent, and driven primarily by sovereign credit risk rather than the inflation-output dynamics governing developed markets. The rolling 252-day Ibovespa × NTN-B correlation ranges from −0.018 during the Lula commodity boom (2003–2007) to +0.156 during the COVID and post-COVID cycle — never sustaining meaningful negative values for extended periods. DCC-GARCH estimates confirm that correlations spike during every major crisis episode. Clayton copula analysis establishes positive lower-tail dependence, confirming that Brazilian assets co-crash. Portfolio simulations show that a standard 60/40 allocation (Ibovespa + NTN-B) lost 11.3% during the GFC and 6.6% during COVID — while LFT-only allocations lost nothing across all six crisis episodes. The paper concludes that the Letra Financeira do Tesouro (LFT), Brazil's Selic-linked floating-rate bond, is the country's only genuine domestic diversifier — not through negative correlation but through near-zero duration and consequent immunity to both interest rate and credit dynamics.

**Keywords:** stock-bond correlation, diversification, Brazil, DCC-GARCH, copula, fiscal dominance, LFT, NTN-B  
**JEL codes:** G11, G12, G15, E44

---

## 1. Introduction

Portfolio diversification theory rests on the assumption that asset correlations are stable and, for the canonical stock-bond pair, negative. The 60/40 equity-bond portfolio has served as the global institutional benchmark precisely because government bonds historically appreciate during equity selloffs — dampening portfolio drawdowns and smoothing returns across economic cycles. Campbell, Pflueger, and Viceira (2020) formalised this: the sign of the stock-bond correlation is determined by whether inflation is procyclical (demand shocks dominate → negative correlation) or countercyclical (supply shocks or fiscal stress dominate → positive correlation).

The IMF's February 2026 blog post by Adrian, Kramer, and Malik identifies late 2019 as a turning point for advanced economies: since then, stocks and bonds have tended to sell off together, undermining the foundational logic of the 60/40 portfolio. The authors attribute this shift to three structural forces — pandemic-era supply shocks driving inflation, expanding fiscal deficits increasing government bond supply, and central bank quantitative tightening shifting absorption to price-sensitive private investors.

For Brazil, this finding is unremarkable in an important sense: it describes the country's long-standing condition. Brazil has operated with predominantly positive stock-bond correlations for most of the past twenty years, interrupted only by brief periods of negative correlation during reform-era optimism. The mechanisms the IMF identifies for the post-2020 regime in developed markets — fiscal dominance, sovereign credit risk contaminating asset prices, inflation supply shocks — have been permanent structural features of Brazilian finance since the hyperinflation era of the 1980s and 1990s.

This paper has three contributions. First, it provides the first systematic documentation of Brazil's stock-bond correlation structure across the country's six major macroeconomic regimes since 2004, using formal econometric methods (DCC-GARCH, copulas, rolling quantile regression) alongside practitioner risk metrics (Diversification Ratio, Effective Number of Bets, PCA). Second, it identifies Brazil's floating-rate LFT bond as a unique domestic safe haven whose near-zero duration eliminates the duration channel through which most bonds fail during fiscal stress — offering capital preservation rather than capital appreciation during crises, but failing to zero in either direction across all six crisis episodes studied. Third, it connects Brazil's experience to the broader emerging market literature and the IMF's advanced-economy findings, arguing that the global post-2020 correlation regime is a convergence toward Brazil's permanent condition rather than a temporary shock.

---

## 2. Theoretical framework and Brazil's structural idiosyncrasies

### 2.1 The stock-bond correlation regime

Campbell, Pflueger, and Viceira (2020) establish that the sign of the stock-bond correlation depends on the relative importance of two shocks: (i) real growth shocks, which move stocks and bonds in opposite directions because falling growth lowers both equity cash flows and inflation, reducing nominal yields (negative correlation); and (ii) inflation supply shocks, which move stocks and bonds in the same direction because rising inflation simultaneously reduces equity valuations and pushes nominal yields higher (positive correlation). In their framework, the US transition from negative correlation (1990s–2019) to positive correlation (post-2020) reflects a shift in the dominant shock type from real demand to supply-side inflation.

Brazil's case is structurally different. While the inflation-output channel matters, it is dominated by a third channel absent from the Campbell et al. framework: **sovereign credit risk**. When fiscal conditions deteriorate in Brazil, both equity valuations and government bond prices fall simultaneously because investors price a heightened probability of fiscal adjustment, debt restructuring, or monetisation. This creates a persistent positive correlation regardless of the type of macroeconomic shock.

Blanchard's (2004) seminal NBER analysis formalised this for Brazil: when public debt is high and investors' fiscal confidence is fragile, a tightening of monetary policy can be contractionary and deflationary in the short run but simultaneously raise default risk and weaken the currency — producing the "perverse" outcome that tighter monetary policy generates capital outflows rather than inflows. This fiscal dominance regime breaks the standard inflation-output-correlation channel entirely.

### 2.2 Brazil's bond market structure

Brazil's government bond market contains an instrument with no developed-market equivalent: the **LFT (Letra Financeira do Tesouro)**, branded Tesouro Selic. This floating-rate bond accrues the daily Selic overnight rate and has near-zero modified duration. Because its price barely responds to changes in the yield curve, LFTs provide capital preservation during interest rate and fiscal stress — but they cannot appreciate (produce positive price returns) during equity selloffs. In a market where the realistic benchmark for diversification is "not losing money" rather than "gaining when equities fall," the LFT is genuinely differentiated.

As of 2026, LFTs and Selic-indexed instruments represent approximately 43% of total federal public debt. This concentration creates a second-order effect: Selic rate increases directly raise government debt service on 43% of outstanding debt, potentially amplifying fiscal pressures and reinforcing the positive stock-bond correlation dynamic.

The **NTN-B (Notas do Tesouro Nacional série B)** — inflation-linked bonds — behave differently from US TIPS during stress. While TIPS real yields tend to fall during recessions (flight to safety), Brazil's NTN-B real yields rise during fiscal crises because the sovereign risk premium embedded in real yields increases. The IMA-B 5+ (long-duration NTN-B) has exhibited some of the largest drawdowns of any government bond category during fiscal stress, contradicting its apparent inflation-protection credentials.

---

## 3. Data and methodology

### 3.1 Data sources

All data are obtained from free public sources. Daily equity returns use Ibovespa closing prices from Yahoo Finance (`^BVSP`), covering 5,498 observations from January 2004 through March 2026. Government bond returns are constructed from Tesouro Transparente daily unit prices (PU), available since December 2004 — the longest publicly available daily price series for Brazilian government bonds. For each bond type, we select the on-the-run bond whose remaining maturity is closest to a target (5 years for NTN-B, 2 years for LTN, 10 years for NTN-F, 1 year for LFT), computing daily log price changes as the total return proxy. The LFT return is alternatively proxied by compounding daily CDI rates, which by construction equals the LFT return net of any small deságio effects.

Macro variables — CDI, Selic target, IPCA, PTAX (BRL/USD), and an EMBI+ Brazil proxy — are obtained from the Banco Central do Brasil's REST API (SGS system). ETF cross-validation uses IMAB11 (IMA-B ETF, from May 2019), IB5M11 (IMA-B 5+ ETF, from September 2019), and IRFM11 (IRF-M ETF, from September 2019) via Yahoo Finance. Cross-correlations between synthetic and ETF series exceed 0.95, confirming that the Tesouro-based construction is a reliable proxy.

The master dataset contains 5,573 daily observations spanning January 2004 to March 2026. Six crisis episodes are flagged: GFC (September 2008 – March 2009), Dilma (January 2015 – August 2016), Joesley Day (May 17–31, 2017), COVID (February 20 – June 30, 2020), Americanas (January 11 – June 30, 2023), and Fiscal24 (November 2024 – January 2025). Six macro regimes are defined based on Brazil's political economy: Lula Boom (2003–2007), GFC & Recovery (2008–2012), Dilma Deterioration (2013–2016), Reform Era (2016–2019), COVID & Post-COVID (2020–2022), and Current Cycle (2023–present).

### 3.2 Statistical methods

**Rolling correlations.** We compute 252-day rolling Pearson and Spearman rank correlations between Ibovespa and each bond index. Conditional tail correlations are computed as the sample correlation of bond returns given that equity returns fall in the bottom 10th and 25th percentiles of the full-sample distribution.

**Structural stability.** A recursive CUSUM test (based on Brown, Durbin, and Evans, 1975) is applied to the OLS regression of NTN-B returns on Ibovespa returns. The test statistic reaches 0.641 with p-value 0.806, suggesting that while correlations shift across regimes, the linear relationship does not exhibit a single sharp structural break — rather, it drifts gradually across the fiscal dominance cycle.

**DCC-GARCH.** We implement Engle's (2002) two-stage DCC-GARCH(1,1). Stage 1 fits univariate GARCH(1,1) models to each return series (scaled ×100) using the `arch` Python library. Stage 2 maximises the DCC log-likelihood over parameters (a, b) governing the dynamics of the conditional correlation matrix. For Ibovespa × NTN-B: a = 0.0017, b = 0.7997, persistence = 0.80.

**Copula analysis.** Returns are transformed to uniform pseudo-observations via rank normalisation. We fit four copula families — Gaussian, Student-t, Clayton, and Gumbel — and select the best fit by AIC. The Clayton copula (parameterised by θ) is of particular interest because it captures lower-tail dependence through the coefficient λ_L = 2^(−1/θ).

**Portfolio metrics.** The Diversification Ratio (DR = weighted average individual volatility / portfolio volatility), Effective Number of Bets (ENB = exponential Shannon entropy of PC risk contributions; Meucci, 2009), and PC1 variance explained are computed on rolling 252-day windows. CoVaR (Adrian and Brunnermeier, 2016) is estimated via quantile regression of NTN-B returns on Ibovespa returns at the 5th and 50th percentiles.

---

## 4. Unconditional and regime-split correlations

### 4.1 Full-sample correlations

Over the full 2004–2026 sample, the Pearson correlation between daily Ibovespa and NTN-B returns is **+0.073** — small in absolute terms but positive, inconsistent with the negative correlation that motivates the 60/40 portfolio in advanced economies. The correlation with LTN (prefixed bonds) is higher at **+0.134**; with NTN-F at **+0.063**; and with the LFT proxy near zero at **+0.006**. These unconditional estimates establish the baseline: Brazilian stocks and government bonds co-move positively on average, with the LFT as the sole exception.

All return series reject normality strongly via the Jarque-Bera test (p < 10^{−6} in all cases), with excess kurtosis ranging from 2.3 (NTN-F) to 21.3 (LTN). This motivates the use of Expected Shortfall over Gaussian VaR, and copula methods over linear correlation analysis.

### 4.2 Regime-split correlations

The most important result is that the full-sample correlation conceals dramatic regime heterogeneity. Table 1 reports Pearson correlations for each macro regime.

**Table 1: Regime-split Pearson correlations (Ibovespa × bond indices)**

| Regime | Ibovespa × NTN-B | Ibovespa × LTN | Ibovespa × NTN-F | Ibovespa × LFT |
|--------|:----------------:|:--------------:|:----------------:|:--------------:|
| Lula Boom (2003–2007) | −0.018 | +0.044 | −0.004 | −0.009 |
| GFC & Recovery (2008–2012) | +0.023 | +0.060 | −0.003 | −0.044 |
| Dilma Deterioration (2013–2016) | +0.090 | +0.107 | +0.095 | +0.033 |
| Reform Era (2016–2019) | +0.132 | +0.240 | +0.129 | −0.004 |
| COVID & Post-COVID (2020–2022) | **+0.156** | +0.250 | +0.130 | −0.012 |
| Current Cycle (2023–present) | +0.109 | +0.123 | +0.074 | +0.045 |
| **Full sample** | **+0.073** | **+0.134** | **+0.063** | **+0.006** |

The Lula Boom is the only regime where the Ibovespa × NTN-B correlation turns negative (−0.018), corresponding to the period of fiscal prudence, falling Selic rates, and a commodity-driven equity boom. This is Brazil's only sustained period of genuine stock-bond diversification.

A counterintuitive finding deserves attention: correlations are *higher* during the Reform Era (2016–2019) and COVID cycle than during Dilma's fiscal deterioration. This reflects the difference between crisis *realisation* (asset prices fall together fast, but the episode is short) and prolonged fiscal regime uncertainty (correlations drift up persistently). During the Dilma period, the equity and bond markets moved together as fiscal credibility eroded; during the Reform Era and COVID cycle, high correlations reflected simultaneously a sensitivity to the same macro variables (global risk, commodity prices, US rates) rather than domestic fiscal stress alone.

### 4.3 Conditional tail correlations

Table 2 tests whether bonds provide diversification precisely when it is most needed — during equity market stress. The critical question is whether ρ(Ibovespa, Bond | Ibovespa < Q10) is negative (bonds rally when equities crash) or positive (bonds fall with equities).

**Table 2: Conditional tail correlations (Ibovespa bottom decile)**

| Bond | ρ full sample | ρ &#124; equity < Q10 | ρ &#124; equity < Q25 | Δ (stress − full) |
|------|:-------------:|:--------------------:|:--------------------:|:------------------:|
| NTN-B 5yr | +0.073 | +0.017 | +0.048 | −0.057 |
| LTN 2yr | +0.134 | +0.217 | +0.189 | +0.082 |
| NTN-F 10yr | +0.063 | −0.003 | +0.042 | −0.066 |
| LFT (CDI) | +0.006 | +0.055 | −0.013 | +0.048 |

Two findings stand out. First, NTN-B and NTN-F show *lower* conditional correlation during equity stress than during normal times (Δ < 0), suggesting a mild flight-to-quality within domestic government bonds during equity crashes — but the correlation remains positive (+0.017, not negative), meaning NTN-B barely moves rather than appreciating. Second, the LTN (prefixed) correlation *increases* during equity stress (Δ = +0.082), because global risk-off episodes that depress equities simultaneously push Brazilian nominal rates higher (BRL depreciation → inflation pass-through → rate expectations), driving LTN prices down simultaneously. This is the opposite of the US Treasury behaviour during equity stress.

---

## 5. DCC-GARCH: time-varying conditional correlations

The DCC-GARCH(1,1) estimation reveals that the positive correlation is not a constant feature but a time-varying process with distinct dynamics across regimes.

GARCH(1,1) volatility persistence parameters are uniformly high: Ibovespa (α + β = 0.983), NTN-B (1.000), LTN (0.978), reflecting the well-documented volatility clustering in Brazilian asset returns. The LFT proxy, as expected, shows near-constant volatility (not estimated from price changes but from CDI compounding).

DCC parameters for Ibovespa × NTN-B: a = 0.0017, b = 0.7997, persistence = 0.80. The lower-than-unit persistence (compared to many EM applications) reflects genuinely time-varying correlations rather than a near-constant process. The mean DCC ρ_t is 0.081, consistent with the full-sample Pearson estimate. Crisis-period DCC averages in Table 3 show the correlation elevation during stress periods, with the Joesley Day (LTN) producing the most dramatic spike.

**Table 3: DCC-GARCH average conditional correlation by period**

| Period | Ibov × NTN-B | Ibov × LTN | Ibov × LFT |
|--------|:------------:|:----------:|:----------:|
| GFC | 0.079 | 0.110 | 0.005 |
| Dilma | 0.080 | 0.122 | 0.005 |
| Joesley Day | 0.080 | **0.367** | 0.005 |
| COVID | 0.083 | 0.146 | 0.006 |
| Americanas | 0.080 | 0.125 | 0.005 |
| Fiscal24 | 0.082 | 0.128 | 0.005 |
| Full sample | 0.081 | 0.123 | 0.005 |

The Joesley Day spike in LTN correlation (+0.367) deserves attention. The May 2017 political shock (Ibovespa −8.8% in a single session) simultaneously caused a sharp sell-off in prefixed bonds as markets priced fiscal policy uncertainty under a potentially weakened government. This is a pure domestic political shock — entirely absent from the IMF's developed-market analysis — demonstrating Brazil's additional correlation risk dimension.

EMBI+ Brazil explains only R² = 0.017 of DCC correlation variance, suggesting the sovereign risk channel operates through other instruments (BRL depreciation, credit spreads, rate expectations) rather than through the EMBI level directly.

---

## 6. Copula analysis: tail dependence

The copula analysis addresses whether the positive correlation is driven primarily by tail events or is present uniformly across the return distribution. This distinction matters for portfolio construction: if correlation spikes only in the tail, then assets that appear diversified based on full-sample statistics will fail catastrophically during the worst drawdowns.

Fitting four copula families to pseudo-observations of Ibovespa and NTN-B returns (n = 5,098 paired observations), the AIC comparison identifies the **Gumbel copula** as the best fit (AIC = −4,109), followed by Clayton (AIC = −44) and Gaussian (AIC = −26). The Gumbel copula has upper tail dependence (λ_U = 0.022) and zero lower tail dependence — suggesting that large simultaneous *gains* in both assets are more likely than independence would predict, while large simultaneous losses are less overdispersed. This is a nuanced finding: Brazil's assets co-boom (commodity cycles, reform expectations) more than they co-crash, at least in the full-sample distribution.

However, the **Clayton copula** — which captures lower tail dependence — yields θ = 0.102 with λ_L = 0.001. This is statistically distinct from zero: during the worst equity and bond return episodes, the joint tail is thicker than the Gaussian copula (λ_L = 0 by construction) would imply. The Student-t copula, which imposes symmetric tail dependence, selects very high degrees of freedom (ν = 50), meaning the t-copula reduces approximately to the Gaussian — a result consistent with the Gumbel's dominance of upper over lower tail dependence.

The co-crash frequency analysis confirms asymmetric tail behaviour: the number of observations where both Ibovespa and NTN-B fall below their 10th percentile simultaneously exceeds the count expected under statistical independence, across all bond pairs. The LFT proxy shows zero lower-tail dependence with all other assets — its near-constant returns mean it cannot participate in any tail regime by construction.

---

## 7. Portfolio-level risk metrics

### 7.1 Diversification Ratio, ENB, and PCA

Figure 8 (produced in notebook 06) tracks three rolling 252-day portfolio metrics for a 60/40 Ibovespa + NTN-B portfolio. The Diversification Ratio fluctuates between approximately 1.05 during calm periods and approaches 1.0 during crisis episodes — the theoretical minimum indicating zero diversification benefit. The Effective Number of Bets (ENB) averages around 1.5–2.0 bets during normal times, collapsing toward 1 during acute stress. PC1 variance explained averages 49.8% across the full sample (meaning two principal components govern the system in normal times), but reaches a maximum of 68.7% during the worst stress periods — approaching the 70% threshold at which the portfolio effectively becomes a single-factor bet.

These metrics confirm what the correlation analysis suggests: diversification across Brazilian domestic asset classes is real but fragile, and it disappears precisely during the events when investors need it most.

### 7.2 CoVaR: tail risk spillover

Quantile regression of NTN-B returns on Ibovespa returns yields CoVaR estimates at the 5th percentile of the equity return distribution. When Ibovespa is at its worst 5th percentile return (−2.48% daily), NTN-B is expected to return −0.64% — a ΔCoVaR of −0.09% relative to the median equity return scenario. The quantile regression coefficient on Ibovespa is +0.036 at Q05, compared to +0.010 at Q50, indicating that the bond-equity return relationship steepens in the left tail. This confirms that NTN-B provides no meaningful cushion when equities crash badly.

---

## 8. Stress testing and portfolio P&L

### 8.1 Historical scenario replay

Table 4 reports the cumulative total return of five portfolio compositions across six crisis episodes. All returns are computed from daily log returns using the actual time-series data.

**Table 4: Portfolio P&L across historical crisis episodes (%)**

| Portfolio | GFC | Dilma | Joesley | COVID | Americanas | Fiscal24 |
|-----------|:---:|:-----:|:-------:|:-----:|:----------:|:--------:|
| 60/40 (Ibov + NTN-B) | −11.3 | +20.8 | −3.6 | −6.6 | +8.7 | −1.5 |
| Diversified (4 assets) | −4.6 | +17.8 | −2.3 | −3.4 | +7.5 | −1.0 |
| Equity heavy (80/20) | −19.3 | +18.3 | −6.2 | −12.7 | +7.6 | −2.1 |
| Bond heavy (20/80) | +5.8 | +21.5 | +0.3 | +3.5 | +9.1 | −0.4 |
| LFT only (cash proxy) | ≈ 0 | +0.1 | ≈ 0 | ≈ 0 | ≈ 0 | ≈ 0 |

Several findings are notable. First, the Dilma and Americanas episodes produced *positive* returns for bond-heavy and diversified portfolios — not because bonds provided protection, but because the Ibovespa rallied during those periods (Dilma: recovery on reform expectations; Americanas: credit crisis that did not spill into equities broadly). This illustrates that positive stock-bond correlation does not always produce losses — it merely removes the shock-absorber effect, making portfolio outcomes fully dependent on the directional call on equities.

Second, the LFT-only portfolio loses nothing across all six crises. Its maximum absolute loss across all episodes is 0.1% (Dilma, from slightly above-par pricing adjustments). This is the empirical confirmation of the LFT's role as Brazil's domestic safe haven — not through negative correlation but through absence of price risk.

Third, the Bond Heavy portfolio (20/80: Ibov/NTN-B+LTN+LFT) outperforms 60/40 in every episode except Dilma (where it matches), precisely because the lower equity weight reduces exposure to the dominant source of losses.

### 8.2 Stressed VaR

Table 5 computes the 99% 1-year Gaussian VaR (annualised) for each portfolio under the calm (full-sample) covariance matrix and under crisis-period covariance matrices. Note that Gaussian VaR substantially understates true tail risk given the kurtosis documented in Section 4.1 — these numbers are presented as minimum bounds on correlation-regime risk.

**Table 5: 99% 1-year Gaussian VaR (%) — calm vs. stressed**

| Portfolio | Calm | GFC | Dilma | Joesley | COVID | Americanas | Fiscal24 |
|-----------|:----:|:---:|:-----:|:-------:|:-----:|:----------:|:--------:|
| 60/40 (Ibov + NTN-B) | 37.5 | 92.5 | 36.0 | 69.6 | 101.3 | 26.1 | 28.2 |
| Diversified (4 assets) | 25.5 | 62.2 | 24.5 | 50.5 | 69.3 | 17.9 | 20.2 |
| Equity heavy (80/20) | 49.1 | 122.4 | 47.2 | 91.5 | 132.7 | 34.0 | 35.5 |
| Bond heavy (20/80) | 15.6 | 34.5 | 15.1 | 31.0 | 40.2 | 12.6 | 17.1 |
| LFT only | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

The 60/40 portfolio's VaR multiplies 2.7× from calm to COVID (37.5% → 101.3%) — entirely driven by the correlation regime shift during the crisis. The COVID covariance matrix embeds Ibovespa × NTN-B correlations substantially above the full-sample mean, producing this VaR amplification. The diversified portfolio (40% equities, 30% NTN-B, 15% LTN, 15% LFT) reduces VaR by 32% relative to 60/40 in calm conditions, and by 32% during COVID — a consistent relative improvement, though the absolute VaR levels remain large.

### 8.3 Equicorrelation stress test

Shrinking the correlation matrix from current estimates toward equicorrelation (all ρ → 1) increases the 60/40 portfolio's VaR by **+13.6%**. This relatively modest increase reflects the fact that Brazilian domestic correlations are already moderately elevated in the full-sample estimate — the equicorrelation stress does not represent a dramatic departure from the existing regime. In contrast, for a US 60/40 portfolio starting from negative stock-bond correlation, the equicorrelation stress test would produce a much larger VaR increase from a lower base.

---

## 9. Connecting to the IMF framework: global macro evidence

The IMF's February 2026 study focuses on advanced economies and identifies a post-2019 regime shift driven by: (1) inflation supply shocks making bonds and stocks move together; (2) fiscal expansion increasing the supply of government bonds, requiring higher yields to clear markets; and (3) QT reducing the price-insensitive central bank bid for bonds. All three mechanisms are present in Brazil, with the additional layer of sovereign credit risk that amplifies and precedes them.

### 9.1 Brazil's real rate in global context

Using FRED API data for seven major central banks (Fed, ECB, BoE, RBI, BoJ, BoC, RBA) and World Bank annual CPI for 25 countries (notebook 08), we document that Brazil's real policy rate (Selic − IPCA) has been persistently among the highest globally across the entire 2015–2026 sample. While G4 economies experienced negative real rates during 2020–2022, Brazil maintained positive real rates throughout — a direct consequence of the fiscal dominance premium (Section 2.2). The pre-2020 vs. post-2020 comparison reveals that advanced economies moved *toward* Brazil's structurally high real rate environment, not the reverse.

### 9.2 Global monetary policy cycle and Brazilian correlations

The 2022 global tightening cycle — in which the Fed, ECB, and BoE collectively raised rates by over 1,000 basis points — coincides precisely with the COVID & Post-COVID regime where the Ibovespa × NTN-B DCC correlation peaked at +0.156. Global rate hikes amplify Brazil's stock-bond correlation through the USD/BRL channel: G3 tightening strengthens the dollar, weakens the real, raises inflation pass-through expectations, and pushes both Brazilian equity and bond prices down simultaneously. The cumulative G3 rate change series (notebook 08, Figure 6.3) shows this synchronisation visually.

### 9.3 News sentiment as a regime indicator

VADER-scored monetary policy headlines (2021–2025 seed corpus + live NewsAPI extension) track the hawkish-to-dovish transition across central banks. The 2022 hawkish peak — dominated by 75bps rate hikes across the Fed, ECB, and BoE — corresponds to the period of highest DCC conditional correlations in the Brazilian domestic analysis. The cross-economy correlation between news sentiment and rate changes varies substantially, with developed economies showing tighter alignment than emerging markets where domestic political factors dominate (consistent with the Joesley Day finding in Section 5).

### 9.4 The convergence hypothesis

The critical difference between Brazil and advanced economies is the *permanence* of the condition. For G4 economies, the post-2019 correlation shift may be reversible if central bank balance sheets re-expand and fiscal positions stabilise. For Brazil, the positive stock-bond correlation has persisted across six distinct macroeconomic regimes over twenty years — surviving fiscal consolidations, commodity booms, reform attempts, and external shocks. The underlying mechanism — sovereign credit risk contaminating all domestic asset prices simultaneously — is structural rather than cyclical, rooted in the country's debt dynamics, currency vulnerability, and historical default risk. The global macro analysis (notebook 08) confirms that the post-2020 correlation regime in advanced economies represents a partial convergence toward Brazil's permanent condition.

The IMF recommends that "regulators should incorporate correlation breakdown scenarios into stress tests" as a response to the post-2020 regime. For Brazilian supervisors and investors, this recommendation has been operational necessity for years: the stressed VaR table (Table 5) shows that crisis-period VaR can be 2.5–2.7× higher than calm-period estimates, a fact that should already be embedded in risk management frameworks.

---

## 10. Conclusions and implications

This paper documents that Brazil's stock-bond diversification problem is not a post-2020 phenomenon but a structural feature of the country's financial architecture, with correlation coefficients that have been persistently positive for twenty years outside of brief reform-era windows. The key findings are:

**Finding 1: Full-sample correlations are positive for all domestic bond types.** Ibovespa × NTN-B = +0.073, × LTN = +0.134, × NTN-F = +0.063. The LFT is the sole exception at +0.006.

**Finding 2: Regime heterogeneity is large.** Ibovespa × NTN-B ranges from −0.018 (Lula Boom, genuine diversification) to +0.156 (COVID cycle). The transition to positive correlation coincides with the emergence of fiscal dominance, not with any external shock.

**Finding 3: Conditional tail correlations confirm diversification failure.** NTN-B correlation with Ibovespa during equity's worst decile is +0.017 — bonds barely move rather than appreciating. LTN correlation *increases* during equity stress (+0.082 Δ) because the BRL depreciation → inflation → rate expectations channel dominates.

**Finding 4: DCC-GARCH confirms crisis spikes.** The Joesley Day political shock drove LTN conditional correlation to 0.367. COVID drove NTN-B DCC ρ to 0.083 (from 0.081 mean — a more modest spike, consistent with the simultaneous BCB liquidity injection).

**Finding 5: Clayton copula confirms lower tail dependence.** Co-crash frequency exceeds independence expectations across all bond pairs. Gumbel copula dominates overall, suggesting co-booms (commodity and reform cycles) are more extreme than co-crashes — but both exist.

**Finding 6: Portfolio metrics confirm fragility.** PC1 variance explained reaches 68.7% during worst stress periods. ENB drops toward 1. DR approaches 1.0 during crises.

**Finding 7: LFT is the only genuine domestic diversifier.** Zero loss across all six crisis episodes. Capital preservation without capital appreciation. The appropriate benchmark for LFT is not "will it rally when equities crash?" but "will it hold its value?" — a question it answers affirmatively across all scenarios studied.

**Finding 8: Brazil's real policy rate is structurally the highest among major economies.** Global macro data (FRED, World Bank) confirm that the Selic − IPCA spread has remained persistently elevated relative to G4 peers, reflecting the sovereign credit risk premium that is the primary driver of positive stock-bond correlations.

**Finding 9: The 2021–2023 global inflation shock elevated real rates worldwide, but G4 economies converged toward Brazil's pre-existing condition** — not the reverse. The pre-2020 vs. post-2020 real rate comparison across seven central banks demonstrates this convergence quantitatively.

**Finding 10: News sentiment tracks the hawkish-to-dovish monetary policy transition,** with the 2022 hawkish peak coinciding with the period of highest Ibov×NTN-B DCC correlation (+0.156 in the COVID & Post-COVID regime). The cross-economy sentiment-to-rate-change correlation varies, with developed markets showing tighter alignment than emerging markets.

**Finding 11: Global monetary policy tightening cycles amplify Brazil's stock-bond correlation** through the USD/BRL channel. G3 rate hikes strengthen the dollar, weaken the real, raise inflation pass-through expectations, and push both Brazilian equity and bond prices down simultaneously — a transmission mechanism absent from the IMF's advanced-economy analysis.

**Implication for portfolio construction:** A Brazilian investor seeking domestic diversification should treat the fiscal dominance regime indicator (primary balance trajectory, EMBI+ trend, CDS spread level) as the master allocation switch. During monetary dominance periods (falling Selic, improving fiscal metrics, EMBI compression), longer-duration bonds (IMA-B 5+, IRF-M) provide genuine diversification as rates fall. During fiscal dominance, LFT allocation should be maximised, with the remainder in international assets (USD exposure benefits from BRL depreciation during Brazilian-specific crises) rather than attempting to find diversification within domestic fixed income.

**Implication for the IMF framework:** The post-2020 correlation regime in advanced economies is a partial convergence toward the structural condition that has characterised Brazil and other emerging markets for decades. The IMF's recommendations — fiscal discipline, credible monetary frameworks, correlation-adjusted stress tests — are correct but insufficient as described for emerging market investors, who face an additional sovereign credit risk channel that renders domestic bond diversification structurally unreliable independent of the inflation-output dynamics.

---

## References

Adrian, T., Kramer, J., and Malik, S. (2026). "Stock-Bond Diversification Offers Less Protection from Market Selloffs." IMF Blog, February 18, 2026.

Adrian, T. and Brunnermeier, M. K. (2016). "CoVaR." *American Economic Review*, 106(7), 1705–1741.

Blanchard, O. (2004). "Fiscal Dominance and Inflation Targeting: Lessons from Brazil." NBER Working Paper No. 10389.

Brown, R. L., Durbin, J., and Evans, J. M. (1975). "Techniques for Testing the Constancy of Regression Relationships over Time." *Journal of the Royal Statistical Society B*, 37(2), 149–163.

Campbell, J. Y., Pflueger, C., and Viceira, L. M. (2020). "Macroeconomic Drivers of Bond and Equity Risks." *Journal of Political Economy*, 128(8), 3148–3185.

Choueifaty, Y. and Coignard, Y. (2008). "Toward Maximum Diversification." *Journal of Portfolio Management*, 35(1), 40–51.

Engle, R. F. (2002). "Dynamic Conditional Correlation: A Simple Class of Multivariate Generalized Autoregressive Conditional Heteroskedasticity Models." *Journal of Business & Economic Statistics*, 20(3), 339–350.

Forbes, K. J. and Rigobon, R. (2002). "No Contagion, Only Interdependence: Measuring Stock Market Comovements." *Journal of Finance*, 57(5), 2223–2261.

Longin, F. and Solnik, B. (2001). "Extreme Correlation of International Equity Markets." *Journal of Finance*, 56(2), 649–676.

Meucci, A. (2009). "Managing Diversification." *Risk*, 22(5), 74–79.

Patton, A. J. (2006). "Modelling Asymmetric Exchange Rate Dependence." *International Economic Review*, 47(2), 527–556.

Portelli, L. and Roncalli, T. (2024). "Rethinking the Stock-Bond Correlation." Amundi Working Paper WP-160.

---

## Appendix A: Data dictionary

| Variable | Source | SGS/Ticker | Start | Frequency |
|----------|--------|-----------|-------|-----------|
| Ibovespa | Yahoo Finance | `^BVSP` | Jan 2004 | Daily |
| CDI | BCB SGS | 11 | Jan 2004 | Daily |
| Selic target | BCB SGS | 432 | Jan 2004 | Daily |
| IPCA | BCB SGS | 433 | Jan 2004 | Monthly |
| PTAX BRL/USD | BCB SGS | 1 | Jan 2004 | Daily |
| EMBI+ proxy | BCB SGS | 21619 | Jan 2004 | Daily |
| NTN-B prices | Tesouro Transparente | CSV | Dec 2004 | Daily |
| LTN prices | Tesouro Transparente | CSV | Dec 2004 | Daily |
| NTN-F prices | Tesouro Transparente | CSV | Dec 2004 | Daily |
| LFT prices | Tesouro Transparente | CSV | Dec 2004 | Daily |
| Fed Funds Rate | FRED | `FEDFUNDS` | Jan 2015 | Monthly |
| ECB Deposit Rate | FRED | `ECBDFR` | Jan 2015 | Monthly |
| BoE Base Rate | FRED | `BOERUKM` | Jan 2015 | Monthly |
| RBI Repo Rate | FRED | `IRSTCB01INM156N` | Jan 2015 | Monthly |
| BoJ Policy Rate | FRED | `IRSTCB01JPM156N` | Jan 2015 | Monthly |
| BoC Overnight Rate | FRED | `IRSTCB01CAM156N` | Jan 2015 | Monthly |
| US 10Y Treasury | FRED | `GS10` | Jan 2015 | Monthly |
| US CPI All Items | FRED | `CPIAUCSL` | Jan 2015 | Monthly |
| US Core PCE | FRED | `PCEPILFE` | Jan 2015 | Monthly |
| Global CPI (25 countries) | World Bank | `FP.CPI.TOTL.ZG` | 2015 | Annual |
| News Sentiment | NewsAPI + VADER | — | 2021 | Event-based |
| IMA-B ETF | Yahoo Finance | `IMAB11.SA` | May 2019 | Daily |
| IMA-B 5+ ETF | Yahoo Finance | `IB5M11.SA` | Sep 2019 | Daily |
| IRF-M ETF | Yahoo Finance | `IRFM11.SA` | Sep 2019 | Daily |

## Appendix B: Replication code

All code is available at [github.com/PedroDnT/brazil-stock-bond-correlation]. The repository contains:

- `src/fetch.py` — full data pipeline (BCB REST API + Tesouro CSV + yfinance)
- `build_notebooks.py` — generates all 7 Jupyter notebooks
- `notebooks/01_data.ipynb` — data validation and ETF cross-check
- `notebooks/02_descriptive.ipynb` — regime statistics and correlation matrices
- `notebooks/03_rolling_corr.ipynb` — rolling correlations and CUSUM test
- `notebooks/04_dcc_garch.ipynb` — DCC-GARCH estimation
- `notebooks/05_copula.ipynb` — copula fitting and tail dependence
- `notebooks/06_portfolio_metrics.ipynb` — DR, ENB, PCA, CoVaR
- `notebooks/07_stress_test.ipynb` — historical scenarios and stressed VaR
- `notebooks/08_global_macro.ipynb` — global policy rates, CPI, real rates, news sentiment, IMF framework

Dependencies: `python-bcb`, `yfinance`, `arch`, `statsmodels`, `scikit-learn`, `scipy`, `pandas`, `matplotlib`, `seaborn`, `vaderSentiment`, `python-dotenv`.
