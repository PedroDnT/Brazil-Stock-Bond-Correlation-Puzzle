# Quantifying hidden correlation risk in multi-asset portfolios: methods and application to Brazil

**Standard portfolio risk metrics systematically understate the probability and severity of simultaneous losses across asset classes.** This section provides a dual-audience technical guide — for researchers and practitioners — to detect, measure, and monitor the "hidden risk" that correlations embed in multi-asset portfolios. Applied to the Brazilian market (Ibovespa equities, government bonds LFT/LTN/NTN-B, and plain vanilla debentures), these methods reveal a structurally hostile diversification environment: Brazil's stock-bond correlation has been **persistently positive and volatile for over 18 years**, driven by sovereign credit risk rather than the inflation-output dynamics that govern developed markets. As the IMF warned in February 2026, "models calibrated on historical correlations may underestimate new risks" — a warning that has described Brazil's reality for far longer than it has described developed economies.

This guide proceeds in six parts: (1) why traditional metrics fail, (2) academic econometric methods, (3) practitioner risk metrics, (4) a cross-method comparison, (5) a Python implementation roadmap for Brazilian data, and (6) Brazil-specific empirical expectations. It builds on the government bond and debenture analyses in Sections 1–2 of this paper and directly extends the IMF's February 2026 study by Adrian, Kramer, and Malik on the breakdown of stock-bond diversification.

---

## 1. Why standard portfolio metrics hide correlation risk

### The Markowitz assumption that correlations hold still

The mean-variance framework treats the covariance matrix $\boldsymbol{\Sigma}$ as a fixed input. Optimal portfolio weights $\mathbf{w} = \frac{1}{\lambda}\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}$ depend on the **inverse** of the covariance matrix, which amplifies estimation errors catastrophically — a phenomenon Michaud (1989) termed "error maximization." When correlations shift between regimes, a portfolio optimized under one regime can be disastrously suboptimal under another. Mynbayeva, Lamb, and Zhao (2022, *European Journal of Operational Research*) demonstrated that even with normally distributed data and shrinkage estimators, mean-variance optimization fails badly because asset mean returns cannot be confidently distinguished.

Four specific failure channels undermine traditional metrics:

**Time-varying correlations.** VaR, volatility, and Sharpe ratio all embed the assumption that distributional parameters are stable over the measurement horizon. Forbes and Rigobon (2002, *Journal of Finance*) showed that correlation coefficients are conditional on market volatility: during turmoil, unadjusted correlations are **mechanically biased upward** by heteroskedasticity. After correction, the apparent correlation surge during the 1997 Asian crisis, 1994 Mexican devaluation, and 1987 U.S. crash largely vanished — suggesting that true correlations were already high during calm periods, and diversification was always less effective than assumed.

**Tail dependence.** Longin and Solnik (2001, *Journal of Finance*) applied extreme value theory to 38 years of equity data across five major markets and found that the correlation of large negative returns does **not** converge to zero as the threshold increases — directly contradicting the multivariate normal assumption. They rejected multivariate normality for the **negative tail but not the positive tail**, establishing that diversification fails asymmetrically: correlations spike in bear markets but not in bull markets. Standard deviation, a symmetric measure, is blind to this asymmetry.

**Illiquidity and stale pricing.** Getmansky, Lo, and Makarov (2003) formalized how illiquid assets exhibit serial correlation in reported returns because prices are stale. Smoothed returns reduce observed variance, bias market beta toward zero, and create the **illusion of low correlation and diversification** that vanishes during liquidation events. AllianceBernstein (2022) estimated that a significant portion of the diversification benefit attributed to illiquid assets is "fake" — an artifact of mark-to-market frequency differences. For Brazilian debentures, which trade infrequently in secondary markets, this channel is particularly dangerous.

**Positive skewness masking left-tail co-movement.** Strategies with positive skewness in normal times (carry trades, short volatility, credit harvesting) can mask catastrophic left-tail correlation. Because tail dependence is asymmetric — strong in the left tail, absent in the right — aggregate portfolio statistics during calm periods systematically understate crash co-movement risk.

### The CAPM breaks during regime switches

CAPM beta, defined as $\beta_i = \text{Cov}(r_i, r_m) / \text{Var}(r_m)$, is unstable when both numerator and denominator shift across regimes. Campbell, Pflueger, and Viceira (2020, *Journal of Political Economy*) identified a structural break in stock-bond comovements around 2001, driven by a shift in the inflation–output gap correlation from negative (stagflation era) to positive (demand-driven era). Under the pre-2001 regime, bond beta was positive (bonds were risky like stocks, $\rho \approx +0.50$); under the post-2001 regime, it turned negative (bonds hedged stocks, $\rho \approx -0.66$). Their 2025 update confirms that the post-COVID inflation episode has temporarily reversed this, aligning with the IMF's February 2026 finding that "the end of 2019 marked a structural shift."

For Brazil, the PUC-Rio dissertation by Cardoso (2024) demonstrates that the stock-bond beta has been **positive and volatile for the last 18 years** — country risk (EMBI+, CDS) dominates the inflation-output channel that governs developed-market correlations.

---

## 2. Academic methods to quantify correlation risk

Each method below is presented with intuition, formal definition, what it measures, limitations, and key references. Methods are ordered from simplest to most complex.

### 2A. Rolling and conditional correlations

**Intuition.** The simplest approach to time-varying dependence: compute correlation within a sliding window and observe how it evolves.

**Rolling Pearson correlation** over window $w$:

$$\hat{\rho}_t^{(w)} = \frac{\sum_{s=t-w+1}^{t}(r_s^A - \bar{r}^A)(r_s^B - \bar{r}^B)}{\sqrt{\sum(r_s^A - \bar{r}^A)^2 \cdot \sum(r_s^B - \bar{r}^B)^2}}$$

Typical choices: $w = 252$ (12-month daily) or $w = 756$ (36-month). Rolling Spearman rank correlation replaces values with ranks and uses $\hat{\rho}_S = 1 - 6\sum d_s^2 / [w(w^2 - 1)]$, providing robustness to outliers and nonlinear monotonic relationships.

**Conditional correlation** isolates dependence during stress: $\rho_{\text{tail}} = \text{Corr}(r^A, r^B \mid r^{\text{equity}} < F^{-1}(0.10))$, where $F^{-1}(0.10)$ is the 10th percentile of equity returns. This directly tests the hypothesis that correlations intensify during equity sell-offs. For Brazil, conditioning on Ibovespa's worst decile captures episodes like Joesley Day (May 2017, Ibovespa $-8.8\%$) and COVID (March 2020, multiple circuit breakers).

**Structural break tests** formalize the question of whether correlation regimes have shifted. The **Bai-Perron (1998, 2003)** test identifies an unknown number $m$ of breakpoints by minimizing global sum of squared residuals via dynamic programming. The sequential test $\sup F_T(l+1|l)$ tests $l+1$ versus $l$ breaks, while $UD\max = \max_{1 \leq k \leq M} \sup F_T(k)$ tests zero versus an unknown number of breaks. The **Chow test** for known breakpoints (e.g., 2013 taper tantrum, 2015 Dilma crisis, 2020 COVID) uses $F = (SSR_R - SSR_U)/k \div SSR_U/(T-2k) \sim F(k, T-2k)$.

**Limitations.** Rolling correlations are sensitive to window length and embed the Forbes-Rigobon bias — conditioning on high-volatility sub-periods mechanically inflates measured correlations. Conditional correlations suffer from small samples in the tails. Bai-Perron requires a trimming parameter and assumes a linear model.

### 2B. DCC-GARCH: dynamic conditional correlation

**Intuition.** The DCC model (Engle, 2002, *Journal of Business & Economic Statistics*) decomposes the time-varying covariance matrix into individual asset volatilities and a dynamic correlation matrix via a computationally tractable two-stage estimation.

**Stage 1 — Univariate GARCH(1,1)** for each asset $i$:

$$h_{i,t} = \omega_i + \alpha_i r_{i,t-1}^2 + \beta_i h_{i,t-1}$$

Standardized residuals: $\epsilon_{i,t} = r_{i,t} / h_{i,t}^{1/2}$.

**Stage 2 — DCC dynamics:**

$$Q_t = (1 - a - b)\bar{Q} + a\,\epsilon_{t-1}\epsilon_{t-1}' + b\,Q_{t-1}$$

$$R_t = (\text{diag}(Q_t))^{-1/2}\,Q_t\,(\text{diag}(Q_t))^{-1/2}$$

where $\bar{Q} = T^{-1}\sum \epsilon_t\epsilon_t'$ is the unconditional correlation matrix, and $a > 0, b > 0, a + b < 1$. The pairwise time-varying correlation is $\rho_{ij,t} = q_{ij,t}/\sqrt{q_{ii,t} \cdot q_{jj,t}}$. Rising $\rho_t$ toward $+1$ during crises indicates contagion; the speed of mean reversion is governed by $a + b$.

The **Asymmetric DCC (AG-DCC)** of Cappiello, Engle, and Sheppard (2006, *Journal of Financial Econometrics*) adds a leverage term:

$$Q_t = (\bar{Q} - A'\bar{Q}A - B'\bar{Q}B - G'\bar{N}G) + A'\epsilon_{t-1}\epsilon_{t-1}'A + B'Q_{t-1}B + G'n_{t-1}n_{t-1}'G$$

where $n_t = I[\epsilon_t < 0] \odot \epsilon_t$ captures the empirical fact that correlations increase more after joint negative shocks — the **leverage effect in correlations**.

**What it measures.** Time-varying pairwise correlations with correlation clustering (high correlation at $t$ implies high at $t+1$) and mean reversion.

**Limitations.** Standard DCC assumes scalar dynamics (all pairs share the same $a, b$). Gaussian quasi-likelihood underestimates tail dependence. Two-stage estimation is consistent but not fully efficient. Cannot capture nonlinear dependence beyond the correlation matrix.

### 2C. Copula-based tail dependence

**Intuition.** Copulas decompose any joint distribution into marginal distributions and a pure dependence structure, enabling modeling of nonlinear, asymmetric tail dependence that linear correlation entirely misses.

**Sklar's Theorem (1959):** For any joint CDF $F(x_1, x_2) = C(F_1(x_1), F_2(x_2))$, where $C$ is the copula — unique for continuous marginals.

**Tail dependence coefficients** measure the limiting probability of simultaneous extremes:

$$\lambda_L = \lim_{u \to 0^+} \Pr[U_2 \leq u \mid U_1 \leq u] = \lim_{u \to 0^+} \frac{C(u,u)}{u}$$

$$\lambda_U = \lim_{u \to 1^-} \Pr[U_2 > u \mid U_1 > u] = \lim_{u \to 1^-} \frac{1 - 2u + C(u,u)}{1 - u}$$

The choice of copula family determines tail behavior:

| Copula | Formula | $\lambda_L$ | $\lambda_U$ | Best for |
|--------|---------|-------------|-------------|----------|
| **Gaussian** | $\Phi_2(\Phi^{-1}(u_1), \Phi^{-1}(u_2); \rho)$ | 0 | 0 | Symmetric, no tail dependence |
| **Student-t** | $T_2(T_\nu^{-1}(u_1), T_\nu^{-1}(u_2); \rho, \nu)$ | $2t_{\nu+1}\left(-\sqrt{\frac{(\nu+1)(1-\rho)}{1+\rho}}\right)$ | Same | Symmetric crashes and booms |
| **Clayton** | $(u_1^{-\theta} + u_2^{-\theta} - 1)^{-1/\theta}$ | $2^{-1/\theta}$ | 0 | **Lower-tail co-crashes** |
| **Gumbel** | $\exp(-[(-\ln u_1)^\theta + (-\ln u_2)^\theta]^{1/\theta})$ | 0 | $2 - 2^{1/\theta}$ | Upper-tail co-booms |

**Why Student-t and Clayton copulas fit emerging market data better than Gaussian.** The Gaussian copula produces $\lambda_L = \lambda_U = 0$ for $\rho < 1$ — extremes are asymptotically independent even under high correlation. Emerging market assets exhibit significant lower tail dependence (correlated crashes driven by capital flight, sovereign risk, and currency crises), making the Clayton copula ($\lambda_L = 2^{-1/\theta} > 0$) and Student-t copula (symmetric $\lambda > 0$) far more appropriate.

**Time-varying copulas** (Patton, 2006, *International Economic Review*) allow tail dependence to evolve. For the SJC copula: $\tau_t^U = \Lambda(\omega_U + \beta_U \tau_{t-1}^U + \alpha_U \cdot \frac{1}{10}\sum_{j=1}^{10}|u_{t-j} - v_{t-j}|)$, where $\Lambda$ is the logistic function mapping to $(0,1)$.

**Limitations.** High-dimensional copulas require vine structures (computationally expensive). Family misspecification affects tail estimates. Tail coefficients require large samples. Time-varying parameters can be hard to identify.

### 2D. CoVaR: conditional value-at-risk for contagion

**Intuition.** CoVaR (Adrian and Brunnermeier, 2016, *American Economic Review*) measures the VaR of one asset (or the system) conditional on another asset being in distress, capturing directional tail risk spillover.

**Formal definition:**

$$\Pr\left(X^j \leq \text{CoVaR}_q^{j|C(X^i)} \mid C(X^i)\right) = q$$

**Delta-CoVaR** isolates the marginal contribution of distress:

$$\Delta\text{CoVaR}_q^{j|i} = \text{CoVaR}_q^{j|X^i = \text{VaR}_q^i} - \text{CoVaR}_q^{j|X^i = \text{VaR}_{50}^i}$$

**Estimation via quantile regression:**

$$\hat{X}_q^{\text{system}|X^i} = \hat{\alpha}_q^i + \hat{\beta}_q^i X^i$$

$$\text{CoVaR}_q^i = \hat{\alpha}_q^i + \hat{\beta}_q^i \cdot \text{VaR}_q^i$$

$$\Delta\text{CoVaR}_q^i = \hat{\beta}_q^i(\text{VaR}_q^i - \text{VaR}_{50}^i)$$

For Brazil, the natural application is to estimate the CoVaR of IDA-DI (debenture) returns given Ibovespa at its 5th percentile. A time-varying version with state variables $M_{t-1}$ (EMBI+, CDS, VIX) produces $\Delta\text{CoVaR}_{q,t}^i = \hat{\beta}_q^{s|i}(\text{VaR}_{q,t}^i - \text{VaR}_{50,t}^i)$.

**Limitations.** Linear quantile regression may miss nonlinear dependencies. Sensitive to conditioning event specification. Captures only pairwise, not network, effects. Noisy in small samples.

### 2E. Regime-switching models

**Intuition.** Asset correlations shift between discrete regimes governed by an unobserved Markov chain. Hamilton's (1989, *Econometrica*) filter probabilistically infers the current regime from observed data.

**Bivariate MS-VAR for correlation regimes:**

$$\begin{pmatrix} r_t^{\text{stock}} \\ r_t^{\text{bond}} \end{pmatrix} = \mu_{S_t} + \Phi_{S_t}\begin{pmatrix} r_{t-1}^{\text{stock}} \\ r_{t-1}^{\text{bond}} \end{pmatrix} + \varepsilon_t, \quad \varepsilon_t \sim N(0, \Sigma_{S_t})$$

with $\Sigma_{S_t}$ regime-dependent, yielding different correlations $\rho_{S_t}$ per state. The **transition matrix** $P$ with elements $p_{ij} = \Pr(S_t = j \mid S_{t-1} = i)$ governs regime switches. **Expected duration** of regime $i$ is $E[D_i] = 1/(1-p_{ii})$.

For Brazil, linking regime transitions to fiscal dominance indicators (primary balance, EMBI+, CDS 5-year) tests whether fiscal deterioration causes the transition to the high-correlation state. Favero and Giavazzi (2004) showed that EMBI is "the single variable that describes investor sentiment about Brazil" — all financial variables fluctuate in parallel with it.

**Limitations.** Number of regimes must be pre-specified. Assumes discrete shifts rather than smooth transitions. Parameter proliferation with additional states.

### 2F. Entropy and information-theoretic measures

**Intuition.** Information-theoretic measures capture **all** forms of statistical dependence — linear and nonlinear — unlike correlation.

**Mutual information:**

$$I(X;Y) = \sum_{x,y} p(x,y)\log\frac{p(x,y)}{p(x)p(y)} = H(X) + H(Y) - H(X,Y)$$

MI equals zero if and only if $X \perp Y$, detecting nonlinear dependencies invisible to Pearson correlation.

**Transfer entropy** (Schreiber, 2000) from $J$ to $I$:

$$T_{J \to I} = \sum p(i_{t+1}, i_t^{(k)}, j_t^{(l)})\log\frac{p(i_{t+1} | i_t^{(k)}, j_t^{(l)})}{p(i_{t+1} | i_t^{(k)})}$$

This is a nonlinear, directional generalization of Granger causality: $T_{J \to I} > 0$ implies $J$ provides predictive information about $I$ beyond $I$'s own history. For Brazil, transfer entropy can establish whether equity market stress **leads** bond market stress (or vice versa) and whether EMBI/CDS Granger-causes correlation regime transitions.

**Kullback-Leibler divergence** $D_{KL}(P\|Q) = \sum P(x)\log\frac{P(x)}{Q(x)}$ measures distributional regime shifts. Spikes in KL divergence between current and historical return distributions serve as early warning indicators of structural breaks.

**Limitations.** Requires large samples for reliable estimation. Sensitive to binning/bandwidth choices. Computationally intensive. For Gaussian variables, transfer entropy reduces exactly to Granger causality — providing no added value.

---

## 3. Practitioner methods to quantify correlation risk

### 3A. Correlation-adjusted VaR and expected shortfall

**Gaussian (parametric) VaR** for a portfolio:

$$\text{VaR}_\alpha = -V_0 \cdot z_\alpha \cdot \sqrt{\mathbf{w}^\top \boldsymbol{\Sigma} \mathbf{w}}$$

where $z_\alpha$ is the normal quantile (e.g., **2.326 for 99%**). This captures correlation through $\boldsymbol{\Sigma}$ but assumes multivariate normality. **Historical simulation VaR** uses the actual joint return distribution, naturally capturing whatever correlation structure existed in the sample period — but only that period's correlations.

**Expected Shortfall** (CVaR/ES) is the average loss beyond VaR:

$$\text{ES}_\alpha = -\frac{1}{1-\alpha}\int_\alpha^1 \text{VaR}(u)\,du$$

ES is a **coherent risk measure** (sub-additive, properly reflecting diversification) and is now mandated by Basel FRTB at **97.5% confidence**, replacing VaR. For Gaussian returns: $\text{ES}_\alpha = \frac{\phi(z_\alpha)}{1-\alpha} \cdot \sigma_p$.

**Stressed VaR** replaces the covariance matrix with one from a crisis period:

$$\text{SVaR}_\alpha = -V_0 \cdot z_\alpha \cdot \sqrt{\mathbf{w}^\top \boldsymbol{\Sigma}_{\text{stressed}} \mathbf{w}}$$

where $\boldsymbol{\Sigma}_{\text{stressed}} = \text{diag}(\boldsymbol{\sigma}) \cdot \mathbf{C}_{\text{stressed}} \cdot \text{diag}(\boldsymbol{\sigma})$, using $\mathbf{C}_{\text{stressed}}$ from March 2020 or January 2023 (Americanas). Packham and Woebbeking (2023) showed that the difference between stressed and unstressed VaR can exceed **50% of unstressed VaR** during normal markets.

**Monitoring.** Track the ES/VaR ratio over time: values above **1.5 signal fat tails**. Compare Gaussian VaR versus Historical VaR divergence as a correlation instability indicator.

### 3B. Diversification ratio

**Definition** (Choueifaty and Coignard, 2008):

$$\text{DR}(\mathbf{w}) = \frac{\sum_i w_i \sigma_i}{\sigma(\mathbf{w})} = \frac{\mathbf{w}^\top \boldsymbol{\sigma}}{\sqrt{\mathbf{w}^\top \boldsymbol{\Sigma} \mathbf{w}}}$$

When $\text{DR} = 1$, the portfolio has **no diversification benefit** — equivalent to holding a single asset or perfectly correlated assets. When $\text{DR} > 1$, genuine diversification exists. The maximum for $N$ uncorrelated, equal-volatility, equal-weight assets is $\text{DR} = \sqrt{N}$.

Portelli and Roncalli (2024, Amundi WP-160) established a critical insight: the aggregate stock-bond correlation equals the average individual stock-bond correlation **amplified** by the diversification ratio: $\rho_{S,B} = \text{DR}(\mathbf{w}) \cdot \sum_i \omega_i \rho_{i,B}$. This means **portfolio diversification simultaneously reduces volatility risk but increases correlation risk** — volatility risk decreases by approximately **3× while correlation risk increases by ~2×** in well-diversified portfolios.

**Decomposition:** $\text{DR}(\mathbf{w}) = [\bar{\rho}(\mathbf{w})(1 - \text{CR}(\mathbf{w})) + \text{CR}(\mathbf{w})]^{-1/2}$, where $\bar{\rho}$ is the volatility-weighted average correlation and $\text{CR}$ is the concentration ratio. Tracking both components distinguishes whether DR changes come from correlation shifts versus weight concentration.

### 3C. PCA on returns: the "all correlations go to one" detector

**Eigendecomposition** of the correlation matrix: $\boldsymbol{\Sigma} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^\top$, with variance explained by the $k$-th PC: $\text{VE}_k = \lambda_k / \sum \lambda_i$.

During crises, **PC1 dominates**. Empirical evidence: during COVID-19 (March 2020), PC1 explained approximately **90% of total variance** in global equities, up from ~55% in normal times. This single metric serves as a real-time correlation regime indicator:

- **PC1 > 70%:** Portfolio has effectively become a single-factor bet — diversification has collapsed
- **PC1 > 50%:** Elevated systemic risk, monitor closely
- **Normal range:** PC1 explains 30–50% of variance

**Monitoring.** Run rolling-window PCA (252-day window) on standardized returns. Track PC1 variance explained and monitor PC loadings — if all assets load similarly on PC1, market-wide contagion is occurring.

### 3D. Stress testing and scenario analysis

Four complementary approaches:

**Historical scenario replay.** Apply exact asset returns from crisis periods to the current portfolio: $\text{Loss}_{\text{scenario}} = \sum_i w_i \cdot r_{i,\text{crisis}}$. For Brazil, the critical scenarios are March 2020 (COVID — Ibovespa circuit breakers, debenture spreads from CDI+1.3% to CDI+5%+), January 2023 (Americanas — credit spreads +100bps, stock -77% in one day), 2015 (Dilma crisis — stocks and bonds falling simultaneously), and May 2017 (Joesley Day — Ibovespa -8.8%, real at largest daily depreciation in 14 years).

**Hypothetical stress.** Design forward-looking scenarios: e.g., credit spread widening +300bps, Ibovespa -20%, USDBRL +20% simultaneously. Derive secondary movements via conditional covariance: $E[\Delta\boldsymbol{\beta}_u | \Delta\boldsymbol{\beta}_s] = \boldsymbol{\Sigma}_{us}\boldsymbol{\Sigma}_{ss}^{-1}\Delta\boldsymbol{\beta}_s$.

**Correlation stress.** Shrinkage toward equicorrelation: $\mathbf{C}_{\text{stressed}} = (1-\alpha)\mathbf{C} + \alpha\mathbf{C}_1$ where $\mathbf{C}_1$ is the all-ones matrix. As $\alpha \to 1$, all correlations approach +1. The stressed matrix must remain positive semi-definite — use Higham's (2002) nearest correlation matrix algorithm.

**Reverse stress testing.** Start with a target adverse outcome (portfolio loss > $X\%$) and work backward: $\boldsymbol{\beta}^* = \arg\max_{\boldsymbol{\beta}} \text{VaR}(\boldsymbol{\beta})$ subject to $(\boldsymbol{\beta} - \boldsymbol{\mu})^\top\boldsymbol{\Sigma}_\beta^{-1}(\boldsymbol{\beta} - \boldsymbol{\mu}) \leq h$, identifying the most plausible worst-case correlation configuration.

### 3E. Liquidity-adjusted risk metrics for Brazilian debentures

Brazilian debentures trade infrequently — pre-COVID, "one of the primary challenges was the scarcity of buyers" (D'Aurea, Santander Asset). Stale pricing creates the false diversification described in Section 1.

**Liquidity-adjusted VaR** (Bangia et al., 1999):

$$\text{LVaR} = \text{VaR} + \sum_{j=1}^n \frac{(\mu_j + \lambda\sigma_j)\alpha_j}{2}$$

where $\mu_j$ and $\sigma_j$ are the mean and standard deviation of the proportional bid-ask spread, and $\lambda$ is the confidence parameter.

**Days-to-liquidate:** $\text{DTL}_i = \text{Shares held}_i / (\text{ADTV}_i \times \text{Participation Rate})$, with typical participation rate of 10–20%. For infrastructure debentures (*debêntures incentivadas*), DTL routinely exceeds **10 business days** during stress.

**The illiquidity-diversification tradeoff.** Illiquid debentures offer a spread premium (CDI + 1.3% to CDI + 3% depending on credit quality and market conditions), but including them in a diversified portfolio incurs hidden costs: wider bid-ask spreads during rebalancing, higher market impact during crisis liquidation, and stale pricing that artificially suppresses measured correlations. During Americanas (January 2023), the BCB Financial Stability Report documented "significant and abrupt increase in securities spreads, reducing fund returns, causing waves of redemptions."

### 3F. Effective number of bets

**Meucci (2009, *Risk Magazine*)** transforms correlated assets into uncorrelated "principal portfolios" via PCA, then measures how evenly risk distributes across them.

**Step 1:** Eigendecompose $\boldsymbol{\Sigma} = \mathbf{E}\boldsymbol{\Lambda}\mathbf{E}^\top$. **Step 2:** Express weights in PC space: $\tilde{\mathbf{w}} = \mathbf{E}^\top\mathbf{w}$. **Step 3:** Compute the diversification distribution: $p_k = \tilde{w}_k^2\lambda_k / \sum_i \tilde{w}_i^2\lambda_i$. **Step 4:** ENB via exponential Shannon entropy:

$$N_{\text{Ent}} = \exp\left(-\sum_{k=1}^N p_k \ln p_k\right)$$

ENB ranges from 1 (fully concentrated — all risk from one PC) to $N$ (fully diversified — equal risk across all PCs). An alternative Herfindahl-based measure is $N_{\text{HHI}} = 1/\sum p_k^2$.

During crises, PC1 dominates and all $p_k$ for $k \geq 2$ collapse, driving ENB sharply toward 1. This makes ENB a powerful **real-time diversification collapse indicator** — perhaps the single most interpretable metric for board-level reporting.

---

## 4. Cross-method comparison and synthesis

| Method | What it measures | Academic rigor | Practitioner usability | Data needs | Computation | Best for detecting | Python libraries |
|--------|-----------------|---------------|----------------------|------------|-------------|-------------------|-----------------|
| Rolling correlation | Time-varying linear dependence | Low | **High** | Returns only | Trivial | Gradual regime shifts | `pandas` |
| Bai-Perron breaks | Structural break dates | **High** | Medium | Returns only | Moderate | Discrete regime changes | `statsmodels` |
| DCC-GARCH | Dynamic conditional correlation | **High** | Medium | Returns only | Moderate | Correlation clustering, mean reversion | `arch` + `scipy` / `mvgarch` |
| ADCC-GARCH | Asymmetric correlation dynamics | **High** | Medium | Returns only | High | Leverage effect in correlations | `rmgarch` (R) |
| Copulas ($\lambda_L$, $\lambda_U$) | Tail dependence | **High** | Low | Returns (large sample) | High | **Crash co-movement** | `pycop`, `skfolio`, `scipy` |
| Time-varying copulas | Dynamic tail dependence | **High** | Low | Returns (very large) | Very high | Evolving tail risk | Custom / R `VineCopula` |
| CoVaR / $\Delta$CoVaR | Tail risk spillover | **High** | Medium | Returns + state vars | Moderate | **Contagion direction** | `statsmodels` (QuantReg) |
| Markov-switching | Discrete correlation regimes | **High** | Low | Returns + macro vars | High | Regime identification, duration | `statsmodels.tsa` |
| Mutual information | Total (incl. nonlinear) dependence | Medium | Low | Returns (large) | Moderate | Nonlinear dependencies | `sklearn.metrics` |
| Transfer entropy | Directional information flow | Medium | Low | Returns (large) | High | Lead-lag contagion | `PyInform` |
| Stressed VaR/ES | Portfolio tail loss under stress | Medium | **High** | Cov matrix + scenarios | Low | **Correlation regime impact** | `numpy`, `scipy` |
| Diversification Ratio | Portfolio diversification level | Medium | **High** | Weights + cov matrix | Trivial | Diversification collapse | `numpy` |
| PCA (PC1 share) | Factor concentration | Medium | **High** | Returns | Low | **"All correlations → 1"** | `sklearn.decomposition` |
| ENB (Meucci) | Independent bets count | Medium | **High** | Weights + cov matrix | Low | Overall diversification quality | `numpy` |
| LVaR | Liquidity-adjusted tail risk | Low | **High** | Bid-ask, volume | Low | **Illiquidity risk** | Custom |
| Reverse stress test | Worst-case correlation scenario | Medium | **High** | Full model | High | Unknown-unknowns | Custom |

**Synthesis.** No single method captures all dimensions of hidden correlation risk. The recommended stack combines:

- **For detection:** Rolling correlation + PCA monitoring (daily, cheap) as early warning; DCC-GARCH (weekly/monthly) for formal time-varying estimates
- **For quantification:** Copula tail dependence coefficients for crash co-movement probability; CoVaR for directional spillover magnitude; ENB for portfolio-level diversification quality
- **For action:** Stressed VaR/ES for capital adequacy; reverse stress testing for limit-setting; Diversification Ratio for rebalancing triggers; LVaR for Brazilian debenture sizing

---

## 5. Python implementation roadmap for the Brazilian market

### Data pipeline: from BCB to returns

The `python-bcb` library provides free access to the BCB SGS system. Core data series:

```python
from bcb import sgs

data = sgs.get({
    'Ibovespa': 7,       # Daily equity index
    'CDI': 11,           # Daily interbank rate
    'IMA_B': 12466,      # IPCA-linked bond index (NTN-B proxy)
    'IMA_B5plus': 12468, # IMA-B 5+ (long-duration NTN-B)
    'IPCA': 433,         # Monthly inflation
}, start='2010-01-01')
```

The `pyield` library (`pip install pyield`, requires Python ≥ 3.11) provides ANBIMA bond data, DI futures, and yield curve tools. For government bonds: `yd.ntnb.data("2024-06-28")` returns indicative rates; for DI futures: `yd.futures("2024-06-28", "DI1")` with flat-forward interpolation via `yd.Interpolator`.

**IDA-DI (debenture index)** requires ANBIMA Feed API access (OAuth2, paid for non-members) at `api.anbima.com.br`. Free proxies include ETF prices (e.g., via Yahoo Finance) or constructing a synthetic CDI + credit spread series. **IMA-S** (LFT/Selic-linked index) SGS code should be verified via the BCB SGS portal; if unavailable, CDI accumulation provides a close proxy since LFTs track the Selic rate with minimal spread.

**Return computation:**

```python
import numpy as np
log_returns = np.log(data / data.shift(1)).dropna()  # Preferred for statistical properties
simple_returns = data.pct_change().dropna()            # For aggregation/reporting
```

### Method-by-method implementation

**Rolling correlations** are native to pandas: `log_returns['Ibovespa'].rolling(252).corr(log_returns['IMA_B'])`. For conditional correlations, filter to observations where equity returns fall below the 10th percentile, then compute the correlation on the filtered subsample.

**DCC-GARCH** requires a two-step approach because the `arch` library (v8.0.0) supports **univariate GARCH only** — no native DCC module. Step 1: fit `arch_model(returns*100, vol='GARCH', p=1, q=1)` per asset to extract standardized residuals. Step 2: estimate DCC parameters ($a$, $b$) via maximum likelihood on the standardized residuals using `scipy.optimize.minimize`. The `mvgarch` package (`pip install mvgarch`) wraps both steps. For ADCC, the R package `rmgarch` remains the most reliable option.

**Copula fitting** uses pseudo-observations via `scipy.stats.rankdata` to transform returns to uniform margins. The `pycop` library estimates parametric copulas (Clayton, Gumbel, Student-t) and computes analytical tail dependence coefficients. The `skfolio` library integrates copulas with portfolio optimization. For the Student-t copula, the analytical tail dependence is $\lambda = 2t_{\nu+1}(-\sqrt{(\nu+1)(1-\rho)/(1+\rho)})$, computed directly from fitted $\rho$ and $\nu$.

**CoVaR** via quantile regression uses `statsmodels.formula.api.quantreg('r_system ~ r_institution', data).fit(q=0.05)`. The coefficient $\hat{\beta}_{0.05}$ on the institution's return captures tail spillover. $\Delta\text{CoVaR} = \hat{\beta}_{0.05}(\text{VaR}_{0.05} - \text{VaR}_{0.50})$.

**PCA** uses `sklearn.decomposition.PCA()` on `StandardScaler`-transformed returns (standardization ensures PCA operates on the correlation matrix rather than the covariance matrix). Rolling PCA with a 252-day window produces a time series of PC1 variance explained.

**Diversification Ratio** is a direct computation: `DR = weights @ np.sqrt(np.diag(cov)) / np.sqrt(weights @ cov @ weights)`. **ENB** requires eigendecomposition of the covariance matrix, projection of weights into PC space, computation of the diversification distribution $p_k$, and $N_{\text{Ent}} = \exp(-\sum p_k \ln p_k)$.

### Free versus paid data

All BCB SGS series (Ibovespa, CDI, IMA-B, IMA-B 5+, IPCA, PTAX) are **freely available** via `python-bcb`. The `pyield` library fetches recent ANBIMA indicative rates and B3 futures data without requiring a paid subscription. **IDA-DI historical time series**, detailed IMA compositions, and intraday data require ANBIMA Feed API credentials (paid for non-members). Free alternatives include ETF proxies (IMAB11, IB5M11) via Yahoo Finance, and synthetic CDI + spread series constructed from public debenture data.

---

## 6. What these methods will reveal about Brazil

### Country risk drives everything: the positive beta puzzle

The most important Brazil-specific finding these methods will surface is the **persistent positive stock-bond correlation** documented by Cardoso (2024, PUC-Rio). Unlike developed markets where the Campbell-Pflueger-Viceira inflation-output mechanism governs the sign, Brazil's correlation is dominated by sovereign credit risk. CDS and EMBI+ explain **71% and 56% of stock return variation** respectively (standardized regressions). When country risk rises, both stocks and bonds sell off — forcing a positive beta that the inflation channel alone cannot overcome. Controlling for EMBI or CDS, the bond-stock beta becomes smaller and occasionally negative during disinflation periods (2017–2019), but the country risk channel dominates in sample.

### DCC-GARCH will show sharp correlation spikes

Time-varying correlations between Ibovespa and IMA-B (NTN-B) are expected to spike to **+0.7 to +0.8** during Americanas (January 2023) and COVID (March 2020), with rapid mean reversion governed by the $a + b$ persistence parameter. The ADCC specification will capture the asymmetry: joint negative shocks (fiscal deterioration announcements, commodity crashes, EM risk-off events) will produce larger correlation increases than equivalent positive shocks. During the Joesley Day sell-off, the DCC path should show a near-instantaneous jump followed by slower decay — the Ibovespa dropped **8.8%** while DI futures surged and the real experienced its largest daily depreciation in over 14 years.

### Copula analysis will confirm asymmetric crash dependence

Clayton copula dominance in Brazilian asset pairs is the expected finding, reflecting strong lower-tail dependence ($\lambda_L > 0$) with negligible upper-tail dependence ($\lambda_U \approx 0$). Brazilian assets **co-crash** but do not co-boom with the same intensity. This is consistent with the country risk channel: capital flight events (fiscal crises, political shocks, global risk-off) simultaneously depress equities, widen bond spreads, and freeze debenture liquidity. The Gaussian copula ($\lambda_L = 0$) will be formally rejected; the Student-t copula will provide a reasonable fit with low degrees of freedom ($\nu \approx 4$–$6$), consistent with fat-tailed emerging market returns.

### PCA and ENB will collapse during fiscal stress

During calm periods (e.g., 2017–2019 disinflation, post-reform optimism), PC1 should explain approximately **40–50%** of variance across an Ibovespa + IMA-B + IDA-DI portfolio, with **4–5 effective independent bets**. During fiscal stress episodes, PC1 will surge above **70–80%** and ENB will collapse to approximately **1.5** — the portfolio effectively becomes a single bet on Brazilian sovereign risk. The Diversification Ratio of a typical Ibovespa + IMA-B + IDA-DI portfolio will approach **1.0** during crises, indicating near-complete loss of diversification benefit. Itaú Macro Vision (January 2025) warns that "recent developments in fiscal variables have been pointing to greater risk of fiscal dominance," with debt/GDP projected to reach **85% by 2026** — suggesting the high-correlation regime may become more frequent.

### Debenture liquidity will amplify measured correlation breakdown

The LVaR/VaR ratio for debenture allocations will diverge sharply during stress. During COVID, debenture spreads surged from **CDI + 1.3% to CDI + 5%+** with extremely low secondary market liquidity. During Americanas (January 2023), credit spreads widened **~100bps** over the year. However, ANBIMA's mark-to-market system was described as "very efficient" during Americanas, preventing the technical cascade that amplified COVID losses — an institutional improvement worth monitoring. The days-to-liquidate metric for mid-tier debentures will likely exceed **10 business days** during stress, making the illiquidity premium versus diversification cost tradeoff a first-order portfolio construction consideration.

### Connecting to the IMF's February 2026 framework

The IMF study by Adrian, Kramer, and Malik identified the end of 2019 as a global turning point for stock-bond diversification, driven by inflation surprises, widening fiscal deficits, and central bank quantitative tightening. Brazil provides both a cautionary tale and a natural experiment. The positive stock-bond correlation regime that developed markets entered only post-2020 has characterized Brazil for nearly two decades. The Brazilian experience suggests that the conditions the IMF identifies — fiscal supply pressures, inflation volatility, sovereign credit concerns — do not reverse quickly. The policy recommendation that "regulators should incorporate correlation breakdown scenarios into stress tests" is not new for Brazil — it has been the reality that Brazilian risk managers have navigated since the global financial crisis, making the methods described in this guide not merely academic exercises but operational necessities.

---

## Conclusion: an operational hierarchy for correlation risk

The central insight from this analysis is not that any single method is superior, but that **correlation risk is multi-dimensional** and requires a layered monitoring architecture. The recommended implementation sequence for a Brazilian multi-asset portfolio prioritizes speed-to-insight:

**Layer 1 (daily, automated):** Rolling correlations (63-day and 252-day), PCA PC1 share, Diversification Ratio, and ENB — all computable on free BCB data with trivial computational cost. These serve as early warning indicators.

**Layer 2 (weekly/monthly, scheduled):** DCC-GARCH time-varying correlations, conditional correlations at the 10th percentile, and CoVaR between asset pairs. These provide formal econometric estimates of correlation dynamics and tail spillover.

**Layer 3 (quarterly, analytical):** Copula fitting with tail dependence coefficients, Markov-switching regime identification, and full stress testing (historical scenarios, hypothetical scenarios, reverse stress tests). These provide deep structural insight into the dependence architecture.

**Layer 4 (as-needed, strategic):** Transfer entropy for contagion direction analysis, mutual information for nonlinear dependence detection, and Bai-Perron structural break testing for formal regime change identification.

The Brazilian market's structurally positive stock-bond correlation, driven by sovereign credit risk rather than the inflation-output dynamics of developed markets, means that the "hidden risk" of correlations is in fact the **dominant risk** in Brazilian multi-asset portfolios. The methods in this guide do not merely supplement traditional risk measures — for Brazil, they reveal the primary risk that traditional measures conceal.