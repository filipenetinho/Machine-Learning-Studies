# 2.2 — Time Series Forecasting Models

> **Module:** Artificial Intelligence II
> **What's covered:** evaluation protocol and baselines · AR / MA / ARMA · ARIMA · SARIMA · Prophet · exogenous variables (SARIMAX, Prophet with regressors) · tree-based forecasting with temporal feature engineering.
> **Data:** four real reference series + a case on Brazil's unemployment rate vs. GDP.

---

## Table of Contents

1. [Four reference series](#1-four-reference-series)
2. [Evaluation protocol](#2-evaluation-protocol)
3. [Missing values: interpolation](#3-missing-values-interpolation)
4. [Baselines](#4-baselines)
5. [Models for stationary series: AR, MA, ARMA](#5-models-for-stationary-series-ar-ma-arma)
6. [Order search and parsimony](#6-order-search-and-parsimony)
7. [Prediction intervals](#7-prediction-intervals)
8. [Non-stationary series: differencing and ARIMA](#8-non-stationary-series-differencing-and-arima)
9. [Seasonal differencing and SARIMA](#9-seasonal-differencing-and-sarima)
10. [Prophet](#10-prophet)
11. [Results on the four series](#11-results-on-the-four-series)
12. [Exogenous variables: SARIMAX and Prophet regressors](#12-exogenous-variables-sarimax-and-prophet-regressors)
13. [Tree-based forecasting and temporal feature engineering](#13-tree-based-forecasting-and-temporal-feature-engineering)
14. [Points of attention](#14-points-of-attention)
15. [Model-selection cheat-sheet](#15-model-selection-cheat-sheet)
16. [Pseudocode](#16-pseudocode)
17. [Interview questions](#17-interview-questions)
18. [References](#18-references)

---

## 1. Four reference series

| Panel | Series | Period | Horizon | Behavior |
|---|---|---|---|---|
| **A** | IPCA (monthly inflation) | Jan/15 – Jun/26 | 6 months | Oscillates around a mean, no trend |
| **B** | Global temperature (annual) | 1850 – 2025 | 5 years | Rises over time, no annual repetition |
| **C** | ANAC air passengers (monthly) | Jan/22 – Jun/26 | 6 months | Rises **and** repeats the same pattern every year |
| **D** | Retail revenue index (monthly) | Jan/00 – May/26 | 12 months | Annual pattern with **growing peaks** (multiplicative seasonality) |

Sources: Central Bank of Brazil, Berkeley Earth/OWID, ANAC, IBGE/PMC. (Retail revenue is an index of each month vs. the 2022 monthly average, in %.)

---

## 2. Evaluation protocol

A four-step methodology valid for every model family:

| Step | Description |
|---|---|
| **1. Error metric** | Usually **RMSE** (as in non-temporal projection); MAE and MAPE are valid alternatives |
| **2. External test (out-of-time)** | Hold out the **most recent** portion of the series, at least as long as the forecast horizon (to forecast 6 months, test ≥ last 6 months) |
| **3. Train and compare** | Train models of different complexity; evaluate error on train and on the external test |
| **4. Final model** | Prefer the lowest error, also considering **parsimony** and the **practical validity** of test predictions |

> ⚠️ **Never split time series randomly.** The test must be in the *future* relative to training, otherwise you leak future information.

```
|────────────── TRAIN (past) ──────────────|── TEST (most recent h) ──|  → then FORECAST →
```

---

## 3. Missing values: interpolation

Time models need the **complete** series. Deleting rows would break the regular time spacing → **interpolate** from observed neighbors.

| Method | How it fills | When to use |
|---|---|---|
| **Linear** | Straight line between adjacent neighbors | Default for short gaps (unlikely abrupt trend change) |
| **Quadratic** | Parabola fitted around the gap | Follows local curvature; preserves peaks/valleys better |
| **Cubic** | Third-degree polynomial with continuous slope and curvature at joins | Smooth curves |
| **Seasonal** | Value at the same point of the previous cycle | Strongly seasonal series |

```python
serie.interpolate(method="linear" | "quadratic" | "cubic")
serie.fillna(serie.shift(m))            # seasonal (m = periodicity)
```

Interpolation can also be used to **replace outliers**.

---

## 4. Baselines

Trivial rules that serve as a **floor** for judging forecasts:

| Baseline | Rule | Formula |
|---|---|---|
| **Naive** | Next values = last value | $\hat Y_{t+h} = Y_t$ |
| **Seasonal naive** | Next values = same point of the previous seasonal cycle | $\hat Y_{t+h} = Y_{t+h-m}$ |
| **Drift** | Extend the average slope of the history | $\hat Y_{t+h} = Y_t + h\cdot\dfrac{Y_t - Y_1}{t-1}$ |

> 🎯 **No model is really useful unless it beats the best naive baseline.**

---

## 5. Models for stationary series: AR, MA, ARMA

Stationary series are simpler to model (no trend, no seasonality). ARMA is the general form; AR and MA are special cases.

### 5.1 AR(p) — Autoregressive

$$Y_t = c + \phi_1 Y_{t-1} + \dots + \phi_p Y_{t-p} + \varepsilon_t$$

- c: constant (intercept); φ₁…φₚ ("phi"): weights of past values, to be estimated.
- εₜ: random error, **white noise** (mean 0, constant variance).
- **Order p** = number of past instants used. AR(1): Yₜ = c + φ₁Yₜ₋₁ + εₜ.
- `SARIMAX(y_train, order=(p, 0, 0), trend="c").fit()`

**Formal definition of PACF:** for each lag k, the sample partial autocorrelation is the **coefficient of the last term** of an AR(k) fitted to the series. In AR(2): $y_t = \hat c + \hat\phi_1 y_{t-1} + \hat\phi_2 y_{t-2} + e_t$, PACF(2) = φ̂₂. Each PACF stick is this coefficient, lag by lag.

### 5.2 MA(q) — Moving Average

$$Y_t = \mu + \theta_1\varepsilon_{t-1} + \dots + \theta_q\varepsilon_{t-q} + \varepsilon_t$$

- μ: fixed mean around which the series oscillates; θ₁…θ_q ("theta"): weights of past **errors**.
- `SARIMAX(y_train, order=(0, 0, q), trend="c").fit()`

> ⚠️ Don't confuse with the descriptive moving average from 2.1 (smoothing tool, not a model).

**Rationale:** the errors are part of the process structure — a **latent, unobservable** white-noise series that **generates** the oscillations of the observed series. MA is indirectly related to AR: expanding MA(1) at t = 2, $Y_2 = \mu + \theta_1\varepsilon_1 + \varepsilon_2 = \mu + \theta_1(Y_1 - \mu) + \varepsilon_2$ — each past error can be rewritten in terms of past values. So MA also imposes dependence between Yₜ and its predecessors, but differently from AR (short memory: a shock lasts exactly q periods).

### 5.3 ARMA(p, q)

$$Y_t = c + \phi_1 Y_{t-1} + \dots + \phi_p Y_{t-p} + \theta_1\varepsilon_{t-1} + \dots + \theta_q\varepsilon_{t-q} + \varepsilon_t$$

- Each value depends on past **values** and past **errors**.
- ARMA(3, 0) = AR(3); ARMA(0, 4) = MA(4).
- `SARIMAX(y_train, order=(p, 0, q), trend="c").fit()`

### Identification from ACF/PACF (classic Box-Jenkins rules)

| Model | ACF | PACF |
|---|---|---|
| AR(p) | Decays gradually | **Cuts off after lag p** |
| MA(q) | **Cuts off after lag q** | Decays gradually |
| ARMA(p, q) | Decays gradually | Decays gradually |

---

## 6. Order search and parsimony

In AR/MA/ARMA, the **hyperparameters are the orders**. The course uses an **exhaustive grid search**:

- **Grid:** all (p, q) combinations from 0 to a defined maximum.
- **Parsimony rule:** only combinations with **p + q < n/5** (n = training length). *Rule:* < n/5 safe; between n/5 and n/2 increased overfitting risk; > n/2 potentially infeasible.
- **Fit and discard:** fit each on train; discard those that didn't converge or lacked observations.
- **Choice:** typically **lowest external-test RMSE** among models whose **train-vs-test error variation is within a defined limit**. The limit is larger for longer horizons (forecasting far ahead naturally separates train and test errors).

> 💡 An alternative common in practice is choosing by information criteria (AIC/BIC) on the training data, then validating out-of-time.

---

## 7. Prediction intervals

The point forecast is the most likely value, not the only one. Models also return a **prediction interval** (shaded band in charts), usually with **95% coverage** (`alpha=0.05`).

- With **AR terms**, uncertainty **accumulates** at each step → the band opens like a **funnel**.
- With **MA terms**, it **stabilizes after q steps**.
- Wide bands don't mean the model is wrong — they mean forecasting that horizon is hard.

`model.get_forecast(h).conf_int(alpha=0.05)`

**ARMA on the four series:** beats the baseline in panels A and C; well above it in D; in B no ARMA passed the stability filter. Without trend or seasonality in the model, the forecast **converges to the mean** in a few steps, and the interval widens with the horizon.

---

## 8. Non-stationary series: differencing and ARIMA

For non-stationary series, apply transformations that make them stationary, then model with what we already know. ARIMA and SARIMA generalize ARMA to series with trend and/or seasonality — **as long as variance is constant over time** (otherwise log/Box-Cox first).

### Differencing

$$Z_t = Y_t - Y_{t-1}$$

Removes the trend. If the trend is non-linear, difference again: $W_t = Z_t - Z_{t-1}$. **In most cases, two differences are enough.** (Differencing doesn't eliminate the correlation between the original values.)

Examples: one difference removes linear trends; a non-linear (e.g., quadratic) trend needs two.

### ARIMA(p, d, q)

$$Z_t = \phi_1 Z_{t-1} + \dots + \phi_p Z_{t-p} + \theta_1\varepsilon_{t-1} + \dots + \theta_q\varepsilon_{t-q} + \varepsilon_t,\qquad Z_t = (1-B)^d\,Y_t$$

- **Backshift (lag) operator B:** $B\,Y_t = Y_{t-1}$, so $(1-B)Y_t = Y_t - Y_{t-1}$ (the difference itself).
- ARIMA = an **ARMA fitted to the d-times differenced series** ("I" = **Integrated**).
- Orders (p, d, q): AR terms on the differenced series, number of differences, MA terms.
- Examples: ARIMA(1,1,1); ARIMA(1,2,2); ARIMA(2,0,2) ≡ ARMA(2,2).
- `SARIMAX(y_train, order=(p, d, q)).fit()`
- Grid search over (p, d, q); **d > 2 not worth testing**.

**Results:** differencing solves panel B (RMSE **0.13** vs. **0.16** baseline). ARIMA beats the baseline in C; D still above. The forecast **extends the recent trend** instead of reverting to the mean (unlike ARMA).

---

## 9. Seasonal differencing and SARIMA

### Seasonal differencing (deseasonalization)

$$Z_t = Y_t - Y_{t-m}$$

m = periodicity: **7** for daily data (weekly cycle), **12** monthly, **4** quarterly. With trend, combine with simple differencing. Usually **one seasonal difference and up to two simple differences** suffice.

### SARIMA(p, d, q) × (P, D, Q, m)

$$
\begin{aligned}
W_t =\ & \phi_1 W_{t-1} + \dots + \phi_p W_{t-p} + \theta_1\varepsilon_{t-1} + \dots + \theta_q\varepsilon_{t-q} \\
     & + \Phi_1 W_{t-m} + \dots + \Phi_P W_{t-Pm} + \Theta_1\varepsilon_{t-m} + \dots + \Theta_Q\varepsilon_{t-Qm} \\
     & + \text{cross terms} + \varepsilon_t
\end{aligned}
$$

with $W_t = (1-B)^d(1-B^m)^D\,Y_t$.

- Combines a regular ARMA and a **seasonal ARMA**, both on the doubly-differenced series.
- Lowercase = regular terms (except m, the periodicity); **uppercase = seasonal terms**: Φ for values, Θ for errors.
- **Cross terms** come from multiplying regular and seasonal polynomials, at summed lags (e.g., φ₁Φ₁Wₜ₋ₘ₋₁).

Example — SARIMA(1,1,1)×(1,1,1,12):

$$W_t = \phi_1 W_{t-1} + \Phi_1 W_{t-12} - \phi_1\Phi_1 W_{t-13} + \theta_1\varepsilon_{t-1} + \Theta_1\varepsilon_{t-12} + \theta_1\Theta_1\varepsilon_{t-13} + \varepsilon_t$$

`SARIMAX(y_train, order=(p, d, q), seasonal_order=(P, D, Q, m)).fit()`

**Search:** grid over (p, d, q) × (P, D, Q); **d ≤ 2, D ≤ 1**. Parsimony: **p + q + P + Q < n/5**. P and Q tend to be smaller than p and q (they refer to older, potentially less relevant instants).

**Results:** with seasonal terms, **all four panels** have an approved model; biggest gains in C and D (seasonality dominates). The forecast repeats the previous cycle's pattern adjusted to the most recent level.

Panel D: SARIMA(0,0,3)×(1,1,1,12) — seasonal amplitude grows with the level; the model underestimates the Dec/2025 peak (143.7 vs. actual 144.7).

---

## 10. Prophet

**Prophet** (Facebook, 2017; Taylor & Letham) can forecast many series in parallel ("at scale"). It treats the series as a **sum of components**, not as dependence between instants.

$$y(t) = g(t) + s(t) + h(t) + \varepsilon_t$$

- **g(t) — trend:** piecewise-linear segments that change slope at some **changepoints**.
- **s(t) — seasonality:** sum of sines and cosines (Fourier series) with period m.
- **h(t) — holidays / special dates**, supplied in a separate table.
- εₜ: error. Seasonality can be **additive or multiplicative** (`seasonality_mode`).

| ✅ Advantages | ❌ Disadvantages |
|---|---|
| Few parameters to tune | **Ignores short-term autocorrelation** — exactly what ARIMA families exploit |
| Tolerant to missing values and level shifts | Tends to **over-smooth** short or noisy series |
| Direct reading of trend, seasonality and holiday effects (interpretable) | |

**Hyperparameters** (no orders as in SARIMAX — they control how much each component can move):

| Hyperparameter | Meaning | Default |
|---|---|---|
| `changepoint_prior_scale` | How much the trend may change slope. Low → rigid; high → flexible | 0.05 |
| `seasonality_prior_scale` | How far seasonality may deviate. High → sharper peaks/valleys | 10 |
| `yearly_seasonality` | # sine/cosine pairs (Fourier order) drawing the annual cycle. Higher → more detail, more overfitting risk | `"auto"` |
| `seasonality_mode` | `additive` (constant cycles) or `multiplicative` (cycles proportional to level) | `"additive"` |

Grid used: `changepoint_prior_scale` ∈ {0.01, 0.1, 0.5}; `seasonality_prior_scale` ∈ {1, 10}; `yearly_seasonality` ∈ {5, 10}; `seasonality_mode` ∈ {additive, multiplicative} (only for series D).

**Results:** Prophet is the **best of all families in panel C**, behind SARIMA in D; in B no configuration passed the stability filter. Forecasts extend the trend and repeat seasonality; the interval **barely widens** with the horizon (unlike ARIMA). Panel D with **multiplicative** seasonality: RMSE **2.68 vs. 4.29** additive — December is modeled better (still slightly overshoots the peak).

---

## 11. Results on the four series

Stability filter: max **20%** RMSE variation train vs. external test in panels A, B, C; **50%** in D (wider forecast amplitude).

| Panel | Best technique |
|---|---|
| A · IPCA (stationary) | **ARMA** |
| B · Temperature (trend) | **ARIMA / SARIMA** |
| C · ANAC passengers (trend + seasonality) | **Prophet** |
| D · Retail (multiplicative seasonality) | **SARIMA** |

> 💡 **There's no universal winner** — the best family depends on the series' structure. Always compare against baselines.

---

## 12. Exogenous variables: SARIMAX and Prophet regressors

We can include **explanatory (exogenous) variables** that also vary over time.

### Case: unemployment rate

Forecast Brazil's **unemployment rate** for the next **4 quarters** using GDP. GDP enters as the **% change vs. the previous quarter** (captures the pace of activity regardless of GDP's own trend and seasonality).

| Variable | Description |
|---|---|
| `TRIMESTRE` | Quarter, 2012/Q1 – 2026/Q1 |
| `TAXA_DESOCUPACAO` | % of the labor force unemployed |
| `PIB_TRIMESTRAL_MI` | Quarterly GDP at current prices (R$ million) |

EDA: unemployment rises until 2017, spikes in the pandemic, and falls to the series' lowest level in 2024–2026. Nominal GDP grows with strong quarterly seasonality. The two **variations move in opposite directions** (GDP contractions coincide with unemployment increases). ACF decays slowly (persistence — unemployment takes years to change level); PACF doesn't cut off after lag 1.

> Both techniques **don't accept missing values in exogenous variables** — drop those rows. And **forecasting Y requires future values of X** (see §14).

### SARIMAX

$$W_t = \beta_1 X_{1,t} + \dots + \beta_k X_{k,t} + \text{SARIMA terms}$$

A multiple linear regression on k exogenous variables + a SARIMA on the same series (the SARIMA terms capture the series' own dynamics). Keeps coefficient interpretation and prediction intervals.

`SARIMAX(y_train, exog=X_train, order=(p,d,q), seasonal_order=(P,D,Q,m)).fit()`

Results: 384 models tested → 33 passed the filters. Chosen: **SARIMAX(2,1,0)×(1,0,1,4)** → external-test error **0.42 p.p.** vs. **0.98 p.p.** for the best baseline. Refit on the full series and forecast 4 quarters using GDP variation forecast by its own best baseline (seasonal naive, RMSE 0.89).

> The **GDP coefficient was not significant** (p = 0.55). The series' own dynamics (AR and seasonal terms) dominate. **An exogenous variable doesn't always add information beyond the series' history.**

### Prophet with regressors

Each regressor gets its own coefficient estimated with the rest; future values must be provided; can be additive or multiplicative; lags of the regressor can be added as new columns (feature engineering). `m.add_regressor("variacao_pib")`

Results: only 2 of 12 configurations passed the stability filter; chosen one errs **0.90 p.p.** vs. 0.42 for SARIMAX. With `changepoint_prior_scale=0.01` the trend becomes too rigid (error > 5 p.p.). Refit on the full series, Prophet projects unemployment **below 4%** — outside anything observed: without lags of the response, it just extends the recent downward trend.

---

## 13. Tree-based forecasting and temporal feature engineering

We can forecast with regression trees, random forest and boosting.
- When the past–present relationship is **non-linear or regime-changing**, trees can beat linear models.
- To extend **well-behaved trend and seasonality**, linear models are usually better (trees **can't extrapolate**).

**Why trees?** Y may relate to X at the same instant or with lags (e.g., Yₜ ~ Xₜ₋₂). In SARIMAX/Prophet, testing this is iterative (try Xₜ₋₁, then Xₜ₋₂…) and you can't include many lagged/aggregated versions of the same variable (collinearity, overfitting). Trees **select variables automatically** at each split, and collinearity doesn't matter much (no coefficients estimated).

**But trees don't see time:** each instant is an independent row — even the target's autocorrelation isn't modeled. → **feature engineering** must rebuild the chronological structure as columns.

### 13.1 Non-seasonal aggregations (last k instants)

Notation: Xₜ original variable, Wₜ new feature (can also be applied to Y).

| Feature | Formula |
|---|---|
| **Lag** | $W_t = X_{t-k}$ |
| **Rolling mean** | $W_t = (X_{t-1} + \dots + X_{t-k})/k$ |
| **Rolling median** | $W_t = \text{med}(X_{t-1},\dots,X_{t-k})$ |
| **Rolling min / max** | $W_t = \min/\max(X_{t-1},\dots,X_{t-k})$ |
| **Rolling std** | $W_t = \text{sd}(X_{t-1},\dots,X_{t-k})$ |
| **Step-1 change** | $W_t = X_{t-k} - X_{t-k-1}$ |
| **Rolling mean of changes** | $W_t = [(X_{t-1}-X_{t-2}) + \dots + (X_{t-k}-X_{t-k-1})]/k$ |
| **Rolling median of changes** | $W_t = \text{med}[(X_{t-1}-X_{t-2}),\dots,(X_{t-k}-X_{t-k-1})]$ |

### 13.2 Seasonal aggregations (periodicity m)

| Feature | Formula |
|---|---|
| **Seasonal lag** | $W_t = X_{t-k\cdot m}$ |
| **Seasonal step-1 change** | $W_t = X_{t-k m} - X_{t-k m-1}$ |
| **Seasonal-step change** | $W_t = X_{t-k m} - X_{t-(k+1)m}$ |

### 13.3 Calendar features
Day of month (1–31) and day of week (1–7) for daily series; week of year (1–53); month (1–12); quarter (1–4); year.

### 13.4 Atypical-event flags
Dummies marking instants where something affected Y: unplanned production stops; Black Friday in daily sales; a merger quarter in a new-customer series.

### Case: unemployment with trees

Engineering on unemployment and GDP variation produced **56 columns**: lags 1–4; rolling mean/median/min/max/std in windows 2, 3, 4; differences and their rolling means; same quarter 1 and 2 years ago + changes; quarter number (1–4).

> 🔒 **All aggregations start from lag 1**, so each row only uses information **prior** to the instant being predicted (no leakage).

### Four specific precautions

| # | Precaution |
|---|---|
| 1. **Feature engineering** | Temporal dependence is represented as columns (lags, aggregations…) |
| 2. **Temporal cross-validation** | High overfitting risk → CV with folds that **respect time order** (`TimeSeriesSplit`) |
| 3. **Missing values** | Lag features leave the first rows empty → use only algorithms with **native missing handling** (so **AdaBoost and classic Gradient Boosting were excluded**) |
| 4. **Hyperparameters** | Minimum observations per node must be **much smaller** than in non-temporal projection (series are short) |

**Time-series CV (expanding window):**

```
fold 1: [train      ][test]
fold 2: [train           ][test]
fold 3: [train                ][test]
fold 4: [train                     ][test]
fold 5: [train                          ][test]   → then external test (most recent)
```

**Results:** 150 models tested, 40 approved; training error near **zero** shows how much the tree memorizes a small dataset. Chosen: **XGBoost** (model 83), best in internal test → **1.87 p.p.** on external test. Predictions stay **stuck at the training level** (no extrapolation). Recursive forecast: each predicted quarter feeds the next quarter's lags → forecast between 6.8% and 7.8%. **No prediction interval** — trees give only point values.

Importance (impurity & permutation agree; SHAP too): **previous unemployment** and the **2-quarter mean** dominate; GDP variation enters via 4-quarter aggregations. SHAP direction: higher recent unemployment → higher current unemployment.

### Final comparison — unemployment

| Technique | External-test RMSE |
|---|---|
| Best baseline | 0.98 p.p. |
| **SARIMAX** | **0.42 p.p.** ✅ |
| Prophet | 0.90 p.p. |
| Trees (XGBoost) | 1.87 p.p. |

> With a small training set (**53 observations**) and well-behaved trend and seasonality, having the **temporal structure built into the model** pays off more than tree flexibility.

---

## 14. Points of attention

**(1) Future values of exogenous variables.** If Yₜ ~ Xₜ₋₁ and we forecast 2 steps ahead, Yₜ₊₁ depends on Xₜ (known) but Yₜ₊₂ depends on Xₜ₊₁ (**unknown**). When the impact horizon of X is shorter than the forecast horizon, we must **forecast X too** — by simulating business scenarios, or with separate models for X (beware of **error accumulation**).

**(2) Recursive forecasting with trees.** ARIMA/SARIMA incorporate lags of Y automatically and forecast cumulatively. In trees, lags of Y are manual columns → if Yₜ₊₂ depends on Yₜ₊₁ (unknown), **redo the feature engineering at each step using previous predictions**.

**(3) Collinearity from temporal features.** Many versions of the same variable cause collinearity. It doesn't hurt predictive power but can hurt interpretation of SARIMAX coefficients (linear). If SARIMAX is used only for prediction, mild/moderate collinearity is fine — but be **parsimonious** (each column = one more coefficient from the same small sample). Trees just pick the useful columns at each split.

---

## 15. Model-selection cheat-sheet

```
Is the series stationary (ADF/KPSS, plot, ACF)?
├── YES → AR / MA / ARMA   (use ACF/PACF to guess p, q; grid search; compare with naive)
└── NO
    ├── Trend only            → ARIMA (d = 1 or 2)            | or drift baseline
    ├── Seasonality (+trend)  → SARIMA (D = 1, m = period)     | or Prophet | seasonal naive
    │     amplitude grows with level → log transform or multiplicative Prophet
    ├── Exogenous drivers     → SARIMAX / Prophet + regressors (need future X!)
    └── Non-linear / regime changes / many candidate drivers / lots of data
                              → Trees / boosting with lag & rolling features + TimeSeriesSplit
Always: out-of-time test ≥ horizon · beat the best baseline · check stability train vs. test
```

---

## 16. Pseudocode

### 16.1 Baselines

```text
FUNCTION Naive(Y, h):          RETURN [Y[n]] * h
FUNCTION SeasonalNaive(Y, h, m): RETURN [Y[n + i - m · ceil(i/m)] FOR i IN 1..h]
FUNCTION Drift(Y, h):          slope ← (Y[n] - Y[1]) / (n - 1);  RETURN [Y[n] + i · slope FOR i IN 1..h]
```

### 16.2 Out-of-time grid search for (S)ARIMA

```text
PROCEDURE SearchSARIMA(Y, h, max_orders, m, stability_limit)
    train ← Y[1 .. n-h];  test ← Y[n-h+1 .. n]
    best_baseline ← min RMSE over {Naive, SeasonalNaive, Drift} on test
    candidates ← []
    FOR p, d, q, P, D, Q IN grid (d ≤ 2, D ≤ 1):
        IF p + q + P + Q ≥ len(train) / 5: CONTINUE                    # parsimony
        TRY model ← SARIMAX(train, order=(p,d,q), seasonal_order=(P,D,Q,m)).fit()
        EXCEPT non-convergence: CONTINUE
        rmse_train ← RMSE(train, model.fitted)
        rmse_test  ← RMSE(test, model.forecast(h))
        IF |rmse_test - rmse_train| / rmse_train ≤ stability_limit:     # stability filter
            candidates.append((p,d,q,P,D,Q), rmse_test)
    best ← argmin rmse_test in candidates
    IF best.rmse_test ≥ best_baseline: WARN "model does not beat baseline"
    final ← SARIMAX(Y, best orders).fit()                               # refit on full series
    RETURN final.get_forecast(h) with 95% interval
```

### 16.3 ARMA forecasting recursion

```text
FUNCTION ForecastARMA(Y, ε̂, c, φ[1..p], θ[1..q], h)
    FOR step IN 1..h:
        t ← n + step
        ar_part ← Σ_{i=1}^{p} φ[i] · Y[t-i]           # uses forecasts for t-i > n
        ma_part ← Σ_{j=1}^{q} θ[j] · ε̂[t-j]           # future errors = 0 (their expectation)
        Y[t] ← c + ar_part + ma_part
    RETURN Y[n+1 .. n+h]
# → MA influence vanishes after q steps; AR forecasts decay toward the mean c/(1 - Σφ)
```

### 16.4 Temporal feature engineering (no leakage)

```text
FUNCTION MakeTemporalFeatures(df, cols, windows, m)
    FOR each column x IN cols (including target y):
        FOR k IN 1..4:            df[x_lag_k]  ← shift(x, k)
        base ← shift(x, 1)                                   # start from lag 1!
        FOR w IN windows:
            df[x_mean_w] ← rolling_mean(base, w);  df[x_med_w] ← rolling_median(base, w)
            df[x_min_w]  ← rolling_min(base, w);   df[x_max_w] ← rolling_max(base, w)
            df[x_std_w]  ← rolling_std(base, w)
            df[x_diffmean_w] ← rolling_mean(diff(base), w)
        df[x_diff1]   ← base - shift(x, 2)
        df[x_slag_1]  ← shift(x, m);  df[x_slag_2] ← shift(x, 2m)
        df[x_sdiff]   ← shift(x, m) - shift(x, 2m)
    df[calendar] ← quarter / month / weekday of index
    df[event_flags] ← 1 on atypical events, else 0
    RETURN df
```

### 16.5 Recursive multi-step forecasting with a tree model

```text
FUNCTION RecursiveForecast(model, history_df, future_exog, h)
    preds ← []
    FOR step IN 1..h:
        row ← MakeTemporalFeatures(history_df) at the next timestamp   # uses past preds as lags
        row.exog ← future_exog[step]                                    # scenario / separate model
        ŷ ← model.predict(row)
        preds.append(ŷ)
        history_df.append(timestamp, y = ŷ, exog = future_exog[step])
    RETURN preds        # point forecasts only (no native interval)
```

### 16.6 Time-series cross-validation

```text
FUNCTION TimeSeriesSplit(n, n_splits, test_size)
    FOR i IN 1..n_splits:
        train_end ← n - (n_splits - i + 1) · test_size
        YIELD train = [1 .. train_end], test = [train_end + 1 .. train_end + test_size]
```

Python reference:

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.model_selection import TimeSeriesSplit

m = SARIMAX(y_tr, exog=X_tr, order=(2,1,0), seasonal_order=(1,0,1,4)).fit(disp=False)
fc = m.get_forecast(steps=4, exog=X_future); fc.predicted_mean; fc.conf_int(alpha=0.05)

p = Prophet(changepoint_prior_scale=0.1, seasonality_prior_scale=10,
            yearly_seasonality=10, seasonality_mode="multiplicative")
p.add_regressor("variacao_pib"); p.fit(df_prophet)          # columns ds, y, variacao_pib
p.predict(future_df)

tscv = TimeSeriesSplit(n_splits=5)
```

---

## 17. Interview questions

1. **How do you validate a forecasting model?** Out-of-time holdout (most recent data, ≥ horizon) and/or rolling-origin (expanding window) CV; compare with naive baselines.
2. **Why are naive baselines important?** Many sophisticated models fail to beat them; a model that doesn't beat seasonal naive adds no value.
3. **AR vs. MA?** AR regresses on past values (long, decaying memory); MA on past shocks (memory of exactly q periods).
4. **How do you pick p and q?** PACF cut-off suggests p, ACF cut-off suggests q; then grid search with AIC/BIC or out-of-time error, with parsimony.
5. **What does the "I" in ARIMA mean?** Integrated — the model is fitted on the d-times differenced series.
6. **Explain SARIMA(1,1,1)(1,1,1,12).** One AR and one MA term, one regular difference; one seasonal AR, one seasonal MA, one seasonal difference at lag 12.
7. **Why do ARIMA intervals widen with the horizon?** Uncertainty accumulates through the AR recursion.
8. **Prophet vs. SARIMA?** Prophet: additive components (piecewise trend + Fourier seasonality + holidays), robust and easy, ignores short-term autocorrelation. SARIMA: explicit autocorrelation structure, better for short-memory dynamics.
9. **What's the main difficulty of using exogenous variables?** You need their future values — forecast or scenario them, with compounding error.
10. **How do you use XGBoost for time series?** Build lag/rolling/calendar features from past data only, use TimeSeriesSplit, forecast recursively (or direct multi-horizon models); remember trees don't extrapolate trends (consider differencing the target).
11. **Why can't you shuffle in time-series CV?** It leaks future information into training and overestimates performance.
12. **Additive vs. multiplicative seasonality — how do you decide?** If seasonal amplitude grows with level → multiplicative (or log-transform and use additive).

---

## 18. References

- Box, G. E. P.; Jenkins, G. M. *Time Series Analysis: Forecasting and Control*. Holden-Day, 1970.
- Hyndman, R. J.; Athanasopoulos, G. et al. *Forecasting: Principles and Practice, the Pythonic Way*. OTexts, 2024. <https://otexts.com/fpppy>
- Morettin, P. A.; Toloi, C. M. C. *Análise de Séries Temporais*. 2nd ed. Blucher, 2006.
- Shumway, R. H.; Stoffer, D. S. *Time Series Analysis and Its Applications*. 2nd ed. Springer, 2006.
- Taylor, S. J.; Letham, B. "Forecasting at Scale." *The American Statistician*, 2018.

---

⬅️ Previous: [2.1 — Introduction to Time Series](../2.1%20-%20Introduction%20to%20Time%20Series%20Analysis/README.md) · 🏠 [Index](../../README.md) · ➡️ Next: [2.3 — Logistic Regression](../2.3%20-%20Classification%20-%20Logistic%20Regression/README.md)
