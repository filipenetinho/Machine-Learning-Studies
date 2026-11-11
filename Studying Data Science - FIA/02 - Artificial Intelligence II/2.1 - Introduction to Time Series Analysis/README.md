# 2.1 — Introduction to Time Series Analysis

> **Module:** Artificial Intelligence II
> **Case study:** a software company wants to estimate how many accesses its mobile app will receive in the **next 5 business days**, using the daily history of the last 90 business days.

---

## Table of Contents

1. [What a time series is](#1-what-a-time-series-is)
2. [Notation](#2-notation)
3. [Components: trend and seasonality](#3-components-trend-and-seasonality)
4. [Decomposition (additive vs. multiplicative)](#4-decomposition-additive-vs-multiplicative)
5. [Smoothing](#5-smoothing)
6. [Stationarity](#6-stationarity)
7. [Stationarity tests: ADF and KPSS](#7-stationarity-tests-adf-and-kpss)
8. [Autocorrelation vs. white noise](#8-autocorrelation-vs-white-noise)
9. [ACF and PACF](#9-acf-and-pacf)
10. [Reading ACF/PACF patterns](#10-reading-acfpacf-patterns)
11. [Case: app accesses](#11-case-app-accesses)
12. [Pseudocode](#12-pseudocode)
13. [Interview questions](#13-interview-questions)
14. [References](#14-references)

---

## 1. What a time series is

A **time series** is a set of observations of a variable (usually quantitative) **indexed by time**. Examples:
- Daily USD/BRL exchange rate, 2020-01-01 to 2025-12-31.
- Gross monthly revenue, Jan–Jun 2026.
- Annual average daily rainfall in a city, 1960–2025.

Each row is a distinct, sequential instant, so observations have **strong dependence**: today's exchange rate is heavily influenced by yesterday's.

> ⚠️ Linear regression **assumes independent observations** — that's why time series need their own models.

Applications:

| Problem | Motivation |
|---|---|
| Units sold next month (monthly sales history) | Stock sizing, pricing |
| E-commerce site visits in the next weeks (2 years of weekly history) | Server capacity, resource planning, UX under peak demand |
| Hotel occupancy at the next year-end (previous year-ends) | Booking management and pricing; public safety and health planning |

**Goal of time-series modeling:** project the most plausible future values of a series from its past behavior. Before building any model, understand the series' structure.

---

## 2. Notation

- **Y** is a quantitative variable indexed by time (the series); **Yₜ** is its value at time *t*.
- Regardless of granularity (days, months…), time is represented by positive integers **t = 1, …, n**.

Example (monthly headcount, n = 5):

| | Jan/26 | Feb/26 | Mar/26 | Apr/26 | May/26 |
|---|---|---|---|---|---|
| t | 1 | 2 | 3 | 4 | 5 |
| Yₜ | 235 | 260 | 278 | 290 | 262 |

---

## 3. Components: trend and seasonality

A **line chart** is the simplest way to see a series' behavior.

| Component | Definition |
|---|---|
| **Trend** | Values grow or decline over time. One series can have different trends in different periods. |
| **Seasonality** | A **cyclical repetition** of behavior over periods of **fixed length** (days of the week, months of the year…). |
| **Residual (noise)** | What's left: short-term irregular fluctuations. |

> (Cycles of *variable* length, like business cycles, are different from seasonality.)

Exercise — classify series as: no trend/no seasonality · trend/no seasonality · no trend/seasonality · trend/seasonality. (Online simulator: <https://www.mysimulator.uk/data-science/time-series-decomposition/>.)

---

## 4. Decomposition (additive vs. multiplicative)

**Decomposing** = separating the series into its components.

**Additive:** observed value = trend + seasonality + residual

$$Y_t = T_t + S_t + R_t$$

**Multiplicative:** used when the **seasonal amplitude grows with the series level** (instead of staying constant)

$$Y_t = T_t \times S_t \times R_t$$

> 💡 A multiplicative series becomes additive after a **log transform**: log Yₜ = log Tₜ + log Sₜ + log Rₜ.

If the seasonal bars in the decomposition plot have similar heights over time → additive decomposition.

```python
from statsmodels.tsa.seasonal import seasonal_decompose
seasonal_decompose(y, model="additive", period=12).plot()
```

---

## 5. Smoothing

An observed series = **general background behavior** (of most interest for forecasting) + **short-term fluctuations** (often hard to model). **Smoothing** attenuates the fluctuations so trend and seasonality become visible.

> Smoothing methods are **descriptive** — they help see what the series is made of. **They are not forecasting models.**

### 5.1 Simple (trailing) moving average of order k

Replaces each point by the average of the **k most recent** observations, including itself:

$$MM_t = \frac{1}{k}\sum_{i=0}^{k-1} Y_{t-i}$$

Larger window → smoother curve, but it **lags** behind recent oscillations.

### 5.2 Centered moving average of order k

Averages observations **around** t, so the smoothed value doesn't lag:

$$MMC_t = \frac{1}{k}\sum_{i=-m}^{m} Y_{t+i},\qquad m = \frac{k-1}{2}$$

The order must be **odd** to be centered. For an **even** seasonal period (e.g., 12 months), take the average of **two consecutive order-12 moving averages** (a "2×12-MA"), which re-centers the result and cancels the annual oscillation → isolates the trend.

### 5.3 Simple Exponential Smoothing (SES)

Weighted average between the current observation and the previous smoothed value, with weight **α ∈ (0, 1)**:

$$S_t = \alpha Y_t + (1-\alpha)\,S_{t-1}$$

- High α → follows the series closely (little smoothing).
- Low α → approaches the historical average (heavy smoothing).
- Unrolling it: weights on past observations decay **exponentially**: α, α(1−α), α(1−α)², …

| Method | Weights | Lag | Note |
|---|---|---|---|
| Simple MA | Equal on last k | Lags | Good for visual trend |
| Centered MA | Equal around t | No lag | Needs future values → only descriptive |
| SES | Exponentially decaying | Some lag | Basis of Holt / Holt-Winters (2.2) |

---

## 6. Stationarity

A series is **stationary** when its behavior **doesn't depend on when it's observed**. In practice:
1. It varies around a **constant mean**;
2. With **constant variance**;
3. The relationship between two instants depends **only on the distance (lag)** between them, not on their position in time.

> ⚠️ Stationarity implies **no trend and no seasonality** — but **not the reverse**. A series with neither can still be non-stationary by changing **level** (structural breaks in the mean) or **volatility** (changes in dispersion).

> 💡 **Why it matters:** many classical models (ARMA) require stationarity — the past is only a reliable guide to the future if the statistical properties don't change. Non-stationary series are usually made stationary by **differencing** (Yₜ − Yₜ₋₁) or transformations (log).

---

## 7. Stationarity tests: ADF and KPSS

Visual reading is subjective (weak trends, smooth breaks). Use hypothesis tests:

| Test | H₀ | Idea | Watch out |
|---|---|---|---|
| **ADF — Augmented Dickey-Fuller** | Series is **NOT** stationary (has a unit root) | Looks for evidence that the series is **pulled back to the mean** when it moves away | With **small samples** it may lack power to reject H₀ even for truly stationary series |
| **KPSS** | Series **IS** stationary | Tracks the **cumulative sum of deviations** from the mean | If the series goes up and down in long stretches, movements cancel in the cumulative sum → may fail to reject H₀ for non-stationary series |

> 💡 Because their null hypotheses are opposite, **use both**:
>
> | ADF | KPSS | Conclusion |
> |---|---|---|
> | Rejects H₀ (p < 0.05) | Doesn't reject | **Stationary** (both agree) |
> | Doesn't reject | Rejects | **Non-stationary** (both agree) |
> | Disagree | | Inconclusive → look at plots, sample size, consider differencing |

Examples from class:
- Monthly ITUB4 / PETR4 quotes (2011–2026): one case where both agree on stationarity (e.g., returns), another where both agree on non-stationarity (prices).
- Monthly ABEV3 (2021–2025): small sample → ADF wrongly concludes non-stationarity; KPSS correctly detects stationarity.
- Daily Dow Jones (2021–2023): ups and downs cancel → KPSS wrongly concludes stationarity; ADF correctly detects non-stationarity.

---

## 8. Autocorrelation vs. white noise

- **Monthly IPCA (Brazilian inflation) variation**: stationary **but autocorrelated** — consecutive months are related.
- **Monthly ITUB4 return**: stationary **white noise** — each value is not influenced by previous ones; **no autocorrelation**.

**White noise:** independent values with constant mean (usually 0) and constant variance → **unpredictable** from its past. If a model's residuals are white noise, the model has extracted all the predictable structure.

### Autocorrelation coefficient

The **Pearson correlation between the series and a lagged copy of itself**, for a given **lag**:

$$\rho_k = \text{corr}(Y_t,\ Y_{t-k})$$

Example (monthly headcount): lag-1 autocorrelation = 0.307; lag-2 = −0.490.

| t | Yₜ | Yₜ₋₁ (lag 1) | Yₜ₋₂ (lag 2) |
|---|---|---|---|
| 1 | 235 | – | – |
| 2 | 260 | 235 | – |
| 3 | 278 | 260 | 235 |
| 4 | 290 | 278 | 260 |
| 5 | 262 | 290 | 278 |

Notes:
- Autocorrelation tends to **decrease with larger lags** (an instant relates more to its neighbors) — **except for seasonality**.
- **Population** autocorrelations are unknown parameters; we compute **sample** autocorrelations as estimates.

---

## 9. ACF and PACF

### ACF — Autocorrelation Function
Associates an autocorrelation value with each lag. Shown as a **stick plot**: each stick is the sample autocorrelation at that lag; the **shaded band** marks values compatible with **zero population autocorrelation** (a hypothesis test, roughly ±1.96/√n).

### PACF — Partial Autocorrelation Function
The **partial autocorrelation of order k** = how much **additional** information lag k gives to explain the series **once lags 1 … k−1 are already known**.

Example (order 2): how much more do the lag-2 values help explain the original series, given that the lag-1 column is already known?

> 💡 Intuition: ACF(2) includes the *indirect* effect Yₜ₋₂ → Yₜ₋₁ → Yₜ. PACF(2) removes it, leaving only the *direct* effect. Technically, PACF(k) is the coefficient of Yₜ₋ₖ in a regression of Yₜ on Yₜ₋₁, …, Yₜ₋ₖ.

---

## 10. Reading ACF/PACF patterns

| Series | ACF | PACF |
|---|---|---|
| **White noise** | No stick outside the band after lag 0 | Same → the past carries no information about the future |
| **Stationary (short memory)** | Decays **fast** | **Cuts off** after the first lags |
| **Single trend** | Decays **slowly**, stays high for many lags (trend keeps distant values similar) | Big spike at lag 1 |
| **Trend reversal** | Falls faster than single trend; **negative** at long lags | |
| **Seasonal (period 12)** | **Oscillates**, rising again at **multiples of the seasonal period** (12, 24, …) | Spikes near seasonal lags |
| **Trend + seasonality** | Both effects combined: slow decay + regular peaks | |

These patterns guide model choice in 2.2 (e.g., PACF cut-off at lag p → AR(p); ACF cut-off at lag q → MA(q)).

---

## 11. Case: app accesses

Data: `DIA_UTIL` (business day), `QTD_ACESSOS` (accesses), last 90 business days. Goal: forecast the next 5 business days to plan infrastructure and avoid instability on peak days.

1. **Observed series:** oscillates around a stable level, no sustained growth, no fixed-period repetition → suggests **stationarity**.
2. **ACF:** fast decay confirms **no trend**; sticks outside the band in the first lags → **not white noise** (there is exploitable structure).
3. **PACF:** sticks outside the band in the first lags → recent instants carry **their own information** about the current value beyond what earlier lags explain.
4. **Diagnosis:** the series is **stationary and autocorrelated** → next step (2.2): build a model to project it (an AR/ARMA-type model is a natural candidate).

---

## 12. Pseudocode

### 12.1 Smoothers

```text
FUNCTION MovingAverage(Y[1..n], k)
    FOR t IN k..n:  MM[t] ← (1/k) · Σ_{i=0}^{k-1} Y[t-i]
    RETURN MM

FUNCTION CenteredMovingAverage(Y, k)              # k odd
    m ← (k - 1) / 2
    FOR t IN (m+1)..(n-m):  MMC[t] ← (1/k) · Σ_{i=-m}^{m} Y[t+i]
    RETURN MMC

FUNCTION CenteredMA_EvenPeriod(Y, s)               # e.g., s = 12
    A ← MovingAverage(Y, s)                        # trailing order-s MA
    B ← MovingAverage(A, 2)                        # average of two consecutive → re-centered
    RETURN shift(B, -s/2)

FUNCTION SimpleExpSmoothing(Y, α)
    S[1] ← Y[1]
    FOR t IN 2..n:  S[t] ← α · Y[t] + (1 - α) · S[t-1]
    RETURN S
```

### 12.2 Additive decomposition (classical)

```text
FUNCTION ClassicalDecomposition(Y, s)
    T ← CenteredMA_EvenPeriod(Y, s)   (or CenteredMovingAverage if s odd)    # trend
    D ← Y - T                                                                # detrended
    FOR each season position j IN 1..s:
        S_j ← mean of D at all t with (t mod s) = j
    S_j ← S_j - mean(S_1..S_s)                                               # seasonal effects sum to 0
    S[t] ← S_{t mod s}
    R ← Y - T - S                                                            # residual
    RETURN T, S, R
# Multiplicative: replace "-" with "/" (and normalize S to average 1), or apply log first.
```

### 12.3 ACF and PACF

```text
FUNCTION ACF(Y, max_lag)
    Ȳ ← mean(Y);  c0 ← Σ (Y[t] - Ȳ)²
    FOR k IN 0..max_lag:
        ρ[k] ← Σ_{t=k+1}^{n} (Y[t] - Ȳ)(Y[t-k] - Ȳ) / c0
    band ← ± 1.96 / sqrt(n)
    RETURN ρ, band

FUNCTION PACF(Y, max_lag)
    FOR k IN 1..max_lag:
        fit OLS:  Y[t] = c + φ1·Y[t-1] + ... + φk·Y[t-k] + e
        pacf[k] ← φk                                 # direct effect of lag k
    RETURN pacf
```

### 12.4 Stationarity check

```text
PROCEDURE CheckStationarity(Y, α = 0.05)
    p_adf  ← ADF(Y).p_value          # H0: non-stationary
    p_kpss ← KPSS(Y).p_value         # H0: stationary
    IF p_adf < α AND p_kpss ≥ α:  RETURN "stationary"
    IF p_adf ≥ α AND p_kpss < α:  RETURN "non-stationary → difference / log and re-test"
    RETURN "inconclusive → inspect plot, sample size, structural breaks"
```

Python reference:

```python
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
y.rolling(7).mean(); y.rolling(13, center=True).mean(); y.ewm(alpha=0.3, adjust=False).mean()
adfuller(y)[1]; kpss(y, regression="c")[1]
plot_acf(y, lags=24); plot_pacf(y, lags=24, method="ywm")
```

---

## 13. Interview questions

1. **Why can't you use ordinary linear regression directly on a time series?** Observations are autocorrelated (violates independence); residuals are correlated → biased inference; also risk of spurious regression with trends.
2. **Define stationarity.** Constant mean, constant variance, and autocovariance depending only on the lag.
3. **No trend and no seasonality → stationary?** Not necessarily: level shifts or volatility changes break stationarity.
4. **ADF vs. KPSS?** Opposite null hypotheses (ADF: unit root/non-stationary; KPSS: stationary). Use both.
5. **How do you make a series stationary?** Differencing (regular and/or seasonal), log/Box-Cox transforms to stabilize variance, detrending.
6. **ACF vs. PACF?** ACF is total correlation with lag k; PACF is the direct correlation with lag k after removing intermediate lags.
7. **What does a slowly decaying ACF indicate?** Trend / non-stationarity (unit root).
8. **What does an ACF with peaks at 12, 24, 36 indicate?** Monthly seasonality with period 12.
9. **What is white noise and why do we want residuals to be white noise?** Uncorrelated, constant-variance noise; if residuals are white noise, the model captured all predictable structure.
10. **Additive vs. multiplicative decomposition?** Additive when seasonal amplitude is constant; multiplicative when it scales with the level (log makes it additive).
11. **Is a moving average a forecasting model?** In this context (smoothing), no — it's descriptive. (Don't confuse with the MA(q) model in ARIMA, which models past *errors*.)

---

## 14. References

- Dickey, D. A.; Fuller, W. A. "Distribution of the Estimators for Autoregressive Time Series with a Unit Root." *JASA*, 1979.
- Hyndman, R. J.; Athanasopoulos, G.; Garza, A.; Challu, C.; Mergenthaler, M.; Olivares, K. G. *Forecasting: Principles and Practice, the Pythonic Way*. OTexts, 2024. <https://otexts.com/fpppy> *(recommended reading)*
- Kwiatkowski, D. et al. "Testing the Null Hypothesis of Stationarity Against the Alternative of a Unit Root." *Journal of Econometrics*, 1992.
- Morettin, P. A.; Toloi, C. M. C. *Análise de Séries Temporais*. 2nd ed. Blucher, 2006.
- Shumway, R. H.; Stoffer, D. S. *Time Series Analysis and Its Applications*. 2nd ed. Springer, 2006.

---

⬅️ Previous: [1.7 — Applied Python for Modeling](../../01%20-%20Artificial%20Intelligence%20I/1.7%20-%20Applied%20Python%20for%20Modeling/README.md) · 🏠 [Index](../../README.md) · ➡️ Next: [2.2 — Time Series Forecasting Models](../2.2%20-%20Time%20Series%20Forecasting%20Models/README.md)
