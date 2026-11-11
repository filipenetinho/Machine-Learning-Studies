# 1.7 — Extra Topic: Applied Python for Modeling

> **Module:** Artificial Intelligence I (extra topic)
> **Case study:** freight pricing — 1 million shipments with the characteristics known at quote time and the actual cost incurred (stored as **Parquet**: ~16 MB on disk vs. ~100 MB as text).

Six practical topics across the modeling lifecycle:

| Stage | Topic |
|---|---|
| Data preparation & visualization | **Polars** (fast data manipulation) · **Plotly** (interactive charts) |
| Model building | **Optuna** (history-guided hyperparameter optimization) |
| Model diagnostics | **Segmented error analysis** |
| Model use | **Prediction intervals** · **Model monitoring (drift, PSI)** |

---

## Table of Contents

1. [The freight-pricing case](#1-the-freight-pricing-case)
2. [Polars](#2-polars)
3. [Plotly](#3-plotly)
4. [Optuna](#4-optuna)
5. [Segmented error analysis](#5-segmented-error-analysis)
6. [Prediction intervals](#6-prediction-intervals)
7. [Model monitoring: drift and PSI](#7-model-monitoring-drift-and-psi)
8. [Pseudocode](#8-pseudocode)
9. [Interview questions](#9-interview-questions)
10. [References](#10-references)

---

## 1. The freight-pricing case

A carrier must estimate each shipment's **freight cost** to support pricing and margin analysis. Costs vary with weight, distance, transport mode and service level.

| Variable | Description |
|---|---|
| `distancia_km` | Distance (km) |
| `peso_kg` | Cargo weight (kg) |
| `volume_m3` | Cargo volume (m³) |
| `nro_volumes` | Number of packages |
| `modal_transporte` | Mode: road, air, rail or river |
| `tipo_carga` | Cargo type: regular, fragile, hazardous |
| `regiao_destino` | Destination macro-region |
| `nivel_urgencia` | Service level: economy, normal, express, super-express |
| `zona_destino` | Urban or rural |
| `necessita_seguro` | Insurance contracted |
| `transportadora` | Carrier operator |
| `tipo_embalagem` | Packaging |
| `faixa_horario_coleta` | Pickup time slot |
| `custo_frete` | **Freight cost (R$) — target** |

EDA highlights: cost ranges from **R$ 7 to R$ 85,349**; on a log scale it looks roughly normal (log-normal target). Huge cost differences by mode — river and rail have far higher average costs than road and air (which concentrate most shipments).

---

## 2. Polars

**Polars** is a DataFrame library written in **Rust** that parallelizes processing across CPU cores.

- Operations are **expressions** inside `select`, `filter`, `with_columns`, `group_by`. Chaining them reads like a **SQL query**.
- Supports **lazy execution** (`scan_parquet` + `collect`) with query optimization.
- Speed-ups grow with data volume; on a few thousand rows pandas and Polars take about the same time.
- Polars DataFrames are accepted directly by scikit-learn.
- Docs: <https://docs.pola.rs>

Benchmarks in class (1M rows, local machine):

| Operation | Speed-up vs. pandas |
|---|---|
| Read the whole file | **26.9×** |
| Filter air shipments above 500 kg | **24.1×** |
| Mean and median cost by mode | 2.4× |

> The gain is biggest for full scans (I/O) and shrinks as the operation becomes compute-bound.

```python
import polars as pl
df = pl.read_parquet("frete.parquet")
(df.filter((pl.col("modal_transporte") == "aereo") & (pl.col("peso_kg") > 500))
   .group_by("modal_transporte")
   .agg(pl.col("custo_frete").mean().alias("mean"),
        pl.col("custo_frete").median().alias("median")))
```

---

## 3. Plotly

**Plotly** creates **interactive** charts: hover tooltips, zoom, range selection.

- `plotly.express` builds a chart in a single call; accepts pandas and Polars DataFrames.
- Export to **HTML** (keeps interactivity) or static images for reports.
- Docs: <https://plotly.com/python>

Case charts: mean cost by mode (bar), cost distribution (histogram on **log scale** to fit R$ 7 – R$ 85k), cost vs. weight by mode (scatter), cost by urgency (boxplot).

> 💡 For plots that draw individual points, use a **sample** (here 100,000 shipments) — plotting 1M points is slow and unreadable.

```python
import plotly.express as px
px.histogram(df, x="custo_frete", log_x=True)
px.scatter(df.sample(100_000), x="peso_kg", y="custo_frete", color="modal_transporte", opacity=.3)
fig.write_html("chart.html")
```

---

## 4. Optuna

**Problem with random search:** each combination is drawn **independently** — the last trial learns nothing from previous ones.

**Optuna** is an optimization framework that samples hyperparameters **guided by history**: at each trial, it models which regions of the search space have produced the best results and concentrates the next draws there.

- Default sampler: **TPE — Tree-structured Parzen Estimator** (Bergstra et al., 2011). (A form of Bayesian optimization: it models the density of "good" vs. "bad" hyperparameters and samples where the ratio good/bad is highest.)
- Library: `optuna` (Akiba et al., 2019) — <https://optuna.org>

**Pruning:** bad trials don't need to run to completion. After each CV fold, the partial result is reported; if the trial is already **worse than the median of previous trials** at that step (MedianPruner), it's stopped and the budget goes to the next combination. → more combinations tested in the same time, focused on the most promising.

### Case strategy

- Sample of **20,000** observations for modeling.
- Holdout 80/20 + 5-fold CV.
- Algorithm: **XGBoost** (LightGBM and RF are good alternatives).
- **60 Optuna trials (TPE)** over:

| Hyperparameter | Range |
|---|---|
| `objective` | `reg:squarederror` or **`reg:gamma`** |
| `n_estimators` | 100 – 500 |
| `max_depth` | 2 – 10 |
| `learning_rate` | 0.03 – 0.3 (log scale) |
| `subsample` | 0.6 – 1.0 |
| `colsample_bytree` | 0.5 – 1.0 |
| `gamma` | 0 – 5 |
| `reg_alpha`, `reg_lambda` | 1e-5 – 100 (log scale) |

- **Model choice by lowest mean MAPE** on internal-test folds — a **relative** metric (%) instead of absolute (R$), because cost levels are very heterogeneous and absolute metrics would be dominated by expensive shipments.

**Gamma deviance** — an alternative loss for **positive, right-skewed** targets. It weighs the error *relative to the predicted value*, preventing large-magnitude observations from dominating the fit:

$$D = 2\sum_i\left[\frac{y_i - \hat y_i}{\hat y_i} - \log\frac{y_i}{\hat y_i}\right]$$

Results:
- **Search history:** almost all the gain comes from the first trials; from halfway on, little error reduction.
- **Hyperparameter importance** (Optuna's fANOVA-style plot): low-importance hyperparameters can be **fixed** in a next round, freeing budget for the others.
- **Optuna vs. random search** (same trial budget, same pruning rule): MAPE **0.225 vs. 0.227** — a modest gain here; the advantage grows with larger/more complex search spaces.

---

## 5. Segmented error analysis

**A single error metric for the whole dataset doesn't say where the model works better or worse.** With costs from R$ 7 to R$ 85k, **absolute error is dominated by expensive shipments** and **percentage error by cheap ones**. → Segment the metrics.

**New metric — Mean Percentage Error (MPE, "mean bias"):**

$$MPE = \frac{1}{n}\sum_i \frac{y_i - \hat y_i}{y_i}$$

It keeps the sign, so errors cancel → it indicates **direction**, not magnitude.
- **MPE > 0** → model tends to **under-predict** (ŷ < y).
- **MPE < 0** → model tends to **over-predict** (ŷ > y).

Case findings:
- **By mode:** MAPE well balanced across modes, 22–24%.
- **By cost band:** MAPE a bit higher for **cheaper** shipments, where the model tends to **over-predict**; for the **most expensive** band, error is lower and the direction flips (**under-predicts**). (A typical "regression to the mean" pattern.)

---

## 6. Prediction intervals

A point prediction doesn't say how much we can trust it. As we built confidence intervals for parameters, we can build **prediction intervals** for the target. (CI = uncertainty about a *parameter*; PI = range of plausible values for a *new observation* — always wider.)

**Empirical (residual-quantile) method — two steps:**
1. **Measure error on held-out data** (from holdout + CV): absolute residual (y − ŷ) or percentage residual (y − ŷ)/ŷ.
2. **Take percentiles of the errors.** E.g., P5 and P95 → an interval with **90% target coverage**, to be checked on data separate from the one that produced the percentiles (external test).

With percentage residuals, multiply the percentile by ŷ to return to the original scale:

$$PI_{90\%}(x) = \left[\hat y\,(1 + q_{0.05}),\ \hat y\,(1 + q_{0.95})\right]$$

(This is the idea behind **split conformal prediction**.)

Evaluate:
- **Coverage:** % of cases where the actual value falls inside the interval — intended vs. observed on external test.
- **Width:** distance between the bounds — how much imprecision the interval admits (absolute or %).

Case: in ~**90%** of shipments the actual cost falls inside the band; coverage holds in **every cost band** (slightly higher, ~92%, in middle bands). A log-scale chart helps visualize intervals around small values.

---

## 7. Model monitoring: drift and PSI

A model is valid only while the **data-generating mechanism** stays essentially the same.
- **Abrupt changes:** pandemic lockdown, new regulation, disruptive product launch.
- **Gradual changes:** customer behavior, economic scenario, service quality.

**Data drift** (input distributions change) may or may not cause **prediction drift** (distorted predictions).

Related concepts:
- **Covariate / data drift:** P(X) changes.
- **Prediction drift:** P(ŷ) changes.
- **Concept drift:** the relationship P(Y | X) changes.

### Population Stability Index (PSI)

Compares the distribution in a **new period** with the **reference** distribution (usually the model-building data). Works for **features or predictions**.

$$PSI = \sum_i (p_i - q_i)\cdot\ln\!\left(\frac{p_i}{q_i}\right)$$

pᵢ, qᵢ = share of observations in bin i in the new period and in the reference.

- Bins are defined **once on the reference** and reused: quantitative variables → **deciles**; qualitative → their categories.
- Market rule of thumb:

| PSI | Reading |
|---|---|
| ≤ 0.10 | Stable |
| 0.10 – 0.20 | Moderate shift |
| > 0.20 | Large shift |

### Case scenarios

| Scenario | What changed | PSI result |
|---|---|---|
| **A** | Operation as before | Stable |
| **B** | Portfolio moved to farther destinations and to rail/river | Shift in **price-forming variables and in predictions** (crosses 0.20) |
| **C** | Different carrier, standard packaging and pickup time; plus a fuel-price hike made transport pricier (mainly road and air) | Shift in 3 **low-weight** variables (crosses 0.20) but **predictions unaffected** |

> ⚠️ **PSI only detects changes in marginal distributions** of features or predictions. If the structural change is only in the **relationship between features and target** (concept drift — like the fuel hike in scenario C), PSI stays stable but prediction quality drops. Only **error analysis** catches this — which requires waiting for the actual outcome. PSI, on the other hand, can be computed **at scoring time**, before the outcome is known.

---

## 8. Pseudocode

### 8.1 Optuna with pruning

```text
FUNCTION Objective(trial)
    params ← {
        objective:        trial.choice(["reg:squarederror", "reg:gamma"]),
        n_estimators:     trial.int(100, 500),
        max_depth:        trial.int(2, 10),
        learning_rate:    trial.float(0.03, 0.3, log = TRUE),
        subsample:        trial.float(0.6, 1.0),
        colsample_bytree: trial.float(0.5, 1.0),
        gamma:            trial.float(0, 5),
        reg_alpha:        trial.float(1e-5, 100, log = TRUE),
        reg_lambda:       trial.float(1e-5, 100, log = TRUE) }
    scores ← []
    FOR fold f IN 1..5:
        model ← XGBRegressor(params).fit(train_part(f))
        scores.append( MAPE(val_part(f), model.predict(val_part(f))) )
        trial.report(mean(scores), step = f)
        IF trial.should_prune():                       # worse than median of previous trials
            RAISE TrialPruned
    RETURN mean(scores)

study ← create_study(direction = "minimize", sampler = TPE, pruner = MedianPruner)
study.optimize(Objective, n_trials = 60)
best_params ← study.best_params
```

### 8.2 TPE intuition

```text
REPEAT for each new trial:
    split past trials into GOOD (best γ%, e.g. 20%) and BAD (rest) by score
    l(h) ← density estimate of hyperparameters in GOOD
    g(h) ← density estimate of hyperparameters in BAD
    draw many candidates from l(h)
    choose candidate maximizing l(h) / g(h)        # ∝ expected improvement
    evaluate it, add to history
```

### 8.3 Segmented error analysis

```text
FUNCTION SegmentedErrors(y, ŷ, segment)
    FOR each group g in unique(segment):
        idx ← segment == g
        MAE[g]  ← mean(|y[idx] - ŷ[idx]|)
        MAPE[g] ← mean(|y[idx] - ŷ[idx]| / y[idx])
        MPE[g]  ← mean((y[idx] - ŷ[idx]) / y[idx])     # >0 under-predicts, <0 over-predicts
        n[g]    ← count(idx)
    RETURN table(MAE, MAPE, MPE, n)
# segments: transport mode, cost band (quantiles of y), region, ...
```

### 8.4 Empirical prediction interval

```text
FUNCTION FitInterval(oof_y, oof_ŷ, coverage = 0.90)
    r ← (oof_y - oof_ŷ) / oof_ŷ                         # percentage residuals on held-out folds
    lo ← quantile(r, (1 - coverage)/2)                   # P5
    hi ← quantile(r, 1 - (1 - coverage)/2)               # P95
    RETURN lo, hi

FUNCTION PredictInterval(ŷ_new, lo, hi)
    RETURN [ŷ_new · (1 + lo), ŷ_new · (1 + hi)]

FUNCTION EvaluateInterval(y_test, lower, upper)
    coverage ← mean(lower ≤ y_test ≤ upper)              # should be ≈ 0.90
    width    ← mean((upper - lower) / ŷ_test)
    RETURN coverage, width
```

### 8.5 PSI

```text
FUNCTION PSI(reference, new, n_bins = 10, eps = 1e-6)
    IF variable is quantitative:
        edges ← deciles of reference                     # defined ONCE, reused every period
        q ← histogram(reference, edges) / |reference|
        p ← histogram(new, edges) / |new|
    ELSE:
        categories ← categories in reference
        q ← frequency of each category in reference
        p ← frequency of each category in new
    q ← max(q, eps);  p ← max(p, eps)                    # avoid log(0)
    RETURN Σ (p - q) · ln(p / q)

# monitoring: for each feature and for the predictions, compute PSI each period
#   ≤ 0.10 stable | 0.10–0.20 moderate | > 0.20 large → investigate / retrain
# when actual outcomes arrive, also track MAPE/MPE over time (catches concept drift)
```

---

## 9. Interview questions

1. **Why might Polars be faster than pandas?** Rust implementation, multi-threaded, columnar (Arrow) memory, lazy query optimization.
2. **Why Parquet over CSV?** Columnar, compressed, typed schema, fast column-selective reads.
3. **Random search vs. Bayesian optimization (TPE)?** Random draws are independent; TPE uses past results to focus sampling on promising regions.
4. **What is pruning in Optuna?** Early stopping of unpromising trials based on intermediate results (e.g., worse than the median).
5. **Why choose MAPE over RMSE for freight pricing?** Targets span orders of magnitude; RMSE would be dominated by expensive shipments; MAPE evaluates relative error.
6. **When would you use a gamma objective?** Positive, right-skewed continuous targets whose variance grows with the mean (costs, claim amounts).
7. **What does MPE tell you that MAPE doesn't?** Direction of bias (systematic over- or under-prediction).
8. **Confidence interval vs. prediction interval?** CI covers a parameter (e.g., mean response); PI covers a new observation and includes irreducible noise — wider.
9. **How do you validate a prediction interval?** Check empirical coverage and width on data not used to compute the residual quantiles.
10. **What is PSI and its thresholds?** Distribution-shift metric; ≤0.1 stable, 0.1–0.2 moderate, >0.2 large.
11. **Can a model degrade with stable PSI?** Yes — concept drift (P(Y|X) changes) doesn't show in input distributions; only monitoring errors with observed outcomes reveals it.
12. **Data drift vs. concept drift?** P(X) changes vs. P(Y|X) changes.

---

## 10. References

- Akiba, T. et al. "Optuna: A Next-Generation Hyperparameter Optimization Framework." *KDD*, 2019.
- Bergstra, J. et al. "Algorithms for Hyper-Parameter Optimization." *NeurIPS*, 2011.
- Siddiqi, N. *Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring*. Wiley, 2006.
- Evidently AI — Data drift: <https://www.evidentlyai.com/ml-in-production/data-drift>
- DataCamp — Understanding data drift and model drift: <https://www.datacamp.com/tutorial/understanding-data-drift-model-drift>

---

⬅️ Previous: [1.6 — Random Forest and Boosting (Regression)](../1.6%20-%20Random%20Forest%20and%20Boosting%20for%20Regression/README.md) · 🏠 [Index](../../README.md) · ➡️ Next: [2.1 — Introduction to Time Series Analysis](../../02%20-%20Artificial%20Intelligence%20II/2.1%20-%20Introduction%20to%20Time%20Series%20Analysis/README.md)
