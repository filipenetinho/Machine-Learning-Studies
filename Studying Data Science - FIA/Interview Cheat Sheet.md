# 🎯 Interview Cheat Sheet — Data Science (FIA course review)

A quick review of the whole course. Each topic links to its full notes.

---

## 1. Which model do I use?

```
Is there a target (Y)?
├── NO → UNSUPERVISED (2.6)
│        ├── Group similar observations ......... hierarchical (small n) · k-means (large n) · DBSCAN (odd shapes / noise)
│        └── Compress many columns .............. PCA / dimensionality reduction
└── YES → SUPERVISED
         ├── Y quantitative (regression / "projection")
         │     ├── indexed by time, forecast the future (2.1, 2.2)
         │     │     stationary → ARMA · trend → ARIMA · seasonality → SARIMA / Prophet
         │     │     exogenous drivers → SARIMAX / Prophet + regressors · non-linear → trees + lag features
         │     └── not temporal
         │           explain drivers → linear regression (1.3) (+ regularization, 1.4)
         │           predict, non-linear → regression tree (1.5) → RF / boosting (1.6)
         └── Y qualitative (classification)
               ├── binary → logistic regression (2.3) · decision tree (2.4) · RF / boosting (2.5)
               └── multinomial → multinomial tree / ensembles (2.4, 2.5)
```

**Statistical learning** (linear/logistic regression, ARIMA) → interpretation and inference.
**Machine learning** (trees, RF, boosting) → predictive performance.

---

## 2. Formulas you should know by heart

| Concept | Formula |
|---|---|
| Standard deviation | $s = \sqrt{\sum(x_i-\bar x)^2/(n-1)}$ |
| Coefficient of variation | $CV = s/\bar x$ |
| Pearson correlation | $r = \dfrac{\sum(x-\bar x)(y-\bar y)}{\sqrt{\sum(x-\bar x)^2\sum(y-\bar y)^2}}$ |
| OLS slope | $\hat\beta_1 = \dfrac{\sum(x-\bar x)(y-\bar y)}{\sum(x-\bar x)^2}$, $\hat\beta_0 = \bar y - \hat\beta_1\bar x$ |
| R² / adjusted R² | $1 - \dfrac{SS_{res}}{SS_{tot}}$ · $1-(1-R^2)\dfrac{n-1}{n-k-1}$ |
| VIF | $1/(1-R_j^2)$ |
| Ridge / LASSO penalty | $\alpha\sum\beta_j^2$ · $\alpha\sum\lvert\beta_j\rvert$ |
| Gradient step | $\beta \leftarrow \beta - \eta\nabla J(\beta)$ |
| Logistic | $p = 1/(1+e^{-(\beta_0+\beta_1x)})$ · $\ln\frac{p}{1-p} = \beta_0+\beta_1x$ · OR $= e^{\beta}$ |
| Log-loss | $-\sum[y\ln p + (1-y)\ln(1-p)]$ |
| Gini / entropy | $1-\sum p_k^2$ · $-\sum p_k\log_2 p_k$ |
| Impurity reduction | $\Delta I = I(\text{parent}) - \sum \frac{n_c}{n}I(c)$ |
| GB update | $F_m = F_{m-1} + \eta\,h_m$, $h_m$ fit to residuals (MSE) or $y-p$ (log-loss) |
| AR(p) / MA(q) | $Y_t = c+\sum\phi_iY_{t-i}+\varepsilon_t$ · $Y_t = \mu+\sum\theta_j\varepsilon_{t-j}+\varepsilon_t$ |
| Differencing | $Y_t - Y_{t-1}$ · seasonal: $Y_t - Y_{t-m}$ |
| PSI | $\sum (p_i-q_i)\ln(p_i/q_i)$ — ≤ 0.10 stable, 0.10–0.20 moderate, > 0.20 large |
| Brier score | $\frac{1}{n}\sum(\hat p_i - y_i)^2$ |
| Euclidean / z-score | $\sqrt{\sum(x_i-x_j)^2}$ · $(x-\bar x)/s$ |
| SHAP additivity | $\hat y_i = \hat y_0 + \sum_j\phi_i(X_j)$ |

---

## 3. Which metric for which problem

| Problem | Metrics | Watch out |
|---|---|---|
| Regression | RMSE (penalizes big errors) · MAE (robust, easy to explain) · MAPE (relative) · R² | MAPE breaks when y ≈ 0; RMSE is dominated by large values |
| Regression, values on very different scales | MAPE / MPE by segment, gamma deviance | Look at errors **by segment** (1.7) |
| Forecasting | RMSE / MAE on an **out-of-time** test ≥ horizon | Always compare with naive / seasonal naive / drift |
| Binary classification | AUC (ranking) · sensitivity / specificity / PPV / NPV at a cutoff · Brier (calibration) | Accuracy is misleading with imbalanced classes |
| Multiclass | AUC-OVR / OVO · balanced accuracy · per-class recall and precision | Plain accuracy reflects the majority class |
| Clustering | Within vs. between homogeneity, dendrogram jumps, WCSS / elbow, silhouette | **Interpretability** and business use decide |

**Confusion matrix:**

```
                 Pred 0     Pred 1
   Obs 0          TN         FP        specificity = TN/(TN+FP)
   Obs 1          FN         TP        sensitivity = TP/(TP+FN)
             NPV=TN/(TN+FN) PPV=TP/(TP+FP)       accuracy = (TP+TN)/N
```

---

## 4. Validation in one picture

```
NON-TEMPORAL:  [ TRAIN 80%  → k-fold CV for hyperparameters ] [ EXTERNAL TEST 20% ]  (stratify for classification)
TEMPORAL:      [ train → | test ] [ train ———→ | test ] ...  (TimeSeriesSplit) ... [ MOST RECENT = out-of-time test ]
```

Choose the model with **low internal-test error**, **stable vs. train** (no overfitting) and **stable vs. external test** (generalizes). Then **retrain on the full training set**.

Tuning: grid search (exhaustive) → **random search** (best cost/benefit) → Bayesian / **Optuna TPE** with pruning.

---

## 5. Concepts interviewers love

| Topic | One-line answer |
|---|---|
| **Bias–variance** | Simple → high bias ("bad but stable"); complex → high variance ("good but unstable") |
| **Overfitting signs** | Train error ≪ test error; trees with many splits and tiny leaves |
| **Data leakage** | Features using information unavailable at prediction time; preprocessing fitted on test data |
| **p-value** | P(estimate at least this extreme \| H₀ true) — *not* P(H₀ true) |
| **Type I / II** | False positive (α) / false negative |
| **Collinearity** | Hurts interpretation and coefficient stability, much less prediction → VIF, drop variables, Ridge |
| **Dummy trap** | Drop one category (the reference); its effect goes into the intercept |
| **Scaling** | Needed for regularization, SGD and clustering; **not** for trees |
| **Bagging vs. boosting** | Parallel + average (↓ variance) vs. sequential + fix errors (↓ bias) |
| **GB residuals** | Each tree fits the negative gradient: residuals (MSE) or y − p (log-loss) |
| **Feature importance** | Impurity (fast, biased toward high cardinality) · permutation (model-agnostic, hurt by correlated features) · SHAP (direction + local) — none proves causality |
| **Imbalance** | Cutoff = event rate, class weights, stratification, AUC / balanced accuracy, then **calibrate** |
| **Calibration** | Monotonic → AUC barely changes; Brier ↓; needed for pricing / expected loss |
| **AUC > 0.95** | Suspect leakage |
| **Stationarity** | Constant mean, variance and lag-only autocovariance; ADF (H₀ non-stationary) + KPSS (H₀ stationary) |
| **ACF / PACF** | PACF cuts off at p → AR(p); ACF cuts off at q → MA(q); slow ACF decay → trend; peaks at 12, 24 → seasonality |
| **Trees and time** | Trees don't see time or extrapolate → lag/rolling/calendar features from lag ≥ 1, recursive forecasting |
| **Exogenous variables** | You need their **future** values (scenarios or separate models; errors accumulate) |
| **Drift** | Data drift P(X), prediction drift P(ŷ), concept drift P(Y\|X) — PSI can't detect concept drift |
| **k-means vs. hierarchical vs. DBSCAN** | k-means: fast, needs k, globular shapes · hierarchical: dendrogram, small n · DBSCAN: arbitrary shapes, noise, no k |

---

## 6. Common traps

- A segment coded 1/2/3 is **qualitative**.
- **Mean ≫ median** → right skew; report the median.
- Average growth rates → **geometric** mean; average speeds over equal distances → **harmonic** mean.
- Same standard deviation ≠ same variability → compare **CV**.
- Stepwise backward: remove **one variable at a time**.
- Regular R² always goes up when you add variables → use **adjusted R²**.
- A 50% cutoff with 2% events → model looks 98% accurate and catches almost nothing.
- Weighted trees show **50/50 at the root**: the probabilities are **not** real frequencies until calibrated.
- Prophet can extrapolate into impossible values (unemployment < 4%) because it has no autoregressive terms.
- With 53 observations and well-behaved seasonality, **SARIMAX beat XGBoost**: more flexibility doesn't always win.
- Removing an exogenous variable can cost nothing: GDP was **not significant** for unemployment (p = 0.55).
- Business-rule quadrants ≠ natural clusters.

---

🏠 [Back to index](README.md)
