# 1.6 — Projection Models: Random Forest and Boosting (Regression)

> **Module:** Artificial Intelligence I
> **Case study:** the same bookstore cost-price problem from 1.5, now with tree **ensembles** (Random Forest, AdaBoost, Gradient Boosting, HistGB, XGBoost, LightGBM, CatBoost).
> **Extra case:** fuel efficiency (km/l) of 500 monitored vehicles.

Tree ensembles are currently the **state of the art for tabular data**. They capture more complex patterns than a single tree, at a higher computational cost.

---

## Table of Contents

1. [Why ensembles?](#1-why-ensembles)
2. [Random Forest (bagging)](#2-random-forest-bagging)
3. [Boosting: the core idea](#3-boosting-the-core-idea)
4. [AdaBoost](#4-adaboost)
5. [Gradient Boosting](#5-gradient-boosting)
6. [Missing values in tree models](#6-missing-values-in-tree-models)
7. [Modern gradient-boosting variants](#7-modern-gradient-boosting-variants)
8. [Hyperparameter cheat-sheet](#8-hyperparameter-cheat-sheet)
9. [Case: bookstore results](#9-case-bookstore-results)
10. [Additional case: fuel efficiency](#10-additional-case-fuel-efficiency)
11. [Pseudocode](#11-pseudocode)
12. [Interview questions](#12-interview-questions)
13. [References](#13-references)

---

## 1. Why ensembles?

An **ensemble** combines many models into one. For projection, the ensemble members are **regression trees**.

| Algorithm | Authors / year |
|---|---|
| Random Forest | Breiman, 2001 |
| AdaBoost | Freund & Schapire, 1995 |
| Gradient Boosting | Friedman, 1999/2001 |
| Histogram-based Gradient Boosting | (Friedman's GB with binned features; scikit-learn implementation inspired by LightGBM) |
| XGBoost | Chen & Guestrin, 2016 |
| LightGBM | Ke et al., 2017 |
| CatBoost | Prokhorenkova, Gusev, Gulin et al. (Yandex), 2018 |

**Two families:**

```
BAGGING (Random Forest)                    BOOSTING (AdaBoost, GBM, XGBoost...)
 trees built in PARALLEL, independently     trees built SEQUENTIALLY, each fixing the previous
 deep-ish trees, each on a bootstrap         shallow "weak" trees (stumps / small depth)
 final = AVERAGE of trees                    final = WEIGHTED SUM of trees
 mainly reduces VARIANCE                     mainly reduces BIAS (and variance with shrinkage)
```

---

## 2. Random Forest (bagging)

**Bagging = Bootstrap AGGregatING.**

Algorithm steps:

| Step | What happens |
|---|---|
| 1. **Bagging** | Draw a **subsample with replacement** (bootstrap) — typically 60–100% of the observations |
| 2. **Tree** | Build a tree; at **each split**, pick the best among **m features randomly drawn** from the k available (usually m = √k, log₂k, or k) |
| 3. **Partial prediction** | Predict each observation with that tree |
| 4. **Forest** | Repeat 1–3 many times → a set of "parallel" trees |
| 5. **Final prediction** | **Average** of the trees' predictions |

> 💡 **Why it works:** averaging many low-bias but high-variance trees reduces variance. The two sources of randomness (bootstrap rows + random feature subsets per split) **decorrelate** the trees — averaging correlated trees helps much less.

### Hyperparameters (`RandomForestRegressor`)

Ensembles can overfit even more → tune with random search + CV.

| Hyperparameter | Meaning | Usual values |
|---|---|---|
| `n_estimators` 🆕 | Number of trees in the forest | 100 – 500 (more for complex patterns) |
| `criterion` | Impurity: `"squared_error"` (MSE, default), `"absolute_error"` (MAE — less weight to outliers, slower), `"poisson"` (for count targets) | `squared_error` |
| `min_impurity_decrease` | Minimum impurity reduction to split | 0 – 0.05 |
| `min_samples_split` | Min. obs. in parent | data-dependent |
| `min_samples_leaf` | Min. obs. in children | data-dependent |
| `max_depth` | Max levels | 2 – 10 |
| `max_samples` 🆕 | % of observations per tree (bootstrap size) | 0.6 – 1.0 |
| `max_features` 🆕 | # features considered per split | `"sqrt"`, `"log2"`, `None` (all) |

Metrics: RMSE, MAE, MAPE, R² (same as before).

> ⚠️ With a forest there's **no single tree to visualize** → interpretation relies on importance measures and SHAP.

---

## 3. Boosting: the core idea

Boosting combines many **weak learners** (usually shallow trees) into a strong model. The core idea:

> **Sequentially fit new models that fix the errors of the previous ones, focusing learning on the hardest-to-predict observations.**

The oldest variant is **AdaBoost** (adaptive boosting), by Yoav Freund and Robert Schapire (1995).

---

## 4. AdaBoost

| Step | What happens |
|---|---|
| 1. **Initial tree** | Fit a simple / weak tree (a **stump**) using **observation weights** (equal at the first iteration) |
| 2. **Error** | Compute the tree's error metric, **weighted** by observation weights |
| 3. **Reweighting** | Give the **tree** a weight (lower error → higher weight) and **update observation weights** (larger residual → larger weight increase) |
| 4. **Boosting** | Repeat 1–3 → a **sequential** set of weak trees |
| 5. **Final prediction** | Average of the trees' predictions **weighted by tree weights** (for regression, scikit-learn's AdaBoost.R2 uses a weighted median) |

**Learning rate (η):** shrinks each tree's contribution → more gradual learning, less overfitting. Early trees fix big errors; later ones refine specific patterns.

- η too **small** → slow learning; may not converge with few trees (**underfitting**).
- η too **large** → converges quickly on training data but generalizes poorly (**overfitting**).
- η is a central hyperparameter → tune via CV (it trades off with `n_estimators`).

| Hyperparameter (`AdaBoostRegressor`) | Usual values |
|---|---|
| `n_estimators` | 10 – 500 |
| `learning_rate` 🆕 | 0.0001 – 0.1 |

---

## 5. Gradient Boosting

Built on the **gradient** concept (the same as SGD in 1.4).

**Intuition:** many models learn by minimizing a cost function. For a 1-parameter model, each β gives an error; the **slope of the tangent** (the gradient) points to where the error rises; at the minimum, the slope is zero. Algorithms take steps "downhill"; the **learning rate** is the step size (too big → overshoot; too small → slow).

> 💡 **In gradient boosting, instead of optimizing a parameter, we optimize the predictions themselves: each new tree is one step downhill.**

| Step | What happens |
|---|---|
| 1. **Initial prediction** | A constant for all observations — usually the **mean** of Y |
| 2. **Residuals** | Compute the (negative) **gradient of the loss** w.r.t. the predictions — for **MSE this equals the residuals** (y − ŷ) |
| 3. **Fit a tree** | Fit a weak tree using the **residuals as the target** |
| 4. **Boost** | Update predictions: **ŷ ← ŷ + η · tree(x)** (learning rate smooths each tree's impact → prevents overfitting) |
| 5. **Repeat** | Steps 2–4 many times. **Final prediction = initial prediction + sum of all trees' (scaled) contributions** |

Technical note: the gradient is the partial derivative of the loss with respect to the model's predictions; we move along the **negative** gradient. For L = ½(y − ŷ)², ∂L/∂ŷ = −(y − ŷ) → the negative gradient is exactly the residual (chain rule + derivative of a sum).

$$F_0(x) = \bar y,\qquad F_m(x) = F_{m-1}(x) + \eta\, h_m(x),\quad h_m \approx -\frac{\partial L(y, F_{m-1}(x))}{\partial F_{m-1}(x)}$$

### Hyperparameters (`GradientBoostingRegressor`)

| Hyperparameter | Meaning | Usual values |
|---|---|---|
| `n_estimators` | # trees in the sequence | 10 – 300 |
| `learning_rate` | η — fraction of the error corrected by each new tree | 0.0001 – 0.1 |
| `min_impurity_decrease` | | 0 – 0.05 |
| `min_samples_split`, `min_samples_leaf` | | data-dependent |
| `subsample` | % of observations per tree (→ **stochastic** gradient boosting) | 0.6 – 1.0 |
| `max_features` | # features per tree/split | `sqrt`, `log2`, `None` |
| `max_depth` | | 2 – 10 |

---

## 6. Missing values in tree models

| Implementation (scikit-learn) | Missing values in quantitative features |
|---|---|
| `DecisionTreeRegressor`, `RandomForestRegressor` | **Native support**: at each split, the algorithm decides whether missing values go left or right, whichever reduces impurity most |
| `AdaBoostRegressor`, `GradientBoostingRegressor` | **No support** → treat beforehand, or use variants that handle missing values (HistGB, XGBoost, LightGBM) |
| Qualitative features | One-hot encoding already handles missings as a separate category |

---

## 7. Modern gradient-boosting variants

### Histogram-based Gradient Boosting (`HistGradientBoostingRegressor`)
Optimized GB: continuous features are **discretized into bins** (like a histogram) before building trees; best splits are searched among bins, not over continuous scales.
- ✅ Faster training, especially on large data · lower memory (compact bin indices) · built-in **L2 regularization** · **native missing values**
- ❌ Sensitive to the number of bins (no universal value)

### XGBoost — Extreme Gradient Boosting (`XGBRegressor`)
Adds **penalty terms (α — L1, λ — L2)** to the loss so no node/tree dominates residual reduction → extra overfitting protection.
- ✅ Parallelism at tree-construction level (`n_jobs=-1`) · **automatic pruning** of low-gain splits (`gamma`) · L1 + L2 regularization · native missing values
- ❌ More hyperparameters (more tuning effort) · memory and time can grow a lot on large datasets

XGBoost's regularized objective (for reference):

$$\mathcal{L} = \sum_i l(y_i,\hat y_i) + \sum_t\left[\gamma T_t + \tfrac{1}{2}\lambda\lVert w_t\rVert^2 + \alpha\lVert w_t\rVert_1\right]$$

(T = # leaves, w = leaf weights.)

### LightGBM (`LGBMRegressor`)
Uses histograms (like HistGB) and **leaf-wise growth**: always splits the leaf with the largest gain → deeper trees in some branches, focused on larger local gains.
- ✅ Very fast · low memory · leaf-wise growth improves predictive performance · L1 + L2 · native missing values
- ❌ More hyperparameters; leaf-wise growth can overfit small datasets (control `num_leaves`, `min_child_samples`)

```
LEVEL-WISE (XGBoost default)          LEAF-WISE (LightGBM)
        ●                                     ●
      ●   ●                                 ●   ●
     ● ● ● ●   ← whole level              ●   ●      ← only the best leaf grows
                                         ● ●
```

### CatBoost — Categorical Boosting (`CatBoostRegressor`)
Handles **qualitative features natively and efficiently** — no manual encoding (it uses ordered target statistics).
- ✅ Native categorical support (performs *better* without encoding) · needs less tuning than other boosters
- ❌ Slower training on large data · less flexibility in loss functions · (per the course) no native missing-value support

---

## 8. Hyperparameter cheat-sheet

| Algorithm | Main hyperparameters (usual ranges) |
|---|---|
| **HistGradientBoosting** | `max_iter` (10–300) · `learning_rate` (0.0001–0.1) · `min_samples_leaf` · `max_leaf_nodes` · `max_bins` (10–255) · `l2_regularization` (1e-5–100) · `max_depth` (2–10) |
| **XGBoost** | `n_estimators` (10–300) · `learning_rate` (0.0001–0.1) · `colsample_bytree` (0.5–1.0) · `subsample` (0.6–1.0) · `gamma` (0–5, min loss reduction to split) · `reg_alpha` L1 (1e-5–100) · `reg_lambda` L2 (1e-5–100) · `max_depth` (2–10) |
| **LightGBM** | `n_estimators` (10–300) · `learning_rate` (0.0001–0.1) · `min_child_samples` · `num_leaves` · `subsample` (0.6–1.0) · `reg_alpha` · `reg_lambda` · `max_depth` (2–10) |
| **CatBoost** | `iterations` (10–200) · `learning_rate` (0.0001–0.1) · `subsample` (0.6–1.0) · `l2_leaf_reg` (1e-5–100) · `depth` (2–8) |

---

## 9. Case: bookstore results

**Common strategy:** 80/20 holdout · 5-fold CV · random search with 100 iterations · choose the lowest mean internal-test RMSE, **as long as it is stable vs. train-fold RMSE and external-test RMSE (max 10% variation in both)** · seed 123.

**Baseline (1.5):** regression tree with `min_impurity_decrease=0.0466, min_samples_split=121, min_samples_leaf=54, max_depth=5`.

### Random Forest
Search: `n_estimators` 10–500, `min_impurity_decrease` 0–0.05, `min_samples_split` 50–200, `min_samples_leaf` 20–50, `max_samples` 0.6–1, `max_features` ∈ {sqrt, log2, None}, `max_depth` 2–10.

Chosen (model 16): `n_estimators=264, min_impurity_decrease=0.0356, min_samples_split=166, min_samples_leaf=21, max_samples=0.942, max_features=None, max_depth=6`.
→ **RMSE slightly lower than the single tree.**

### Boosting (XGBoost won among AdaBoost/GB/HistGB/XGBoost/LightGBM)
Chosen (model 378): `n_estimators=297, learning_rate=0.0124, colsample_bytree=0.9262, subsample=0.6089, gamma=4.7816, reg_alpha=0.0152, reg_lambda=8.2585, max_depth=4`.
→ **RMSE a bit lower than Random Forest.**

### CatBoost (with raw categorical variables, no one-hot)
Chosen (model 81): `iterations=167, learning_rate=0.0755, subsample=0.7690, l2_leaf_reg=0.9276, depth=5`.
→ Internal-test RMSE slightly **higher**, though external-test metrics improved a little. Given the small difference and XGBoost being lighter, **XGBoost was chosen as the final model**.

> 💡 **Lesson:** when models perform about the same, prefer the simpler / lighter / more maintainable one.

XGBoost diagnostics: residuals symmetric around zero with roughly constant variance.

Importance:
- **Impurity:** category (university) > pages > color > cover.
- **Permutation:** pages > category > color > cover.
- **SHAP:** pages > category > color > cover. Higher predicted prices for books with **more pages, university category, not black print, not paperback**.

---

## 10. Additional case: fuel efficiency

An automaker monitored 500 vehicles of one model for a month to estimate **average efficiency (km/l)** from usage, driving style and preventive maintenance.

Features: % urban km, # trips, avg. trip distance, avg. speed, # harsh accelerations, # harsh brakings, A/C usage hours, avg. tire pressure (PSI), km since oil change, km since air-filter change, km since last service, avg. engine temperature, % time in traffic jams, gasoline type (regular/additive), driving style (economic/moderate/aggressive).

Exercise: (a) bivariate analysis; (b) split + one-hot fitted on train; (c) define search spaces for **regularized SGD regression, regression tree, random forest, AdaBoost, GB, HistGB, XGBoost, LightGBM**; (d) 5-fold CV + 100-iteration random search, record train / internal-test / external-test RMSE, pick and retrain the final model; (e) MAE, MAPE, R² and residuals on external test; (f) impurity, permutation and SHAP importance; (g) SHAP directions vs. business intuition.

---

## 11. Pseudocode

### 11.1 Random Forest (regression)

```text
FUNCTION RandomForestFit(X[n×k], y, n_estimators, max_samples, max_features, tree_params)
    forest ← []
    FOR t IN 1..n_estimators:
        idx ← sample round(max_samples · n) row indices WITH replacement      # bootstrap
        m ← {sqrt: √k, log2: log2(k), None: k}[max_features]
        tree ← BuildTree(X[idx], y[idx], tree_params,
                         at_each_split = consider only m randomly chosen features)
        forest.append(tree)
    RETURN forest

FUNCTION RandomForestPredict(forest, x)
    RETURN mean( tree.predict(x) FOR tree IN forest )
```

### 11.2 AdaBoost (regression, AdaBoost.R2 sketch)

```text
FUNCTION AdaBoostFit(X, y, n_estimators, η)
    w[i] ← 1/n for all i
    models ← [];  alphas ← []
    FOR t IN 1..n_estimators:
        h ← fit weak tree (stump) on (X, y) using sample weights w
        e[i] ← |y[i] - h(x[i])| / max_i |y[i] - h(x[i])|          # normalized loss ∈ [0,1]
        ε ← Σ w[i] · e[i]                                          # weighted error
        IF ε ≥ 0.5: BREAK
        β ← ε / (1 - ε)
        α ← η · log(1/β)                                            # lower error → larger tree weight
        w[i] ← w[i] · β^(η·(1 - e[i]))                              # well-predicted obs. lose weight
        w ← w / Σ w                                                 # → hard obs. gain relative weight
        models.append(h);  alphas.append(α)
    RETURN models, alphas

FUNCTION AdaBoostPredict(models, alphas, x)
    RETURN weighted_median( h(x) FOR h IN models, weights = alphas )
```

### 11.3 Gradient Boosting (regression, squared loss)

```text
FUNCTION GradientBoostingFit(X, y, n_estimators, η, max_depth, subsample)
    F0 ← mean(y)
    F[i] ← F0 for all i
    trees ← []
    FOR m IN 1..n_estimators:
        r[i] ← y[i] - F[i]                         # negative gradient of ½(y-F)² = residual
        idx ← random subset of rows of size subsample·n (without replacement)
        h ← fit regression tree on (X[idx], r[idx]) with max_depth   # weak learner on residuals
        F[i] ← F[i] + η · h(x[i])                  # small step downhill
        trees.append(h)
    RETURN F0, trees

FUNCTION GradientBoostingPredict(F0, trees, η, x)
    RETURN F0 + η · Σ h(x) FOR h IN trees
```

### 11.4 Model-comparison loop (as used in the course)

```text
FOR each algorithm A IN {SGD-regularized, Tree, RF, AdaBoost, GB, HistGB, XGBoost, LightGBM, CatBoost}:
    results[A] ← RandomizedSearchCV(A, space[A], n_iter=100, cv=5, scoring=RMSE, return_train_score)
    FOR each candidate c in results[A]:
        keep c IF |RMSE_train(c) - RMSE_internal(c)| / RMSE_internal(c) ≤ 10%
                AND |RMSE_external(c) - RMSE_internal(c)| / RMSE_internal(c) ≤ 10%
    best[A] ← stable candidate with lowest RMSE_internal
final ← argmin_A best[A].RMSE_internal  (tie → simpler / faster / easier to maintain)
RETRAIN final on full training set; report external-test RMSE, MAE, MAPE, R²; explain with SHAP
```

Python reference:

```python
from sklearn.ensemble import (RandomForestRegressor, AdaBoostRegressor,
                              GradientBoostingRegressor, HistGradientBoostingRegressor)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

rf  = RandomForestRegressor(n_estimators=264, max_depth=6, min_samples_leaf=21,
                            max_samples=0.942, max_features=None, random_state=123)
xgb = XGBRegressor(n_estimators=297, learning_rate=0.0124, max_depth=4, subsample=0.6089,
                   colsample_bytree=0.9262, gamma=4.78, reg_alpha=0.0152, reg_lambda=8.26, n_jobs=-1)
cat = CatBoostRegressor(iterations=167, learning_rate=0.0755, depth=5, l2_leaf_reg=0.93,
                        cat_features=cat_cols, verbose=0)
```

---

## 12. Interview questions

1. **Bagging vs. boosting?** Bagging trains independent models on bootstrap samples and averages them (reduces variance). Boosting trains models sequentially, each correcting its predecessors (reduces bias).
2. **Why does Random Forest sample features at each split?** To decorrelate the trees; averaging correlated trees barely reduces variance.
3. **What is out-of-bag (OOB) error?** Each tree doesn't see ~37% of rows (bootstrap); predicting those gives a free validation estimate.
4. **In gradient boosting, what is each tree fitted on?** The negative gradient of the loss w.r.t. current predictions — the residuals for MSE.
5. **Role of the learning rate in boosting?** Shrinks each tree's contribution; smaller η needs more trees but generalizes better.
6. **Can Random Forest overfit by adding more trees?** Adding trees doesn't increase overfitting (error converges); overfitting comes from tree depth/leaf size. Boosting **can** overfit with too many trees → early stopping.
7. **XGBoost vs. LightGBM vs. CatBoost?** XGBoost: regularized objective, level-wise growth, pruning by gamma. LightGBM: histograms + leaf-wise growth, very fast on large data. CatBoost: native categorical handling with ordered target statistics, little tuning.
8. **Why are ensembles less interpretable and how do you explain them?** No single tree; use feature importance, permutation importance, SHAP, PDP/ICE.
9. **How would you choose between two models with similar performance?** Prefer the simpler, faster, cheaper-to-maintain, more stable one.
10. **Which boosting implementations handle missing values natively?** HistGB, XGBoost, LightGBM (and sklearn trees/RF recently); classic sklearn GB and AdaBoost don't.

---

## 13. References

- Albon, C. *Machine Learning with Python Cookbook*. O'Reilly, 2018.
- Géron, A. *Hands-On Machine Learning with Scikit-Learn, Keras and TensorFlow*. O'Reilly, 2019.
- James, G. et al. *An Introduction to Statistical Learning — with Applications in Python*. 2nd ed. Springer, 2023.
- Morettin, P. A.; Singer, J. M. *Estatística e Ciência de Dados*. LTC, 2022.
- Breiman, L. "Random Forests." *Machine Learning*, 45, 2001.
- Friedman, J. H. "Greedy Function Approximation: A Gradient Boosting Machine." *Annals of Statistics*, 2001.
- Chen, T.; Guestrin, C. "XGBoost: A Scalable Tree Boosting System." *KDD*, 2016.
- Ke, G. et al. "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." *NeurIPS*, 2017.

---

⬅️ Previous: [1.5 — Regression Trees](../1.5%20-%20Regression%20Trees/README.md) · 🏠 [Index](../../README.md) · ➡️ Next: [1.7 — Applied Python for Modeling](../1.7%20-%20Applied%20Python%20for%20Modeling/README.md)
