# 2.5 — Classification Models: Random Forest and Boosting

> **Module:** Artificial Intelligence II
> **Case studies (same as 2.4, now with ensembles):** home-insurance claims (binary, 6.67% events) and health-insurance contract outcome (5 categories).
> **Prerequisites:** [1.6 — Random Forest and Boosting for regression](../../01%20-%20Artificial%20Intelligence%20I/1.6%20-%20Random%20Forest%20and%20Boosting%20for%20Regression/README.md) and [2.4 — Decision Trees](../2.4%20-%20Classification%20-%20Decision%20Trees/README.md).

The class material for this topic is short: the algorithms are the ones from 1.6, adapted to classification. These notes add the details that explain **how** that adaptation works, which is what interviewers usually ask about. Sections marked *(extra)* go beyond the slides.

---

## Table of Contents

1. [What changes from regression to classification](#1-what-changes-from-regression-to-classification)
2. [Random Forest for classification](#2-random-forest-for-classification)
3. [AdaBoost for classification](#3-adaboost-for-classification)
4. [Gradient boosting for classification](#4-gradient-boosting-for-classification)
5. [Class weighting: class_weight vs. sample_weight](#5-class-weighting-class_weight-vs-sample_weight)
6. [Python functions](#6-python-functions)
7. [Modeling strategy (same as 2.4)](#7-modeling-strategy-same-as-24)
8. [Case studies](#8-case-studies)
9. [Pseudocode](#9-pseudocode)
10. [Interview questions](#10-interview-questions)
11. [References](#11-references)

---

## 1. What changes from regression to classification

The Random Forest and boosting algorithms from the projection models also work for **binary and multinomial classification**.

- In **Random Forest** and **AdaBoost**, each tree becomes a **decision (classification) tree**, using **Gini or entropy** as the impurity (as in 2.4).
- In **gradient boosting**, each tree is **still a regression tree**: it's fitted to the **pseudo-residuals of the log-loss** and added to the previous trees **on the logit scale**.

The general steps of each algorithm stay the same. **Three things change:**

1. **How trees are combined** in the final prediction → it now produces a **probability per category**.
2. The **loss function** in gradient boosting → no longer squared error (it's **log-loss**).
3. **Class weighting** → not every algorithm accepts `class_weight`. The alternative is a **vector of weights per observation** (`sample_weight`), computed with the same criterion and passed at fit time.

The modeling strategy, performance measures and probability calibration are the same as for a single decision tree.

---

## 2. Random Forest for classification

Same bagging steps as in 1.6, with classification trees:

1. Bootstrap sample of rows (`max_samples`, 60–100%).
2. Grow a classification tree; at each split consider `max_features` random features; impurity = Gini or entropy.
3. Each tree gives, for observation x, the class proportions of the leaf where x lands.
4. Repeat for `n_estimators` trees.
5. **Final probability = average of the trees' probabilities** (scikit-learn uses soft voting). Classic Random Forest used **majority vote** of predicted classes.

$$\hat p_k(x) = \frac{1}{T}\sum_{t=1}^{T}\hat p_{t,k}(x)$$

*(extra)* Calibration tendency: averaging pulls probabilities **toward the middle** (rarely near 0 or 1) → calibrate with isotonic/sigmoid when probabilities matter (pricing, expected loss).

Key hyperparameters: `n_estimators`, `criterion` (gini/entropy), `max_depth`, `min_samples_split`, `min_samples_leaf`, `min_impurity_decrease`, `max_samples`, `max_features`, `class_weight` (`None`, `balanced`, `balanced_subsample`).

---

## 3. AdaBoost for classification

Same sequential scheme as in 1.6, with classification stumps:

1. Start with equal observation weights.
2. Fit a weak tree (stump) with those weights.
3. Compute its **weighted error rate** ε (share of weight on misclassified observations).
4. Give the tree a weight $\alpha = \eta\cdot\ln\frac{1-\varepsilon}{\varepsilon}$ (plus $\ln(K-1)$ in the multiclass SAMME variant): lower error → larger weight.
5. **Increase the weights of misclassified observations** (multiply by $e^{\alpha}$), renormalize → the next stump focuses on the hard cases.
6. Final prediction: **weighted vote** of all stumps (weights α); probabilities derived from the weighted vote.

---

## 4. Gradient boosting for classification

The key idea: **boost on the log-odds (logit) scale** and convert to probability with the sigmoid at the end.

**Binary case (log-loss):**

$$L(y, F) = -\big[y\ln p + (1-y)\ln(1-p)\big],\qquad p = \sigma(F) = \frac{1}{1+e^{-F}}$$

The negative gradient of the log-loss with respect to F is remarkably simple:

$$-\frac{\partial L}{\partial F} = y - p$$

So each new **regression tree** is fitted to the **pseudo-residuals rᵢ = yᵢ − pᵢ** (observed label minus current predicted probability), and the model is updated on the logit scale:

$$F_0 = \ln\frac{\bar y}{1-\bar y},\qquad F_m(x) = F_{m-1}(x) + \eta\,h_m(x),\qquad \hat p(x) = \sigma(F_M(x))$$

- F₀ = log-odds of the overall event rate.
- *(extra)* Leaf values are usually adjusted with a Newton step: $\gamma_{leaf} = \frac{\sum r_i}{\sum p_i(1-p_i)}$.

**Multiclass case (K categories):** fit **K trees per iteration** (one per class), each on its own residual $y_{ik} - p_{ik}$; probabilities come from the **softmax**:

$$p_k(x) = \frac{e^{F_k(x)}}{\sum_{j=1}^{K} e^{F_j(x)}}$$

XGBoost, LightGBM, CatBoost and HistGradientBoosting follow the same principle (objectives like `binary:logistic`, `multi:softprob`), with the extras from 1.6 (regularization, histograms, leaf-wise growth, native categoricals, native missing values).

---

## 5. Class weighting: class_weight vs. sample_weight

As in 2.4, with imbalanced data we weight classes **inversely to their frequency** so they contribute equally:

$$w_k = \frac{M}{K\cdot n_k}\qquad(\text{binary: } w_k = M/(2n_k))$$

| Algorithm | How to weight |
|---|---|
| `RandomForestClassifier`, `DecisionTreeClassifier`, `HistGradientBoostingClassifier`, `LGBMClassifier` | `class_weight="balanced"` |
| `AdaBoostClassifier`, `GradientBoostingClassifier`, `CatBoostClassifier` (also `auto_class_weights`), `XGBClassifier` (binary also `scale_pos_weight = n₀/n₁`) | **`sample_weight`** vector passed to `.fit()` |

```python
from sklearn.utils.class_weight import compute_sample_weight
sw = compute_sample_weight(class_weight="balanced", y=y_train)
model.fit(X_train, y_train, sample_weight=sw)
```

> ⚠️ When using CV with sample weights, compute the weights **inside each training fold** (or pass them through the search's `fit` params) — and remember: weighted models produce **uncalibrated** probabilities → calibrate (2.4 §8).

---

## 6. Python functions

| Algorithm | Function | Library |
|---|---|---|
| Random Forest | `RandomForestClassifier(...).fit(X, y)` | scikit-learn |
| AdaBoost | `AdaBoostClassifier(...).fit(X, y)` | scikit-learn |
| Gradient Boosting | `GradientBoostingClassifier(...).fit(X, y)` | scikit-learn |
| Histogram-based GB | `HistGradientBoostingClassifier(...).fit(X, y)` | scikit-learn |
| XGBoost | `XGBClassifier(...).fit(X, y)` | xgboost |
| LightGBM | `LGBMClassifier(...).fit(X, y)` | lightgbm |
| CatBoost | `CatBoostClassifier(...).fit(X, y)` | catboost |

---

## 7. Modeling strategy (same as 2.4)

- **Stratified** holdout (80/20) and **stratified** k-fold CV (k = 5).
- Random search (or Optuna) over hyperparameters (see the ranges in 1.6).
- Class weighting (`class_weight` or `sample_weight`).
- Select by **AUC** (binary) or **AUC-OVR** (multiclass) on internal-test folds, checking stability vs. train and external test; balanced accuracy as a complementary metric.
- **Calibrate** probabilities (isotonic if ≥ 1,000 obs., otherwise sigmoid); check the calibration curve and Brier score.
- Cutoff: event rate (binary, calibrated); argmax of weighted probabilities or business rules (multiclass).
- Explain with impurity/permutation importance and SHAP (per class in the multiclass case).

---

## 8. Case studies

### Home-insurance claims (binary)
- Probability a policy registers a claim in the next 12 months; feeds premium pricing and risk acceptance.
- 40,000 policies; **6.67%** with ≥ 1 claim; minority class is the focus.
- Features: property age, insured value, customer tenure, claims in the last 5 years, property type, region, monitored alarm. Target: `SINISTRO_12M`.
- Notebook: `Sinistro_Residencial_Random_Forest_Boosting.ipynb` — compare the ensembles against the single tree from 2.4 (AUC, Brier after calibration, stability).

### Health-insurance operator (multinomial)
- Most likely outcome in 3 months: CancInvol, CancVol, Down, Mantem, Up.
- 30,000 customers with ≥ 12 months of plan.
- Features: age, plan tenure, dependents, cost/income ratio, uses and distinct services in 12 months, complaints, satisfaction, late payments, moved residence. Target: `STATUS`.
- Notebook: `Operadora_Saude_Random_Forest_Boosting.ipynb` — compare against the tree's AUC-OVR = 0.790.

> 💡 Things to look at when comparing with the single tree: the ensemble should raise AUC/AUC-OVR, give **more distinct probability values** (the single tree only has one value per leaf), and be **more stable** across folds. The cost is interpretability → use SHAP.

---

## 9. Pseudocode

### 9.1 Random Forest classifier

```text
FUNCTION RFClassifierFit(X, y, T, max_samples, max_features, tree_params, class_weight)
    forest ← []
    FOR t IN 1..T:
        idx ← bootstrap sample of size max_samples · n
        tree ← BuildClassTree(X[idx], y[idx], tree_params,
                              criterion ∈ {gini, entropy},
                              per-split random subset of max_features features,
                              weights = class_weight)
        forest.append(tree)
    RETURN forest

FUNCTION RFPredictProba(forest, x):  RETURN mean( tree.PredictProba(x) FOR tree IN forest )   # per class
```

### 9.2 AdaBoost (binary, SAMME with y ∈ {−1, +1})

```text
FUNCTION AdaBoostClfFit(X, y, M, η)
    w[i] ← 1/n
    FOR m IN 1..M:
        h_m ← fit stump on (X, y) with sample weights w
        ε ← Σ w[i] · 1[h_m(x_i) ≠ y_i] / Σ w[i]
        α_m ← η · ln((1 - ε) / ε)
        w[i] ← w[i] · exp(α_m · 1[h_m(x_i) ≠ y_i])     # misclassified gain weight
        w ← w / Σ w
    RETURN {h_m, α_m}

FUNCTION AdaBoostClfPredict(model, x):  RETURN sign( Σ α_m · h_m(x) )
```

### 9.3 Gradient boosting classifier (binary log-loss)

```text
FUNCTION GBClassifierFit(X, y ∈ {0,1}, M, η, max_depth, sample_weight = 1)
    F0 ← ln( mean(y) / (1 - mean(y)) )                 # log-odds of the event rate
    F[i] ← F0
    trees ← []
    FOR m IN 1..M:
        p[i] ← 1 / (1 + exp(-F[i]))
        r[i] ← y[i] - p[i]                             # pseudo-residuals = -∂logloss/∂F
        h ← fit REGRESSION tree on (X, r) with max_depth, weights sample_weight
        FOR each leaf ℓ of h:                          # Newton step for leaf value
            γ_ℓ ← Σ_{i∈ℓ} w_i·r[i] / Σ_{i∈ℓ} w_i·p[i](1 - p[i])
        F[i] ← F[i] + η · γ_{leaf(x_i)}
        trees.append(h with γ)
    RETURN F0, trees

FUNCTION GBClassifierProba(F0, trees, η, x)
    F ← F0 + η · Σ γ_{leaf_h(x)} FOR h IN trees
    RETURN 1 / (1 + exp(-F))
```

### 9.4 Multiclass gradient boosting (softmax)

```text
FUNCTION GBMulticlassFit(X, y ∈ {1..K}, M, η)
    F[i,k] ← ln(prior_k)  for all i, k
    FOR m IN 1..M:
        P[i,k] ← exp(F[i,k]) / Σ_j exp(F[i,j])          # softmax
        FOR k IN 1..K:
            r[i] ← 1[y_i = k] - P[i,k]
            h_{m,k} ← fit regression tree on (X, r)
            F[:,k] ← F[:,k] + η · h_{m,k}(X)
    RETURN all trees
# prediction: softmax of accumulated F; class = argmax
```

### 9.5 Full workflow

```text
PROCEDURE EnsembleClassificationProject(X, y)
    split stratified 80/20; StratifiedKFold(5)
    sw ← compute_sample_weight("balanced", y_train)      # or class_weight="balanced"
    FOR A IN {RF, AdaBoost, GB, HistGB, XGBoost, LightGBM, CatBoost}:
        best[A] ← RandomizedSearchCV(A, space[A], scoring = AUC | AUC-OVR, cv = StratifiedKFold,
                                     fit_params = {sample_weight: sw})
    final ← best stable model by internal AUC (tie → simpler/lighter)
    cal ← CalibratedClassifierCV(final, method = isotonic if n ≥ 1000 else sigmoid)
    REPORT external-test AUC, Brier, calibration curve, confusion matrix at chosen cutoff
    EXPLAIN with SHAP (per class if multinomial)
```

Python reference:

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

rf  = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=50,
                             class_weight="balanced", random_state=123).fit(X_tr, y_tr)
sw  = compute_sample_weight("balanced", y_tr)
gb  = GradientBoostingClassifier(learning_rate=0.05, n_estimators=300, max_depth=3).fit(X_tr, y_tr, sample_weight=sw)
xgb = XGBClassifier(objective="multi:softprob", eval_metric="mlogloss").fit(X_tr, y_tr, sample_weight=sw)
cal = CalibratedClassifierCV(xgb, method="isotonic", cv=5).fit(X_tr, y_tr)
```

---

## 10. Interview questions

1. **How does Random Forest produce class probabilities?** Average of the per-tree leaf class proportions (soft voting).
2. **In gradient boosting for classification, what does each tree fit?** A regression tree on the pseudo-residuals y − p (negative gradient of log-loss w.r.t. the log-odds).
3. **Why does GB for classification work on the logit scale?** Additive updates on log-odds are unbounded; the sigmoid/softmax maps the sum to valid probabilities.
4. **How does multiclass gradient boosting work?** K trees per iteration (one per class) with softmax probabilities.
5. **How do you handle class imbalance when an algorithm has no `class_weight`?** Pass `sample_weight` (inverse frequency), or `scale_pos_weight` in XGBoost; or resample.
6. **Are ensemble probabilities calibrated?** Often not: RF compresses toward the middle; boosting can be over-confident; class weighting distorts levels. Check calibration curves and Brier; calibrate.
7. **Bagging vs. boosting in terms of bias/variance?** Bagging reduces variance; boosting reduces bias (and controls variance via shrinkage, subsampling, depth).
8. **Why prefer AUC over accuracy for model selection here?** Prevalence-insensitive and threshold-independent ranking quality.
9. **AdaBoost weight update intuition?** Misclassified points get more weight, so the next weak learner focuses on them; accurate learners get bigger votes.
10. **How would you explain an XGBoost classifier to a business stakeholder?** Global SHAP importance for drivers, beeswarm for direction, waterfall for individual decisions; plus calibrated probabilities as expected frequencies.

---

## 11. References

- Breiman, L. et al. *Classification and Regression Trees*. Chapman & Hall, 1984.
- Grus, J. *Data Science from Scratch*. 2nd ed. O'Reilly, 2019.
- Hastie, T.; Tibshirani, R.; Friedman, J. *The Elements of Statistical Learning*. 2nd ed. Springer, 2009.
- James, G. et al. *An Introduction to Statistical Learning — with Applications in Python*. 2nd ed. Springer, 2023.
- Morettin, P. A.; Singer, J. M. *Estatística e Ciência de Dados*. 2nd ed. LTC, 2022.
- Niculescu-Mizil, A.; Caruana, R. "Predicting Good Probabilities with Supervised Learning." *ICML*, 2005.
- Friedman, J. H. "Greedy Function Approximation: A Gradient Boosting Machine." *Annals of Statistics*, 2001.

---

⬅️ Previous: [2.4 — Decision Trees](../2.4%20-%20Classification%20-%20Decision%20Trees/README.md) · 🏠 [Index](../../README.md) · ➡️ Next: [2.6 — Segmentation (Clustering)](../2.6%20-%20Segmentation%20Models%20-%20Clustering/README.md)
