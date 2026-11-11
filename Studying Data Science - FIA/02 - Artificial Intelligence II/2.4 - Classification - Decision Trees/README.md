# 2.4 — Classification Models: Decision Trees

> **Module:** Artificial Intelligence II
> **Case studies:** (1) **home-insurance claims** — probability a policy has a claim in the next 12 months (6.67% events, imbalanced, binary); (2) **health-insurance operator** — most likely contract outcome in 3 months among 5 categories (multinomial).
> **Also covered:** class imbalance in trees, stratified validation, **probability calibration** (calibration curve, Brier score, sigmoid/isotonic).

---

## Table of Contents

1. [From regression trees to decision trees](#1-from-regression-trees-to-decision-trees)
2. [Impurity: Gini and entropy](#2-impurity-gini-and-entropy)
3. [Predicted values and classification](#3-predicted-values-and-classification)
4. [Hyperparameters and modeling strategy](#4-hyperparameters-and-modeling-strategy)
5. [Class imbalance in trees](#5-class-imbalance-in-trees)
6. [Stratified validation](#6-stratified-validation)
7. [Case: home-insurance claims](#7-case-home-insurance-claims)
8. [Probability calibration](#8-probability-calibration)
9. [Multinomial decision trees](#9-multinomial-decision-trees)
10. [Case: health-insurance operator](#10-case-health-insurance-operator)
11. [Pseudocode](#11-pseudocode)
12. [Interview questions](#12-interview-questions)
13. [References](#13-references)

---

## 1. From regression trees to decision trees

Applications:

| Question | Motivation |
|---|---|
| A bank customer's satisfaction level (very satisfied … very dissatisfied) from transactions and relationship | Find satisfaction drivers, reduce churn |
| Probability a patient has each of 3 diseases under investigation | Faster diagnosis, earlier adequate treatment |
| Next product category (of 8) a retail customer will buy, from 24 months of purchases | Personalized offers, repurchase, stock planning |

The **decision (classification) tree** is the regression-tree algorithm (1.5) for a **qualitative target**. Same procedure: recursively split into nodes by binary conditions on features until a stopping criterion is met. **Three things change:**

1. The **impurity measure** (no longer MSE/MAE).
2. The **predicted value** in each leaf → a **probability per category**.
3. **Evaluation** → classification metrics, with attention to **probability calibration**.

Two variants: **binary** (2 categories) and **multinomial** (3+).

---

## 2. Impurity: Gini and entropy

**Impurity** = how mixed the target categories are within a node. Pure node = one category only.

### Gini index

**Probability of misclassifying a randomly drawn observation** of the node if it were labeled according to the node's own proportions:

$$G = 1 - p_0^2 - p_1^2$$

- Max **0.5** when both categories are 50/50; **0** when the node is pure.

### Entropy

**Amount of information needed to identify the category** of an observation in the node (more mixed → more information):

$$H = -p_0\log_2 p_0 - p_1\log_2 p_1$$

- Max **1** at 50/50; **0** when pure. Same min/max points as Gini.

### Gini or entropy?

Same shape, rank splits almost always the same way (2 × Gini ≈ entropy curve). **Gini is faster** (no logarithm). In practice, treat the criterion as **one more hyperparameter** to test.

The split chosen is the one with the largest reduction from the parent impurity to the **weighted average** of the children's impurities (**information gain** when using entropy).

### Worked example

```
                 Node 0: 100 obs (65 class 0 / 35 class 1)
                 Gini = 0.455   Entropy = 0.934
                /                                \
   Node 1: 70 obs (56 / 14)              Node 2: 30 obs (9 / 21)
   Gini = 0.320  Entropy = 0.722         Gini = 0.420  Entropy = 0.881

Weighted average: Gini = 0.7·0.320 + 0.3·0.420 = 0.350   Entropy = 0.770
Reduction: Gini −0.105 (−23%)   Entropy −0.164 (−18%)
```

Check: Gini(node 0) = 1 − 0.65² − 0.35² = 1 − 0.4225 − 0.1225 = 0.455 ✓.

---

## 3. Predicted values and classification

In a regression tree the leaf predicted the mean/median. In a decision tree, each leaf predicts the **proportion of class 1**:

$$\hat p_j = \frac{n_{j,1}}{n_j}$$

(nⱼ = observations in leaf j; nⱼ,₁ = those with Y = 1.)

This estimates **P(Y = 1 | X)** — the same quantity logistic regression estimates. The difference is *how*: logistic uses a **linear function of the features**; the tree uses the **observed frequency within the profile**.

**Classification:** as with logistic regression, apply a cutoff. Balanced data → 50%. Rare event → lower cutoff; the **overall event rate** is the usual starting point.

---

## 4. Hyperparameters and modeling strategy

Same as regression trees plus two new ones:

| Hyperparameter (`DecisionTreeClassifier`) | Meaning | Values |
|---|---|---|
| `criterion` 🆕 | Impurity measure to minimize | `gini` or `entropy` |
| `class_weight` 🆕 | Weight of each category in the impurity calculation | `None` or `balanced` |
| `min_impurity_decrease` | Minimum impurity reduction to split | 0 – 0.05 |
| `min_samples_split` | Min. obs. in the parent | depends on dataset size |
| `min_samples_leaf` | Min. obs. in each child | depends on dataset size |
| `max_depth` | Max levels | 2 – 10 |

**Strategy:** holdout for external test + CV to choose hyperparameters + random search. **What changes is the selection metric** (instead of RMSE):
- **AUC** — discrimination independent of the cutoff.
- **Balanced accuracy** — simple mean of sensitivity and specificity (equal weight to both classes).

Final evaluation: accuracy, sensitivity, specificity, PPV, NPV, AUC.

---

## 5. Class imbalance in trees

A dataset is **balanced** when categories appear in similar proportions (p₀ ≈ p₁ ≈ 50%); otherwise **imbalanced** — and the root node already starts with **lower impurity** because the majority class dominates.

Often the **minority** class is what matters (fraud, default, failure, claim). Imbalance affects **(a) the choice of splits** and **(b) the reading of predicted probabilities**.

### Effect on the choice of splits

Root: 100 obs, **90 class 0 / 10 class 1** → Gini = 1 − 0.9² − 0.1² = **0.180**.

| | Proposal 1 — isolates the minority | Proposal 2 — isolates the majority |
|---|---|---|
| Node 1 | 38 obs (30 / 8) → Gini 0.332 | 10 obs (5 / 5) → Gini 0.500 |
| Node 2 | 62 obs (60 / 2) → Gini 0.062 | 90 obs (85 / 5) → Gini 0.105 |
| Hit rate class 1 (node 1 → 1) | **80%** | 50% |
| Hit rate class 0 (node 2 → 0) | 67% | **94%** |
| Weighted Gini | 0.38·0.332 + 0.62·0.062 ≈ **0.165** | 0.10·0.5 + 0.90·0.105 ≈ **0.144** |
| Gini reduction | ≈ 0.015 | ≈ **0.036** ✅ chosen |

The algorithm picks **proposal 2**, which identifies the minority class *worse*. Why? **Impurity indices weigh each class by its frequency** — getting the majority class right brings more impurity reduction than getting the minority right.

**Two fixes when the minority class matters:**
1. **Weight the classes** in the impurity calculation (`class_weight="balanced"`).
2. **Select the model by a prevalence-insensitive metric** — AUC or balanced accuracy.

### Weighted impurity

Each class gets a weight **inversely proportional to its frequency**, so all classes contribute equally:

$$G_w = 1 - \frac{w_0 p_0^2 + w_1 p_1^2}{(w_0 p_0 + w_1 p_1)^2},\qquad w_0 = \frac{M}{2n_0},\quad w_1 = \frac{M}{2n_1}$$

(M = total observations; n₀, n₁ = class counts.) `DecisionTreeClassifier(class_weight="balanced")`

---

## 6. Stratified validation

With imbalance, a random split can leave the minority class **under-represented** in small sets (external test, internal-test folds). **Stratifying** keeps the same class distribution as the full dataset in every subset.

Example: 100 obs (80 class 0, 20 class 1), 75% train / 25% external test, 5-fold stratified CV:
- **Train:** 75 obs → 60 / 15.
- **Each internal-test fold:** 15 obs → 12 / 3.
- **External test:** 25 obs → 20 / 5.

`train_test_split(..., stratify=y)` and `StratifiedKFold(...)`.

---

## 7. Case: home-insurance claims

**Question:** probability a home-insurance policy registers a claim in the **next 12 months**.
**Motivation:** the probability feeds **premium pricing** and **risk acceptance**. Too high → expensive policy (may lose the customer); too low → cheap policy (portfolio exposed to indemnity costs not covered by premiums).
**Data:** 40,000 policies in force at the start; **6.67%** had ≥ 1 claim at the end of 12 months. Minority class is the focus. No missing values or duplicates.

| Variable | Description |
|---|---|
| `IDADE_IMOVEL_ANOS` | Property age (years) |
| `VALOR_SEGURADO` | Insured value (R$) |
| `TEMPO_CLIENTE_ANOS` | Customer tenure with the insurer (years) |
| `QTD_SINISTROS_5A` | # claims in the last 5 years |
| `TIPO_IMOVEL` | Apartment, house, townhouse |
| `REGIAO` | Capital, metro area, countryside |
| `SISTEMA_ALARME` | Monitored alarm (yes/no) |
| `SINISTRO_12M` | **Target** (1/0) |

**Strategy:** stratified 80/20 holdout · stratified 5-fold CV · random search (100 iterations): `criterion` ∈ {gini, entropy}, `class_weight="balanced"`, `min_impurity_decrease` 0–0.05, `min_samples_split` 100–200, `min_samples_leaf` 50–100, `max_depth` 2–10 · choose by **highest mean internal-test AUC**, as long as stable · seed 123.

**Chosen (model 57):** `criterion=entropy, class_weight=balanced, min_impurity_decrease=0.002455, min_samples_split=181, min_samples_leaf=55, max_depth=5`.

### The effect of weighting on reading the tree

Node proportions shown in the tree plot are computed **with the `balanced` weights**, not observed frequencies — the root shows **50/50** as if the data were balanced. For the highlighted profiles:
- Highest-propensity leaf shows **80.6%**, but its **observed** claim rate is **22.9%**.
- Lowest-propensity leaf shows **15.7%** vs. observed **1.3%**.

> The **ranking** of profiles is correct (that's what supports the AUC). What's off is the **level** of the probabilities → they're **uncalibrated**.

---

## 8. Probability calibration

Under the frequentist interpretation, "the model predicts 30% for this profile" should mean "about 30% of observations with this profile have the event". In the claims model, the **mean predicted probability is 42.6% vs. an observed rate of 6.7%** → not calibrated.

### Calibration curve (reliability diagram)

1. Split the probability scale into **10 equal-width bins** (0–0.1, 0.1–0.2, …).
2. In each bin compute the **mean predicted probability** and the **observed event rate**.
3. Plot mean predicted (x) vs. observed rate (y).

- Well calibrated → points around the **45° diagonal**.
- **Below** the diagonal → model **over-predicts** (predicts higher than observed); above → under-predicts.
- Empty bins are omitted.

Typical behaviors: **logistic regression tends to be well calibrated**; trees and ensembles can deviate (e.g., Random Forest pushes probabilities toward the middle; boosting/naive Bayes toward the extremes).

### Brier score

$$BS = \frac{1}{n}\sum_{i=1}^{n}(\hat p_i - y_i)^2$$

- 0 to 1, **lower is better**. Drops when the model gives high probabilities where y = 1 and low ones where y = 0 → combines **discrimination** and **calibration** in one number.
- In practice: compare the **same model before and after calibration**.

### Calibration methods

Calibrating = fitting a **non-decreasing function** mapping predicted probability → observed rate.

| Method | How | Pros / cons |
|---|---|---|
| **Sigmoid (Platt scaling)** | Fits a logistic function on the predicted probabilities (2 parameters) | Smooth → **stable on small data**, but can't follow deviations outside that shape |
| **Isotonic regression** | Fits a **piecewise-constant increasing** function, no fixed shape | More flexible, usually calibrates better, but needs volume — use with **≥ 1,000 observations** |

`CalibratedClassifierCV(estimator, method="sigmoid" | "isotonic", cv=...)` — calibrate on data **not** used to train the base model (CV or a separate set).

### What calibration changes

Calibration is an **increasing transformation**: it repositions the level and keeps the ranking practically unchanged.
- **Changes little:** **AUC** (ranking barely changes).
- **Changes:** **Brier score**, the direct reading of probabilities as expected frequencies, and **any calculation multiplying probability by money** (expected loss, pricing).
- **The cutoff moves:** with weighted probabilities it's around 50%; with calibrated probabilities, around the **overall event rate**.

### Case results

- Isotonic regression (> 1,000 obs): curve near the diagonal; **Brier 0.212 → 0.059**.
- Mean predicted probability **42.6% → 6.8%** (observed 6.7%).
- **AUC stable** across train/internal/external test and barely affected by calibration.
- With calibrated probabilities, a **50% cutoff would classify every policy as "no claim"** → use the event rate (**6.67%**) as cutoff.

| Setting | Cutoff |
|---|---|
| Original (weighted) probabilities | ~50% |
| Calibrated probabilities | ~6.67% (event rate) |

**Importance (impurity & permutation agree):** claim history dominates, then tenure, then property characteristics.
**SHAP — higher claim probability for:** more claims in the last 5 years · shorter tenure · older property and higher insured value · **no alarm**.

**Conclusion:** the model separates policies well — the highest-propensity profile's observed claim rate is **17× higher** than the lowest. With calibrated probabilities, they can enter pricing: a policy with 20% probability and known average indemnity has **expected loss = 0.20 × average indemnity**.

---

## 9. Multinomial decision trees

For a target with **K > 2** categories, estimate probabilities for each. Same algorithm, changing:
- impurity sums over the K categories;
- the predicted category per leaf — **no cutoff**;
- performance measures (no single positive/negative class).

### Generalized Gini and entropy

$$G = 1 - \sum_{k=1}^{K} p_k^2 \qquad H = -\sum_{k=1}^{K} p_k\log_2 p_k$$

Maxima depend on K (all equal proportions): **Gini max = 1 − 1/K**, **entropy max = log₂K**. For K = 4: Gini ∈ [0, 0.75], entropy ∈ [0, 2].

Example (K = 4):

```
          Node 0: 100 obs (8 · 12 · 25 · 55)   Gini 0.614   Entropy 1.633
            /                                     \
  Node 1: 60 obs (2 · 3 · 10 · 45)       Node 2: 40 obs (6 · 9 · 15 · 10)
  Gini 0.406   Entropy 1.122             Gini 0.724   Entropy 1.925
Weighted: Gini 0.533 · Entropy 1.443  → reduction 13% (Gini), 12% (entropy)
```

### Predicted category

Each leaf gives a probability per category: $\hat p_{j,k} = n_{j,k}/n_j$.

A first idea: predict the category with the highest probability. But under imbalance this **favors frequent categories**: if one category is 80% of the data, few leaves will have another majority, and the model **never predicts the others** — same problem as the 50% cutoff in the binary case.

**Same fix:** **weight the categories** (`class_weight="balanced"`), changing both the impurity and the leaf proportions. Consequence: probabilities become **uncalibrated** → calibrate (per category, one-vs-rest). `DecisionTreeClassifier` handles binary and multinomial targets the same way.

### Selection metrics (multiclass)

| Metric | Definition |
|---|---|
| **AUC-OVR (one-vs-rest)** | Mean of the K ROC AUCs of each category vs. all others: $\frac{1}{K}\sum_k AUC_k$ |
| **AUC-OVO (one-vs-one)** | Mean over all K(K−1)/2 pairs: $\frac{2}{K(K-1)}\sum_{j<k} AUC_{jk}$ |
| **Balanced accuracy** | Mean of the K per-class hit rates: $\frac{1}{K}\sum_k \text{hit}_k$ |

Plain accuracy is still available, but under imbalance it mostly reflects the majority class.

### K × K confusion matrix

| | Pred = 1 | Pred = 2 | … | Pred = K |
|---|---|---|---|---|
| **Obs = 1** | ✅ hit | error | | error |
| **Obs = 2** | error | ✅ hit | | error |
| **Obs = K** | error | error | | ✅ hit |

- **Hit rate (recall) per category:** hitsₖ / observedₖ.
- **Precision per category:** hitsₖ / predictedₖ.

---

## 10. Case: health-insurance operator

**Question:** most likely contract outcome for each customer in the next 3 months: **cancel for non-payment (CancInvol)**, **cancel by own decision (CancVol)**, **downgrade (Down)**, **keep (Mantem)** or **upgrade (Up)**.
**Motivation:** each outcome calls for a **different retention action** — treating all at-risk customers the same wastes budget.
**Data:** 30,000 customers with ≥ 12 months of plan; most customers **keep** (67%). No duplicates/missings.

| Variable | Description |
|---|---|
| `IDADE` | Age |
| `TEMPO_PLANO` | Years with the plan |
| `QTDE_DEPENDENTES` | # dependents |
| `RELACAO_CUSTO_RENDA` | Annual plan cost / declared annual income |
| `QTDE_UTILIZACOES_12M` | # service uses in 12 months |
| `QTDE_SERVICOS_12M` | # distinct services used (consultation, exam, ER, hospitalization) |
| `QTDE_RECLAMACOES_12M` | # complaints in 12 months |
| `NIVEL_SATISFACAO` | Avg. satisfaction 0–10 |
| `QTDE_ATRASOS_12M` | # late payments in 12 months |
| `FLAG_MUDANCA_RESID_12M` | Moved residence in 12 months |
| `STATUS` | **Target** — CancInvol, CancVol, Down, Mantem, Up |

**Strategy:** stratified 80/20 + stratified 5-fold CV · random search (100): `criterion` ∈ {gini, entropy}, `class_weight="balanced"`, `min_impurity_decrease` 0–0.05, `min_samples_split` 300–800, `min_samples_leaf` 150–400, `max_depth` 2–10 · choose by **highest mean internal-test AUC-OVR**, stable vs. train · seed 123.

**Chosen (model 90):** `entropy, balanced, min_impurity_decrease=0.000387, min_samples_split=723, min_samples_leaf=313, max_depth=10` → **62 leaves**. Root shows 20% per category (balanced weights). Calibration done **per category (one-vs-rest)**.

**Performance:**
- **AUC-OVR 0.790** (external test), stable vs. train (0.802) and internal test (0.789); **AUC-OVO 0.817**.
- Cancellations best separated (AUC **0.847** and **0.828**); **"Mantem" hardest (0.684)** — it groups customers *without* any distinctive signal; the model reaches it **by exclusion**.

| Category | Hit rate (recall) | Precision |
|---|---|---|
| Cancel – non-payment | 62% | 25% |
| Cancel – voluntary | 57% | 19% |
| Downgrade | 40% | 33% |
| Keep | 38% | **81%** |
| Upgrade | 68% | 29% |
| **Overall** | Accuracy **44%** (vs. 67% predicting always "keep") · **Balanced accuracy 53%** (vs. 20% at random) | |

> Rare categories have low precision because the model flags **many more customers than the actual number of cases** — that's the price of identifying most of them.

**SHAP by category — higher probability of…**

| Outcome | Drivers |
|---|---|
| **Cancel for non-payment** | More late payments · higher cost/income ratio · shorter tenure |
| **Cancel voluntarily** | More complaints · less variety of services used · lower satisfaction |
| **Downgrade** | Higher cost/income ratio · more dependents |
| **Keep** | Lower cost/income · longer tenure · fewer late payments · fewer complaints |
| **Upgrade** | Lower cost/income · fewer complaints and late payments · higher satisfaction |

**Conclusion:** AUC-OVR 0.790, good identification of both cancellation types (largest losses). Actions: non-payment → **debt renegotiation / billing review**; voluntary → **address open complaints**; downgrade → **offer an intermediate plan** before the customer asks.

---

## 11. Pseudocode

### 11.1 Impurity functions

```text
FUNCTION Gini(counts[1..K], weights = 1)
    w ← weights · counts;  p ← w / Σ w
    RETURN 1 - Σ p_k²

FUNCTION Entropy(counts[1..K], weights = 1)
    w ← weights · counts;  p ← w / Σ w
    RETURN -Σ_{p_k > 0} p_k · log2(p_k)

FUNCTION BalancedWeights(y)
    M ← |y|;  K ← #classes
    FOR each class k:  w[k] ← M / (K · count(y = k))          # binary: M / (2 n_k)
    RETURN w
```

### 11.2 Growing a classification tree

```text
FUNCTION BuildClassTree(data, depth, params, w)
    counts ← class counts in data
    node.proba ← (w · counts) / Σ (w · counts)                 # weighted proportions per class
    IF depth = max_depth OR |data| < min_samples_split OR node is pure:
        RETURN Leaf(node)
    parent ← Impurity(counts, w)                               # Gini or Entropy (criterion)
    best ← NONE
    FOR each feature f, each threshold t:
        L, R ← split(data, f ≤ t)
        IF |L| < min_samples_leaf OR |R| < min_samples_leaf: CONTINUE
        child ← (W(L)/W(data))·Impurity(L) + (W(R)/W(data))·Impurity(R)     # W = weighted size
        gain ← parent - child
        IF gain > best.gain: best ← (f, t, gain, L, R)
    IF best = NONE OR weighted_gain(best) < min_impurity_decrease: RETURN Leaf(node)
    node.left  ← BuildClassTree(best.L, depth+1, params, w)
    node.right ← BuildClassTree(best.R, depth+1, params, w)
    RETURN node

FUNCTION PredictProba(tree, x):   RETURN leaf_of(x).proba
FUNCTION PredictBinary(tree, x, cutoff):  RETURN 1 IF PredictProba(tree, x)[1] ≥ cutoff ELSE 0
FUNCTION PredictMulti(tree, x):   RETURN argmax_k PredictProba(tree, x)[k]
```

### 11.3 Calibration curve and Brier score

```text
FUNCTION CalibrationCurve(y, p̂, n_bins = 10)
    FOR b IN 1..n_bins:
        idx ← p̂ ∈ [(b-1)/n_bins, b/n_bins)
        IF idx is empty: CONTINUE
        x[b] ← mean(p̂[idx]);  y_obs[b] ← mean(y[idx])
    PLOT (x, y_obs) against the 45° line
    # points below the diagonal → model over-predicts

FUNCTION Brier(y, p̂):  RETURN mean( (p̂ - y)² )
```

### 11.4 Isotonic calibration (Pool Adjacent Violators)

```text
FUNCTION FitIsotonic(p̂_cal, y_cal)          # on data NOT used to train the base model
    sort pairs by p̂_cal
    blocks ← each observation is a block with value y
    WHILE some adjacent blocks violate monotonicity (left.value > right.value):
        merge them; value ← weighted mean of y in merged block
    RETURN step function g: p̂ → calibrated probability

FUNCTION FitSigmoid(p̂_cal, y_cal)           # Platt scaling
    fit logistic regression: y ~ a · logit(p̂) + b       (or a · p̂ + b)
    RETURN g(p̂) = 1 / (1 + exp(-(a · s(p̂) + b)))

# after calibration: AUC ≈ unchanged, Brier ↓, mean(g(p̂)) ≈ event rate,
# cutoff ≈ event rate; expected_loss = g(p̂) × average indemnity
```

### 11.5 Multiclass evaluation

```text
FUNCTION AUC_OVR(y, P̂[n×K]):  RETURN mean_k AUC( y == k, P̂[:, k] )
FUNCTION AUC_OVO(y, P̂):        RETURN mean over pairs (j<k) of AUC on obs with y ∈ {j,k}
FUNCTION BalancedAccuracy(y, ŷ):  RETURN mean_k ( Σ(y=k ∧ ŷ=k) / Σ(y=k) )
FUNCTION PerClassPrecision(y, ŷ, k):  RETURN Σ(y=k ∧ ŷ=k) / Σ(ŷ=k)
```

Python reference:

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import roc_auc_score, brier_score_loss, balanced_accuracy_score

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.2, stratify=y, random_state=123)
search = RandomizedSearchCV(DecisionTreeClassifier(class_weight="balanced", random_state=123),
                            space, n_iter=100, scoring="roc_auc",          # "roc_auc_ovr" for multiclass
                            cv=StratifiedKFold(5, shuffle=True, random_state=123)).fit(X_tr, y_tr)
cal = CalibratedClassifierCV(search.best_estimator_, method="isotonic", cv=5).fit(X_tr, y_tr)
p = cal.predict_proba(X_te)[:, 1]
roc_auc_score(y_te, p); brier_score_loss(y_te, p); calibration_curve(y_te, p, n_bins=10)
y_hat = (p >= y_tr.mean()).astype(int)                 # cutoff = event rate
```

---

## 12. Interview questions

1. **Gini vs. entropy?** Both measure node impurity, peak at equal proportions and are zero for pure nodes; nearly identical splits; Gini is cheaper (no log).
2. **Compute the Gini of a node with 70% / 30%.** 1 − 0.49 − 0.09 = 0.42.
3. **Why do trees struggle with imbalanced classes?** Impurity reduction is driven by the majority class, so splits favor isolating it; fix with class weights and AUC/balanced-accuracy-based selection.
4. **What does `class_weight="balanced"` do and what's the side effect?** Weights classes inversely to frequency; probabilities become uncalibrated (inflated for the minority).
5. **What is calibration and why does it matter?** Agreement between predicted probabilities and observed frequencies; essential when probabilities feed pricing, expected-loss or budget calculations.
6. **Does calibration change AUC?** Barely — it's a monotonic transformation preserving rank.
7. **Platt vs. isotonic?** Platt: 2-parameter sigmoid, stable on small data. Isotonic: non-parametric monotonic step function, more flexible, needs ≥ ~1,000 observations; risk of overfitting on small sets.
8. **What is the Brier score?** Mean squared error of probabilities; lower is better; mixes discrimination and calibration.
9. **Why use stratified splits?** Preserve class proportions in train/test/folds, especially for rare classes.
10. **How do you evaluate a multiclass model?** K×K confusion matrix, per-class recall/precision, balanced accuracy, macro AUC-OVR/OVO, log-loss.
11. **Why is plain accuracy misleading in the health-insurance case?** Predicting always "keep" gives 67% accuracy but is useless; balanced accuracy (53% vs. 20% random) is fairer.
12. **Decision tree vs. logistic regression for P(Y=1|X)?** Logistic: linear in log-odds, smooth, well calibrated, extrapolates. Tree: piecewise-constant frequencies per profile, captures interactions/non-linearity, needs calibration under weighting.

---

## 13. References

- Breiman, L. et al. *Classification and Regression Trees*. Chapman & Hall, 1984.
- Hastie, T.; Tibshirani, R.; Friedman, J. *The Elements of Statistical Learning*. 2nd ed. Springer, 2009.
- James, G. et al. *An Introduction to Statistical Learning — with Applications in Python*. 2nd ed. Springer, 2023.
- Morettin, P. A.; Singer, J. M. *Estatística e Ciência de Dados*. 2nd ed. LTC, 2022.
- Niculescu-Mizil, A.; Caruana, R. "Predicting Good Probabilities with Supervised Learning." *ICML*, 2005.
- Lundberg, S. M.; Lee, S.-I. "A Unified Approach to Interpreting Model Predictions." *NeurIPS*, 2017.

---

⬅️ Previous: [2.3 — Logistic Regression](../2.3%20-%20Classification%20-%20Logistic%20Regression/README.md) · 🏠 [Index](../../README.md) · ➡️ Next: [2.5 — Random Forest and Boosting (Classification)](../2.5%20-%20Classification%20-%20Random%20Forest%20and%20Boosting/README.md)
