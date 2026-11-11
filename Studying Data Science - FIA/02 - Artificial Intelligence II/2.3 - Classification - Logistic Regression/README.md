# 2.3 — Classification Models: Logistic Regression

> **Module:** Artificial Intelligence II
> **Case studies:** (1) **delivery-app repurchase** — probability a customer orders again in 7 days without any commercial stimulus (balanced classes); (2) **online card fraud** — probability a transaction is fraudulent (2.03% events, imbalanced).

---

## Table of Contents

1. [Classification recap and applications](#1-classification-recap-and-applications)
2. [The logistic function](#2-the-logistic-function)
3. [Simple logistic regression](#3-simple-logistic-regression)
4. [Odds and the log-odds (logit) link](#4-odds-and-the-log-odds-logit-link)
5. [Interpreting the parameters](#5-interpreting-the-parameters)
6. [Multiple logistic regression](#6-multiple-logistic-regression)
7. [Case: delivery-app repurchase](#7-case-delivery-app-repurchase)
8. [From probability to classification](#8-from-probability-to-classification)
9. [Confusion matrix and performance measures](#9-confusion-matrix-and-performance-measures)
10. [ROC curve and AUC](#10-roc-curve-and-auc)
11. [Class imbalance: the fraud case](#11-class-imbalance-the-fraud-case)
12. [Regularized logistic regression](#12-regularized-logistic-regression)
13. [End-to-end workflow](#13-end-to-end-workflow)
14. [Pseudocode](#14-pseudocode)
15. [Interview questions](#15-interview-questions)
16. [References](#16-references)

---

## 1. Classification recap and applications

Supervised learning with a **qualitative** target:
- **Binary:** 2 categories (fraud yes/no).
- **Multinomial:** 3+ categories (cancellation reason: price, service, quality).

| Question | Motivation |
|---|---|
| Probability a customer defaults if granted a personal loan | Sustainable credit, limits matching each profile, lower losses |
| Probability an insured driver has an auto claim (demographics + usage habits) | Identify claim drivers, better pricing, lower risk |
| Probability a customer responds to a discount campaign (purchase history, engagement) | Give discounts to those who actually change behavior, not to those who'd buy anyway |

---

## 2. The logistic function

Since the target is qualitative, we model the **probability** of each category.

Notation:
- **Y** — binary target: 1 if the event occurs, 0 otherwise (e.g., Y = 1 if the customer cancels their phone plan in 12 months).
- **X** — explanatory variable (quantitative or qualitative) (e.g., satisfaction score with the last service).
- **p** — probability of the event for each observation, which **depends on X** (e.g., P(cancel | gave score 6 out of 10)).

The **logistic function** is an S-shaped curve that accepts any real value and always returns a value **strictly between 0 and 1**:

$$f(x) = \frac{e^{\beta_0 + \beta_1 x}}{1 + e^{\beta_0 + \beta_1 x}} = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x)}}$$

e ≈ 2.7183 is **Euler's number** (used by convention for computational simplicity — like π ≈ 3.1416 is the area of a unit circle).

Properties:
- The **sign of β₁** decides whether the curve increases (β₁ > 0) or decreases (β₁ < 0); its **magnitude** sets how fast the probability changes along X.
- If **β₁ = 0**, X no longer matters: the function is a constant that depends only on β₀. If also β₀ = 0, p = 0.5.

```
 p
1.0 ┤                       ●●●●●●
    │                  ●●●
0.5 ┤             ●●              ← steepness set by |β1|
    │        ●●●
0.0 ┤●●●●●●
    └──────────────────────────────► X
```

---

## 3. Simple logistic regression

Estimates the probability of a binary event (Y = 0/1) as a function of **one** X.

**Theoretical model:**

$$p = \frac{e^{\beta_0 + \beta_1 X}}{1 + e^{\beta_0 + \beta_1 X}}$$

- β₀, β₁: fixed, unknown population parameters.
- **β₀ (intercept):** sets the general level around which probabilities oscillate.
- **β₁ (slope):** measures the weight of X on those probabilities.

> 💡 **No error term!** Uncertainty is already embedded in the concept of probability: for the same x, Y can still turn out 0 or 1. (Formally, Y ~ Bernoulli(p).)

**Fitted model:** $\hat p = \dfrac{e^{\hat\beta_0 + \hat\beta_1 X}}{1 + e^{\hat\beta_0 + \hat\beta_1 X}}$ — p̂ is the predicted probability for each observation.

Parameters are estimated by **Maximum Likelihood** (not OLS). `sm.Logit(y, sm.add_constant(x)).fit()`

---

## 4. Odds and the log-odds (logit) link

**Odds ("chance")** = probability the event occurs / probability it doesn't:

$$\text{odds} = \frac{p}{1-p}$$

Example: P(cancel) = 20% → odds = 0.20 / 0.80 = **0.25 = 1/4** → "1 to 4": for every customer who cancels, 4 don't. Odds range from 0 to +∞.

With algebra, the model becomes linear in the **log of the odds**:

$$\ln\!\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 X$$

The log-odds (**logit**) ranges from −∞ to +∞, so a linear equation can be used **without violating probability limits**. This is the link with linear regression (logistic regression is a **GLM** with logit link).

---

## 5. Interpreting the parameters

On the **log-odds scale**:
- **β₀** = log-odds of the event for an observation with **X = 0** (often little practical meaning if X = 0 doesn't exist in the sample).
- **β₁** = how much the **log-odds increase per 1-unit increase in X**. Positive → direct relationship; negative → inverse.

> 💡 **Odds ratio (extra, very common in interviews):** exponentiating gives a multiplicative effect on the odds: **OR = e^β₁**. "Each extra unit of X multiplies the odds by e^β₁." OR > 1 increases the odds; OR < 1 decreases them.

---

## 6. Multiple logistic regression

$$p = \frac{e^{\beta_0 + \beta_1 X_1 + \dots + \beta_k X_k}}{1 + e^{\beta_0 + \beta_1 X_1 + \dots + \beta_k X_k}}
\qquad\Longleftrightarrow\qquad
\ln\!\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 X_1 + \dots + \beta_k X_k$$

- k quantitative or qualitative variables, each with its own coefficient.
- As in linear regression, each coefficient is the effect on the (log-)odds **holding the other variables fixed**.
- Qualitative variables must be turned into **dummies** (one-hot, with a reference category).

**Reading the output** (`print(model.summary())`):

| Column | Meaning |
|---|---|
| `coef` | Estimated coefficient on the log-odds scale; **sign = direction** of the effect |
| `std err` | How much the coefficient would vary across samples of the same size (smaller = more stable) |
| `z` | coef / std err (standardized estimate; larger \|z\| → farther from zero). Logistic uses **z** (Wald test), linear uses t |
| `P>\|z\|` | p-value of H₀: βⱼ = 0 vs. Hₐ: βⱼ ≠ 0. Small (≤ 5%) → reject nullity |
| `[0.025 0.975]` | 95% CI; if it **contains zero**, we can't say the variable is associated with the odds |

**Variable selection:** **stepwise backward** — remove one variable at a time, always the highest p-value, until only significant ones remain (5% throughout the case).

**Collinearity and VIF:** same as in linear regression.

| VIF | Collinearity |
|---|---|
| = 1 | Absent |
| 1 – 2 | Weak |
| 2 – 5 | Moderate |
| 5 – 10 | High |
| > 10 | Very high |

- Collinearity doesn't affect **performance**, only the **interpretation** of effects.
- For qualitative variables with many categories, dummies' VIF is naturally higher (they're correlated by construction).

---

## 7. Case: delivery-app repurchase

**Question:** probability a customer places a new order in the **next 7 days without any commercial stimulus** → choose who should receive a discount coupon.
**Data:** 50,000 customers with ≥ 1 order in the last 30 days; ~**half** reordered within 7 days (**balanced** classes). No campaign was active during the window, so the target captures **spontaneous** repurchase.

| Variable | Description |
|---|---|
| `DIAS_DESDE_ULTIMO_PEDIDO` | Days since last order (recency) |
| `QTD_PEDIDOS_30D` | # orders in last 30 days (frequency) |
| `TICKET_MEDIO_30D` | Average order value in last 30 days (R$) |
| `NOTA_MEDIA_AVALIACOES_3M` | Average rating given by the customer in last 3 months (1–5) |
| `ATRASO_ALGUM_PEDIDO_30D` | Any late order in last 30 days (yes/no) |
| `ASSINANTE_CLUBE` | Free-delivery club subscriber (yes/no) |
| `RECOMPRA_7D` | **Target** — reordered within 7 days (1/0) |

EDA: no duplicates, no missing values (if there were: remove duplicates; fill/remove missings — required only for fitting, as in linear regression).

**Stepwise backward:**
- Model 1 (all): highest p-value = **ticket (0.709)** → remove.
- Model 2: next highest = **late order (0.593)** → remove. Remaining coefficients barely changed → no relevant collinearity.
- **Model 3:** all four remaining variables p < 5% (all CIs exclude 0) → **final model**. VIF < 1.4 everywhere (orders and club slightly higher — subscribers order more often — but weak).

**Interpretation (Model 3), holding others fixed:**

| Variable | Coef (log-odds) | Odds ratio e^β | Reading |
|---|---|---|---|
| Days since last order | −0.0793 | 0.924 | Each extra day → log-odds −0.079 (odds ×0.92). Probability rises as **recency decreases** |
| # orders 30d | +0.1168 | 1.124 | Each extra order → log-odds +0.117 (odds ×1.12). Rises with **frequency** |
| Average rating | +0.9483 | 2.58 | Each extra rating point → log-odds +0.948 (odds ×2.6). Rises with **satisfaction** |
| Club subscriber | +0.5240 | 1.69 | Subscribers have log-odds 0.52 higher (odds ×1.7) |

**Final equation:**

$$\hat p = \frac{e^{-3.7387 - 0.0793x_1 + 0.1168x_2 + 0.9483x_3 + 0.5240x_4}}{1 + e^{-3.7387 - 0.0793x_1 + 0.1168x_2 + 0.9483x_3 + 0.5240x_4}}$$

(x₁ = days since last order, x₂ = orders in 30 days, x₃ = average rating, x₄ = 1 if club subscriber.)

> Property: the **average of predicted probabilities equals the sample repurchase rate** exactly (a property of maximum-likelihood logistic regression with an intercept).

**Predictions for new customers** (computed from the equation):

| Customer | Profile | Linear predictor z | p̂ |
|---|---|---|---|
| A | 3 days since last order; 8 orders/month; rating 4.7; subscriber | −3.7387 − 0.238 + 0.934 + 4.457 + 0.524 ≈ **1.94** | **≈ 87%** |
| B | 25 days; 1 order/month; rating 3.9; not subscriber | −3.7387 − 1.983 + 0.117 + 3.698 ≈ **−1.91** | **≈ 13%** |

> Predictions only hold for customers with a profile similar to the training base — extrapolation isn't guaranteed.

---

## 8. From probability to classification

The model gives a probability, but many decisions need a binary answer (who gets the coupon?). Define a **cutoff (threshold)**:
- p̂ ≥ cutoff → **Ŷ = 1**
- p̂ < cutoff → **Ŷ = 0**

For **balanced** data, the natural cutoff is **50%** (predict the more likely class).

---

## 9. Confusion matrix and performance measures

| | **Predicted = 0** | **Predicted = 1** |
|---|---|---|
| **Observed = 0** | True Negative (TN) | False Positive (FP) |
| **Observed = 1** | False Negative (FN) | True Positive (TP) |

A good model concentrates observations on the main diagonal.

| Measure | Formula | Meaning | Case (cutoff 50%) |
|---|---|---|---|
| **Accuracy** | (TP + TN) / N | % correctly classified (0 or 1) | (16,884 + 17,561) / 50,000 = **68.9%** |
| **Sensitivity (recall, TPR)** | TP / (TP + FN) | % of actual 1s detected | 17,561 / 25,038 = **70.1%** |
| **Specificity (TNR)** | TN / (TN + FP) | % of actual 0s detected | 16,884 / 24,962 = **67.6%** |
| **Positive precision (PPV, precision)** | TP / (TP + FP) | % of predicted 1s that are truly 1 | 17,561 / 25,639 = **68.5%** |
| **Negative precision (NPV)** | TN / (TN + FN) | % of predicted 0s that are truly 0 | 16,884 / 24,361 = **69.3%** |

Business reading: of every 100 customers, the model classifies 69 correctly; of 100 who repurchase, it identifies 70; of 100 who don't, it identifies 68; of 100 flagged as repurchasers, 69 actually repurchase; of 100 flagged as non-repurchasers, 69 actually don't.

Suggested interpretation for hit-rate measures (context-dependent):

| Value | Discrimination |
|---|---|
| < 50% | None — worse than random |
| 50 – 60% | Weak |
| 60 – 70% | Satisfactory |
| 70 – 80% | Good |
| > 80% | Very good |

> PPV and NPV have **no reference level**: they depend heavily on how frequent the event is. The rarer the event, the lower the PPV tends to be.

(Extra: **F1-score** = harmonic mean of precision and recall = 2·PPV·TPR / (PPV + TPR).)

---

## 10. ROC curve and AUC

**ROC (Receiver Operating Characteristic)** analysis was developed in the 1950s–60s to evaluate signal detection in radars. In classification, the ROC curve shows how **sensitivity** (y-axis) and **1 − specificity** (x-axis, false-positive rate) vary as the **cutoff moves**.

- The more discriminative the model, the more the curve moves away from the diagonal toward the **top-left corner** (both high).
- The **diagonal** = a model with no separating ability.

**AUC (Area Under the Curve)** summarizes discrimination across **all cutoffs** in one number. It equals the **probability that, drawing one case with the event and one without, the model assigns a higher probability to the event case**.

| AUC | Discrimination |
|---|---|
| < 0.5 | None (worse than random) |
| 0.5 – 0.6 | Low |
| 0.6 – 0.7 | Acceptable |
| 0.7 – 0.8 | Good |
| 0.8 – 0.9 | Very good |
| 0.9 – 0.95 | Excellent |
| > 0.95 | Excellent, **but suspicious*** |

\* Check feature consistency — especially whether any feature has a **deterministic relationship with the target** or refers to a **moment after the prediction** (leakage).

Case: final model **AUC = 0.758 → good** discrimination.

(Extra: **Gini = 2·AUC − 1**, common in credit scoring. **KS** = max difference between the cumulative distributions of scores for events and non-events.)

---

## 11. Class imbalance: the fraud case

In many problems the event is **rare**: fraud, severe default, equipment failure, short-term churn. The model and interpretation stay the same, but the **classification step changes**: the 50% cutoff stops making sense, and **accuracy becomes uninformative**.

> Example: with 1% events, a model that predicts "no event" for everyone has **99% accuracy** without catching a single case.

### Case: online card fraud

**Question:** probability an e-commerce transaction is fraudulent **at authorization time** → reduce fraud losses without blocking too many legitimate transactions (friction, card abandonment).
**Data:** 30,000 recent transactions; **610 confirmed frauds** (after chargeback) → **2.03%** (≈ 50 legit : 1 fraud).

| Variable | Description |
|---|---|
| `VALOR_TRANSACAO` | Transaction amount (R$) |
| `PERCENTIL_VALOR_HISTORICO` | Position of the amount in the customer's own purchase history (0–100) |
| `QTD_TRANSACOES_24H` | # card transactions in the last 24 h |
| `DISPOSITIVO_NOVO` | Purchase from an unrecognized device (yes/no) |
| `MADRUGADA` | Transaction between 0h and 6h (yes/no) |
| `CATEGORIA_ESTABELECIMENTO` | Merchant category: food, apparel, services, electronics, other |
| `FRAUDE` | **Target** (1/0) |

Model 1: **all variables significant** at 5%, all VIF < 1.4, **AUC = 0.862** (very good).

**Cutoff 50% on imbalanced data:**

| Measure | Reading |
|---|---|
| Accuracy **98.0%** | "Excellent" (apparently) |
| Specificity ≈ 100% | Almost no wrongful blocks |
| Sensitivity **≈ 4%** | Only **26 of 610** frauds detected (only 40 transactions flagged) |

98.0% accuracy is practically the same as classifying everything as legitimate (97.97%). The model **does discriminate** (AUC 0.862), but the **cutoff makes it useless**.

**Why:** with 2% events, the model assigns **low probabilities to almost all transactions**; very few exceed 50%.

**Redefining the cutoff:** the usual cutoff for imbalanced data is the **overall event rate** (here **2.03%**) — the natural probability of the event for a random observation. A transaction above 2.03% is more suspicious than average; below, less. (Same logic as 50% for balanced data, where the overall rate is 50%.)

→ With cutoff = 2.03%, accuracy drops a lot, but the model becomes **useful**: it identifies **almost 8 of every 10 frauds**.

**The trade-off:**
- **Lowering the cutoff ↑ sensitivity, ↓ specificity and ↓ PPV**: to catch most frauds, accept blocking some legitimate transactions.
- **No cutoff maximizes sensitivity and specificity at the same time.**
- Accuracy closely follows specificity when the negative class dominates.

> 💡 In practice the cutoff is a **business decision**: weigh the cost of a missed fraud (FN) vs. the cost of blocking a good customer (FP), or set a capacity limit (e.g., the analyst team can review N transactions/day → rank by probability).

Other common tools for imbalance (beyond the course): class weights (`class_weight="balanced"`), under/over-sampling (SMOTE), and PR curves / average precision instead of ROC when positives are very rare.

---

## 12. Regularized logistic regression

Same regularization ideas as in linear regression (1.4): penalize large coefficients → less overfitting, softer collinearity. Only the **cost function** changes.

### Cost function: negative log-likelihood (log-loss)

$$\mathcal{J}(\beta) = -\sum_{i=1}^{n}\Big[y_i\ln p_i + (1-y_i)\ln(1-p_i)\Big]$$

Reading: when the event occurred (yᵢ = 1), cost grows as p̂ moves away from 1; when it didn't (yᵢ = 0), cost grows as p̂ moves away from 0.

| Regularization | Cost function | Effect | Hyperparameters |
|---|---|---|---|
| **Ridge (L2)** | log-loss + $\alpha\sum_j\beta_j^2$ | Shrinks large coefficients **without zeroing**; under collinearity, spreads weight among correlated variables | α ≥ 0 (α = 0 → plain logistic) |
| **LASSO (L1)** | log-loss + $\alpha\sum_j\lvert\beta_j\rvert$ | Penalizes small coefficients proportionally more → **zeroes some** → automatic **variable selection** (alternative to stepwise) | α |
| **Elastic Net** | log-loss + $\alpha\left[(1-\lambda)\sum\beta_j^2 + \lambda\sum\lvert\beta_j\rvert\right]$ | Handles collinearity and drops weak variables at once | α (total strength), λ / ρ ∈ [0, 1] (L1 vs. L2 mix; `l1_ratio`) |

Can be minimized with **SGD** (mini-batches → scales to very large data):

`SGDClassifier(loss="log_loss", penalty="elasticnet", alpha=..., l1_ratio=...).fit(X, y)`

> Note: in scikit-learn's `LogisticRegression`, the strength is `C = 1/α` (smaller C = stronger penalty). Remember to **standardize** features.

---

## 13. End-to-end workflow

```
INITIAL DEFINITIONS
 1. Problem definition   → translate the business question into a BINARY event with an explicit TIME WINDOW
 2. Variable definition  → features that already exist at the moment the model will be applied (prior to the target)
DATA PREPARATION
 3. EDA                  → treat missing values; assess outliers
 4. Bivariate analysis   → compare each feature between the two target groups
 5. Dummies              → one-hot encoding with a reference category
ESTIMATION & SELECTION
 6. Estimation           → fit with all candidate features
 7. Variable selection   → stepwise backward by p-value, one at a time
 8. Collinearity         → VIF; refit without redundant variables; compare
EVALUATION & USE
 9. Performance          → AUC (cutoff-independent discrimination)
10. Cutoff               → choose using sensitivity, specificity (and business costs)
11. Application          → score new data; prioritize actions by ranking probabilities
(iterative: after diagnostics, go back to estimation if needed)
```

---

## 14. Pseudocode

### 14.1 Logistic regression by gradient descent (maximum likelihood)

```text
FUNCTION Sigmoid(z):  RETURN 1 / (1 + exp(-z))

FUNCTION FitLogistic(X[n×k], y[n], η, epochs, α = 0, l1_ratio = 0)
    X ← [1 | standardize(X)]
    β ← zeros(k + 1)
    FOR epoch IN 1..epochs:
        p ← Sigmoid(X · β)
        grad ← Xᵀ · (p - y) / n                          # gradient of the mean log-loss
        penalty ← α · ( l1_ratio · sign(β) + (1 - l1_ratio) · 2β );  penalty[0] ← 0
        β ← β - η · (grad + penalty)
    RETURN β

FUNCTION LogLoss(y, p):  RETURN -mean( y·ln(p) + (1-y)·ln(1-p) )
FUNCTION PredictProba(β, x):  RETURN Sigmoid(β0 + Σ βj xj)
```

### 14.2 Stepwise backward (logistic)

```text
FUNCTION StepwiseBackwardLogit(X, y, α_sig = 0.05)
    feats ← all columns
    LOOP
        m ← Logit(y, add_constant(X[feats])).fit()
        worst ← argmax p_value over feats
        IF p_value[worst] > α_sig: REMOVE worst; CONTINUE
        BREAK
    RETURN m, feats
```

### 14.3 Confusion matrix, metrics, ROC/AUC

```text
FUNCTION Classify(p̂, cutoff):  RETURN [1 IF p ≥ cutoff ELSE 0 FOR p IN p̂]

FUNCTION Metrics(y, ŷ)
    TP ← Σ(y=1 ∧ ŷ=1); TN ← Σ(y=0 ∧ ŷ=0); FP ← Σ(y=0 ∧ ŷ=1); FN ← Σ(y=1 ∧ ŷ=0)
    RETURN { accuracy: (TP+TN)/n, sensitivity: TP/(TP+FN), specificity: TN/(TN+FP),
             ppv: TP/(TP+FP), npv: TN/(TN+FN) }

FUNCTION ROC_AUC(y, p̂)
    points ← []
    FOR cutoff IN sorted(unique(p̂), descending) ∪ {0}:
        m ← Metrics(y, Classify(p̂, cutoff))
        points.append( (1 - m.specificity, m.sensitivity) )
    AUC ← trapezoidal area under points
    # equivalently: AUC = P( p̂(random positive) > p̂(random negative) )
    RETURN points, AUC
```

### 14.4 Choosing a cutoff on imbalanced data

```text
FUNCTION ChooseCutoff(y, p̂, strategy)
    IF strategy = "event_rate":   RETURN mean(y)                      # e.g., 2.03%
    IF strategy = "youden":       RETURN argmax_c (sensitivity(c) + specificity(c) - 1)
    IF strategy = "cost":         RETURN argmin_c ( cost_FN · FN(c) + cost_FP · FP(c) )
    IF strategy = "capacity":     RETURN p̂ value at rank N (top-N most suspicious)
```

Python reference:

```python
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, classification_report
from sklearn.linear_model import SGDClassifier, LogisticRegression

X = sm.add_constant(pd.get_dummies(df[feats], drop_first=True).astype(float))
m = sm.Logit(df["RECOMPRA_7D"], X).fit(); print(m.summary())
np.exp(m.params)                                     # odds ratios
p = m.predict(X)
cut = df["RECOMPRA_7D"].mean()                       # 0.5 if balanced; event rate if imbalanced
confusion_matrix(y, (p >= cut).astype(int)); roc_auc_score(y, p)
SGDClassifier(loss="log_loss", penalty="elasticnet", alpha=1e-4, l1_ratio=0.5)
```

---

## 15. Interview questions

1. **Why not use linear regression for a binary target?** Predictions can fall outside [0, 1], errors aren't normal/homoscedastic; the logit link fixes the range.
2. **How is logistic regression estimated?** Maximum likelihood (minimizing log-loss), via iterative methods (Newton-Raphson/IRLS or gradient descent).
3. **Interpret a coefficient of 0.5.** Each unit increase raises the log-odds by 0.5 → multiplies the odds by e^0.5 ≈ 1.65 (+65%), holding others fixed.
4. **Odds vs. probability?** Odds = p/(1−p). p = 0.2 → odds 0.25.
5. **What does AUC mean?** Probability that a random positive gets a higher score than a random negative; threshold-independent ranking quality.
6. **Why is accuracy misleading with imbalanced classes?** Predicting the majority class gives high accuracy with zero recall.
7. **How do you pick a threshold?** Business costs of FP vs. FN, capacity constraints, Youden's J, target precision/recall; default to the event rate for imbalanced data.
8. **Precision vs. recall trade-off?** Lowering the threshold raises recall and lowers precision.
9. **AUC = 0.99 — what do you suspect?** Data leakage (a feature that is post-event or deterministic of the target).
10. **Does class imbalance bias the coefficients?** Mainly the intercept (calibration); the ranking (AUC) is usually fine. Use proper thresholds, class weights or resampling as needed.
11. **What is log-loss?** −mean[y·ln p + (1−y)·ln(1−p)]; penalizes confident wrong predictions heavily.
12. **L1 vs. L2 in logistic regression?** L1 zeros coefficients (selection), L2 shrinks (stability under collinearity).
13. **Why must features exist before the target window?** Otherwise the model uses information unavailable at scoring time (leakage).
14. **What property does the mean of predicted probabilities have?** With an intercept, it equals the observed event rate in training.

---

## 16. References

- Hosmer, D. W.; Lemeshow, S.; Sturdivant, R. X. *Applied Logistic Regression*. 3rd ed. Wiley, 2013.
- Agresti, A. *Categorical Data Analysis*. 3rd ed. Wiley, 2013.
- Kutner, M. H.; Nachtsheim, C. J.; Neter, J.; Li, W. *Applied Linear Statistical Models*. 5th ed. McGraw-Hill Irwin, 2005.

---

⬅️ Previous: [2.2 — Time Series Forecasting Models](../2.2%20-%20Time%20Series%20Forecasting%20Models/README.md) · 🏠 [Index](../../README.md) · ➡️ Next: [2.4 — Decision Trees](../2.4%20-%20Classification%20-%20Decision%20Trees/README.md)
