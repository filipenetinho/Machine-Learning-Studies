# 1.5 — Projection Models: Regression Trees (CART)

> **Module:** Artificial Intelligence I
> **Case study:** a new bookstore wants to understand how the **cost price** of 17,000 titles varies with their characteristics (category, pages, edition, cover, paper density, color, year).
> **Also covered:** model interpretability (Explainable AI) — impurity importance, permutation importance and **SHAP values**.

---

## Table of Contents

1. [Projection models and decision trees](#1-projection-models-and-decision-trees)
2. [What a regression tree is](#2-what-a-regression-tree-is)
3. [Use cases](#3-use-cases)
4. [Strengths and weaknesses](#4-strengths-and-weaknesses)
5. [How the algorithm works (CART)](#5-how-the-algorithm-works-cart)
6. [Stopping criteria and tree vocabulary](#6-stopping-criteria-and-tree-vocabulary)
7. [Hyperparameters and overfitting](#7-hyperparameters-and-overfitting)
8. [Performance evaluation](#8-performance-evaluation)
9. [Case: bookstore](#9-case-bookstore)
10. [Model interpretability (XAI)](#10-model-interpretability-xai)
11. [Additional case: private-label card spend](#11-additional-case-private-label-card-spend)
12. [Pseudocode](#12-pseudocode)
13. [Interview questions](#13-interview-questions)
14. [References](#14-references)

---

## 1. Projection models and decision trees

**Projection (regression) methods** aim to predict the behavior of a **continuous target**. Linear regression was the first; here we start a new family: **tree-based models**.

- Target **qualitative** → **classification tree** (see 2.4).
- Target **quantitative** → **regression tree** (this module).

No data-driven algorithm is infallible — each has strengths and points of attention.

---

## 2. What a regression tree is

Unlike linear regression (an optimal **equation** from math/statistics), a decision tree is a **computational algorithm** — an ordered sequence of steps.

> A regression tree **sequentially partitions** the data into groups (**nodes**), using the combinations of features (X₁, X₂, …) that best differentiate the **mean of the target Y**.

- The **prediction** of a leaf = the **mean of Y** of the training observations in that leaf.
- Outliers in **Y** distort predictions → consider capping them at percentiles (e.g., P1 and P99, *winsorizing*).
- Outliers in **X** have little or no effect (splits only depend on order).

Classic example: predicting baseball players' salaries from years in the league and hits.

```
                  Years < 4.5?
                 /            \
               yes             no
        ŷ = 5.11           Hits < 117.5?
                           /          \
                       ŷ = 6.00     ŷ = 6.74      (log salary)
```

---

## 3. Use cases

| Question | Motivation |
|---|---|
| Total fuel consumption of a truck fleet by mileage, vehicle traits, traffic | Operational efficiency, lower fueling costs |
| Expected crop yield (soy, corn) from weather, soil quality, irrigation | Storage, transport and sales decisions |
| Expected leads per sales campaign by type, time slot, targeting, channel, investment | Acquisition efficiency, marketing ROI |
| A customer's monthly income from age, tenure, investments, card spend | Customer value with credit-risk control |

---

## 4. Strengths and weaknesses

| ✅ Strengths | ⚠️ Points of attention |
|---|---|
| Easy-to-interpret structure | Strong tendency to **overfit** |
| Models **non-linear** relationships naturally | **Poor extrapolation** (can't predict beyond the range of training Y) |
| Models **interactions¹** between features naturally | Limited predictive performance alone (vs. ensembles) |
| **No scaling** needed | **Unstable**: small data changes → very different tree |
| Robust to outliers in features | Predictions are **piecewise constant** (discontinuous, step-like) |
| Foundation of ensembles: Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost | |

¹ **Interaction effect:** the relationship between X₁ and Y differs depending on the value of X₂. (Trees capture this because a split on X₂ followed by a split on X₁ creates different X₁ effects per branch.)

---

## 5. How the algorithm works (CART)

We study the most common method, **CART — Classification and Regression Trees** (Breiman et al., 1984), implemented in scikit-learn.

**Goal:** reduce the **dispersion (impurity)** of Y within nodes → make nodes increasingly **homogeneous**. At each node, the algorithm searches for the **variable and cut point that most reduce Y's variability** in the resulting nodes.

### Impurity measures

With ŷ = node mean:

$$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat y_i)^2 \qquad MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat y_i|$$

**MSE is the usual choice** — it penalizes larger errors more severely. (Minimizing MSE within a node ⇔ the node's prediction is its mean; for MAE it's the median.)

Illustration: original data, ŷ = 30, MSE = 200, MAE = 12 → split into 3 groups (ŷ = 16, 28, 46): MSE = 48, MAE ≈ 6 → split into 5 groups (ŷ = 10, 20, 30, 40, 50): MSE = MAE = 0 (total impurity reduction).

### Step by step

**Step 0.** The tree starts with a **root node (node 0)** containing **all** observations. Dispersion is maximal.

**Step 1.** Split node 0 into two child nodes by a **binary condition** on some feature. The variable and cut are chosen automatically to **maximize the impurity reduction**:

$$\Delta I = I(0) - \left[\frac{n_1}{n_0}\,I(1) + \frac{n_2}{n_0}\,I(2)\right]$$

- ΔI: reduction in dispersion (MSE or MAE)
- I(0), I(1), I(2): impurities of parent and children
- n₀, n₁, n₂: number of observations in each node (children are **weighted by size**)

Split examples:
- Quantitative X: `node 1: X ≤ 100` | `node 2: X > 100`
- Qualitative X: `node 1: X = "Category A"` | `node 2: X ≠ "Category A"`

> In scikit-learn, qualitative variables must be **one-hot encoded** first.

**Step 2.** Repeat recursively on each child. A perfectly "pure" tree would have zero dispersion in every node (= memorizing the data → overfitting).

This is a **greedy** algorithm: it chooses the best split *now*, without looking ahead.

---

## 6. Stopping criteria and tree vocabulary

A node stops splitting when it meets one of:

1. No variable reduces impurity **enough** (e.g., < 1% reduction).
2. Too few observations in the node or in the resulting children (e.g., < 50).
3. The branch reached a **maximum depth** (e.g., level 3).

Vocabulary:

```
                     [Root / initial node]                ← level 0
                      /                \
         [Intermediate node]      [Intermediate node]     ← level 1
           /          \              /         \
      [Leaf]       [Leaf]       [Leaf]       [Leaf]       ← level 2  (final nodes / leaves)
```

- **Parent node:** the node being split. **Child nodes:** the two resulting nodes.
- **Leaves (terminal nodes):** each one is a **profile**. With 4 leaves, the data was divided into 4 profiles.

---

## 7. Hyperparameters and overfitting

Hyperparameters control growth and complexity, node count and size, split conditions, the overfitting risk, and the balance between performance and interpretability. The main ones are the **stopping criteria**:

| Hyperparameter (scikit-learn) | Meaning | Usual values |
|---|---|---|
| `min_impurity_decrease` | Minimum impurity reduction to justify a split | 0 – 0.05 |
| `min_samples_split` | Minimum # observations in the **parent** to attempt a split | Depends on dataset size |
| `min_samples_leaf` | Minimum # observations in each **child** (leaf) | Depends on dataset size |
| `max_depth` | Maximum # levels | 2 – 10 |

`DecisionTreeRegressor(...).fit(X, y)`

> **Thought experiment:** with 200 rows and `min_impurity_decrease=0, min_samples_split=2, min_samples_leaf=1, max_depth=10`, the tree can grow until leaves have one observation each → **memorizes the data → overfitting**.

Trees overfit more easily than linear regression because each split greedily optimizes the fit. Signs of overfitting: **many splits** and **leaves with very few observations**. Remedy: holdout + cross-validation + hyperparameter search (e.g., random search).

---

## 8. Performance evaluation

Same metrics as other projection models, on train, internal test and external test:

$$RMSE = \sqrt{\tfrac{1}{n}\sum(y_i-\hat y_i)^2}\quad MAE = \tfrac{1}{n}\sum|y_i-\hat y_i|\quad MAPE = \tfrac{1}{n}\sum\left|\tfrac{y_i-\hat y_i}{y_i}\right|\quad R^2 = 1-\tfrac{\sum(y_i-\hat y_i)^2}{\sum(y_i-\bar y)^2}$$

---

## 9. Case: bookstore

**Business:** a book retailer starting next month wants to understand how cost price varies with characteristics → support negotiation with publishers and pricing.

Bivariate insights:
- University-category books are pricier; hardcover slightly pricier than paperback.
- Color printing pricier than black; high-density paper slightly pricier.
- More pages → higher price (high dispersion).

**Modeling strategy:**
- Holdout: 80% train / 20% external test.
- 5-fold CV inside train.
- Random search, 100 iterations: `min_impurity_decrease ∈ [0, 0.05]`, `min_samples_split ∈ [100, 200]`, `min_samples_leaf ∈ [50, 100]`, `max_depth ∈ [2, 10]`.
- Choice: lowest mean RMSE on internal-test folds, stable vs. train folds and with little variation on external test. Random seed: 123.

**Chosen hyperparameters:** `min_impurity_decrease=0.0466`, `min_samples_split=121`, `min_samples_leaf=54`, `max_depth=5` → retrained tree with **12 profiles (leaves)**.

Example profiles:
- **High price:** > 701 pages, university category, hardcover → **ŷ = R$ 108.53**.
- **Low price:** 350–400 pages, not university, black print → **ŷ = R$ 40.69**.

Performance: **stable RMSE** across train, internal test and external test. Residuals symmetric around zero with roughly constant variance.

**Remarks:**
1. Leaf configuration is **very sensitive** — small changes in data or hyperparameters can yield profiles with quite different interpretations.
2. **Low diversity of predicted values**: every observation in a leaf gets the same ŷ → "columned" pattern in the residuals-vs-predicted plot.

---

## 10. Model interpretability (XAI)

ML models focus on performance more than interpretation, but there's growing demand to explain them → **Explainable AI (XAI)**. Three techniques:

### 10.1 Impurity-based importance (trees only)

Total impurity reduction produced by a variable across all nodes split on it, weighted by the share of observations in each split:

$$\text{Importance}(X) = \sum_{j=1}^{J} \frac{n_j}{n}\,\Delta I_j(X)$$

- J: number of split nodes; nⱼ: observations in node j; n: total observations
- ΔIⱼ(X): impurity reduction from X at node j (0 if the split doesn't use X)

Usually normalized to sum to 1. `model.feature_importances_`.
Case ranking: **pages > category (university) > color > cover type**.

### 10.2 Permutation importance (any model)

**Randomly shuffle** one feature's values — breaking its link with the target — and measure how much performance degrades:

$$\text{Importance}(X) = RMSE_{perm}(X) - RMSE_{orig}$$

Repeat the shuffle many times (≥ 30 typically) and average. `sklearn.inspection.permutation_importance(...)`. Same ranking in the case.

### 10.3 SHAP values

**SHapley Additive exPlanations** (Lundberg & Lee, 2017), inspired by Shapley's (1953) cooperative game theory: what is each *player* (feature) worth to the *game's outcome* (prediction)?

SHAP values φᵢ(Xⱼ) are computed **for each observation i and each feature j**: the impact of feature Xⱼ on the model's prediction for observation i.

> **Question SHAP answers:** "How much did this feature push the prediction up or down?"

$$\hat y_i = \hat y_0 + \sum_{j=1}^{k}\phi_i(X_j)$$

- Start at a **base prediction** ŷ₀ (the average prediction); each feature adds a positive or negative "push" until the final prediction.
- **SHAP > 0** → pushes prediction **up**; **SHAP < 0** → pushes **down**; large |SHAP| → **more influence**.
- **Global importance:** rank features by **mean |SHAP|** across the dataset.

**Formula ("hard level"):**

$$\phi_i(X_j) = \sum_{S \subseteq K\setminus\{X_j\}} \frac{|S|!\,(k-|S|-1)!}{k!}\left[\hat y_i(S\cup\{X_j\}) - \hat y_i(S)\right]$$

- K: set of all k features; S: each subset not containing Xⱼ
- **[ŷᵢ(S ∪ {Xⱼ}) − ŷᵢ(S)]**: marginal impact of adding Xⱼ to subset S
- **Weight** |S|!(k−|S|−1)!/k!: share of feature orderings where exactly S comes before Xⱼ. Larger weights when S is very small (Xⱼ's impact in scarcity) or very large (impact when many features are present)
- The sum = weighted average of Xⱼ's marginal contributions over all possible subsets

Numeric example: two features; prediction ŷ = 12.5; φᵢ(X₁) = −0.15 → X₁ lowered this prediction by 0.15; without X₁ the model would predict ~12.65.

Case (`shap.Explainer(...)`):
- Global ranking by mean |SHAP|: pages > color > category (university) > cover > category (children's).
- **Beeswarm plot** (direction per observation): higher prices for books with **more pages, not black print, university category, not paperback**.
- **Waterfall plot** (one observation): base ŷ₀ = 38.62 → final ŷ = 44.95; 437 pages: **+R$ 8.80**; black print: **−R$ 1.28**; not university: **−R$ 0.96**; …

### 10.4 Comparing the three

| | Impurity reduction | Permutation | SHAP |
|---|---|---|---|
| Looks at… | Inside the tree | Shuffle the feature, measure the drop | Splits each prediction into "pushes" |
| Measures | Reduction of Gini / entropy / MSE in splits | Performance drop when a feature is shuffled | Contribution of each feature to each prediction |
| Best for | Quick screening in tree models | Validating importance on a business metric | Explaining individual cases + global view |
| Strength | Very fast, automatic | **Model-agnostic** | Shows **direction** (up/down) |
| Watch out | **Biased toward features with many possible splits** (high cardinality / continuous) | **Correlated features can mask importance** | More complex; does **not** prove causality |

**In practice:** impurity to explore quickly → permutation to validate impact → SHAP to explain decisions and cases.

> 🎯 **Key message:** all three explain the **model**, not the real world. They do **not prove causality**.

---

## 11. Additional case: private-label card spend

A department-store chain wants to estimate the potential spend of new customers with the store's private-label card (after 3 months of ownership) → target acquisition/activation, set initial limits, tailor communication. Exercise steps:

a) Bivariate analysis (scatter for quantitative, boxplots for qualitative).
b) Train/external-test split; **fit one-hot encoding on train**, apply to test.
c) Define hyperparameter search ranges.
d) 5-fold CV + 100-iteration random search; store mean train RMSE, mean internal-test RMSE, external-test RMSE; choose a final model.
e) Retrain on full train; interpret the leaf profiles — do they make business sense?
f) External-test MAE, MAPE, R²; residual plots.
g) Importance by impurity, permutation and SHAP — do they agree?
h) SHAP directions — coherent with the bivariate analysis and business intuition?

---

## 12. Pseudocode

### 12.1 Growing a CART regression tree

```text
FUNCTION BuildTree(data, depth, params)
    node.prediction ← mean(data.y)
    node.n ← |data|
    IF depth = params.max_depth
       OR |data| < params.min_samples_split
       OR all y equal:
        RETURN Leaf(node)

    best ← (gain = 0)
    parent_impurity ← MSE(data.y)                     # = variance of y in node
    FOR each feature f:
        FOR each candidate threshold t in sorted unique values of f (midpoints):
            L ← data[f ≤ t];  R ← data[f > t]
            IF |L| < params.min_samples_leaf OR |R| < params.min_samples_leaf: CONTINUE
            child_impurity ← (|L|/|data|)·MSE(L.y) + (|R|/|data|)·MSE(R.y)
            gain ← parent_impurity - child_impurity
            IF gain > best.gain: best ← (f, t, gain, L, R)

    IF best.gain · |data| / N_total < params.min_impurity_decrease:   # weighted decrease
        RETURN Leaf(node)

    node.rule ← (best.f ≤ best.t)
    node.left  ← BuildTree(best.L, depth + 1, params)
    node.right ← BuildTree(best.R, depth + 1, params)
    RETURN node

FUNCTION Predict(node, x)
    WHILE node is not a leaf:
        node ← node.left IF x[node.rule.f] ≤ node.rule.t ELSE node.right
    RETURN node.prediction                             # mean of y in that leaf
```

### 12.2 Permutation importance

```text
FUNCTION PermutationImportance(model, X_test, y_test, n_repeats = 30)
    base ← RMSE(y_test, model.predict(X_test))
    FOR each feature j:
        drops ← []
        REPEAT n_repeats times:
            X_perm ← copy(X_test);  SHUFFLE column j of X_perm
            drops.append( RMSE(y_test, model.predict(X_perm)) - base )
        importance[j] ← mean(drops)
    RETURN importance
```

### 12.3 Exact SHAP value (conceptual, exponential cost)

```text
FUNCTION ShapValue(model, x_i, j, all_features K)
    k ← |K|;  phi ← 0
    FOR each subset S of K \ {j}:
        weight ← |S|! · (k - |S| - 1)! / k!
        phi ← phi + weight · ( f(x_i; S ∪ {j}) - f(x_i; S) )
        # f(x; S) = expected model prediction when only features in S are "known"
        #           (others marginalized over background data)
    RETURN phi
# Property: base_value + Σ_j ShapValue(...) = model.predict(x_i)
# TreeSHAP computes this exactly in polynomial time for tree models.
```

Python reference:

```python
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import RandomizedSearchCV
from sklearn.inspection import permutation_importance
from scipy.stats import randint, uniform
import shap

space = {"min_impurity_decrease": uniform(0, 0.05),
         "min_samples_split": randint(100, 201),
         "min_samples_leaf": randint(50, 101),
         "max_depth": randint(2, 11)}
search = RandomizedSearchCV(DecisionTreeRegressor(random_state=123), space, n_iter=100, cv=5,
                            scoring="neg_root_mean_squared_error", return_train_score=True,
                            random_state=123).fit(X_train, y_train)
tree = search.best_estimator_
plot_tree(tree, feature_names=X_train.columns, filled=True)
tree.feature_importances_
permutation_importance(tree, X_test, y_test, n_repeats=30, random_state=123)
explainer = shap.Explainer(tree, X_train); sv = explainer(X_test)
shap.plots.beeswarm(sv); shap.plots.waterfall(sv[0])
```

---

## 13. Interview questions

1. **How does a regression tree choose a split?** Greedily: for every feature and threshold, compute the weighted impurity (MSE) of the children; pick the split with the largest impurity decrease.
2. **What does a leaf predict?** The mean of training Y in that leaf (median if criterion is MAE).
3. **Why do trees overfit and how do you control it?** They can keep splitting until leaves are pure. Control via max_depth, min_samples_leaf, min_samples_split, min_impurity_decrease, pruning (ccp_alpha), and CV tuning.
4. **Do trees need feature scaling?** No — splits depend on ordering only.
5. **Why can't trees extrapolate?** Predictions are means of training targets — never outside the training Y range.
6. **What is an interaction effect and why do trees capture it?** The effect of X₁ on Y depends on X₂; nested splits create different X₁ effects per branch.
7. **Why are single trees unstable?** Greedy top splits change with small data perturbations, cascading into a different tree → motivates bagging / Random Forest.
8. **Impurity vs. permutation importance?** Impurity is fast but biased toward high-cardinality features and computed on training data; permutation is model-agnostic and computed on held-out data, but correlated features share/mask importance.
9. **What is a SHAP value?** The Shapley-value contribution of a feature to one prediction relative to the base value; contributions sum to the prediction minus the base value.
10. **Does a high SHAP importance mean causality?** No — it explains the model's behavior, not the data-generating process.

---

## 14. References

- Breiman, L. et al. *Classification and Regression Trees*. Chapman & Hall, 1984.
- Grus, J. *Data Science from Scratch*. 2nd ed. O'Reilly, 2019.
- Morettin, P. A.; Singer, J. M. *Estatística e Ciência de Dados*. LTC, 2022.
- James, G. et al. *An Introduction to Statistical Learning — with Applications in Python*. 2nd ed. Springer, 2023.
- Lundberg, S. M.; Lee, S.-I. "A Unified Approach to Interpreting Model Predictions." *NeurIPS*, 2017.
- Shapley, L. S. "A Value for n-Person Games." *Contributions to the Theory of Games II*, 1953.

---

⬅️ Previous: [1.4 — Validation and Regularization](../1.4%20-%20Validation%20and%20Regularization/README.md) · 🏠 [Index](../../README.md) · ➡️ Next: [1.6 — Random Forest and Boosting (Regression)](../1.6%20-%20Random%20Forest%20and%20Boosting%20for%20Regression/README.md)
