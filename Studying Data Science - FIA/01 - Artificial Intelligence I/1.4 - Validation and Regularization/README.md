# 1.4 — Model Validation and Regularization

> **Module:** Artificial Intelligence I
> **Case study:** used-car resale pricing (vehicle age, mileage, engine power → resale price).
> **Big question:** a model that fits the *sample* well — does it also work on the *population* (new data)? **Not necessarily.**

---

## Table of Contents

1. [Why validate?](#1-why-validate)
2. [Underfitting and overfitting](#2-underfitting-and-overfitting)
3. [The bias–variance trade-off](#3-the-biasvariance-trade-off)
4. [Validation strategies](#4-validation-strategies)
5. [Case: used-car pricing](#5-case-used-car-pricing)
6. [Regularized linear regression: Ridge, LASSO, Elastic Net](#6-regularized-linear-regression-ridge-lasso-elastic-net)
7. [Stochastic Gradient Descent (SGD)](#7-stochastic-gradient-descent-sgd)
8. [Feature scaling (standardization)](#8-feature-scaling-standardization)
9. [Hyperparameters and how to tune them](#9-hyperparameters-and-how-to-tune-them)
10. [Statistical vs. ML philosophy](#10-statistical-vs-ml-philosophy)
11. [Practice case studies](#11-practice-case-studies)
12. [Pseudocode](#12-pseudocode)
13. [Interview questions](#13-interview-questions)
14. [References](#14-references)

---

## 1. Why validate?

In 1.3 we evaluated fit quality **on the same data used to build the model**. For a **predictive** model, we must guarantee its predictions **generalize** to new observations.

```
Bad model on the sample   → bad model on the population      ✔ (always)
Good model on the sample  → good model on the population?    ✘ (not necessarily — overfitting!)
```

> 💡 *Reflection:* why didn't we validate in 1.3? Because those models were mostly **diagnostic** (explaining relationships, using statistical inference — p-values and CIs already account for sampling uncertainty). Validation becomes essential when the goal is **prediction**.

---

## 2. Underfitting and overfitting

| | Underfitting | Overfitting |
|---|---|---|
| Training performance | **Bad** | **Excellent** |
| Performance on new data | Bad | **Bad** |
| Why | Model too simple; can't capture the pattern | Model learned **sample-specific noise** instead of global patterns |
| How you notice | Error metrics in diagnostics are already poor | Big gap between training and test performance |

The high quality of an overfitted model on the sample is **artificial**: it memorized the training data.

---

## 3. The bias–variance trade-off

The **total prediction error** on new data, as a function of model complexity, decomposes into:

- **Bias:** systematic deviation of predictions from the true values.
- **Variance:** how much predictions vary (instability across samples).

$$\text{Expected error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible noise}$$

```
error
  │\                                   /
  │ \      total error               /
  │  \___                      ___/  ← variance grows
  │      \____           _____/
  │   bias²   \___ ★ ___/
  │               ‾‾‾‾‾          (★ = sweet spot)
  └──────────────────────────────────► model complexity
    UNDERFITTING                 OVERFITTING
  (high bias, low variance)    (low bias, high variance)
  "bad but stable"             "good but unstable"
```

---

## 4. Validation strategies

From least to most rigorous:

| # | Method | Idea |
|---|---|---|
| 1 | **Holdout (simple) validation** | Reserve part of the sample to test the model after building it |
| 2 | **k-fold cross-validation** | Rotate which part is held out during building → broader testing; also used to **tune hyperparameters** |
| 3 | **Holdout + cross-validation** | Combine 1 and 2 — **preferred** |
| 4 | **Nested cross-validation** | Variant of 3, also rotating the outer holdout many times |

### 4.1 Holdout validation

1. Randomly split into **train** (70–80%) and **test**.
2. Fit one or more candidate models on train.
3. Apply each to test.
4. Compare train vs. test performance (R², MAE, MAPE for regression).
5. Choose a model that **performs well on test and is stable vs. train**.
6. Retrain on the full training set for final predictions.

### 4.2 k-fold cross-validation

```
Full sample split into k folds (e.g., k = 5):

Iteration 1:  [TEST ][train][train][train][train]
Iteration 2:  [train][TEST ][train][train][train]
Iteration 3:  [train][train][TEST ][train][train]
Iteration 4:  [train][train][train][TEST ][train]
Iteration 5:  [train][train][train][train][TEST ]

CV score = average of the k test scores (also look at their spread)
```

- ✅ More robust: every observation is used for testing once.
- ❌ No fully untouched set for the **final** choice: the same data serves as train and test at different times.

### 4.3 Holdout + cross-validation (recommended)

```
FULL SAMPLE
├── OUTER LAYER: TRAIN (80%) ─────────────────────────┐   TEST (20%) = "external test"
│     INNER LAYER: k-fold CV on the training set      │   used ONCE at the end
│       fold 1: [int.test][train][train][train][train]│
│       fold 2: [train][int.test][train][train][train]│
│       ...                                           │
└─────────────────────────────────────────────────────┘
```

- **Internal test** (CV folds): choose model / hyperparameters.
- **External test** (holdout): final unbiased check of generalization.

---

## 5. Case: used-car pricing

A dealership wants the ideal resale price of a used car from its features (age, mileage, engine power).

> 💡 **Sample design tip:** include only transactions that represent the behavior you want the model to reproduce. Exclude by business rule: cars that sat unsold for too long, auction vehicles, employee sales, collector cars, etc.

- Correlogram: age and mileage both correlate with price — and **with each other (r = 0.951)**.
- **Model 1** (all features, 80% train): age and power significant; **mileage not significant** (only ~R$ 89 impact per 1,000 km?!). Contradicts the bivariate analysis → **VIF very high (~11)** for age and mileage → collinearity.
- **Model 2** (drop mileage): all significant; age coefficient changed a lot, its standard error shrank and CI narrowed; VIF ≈ 1.0.

Validation (80/20 holdout + 5-fold CV inside train):

| | Model 1 (age, km, power) | Model 2 (age, power) |
|---|---|---|
| Max VIF | ≈ 11 | ≈ 1.0 |
| Coefficients | Unreliable | Stable & interpretable |
| MAE train (fold avg.) | R$ 7,551 | R$ 7,715 |
| MAE internal test (fold avg.) | R$ 7,766 | R$ 7,840 |
| MAE external test | R$ 7,588 | R$ 7,660 |

Both showed some drop in adjusted R² from train to external test, but MAE barely changed → **no major deterioration**. Treating collinearity didn't change predictive performance much — but made interpretation safe.

> **Key lesson:** collinearity hurts **interpretation** more than **prediction**.

**But what if there are *many* correlated features?** Deciding which to keep becomes hard → use **regularization**.

---

## 6. Regularized linear regression: Ridge, LASSO, Elastic Net

**Regularization** penalizes variables with little contribution: it **shrinks** or **zeroes** coefficients of irrelevant variables, performing variable selection and **controlling overfitting**.

### Traditional OLS cost function

$$J(\beta) = \sum_{i=1}^{n}\left(y_i - \hat y_i\right)^2$$

(Sum of squared residuals — equivalent to minimizing RMSE.)

### Ridge (L2)

$$J(\beta) = \sum_i (y_i - \hat y_i)^2 + \alpha \sum_{j=1}^{k} \beta_j^2$$

- Penalizes **large** coefficients → **shrinkage** (pulls them toward zero, but rarely exactly zero).
- Especially useful under **collinearity** (spreads weight among correlated features, stabilizes estimates).

### LASSO (L1)

$$J(\beta) = \sum_i (y_i - \hat y_i)^2 + \alpha \sum_{j=1}^{k} |\beta_j|$$

- Penalizes **absolute** values → tends to set small coefficients **exactly to zero** → **automatic feature selection**, sparser model.

### Elastic Net (L1 + L2)

$$J(\beta) = \sum_i (y_i - \hat y_i)^2 + \alpha\left[\lambda \sum_j |\beta_j| + (1-\lambda)\sum_j \beta_j^2\right]$$

- Combines both: handles collinearity **and** selects variables.
- **α > 0:** overall penalty strength (α = 0 → plain OLS; larger α → stronger penalty).
- **λ ∈ [0, 1]:** mix between LASSO and Ridge (e.g., 60% LASSO, 40% Ridge). In scikit-learn this is `l1_ratio`.

> 💡 The intercept β₀ is not penalized.

| | Ridge (L2) | LASSO (L1) | Elastic Net |
|---|---|---|---|
| Penalty | Σβ² | Σ\|β\| | mix |
| Zeroes coefficients? | No (shrinks) | **Yes** | Yes |
| Handles collinearity | **Best** | Picks one arbitrarily | Good |
| Feature selection | No | **Yes** | Yes |

Geometric intuition: the L1 constraint region is a diamond with corners on the axes, so the optimum often lands on a corner (a coefficient = 0); the L2 region is a circle, so it doesn't.

In Python (as in class): `SGDRegressor(penalty="l2"|"l1"|"elasticnet", alpha=..., l1_ratio=...)`. Alternatives: `Ridge`, `Lasso`, `ElasticNet`.

---

## 7. Stochastic Gradient Descent (SGD)

Regularized models in scikit-learn (via `SGDRegressor`) are fitted with **SGD** instead of the closed-form OLS solution.

**Idea:**
- Search the coefficients that minimize the cost function **by successive steps**: at each step move in the direction that most reduces the error (the negative **gradient**), updating all coefficients simultaneously.
- Instead of the whole dataset at each step, use **subsamples** (mini-batches / single observations) → each step is faster and scales to big data.
- Step size = **learning rate**, which can adapt over iterations (big steps first, fine adjustments near the minimum).

```
        cost J(β)
          │ \                 /
          │  \  ●→           /
          │   \    ●→       /
          │    \      ●→   /
          │     \       ●→/
          │      \___●___/        ← minimum
          └────────────────────► β
```

Update rule: $\beta \leftarrow \beta - \eta \cdot \nabla J(\beta)$, where η is the learning rate.

---

## 8. Feature scaling (standardization)

**Is SGD affected by the scale of the features? Yes.** The penalty terms depend directly on coefficient magnitudes, and gradients depend on feature scales. Features with large values (e.g., mileage in km vs. age in years) would dominate.

→ **Standardize** features before fitting:

$$z = \frac{x - \bar x}{s}$$

(mean 0, standard deviation 1). Fit the scaler on **training data only** and apply it to test/production (avoid leakage).

---

## 9. Hyperparameters and how to tune them

**Hyperparameters** govern *how the algorithm works*. They are **not estimated** from the data during fitting — they're set beforehand. (Parameters = coefficients β, learned; hyperparameters = α, λ, learning rate…, chosen.)

Common hyperparameters in regularized regression (scikit-learn names):

| Name | Meaning |
|---|---|
| `penalty` | Type of penalty: `l2`, `l1`, `elasticnet` |
| `alpha` | Strength of the penalty (α) |
| `l1_ratio` | LASSO/Ridge mix (λ) |
| `eta0` | Initial learning rate |
| `learning_rate` | Schedule for updating the learning rate (`constant`, `optimal`, `invscaling`, `adaptive`) |

Optimal values depend on the data (complexity, size…) → they must be **tuned per model**, usually via **cross-validation**.

### Tuning methods

| Method | How | Pros / cons |
|---|---|---|
| **Grid search** | Tests **all** combinations in a defined grid | Exhaustive, but cost grows exponentially with # hyperparameters |
| **Randomized search** | Samples a fixed number of random combinations from ranges | Usually as good as grid with far fewer trials (explores more distinct values) → **best cost/quality balance** |
| **Bayesian search** | Models the cost function from past trials to choose the next promising values | Most efficient, but more complex and sequential (hard to parallelize) |

**How to choose the final hyperparameters** — a model with:
- **Low** average error on the internal-test folds;
- **Stable** error internal-test vs. train (no overfitting);
- **Stable** error internal-test vs. external test (generalizes).

Then **retrain on the full training set**. In Python: `RandomizedSearchCV(...)`.

---

## 10. Statistical vs. ML philosophy

With regularization:
- We **didn't** select variables by p-value — **LASSO** zeroes irrelevant ones.
- We **didn't** remove variables for collinearity — **Ridge** shrinkage controls it.

Regularization is aligned with the **machine-learning philosophy: focus on optimization (prediction), less on statistical interpretation.**

---

## 11. Practice case studies

Each asks for a **diagnostic** (which features matter and how much) and a **predictive** answer (estimate for a new observation):

| Case | Target |
|---|---|
| A. Used-car pricing (extended) | Resale listing price |
| B. Short-term rental pricing | Nightly rate |
| C. Professionals' compensation (HR consultancy) | Monthly salary |
| D. Health-insurance cost | Annual cost per insured |
| E. E-commerce customer spend | Annual spend |
| F. Marketing mix & sales | Sales by market (media investment + market traits) |
| G. Residential energy consumption | Monthly kWh |
| H. School performance | Final grade |
| I. Freight pricing | Shipping cost (weight, distance, mode, service level) |
| J. Agricultural productivity | Soybean yield per plot |

---

## 12. Pseudocode

### 12.1 Holdout + k-fold CV + randomized search

```text
PROCEDURE TrainWithValidation(X, y, model_family, search_space, k = 5, n_iter = 50)
    (X_train, y_train), (X_ext_test, y_ext_test) ← random_split(X, y, test_size = 0.20)

    best ← NONE
    REPEAT n_iter times:                                  # randomized search
        h ← sample one hyperparameter combination from search_space
        folds ← split X_train into k folds
        train_errors ← []; val_errors ← []
        FOR each fold f:
            X_tr, y_tr ← X_train without fold f
            X_val, y_val ← fold f
            scaler ← fit_standardizer(X_tr)               # fit on train part ONLY
            m ← model_family(h).fit(scaler(X_tr), y_tr)
            train_errors.append( MAE(y_tr,  m.predict(scaler(X_tr))) )
            val_errors.append(   MAE(y_val, m.predict(scaler(X_val))) )
        score ← mean(val_errors)
        IF score < best.score: best ← (h, score, mean(train_errors))

    # check stability: mean(train_errors) ≈ mean(val_errors)  → no overfitting
    scaler ← fit_standardizer(X_train)
    final ← model_family(best.h).fit(scaler(X_train), y_train)    # retrain on full train
    ext_error ← MAE(y_ext_test, final.predict(scaler(X_ext_test)))
    # check: ext_error ≈ best.score → generalizes
    RETURN final, scaler, best.h, ext_error
```

### 12.2 Elastic Net fitted with SGD

```text
FUNCTION SGD_ElasticNet(X[n×k], y[n], alpha, l1_ratio, eta0, epochs, batch_size)
    X ← standardize(X)
    β0 ← 0;  β[1..k] ← 0
    FOR epoch IN 1..epochs:
        SHUFFLE rows of (X, y)
        FOR each mini-batch B of size batch_size:
            ŷ ← β0 + X[B] · β
            err ← ŷ - y[B]
            grad_β0 ← (2 / |B|) · Σ err
            grad_β  ← (2 / |B|) · X[B]ᵀ · err
                       + alpha · ( l1_ratio · sign(β) + (1 - l1_ratio) · 2 · β )
            η ← learning_rate_schedule(eta0, epoch)       # large steps → fine steps
            β0 ← β0 - η · grad_β0                         # intercept not penalized
            β  ← β  - η · grad_β
    RETURN β0, β

# l1_ratio = 0 → Ridge;  l1_ratio = 1 → LASSO;  alpha = 0 → plain OLS
```

### 12.3 Grid vs. randomized search

```text
GridSearch(space):        FOR every combination in cartesian_product(space): evaluate by CV
RandomizedSearch(space):  FOR i in 1..n_iter: h ← random draw from each range; evaluate by CV
BayesianSearch(space):    fit surrogate model of score(h) on past trials;
                          pick next h maximizing expected improvement; repeat sequentially
```

Python reference:

```python
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from scipy.stats import loguniform, uniform

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
pipe = Pipeline([("scaler", StandardScaler()), ("model", SGDRegressor(max_iter=5000))])
space = {
    "model__penalty": ["l2", "l1", "elasticnet"],
    "model__alpha": loguniform(1e-5, 1e1),
    "model__l1_ratio": uniform(0, 1),
    "model__eta0": loguniform(1e-4, 1e-1),
    "model__learning_rate": ["constant", "optimal", "invscaling", "adaptive"],
}
search = RandomizedSearchCV(pipe, space, n_iter=50, cv=5,
                            scoring="neg_mean_absolute_error", return_train_score=True)
search.fit(X_tr, y_tr)
print(search.best_params_, -search.best_score_)
print("external MAE:", mean_absolute_error(y_te, search.best_estimator_.predict(X_te)))
```

---

## 13. Interview questions

1. **Explain the bias–variance trade-off.** Simple models: high bias/low variance (underfit). Complex models: low bias/high variance (overfit). Aim for the complexity minimizing total error on unseen data.
2. **How do you detect overfitting?** Training error ≪ validation/test error.
3. **Holdout vs. k-fold CV?** Holdout is cheap but depends on one random split; k-fold uses all data for validation and gives a variance estimate. Best: k-fold for tuning + a final untouched holdout.
4. **Why keep an external test set if you already do CV?** CV scores are used to *choose* the model, so they're optimistically biased; the untouched holdout gives an unbiased final estimate.
5. **Ridge vs. LASSO?** L2 shrinks (good for collinearity); L1 zeroes (feature selection). Elastic Net mixes both.
6. **Why must features be scaled before regularization/SGD?** Penalties and gradients depend on coefficient magnitude, which depends on feature scale.
7. **Parameter vs. hyperparameter?** Parameters are learned from data (β); hyperparameters are set before training (α, learning rate, tree depth) and tuned with CV.
8. **Grid vs. random vs. Bayesian search?** Exhaustive vs. random sampling (usually more efficient) vs. model-guided sequential search.
9. **What happens with α → ∞ in Ridge?** All coefficients → 0 (model predicts the mean) → underfitting.
10. **Does collinearity hurt prediction?** Usually little; it mostly hurts interpretation and coefficient stability.
11. **Where can data leakage sneak into CV?** Fitting the scaler/imputer/feature selection on the full dataset before splitting. Use pipelines.
12. **What is the learning rate in gradient descent?** Step size; too large → diverge/oscillate; too small → slow convergence.

---

## 14. References

- Forsyth, D. A. *Applied Machine Learning*. Springer, 2019.
- James, G. et al. *An Introduction to Statistical Learning — with Applications in Python*. 2nd ed. Springer, 2023.
- Grus, J. *Data Science from Scratch: First Principles with Python*. 2nd ed. O'Reilly, 2019.
- Raschka, S. *Model Evaluation, Model Selection, and Algorithm Selection in Machine Learning*. arXiv:1811.12808, 2018.
- Fortmann-Roe, S. *Understanding the Bias-Variance Tradeoff*. <https://scott.fortmann-roe.com/docs/BiasVariance.html>
- Amidi, A.; Amidi, S. *CS-229 Machine Learning Tips and Tricks cheatsheet*. Stanford.

---

⬅️ Previous: [1.3 — Linear Regression](../1.3%20-%20Linear%20Regression/README.md) · 🏠 [Index](../../README.md) · ➡️ Next: [1.5 — Regression Trees](../1.5%20-%20Regression%20Trees/README.md)
