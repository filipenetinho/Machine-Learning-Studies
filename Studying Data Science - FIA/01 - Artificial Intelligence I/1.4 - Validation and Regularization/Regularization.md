# Preventing Overfitting with Regularization

> 📗 **Companion notes** for *The StatQuest Illustrated Guide to Machine Learning* (Josh Starmer), **Chapter 9**.
> Written in my own words with my own examples. Course notes: [1.4 — Validation and Regularization](README.md).

---

## 1. The idea: trade a little bias for a lot less variance

With few training points, least squares can fit them *too* well. The slope ends up steep because it chases those exact points, and predictions on new data suffer.

**Regularization** adds a **penalty for large coefficients** to the loss. The fitted line becomes a bit worse on training data (more bias) but much more stable on new data (less variance).

---

## 2. Ridge regression (L2 penalty)

$$\text{Loss}_{ridge} = SSR + \lambda\sum_j \beta_j^2$$

- λ (lambda) ≥ 0 sets how much we care about small coefficients.
- λ = 0 → plain least squares.
- Larger λ → coefficients shrink **toward** zero, but never exactly zero.
- The intercept is **not** penalized (it just sets the level).

### Toy example (my own)

Two training points, (1, 2) and (2, 5). Least squares gives slope 3 through both points (SSR = 0).
With ridge and λ = 1, a gentler slope (say 2.4) has SSR > 0 but a penalty of 2.4² = 5.76 instead of 3² = 9. If the true trend is gentler, the ridge line predicts new points better.

> 💡 Ridge also makes it possible to fit models with **more features than rows** and stabilizes coefficients when features are correlated.

---

## 3. Lasso regression (L1 penalty)

$$\text{Loss}_{lasso} = SSR + \lambda\sum_j |\beta_j|$$

- Same idea, but the penalty uses absolute values.
- Key difference: lasso can shrink coefficients **all the way to zero** → it **removes** useless features (automatic feature selection).

| | Ridge (L2) | Lasso (L1) |
|---|---|---|
| Penalty | Σβ² | Σ\|β\| |
| Coefficients reach exactly 0? | No | **Yes** |
| Best when… | Most features are somewhat useful; correlated features | Many features are useless; you want a simpler model |

**Elastic Net** mixes both penalties: λ₁Σ|β| + λ₂Σβ².

---

## 4. Choosing λ

There's no formula: try a range of λ values with **cross validation** and keep the one with the lowest average validation error (see [Cross Validation](Cross%20Validation.md)).

```
validation error
   │\                         /
   │ \  overfit         underfit
   │  \___           ___/
   │      \___★___/            ★ best λ
   └──────────────────────────► λ  (0 = plain least squares … large = flat line)
```

---

## 5. Regularization beyond linear regression

The same "loss + penalty" idea is everywhere:
- **Logistic regression** with L1/L2 (course 2.3).
- **XGBoost / LightGBM** penalties on leaf weights (course 1.6).
- **Neural networks** with weight decay (L2).

> ⚠️ Always **standardize** features first: the penalty compares coefficient sizes, and a feature measured in grams would otherwise get a tiny coefficient and escape the penalty compared to the same feature in kilograms.

---

## 6. Pseudocode

```text
FUNCTION RidgeLoss(β0, β, X, y, λ):  RETURN Σ (y - β0 - X·β)² + λ · Σ β_j²
FUNCTION LassoLoss(β0, β, X, y, λ):  RETURN Σ (y - β0 - X·β)² + λ · Σ |β_j|

FUNCTION RidgeClosedForm(X_std, y, λ)            # X_std standardized, y centered
    RETURN (X_stdᵀ X_std + λ I)⁻¹ X_stdᵀ y

FUNCTION ChooseLambda(X, y, lambdas, k = 5)
    FOR λ IN lambdas:
        cv_error[λ] ← KFoldCV(X, y, () → Pipeline(Standardize, Ridge(λ)), k)
    RETURN λ with the lowest mean cv_error
```

---

## 7. Self-check

1. What happens to ridge coefficients as λ → ∞? → All shrink toward 0; the model approaches a flat line at the mean.
2. You have 200 features and suspect only 10 matter. Ridge or lasso? → **Lasso** (or Elastic Net).
3. Why isn't the intercept penalized? → It only shifts the level of predictions; shrinking it would bias every prediction.
4. How do you pick λ? → Cross validation over a grid of values.

---

🔗 Related: [1.4 course notes](README.md) · [Cross Validation](Cross%20Validation.md) · [Gradient Descent](Gradient%20Descent.md)
