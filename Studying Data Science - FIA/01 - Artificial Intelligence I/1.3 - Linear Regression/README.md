# 1.3 — Linear Regression (Simple and Multiple)

> **Module:** Artificial Intelligence I
> **Case studies:** (1) insurance brokers' revenue vs. experience; (2) credit-card limit for new customers; (3) starting salaries of data analysts (qualitative variables); (4) discount coupons vs. incremental sales (non-linear transformations).

Linear regression is the foundation of supervised learning for a **quantitative target**. It's also the model whose ideas (fit by minimizing loss, coefficients, residuals, overfitting, inference) reappear in almost every other algorithm.

---

## Table of Contents

1. [Linear relationship and the line equation](#1-linear-relationship-and-the-line-equation)
2. [Pearson correlation](#2-pearson-correlation)
3. [Simple linear regression](#3-simple-linear-regression)
4. [Estimation: Ordinary Least Squares (OLS)](#4-estimation-ordinary-least-squares-ols)
5. [Inference on the parameters](#5-inference-on-the-parameters)
6. [Diagnostics: fit quality, residuals, influential points](#6-diagnostics-fit-quality-residuals-influential-points)
7. [Prediction](#7-prediction)
8. [Multiple linear regression](#8-multiple-linear-regression)
9. [Variable selection: stepwise backward](#9-variable-selection-stepwise-backward)
10. [Collinearity and VIF](#10-collinearity-and-vif)
11. [Qualitative variables: dummies / one-hot encoding](#11-qualitative-variables-dummies--one-hot-encoding)
12. [Non-linear relationships: Spearman and transformations](#12-non-linear-relationships-spearman-and-transformations)
13. [End-to-end workflow](#13-end-to-end-workflow)
14. [Pseudocode](#14-pseudocode)
15. [Interview questions](#15-interview-questions)
16. [References](#16-references)

---

## 1. Linear relationship and the line equation

Two quantitative variables have a **linear relationship** when, as one increases, the other increases or decreases at a **constant rate**. On a scatter plot, the points look like a straight line (with noise around it). Classic example: height vs. weight.

**Line equation:**

$$y = b_0 + b_1 x$$

- **b₀ — intercept:** value of *y* when *x* = 0.
- **b₁ — slope:** change in *y* when *x* increases by **1 unit**.

Example: with b₀ = −100 and b₁ = 1.1, a person 174 cm tall has predicted weight −100 + 1.1·174 = **91.4 kg**. Everyone *on* the line follows a rule — a simplification of the general behavior (real points vary around it).

Business examples of (roughly) linear relationships: revenue vs. active customers; average handling time vs. tickets received; sales vs. ad spend (within limits); units produced vs. total production cost.

---

## 2. Pearson correlation

The **Pearson correlation coefficient (r)** measures the strength of the **linear** relationship between two quantitative variables in a sample:

$$r = \frac{\sum_i (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_i (x_i - \bar{x})^2}\,\sqrt{\sum_i (y_i - \bar{y})^2}}$$

- r ∈ [−1, 1]
- r → 1: strong positive linear relationship
- r → −1: strong negative linear relationship
- r → 0: weak / no **linear** relationship (there may still be a non-linear one!)

Suggested interpretation (subjective, context-dependent):

| \|r\| | Linear relationship |
|---|---|
| ≥ 0.9 | Very strong |
| 0.7 – 0.9 | Strong |
| 0.4 – 0.7 | Moderate |
| 0.2 – 0.4 | Weak |
| < 0.2 | Very weak |

Case: experience × monthly revenue → **r = 0.725 (strong)**.

A **pairplot / correlogram** (`sns.pairplot`) shows scatter plots for every pair of variables; a **correlation matrix heatmap** consolidates all pairwise r values.

> ⚠️ Correlation ≠ causation. And r only captures *linear* association.

---

## 3. Simple linear regression

Explains a quantitative target with **one** explanatory variable.

**Theoretical (population) model:**

$$Y = \beta_0 + \beta_1 X + \varepsilon$$

- *Y*: quantitative response
- *X*: explanatory variable (quantitative or qualitative)
- β₀, β₁: **fixed, unknown population parameters**
- ε: theoretical error

**Assumptions:**
1. Observations are **independent**.
2. Errors ε follow a **normal distribution** with **mean zero** and **constant variance** (homoscedasticity).

> 💡 **The normal distribution** appears "naturally" in many phenomena (height, weight, measurement errors, daily temperatures, industrial dimensions). It's defined by its mean μ and standard deviation σ. The Galton board is a classic illustration: many small random effects add up to a bell curve (Central Limit Theorem intuition).

Before modeling, EDA matters especially for:
- **Missing values:** OLS can't handle them (in Y or X). Remove or replace them (or bin the variable and add a "no information" category).
- **Outliers:** can have huge impact, especially in Y.

Case: of 32 brokers, 2 had missing revenue — they were no longer active, so they were removed.

---

## 4. Estimation: Ordinary Least Squares (OLS)

We use the sample to **estimate** β₀ and β₁ — find the line that best fits the data. **OLS** finds the values that **minimize the sum of squared vertical distances** between points and the line:

$$\min_{\beta_0,\beta_1} \sum_i \left(y_i - \beta_0 - \beta_1 x_i\right)^2$$

Closed-form solution for simple regression:

$$\hat\beta_1 = \frac{\sum (x_i-\bar x)(y_i-\bar y)}{\sum (x_i-\bar x)^2} = r\cdot\frac{s_y}{s_x} \qquad \hat\beta_0 = \bar y - \hat\beta_1 \bar x$$

**Fitted model:** $\hat Y = \hat\beta_0 + \hat\beta_1 X$ — hats (^) denote estimates and predicted values.

**Residual:** $e = Y - \hat Y$ (actual − predicted).

Properties:
- OLS residuals **always average zero**.
- OLS **does not work with missing values**.

Case: $\hat y = 110.53 + 9.79\,x$ (`sm.OLS(y, sm.add_constant(x)).fit()`)
- **β̂₀ = 110.53:** a broker with zero experience would bill ~R$ 110k/month on average (hypothetical extrapolation — no such broker in the sample).
- **β̂₁ = 9.79:** each extra year of experience adds ~R$ 9.8k/month on average.

---

## 5. Inference on the parameters

We don't actually care about β̂ — we care about the **true** β. Any estimate from a sample has a **margin of error**, which:
- ↓ decreases with larger **sample size**
- ↑ increases with greater **data variability**
- ↑ increases with higher desired **confidence**

(Example: a survey estimated 60% of US undergrads own an iPhone; the true population value was 53.8%. Samples carry error.)

### 5.1 Confidence intervals

$$CI(\beta_j;\gamma) = \hat\beta_j \pm ME(\hat\beta_j)$$

γ = confidence level (usually 90% or 95%; 99% for stricter problems). Interpretation: if we drew many samples of the same size, about γ% of the intervals would contain the true parameter.

Case: CI(β₁; 95%) = [6.20; 13.39]; CI(β₀; 95%) = [76.50; 144.56].

### 5.2 Hypothesis test on β₁

| | |
|---|---|
| **H₀: β₁ = 0** | Flat (horizontal) line → **no** linear relationship between X and Y. The model reduces to Y = β₀ + ε |
| **Hₐ: β₁ ≠ 0** | There is a trend → **there is** a linear relationship |

If the 95% CI for β₁ **does not contain zero**, reject H₀. Case: [6.20; 13.39] excludes 0 → there **is** a statistically significant linear relationship between experience and revenue.

The same test on β₀ (H₀: β₀ = 0, line passes through the origin) has less practical importance; if β₀ = 0 you could use the more parsimonious Y = β₁X + ε.

### 5.3 p-value

> **The p-value is the probability of observing an estimate as far from zero as β̂ (or farther), assuming H₀ is true.**

- Small p-value → reject H₀; large → don't reject.
- Threshold = **significance level α** (commonly 5%).
- Deciding with p < 0.05 is **equivalent** to checking whether the 95% CI excludes zero.

Case: p(β₁) ≈ 0.0000057 → under H₀, getting 9.79 or more extreme is almost impossible → conclude Hₐ.

### 5.4 Type I and Type II errors

| | **Reality: H₀ true** (no relationship) | **Reality: H₀ false** (relationship exists) |
|---|---|---|
| **Decide: reject H₀** | ❌ **Type I error** (false positive) | ✅ |
| **Decide: don't reject H₀** | ✅ | ❌ **Type II error** (false negative) |

When we reject H₀ with p < α, **α is the maximum probability of a Type I error**.

---

## 6. Diagnostics: fit quality, residuals, influential points

"Is the model good?" — three checks.

### 6.1 Fit quality

**Coefficient of determination R²** — % of the target's variability explained by the model, 0–100%:

$$R^2 = \frac{\sum_i (\hat y_i - \bar y)^2}{\sum_i (y_i - \bar y)^2} = 1 - \frac{\sum_i (y_i-\hat y_i)^2}{\sum_i (y_i - \bar y)^2}$$

- If X explains nothing, every ŷ = ȳ → R² = 0.
- If X explains perfectly, ŷ = y → R² = 1.
- In simple regression, **R² = r²**.

| R² | Explanatory power |
|---|---|
| ≥ 0.80 | Very good |
| 0.60 – 0.80 | Good |
| 0.40 – 0.60 | Moderate |
| 0.20 – 0.40 | Low |
| < 0.20 | Very low |

**Error metrics:**

| Metric | Formula | Notes |
|---|---|---|
| **MAE** | $\frac{1}{n}\sum \lvert e_i\rvert$ | Average error in the target's unit — easy for business |
| **MAPE** | $\frac{1}{n}\sum \lvert e_i / y_i\rvert$ | Average % error; **undefined when y can be 0** |
| **RMSE** | $\sqrt{\frac{1}{n}\sum e_i^2}$ | Penalizes large errors more; good for comparing models technically |

Case: R² = 0.526 (experience explains 52.6% of revenue variability); MAE = R$ 37,792; MAPE = 27.0%.

### 6.2 Residual properties

| Property | How to check | What you want |
|---|---|---|
| **Normality** | Histogram + **Q-Q plot** (sample percentiles of residuals vs. theoretical normal percentiles) | Points along the diagonal |
| **Homoscedasticity** (constant variance) | Scatter of residuals vs. predicted values | A random cloud — no funnel shape |
| **Independence** | Domain knowledge of the sample (no simple test); plot residuals vs. time for temporal data | No pattern; most common violation is **time series** |

### 6.3 Influential observations — Cook's distance

Measures how much **removing each observation changes the fitted model**. Ranges from 0 to +∞; high values deserve attention. A common rule of thumb for small datasets: **D > 4/n**.

An influential point isn't necessarily a univariate outlier in x or y — it may be a **bivariate** outlier relative to the point cloud.

Case: observations 28 and 29 exceeded 4/n. Refitting without them changed β̂₁ from 9.79 to 8.91 and R² from 0.526 to 0.521 — not a material improvement, so the original model was kept.

---

## 7. Prediction

For a new broker with 13 years: ŷ = 110.53 + 9.79·13 = **237.83 → ~R$ 238k/month**.

> ⚠️ **Don't extrapolate**: the prediction is only valid for new observations with a profile similar to the sample (same range of X and same population).

---

## 8. Multiple linear regression

Explains Y with **two or more** explanatory variables.

$$Y = \beta_0 + \beta_1 X_1 + \dots + \beta_k X_k + \varepsilon$$

Same assumptions (independence, ε ~ Normal(0, σ²)). Geometrically it is a **hyperplane** (with 2 features, a plane in 3-D).

**Interpretation:** βⱼ = change in Y when Xⱼ increases 1 unit, **holding all other variables fixed** (*ceteris paribus*).

**Fitted:** $\hat Y = \hat\beta_0 + \hat\beta_1 X_1 + \dots + \hat\beta_k X_k$ (OLS in matrix form: $\hat\beta = (X^\top X)^{-1}X^\top y$).

### Reading a regression output (statsmodels `summary()`)

| Column | Meaning |
|---|---|
| `coef` | Estimated β̂ⱼ |
| `std err` | Standard error of β̂ⱼ — how volatile it would be across samples |
| `t` | coef / std err — a standardized coefficient |
| `P>\|t\|` | p-value for H₀: βⱼ = 0 |
| `[0.025 0.975]` | 95% confidence interval |
| `Adj. R-squared` | Adjusted R² |

### Adjusted R²

Plain R² **never decreases** when you add variables, even useless ones. **Adjusted R²** penalizes extra variables and can drop if a variable hurts:

$$R^2_{adj} = 1 - (1 - R^2)\frac{n-1}{n-k-1}$$

(n = observations, k = explanatory variables.)

### Case: credit-card limit
2,500 historical customers whose limit was deemed appropriate (reasonable utilization, no default in 12 months). Candidate features: credit score, total overdue amount (12m), # credit inquiries (12m), average income (12m), employment time, age, # dependents.

---

## 9. Variable selection: stepwise backward

In multiple regression some variables may be significant and others not. **Stepwise backward:**

1. Fit with all variables.
2. Remove the variable with the **highest p-value** above α.
3. **Refit** and repeat until all remaining are significant.

> 💡 **Remove one variable at a time** — removing one can change the significance of the others.

Case: Model 1 → drop **age** (highest p-value) → Model 2 → drop **# dependents** → Model 3 (all significant). Adjusted R² barely moved: 63.3% → 63.2%.

Model 3 coefficients (each "holding the others fixed"):
- +1 credit-score point → **+R$ 1.41** limit (r = 0.592 bivariate ✓)
- +R$ 1 overdue → **−R$ 0.07** (r = −0.268 ✓)
- +1 credit inquiry → **−R$ 54.28** (r = −0.422 ✓)
- +R$ 1 income → **+R$ 0.10** (r = 0.636 ✓)
- +1 year employment → **+R$ 9.57** (r = 0.154 ✓)

Always check that signs are **coherent with the business and with the bivariate analysis**.

**Variable importance:** lower p-value = more important; for "visual ties", larger |t| = more important.

---

## 10. Collinearity and VIF

**Collinearity (multicollinearity):** strong linear correlation between explanatory variables. It distorts estimation because it's hard to isolate each variable's individual effect → unstable coefficients, inflated standard errors, flipped signs.

Checking pairwise correlations is a start, but it doesn't scale and doesn't work for qualitative variables. Use the **Variance Inflation Factor**:

$$VIF(X_i) = \frac{1}{1 - R_i^2}$$

where $R_i^2$ is the R² of a regression of **Xᵢ on all other explanatory variables** (excluding Y). It answers: *"How well can I model this feature from the others?"*

| VIF | Collinearity |
|---|---|
| = 1 | Absent |
| 1 – 2 | Weak |
| 2 – 5 | Moderate |
| 5 – 10 | High |
| ≥ 10 | Very high |

> High VIF doesn't automatically mean distortion — refit without the variable and compare.

Case: credit score and # inquiries were collinear (VIF 1.75). Model 4 dropped # inquiries → score's VIF fell to 1.15, its coefficient rose 1.41 → 1.68, adjusted R² 63.2% → 62.5% (acceptable). Model 4: R² = 0.625, MAE = R$ 410.28, MAPE = 11.0%.

Model 5 removed influential observations (Cook): adjusted R² 0.625 → 0.655, MAE/MAPE unchanged → adopted for deployment:

$$\hat y = 2091.78 + 1.6251\,x_{score} - 0.0680\,x_{overdue} + 0.1039\,x_{income} + 8.7853\,x_{employment}$$

---

## 11. Qualitative variables: dummies / one-hot encoding

Regression needs numbers. Convert categorical variables to **dummies** using **one-hot encoding**: one binary (0/1) column per category.

```
GENDER = female → GENDER_female=1, GENDER_male=0, GENDER_other=0
```

> ⚠️ **Dummy-variable trap:** drop **one** dummy per variable (the *reference category*). The full set is perfectly collinear (they always sum to 1, same as the intercept). The dropped category's effect is absorbed by the **intercept**, and each remaining dummy's coefficient is the **difference vs. the reference**.

**Geometry:** with a quantitative X₁ and a dummy X₂:
- X₂ = 0 → Y = β₀ + β₁X₁
- X₂ = 1 → Y = (β₀ + β₂) + β₁X₁

Two **parallel lines** (same slope, different intercepts) — a "plane with a jump".

Case — tech salaries (200 new data analysts; features: level, education, English, gender, prior experience, salary expectation):
- Mid-level ("pleno") earns **R$ 2,408.85 more** than junior, holding others fixed.
- Male earns R$ 228.85 more than female, but **not statistically significant** at 5% — could be sample coincidence, not a population pattern.

Boxplots are the go-to bivariate chart when an explanatory variable is qualitative.

---

## 12. Non-linear relationships: Spearman and transformations

### Spearman correlation

Measures **monotonic** (not necessarily linear) association. It is **Pearson applied to the ranks** of the variables; works for quantitative or ordinal variables.

If **Spearman ≫ Pearson**, the relationship is monotonic but **non-linear**.

Case — coupons: % discount × incremental units: Pearson 0.884, Spearman 0.975 → non-linear (looks exponential, y ~ eˣ).

### Options for non-linearity
1. **Transform** variables to linearize the relationship.
2. Use **more flexible models** (e.g., trees, ML).

Common transformations:
- Transform **X** when dispersion is constant: log(x), √x, x², 1/x…
- Transform **Y** when dispersion **grows** with the level: log(y), √y.

Case: replacing y with **log(y)** raised Pearson from 0.884 to **0.981** → now a simple linear regression works. (Remember to back-transform predictions: ŷ = exp(prediction).)

---

## 13. End-to-end workflow

```
DEFINITION & COLLECTION
 1. Problem definition    → translate business question: explain relationships / predict a quantitative variable
 2. Variable definition   → quantitative target Y; candidate features X1..Xk
 3. Data collection       → representative sample of the population of interest
DATA PREPARATION
 4. EDA                   → treat missing values and outliers (regression can't handle them)
 5. Bivariate analysis    → scatter plots, boxplots, Pearson & Spearman
 6. Variable preparation  → transform non-linear quantitatives; one-hot encode qualitatives
ESTIMATION & SELECTION
 7. Estimation            → OLS with all variables
 8. Variable selection    → p-values; stepwise backward, one at a time
 9. Collinearity          → VIF and correlations; remove redundant variables; refit
DIAGNOSTICS & USE
10. Diagnostics           → R², MAE, MAPE, RMSE; residual properties; Cook's distance
11. Interpretation        → coefficients in business terms; importance ranking (p-value / |t|)
12. Prediction            → estimate Y for new observations, never extrapolating beyond the sample profile
(iterative: diagnostics can send you back to estimation)
```

---

## 14. Pseudocode

### 14.1 OLS for simple linear regression (from scratch)

```text
FUNCTION FitSimpleOLS(x[1..n], y[1..n])
    x_bar ← mean(x);  y_bar ← mean(y)
    Sxy ← Σ (x[i] - x_bar) * (y[i] - y_bar)
    Sxx ← Σ (x[i] - x_bar)²
    b1  ← Sxy / Sxx
    b0  ← y_bar - b1 * x_bar
    RETURN (b0, b1)

FUNCTION Predict(b0, b1, x_new)
    RETURN b0 + b1 * x_new
```

### 14.2 OLS for multiple regression + inference

```text
FUNCTION FitOLS(X[n×k], y[n])
    X ← [1 | X]                                 # add intercept column
    beta_hat ← (Xᵀ X)⁻¹ Xᵀ y                      # normal equations
    residuals ← y - X · beta_hat
    sigma² ← Σ residuals² / (n - k - 1)
    se(beta_hat) ← sqrt( diag( sigma² · (Xᵀ X)⁻¹ ) )
    t ← beta_hat / se
    p_value ← 2 · P(T_{n-k-1} > |t|)             # Student-t
    CI_95 ← beta_hat ± t_{0.975, n-k-1} · se
    R² ← 1 - Σ residuals² / Σ (y - mean(y))²
    R²_adj ← 1 - (1 - R²) · (n - 1) / (n - k - 1)
    RETURN beta_hat, se, t, p_value, CI_95, R², R²_adj
```

### 14.3 Stepwise backward selection

```text
FUNCTION StepwiseBackward(X, y, alpha = 0.05)
    features ← all columns of X
    LOOP
        model ← FitOLS(X[features], y)
        worst ← feature (excluding intercept) with the largest p_value
        IF p_value[worst] > alpha:
            REMOVE worst FROM features            # ONE at a time
        ELSE:
            BREAK
    RETURN model, features
```

### 14.4 VIF

```text
FUNCTION VIF(X)
    FOR each feature j in X:
        R²_j ← R² of FitOLS(X without j, X[j])
        vif[j] ← 1 / (1 - R²_j)
    RETURN vif
```

### 14.5 Cook's distance (influential points)

```text
FUNCTION CooksDistance(X, y)
    full ← FitOLS(X, y);  p ← number of parameters;  s² ← MSE(full)
    FOR each observation i:
        model_i ← FitOLS(X without row i, y without row i)
        D[i] ← Σ_j ( ŷ_full[j] - ŷ_model_i[j] )² / (p · s²)
    flagged ← { i : D[i] > 4 / n }
    RETURN D, flagged
```

### 14.6 Full workflow

```text
PROCEDURE LinearRegressionProject(data, target)
    data ← remove or impute missing values; treat outliers
    FOR each feature: check scatter/boxplot vs target; compute Pearson & Spearman
        IF Spearman >> Pearson: try transformations (log, sqrt, square) on X or Y
    data ← one_hot_encode(qualitative features, drop_first = TRUE)
    model, feats ← StepwiseBackward(data[features], data[target])
    WHILE max(VIF(data[feats])) high:
        drop the redundant feature; refit; compare R²_adj and coefficients
    CHECK residuals: Q-Q plot (normality), residual-vs-fitted (homoscedasticity), independence
    D ← CooksDistance; refit without flagged points; keep the better model
    REPORT R², R²_adj, MAE, MAPE, RMSE; interpret coefficients ceteris paribus
    USE model.predict(new_data) only within the training profile
```

Python reference:

```python
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

X = pd.get_dummies(df[features], drop_first=True).astype(float)
X = sm.add_constant(X)
model = sm.OLS(df["y"], X).fit()
print(model.summary())                      # coef, std err, t, p, CI, R², adj R²
vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
cooks = model.get_influence().cooks_distance[0]
sm.qqplot(model.resid, line="45", fit=True)
```

---

## 15. Interview questions

1. **What does OLS minimize?** The sum of squared residuals.
2. **State the assumptions of linear regression.** Linearity, independence of errors, homoscedasticity, normality of errors (for inference), no perfect multicollinearity.
3. **Interpret a coefficient in multiple regression.** Expected change in Y for a one-unit increase in Xⱼ holding all other features constant.
4. **What is a p-value?** P(estimate at least this extreme | H₀ true). It is *not* the probability that H₀ is true.
5. **Type I vs. Type II error?** False positive (reject a true H₀) vs. false negative (fail to reject a false H₀). α controls Type I.
6. **Why use adjusted R²?** R² never decreases when adding variables; adjusted R² penalizes complexity.
7. **What is multicollinearity and how do you detect it?** Correlated features → unstable coefficients. Detect with correlation matrix and VIF (>5 or >10 is concerning). Fix by dropping/combining features or using regularization (Ridge — see 1.4).
8. **What is the dummy-variable trap?** Including all k dummies plus an intercept → perfect collinearity. Drop one (reference category).
9. **MAE vs. RMSE?** RMSE penalizes large errors more; MAE is more robust and easier to explain.
10. **When is MAPE a bad metric?** When the target can be zero or near zero (division blows up); it also penalizes over-forecasts and under-forecasts asymmetrically.
11. **Pearson vs. Spearman?** Linear vs. monotonic (rank-based). Spearman ≫ Pearson suggests a non-linear monotonic relation.
12. **How do you handle heteroscedasticity?** Transform Y (log), use weighted least squares, or robust (HC) standard errors.
13. **What is Cook's distance?** How much the fitted values change when an observation is removed; flags influential points (rule of thumb > 4/n).
14. **Why remove variables one at a time in backward selection?** Removing one changes the others' significance.
15. **R² = 0.3 — is the model useless?** Not necessarily: for diagnostic purposes with significant coefficients it can still be informative; depends on domain (social/behavioral data is noisy).

---

## 16. References

- Morettin, P. A.; Singer, J. M. *Estatística e Ciência de Dados*. 2nd ed. LTC, 2022.
- Anderson, D. R.; Sweeney, D. J.; Williams, T. A. *Statistics for Business and Economics*. Cengage.
- Hastie, T.; Tibshirani, R.; Friedman, J. *The Elements of Statistical Learning*. 2nd ed. Springer, 2009. <https://hastie.su.domains/ElemStatLearn>
- Kutner, M. H. et al. *Applied Linear Regression Models*. 4th ed. McGraw-Hill Irwin, 2004.
- Rencher, A. C.; Schaalje, G. B. *Linear Models in Statistics*. 2nd ed. Wiley, 2008.
- Montgomery, D. C. *Introduction to Linear Regression Analysis*. 5th ed. Wiley, 2012.

---

⬅️ Previous: [1.2 — Introduction to Modeling](../1.2%20-%20Introduction%20to%20Modeling/README.md) · 🏠 [Index](../../README.md) · ➡️ Next: [1.4 — Validation and Regularization](../1.4%20-%20Validation%20and%20Regularization/README.md)
