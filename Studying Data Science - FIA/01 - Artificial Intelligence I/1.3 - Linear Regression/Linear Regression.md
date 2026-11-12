# Linear Regression

> 📗 **Companion notes** for *The StatQuest Illustrated Guide to Machine Learning* (Josh Starmer), **Chapter 4**.
> Written in my own words with my own examples. Course notes: [1.3 — Linear Regression](README.md).

---

## 1. The idea

Linear regression draws the **straight line that best summarizes the trend** between a feature and a numeric target, then uses it to predict.

"Best" means the line with the **smallest sum of squared residuals (SSR)**: the total squared vertical distance between each point and the line. This is called **least squares**.

$$\hat y = b_0 + b_1 x \qquad SSR(b_0, b_1) = \sum_i \big(y_i - (b_0 + b_1 x_i)\big)^2$$

---

## 2. How the best line is found

Think of SSR as a landscape: every possible (intercept, slope) pair has an SSR value. We want the lowest point.

**Two ways to get there:**

| Approach | How | When |
|---|---|---|
| **Analytical** | Set the derivatives of SSR to zero and solve → closed-form formulas | Linear regression (small/medium data) |
| **Iterative** | Start somewhere and step downhill (**gradient descent**) | When there's no formula, or data is huge (see [Gradient Descent](../1.4%20-%20Validation%20and%20Regularization/Gradient%20Descent.md)) |

Closed-form solution for one feature:

$$b_1 = \frac{\sum (x_i - \bar x)(y_i - \bar y)}{\sum (x_i - \bar x)^2}, \qquad b_0 = \bar y - b_1\bar x$$

### Worked example (my own)

Ad spend (R$ thousand) vs. sales (units):

| x | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| y | 3 | 5 | 6 | 8 |

x̄ = 2.5, ȳ = 5.5
Σ(x − x̄)(y − ȳ) = (−1.5)(−2.5) + (−0.5)(−0.5) + (0.5)(0.5) + (1.5)(2.5) = 3.75 + 0.25 + 0.25 + 3.75 = **8**
Σ(x − x̄)² = 2.25 + 0.25 + 0.25 + 2.25 = **5**
b₁ = 8/5 = **1.6**, b₀ = 5.5 − 1.6 × 2.5 = **1.5** → **ŷ = 1.5 + 1.6x**

Each extra R$ 1k of ads ≈ **+1.6 units**.

---

## 3. Is the line any good? R² and the p-value

1. **R²** compares the line with the flat "mean of y" line: how much of the squared error does the slope remove?
   In the example: SSR(mean) = 6.25 + 0.25 + 0.25 + 6.25 = 13; predictions are 3.1, 4.7, 6.3, 7.9 → residuals −0.1, 0.3, −0.3, 0.1 → SSR(line) = 0.01 + 0.09 + 0.09 + 0.01 = 0.2 → R² = (13 − 0.2)/13 ≈ **0.98**.
2. **p-value for R² (F-test):** with very few points, a high R² can happen by chance. The p-value says how often random, unrelated data would give an R² this large.

> 💡 Two questions, two answers: **R²** = *how strong* is the relationship? **p-value** = *how confident* are we that it isn't luck? You want a large R² **and** a small p-value.

The F statistic behind the p-value compares the explained variation per parameter with the unexplained variation per remaining degree of freedom:

$$F = \frac{(SSR_{mean} - SSR_{line}) / (p_{line} - p_{mean})}{SSR_{line} / (n - p_{line})}$$

---

## 4. Multiple linear regression

Same idea with more features: instead of a line, a **plane** (2 features) or hyperplane (more):

$$\hat y = b_0 + b_1x_1 + b_2x_2 + \dots$$

- Still fit by minimizing SSR.
- Each coefficient is the change in ŷ per unit of its feature **with the others held fixed**.
- Adding a feature can never increase training SSR, so plain R² always goes up → use **adjusted R²** or the **p-value comparing the simple and the bigger model** to decide whether the extra feature earns its place.

---

## 5. Things to watch

- **Linear in the coefficients**, not necessarily in x: ŷ = b₀ + b₁x + b₂x² is still "linear regression".
- **Outliers** pull the line strongly because errors are squared.
- **Extrapolation** beyond the range of x in the data is risky.
- **Correlation isn't causation**: the slope describes association.

---

## 6. Pseudocode

```text
FUNCTION FitLine(x, y)
    x̄ ← mean(x);  ȳ ← mean(y)
    b1 ← Σ (x - x̄)(y - ȳ) / Σ (x - x̄)²
    b0 ← ȳ - b1 · x̄
    RETURN b0, b1

FUNCTION Evaluate(x, y, b0, b1, n_params = 2)
    ŷ ← b0 + b1 · x
    ssr_line ← Σ (y - ŷ)²
    ssr_mean ← Σ (y - mean(y))²
    R² ← 1 - ssr_line / ssr_mean
    F  ← ((ssr_mean - ssr_line) / (n_params - 1)) / (ssr_line / (n - n_params))
    p  ← P(F_{n_params - 1, n - n_params} ≥ F)
    RETURN R², p
```

---

## 7. Self-check

1. What does least squares minimize? → The sum of squared vertical distances from the points to the line.
2. R² = 0.9 with p = 0.4 on 3 points. Trust it? → No — with so few points a high R² is easy to get by chance.
3. Adding a useless feature: what happens to R² and adjusted R²? → R² rises slightly; adjusted R² tends to fall.
4. Using the example line, predict sales for R$ 5k of ads. → 1.5 + 1.6 × 5 = **9.5 units** (an extrapolation, so be careful).

---

🔗 Related: [1.3 course notes](README.md) · [Gradient Descent](../1.4%20-%20Validation%20and%20Regularization/Gradient%20Descent.md) · [Regularization](../1.4%20-%20Validation%20and%20Regularization/Regularization.md)
