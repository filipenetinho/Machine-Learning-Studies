# Logistic Regression

> 📗 **Companion notes** for *The StatQuest Illustrated Guide to Machine Learning* (Josh Starmer), **Chapter 6**.
> Written in my own words with my own examples. Course notes: [2.3 — Logistic Regression](README.md).

---

## 1. From a line to an S-curve

For a yes/no target (churned or not, clicked or not), a straight line is a poor fit: it predicts values below 0 and above 1. Logistic regression fits an **S-shaped curve** whose output is always between 0 and 1 and reads as a **probability**.

$$p(x) = \frac{1}{1 + e^{-(b_0 + b_1x)}}$$

Classify by comparing p with a threshold (0.5 by default): p ≥ 0.5 → "yes".

---

## 2. Probability vs. likelihood

These two words get mixed up all the time:

| | Probability | Likelihood |
|---|---|---|
| Fixed | The model (curve) | The **data** we observed |
| Varies | The possible outcomes | The model (curve) |
| Question | "Given this curve, how likely is a churn at x = 10?" | "Given these observed churns, how plausible is this curve?" |

**Fitting = trying many curves and keeping the one with the highest likelihood of the observed data.** This is **maximum likelihood**.

---

## 3. Computing the likelihood (my own example)

Candidate curve predicts these churn probabilities for 4 customers:

| Customer | Churned? | p(churn) from curve | Probability of what actually happened |
|---|---|---|---|
| A | yes | 0.9 | 0.9 |
| B | yes | 0.6 | 0.6 |
| C | no | 0.2 | 1 − 0.2 = 0.8 |
| D | no | 0.4 | 1 − 0.4 = 0.6 |

Likelihood = 0.9 × 0.6 × 0.8 × 0.6 = **0.259**

A better curve would push A and B closer to 1 and C and D closer to 0, raising this product.

> 💡 **Use the log-likelihood.** Multiplying hundreds of probabilities produces numbers so tiny that computers round them to 0 (*underflow*). Logs turn the product into a sum:
> log(0.9) + log(0.6) + log(0.8) + log(0.6) ≈ −0.105 − 0.511 − 0.223 − 0.511 = **−1.35**.
> Maximizing the log-likelihood picks the same curve. Its negative is the **log-loss** used by most software.

There's no closed-form solution, so the curve is found **iteratively** (gradient descent or Newton's method — see [Gradient Descent](../../01%20-%20Artificial%20Intelligence%20I/1.4%20-%20Validation%20and%20Regularization/Gradient%20Descent.md)).

---

## 4. Reading the coefficients

On the **log-odds** scale the model is a straight line:

$$\log\frac{p}{1-p} = b_0 + b_1x$$

- b₁ > 0 → probability increases with x.
- e^{b₁} = **odds ratio**: each unit of x multiplies the odds by that factor.

---

## 5. Features of any kind

Like linear regression, logistic regression accepts continuous and categorical features (as dummies), and many at once. Each coefficient's significance can be tested (Wald test) to see whether it helps predict the outcome.

---

## 6. Limitations

- The decision boundary is **linear** in the features (unless you add transformed features).
- Assumes the log-odds change linearly with each feature.
- Perfectly separable data can make coefficients blow up → use regularization.

---

## 7. Pseudocode

```text
FUNCTION Sigmoid(z):  RETURN 1 / (1 + exp(-z))

FUNCTION LogLikelihood(b, X, y)
    p ← Sigmoid(b0 + X·b)
    RETURN Σ [ y·log(p) + (1 - y)·log(1 - p) ]

FUNCTION FitLogistic(X, y, η, steps)
    b ← zeros
    REPEAT steps times:
        p ← Sigmoid(b0 + X·b)
        gradient ← Xᵀ (y - p)                 # derivative of the log-likelihood
        b ← b + η · gradient                  # ASCENT: we maximize the likelihood
    RETURN b

FUNCTION Classify(b, x, threshold = 0.5):  RETURN Sigmoid(b0 + x·b) ≥ threshold
```

---

## 8. Self-check

1. Why not use linear regression for a yes/no target? → It can predict impossible values (< 0 or > 1).
2. Probability vs. likelihood in one line? → Probability: fixed model, varying outcomes. Likelihood: fixed data, varying model.
3. Why maximize the log-likelihood instead of the likelihood? → Same optimum, but sums avoid numerical underflow.
4. b₁ = 0.7. Odds ratio? → e^0.7 ≈ **2.01**: each unit roughly doubles the odds.

---

🔗 Related: [2.3 course notes](README.md) · [Assessing Model Performance](Assessing%20Model%20Performance.md) · [Naive Bayes](Naive%20Bayes.md)
