# Fundamental Concepts in Statistics

> 📗 **Companion notes** for *The StatQuest Illustrated Guide to Machine Learning* (Josh Starmer), **Chapter 3** and the probability / statistics parts of the **Appendices**.
> Written in my own words with my own examples. Course notes: [1.1 — EDA](README.md).

---

## 1. Histograms: seeing where the data piles up

A histogram splits the range of a variable into **bins** and counts how many observations fall in each. Tall bars = values that are common; short bars = rare values.

- **Too many bins** → noisy, every bar is tiny.
- **Too few bins** → the shape disappears.
- Try a few bin widths before drawing conclusions.

> 💡 A histogram is already a crude **model of probability**: if 30 of 200 customers spend between R$ 100 and R$ 150, the chance that the next customer lands there is roughly 30/200 = 15%.

---

## 2. Probability distributions

A **probability distribution** is a smooth, idealized version of the histogram: it tells us how likely each value (or range of values) is, without needing the raw data.

### 2.1 Discrete distributions (countable outcomes)

| Distribution | Use when… | My example |
|---|---|---|
| **Binomial** | Counting successes in *n* independent yes/no trials with the same success probability *p* | Out of 10 emails sent, how many get opened if each opens with p = 0.3? |
| **Poisson** | Counting events in a fixed interval when they happen at an average rate λ | Support tickets per hour, averaging 4 |

Binomial probability of exactly *k* successes:

$$P(k) = \binom{n}{k} p^k (1-p)^{n-k}$$

Example: P(exactly 3 opens out of 10, p = 0.3) = 120 × 0.3³ × 0.7⁷ ≈ **0.267**.

Poisson probability of exactly *k* events:

$$P(k) = \frac{e^{-\lambda}\lambda^k}{k!}$$

Example: P(exactly 2 tickets in an hour, λ = 4) = e⁻⁴ × 16 / 2 ≈ **0.147**.

### 2.2 Continuous distributions (measured outcomes)

For continuous data, the probability of one *exact* value is 0 (what's the chance of a delivery taking exactly 31.000000… minutes?). Instead, probability is the **area under the curve** between two values.

| Distribution | Shape | Use when… |
|---|---|---|
| **Normal (Gaussian)** | Symmetric bell, defined by mean μ and standard deviation σ | Heights, measurement errors, many sums of small effects |
| **Exponential** | Starts high, decays | Waiting time until the next event |
| **Uniform** | Flat | Every value in a range equally likely (random number generators) |

Normal-distribution rule of thumb: about **68%** of values within μ ± 1σ, **95%** within μ ± 2σ, **99.7%** within μ ± 3σ.

> ⚠️ The height of a continuous curve (the **density**) is *not* a probability; it can even be greater than 1. Only areas are probabilities.

---

## 3. Probability basics (Appendix-style refresher)

- **Independent events:** one doesn't change the other's probability → multiply: P(A and B) = P(A) × P(B).
  Example: 60% of users prefer dark mode. Two random users both prefer it: 0.6 × 0.6 = **0.36**.
- **Order matters for a specific sequence:** first two prefer dark mode, third prefers light: 0.6 × 0.6 × 0.4 = **0.144**.
- **Mean / expected value:** the long-run average. For a discrete variable, E[X] = Σ x · P(x).
- **Variance & standard deviation:** average squared distance from the mean, and its square root (same units as the data).

> 💡 **Population vs. estimate:** with data we only *estimate* μ and σ. The sample variance divides by **n − 1** instead of n, because the sample mean is "pulled" toward the data and makes raw deviations look too small.

---

## 4. Models, residuals and sums of squares

A **model** is an approximation of a relationship, e.g., a line predicting delivery time from distance. No model is perfect, so we measure how far it misses:

- **Residual** = observed − predicted.
- **Sum of squared residuals (SSR)** = Σ residual². Squaring keeps errors positive and punishes big misses more.
- **Mean squared error (MSE)** = SSR / n. Unlike SSR, it doesn't grow just because you have more data, so you can compare datasets of different sizes.

My example — 4 deliveries, model predicts 20, 25, 30, 35 minutes; actual 22, 24, 33, 35:

| Residual | 2 | −1 | 3 | 0 |
|---|---|---|---|---|
| Squared | 4 | 1 | 9 | 0 |

SSR = 14, MSE = 14 / 4 = **3.5**.

---

## 5. R²: how much better than "just guess the mean"?

$$R^2 = \frac{SSR(\text{mean}) - SSR(\text{model})}{SSR(\text{mean})} = 1 - \frac{SSR(\text{model})}{SSR(\text{mean})}$$

- The baseline is a flat line at the mean of Y.
- R² = 0.7 → the model removes **70%** of the squared error the mean would make.
- R² = 0 → no better than the mean. (On new data, it can even be negative: worse than the mean.)

> 💡 R² is the same quantity as in [1.3 — Linear Regression](../1.3%20-%20Linear%20Regression/README.md), here explained as a "percentage of error removed".

---

## 6. p-values: is the difference real or luck?

A **p-value** answers: *if there were really no difference (the null hypothesis), how surprising would results like mine be?*

- Small p-value (commonly < 0.05) → results this extreme would be rare under "no difference" → we **reject** the null.
- Large p-value → the data are compatible with "no difference" → we **don't reject** it (that's not the same as proving there's no difference).

Example: a new checkout page converts 5.6% vs. 5.0% for the old one, p = 0.30. With this sample, a 0.6-point gap is easily produced by chance → **not enough evidence** that the new page is better.

> ⚠️ Common traps:
> - The p-value is **not** the probability that the null hypothesis is true.
> - A tiny p-value doesn't mean a **large** effect — with huge samples, trivial differences become "significant".
> - Running many tests inflates **false positives** (at α = 0.05, about 1 in 20 tests on pure noise will look significant).

---

## 7. Pseudocode

```text
FUNCTION Histogram(x, n_bins)
    edges ← n_bins + 1 equally spaced values from min(x) to max(x)
    counts[b] ← number of x in [edges[b], edges[b+1])
    RETURN counts / length(x)                         # ≈ probability per bin

FUNCTION SSR(y, ŷ):  RETURN Σ (y[i] - ŷ[i])²
FUNCTION MSE(y, ŷ):  RETURN SSR(y, ŷ) / length(y)

FUNCTION RSquared(y, ŷ)
    baseline ← SSR(y, [mean(y)] * length(y))
    RETURN (baseline - SSR(y, ŷ)) / baseline

FUNCTION PermutationPValue(group_a, group_b, n_shuffles = 10000)
    observed ← |mean(group_a) - mean(group_b)|
    pooled ← group_a ∪ group_b;  count ← 0
    REPEAT n_shuffles times:
        SHUFFLE pooled
        a* ← first |group_a| values;  b* ← the rest
        IF |mean(a*) - mean(b*)| ≥ observed: count ← count + 1
    RETURN count / n_shuffles        # share of "no difference" worlds at least as extreme
```

---

## 8. Self-check

1. Why is P(delivery takes exactly 30 minutes) = 0 for a continuous variable? → Probability is an area; a single point has zero width.
2. Which distribution for "number of app crashes per day"? → **Poisson**.
3. Model SSR = 40, mean-only SSR = 160. R²? → (160 − 40) / 160 = **0.75**.
4. p = 0.03 at α = 0.05. Does that prove the effect is large? → No — it says the effect is unlikely to be pure chance, not how big it is.

---

🔗 Related: [1.1 course notes](README.md) · [Linear Regression](../1.3%20-%20Linear%20Regression/Linear%20Regression.md)
