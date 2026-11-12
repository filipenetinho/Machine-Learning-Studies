# Gradient Descent

> 📗 **Companion notes** for *The StatQuest Illustrated Guide to Machine Learning* (Josh Starmer), **Chapter 5** (plus the derivative / chain-rule refreshers in the **Appendices**).
> Written in my own words with my own examples. Course notes: [1.4 — Validation and Regularization](README.md) (SGD section).

---

## 1. Why we need it

Linear regression has a formula for its best parameters. Most models don't (logistic regression, neural networks, boosting…). **Gradient descent** is a general recipe to find parameters that minimize a **loss function** when no formula exists.

Mental picture: you're on a foggy hill and want the lowest point. You can't see the valley, but you can feel the slope under your feet → take a step downhill, feel again, repeat.

---

## 2. Ingredients

| Ingredient | Meaning |
|---|---|
| **Loss function** | Measures how bad the current parameters are (e.g., SSR for regression, log-loss for classification) |
| **Derivative / gradient** | Slope of the loss with respect to each parameter: tells which direction is uphill and how steep |
| **Learning rate (η)** | Scales the slope into a step size |
| **Stopping rule** | Stop when steps get tiny or after a maximum number of steps |

Update rule for each parameter θ:

$$\text{step} = \eta \cdot \frac{\partial\,\text{Loss}}{\partial\theta}, \qquad \theta_{new} = \theta_{old} - \text{step}$$

- Steep slope → big step (we're far from the bottom).
- Flat slope → tiny step (we're close).
- Minus sign → move **against** the slope (downhill).

---

## 3. A tiny worked example (my own)

Fit only the intercept b of ŷ = b to the points y = {2, 4, 6} (the answer should be the mean, 4).

Loss: SSR(b) = (2 − b)² + (4 − b)² + (6 − b)²
Derivative (chain rule): dSSR/db = −2(2 − b) − 2(4 − b) − 2(6 − b) = 6b − 24

Learning rate η = 0.1, start at b = 0:

| Step | b | Slope 6b − 24 | Step size = 0.1 × slope | New b |
|---|---|---|---|---|
| 1 | 0.00 | −24.0 | −2.40 | 2.40 |
| 2 | 2.40 | −9.6 | −0.96 | 3.36 |
| 3 | 3.36 | −3.84 | −0.38 | 3.74 |
| 4 | 3.74 | −1.54 | −0.15 | 3.90 |
| … | → 4.00 | → 0 | → 0 | |

Steps shrink automatically as the slope flattens near the minimum.

> 💡 **Chain rule refresher:** for (y − b)², the outer function is u² (derivative 2u) and the inner is u = y − b (derivative −1) → 2(y − b)(−1) = −2(y − b).

---

## 4. The learning rate matters

| η | Behavior |
|---|---|
| Too small | Converges, but painfully slowly |
| Good | Steady, fast descent |
| Too large | Overshoots the valley, bounces around or even diverges |

In the example, η = 0.4 would give step 1 → b = 9.6, step 2 → b = −3.84… it explodes. Practical tools: learning-rate **schedules** (start big, shrink over time) or adaptive methods.

---

## 5. More than one parameter

With intercept and slope (or thousands of weights), compute the derivative with respect to **each** parameter. Together they form the **gradient**; update all parameters at the same time with the same rule.

---

## 6. Stochastic and mini-batch gradient descent

Computing the gradient on **all** rows each step is expensive for big data.

| Variant | Gradient uses… | Trade-off |
|---|---|---|
| **Batch** | All rows | Exact but slow per step |
| **Stochastic (SGD)** | One random row | Very fast, very noisy path |
| **Mini-batch** | A small random subset (e.g., 32–512 rows) | The usual compromise |

The noisy path still heads downhill on average and can even help escape shallow bumps. This is what `SGDRegressor` / `SGDClassifier` use in the course.

> ⚠️ Gradient descent is sensitive to **feature scale**: standardize features first, or one parameter's slope will dominate.

---

## 7. Pseudocode

```text
FUNCTION GradientDescent(loss_gradient, θ_init, η, max_steps = 1000, tol = 1e-6)
    θ ← θ_init
    FOR step IN 1..max_steps:
        g ← loss_gradient(θ)                 # vector of partial derivatives
        Δ ← η · g
        θ ← θ - Δ
        IF max(|Δ|) < tol: BREAK             # steps are tiny → we're at the bottom
    RETURN θ

FUNCTION MiniBatchSGD(X, y, θ_init, η, epochs, batch_size)
    θ ← θ_init
    FOR epoch IN 1..epochs:
        SHUFFLE rows
        FOR each batch B of batch_size rows:
            θ ← θ - η · gradient_of_loss_on(B, θ)
        η ← schedule(η, epoch)               # optional: shrink learning rate
    RETURN θ
```

---

## 8. Self-check

1. Why the minus sign in θ − η·slope? → To move opposite to the slope, i.e., downhill.
2. Why do steps get smaller near the minimum even with fixed η? → The slope itself approaches zero.
3. Loss jumps around and grows each step. First thing to try? → Lower the learning rate.
4. Why standardize features before gradient descent? → So all parameters' gradients are on comparable scales and one step size works for all.

---

🔗 Related: [Linear Regression](../1.3%20-%20Linear%20Regression/Linear%20Regression.md) · [Logistic Regression](../../02%20-%20Artificial%20Intelligence%20II/2.3%20-%20Classification%20-%20Logistic%20Regression/Logistic%20Regression.md) · [Neural Networks](../../03%20-%20Extra%20Topics/Neural%20Networks/Neural%20Networks.md)
