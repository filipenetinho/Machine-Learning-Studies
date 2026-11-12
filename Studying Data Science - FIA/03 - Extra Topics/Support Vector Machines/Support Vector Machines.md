# Support Vector Classifiers and Machines (SVMs)

> 📗 **Companion notes** for *The StatQuest Illustrated Guide to Machine Learning* (Josh Starmer), **Chapter 11**.
> Written in my own words with my own examples. Not part of the FIA course modules, so it lives in **Extra Topics**.

---

## 1. Maximal margin: the starting point

Two classes on a line (e.g., transaction amount: legit vs. fraud). A simple classifier puts a **threshold** between them. The **maximal margin** choice puts it exactly halfway between the closest points of each class, so the gap on both sides (the **margin**) is as large as possible.

**Problem:** it's extremely sensitive to outliers. One mislabeled fraud sitting among legit transactions drags the threshold to a bad place.

---

## 2. Support vector classifier: allow some mistakes

A **soft margin** deliberately lets a few training points be misclassified or sit inside the margin, in exchange for a threshold that generalizes better.

- How many violations to allow is a hyperparameter (often called **C**), tuned with **cross validation**.
- The points on the edge of, or inside, the margin are the **support vectors**: they alone determine the boundary.
- In 2 features the boundary is a line; in 3, a plane; in more, a **hyperplane**.

> 💡 Same bias–variance trade-off as everywhere: a soft margin adds a little bias (some training errors) to cut variance (less sensitivity to individual points).

---

## 3. When no straight boundary works

Example (my own): a medication works only for a **middle range of doses**, so "worked" points sit between two groups of "didn't work" points on the dose axis. No single threshold separates them.

**Trick:** add a new dimension. Plot each point at (dose, dose²). In that 2-D space the middle-dose points sit low and the extreme ones high, so a **straight line can now separate them**. A support vector classifier in the new space = a curved boundary in the original space.

That's a **support vector machine**: transform the data into a higher-dimensional space where a linear separator works.

---

## 4. Kernels: the shortcut

Computing the new coordinates explicitly can be expensive (or infinite-dimensional). **Kernel functions** compute the **relationships (similarities) between pairs of points** as if they were in the higher-dimensional space, without ever building the new coordinates. This is the **kernel trick**.

| Kernel | Formula (for two points a, b) | Intuition |
|---|---|---|
| **Polynomial** | (a·b + r)ᵈ | Adds polynomial combinations of features up to degree d; r and d tuned by CV |
| **Radial basis function (RBF)** | e^{−γ ‖a − b‖²} | Like a weighted nearest-neighbor vote: nearby points influence the classification a lot, distant points almost nothing. Corresponds to an **infinite**-dimensional space |

Example of the polynomial kernel with r = 1/2 and d = 2 in 1-D: (ab + ½)² = a²b² + ab + ¼, which is the dot product of the vectors (a, a², ½) and (b, b², ½). So the kernel gives the high-dimensional relationship directly.

---

## 5. Strengths and weaknesses

| ✅ | ⚠️ |
|---|---|
| Strong performance on small/medium, high-dimensional data | Slow on very large datasets |
| Flexible boundaries via kernels | Several hyperparameters (C, kernel, γ, d) to tune |
| Only support vectors matter → robust to far-away points | No natural probability output (needs extra calibration) |
| | Requires **feature scaling** |

---

## 6. Pseudocode

```text
FUNCTION LinearKernel(a, b):           RETURN a · b
FUNCTION PolyKernel(a, b, r, d):       RETURN (a · b + r)^d
FUNCTION RBFKernel(a, b, γ):           RETURN exp(-γ · ||a - b||²)

FUNCTION TrainSVM(X, y ∈ {-1, +1}, kernel, C)
    X ← standardize(X)
    K[i, j] ← kernel(X[i], X[j]) for all pairs
    α ← solve the quadratic program:
            maximize  Σ α_i - ½ Σ_i Σ_j α_i α_j y_i y_j K[i, j]
            subject to 0 ≤ α_i ≤ C  and  Σ α_i y_i = 0
    support_vectors ← { i : α_i > 0 }
    b ← intercept computed from points with 0 < α_i < C
    RETURN α, b, support_vectors

FUNCTION PredictSVM(x, model)
    s ← Σ_{i ∈ support_vectors} α_i · y_i · kernel(X[i], x) + b
    RETURN sign(s)

# Choose C, kernel and its parameters with cross validation.
```

---

## 7. Self-check

1. What's wrong with the maximal margin classifier? → It's extremely sensitive to outliers.
2. What are support vectors? → The points on or inside the margin; they alone define the boundary.
3. What does the kernel trick avoid? → Explicitly computing high-dimensional coordinates.
4. With the RBF kernel, do far-away points matter much for a new prediction? → Very little.

---

🔗 Related: [Cross Validation](../../01%20-%20Artificial%20Intelligence%20I/1.4%20-%20Validation%20and%20Regularization/Cross%20Validation.md) · [Assessing Model Performance](../../02%20-%20Artificial%20Intelligence%20II/2.3%20-%20Classification%20-%20Logistic%20Regression/Assessing%20Model%20Performance.md)
