# Regression Trees

> 📗 **Companion notes** for *The StatQuest Illustrated Guide to Machine Learning* (Josh Starmer), **Chapter 10** (regression-tree part).
> Written in my own words with my own examples. Course notes: [1.5 — Regression Trees](README.md).

---

## 1. When a line isn't enough

Suppose a drug's effect is low at small doses, high at medium doses and low again at high doses. No straight line captures that. A **regression tree** handles it by cutting the feature into ranges and predicting the **average target in each range**.

```
            Dose < 10?
           /         \
        yes           no
      ŷ = 2        Dose < 25?
                   /       \
                 yes        no
               ŷ = 18     ŷ = 4
```

---

## 2. How a split is chosen

For a numeric feature:
1. Sort the values and try every **midpoint between neighbors** as a threshold.
2. For each threshold, predict each side with **its own average** and compute the **SSR** of the split (left SSR + right SSR).
3. Keep the threshold with the **lowest SSR**.

With several features, do this for each one and keep the feature + threshold with the lowest SSR overall. Then repeat inside each branch.

### Worked example (my own)

| Dose | 5 | 8 | 15 | 20 | 30 | 35 |
|---|---|---|---|---|---|---|
| Effect | 1 | 3 | 17 | 19 | 5 | 3 |

Threshold 11.5 (between 8 and 15):
- Left {1, 3}: mean 2 → SSR = 1 + 1 = 2
- Right {17, 19, 5, 3}: mean 11 → SSR = 36 + 64 + 36 + 64 = 200
- Total = **202**

Threshold 25 (between 20 and 30):
- Left {1, 3, 17, 19}: mean 10 → SSR = 81 + 49 + 49 + 81 = 260
- Right {5, 3}: mean 4 → SSR = 2
- Total = 262

11.5 is better than 25. Checking all midpoints the same way gives the root split; then the right branch {17, 19, 5, 3} is split again (at 25) to separate the high middle from the low end.

---

## 3. When to stop splitting

A tree that keeps splitting ends with one observation per leaf: zero training error, terrible on new data (overfitting). Common stopping rules:
- **Minimum observations per leaf** (e.g., don't split a node with fewer than 20 rows; a typical default in practice).
- **Maximum depth.**
- **Minimum SSR reduction** required to split.

Alternatively, grow a big tree and **prune** it back (cost-complexity pruning), using cross validation to pick how much to cut.

---

## 4. Multiple features and mixed types

- Numeric features: thresholds at midpoints.
- Categorical features: splits like "region = North vs. not North".
- The same feature can appear several times at different depths.

---

## 5. Strengths and limitations

| ✅ | ⚠️ |
|---|---|
| Captures non-linear shapes and interactions | Predictions are **step-shaped** (constant inside a leaf) |
| Easy to read and explain | Can't predict outside the training range of y |
| No feature scaling needed | Unstable: small data changes → different tree |
| | Overfits easily → tune with CV, or use ensembles (Random Forest, boosting — course 1.6) |

---

## 6. Pseudocode

```text
FUNCTION BestSplit(X, y, min_leaf)
    best ← (ssr = ∞)
    FOR each feature f:
        values ← sorted unique X[:, f]
        FOR each midpoint t between consecutive values:
            L ← y[X[:, f] < t];  R ← y[X[:, f] ≥ t]
            IF |L| < min_leaf OR |R| < min_leaf: CONTINUE
            ssr ← Σ (L - mean(L))² + Σ (R - mean(R))²
            IF ssr < best.ssr: best ← (f, t, ssr)
    RETURN best

FUNCTION GrowRegressionTree(X, y, depth, max_depth, min_leaf)
    IF depth = max_depth OR |y| < 2·min_leaf: RETURN Leaf(mean(y))
    s ← BestSplit(X, y, min_leaf)
    IF s is empty: RETURN Leaf(mean(y))
    left  ← GrowRegressionTree(rows with X[:, s.f] < s.t, depth + 1, ...)
    right ← GrowRegressionTree(rows with X[:, s.f] ≥ s.t, depth + 1, ...)
    RETURN Node(s.f, s.t, left, right)
```

---

## 7. Self-check

1. What does a leaf predict? → The average target of the training rows in it.
2. How is a numeric threshold chosen? → Try midpoints between sorted values; keep the one with the lowest total SSR.
3. Why limit leaf size? → To avoid memorizing individual points.
4. Can a regression tree predict a value higher than any target in training? → No.

---

🔗 Related: [1.5 course notes](README.md) · [Classification Trees](../../02%20-%20Artificial%20Intelligence%20II/2.4%20-%20Classification%20-%20Decision%20Trees/Classification%20Trees.md)
