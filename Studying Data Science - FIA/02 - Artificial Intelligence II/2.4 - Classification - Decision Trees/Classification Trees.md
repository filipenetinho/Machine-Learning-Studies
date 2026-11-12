# Classification Trees

> 📗 **Companion notes** for *The StatQuest Illustrated Guide to Machine Learning* (Josh Starmer), **Chapter 10** (classification-tree part).
> Written in my own words with my own examples. Course notes: [2.4 — Decision Trees](README.md).

---

## 1. The idea

A classification tree asks a sequence of yes/no questions about the features and ends in a **leaf** that predicts a class. Each question is chosen so the groups it creates are as **pure** (one class dominating) as possible.

```
           Late payments ≥ 2?
           /              \
         yes               no
   Income < 3k?         ŷ = "pays"
     /       \
   yes        no
ŷ = "defaults"  ŷ = "pays"
```

---

## 2. Measuring purity: Gini impurity

For one leaf with class proportions p₁, p₂, …:

$$\text{Gini} = 1 - \sum_k p_k^2$$

- 0 → perfectly pure (one class only).
- 0.5 → worst case for two classes (50/50).

For a whole split, take the **weighted average** of the two leaves' Gini values (weighted by how many rows each leaf has). Pick the question with the **lowest weighted Gini**.

### Worked example (my own): which question predicts churn better?

10 customers, 5 churned.

**Question A: "Has a support complaint?"**
- Yes: 4 customers, 3 churned → Gini = 1 − (0.75² + 0.25²) = 1 − 0.625 = **0.375**
- No: 6 customers, 2 churned → Gini = 1 − ((1/3)² + (2/3)²) = 1 − 0.556 = **0.444**
- Weighted: 0.4 × 0.375 + 0.6 × 0.444 = **0.417**

**Question B: "Monthly plan (vs. annual)?"**
- Yes: 5 customers, 4 churned → Gini = 1 − (0.8² + 0.2²) = **0.32**
- No: 5 customers, 1 churned → Gini = **0.32**
- Weighted: **0.32**

Question B has the lower impurity → it goes at the root.

---

## 3. Numeric features

1. Sort the values.
2. Try the **midpoint between each pair of neighbors** as a threshold.
3. Compute the weighted Gini for each threshold; keep the best.
4. Compare that best threshold with the other features' best questions.

(Exactly like regression trees, but with Gini instead of SSR — see [Regression Trees](../../01%20-%20Artificial%20Intelligence%20I/1.5%20-%20Regression%20Trees/Regression%20Trees.md).)

---

## 4. Growing and stopping

- Repeat the search inside each branch.
- A node becomes a **leaf** when it's pure, too small, or no split reduces impurity.
- Leaf prediction = the **majority class** (and the class proportions give a probability).
- A feature that never reduces impurity enough simply never gets used → trees do **automatic feature selection**.

**Avoiding overfitting:** limit depth, require a minimum number of rows per leaf, or grow a full tree and **prune** it (cost-complexity pruning), choosing the amount of pruning by cross validation.

> ⚠️ A leaf with very few rows (e.g., 1 customer) gives an overconfident prediction. Minimum leaf sizes keep predictions grounded.

---

## 5. Gini vs. entropy

Entropy, −Σ pₖ log₂ pₖ, is the alternative impurity measure. Both curves have the same shape and usually choose the same splits; Gini is a bit faster (no logarithm).

---

## 6. Pseudocode

```text
FUNCTION Gini(labels)
    RETURN 1 - Σ_k (count(labels = k) / |labels|)²

FUNCTION WeightedGini(left, right)
    n ← |left| + |right|
    RETURN (|left|/n)·Gini(left) + (|right|/n)·Gini(right)

FUNCTION GrowClassificationTree(X, y, depth, max_depth, min_leaf)
    IF Gini(y) = 0 OR depth = max_depth OR |y| < 2·min_leaf:
        RETURN Leaf(majority(y), proportions(y))
    best ← NONE
    FOR each feature f:
        FOR each candidate question q on f          # midpoints, or "category = v"
            L, R ← split y by q
            IF |L| < min_leaf OR |R| < min_leaf: CONTINUE
            IF best is NONE OR WeightedGini(L, R) < best.score: best ← (q, WeightedGini(L, R))
    IF best is NONE OR best.score ≥ Gini(y): RETURN Leaf(majority(y), proportions(y))
    RETURN Node(best.q, GrowClassificationTree(rows where q is true, ...),
                         GrowClassificationTree(rows where q is false, ...))
```

---

## 7. Self-check

1. Gini of a leaf with 8 "yes" and 2 "no"? → 1 − (0.64 + 0.04) = **0.32**.
2. Why weight the leaves' Gini by their size? → A tiny pure leaf shouldn't make a split look good if the big leaf stays mixed.
3. How does a tree handle a useless feature? → It never chooses it for a split.
4. What does a leaf predict? → The majority class; the proportions can serve as probabilities.

---

🔗 Related: [2.4 course notes](README.md) · [Regression Trees](../../01%20-%20Artificial%20Intelligence%20I/1.5%20-%20Regression%20Trees/Regression%20Trees.md) · [Assessing Model Performance](../2.3%20-%20Classification%20-%20Logistic%20Regression/Assessing%20Model%20Performance.md)
