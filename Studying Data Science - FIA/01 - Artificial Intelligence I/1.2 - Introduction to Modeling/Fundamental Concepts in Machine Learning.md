# Fundamental Concepts in Machine Learning

> 📗 **Companion notes** for *The StatQuest Illustrated Guide to Machine Learning* (Josh Starmer), **Chapter 1**.
> Written in my own words with my own examples. Read alongside the course notes: [1.2 — Introduction to Modeling](README.md).

---

## 1. Machine learning in one sentence

Machine learning is a toolbox of methods that turn data into **decisions**. In practice those decisions come in two flavors:

| Task | Output | Example |
|---|---|---|
| **Classification** | A category | Will this credit-card transaction be fraud or not? |
| **Regression** | A number | How many minutes will this delivery take? |

> 💡 Most of what follows in the book (and in the course) is either a way to build one of these two kinds of models, or a way to judge how good the model is.

---

## 2. Many methods, one question: "which one works best?"

For the same data you could fit a straight line, a wiggly curve, a tree, a neural network… The training data alone can't tell you which is best, because a flexible enough method can pass through every training point.

The fair way to compare methods:

1. **Split** the data into **training** data (used to fit the model) and **testing** data (kept aside).
2. Fit each candidate method on the training data.
3. Measure each one's error on the **testing** data.
4. Prefer the method with the **smaller testing error**, not the smaller training error.

### Toy example (my own)

Predicting delivery time (minutes) from distance (km):

| Model | Training error (SSR) | Testing error (SSR) |
|---|---|---|
| Straight line | 120 | **140** ✅ |
| Very wiggly curve | **5** | 610 |

The wiggly curve memorized the training points but fails on new ones → **overfitting**. The line is "worse" on training data but is the better model.

> This is the same idea as the holdout / external test in [1.4](../1.4%20-%20Validation%20and%20Regularization/README.md).

---

## 3. Vocabulary

| Term | Meaning | Course equivalent |
|---|---|---|
| **Dependent variable** | What we want to predict | Target (Y) |
| **Independent variables** | What we use to predict | Features (X) |
| **Training data** | Data used to fit the model | Train set |
| **Testing data** | Data used to evaluate the fitted model | Test / external test |
| **Overfitting** | Great on training data, poor on new data | Overfitting (high variance) |
| **Bias** | A model's inability to capture the true relationship (e.g., a line for a curved pattern) | Bias in the bias–variance trade-off |
| **Variance** | How much the fit changes from one training set to another | Variance in the bias–variance trade-off |

> 💡 "Dependent" because its value *depends on* the others; the features are "independent" because we treat them as given, not because they're statistically independent of each other.

---

## 4. Types of data

```
Data
├── Discrete   — countable, often whole numbers: # of purchases, # of children
│               (also rankings and categories: "low / medium / high", "red / blue")
└── Continuous — measured, can take any value in a range: weight, temperature, time
```

Why it matters: the data type decides which methods, summaries and charts you can use (same message as the variable-type tree in [1.1](../1.1%20-%20Exploratory%20Data%20Analysis%20and%20Data%20Visualization/README.md)).

---

## 5. Pseudocode — comparing methods fairly

```text
PROCEDURE CompareMethods(data, methods, error_metric)
    train, test ← random_split(data, test_fraction = 0.25)
    results ← {}
    FOR each method m IN methods:
        model ← m.fit(train.X, train.y)
        results[m] ← error_metric(test.y, model.predict(test.X))
    best ← method with the lowest results[m]
    RETURN best, results
# A single split can be lucky or unlucky → cross validation repeats this
# with different splits (see "Cross Validation.md" in 1.4).
```

---

## 6. Self-check

1. Is predicting a customer's satisfaction level ("unhappy / neutral / happy") classification or regression? → **Classification** (categories, even though they're ordered).
2. Why can't training error choose between a line and a wiggly curve? → A flexible model can always shrink training error; only data it hasn't seen reveals overfitting.
3. A model has low training error and high testing error. Bias or variance problem? → **Variance** (overfitting).
4. Is "number of app logins this month" discrete or continuous? → **Discrete** (a count).

---

🔗 Related: [Cross Validation](../1.4%20-%20Validation%20and%20Regularization/Cross%20Validation.md) · [1.2 course notes](README.md)
