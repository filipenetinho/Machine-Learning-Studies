# Cross Validation

> 📗 **Companion notes** for *The StatQuest Illustrated Guide to Machine Learning* (Josh Starmer), **Chapter 2**.
> Written in my own words with my own examples. Course notes: [1.4 — Validation and Regularization](README.md).

---

## 1. The problem cross validation solves

To judge a model we need data it hasn't seen. But **which** rows should we hold out? With one random split:
- a "lucky" test set makes a bad model look good;
- an "unlucky" one makes a good model look bad;
- the held-out rows are never used for training, which hurts when data is scarce.

**Cross validation** fixes this by **rotating** the held-out part so that **every row gets used for testing exactly once**, and averaging the results.

---

## 2. k-fold cross validation

1. Shuffle the data and cut it into **k equal blocks (folds)**.
2. For each fold: train on the other k − 1 folds, test on this one, record the error.
3. Summarize the k test errors: their **mean** (expected performance) and **spread** (stability).

```
k = 4
round 1: [TEST][train][train][train]
round 2: [train][TEST][train][train]
round 3: [train][train][TEST][train]
round 4: [train][train][train][TEST]
```

### Toy example (my own)

Comparing two methods on 4 folds (MAE in minutes):

| Fold | Line | Tree |
|---|---|---|
| 1 | 6.1 | 5.2 |
| 2 | 5.8 | 9.4 |
| 3 | 6.4 | 4.9 |
| 4 | 6.0 | 8.7 |
| **Mean** | **6.08** | 7.05 |
| Spread | small | large |

The tree wins on two folds but is erratic; on average the line is better **and** more stable. A single split could have picked either one.

---

## 3. Variants

| Variant | How | When |
|---|---|---|
| **k-fold (k = 5 or 10)** | k rounds, each fold tested once | The usual default |
| **Leave-one-out (LOOCV)** | k = n: each row is its own test set | Very small datasets (costly for large ones) |
| **Stratified k-fold** | Each fold keeps the class proportions | Classification, especially imbalanced classes |
| **Time-series split** | Train only on the past, test on the next block | Temporal data (never shuffle!) — see [2.2](../../02%20-%20Artificial%20Intelligence%20II/2.2%20-%20Time%20Series%20Forecasting%20Models/README.md) |

---

## 4. What cross validation is used for

1. **Comparing methods** (line vs. tree vs. boosting).
2. **Tuning hyperparameters** (tree depth, regularization strength λ, learning rate): run CV for each candidate value and keep the best average.
3. **Estimating** how the final model will perform on new data.

> ⚠️ If CV was used to *choose* among many options, its score is slightly optimistic. Keep a final **untouched test set** for one last honest check (the "holdout + CV" scheme from the course).

> ⚠️ **Leakage inside CV:** any step that learns from data (scaling, imputing, selecting features) must be fit **inside each training fold**, not on the full data beforehand. Pipelines handle this.

---

## 5. Pseudocode

```text
FUNCTION KFoldCV(X, y, make_model, k = 5, metric = MAE)
    idx ← SHUFFLE(1..n)
    folds ← split idx into k nearly equal parts
    scores ← []
    FOR f IN 1..k:
        test_idx  ← folds[f]
        train_idx ← all idx not in folds[f]
        model ← make_model()                          # fresh model every round
        model.fit(X[train_idx], y[train_idx])         # preprocessing fitted here too
        scores.append( metric(y[test_idx], model.predict(X[test_idx])) )
    RETURN mean(scores), std(scores)

FUNCTION TuneWithCV(X, y, candidate_values)
    FOR v IN candidate_values:
        result[v] ← KFoldCV(X, y, () → Model(hyperparameter = v))
    RETURN v with the best mean score (prefer stable / simpler on ties)
```

---

## 6. Self-check

1. With 1,000 rows and 10-fold CV, how many rows train each model? → **900**.
2. LOOCV on 1,000 rows fits how many models? → **1,000**.
3. Why not shuffle a time series before CV? → You'd train on the future and test on the past (leakage).
4. Why fit the scaler inside each fold? → Otherwise test-fold statistics leak into training.

---

🔗 Related: [1.4 course notes](README.md) · [Regularization](Regularization.md) · [Fundamental Concepts in ML](../1.2%20-%20Introduction%20to%20Modeling/Fundamental%20Concepts%20in%20Machine%20Learning.md)
