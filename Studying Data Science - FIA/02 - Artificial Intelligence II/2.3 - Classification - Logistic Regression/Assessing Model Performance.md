# Assessing Model Performance

> 📗 **Companion notes** for *The StatQuest Illustrated Guide to Machine Learning* (Josh Starmer), **Chapter 8**.
> Written in my own words with my own examples. Course notes: [2.3 — Logistic Regression](README.md) (sections 8–11).

---

## 1. The confusion matrix

After classifying test data, count four kinds of outcomes:

|  | **Predicted: yes** | **Predicted: no** |
|---|---|---|
| **Actually yes** | True Positive (TP) | False Negative (FN) |
| **Actually no** | False Positive (FP) | True Negative (TN) |

> ⚠️ Different books and libraries put actual/predicted on different axes. Always read the labels before reading the numbers.

With more than two classes, the matrix is K × K; the diagonal holds the correct predictions.

---

## 2. Metrics from the confusion matrix

| Metric | Formula | Question it answers |
|---|---|---|
| **Sensitivity (recall, TPR)** | TP / (TP + FN) | Of the real positives, how many did we catch? |
| **Specificity (TNR)** | TN / (TN + FP) | Of the real negatives, how many did we correctly clear? |
| **Precision** | TP / (TP + FP) | When we say "positive", how often are we right? |
| **False positive rate** | FP / (FP + TN) = 1 − specificity | How often we raise false alarms |
| **Accuracy** | (TP + TN) / total | Overall share correct (misleading with imbalanced classes) |

### Example (my own): screening for machine failures, 1,000 machines

|  | Pred. fail | Pred. OK |
|---|---|---|
| **Fails** | 40 | 10 |
| **OK** | 60 | 890 |

- Sensitivity = 40 / 50 = **80%**
- Specificity = 890 / 950 = **93.7%**
- Precision = 40 / 100 = **40%** (most alarms are false, because failures are rare)
- Accuracy = 930 / 1,000 = **93%** (but "always OK" would score 95%!)

---

## 3. Thresholds change everything

A model like logistic regression outputs probabilities; the **threshold** turns them into yes/no. Lowering it catches more positives (↑ sensitivity) but raises more false alarms (↓ specificity, usually ↓ precision). The right threshold depends on the **costs**: missing a failure vs. an unnecessary inspection.

Testing every threshold one by one and staring at dozens of confusion matrices is tedious → **ROC curves** summarize them all.

---

## 4. ROC curve and AUC

- **ROC curve:** for every threshold, plot **TPR (sensitivity)** on the y-axis against **FPR (1 − specificity)** on the x-axis, and connect the points.
- The diagonal = random guessing (TPR = FPR).
- The closer the curve hugs the **top-left corner**, the better.
- **AUC (area under the curve)** compresses the curve into one number between 0.5 (random) and 1 (perfect). It equals the probability that a random positive gets a higher score than a random negative.

Use cases:
- Pick a threshold by looking for the point with the trade-off you can afford.
- Compare models: higher AUC = better ranking overall.

---

## 5. Precision–recall curves for rare events

When negatives vastly outnumber positives (fraud, rare failures), the FPR barely moves even with many false alarms, because FP is divided by a huge number of negatives. The ROC curve can then look great while precision is poor.

A **precision–recall curve** (precision on y, recall on x) ignores true negatives and shows the real cost of false alarms. Its area (**average precision**) is a better summary for rare-event problems.

---

## 6. Pseudocode

```text
FUNCTION ConfusionMatrix(y, ŷ)
    TP ← Σ(y=1 ∧ ŷ=1);  FN ← Σ(y=1 ∧ ŷ=0);  FP ← Σ(y=0 ∧ ŷ=1);  TN ← Σ(y=0 ∧ ŷ=0)
    RETURN TP, FN, FP, TN

FUNCTION ROC(y, scores)
    points ← [(0, 0)]
    FOR t IN sorted(unique(scores), descending):
        ŷ ← scores ≥ t
        TP, FN, FP, TN ← ConfusionMatrix(y, ŷ)
        points.append( (FP/(FP+TN), TP/(TP+FN)) )          # (FPR, TPR)
    AUC ← trapezoid area under points
    RETURN points, AUC

FUNCTION PrecisionRecall(y, scores)
    FOR t IN sorted(unique(scores), descending):
        TP, FN, FP, _ ← ConfusionMatrix(y, scores ≥ t)
        curve.append( (TP/(TP+FN), TP/(TP+FP)) )            # (recall, precision)
    RETURN curve
```

---

## 7. Self-check

1. 2% positives, model predicts "negative" for everyone. Accuracy and sensitivity? → **98%** and **0%**.
2. Which axis of the ROC curve is the false positive rate? → The **x-axis**.
3. AUC = 0.5 means? → No better than random ranking.
4. When prefer a precision–recall curve? → When the positive class is rare and false alarms matter.

---

🔗 Related: [2.3 course notes](README.md) · [Logistic Regression](Logistic%20Regression.md) · [Cross Validation](../../01%20-%20Artificial%20Intelligence%20I/1.4%20-%20Validation%20and%20Regularization/Cross%20Validation.md)
