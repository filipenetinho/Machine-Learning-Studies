# Naive Bayes

> 📗 **Companion notes** for *The StatQuest Illustrated Guide to Machine Learning* (Josh Starmer), **Chapter 7**.
> Written in my own words with my own examples. Not covered as its own module in the FIA course; it's filed here next to logistic regression because both are **probabilistic classifiers**.

---

## 1. The idea

Naive Bayes classifies by asking, for each class: *how likely is this class in general* (**prior**) × *how likely are these features if the observation belongs to this class* (**likelihoods**)? The class with the highest score wins.

It's "naive" because it treats every feature as **independent of the others given the class**. That's rarely true (e.g., in a message, "free" and "prize" tend to appear together), but the method still works surprisingly well, especially for text.

$$\text{score}(c) = P(c)\times\prod_j P(x_j \mid c)$$

(From Bayes' theorem, P(c | x) ∝ P(c) · P(x | c); the denominator P(x) is the same for every class, so it can be ignored when comparing.)

---

## 2. Multinomial Naive Bayes (counts of words / tokens)

### Example (my own): is a support message *urgent*?

Training: 6 urgent messages and 14 normal ones → priors P(urgent) = 0.3, P(normal) = 0.7.
Word frequencies within each class:

| Word | P(word \| urgent) | P(word \| normal) |
|---|---|---|
| "down" | 0.30 | 0.05 |
| "please" | 0.10 | 0.25 |
| "now" | 0.25 | 0.05 |
| "thanks" | 0.05 | 0.30 |

New message: **"down now"**
- urgent: 0.3 × 0.30 × 0.25 = **0.0225**
- normal: 0.7 × 0.05 × 0.05 = 0.00175
→ classified as **urgent**.

Word **order is ignored** ("now down" scores the same): the model only sees which words appear and how often ("bag of words").

### The zero-count problem

If a word never appeared in one class during training, its probability is 0 and wipes out the whole product. Fix: **add a pseudocount** (usually +1 to every word count, called **Laplace smoothing**) before computing frequencies.

---

## 3. Gaussian Naive Bayes (continuous features)

For numeric features, model each feature inside each class with a **normal distribution** (mean and standard deviation estimated from that class's training data). The likelihood of a value is the **height of that bell curve** at the value.

Example: classify a transaction as fraud or not using amount and hour. For each class, fit one normal curve for amount and another for hour, then multiply prior × density(amount) × density(hour).

> 💡 **Use logs:** with many features the product becomes tiny → add log-probabilities instead of multiplying probabilities (avoids underflow).

---

## 4. Strengths and weaknesses

| ✅ | ⚠️ |
|---|---|
| Very fast to train and predict | Independence assumption is unrealistic |
| Works with little data and many features (text) | Probabilities are often poorly calibrated (too close to 0 or 1) |
| Simple to explain | Can't capture interactions between features |
| Strong baseline for spam / sentiment / topic classification | Gaussian version assumes bell-shaped features |

> Naive Bayes has **high bias** (a very simple assumption) but **low variance** (stable across samples), which is why it holds up well on small datasets.

---

## 5. Pseudocode

```text
FUNCTION TrainMultinomialNB(docs, labels, α = 1)
    FOR each class c:
        prior[c] ← count(labels = c) / |labels|
        word_counts[c] ← total count of each word in docs of class c + α     # Laplace smoothing
        likelihood[c][w] ← word_counts[c][w] / Σ_w word_counts[c][w]
    RETURN prior, likelihood

FUNCTION PredictNB(doc, prior, likelihood)
    FOR each class c:
        score[c] ← log(prior[c]) + Σ_{w in doc} log(likelihood[c][w])
    RETURN argmax_c score[c]

FUNCTION TrainGaussianNB(X, y)
    FOR each class c:
        prior[c] ← mean(y = c)
        μ[c][j], σ[c][j] ← mean and std of feature j among rows of class c
    RETURN prior, μ, σ
# predict: log prior + Σ_j log NormalPDF(x_j; μ[c][j], σ[c][j]), take the argmax
```

---

## 6. Self-check

1. Why is it called "naive"? → It assumes features are independent within each class.
2. A word never seen in "urgent" messages appears in a new message. What breaks and what's the fix? → Its probability is 0, making the class score 0; add a pseudocount (Laplace smoothing).
3. Why use log-probabilities? → To avoid underflow when multiplying many small numbers.
4. Does Naive Bayes care about word order? → No (bag of words).

---

🔗 Related: [Logistic Regression](Logistic%20Regression.md) · [Fundamental Concepts in Statistics](../../01%20-%20Artificial%20Intelligence%20I/1.1%20-%20Exploratory%20Data%20Analysis%20and%20Data%20Visualization/Fundamental%20Concepts%20in%20Statistics.md)
