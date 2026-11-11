# 1.2 — Introduction to Modeling (Supervised and Unsupervised Learning)

> **Module:** Artificial Intelligence I
> **Goal:** start the modeling journey — learning from data. Covers inferential statistics, probability, sampling, and the vocabulary of modeling.

---

## Table of Contents

1. [Levels of analytical complexity](#1-levels-of-analytical-complexity)
2. [Inferential statistics](#2-inferential-statistics)
3. [Probability](#3-probability)
4. [Sampling methods](#4-sampling-methods)
5. [What is a model?](#5-what-is-a-model)
6. [Statistical learning vs. machine learning](#6-statistical-learning-vs-machine-learning)
7. [Target vs. features](#7-target-vs-features)
8. [Supervised vs. unsupervised learning](#8-supervised-vs-unsupervised-learning)
9. [Predictive vs. diagnostic models](#9-predictive-vs-diagnostic-models)
10. [Data drift](#10-data-drift)
11. [Concept map](#11-concept-map)
12. [Pseudocode](#12-pseudocode)
13. [Quiz (with answers)](#13-quiz-with-answers)
14. [Interview questions](#14-interview-questions)
15. [References](#15-references)

---

## 1. Levels of analytical complexity

```
  ▲  Cognitive     — How do we simulate human reasoning?
  │  Prescriptive  — How do we change the future?
  │  Predictive    — What will happen?            ┐
  │  Diagnostic    — Why did it happen?           ├─ focus of the course (modeling)
  │  Descriptive   — What happened? (= EDA, 1.1)  ┘
```

Example applications:

| Question | Type of model |
|---|---|
| Which demographic/behavioral factors drive default within 60 days after an installment purchase? | Diagnostic classification (credit risk) |
| What is the probability that an insurance policy will be renewed at the end of the next annual cycle? | Predictive classification (churn) |
| Are there groups of telecom customers with similar usage (calls, data, add-ons)? | Unsupervised segmentation (clustering) |

---

## 2. Inferential statistics

### Population and sample

- **Population:** the complete set of elements you want to study.
- **Sample:** a subset used to draw valid conclusions about the whole (since collecting the whole population is often impossible).

### Descriptive vs. inferential

| | Descriptive statistics | Inferential statistics |
|---|---|---|
| Goal | **Describe** what was observed (sample or known population) | **Infer / estimate / predict** unknown population behavior from a known sample |
| Example | "Among 1,200 voters surveyed, 23% chose candidate A" | "Among *all* voters, about X% will choose A" |
| Example | "Among 100k customers active 6 months ago, 14% canceled" | "Among customers active *today*, X% will cancel" |

> 💡 **Core idea:** every model is an inferential tool. It learns from a **sample** (historical data) and generalizes to the **population** (new customers, future periods).

---

## 3. Probability

Probability is a measure (0% to 100%) that quantifies the **uncertainty** of a random (non-deterministic) event.

| Situation | Description | Examples |
|---|---|---|
| **Known probability** | Finite, equally likely outcomes → computed from the problem structure, no data needed | Even number on a die: 3/6 = ½. Heads on a coin: ½. A heart from a deck: 13/52 = ¼ |
| **Unknown probability** | Real-world phenomena; outcomes not equally likely. The true probability (= the event's proportion in the population) exists but must be **estimated from a sample** | Probability of an adult having hypertension; a customer stopping using their account in 12 months; rain on a summer day |

The sample can be:

- **Cross-sectional:** drawn at one moment in time (1,500 adults examined today).
- **Temporal / cohort ("safra", vintage):** a slice of the past whose outcome is already known (customers active 12 months ago → we know who churned).

---

## 4. Sampling methods

### Probability sampling
Each element has a **known, non-zero** probability of being selected → allows generalization and quantifying estimation error.

| Method | How it works | Example |
|---|---|---|
| **Simple random** | Everyone has the same chance | 500 customers drawn for a gift |
| **Stratified** | Split into homogeneous strata, sample within each → guarantees representativeness | Election poll with quotas by gender, age, region, class (drawn randomly within strata) |
| **Systematic** | Random start, then every k-th element | Every 10th sign-up receives a survey |
| **Cluster** | Split into natural groups, draw some groups, observe everyone in them | 20 schools drawn; all teachers in them evaluated |

### Non-probability sampling
Not random with known probability → cheaper, but **risk of bias**, no guarantee of representativeness.

| Method | How it works | Example |
|---|---|---|
| **Convenience** | Whoever is most accessible | Customers who walk into a store and accept to try a piece |
| **Quota** | Fill quotas by characteristic, but choice within quota isn't random | Interview pedestrians until 50 men and 50 women |
| **Purposive (judgment)** | Expert chooses "representative" cases | Audit only the most relevant bank branches |
| **Snowball** | Respondents recruit new participants | Patients with a rare disease via contact networks |

---

## 5. What is a model?

Inferential statistics estimates unknown population quantities (like a probability). A **model generalizes this idea**: instead of estimating a single number, it captures **patterns, relationships or structures involving many variables**, forming a rule that explains and predicts phenomena.

> **Learning from data:** the model learns about the *population* from the *sample*.

Examples: estimating e-commerce sales from ad spend; approving/rejecting loans from personal and financial history; detecting pneumonia in chest X-rays from past diagnoses.

Models may be **mathematical equations** and/or **computational algorithms** (not a strict split — equations are fitted by algorithms, and algorithms have math foundations):

```
Equation:  Linear regression     Y = a + b·X
Equation:  Logistic regression   p = 1 / (1 + e^(−z)),  z = a + b·X
Algorithm: Decision tree         Income > R$ 5,000? ── yes → Approve
                                                   └─ no  → Reject
```

---

## 6. Statistical learning vs. machine learning

| | Statistical learning | Machine learning |
|---|---|---|
| Priority | **Interpretation & inference** about parameters: how each variable influences the response | **Predictive performance** on new observations, even if less interpretable |
| Typical methods | Linear & logistic regression, GLMs, ARIMA | Random forest, gradient boosting, deep neural networks, SVM |

---

## 7. Target vs. features

- **Target (response variable, Y):** the main characteristic of interest — what you want to understand or predict.
- **Features (explanatory variables, X):** characteristics that help explain or predict the target.

Example — insurance renewal:

```
Features (X)                                   Target (Y)
• policy value (R$)                    ┐
• # claims during the term             │
• tenure with the insurer (years)      ├──► MODEL ──► Was the policy renewed?  (yes / no)
• coverage type (full, partial)        │
• acquisition channel (broker, online) ┘
```

---

## 8. Supervised vs. unsupervised learning

### Supervised
Starts from a sample where the **target is known** and learns relationships that explain it. Most common type; subdivided by the target's nature:

| Task | Target | Sub-types / examples |
|---|---|---|
| **Classification** | Qualitative (labels) | **Binary:** is a credit-card transaction fraud? yes/no · **Multinomial:** cancel reason: price, service, quality |
| **Regression ("projection")** | Quantitative (numbers) | **Non-temporal:** energy consumption across households · **Temporal (time series):** stock price next business day |

### Unsupervised
No target — only features. Finds **structural patterns** in the data.

| Task | What it does | Example |
|---|---|---|
| **Segmentation (clustering)** | Groups similar observations | Group retail customers by Recency, Frequency, Monetary value (RFM) |
| **Dimensionality reduction** | Reduces # of columns keeping most information | Represent athletes with 5 main "axes" from 50 physical/performance features (e.g., PCA) |

---

## 9. Predictive vs. diagnostic models

| Predictive | Diagnostic |
|---|---|
| Estimates **unknown target values for new** elements | Understands **how features relate** to the target, without extrapolating |
| Ex.: predict the delay of a new construction project | Ex.: understand which factors drive rent in a São Paulo neighborhood |

### Time windows in predictive models with temporal samples

```
          Historical period → FEATURES              Prediction period → TARGET
  ... T-12  T-11  ...  T-2  T-1   │ REF │   T+1  T+2  T+3 ...
```

- Features come from the **past**; the target comes from the **future** relative to a **reference date (REF)**.
- For **training**, REF is moved back in time so the target is already observed.
- In **production (scoring)**, REF = today.

> ⚠️ This is how you avoid **data leakage**: never let information from after REF enter the features.
>
> Note: every temporal forecasting model is predictive, but not every predictive model is temporal.

---

## 10. Data drift

A model is only valid while the **data-generating mechanism stays essentially the same** (no structural breaks). If data characteristics evolve, what was learned may lose validity.

- **Abrupt changes:** pandemic lockdown, a new regulation, a disruptive product launch.
- **Gradual changes:** customer behavior, economic scenario, service quality.

**Data drift** (input distributions change) can cause **prediction drift** (distorted predictions). Monitoring drift is a core MLOps task. Suggested reading: <https://www.evidentlyai.com/ml-in-production/data-drift>

---

## 11. Concept map

```
POPULATION ── SAMPLE
                 ├── Sampling: cross-sectional
                 │      ├── Probability: simple random, stratified, systematic, cluster
                 │      └── Non-probability: convenience, quota, purposive, snowball
                 └── Sampling: temporal (cohort/vintage)

ANALYSIS
 ├── Descriptive statistics (EDA)
 └── Inferential statistics
       ├── Models
       │    ├── Tradition: statistical learning | machine learning
       │    ├── Supervised: target + features
       │    │     ├── Tasks: binary classif. | multinomial classif. | non-temporal regression | temporal regression
       │    │     └── Purpose: predictive | diagnostic
       │    └── Unsupervised: segmentation | dimensionality reduction | ...
       └── Other techniques: hypothesis tests, confidence intervals
```

---

## 12. Pseudocode

### Choosing the modeling approach

```text
FUNCTION ChooseApproach(problem)
    IF problem has no target variable:
        IF goal = "group similar observations":   RETURN Unsupervised → Clustering
        IF goal = "compress many columns":        RETURN Unsupervised → Dimensionality reduction
    ELSE:
        IF target is qualitative:
            IF 2 categories:  task ← Binary classification
            ELSE:             task ← Multinomial classification
        ELSE (target is quantitative):
            IF target is indexed by time and we predict future instants:
                task ← Temporal regression (time series forecasting)
            ELSE:
                task ← Non-temporal regression
        IF goal = "explain drivers":   purpose ← Diagnostic (prefer interpretable models)
        ELSE:                          purpose ← Predictive (prefer accuracy; validate out-of-sample)
        RETURN (task, purpose)
```

### Building a modeling dataset with a reference date (avoiding leakage)

```text
PROCEDURE BuildTrainingSet(entities, events, REF, feature_window, target_window)
    population ← entities that were ACTIVE at REF          # e.g., students active 1 year ago
    FOR each entity e in population:
        features[e] ← aggregate events of e IN [REF - feature_window, REF)
        target[e]   ← outcome of e IN (REF, REF + target_window]   # e.g., dropped out next semester?
    RETURN (features, target)

# Production:
#   scoring_set ← BuildFeaturesOnly(entities active today, REF = today)
#   predictions ← model.predict(scoring_set)
```

---

## 13. Quiz (with answers)

1. An insurer sent a survey to 5,000 customers who had a claim in the last 3 months; 1,200 answered. **What's the sample with available data?** → **(b) the 1,200 who answered.**
2. A clothing store extracted all customers who bought in 2025 and had complete records. **Which population does this represent?** → **(c) customers who bought in 2025 with complete data** (restricting to complete records defines the population you can actually generalize to).
3. **Where can probability be obtained from the problem structure, without data?** → **(c)** a child of two Aa parents being aa (Mendelian genetics: ¼).
4. A coastal news site runs an online poll and calls it "the opinion of the Brazilian population." → **(c)** respondents self-select (bias) and don't represent Brazil; it's non-probability sampling.
5. Auditor sorts all entries and checks 1 in every 50 from a random start. → **(d) Systematic.**
6. To estimate the probability that a new battery fails within 12 months of use → **(b)** units from lots produced more than 12 months ago, whose fail/no-fail history is known.
7. Which challenge requires a model (inference), not descriptive analysis? → **(c)** probability of new buildings having structural damage within 5 years.
8. Where would a classification model fit? → **(b)** classify a product review as positive, neutral or negative.
9. A retailer wants to group stores with similar profiles → **(c) Unsupervised — segmentation.**
10. Predictive model of university dropout — how to define the population? → **(b)** students active 1 year ago: features up to the reference date, dropout in the following semester.

---

## 14. Interview questions

1. **Difference between descriptive and inferential statistics?** Descriptive summarizes observed data; inferential generalizes from a sample to a population with uncertainty.
2. **Classification vs. regression?** Categorical vs. numeric target.
3. **Supervised vs. unsupervised?** Presence or absence of a labeled target.
4. **What is data leakage and how do you prevent it?** Using information not available at prediction time (e.g., from after the reference date). Prevent with strict time windows (features ≤ REF < target) and pipelines fitted only on training data.
5. **What is data drift? How would you monitor it?** Change in input distributions over time; monitor feature distributions (e.g., PSI, KS test), prediction distributions and performance metrics; retrain when needed.
6. **When would you choose an interpretable statistical model over a black-box ML model?** When the goal is diagnostic/explanatory, under regulation (credit, insurance), or when stakeholders must trust and act on drivers.
7. **Why is stratified sampling useful?** Guarantees representation of important subgroups and reduces variance of estimates.
8. **What's a vintage/cohort ("safra") sample?** A group observed from a past reference date whose outcome is already known — the standard way to build training data for predictive models.

---

## 15. References

- Provost, F.; Fawcett, T. *Data Science for Business*. O'Reilly, 2013.
- Shmueli, G.; Bruce, P.; Gedeck, P.; Patel, N. *Data Mining for Business Analytics: Concepts, Techniques and Applications in Python*. Wiley, 2020.
- James, G.; Witten, D.; Hastie, T.; Tibshirani, R.; Taylor, J. *An Introduction to Statistical Learning with Applications in Python*. Springer, 2023.
- Kuhn, M.; Johnson, K. *Applied Predictive Modeling*. Springer, 2013.

---

⬅️ Previous: [1.1 — EDA](../1.1%20-%20Exploratory%20Data%20Analysis%20and%20Data%20Visualization/README.md) · 🏠 [Index](../../README.md) · ➡️ Next: [1.3 — Linear Regression](../1.3%20-%20Linear%20Regression/README.md)
