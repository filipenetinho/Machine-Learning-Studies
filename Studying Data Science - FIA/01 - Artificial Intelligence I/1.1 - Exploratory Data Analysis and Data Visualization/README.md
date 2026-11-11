# 1.1 — Exploratory Data Analysis (EDA) and Data Visualization

> **Course:** Advanced MBA in Analytics & AI — Data Science (FIA Business School / Labdata)
> **Module:** Artificial Intelligence I
> **Case study used in class:** residential real-estate listings for sale in a city (one row = one property).

---

## Table of Contents

1. [What EDA is and why it comes first](#1-what-eda-is-and-why-it-comes-first)
2. [Levels of data structure](#2-levels-of-data-structure)
3. [Observations, variables and the "grain" of a table](#3-observations-variables-and-the-grain-of-a-table)
4. [Variable types](#4-variable-types)
5. [Uniqueness and completeness (missing values)](#5-uniqueness-and-completeness-missing-values)
6. [Univariate analysis](#6-univariate-analysis)
7. [Outlier analysis](#7-outlier-analysis)
8. [Bivariate analysis](#8-bivariate-analysis)
9. [Data visualization](#9-data-visualization)
10. [Automated EDA](#10-automated-eda)
11. [Pseudocode — a full EDA pipeline](#11-pseudocode--a-full-eda-pipeline)
12. [Interview questions](#12-interview-questions)
13. [References](#13-references)

---

## 1. What EDA is and why it comes first

**Exploratory Data Analysis** is a set of statistical techniques used to *describe* a dataset and generate insights about how the data behaves. It must be done **before** any sophisticated analysis or predictive model.

EDA has three main goals:

| Goal | Question it answers |
|---|---|
| **Knowledge** | What does the data tell me about the business problem? |
| **Consistency** | Is the data correct? (duplicates, gaps, impossible values) |
| **Communication** | How do I show the findings clearly (tables and charts)? |

EDA is **descriptive**: it describes what happened. It may not include diagnosis (*why*), and it never includes prediction (*what will happen*). That's why it is also called **descriptive analysis**.

> 💡 **Concept behind it:** a model is only as good as the data fed to it ("garbage in, garbage out"). EDA is where you discover the problems — duplicated keys, missing values, outliers, skewed distributions — that would silently break a model later.

Typical business questions answered by EDA:
- What are the main demographic profiles of a supermarket loyalty-program member, and which product categories did they buy most in the last 12 months?
- How do "Diamond" credit-card customers' transaction patterns compare to lower tiers?
- Which assets were most searched by brokerage clients, and is there a relationship between investor demographics and portfolio diversification?

---

## 2. Levels of data structure

| Level | Description | Examples |
|---|---|---|
| **Structured** | Table format: rows and columns. Best format for querying and analysis. | Spreadsheets, SQL tables |
| **Semi-structured** | Has internal organization (fields, hierarchy, tags) but no fixed tabular schema. | JSON, XML, e-mail (sender/date are structured; body/attachments are not) |
| **Unstructured** | No predefined column schema; needs interpretation to extract variables. | Images, text, audio, video, social-media posts |

---

## 3. Observations, variables and the "grain" of a table

The first step with any table is to understand its structure:

- **Rows = observations (records).** Distinct elements of some nature: a customer, a company, a region, a transaction, a product, a month…
- **Columns = variables.** Observable characteristics of each record: tenure, number of call-center calls in 12 months, etc.

The **grain** (in Portuguese, *"visão da base"*) is *what one row represents*. In the real-estate case, **each row is one distinct property for sale**, so the grain is *property*.

> 💡 Getting the grain right is essential: it guides how you interpret the data, how you join/aggregate it, and which analyses make sense. Many bugs in analytics engineering are grain bugs (e.g., joining at the wrong level and duplicating rows).

### Data dictionary

You should always have a **data dictionary** describing each variable:

- Variable name
- Description of what it represents
- Type (key, integer, decimal, text…)
- Unit of measure (if applicable)
- Possible values (domain)
- Presence of missing values
- Calculation rule (if applicable)
- Data source / lineage

Example (real-estate case):

| Variable | Description |
|---|---|
| `ID_IMOVEL` | Property ID |
| `BAIRRO` | Neighborhood |
| `TIPO_IMOVEL` | Property type: house or apartment |
| `METRAGEM` | Area in m² (for houses, land area) |
| `VALOR_VENDA` | Current sale price (R$) |
| `INTENSIDADE_LUZ` | Sunlight during the day: none, little, a lot |
| `VAGAS_GARAGEM` | Number of parking spots |
| `FLUXO_VEICULOS` | Traffic on the street: low, intermediate, heavy |
| `COMERCIOS_RAIO_500M` | Number of shops within 500 m |

---

## 4. Variable types

**"Which analysis should I use?" → It depends on the variable type.**

```
Variables
├── Key (identifier)              e.g., CPF, customer ID, hash, company + branch code
├── Quantitative (numeric)
│   ├── Discrete                  countable values: # of children, # of sales
│   └── Continuous                any real value: income, area, price
├── Qualitative (categorical)
│   ├── Nominal                   no order: department, city, gender
│   └── Ordinal                   natural order: satisfaction (Likert), Bronze < Silver < Gold
└── Auxiliary
    ├── Date/time                 used to filter, group, or derive (age, tenure, recency)
    ├── Text                      unstructured; summarized via word clouds, NLP
    └── Location                  address, ZIP, lat/long → derive distances
```

Key points:

- **Key variables** identify records. They often are **personal data** under data-protection laws (Brazil's LGPD, Law 13.709/2018; analogous to GDPR). Check the legal basis, store them securely, and consider **anonymization** before analytical use.
- **Discrete vs. continuous** depends on the **nature** of the information, *not* on rounding. Gasoline price is continuous even though it's stored with 2–3 decimals.
- **A variable ≠ a summary measure.** "Months since each customer's last policy" is a variable (varies across rows). "*Average* months since last policy" is a **summary measure** (a statistic / KPI).
- **Trap:** a segment coded as `1, 2, 3` is still **qualitative**. Numbers used as labels are categories, not quantities.

### Exercise (from class)

| Column | Type |
|---|---|
| CPF (Brazilian tax ID) | Key |
| Age | Quantitative continuous |
| Gender | Qualitative nominal |
| City of residence | Qualitative nominal |
| Monthly income | Quantitative continuous |
| Number of children | Quantitative discrete |

| Column | Type |
|---|---|
| Month | Date |
| Revenue | Quantitative continuous |
| # of sales | Quantitative discrete |
| # of stores | Quantitative discrete |
| Average ticket | Quantitative continuous |
| Target achieved ("< 80%", "80–100%", "> 100%") | Qualitative ordinal |

---

## 5. Uniqueness and completeness (missing values)

### 5.1 Uniqueness analysis

Check whether each entity appears **only once** — i.e., validate the grain. This is done on the **key field(s)**.

- Customer master table keyed by `CUSTOMER_ID` → each ID appears once.
- Orders table at *item* grain → the **combination** `ORDER_ID + PRODUCT_ID` appears once (each one alone can repeat).

Three kinds of duplicates on the main key:

| Case | What happens | What to do |
|---|---|---|
| **1. Fully identical rows** | All columns repeat | Keep one, drop the rest |
| **2. Same values, but some rows have blanks** | Some fields are filled in one row and empty in another | Keep the most complete row (or the one with the most relevant fields filled) |
| **3. Conflicting values** | At least one column disagrees | No universal rule. Investigate the source / pipeline. If impossible, apply a **business rule**: keep the row that brings the **lowest risk** if wrong |

In the case: ID 115 was fully duplicated (keep one); ID 327 had blanks in the second row (keep the first); ID 576 had two different prices (investigate, or keep the most conservative choice).

### 5.2 Completeness (fill-rate) analysis

Count how many rows have a value for each variable → equivalently, count the **missing values**.

Missing values can mean different things:

| # | Meaning | Informative? |
|---|---|---|
| 1 | Information unknown at the source — nothing can be done | Non-informative |
| 2 | Inconsistency in the data pipeline — can be recovered by reviewing extraction/calculation | Non-informative |
| 3 | **Not applicable** — the field *shouldn't* be filled | **Informative** |
| 4 | **Equivalent to a value** — e.g., blank means zero | **Informative** |

> 💡 Example: `COMERCIOS_RAIO_500M` blank may simply mean "zero shops nearby" (case 4) — or a failed geolocation (case 1/2). The *meaning* of the missing value decides the *treatment*.

### 5.3 How missing values are represented

In Python/pandas, empty cells become `NaN` ("not a number"). But sources often encode blanks as text: `"-"`, `"--"`, `"NA"`, `"N/A"`, `"Unknown"`, `"Desconhecido"`… pandas **won't** detect those automatically — you must know the data or look for them during EDA.

---

## 6. Univariate analysis

Analyze **one variable at a time**.

### 6.1 Frequency table

The most elementary EDA tool.

- **Absolute frequency:** count of observations in each category.
- **Relative frequency:** proportion (0–1, or 0–100%).

| Variable type | How to use a frequency table |
|---|---|
| Qualitative | Directly, one row per category |
| Quantitative discrete with few values | Directly, or with ranges (e.g., "4 or more") |
| Quantitative continuous | **Must** create ranges (bins), e.g., income 1–2, 2–3, 3–5, 5–10… minimum wages |

Case example: 52.5% of properties have *a lot* of sunlight, 35.7% *little*, 11.8% *none*. For nearby shops, the most common range is 10–20 (34.4%), and 16.3% had no information.

### 6.2 Summary measures (quantitative variables)

Two families:

**Measures of position (central tendency & location)**

| Measure | Meaning |
|---|---|
| **Mean** | Center of gravity of the values |
| **Median (Q2, P50)** | Cut point: 50% of values below, 50% above |
| **Mode** | Most frequent value — useful when values have low diversity / one dominant value |
| **Min / Max** | Extreme values |
| **Quartiles (Q1, Q3)** | 25% of values below Q1; 25% above Q3 |
| **Percentiles (Pk)** | k% of values below Pk (e.g., P40) |

Case (area, m²): mean 165, median 140, mode 130, min 40, max 602, Q1 100, Q3 210, P1 48, P99 ≈ 458.

**Mean vs. median.** The mean is heavily influenced by extreme values; the median is **robust**. Always compare both — if they differ a lot, the distribution is skewed and the median is usually the better "typical value" (e.g., income, house prices).

**Mean variants**

| Variant | When to use | Example |
|---|---|---|
| **Weighted mean** | Values have different weights | Grades 5, 8, 10 with weights 1, 2, 3 → (5·1 + 8·2 + 10·3) / 6 = **8.5** (simple mean = 7.7) |
| **Geometric mean** | Values combine **multiplicatively** (growth rates, returns) | Growth 2%, 4%, 50% → total = 1.02·1.04·1.50 = 1.591 (+59.1%). Geometric mean = 1.591^(1/3) = **1.167 → 16.7%/year** (simple mean would say 18.7% — wrong) |
| **Harmonic mean** | Values are **rates** (ratios) with equal numerators | 3 legs of 10 km at 20, 40, 60 km/h → 3 / (1/20 + 1/40 + 1/60) = **32.7 km/h** (simple mean says 40 — wrong) |

**Measures of dispersion (spread)**

| Measure | Formula / idea |
|---|---|
| **Variance** | Average of squared deviations from the mean: `s² = Σ(xᵢ − x̄)² / (n − 1)` |
| **Standard deviation** | `s = √s²` — typical deviation from the mean, in the original unit |
| **Mean absolute deviation** | Average of `|xᵢ − x̄|` — less sensitive to extremes than SD |
| **Median absolute deviation (MAD)** | Median of `|xᵢ − median|` — robust |
| **Coefficient of variation (CV)** | `CV = s / x̄` — dispersion relative to the level; unit-free |
| **Range** | `max − min` |
| **Percentile range** | e.g., `P99 − P1` — range that ignores the extremes |

Case (area): SD ≈ 91, mean abs. deviation ≈ 69, median abs. deviation 50, CV ≈ 55%, range 562, P1–P99 range ≈ 410.

**Why the coefficient of variation matters.** Salaries with SD = R$ 600 at every level: for an intern (mean R$ 3,000) CV = 20%; for a manager (mean R$ 21,000) CV ≈ 2.9%. Same SD, very different *relative* variability. Use CV to compare variability between groups with different means.

---

## 7. Outlier analysis

**Outliers** are unusual values — too high (upper) or too low (lower). They can be:

- **Legitimate (correct)** — naturally far from most values (a 105-year-old customer may be real).
- **Illegitimate (errors)** — should be corrected or removed (a supermarket item priced at R$ 1,000,000 when P99 is R$ 350).

Strategies to detect them:

1. List the ~5% smallest and ~5% largest values and look for a **jump**.
2. Compare **min vs. P1** and **max vs. P99**.
3. Compare **range vs. percentile range**.
4. Boxplot rule (see §9): values beyond `Q1 − 1.5·IQR` or `Q3 + 1.5·IQR`.

> ⚠️ Outlier detection is **subjective** — there is no universal rule. Domain knowledge decides whether a value is right or wrong.

Case: area max 602 m² vs. P99 458 m² → upper outliers, but plausible. Price max **R$ 296 million** vs. P99 ≈ R$ 2.9 million → almost certainly a data error.

---

## 8. Bivariate analysis

Evaluates **two variables together** — how they relate.

### 8.1 Two-way frequency table (contingency table / crosstab)

One variable on rows, the other on columns. Quantitative variables must be binned. Percentages can be computed in **three ways**, each answering a different question:

| Percentage base | Question answered |
|---|---|
| **Grand total** | What share of *all* claims is "Auto & Level 3"? |
| **Row total** | Among *Auto* claims, how are severities distributed? |
| **Column total** | Among *Level 3* claims, which product do they come from? |

> 💡 Choosing the right base is the whole point. "% of rows" compares profiles across row groups; "% of columns" compares across column groups.

### 8.2 Comparing summary measures by group

When one variable is quantitative, compute its summary measures **segmented by** the categories of the other. Example: purchase recency (mean, SD, min, max) for customers *with* vs. *without* the store card. Case: area (m²) by property type (house vs. apartment).

### 8.3 Charts

Scatter plots, grouped bars, side-by-side boxplots — see next section.

---

## 9. Data visualization

Goal: **simplify the communication** of results through visual elements.

Principles:
- **Clarity** — the message must be unambiguous.
- **Simplicity** — avoid elements that demand unnecessary effort.
- **Aesthetics** — harmonious colors, sizes, margins.

### 9.1 Visual effectiveness (Cleveland & McGill, 1984)

The human brain decodes some visual encodings more accurately than others:

| Effectiveness | Encodings |
|---|---|
| **High** | Position, length, direction |
| **Medium** | Angle, area, volume, curvature |
| **Low** | Color, pattern |

> Lower-effectiveness charts are not necessarily bad — it depends on the goal.

### 9.2 Chart catalog

**High effectiveness**

| Chart | Use it for |
|---|---|
| **Simple bar** | Frequencies by category; comparing a summary measure across groups; evolution of an indicator over a few periods |
| **Grouped bar** | Comparing frequencies or indicators between 2+ groups (e.g., open vs. click rate by campaign) |
| **Stacked bar** | Indicators that **can be summed** (revenue desktop + mobile), or frequency profiles of groups; can be normalized to 100% (share) |
| **Waterfall** | Positive and negative contributions to the variation of an indicator over time |
| **Line** | Evolution over time; better than bars when there are many points; compare few groups (avoid many crossings). Area fill is OK but the focus is the line direction |
| **Scatter** | Relationship between two quantitative variables; use transparency when points overlap |
| **Histogram** | Distribution of a quantitative variable in bins (bars touch → continuity). Shows center, symmetry, spread |
| **Boxplot** | Compact summary of the distribution; great to **compare distributions across groups** |

**Medium effectiveness**

| Chart | Notes |
|---|---|
| **Stacked area** | Like stacked bars, but compares areas (less precise); can be normalized to 100% |
| **Pie / donut** | Shares of a whole; poor when proportions are close or there are many slices; poor for comparing multiple distributions. Donut is slightly better (arc length) |
| **Bubble** | Scatter + a 3rd variable as bubble area; poor with many/overlapping bubbles |
| **Treemap** | Frequencies of many categories; hard area comparison, labels may become unreadable |

**Low effectiveness**

| Chart | Notes |
|---|---|
| **Heatmap** | Compares a numeric indicator across a grid of two dimensions using color |

**Modern alternatives (Python)**

| Chart | Alternative to | Advantage |
|---|---|---|
| Waffle plot | Pie | Easier to read proportions |
| Strip / swarm plot | Boxplot | Shows every point (swarm avoids overlap) |
| Violin plot | Boxplot | Smoothed density shape |
| Ridgeline plot | Histogram | Compare smoothed distributions of many groups |
| Raincloud plot | Boxplot + histogram | Combines density, boxplot and strip |
| Hexbin plot | Scatter | Shows density of overlap; fast for big data |

### 9.3 Histogram shapes: symmetry and skewness

- **Symmetric:** mirror image around a central line → **mean = median**.
- **Right-skewed (positive skew):** long tail to the right → **mean > median** (e.g., income, prices).
- **Left-skewed (negative skew):** long tail to the left → **mean < median**.

### 9.4 Boxplot anatomy

```
          o   o        ← potential outliers  (> Q3 + 1.5·IQR)
          |
     ─────┴─────       ← upper whisker: largest value ≤ Q3 + 1.5·IQR
     ┌─────────┐       ← Q3 (75th percentile)
     │         │
     ├─────────┤       ← Q2 = median
     │         │
     └─────────┘       ← Q1 (25th percentile)
     ─────┬─────       ← lower whisker: smallest value ≥ Q1 − 1.5·IQR
          o            ← potential outlier   (< Q1 − 1.5·IQR)

IQR (interquartile range) = Q3 − Q1
```

> ⚠️ Boxplot outliers must be judged carefully: in long-tailed distributions, perfectly normal values may be flagged.

---

## 10. Automated EDA

Libraries like `pandas`, `numpy`, `scipy`, `matplotlib` and `seaborn` give you all the building blocks, but every analysis must be coded by hand. **Automated EDA** packages generate a complete report. A popular one is **`ydata-profiling`** (formerly pandas-profiling): it produces an HTML report with tables, charts, missing values, duplicates, correlations, etc.

```python
from ydata_profiling import ProfileReport
ProfileReport(df, title="EDA report").to_file("report.html")
```

---

## 11. Pseudocode — a full EDA pipeline

```text
PROCEDURE ExploratoryDataAnalysis(table, data_dictionary, key_columns)

    # ---------- 1. Structure ----------
    PRINT number of rows and columns
    DEFINE grain ← "what one row represents" (e.g., one property)
    FOR each column c:
        type[c] ← classify as KEY | QUANT_DISCRETE | QUANT_CONTINUOUS
                               | QUAL_NOMINAL | QUAL_ORDINAL | DATE | TEXT | LOCATION
        (use data_dictionary; numbers used as labels → QUAL)

    # ---------- 2. Uniqueness ----------
    dup_keys ← rows whose key_columns appear more than once
    FOR each duplicated key k:
        rows ← all rows with key k
        IF all rows identical:              keep one
        ELSE IF rows differ only by blanks: keep the most complete row
        ELSE:                               investigate source;
                                            if impossible, keep lowest-risk row (business rule)

    # ---------- 3. Completeness ----------
    REPLACE textual blanks ("-", "NA", "N/A", "Unknown", ...) WITH NaN
    FOR each column c:
        missing_count[c] ← count of NaN in c
        missing_pct[c]   ← missing_count[c] / number of rows
        DECIDE meaning: unknown | pipeline error | not applicable | equals a value (e.g., 0)
        TREAT accordingly (keep as category "missing", impute, fix pipeline, fill with 0)

    # ---------- 4. Univariate ----------
    FOR each column c:
        IF type[c] is QUAL or QUANT_DISCRETE with few values:
            frequency_table(c)  → absolute and relative counts
        IF type[c] is QUANT:
            position   ← mean, median, mode, min, max, Q1, Q3, P1, P99
            dispersion ← std, mean_abs_dev, median_abs_dev, CV = std/mean,
                         range = max-min, pct_range = P99-P1
            IF |mean - median| large → note skewness; prefer median
            PLOT histogram and boxplot
            IF continuous: frequency_table(binned(c))

    # ---------- 5. Outliers ----------
    FOR each quantitative column c:
        INSPECT smallest 5% and largest 5% values for jumps
        COMPARE min vs P1, max vs P99, range vs pct_range
        IQR ← Q3 - Q1
        flag ← values < Q1 - 1.5*IQR OR values > Q3 + 1.5*IQR
        FOR each flagged value: decide LEGITIMATE (keep) or ERROR (fix / remove)

    # ---------- 6. Bivariate ----------
    FOR each relevant pair (x, y):
        IF both qualitative (or binned):
            crosstab(x, y) with % of total, % of rows, % of columns
            PLOT grouped or 100%-stacked bars
        IF one qualitative g, one quantitative q:
            summary measures of q grouped by g
            PLOT side-by-side boxplots
        IF both quantitative:
            PLOT scatter (with transparency); compute correlation

    # ---------- 7. Communicate ----------
    CHOOSE high-effectiveness charts (position/length) whenever possible
    WRITE business conclusions for each table/chart
END PROCEDURE
```

Python equivalents of the key steps:

```python
df.shape; df.dtypes; df.head()
df[df.duplicated(subset=key_cols, keep=False)]          # duplicated keys
df = df.replace(["-", "--", "NA", "N/A", "Unknown"], np.nan)
df.isna().sum(); df.isna().mean()                         # missing count / %
df[col].value_counts(); df[col].value_counts(normalize=True)
df[col].describe(percentiles=[.01, .25, .5, .75, .99])
pd.cut(df[col], bins=[...]).value_counts()               # binning
pd.crosstab(df.a, df.b, normalize="index")               # row %
df.groupby("type")["area"].agg(["mean", "median", "std", "min", "max"])
sns.histplot(df, x=col); sns.boxplot(df, x="type", y=col); sns.scatterplot(df, x=a, y=b, alpha=.3)
```

---

## 12. Interview questions

1. **What's the difference between the mean and the median, and when would you prefer the median?** The median is robust to outliers; prefer it for skewed variables (income, price, time-to-event).
2. **Your average growth rates are 2%, 4%, 50%. What's the average annual growth?** Use the geometric mean (≈16.7%), not the arithmetic mean.
3. **Two teams have the same standard deviation of salaries. Do they have the same variability?** Not necessarily in relative terms — compare the **coefficient of variation**.
4. **How do you treat missing values?** First understand their *meaning* (unknown, pipeline bug, not applicable, equals a value). Treatment follows meaning: fix the pipeline, fill with 0, create a "missing" category, impute, or drop.
5. **How do you detect outliers?** Percentile comparisons, jumps in sorted values, IQR rule (1.5·IQR), domain knowledge. Then decide legitimate vs. error.
6. **Is a customer segment coded 1/2/3 a quantitative variable?** No — it's qualitative (labels). Treating it as numeric would imply order and distance that don't exist.
7. **Why are pie charts often discouraged?** Humans compare angles/areas poorly; bars (length/position) are more accurate.
8. **What is the grain of a table and why does it matter?** What one row represents; wrong grain → duplicated or wrong aggregations and joins.
9. **What does a right-skewed histogram imply about mean and median?** Mean > median.
10. **What does "percentage by row" vs. "by column" mean in a crosstab?** Each answers a different conditional question: P(column | row) vs. P(row | column).

---

## 13. References

- Tukey, J. W. *Exploratory Data Analysis*. Addison-Wesley, 1977.
- Morettin, P. A.; Bussab, W. O. *Estatística Básica*. 10th ed. Saraiva, 2023.
- Anderson, D. R.; Sweeney, D. J.; Williams, T. A. *Statistics for Business and Economics*. Cengage.
- Peng, R. D.; Matsui, E. *The Art of Data Science*. Leanpub, 2015. <https://bookdown.org/rdpeng/artofdatascience>
- Cleveland, W. S.; McGill, R. "Graphical perception: theory, experimentation, and application to the development of graphical methods." *JASA*, 79(387), 1984.
- Wilke, C. O. *Fundamentals of Data Visualization*. O'Reilly, 2019. <https://clauswilke.com/dataviz>
- Cairo, A. *The Functional Art*. New Riders, 2012.
- Knaflic, C. N. *Storytelling with Data*. Wiley, 2015.
- Few, S. *Show Me the Numbers*. 2nd ed. Analytics Press, 2012.
- Ware, C. *Information Visualization: Perception for Design*. 4th ed. Morgan Kaufmann, 2021.

---

⬅️ [Back to index](../../README.md) · ➡️ Next: [1.2 — Introduction to Modeling](../1.2%20-%20Introduction%20to%20Modeling/README.md)
