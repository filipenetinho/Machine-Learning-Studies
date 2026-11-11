# 2.6 — Segmentation Models (Cluster Analysis)

> **Module:** Artificial Intelligence II
> **Case studies:** (1) segmenting **10 job candidates** by experience and number of courses (worked by hand); (2) synthetic shapes — blobs, interlocking moons, concentric rings; (3) **European eating habits** (25 countries, 9 food groups); (4) **e-commerce cross-sell** with mixed variables (Gower distance).

---

## Table of Contents

1. [Unsupervised learning and the goal of clustering](#1-unsupervised-learning-and-the-goal-of-clustering)
2. [Applications](#2-applications)
3. [Understanding the problem and choosing variables](#3-understanding-the-problem-and-choosing-variables)
4. [Distance measures: Euclidean and Gower](#4-distance-measures-euclidean-and-gower)
5. [Standardization](#5-standardization)
6. [Hierarchical clustering](#6-hierarchical-clustering)
7. [Partition algorithms: k-means](#7-partition-algorithms-k-means)
8. [DBSCAN](#8-dbscan)
9. [Comparing the algorithms](#9-comparing-the-algorithms)
10. [Additional case studies](#10-additional-case-studies)
11. [Pseudocode](#11-pseudocode)
12. [Interview questions](#12-interview-questions)
13. [References](#13-references)

---

## 1. Unsupervised learning and the goal of clustering

In supervised models the target is known and guides the fit. In **unsupervised** learning there is **no target**: the goal is to find **structure in the variables themselves**.

| Use | What it does |
|---|---|
| **Segmentation (clustering)** | Group similar observations (e.g., customers with similar behavior) — **this module** |
| **Dimensionality reduction** | Summarize many variables into a few new ones that keep most of the information (e.g., PCA) |
| **Outlier detection** | Identify observations that deviate from the general pattern, considering many variables at once |

**Goal of cluster analysis:** group observations so that
- **within each cluster**, observations are **homogeneous** (minimum variability), and
- **between clusters**, groups are **heterogeneous** (maximum variability).

```
   ●●●        ▲▲▲          within-cluster variation  → small
  ●●●●●      ▲▲▲▲▲         between-cluster variation → large
   ●●●        ▲▲▲
         ■■■
        ■■■■■
         ■■■
```

---

## 2. Applications

| Application | Example | Area |
|---|---|---|
| **Customer portfolio tiers** | Bank customers grouped by transactions, product holdings and investments → differentiated service levels | Banking |
| **Eating habits** | Group regions of a country by similarity of diet | Health & nutrition |
| **Genetic sequencing** | Group DNA sequences by length and frequency of patterns to understand gene structure/function | Medicine |
| **Personalized communication** | Segment by socio-demographic profile for relationship marketing | Marketing & CRM |
| **Customer recognition (RFM)** | Group retail customers by **Recency, Frequency, Monetary value** → tailored recognition/relationship strategies | Marketing & CRM |

---

## 3. Understanding the problem and choosing variables

**Case — customer recognition:** create 4 clusters of retail customers by purchase frequency and value. A simple business rule proposed **4 quadrants** (above/below the mean of each axis).

> Is that adequate? **A purely business-rule segmentation doesn't always give the best "rule" for grouping similar observations** — quadrant borders can split natural groups and put dissimilar customers together. → use **statistical methods**.

**How do I choose the variables? Is there a method to select the most important ones, like in regression?**

> *Is a tiger more similar to a cat or to a lion?* It depends on the variables: by size, to the lion; by having stripes… neither; by being a feline, all three. **The choice of variables defines the notion of similarity.**

- Being unsupervised, there's **no statistical criterion** for "important" variables (no target to explain).
- An essential step is **defining the variables that best reflect the purpose** of the segmentation.
- In companies, this is done jointly by **data science and the business area**.

---

## 4. Distance measures: Euclidean and Gower

Once variables are defined, we need to quantify **how similar two observations are** → **distance measures**.

### 4.1 Euclidean distance

Based on the **Pythagorean theorem**; for 2+ **quantitative** variables:

$$d(i, j) = \sqrt{\sum_{v=1}^{p}(x_{iv} - x_{jv})^2}$$

Smaller distance → more similar.

### Case: candidate evaluation

A tech recruiter wants to segment 10 candidates for the hiring manager using (1) **years of experience** and (2) **# post-graduation courses/specializations**.

| Candidate | Experience (yrs) | Courses |
|---|---|---|
| Ana | 9 | 9 |
| Beatriz | 3 | 4 |
| Carlos | 10 | 7 |
| Fernando | 8 | 2 |
| João | 1 | 3 |
| Mariana | 11 | 1 |
| Paula | 4 | 3 |
| Pedro | 9 | 0 |
| Ronaldo | 2 | 5 |
| Sueli | 12 | 8 |

- Ronaldo ↔ Sueli: √((12−2)² + (8−5)²) = √(100 + 9) ≈ **10.4**
- Ronaldo ↔ Ana: √((9−2)² + (9−5)²) = √(49 + 16) ≈ **8.1**
- Ronaldo ↔ Beatriz: √((3−2)² + (5−4)²) = √2 ≈ **1.4** → Ronaldo is most similar to Beatriz.

Visually there seem to be **3 groups**:
- **Group 1 — experienced & highly trained:** Ana, Carlos, Sueli
- **Group 2 — junior:** Beatriz, João, Paula, Ronaldo
- **Group 3 — experienced, few courses:** Fernando, Mariana, Pedro

**Distance matrix:** all pairwise distances. The **diagonal is 0** (distance of an observation to itself) and the matrix is **symmetric** (d(i,j) = d(j,i)). Small distances inside groups, larger between groups.

### 4.2 Gower distance

Works with **quantitative and qualitative** variables together and is **scaled to [0, 1]** (0 = identical).

Partial similarity between i and j for each variable k:

$$\text{quantitative: } s_{ij}^{(k)} = 1 - \frac{|X_i - X_j|}{\max(X) - \min(X)}
\qquad
\text{qualitative: } s_{ij}^{(k)} = \begin{cases}1 & \text{if } Y_i = Y_j\\ 0 & \text{if } Y_i \neq Y_j\end{cases}$$

Gower distance (average over the p variables):

$$D_{ij} = 1 - \frac{1}{p}\sum_{k=1}^{p} s_{ij}^{(k)}$$

---

## 5. Standardization

Variables on different scales get different weights in distance calculations.

Example: obs. 1 = (2, 20), obs. 2 = (7, 70) → √(5² + 50²) = √(25 + 2500) ≈ **50.2** — dominated by X₂, which has a larger scale.

**Z-score** (most common):

$$X_{std} = \frac{X - \text{mean}(X)}{\text{sd}(X)}$$

→ mean 0, standard deviation 1.

Candidate case (experience mean 6.9, sd 4.0; courses mean 4.2, sd 3.0):

| Candidate | Exp. std | Courses std |
|---|---|---|
| Ana | (9 − 6.9)/4.0 = 0.5 | (9 − 4.2)/3.0 = 1.6 |
| Beatriz | −1.0 | −0.1 |
| Carlos | 0.8 | 0.9 |
| Fernando | 0.3 | −0.7 |
| João | −1.5 | −0.4 |
| Mariana | 1.0 | −1.1 |
| Paula | −0.7 | −0.4 |
| Pedro | 0.5 | −1.4 |
| Ronaldo | −1.2 | 0.3 |
| Sueli | 1.3 | 1.3 |

Then recompute the distance matrix on the standardized variables.

**Considerations:**
- Even if variables have similar scales, **always standardize** quantitative variables — don't be hostage to scale differences.
- To **deliberately weight** a variable, multiply it by a factor **after** standardizing: factor > 1 increases its influence; between 0 and 1 decreases it. (E.g., distance 5 becomes 10 when the axis is doubled, 2.5 when halved.)

---

## 6. Hierarchical clustering

### Algorithm (agglomerative)

Repeat two simple steps:
1. Find the **two closest elements** in the distance matrix.
2. **Merge** them into a single element and **recalculate the distance matrix** using the chosen **linkage criterion**. Go back to step 1.

It starts with every observation on its own (no clusters) and ends with **all observations in a single cluster**. Along the way it produces many intermediate scenarios with different numbers of clusters — we choose one using the **dendrogram**.

### Dendrogram

A tree diagram summarizing the sequence of merges vs. the **distance at which each merge happened**.

```
distance
  │            ┌─────────────┐            ← big jump: merging the last 2 clusters
  │            │             │
  │            │             │            ← "cut" here → 2 clusters
  │        ┌───┴───┐     ┌───┴───┐
  │      ┌─┴─┐   ┌─┴─┐  ┌┴─┐   ┌─┴─┐
  │      A   B   C   D  E  F   G   H
  └───────────────────────────────────
```

Reading **bottom-up**, look for a **region of stability**: the ideal number of clusters is the one **just before a merge that requires a large increase in distance**. That's the statistically optimal scenario — but other scenarios may be adopted if they're more interpretable for the business.

### Linkage criteria

After merging E and F, what's the distance from D to the cluster {E, F}? With d(D,E) = 3 and d(D,F) = 2:

| Linkage | Rule | New d(D, {E,F}) |
|---|---|---|
| **Complete** | **Largest** distance between members | max(2, 3) = **3** |
| **Single** | **Smallest** distance | min(2, 3) = **2** |
| **Average** | **Mean** distance | mean(2, 3) = **2.5** |
| *(extra)* **Ward** | Merge that **least increases total within-cluster variance** | — |

> Single linkage tends to "chain" (long, snake-like clusters — good for odd shapes, bad for compact groups); complete and Ward produce compact, similar-size clusters.

### Considerations

- **Computationally expensive**: compares many distances at each step. Usually not feasible beyond a few tens of thousands of observations. Estimated RAM: 10k obs ≈ **800 MB**; 20k ≈ **3 GB**; 30k ≈ **7 GB**; 40k ≈ **13 GB** (the distance matrix grows with n²).
- Being **sequential**, you **can't** fit it on part of the data and "apply" it to classify the rest.
- → Recommended for **small datasets**; for larger ones, use **partition** methods.

### Case: candidates

Hierarchical with Euclidean distance and several linkages: **all except single** found the same **3 clusters** identified visually. Final step: **profile the clusters** with EDA (means of each variable per cluster). Which cluster should the recruiter pick? Depends on the profile sought, the salary budget, etc.

### Procedure summary

1. **Standardize** quantitative variables.
2. Compute the **distance matrix** with an appropriate measure.
3. Run the algorithm testing **one or more linkages**.
4. Inspect the **dendrograms** and pick one or more stable scenarios (e.g., complete with 4 clusters, average with 3).
5. For each scenario, **profile and interpret** the clusters. There's no single right answer — different groupings can be good depending on interpretability.

---

## 7. Partition algorithms: k-means

Partition algorithms require defining **k (number of clusters) in advance**. The most used is **k-means**, for quantitative variables, based on Euclidean distance.

### Steps

1. **Initialize** k random reference points (**centroids**) in the space.
2. **Assign** every observation to the **nearest centroid** (smallest Euclidean distance).
3. **Update** each centroid as the **mean** of the observations assigned to it (a "correction" of the random starting points).
4. **Repeat** steps 2–3 until **no observation changes cluster** (stability).

```
iteration 0:  ✕ random centroids         iteration 1: assign → move centroids to means
iteration 2:  some points switch cluster  …  iteration t: no point changes → converged
```

Because the random initialization affects the result, the process is **repeated with different starting centroids** and the solution with **highest within-cluster homogeneity** (lowest within-cluster sum of squares, *inertia*) is kept. (*extra:* scikit-learn uses smart **k-means++** initialization and `n_init` restarts.)

k-means minimizes:

$$\text{WCSS} = \sum_{c=1}^{k}\sum_{i\in c}\lVert x_i - \mu_c\rVert^2$$

### Case: candidates

k-means with k = 2, 3, 4, 5 → **k = 3 looks ideal**: k = 2 merges very heterogeneous candidates; k = 4 or 5 split similar ones.

### Considerations

- **Much faster** than hierarchical — it doesn't compare all pairs (no full distance matrix) → works on **large datasets**.
- Downside: **k must be set beforehand**. If you have no idea, run a **hierarchical clustering first** (on a random subsample if needed) and use the dendrogram(s) to get initial guesses for k.
- *(extra)* Other common tools for choosing k: **elbow method** (WCSS vs. k) and **silhouette score**.
- Unlike hierarchical, k-means **can score new observations** (assign them to the nearest centroid).

---

## 8. DBSCAN

### Limitation of k-means

k-means works best when the real clusters are **elliptical / globular** (ellipses in 2-D, ellipsoids in 3-D).

| Shape | k-means |
|---|---|
| Blobs | Identifiable ✅ |
| Interlocking moons | Less identifiable ⚠️ |
| Concentric rings | Not identifiable ❌ |

### The algorithm

**DBSCAN — Density-Based Spatial Clustering of Applications with Noise** groups observations into clusters of **arbitrary shape**, finding **high-density regions separated by low-density regions**. It **doesn't require k**.

1. Define a **maximum distance** (`eps`) delimiting each observation's neighborhood, and a **minimum number of observations** (`min_samples`) to form a cluster.
2. Where at least `min_samples` observations lie within `eps`, a cluster forms and **expands** while it keeps finding nearby observations.
3. Observations that belong to no cluster are labeled **noise** — a distinctive feature of DBSCAN.

*(extra vocabulary)* **Core point:** ≥ `min_samples` neighbors within `eps`. **Border point:** within `eps` of a core point but not core itself. **Noise:** neither.

### Hyperparameters

| Hyperparameter | Meaning | Usual values |
|---|---|---|
| `eps` | Max distance for one observation to be in another's neighborhood | (0, 5] with **z-score-standardized** variables |
| `min_samples` | Min. # observations in the neighborhood to form a cluster | 2 – 30, depending on dataset size |

**Smaller values → more complex segmentation (more clusters).** No hyperparameter choice is indisputably superior — in an unsupervised problem there's **no error to minimize** → **interpretation** of the clusters is key.

`DBSCAN(eps=..., min_samples=...).fit_predict(X)`

### Results on the synthetic shapes

| Dataset | Result |
|---|---|
| **Blobs** | DBSCAN (eps 0.60, min_samples 25) ≈ k-means (k = 2) — differ only on border points, and DBSCAN marks some points as noise. Hierarchical (complete, k = 2) wrongly merged the right part of the upper blob (47 obs.) into the lower one |
| **Moons** | Hierarchical **single** (k = 2) captured the shape best; DBSCAN (eps 0.82, min_samples 61) close; **k-means the least appropriate** |
| **Rings** | Hierarchical **single** best. k-means and DBSCAN (eps 0.10, min_samples 2) failed. DBSCAN eps 0.45 / min 20: found the inner ring but split the outer into 4. **eps 0.50 / min 20: both rings perfectly** |

### Practical interpretation

In practice we often care about clusters with **different behavior in summary measures**, not necessarily the geometrically "ideal" clusters. In the rings example, k-means gives two clusters with **different means** (≈ (0.35, 0.33) vs. (−0.36, −0.33)), while the "correct" rings have the **same mean (0, 0)** and differ only in **spread** (sd 0.71 vs. 0.36). Which one is more useful depends on the business question.

### Considerations

- The most useful DBSCAN segmentation may depend on the **EDA of the resulting clusters**, not their geometric shape.
- DBSCAN is **optimized for large datasets** — usually feasible for a few hundred thousand observations (depends on # variables and hardware).
- **Not recommended for qualitative data**: its theory relies on density/proximity notions grounded in quantitative data.

---

## 9. Comparing the algorithms

| | Hierarchical | k-means | DBSCAN |
|---|---|---|---|
| Need k in advance? | No (choose via dendrogram) | **Yes** | No |
| Cluster shapes | Depends on linkage (single → arbitrary; complete/average/Ward → compact) | Globular / elliptical | **Arbitrary** |
| Noise / outliers | Assigned to clusters | Assigned (and they pull centroids) | **Labeled as noise** |
| Data size | Small (≤ tens of thousands; RAM ∝ n²) | **Large** | Large (hundreds of thousands) |
| Score new observations? | No | **Yes** (nearest centroid) | Not natively |
| Variable types | Any, given a suitable distance (e.g., **Gower** for mixed) | Quantitative (Euclidean) | Quantitative |
| Key settings | Distance, linkage, cut | k, initialization, restarts | eps, min_samples |

---

## 10. Additional case studies

### European eating habits
25 European countries, annual per-capita consumption indices of 9 food groups: red meat, white meat, eggs, milk, fish, cereals, carbohydrates (starchy foods), grains/pulses, fruits & vegetables. Goal: group countries with similar diets.

Exercise: (a) brief EDA; (b) standardize; (c) Euclidean distance matrix; (d) hierarchical with 3 linkages — which method and how many clusters from the dendrograms?; (e) which country is most similar to Italy? to Romania? to the Netherlands?; (f) profile clusters; (g) k-means with k = 2–5 — which k?; (h) profile; (i) main differences between hierarchical and k-means solutions.

### E-commerce cross-sell (mixed variables)
2,992 customers who bought **exactly one** electronic product (phone, TV or computer). Segment into **3–5 profiles** by **product category** (qualitative), **amount paid** (R$) and **satisfaction score** for personalized cross-sell.

Exercise: (a) EDA; (b) **Gower** distance matrix; (c) hierarchical clustering on the Gower matrix for k = 3, 4, 5 and profile each scenario; (d) which scenario is best for cross-sell actions, and why?

---

## 11. Pseudocode

### 11.1 Distances and standardization

```text
FUNCTION ZScore(X):  FOR each column v:  X[:, v] ← (X[:, v] - mean(X[:, v])) / sd(X[:, v]);  RETURN X

FUNCTION Euclidean(a, b):  RETURN sqrt( Σ_v (a[v] - b[v])² )

FUNCTION Gower(a, b, types, ranges)
    s ← 0
    FOR each variable k:
        IF types[k] = quantitative:  s ← s + 1 - |a[k] - b[k]| / ranges[k]      # range = max - min
        ELSE:                        s ← s + (1 IF a[k] = b[k] ELSE 0)
    RETURN 1 - s / p

FUNCTION DistanceMatrix(X, dist):  D[i, j] ← dist(X[i], X[j]) for all i, j   # D[i,i] = 0, symmetric
```

### 11.2 Agglomerative hierarchical clustering

```text
FUNCTION Hierarchical(D[n×n], linkage ∈ {single, complete, average})
    clusters ← { {1}, {2}, ..., {n} }
    merges ← []
    WHILE |clusters| > 1:
        (A, B) ← pair of clusters with minimum distance in D
        merges.append( (A, B, D[A, B]) )                   # height in the dendrogram
        C ← A ∪ B
        FOR each other cluster X:
            D[C, X] ← min(D[A,X], D[B,X])                     IF linkage = single
                      max(D[A,X], D[B,X])                     IF linkage = complete
                      (|A|·D[A,X] + |B|·D[B,X]) / (|A|+|B|)   IF linkage = average
        remove A, B from clusters (and D); add C
    RETURN merges                                              # → dendrogram

FUNCTION CutDendrogram(merges, k):  undo the last (k - 1) merges → k clusters
# choose k just before the largest jump in merge height (stability region)
```

### 11.3 k-means

```text
FUNCTION KMeans(X[n×p], k, n_init = 10, max_iter = 300)
    best ← NONE
    REPEAT n_init times:
        μ ← k random observations (or k-means++ seeding)
        REPEAT up to max_iter:
            FOR i IN 1..n:  label[i] ← argmin_c ||X[i] - μ[c]||²          # assignment
            μ_new[c] ← mean of X[label = c]   for each c                   # update
            IF no label changed (or μ_new ≈ μ): BREAK
            μ ← μ_new
        WCSS ← Σ_i ||X[i] - μ[label[i]]||²
        IF WCSS < best.WCSS: best ← (label, μ, WCSS)
    RETURN best

FUNCTION KMeansPredict(μ, x_new):  RETURN argmin_c ||x_new - μ[c]||²
```

### 11.4 DBSCAN

```text
FUNCTION DBSCAN(X, eps, min_samples)
    label[i] ← UNVISITED for all i;  c ← 0
    FOR each point i:
        IF label[i] ≠ UNVISITED: CONTINUE
        N ← { j : dist(X[i], X[j]) ≤ eps }                   # neighborhood (includes i)
        IF |N| < min_samples:  label[i] ← NOISE;  CONTINUE
        c ← c + 1;  label[i] ← c                               # i is a core point → new cluster
        queue ← N \ {i}
        WHILE queue not empty:
            j ← pop(queue)
            IF label[j] = NOISE: label[j] ← c                  # border point
            IF label[j] ≠ UNVISITED: CONTINUE
            label[j] ← c
            N_j ← { m : dist(X[j], X[m]) ≤ eps }
            IF |N_j| ≥ min_samples: queue ← queue ∪ N_j        # j is core → keep expanding
    RETURN label                                               # NOISE = -1 in scikit-learn
```

### 11.5 Full segmentation workflow

```text
PROCEDURE SegmentationProject(data)
    variables ← chosen jointly with the business (they DEFINE similarity)
    IF all quantitative:  X ← ZScore(data[variables]);  optionally multiply columns by weights
    ELSE:                 D ← Gower distance matrix
    IF n small:  run Hierarchical with several linkages → dendrograms → candidate k's
    IF n large:  run Hierarchical on a random subsample to guess k; then KMeans for k candidates
    IF shapes non-globular or noise matters:  run DBSCAN over a grid of (eps, min_samples)
    FOR each candidate solution:
        profile clusters: size, mean/median of each variable, category shares
        name the segments; check business usefulness and actionability
    CHOOSE the solution that is homogeneous within, heterogeneous between, AND interpretable
```

Python reference:

```python
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import KMeans, DBSCAN
import gower                                           # pip install gower

Xs = StandardScaler().fit_transform(df[cols])
Z = linkage(Xs, method="complete", metric="euclidean"); dendrogram(Z, labels=df["name"].values)
labels_h = fcluster(Z, t=3, criterion="maxclust")
km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(Xs); km.labels_, km.inertia_
db = DBSCAN(eps=0.5, min_samples=20).fit_predict(Xs)  # -1 = noise
Dg = gower.gower_matrix(df[["CATEGORIA", "VALOR_PAGO", "NOTA_SATISFACAO"]])
from scipy.spatial.distance import squareform
Zg = linkage(squareform(Dg, checks=False), method="average")
df.groupby(labels_h)[cols].agg(["mean", "median", "count"])   # cluster profiling
```

---

## 12. Interview questions

1. **Supervised vs. unsupervised?** Unsupervised has no target; it finds structure (clusters, low-dimensional representations, anomalies).
2. **What makes a good clustering?** High within-cluster homogeneity, high between-cluster heterogeneity, and business interpretability/actionability.
3. **Why standardize before clustering?** Distances are dominated by large-scale variables otherwise.
4. **How do you cluster mixed numeric and categorical data?** Gower distance + hierarchical (or k-prototypes / k-medoids).
5. **Explain k-means.** Initialize k centroids → assign points to nearest centroid → recompute centroids as means → repeat until stable; restart several times and keep the lowest WCSS.
6. **How do you choose k?** Dendrogram from hierarchical clustering, elbow (WCSS vs. k), silhouette, and business interpretability.
7. **Limitations of k-means?** Need k; assumes spherical/similar-size clusters; sensitive to initialization, scale and outliers; Euclidean/quantitative only.
8. **Single vs. complete vs. average linkage?** Min / max / mean inter-cluster distance; single chains (arbitrary shapes), complete gives compact clusters, average is in between.
9. **How do you read a dendrogram?** Cut where the next merge requires a large jump in distance (stability region).
10. **When would you use DBSCAN?** Arbitrary-shaped clusters, presence of noise/outliers, unknown k; needs quantitative, scaled data and tuning of eps/min_samples.
11. **Why can't hierarchical clustering scale?** It needs the full n×n distance matrix (O(n²) memory, ≥ O(n²) time) and can't score new points.
12. **Is a business-rule segmentation (quadrants) wrong?** Not wrong, but it ignores the data's natural structure; statistical clustering groups truly similar observations.
13. **How do you validate clusters without labels?** Internal metrics (silhouette, WCSS), stability across samples/seeds, profiling, and business validation (do the segments respond differently to actions?).

---

## 13. References

- Härdle, W. K.; Simar, L. *Applied Multivariate Statistical Analysis*. 4th ed. Springer, 2014.
- Johnson, R. A.; Wichern, D. W. *Applied Multivariate Statistical Analysis*. 6th ed. Pearson Prentice Hall, 2007.
- Timm, N. H. *Applied Multivariate Analysis*. Springer-Verlag, 2002.
- Zelterman, D. *Applied Multivariate Statistics with R*. Springer, 2015.
- Ester, M. et al. "A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise." *KDD*, 1996.

---

⬅️ Previous: [2.5 — Random Forest and Boosting (Classification)](../2.5%20-%20Classification%20-%20Random%20Forest%20and%20Boosting/README.md) · 🏠 [Index](../../README.md) · 🎯 [Interview cheat sheet](../../Interview%20Cheat%20Sheet.md)
