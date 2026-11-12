# Neural Networks

> 📗 **Companion notes** for *The StatQuest Illustrated Guide to Machine Learning* (Josh Starmer), **Chapter 12**.
> Written in my own words with my own examples. Not part of the FIA course modules, so it lives in **Extra Topics**.

---

## 1. What a neural network does

A neural network fits a **flexible curve (or surface)** to data by adding together many simple bent shapes. Each "bent shape" comes from one node passing a weighted input through an **activation function**.

```
input x ──(w1, b1)──► [activation] ──(w3)──┐
       └──(w2, b2)──► [activation] ──(w4)──┴──(+ b3)──► output
         input layer     hidden layer              output layer
```

- **Weights (w)** stretch/flip the shapes; **biases (b)** shift them.
- The output is the **sum** of the scaled shapes plus a final bias.
- These weights and biases are the **parameters**, learned from data.

---

## 2. Activation functions

| Activation | Formula | Shape |
|---|---|---|
| **ReLU** | max(0, x) | Bent line: 0 for negatives, identity for positives. The most common choice in hidden layers |
| **SoftPlus** | log(1 + eˣ) | A smooth version of ReLU |
| **Sigmoid** | 1 / (1 + e⁻ˣ) | S-curve from 0 to 1 |

Without activations, stacking layers would just produce another straight line: the bends are what make the network flexible.

### Tiny forward pass (my own numbers, ReLU)

x = 0.5
- Node 1: 2.0·x − 0.5 = 0.5 → ReLU → 0.5
- Node 2: −3.0·x + 2.0 = 0.5 → ReLU → 0.5
- Output: 1.5·0.5 + 1.0·0.5 − 0.2 = **1.05**

---

## 3. Training: backpropagation

Goal: find the weights and biases that minimize a loss, e.g., the **SSR** between predictions and observed values.

1. Start with **random** weights (and biases often at 0).
2. **Forward pass:** compute predictions and the loss.
3. **Backward pass (backpropagation):** use the **chain rule** to compute how the loss changes with each parameter, starting from the output and moving backward through the layers.
4. **Gradient descent:** nudge every parameter against its derivative.
5. Repeat until the loss stops improving.

> 💡 Backpropagation is "just" the chain rule applied efficiently: the derivative for a weight deep in the network = (how the loss changes with the output) × (how the output changes with that node) × (how that node changes with the weight). See [Gradient Descent](../../01%20-%20Artificial%20Intelligence%20I/1.4%20-%20Validation%20and%20Regularization/Gradient%20Descent.md).

---

## 4. More inputs, more outputs, classification

- **Multiple inputs:** each hidden node gets a weighted sum of all inputs → the network bends a **surface** instead of a curve.
- **Multiple outputs:** one output node per class, producing a raw score for each class.
- **ArgMax:** pick the class with the highest score. Easy to read, but useless for training, because its derivative is 0 almost everywhere.
- **SoftMax:** turns raw scores into probabilities that are positive and sum to 1:

$$\text{softmax}(z_k) = \frac{e^{z_k}}{\sum_j e^{z_j}}$$

  Example: scores (2.0, 1.0, 0.1) → probabilities ≈ (0.66, 0.24, 0.10).

- **Cross entropy:** the loss for classification. For one observation it's **−log(probability assigned to the correct class)**. Predicting 0.66 for the right class costs −log(0.66) ≈ 0.42; predicting 0.10 costs ≈ 2.30. Its steep penalty for confident mistakes speeds up learning compared with SSR.

---

## 5. Practical points

| Point | Why |
|---|---|
| Scale the inputs | Gradient descent behaves much better |
| Use cross validation / a validation set | Networks overfit easily |
| Regularize (weight decay, dropout, early stopping) | Controls variance |
| Need lots of data | Many parameters to estimate |
| Harder to interpret | Use SHAP or similar to explain predictions |

---

## 6. Pseudocode (one hidden layer, regression)

```text
FUNCTION ReLU(z):  RETURN max(0, z)

FUNCTION Forward(x, W1, b1, w2, b2)
    h ← ReLU(W1 · x + b1)                 # hidden layer
    RETURN w2 · h + b2, h                 # output, hidden values

FUNCTION Train(X, y, n_hidden, η, epochs)
    W1 ← random small values; b1 ← 0; w2 ← random small values; b2 ← 0
    FOR epoch IN 1..epochs:
        FOR each (x, target) in shuffled data:          # SGD
            ŷ, h ← Forward(x, W1, b1, w2, b2)
            d_out ← -2 (target - ŷ)                       # dSSR/dŷ
            grad_w2 ← d_out · h;   grad_b2 ← d_out
            d_h ← d_out · w2 · 1[h > 0]                   # chain rule through ReLU
            grad_W1 ← outer(d_h, x);   grad_b1 ← d_h
            W1 -= η·grad_W1;  b1 -= η·grad_b1;  w2 -= η·grad_w2;  b2 -= η·grad_b2
    RETURN W1, b1, w2, b2

FUNCTION Softmax(z):  e ← exp(z - max(z));  RETURN e / Σ e
FUNCTION CrossEntropy(probs, true_class):  RETURN -log(probs[true_class])
```

---

## 7. Self-check

1. Why do we need activation functions? → Without them, any stack of layers is still a linear model.
2. What does backpropagation compute? → The derivative of the loss with respect to every weight and bias (via the chain rule).
3. Why SoftMax instead of ArgMax for training? → SoftMax is differentiable; ArgMax's derivative is 0 almost everywhere.
4. Cross entropy when the correct class gets probability 0.9? → −log(0.9) ≈ **0.105**.

---

🔗 Related: [Gradient Descent](../../01%20-%20Artificial%20Intelligence%20I/1.4%20-%20Validation%20and%20Regularization/Gradient%20Descent.md) · [Logistic Regression](../../02%20-%20Artificial%20Intelligence%20II/2.3%20-%20Classification%20-%20Logistic%20Regression/Logistic%20Regression.md)
