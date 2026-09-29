# Changes from `NeuralNetwork_origin.ipynb`

> Generated with AI assistance (Claude).

---

## Cell 0 — imports

Added:

```python
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
```

---

## Cell 7 — `softmax`

```python
# before
return np.exp(z) / np.sum(np.exp(z))

# after
return np.exp(z) / np.sum(np.exp(z), axis=0, keepdims=True)
```

Normalize per column instead of over the whole matrix. Each column is one sample
and needs its own probability distribution summing to 1.

---

## Cell 8 — `feed_forward`

```python
# before
z = np.dot(W[0].T, a)

# after
z = np.dot(W[0].T, a) + b[0]
Z.append(z)
```

Added bias to layer 1 — `b[0]` was never used before. Added `Z.append(z)` so `Z`
has all 4 entries; without it the list was off by one and backprop read the wrong
`Z[l-1]`.

Removed `print("shape of a:", a.shape)` from the loop.

---

## Cell 9 — `calculate_loss`

```python
# before
def calculate_loss(Y, Y_hat, epsilon=1e-10):
    N = Y.shape[1]

# after
def calculate_loss(y, y_hat, epsilon=1e-10):
    N = y.shape[1]
```

Renamed parameters to lowercase so they no longer shadow the global `Y`.
Added `print(J)`.

---

## Cell 12 → Cell 10 — `backpropagation`

Rewritten.

```python
# before
def backpropagation(J, W, A, y, Z):
    gradient_W = []
    gradient_b = []
    e_l = A[-1] - y
    for l in range(len(A), 0, -1):
        gradient_W.append(np.dot(A[l-1], e_l))
        gradient_b.append(e_l)
        e_l = np.dot(W[l], e_l) & np.astype(Z[l-1], np.int8)

backpropagation(J, W, A, Y[0], Z)
```

```python
# after
def backpropagation(W, A, y, Z):
    gradient_W = []
    gradient_b = []
    e_l = (A[-1] - y) / y.shape[1]
    for l in range(len(W)-1, -1, -1):
        gradient_W.append(np.dot(A[l], e_l.T))
        gradient_b.append(np.sum(e_l, axis=1, keepdims=True))

        if l > 0:
            e_l = np.dot(W[l], e_l) * (Z[l-1] > 0)
    gradient_W.reverse()
    gradient_b.reverse()
    return gradient_W, gradient_b
```

What changed and why:

- Dropped the `J` parameter — backprop never uses the loss value.
- Divide by N in `e_l`, to stay consistent with the loss.
- `range(len(W)-1, -1, -1)` instead of `range(len(A), 0, -1)`. `len(A)` is 5 but
  `W` only has indices 0–3.
- `A[l]` instead of `A[l-1]`. Python indexes from 0, the reference material from 1.
- Added `.T` — the formula is `A^(l-1) · E^(l)T`.
- `np.sum(e_l, axis=1, keepdims=True)` instead of `e_l`. Bias has shape (d, 1),
  so the per-sample errors must be summed.
- `*` instead of `&`. `&` is bitwise AND, not element-wise multiply.
- `(Z[l-1] > 0)` instead of `np.astype(Z[l-1], np.int8)`. `astype` just truncates
  decimals; it is not the ReLU derivative.
- `W[l]` instead of the hardcoded `W[1]`.
- Added `if l > 0`. At l = 0, `Z[-1]` points to the last element of the list,
  which has the wrong shape.
- Added `.reverse()`. The loop runs backwards, so `append` left the list in
  reverse order.
- Added a `return` statement — the original returned nothing.

---

## Cell 11 — `gradient_descent` (new)

```python
def gradient_descent(W, b, eta, gradient_W, gradient_b):
    for l in range(len(W)):
        W[l] -= eta * gradient_W[l]
        b[l] -= eta * gradient_b[l]
    return W, b
```

---

## Cell 13 — train/val/test split (new)

```python
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
X_train, X_val, X_test = X_train.T, X_val.T, X_test.T
y_train, y_val, y_test = y_train.T, y_val.T, y_test.T
```

Split on `X` in (60000, 784) form, then transpose to (784, N).
A 70/15/15 split gives 42000 / 9000 / 9000.

---

## Cell 15 — `train` (new)

```python
def train(X_train, y_train, W, b, eta=0.01, epochs=5, d=[784, 512, 128, 32, 10]):
    for epoch in range(epochs):
        X_train, y_train = shuffle(X_train.T, y_train.T)
        X_train = X_train.T
        y_train = y_train.T
        for i in range(X_train.shape[1]):
            y_hat, A, Z = feed_forward(X_train, W, b, d)
            g_W, g_b = backpropagation(W, A, y_train, Z)
            W, b = gradient_descent(W, b, eta, g_W, g_b)
    return W, b
```

---

## Removed cells

- `print(A)`
- `print(Z)`
- `print(Y[:,].shape)`
- the `np.astype(a > 0, np.int8)` scratch test

---

## Unchanged

Cells 1–6 (loading MNIST, plotting samples, flattening, normalizing, one-hot
encoding) and the final `onehot_to_label` cell.
