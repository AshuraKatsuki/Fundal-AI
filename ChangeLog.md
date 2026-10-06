# Changelog

> Generated with AI assistance (Claude).

Changes since the version recorded in `CHANGES.md`.

---

## Cell 11 — `gradient_descent`

Reordered parameters and added a default for `eta`.

```python
# before
def gradient_descent(W, b, eta, gradient_W, gradient_b):

# after
def gradient_descent(W, b, gradient_W, gradient_b, eta=0.001):
```

Added a test call:

```python
print(gradient_descent(W, b, gradient_W, gradient_b, eta=0.001))
```

---

## Cell 13 — split ratio

```python
# before
train_test_split(X, Y, test_size=0.30, random_state=42)
train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# after
train_test_split(X, Y, test_size=0.985, random_state=42)
train_test_split(X_test, y_test, test_size=0.985, random_state=42)
```

Training set size: 42000 → 900.

---

## Cell 14 — shape check (new)

```python
print(y_train.shape)
print(X_train.shape)
```

---

## Cell 15 — `predict` (new)

```python
def predict(X, W, b, d):
    y_hat, _, _ = feed_forward(X, W, b, d)
    index = np.argmax(y_hat, axis=0)
    return np.eye(y_hat.shape[0])[index].T

print(predict(X.T, W, b, d))
```

Returns one-hot predictions with shape (10, N).

---

## Cell 16 — `accuracy` (new)

```python
def accuracy(y_hat, Y):
    accuracy = np.mean(y_hat == Y)*100
    return (f"Accuracy: {accuracy:.2f}%")

print(accuracy(y_hat, Y.T))
```

---

## Cell 17 — `train`

Added per-epoch loss and accuracy reporting:

```python
print(f"Epoch {epoch+1},", "loss:", calculate_loss(y_train, y_hat, epsilon=1e-10))

Y_hat = predict(X.T, W, b, d)
print("Accuracy on train", accuracy(Y_hat, Y.T))
```

Added a docstring describing inputs and outputs.

Changed the number of epochs in the call: 5 → 10.

```python
W, b = train(X_train, y_train, W, b, eta=0.01, epochs=10)
```

---

## Unchanged

Cells 0–10, 12, 18.
