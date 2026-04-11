from logistic import LogisticRegression
import numpy as np

X = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [5, 6],
    [6, 7],
    [7, 8]
])

y = np.array([0, 0, 0, 1, 1, 1])

model = LogisticRegression(n_features=2, lr=0.1, epochs=5000)
model.fit(X, y)
X_test = np.array([
    [2, 2],
    [8, 9]
])

print(model.predict(X_test))
