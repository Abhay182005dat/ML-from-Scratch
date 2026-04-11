import numpy as np
import pandas as pd

"""
Suppose  X -> (n_samples, n_features) and y -> (n_samples,) and w -> (n_features,)

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def initialise_parameters(n_features):
    weights = np.zeros(n_features)
    bias = 0
    return weights, bias

def compute_loss(y_true, y_pred):
    n_samples = len(y_true)
    epsilon = 1e-9
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    # L = −m1 ​∑[ y_true*log(y_pred) + (1−y_true)*log(1−y_pred) ]
    loss = - (1/n_samples) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss


dL/dw = dL/dy^ * dy^/dz * dz/dw
dL/db = dL/dy^ * dy^/dz * dz/db

def compute_gradients(X, y_true, y_pred):
    n_samples = len(y_true)
    
    dw = (1/n_samples) * np.dot(X.T, (y_pred - y_true))  # y_pred - y_true -> (n_samples,) and X -> (n_samples, n_features) 
    db = (1/n_samples) * np.sum(y_pred - y_true)        #  so we need to transpose X to get (n_features, n_samples) and the result will be (n_features,)
    
    return dw, db
"""
class LogisticRegression:
    def __init__(self,n_features, lr=0.01,epochs=100):
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.lr = lr
        self.epochs = epochs

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)
    
    def fit(self, X, y):
        n_samples = X.shape[0]
        for _ in range(self.epochs):
            y_pred = self.predict_proba(X)

            # clip for safety
            epsilon = 1e-9
            y_pred = np.clip(y_pred , epsilon , 1 - epsilon)

            loss =  -(1/n_samples) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
            print(loss)

            # gradients
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)

            # update
            self.weights  = self.weights - self.lr*dw
            self.bias = self.bias - self.lr*db

    def predict(self, X):
        y_pred = self.predict_proba(X)
        return (y_pred >= 0.5).astype(int)