import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    n_samples, n_features = X.shape
    #(n_features, 1)
    w = np.zeros((n_features, ))
    b = 0.0

    for _ in range(steps):

      #(n_samples, 1)
      logits = X @ w + b
      probabilities = _sigmoid(logits)

      #(n_samples, 1)
      dLdlogit = probabilities - y 
      #(n_samples, n_features)
      dLogitdw = X

      #(n_features, 1)
      dLdw = dLogitdw.T @ dLdlogit / n_samples
      dLdb = np.mean(dLdlogit)

      w = w - lr * dLdw
      b = b - lr * dLdb

    return (w, b)
