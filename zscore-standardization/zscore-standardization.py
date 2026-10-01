import numpy as np

def zscore_standardize(X: list, axis: int = 0, eps: float = 1e-12) -> np.ndarray:
    """
    Returns population Z-scores as a NumPy array matching the shape of X.
    """
    X_ = np.array(X, dtype=float)

    mean = np.mean(X_, axis=axis, keepdims=True)
    std = np.std(X_, axis=axis, keepdims=True)

    safe_std = np.maximum(std, eps)

    zscore = (X_ - mean) / safe_std

    return zscore