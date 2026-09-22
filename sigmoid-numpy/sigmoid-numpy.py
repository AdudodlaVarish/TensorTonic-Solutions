import numpy as np

def sigmoid(x: list | float) -> np.ndarray | float:

    x_np = np.asarray(x)
    return 1 / (1 + np.exp(-1 * x_np))