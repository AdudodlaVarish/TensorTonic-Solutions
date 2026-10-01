import numpy as np

def rmsprop_step(
    w: list,
    g: list,
    s: list,
    lr: float = 0.001,
    beta: float = 0.9,
    eps: float = 1e-8,
) -> tuple[list, list]:
    """
    Returns (new_w, new_s) with the same shapes as the inputs.
    """

    w_ = np.array(w, dtype=float)
    g_ = np.array(g, dtype=float)
    s_ = np.array(s, dtype=float)

    s_ = beta * s_ + (1 - beta) * g_ ** 2
    w_ = w_ - lr * g_ / (np.sqrt(s_ + eps))

    return w_.tolist(), s_.tolist()