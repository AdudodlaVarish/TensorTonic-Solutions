import numpy as np

def expected_value_discrete(x: list, p: list) -> float:

    x_ = np.array(x).reshape(-1, 1)
    p_ = np.array(p).reshape(-1, 1)

    return (x_.T @ p_)[0,0].astype(float)

    