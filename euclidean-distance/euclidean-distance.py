import numpy as np

def euclidean_distance(x: list, y: list) -> float:

    _x = np.array(x)
    _y = np.array(y)

    distance = np.sqrt(np.sum((_x - _y) ** 2))
    return distance