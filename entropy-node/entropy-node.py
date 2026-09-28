import numpy as np

def entropy_node(y: list[int]) -> float:

    y_ = np.asarray(y)
    vals, counts = np.unique(y_, return_counts = True)
    p = counts / np.sum(counts)
    p_ = np.log2(p)
    entropy = np.sum(p * p_)
    return -1 * entropy