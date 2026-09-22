import numpy as np

def matrix_trace(A: list) -> float:

    A = np.array(A)
    N, _ = A.shape
    trace = 0.0
    for i in range(N):
        trace += A[i,i]
    return trace