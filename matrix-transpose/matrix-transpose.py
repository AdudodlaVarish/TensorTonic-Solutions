import numpy as np

def matrix_transpose(A: list) -> np.ndarray:

    matrix = np.array(A)
    N, M = matrix.shape
    transpose = np.zeros((M, N))
    for i in range(M):
      for j in range(N):
        transpose[i, j] = matrix[j, i]
    
    return transpose
    
