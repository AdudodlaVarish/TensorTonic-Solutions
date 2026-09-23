import math

def log_loss(y_true: list, y_pred: list, eps: float = 1e-15) -> list:
    """
    Returns a list of loss values.
    """

    Loss = []

    for i in range(len(y_pred)):
      y_pred[i] = max(eps, min(y_pred[i], 1-eps))
      Loss.append(y_true[i] * -1 * math.log(y_pred[i]) + (1 - y_true[i]) * -1 * math.log(1 - y_pred[i]))
    
    return Loss

    