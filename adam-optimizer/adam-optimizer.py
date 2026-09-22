import numpy as np

def adam_step(
    param: list,
    grad: list,
    m: list,
    v: list,
    t: int,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    m_old = np.array(m)
    gradient = np.array(grad)
    v_old = np.array(v)
    param_old = np.array(param)
    
    m_new = m_old * beta1 + (1-beta1) * gradient
    v_new = v_old * beta2 + (1-beta2) * (gradient ** 2)

    m_corrected = m_new / (1 - (beta1 ** t))
    v_corrected = v_new / (1 - (beta2 ** t))

    param_new = param_old - lr * (m_corrected) / (np.sqrt(v_corrected) + eps)

    return (param_new, m_new, v_new)
