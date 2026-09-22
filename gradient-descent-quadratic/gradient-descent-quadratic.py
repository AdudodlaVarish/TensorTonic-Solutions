def gradient_descent_quadratic(a: float, b: float, c: float, x0: float, lr: float, steps: int) -> float:
    
    for _ in range(steps):
        gradient = lambda x: 2 * a * x + b
        x0 = x0 - lr * gradient(x0)

    return x0