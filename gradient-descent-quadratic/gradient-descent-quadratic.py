def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    x = float(x0)
    f = 2*a*x + b       # derivative of the 1D quadratic
    
    for _ in range(steps):
        x = x - lr*f    # implementing gradient descent 
        f = 2*a*x + b   # calculating the grad at the new 'x' value
    
    return x