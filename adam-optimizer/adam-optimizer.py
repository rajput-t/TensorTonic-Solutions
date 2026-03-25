import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    
    m,v,grad,param = np.array(m), np.array(v), np.array(grad), np.array(param)
    # updating momentum and velocity
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad**2

    # bias correction
    M = m / (1 - beta1**t)
    V = v / (1 - beta2**t)

    # updating parameters using the bias corrected momentum and velocity
    param = param - lr * (M / (V**0.5 + eps))

    return (param, m, v)