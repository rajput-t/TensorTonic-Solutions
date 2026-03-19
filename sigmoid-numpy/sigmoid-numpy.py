import numpy as np

def sigmoid(x):
    x=np.array(x)
    """
    Vectorized sigmoid function.
    """
    return 1/(1+np.exp(-x))