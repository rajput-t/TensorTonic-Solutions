import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    pos = np.arange(seq_len)[:,np.newaxis] # shape(seq_len, 1)
    i = np.arange(d_model)[np.newaxis,:]   # shape(1, d_model)

    # calculating the inner angle fraction 
    denominator = 1 / np.power(base,(2 * (i // 2))/d_model) # shape(1, d_model)
    fraction = pos * denominator # shape(seq_len, d_model)

    pe = np.zeros((seq_len,d_model))
    pe[:,0::2] = np.sin(fraction[:,0::2])
    pe[:,1::2] = np.cos(fraction[:,1::2])
    return pe