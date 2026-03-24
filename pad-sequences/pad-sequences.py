import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L)
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    if not seqs:
        return np.array([])

    N = len(seqs)

    # Determine max length
    if max_len is None:
        L = max(len(seq) for seq in seqs)
    else:
        L = max_len

    # Initialize with pad_value
    padded = np.full((N, L), pad_value)

    # Fill values (with truncation if needed)
    for i, seq in enumerate(seqs):
        trunc = seq[:L]  # handles truncation automatically
        padded[i, :len(trunc)] = trunc

    return padded