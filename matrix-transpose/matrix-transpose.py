import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    A = np.array(A)
    m,n = A.shape
    
    AT = np.zeros((n,m))

    for i,row in enumerate(A):
        for j,item in enumerate(row):
            AT[j,i] = A[i,j] 
    
    return AT
