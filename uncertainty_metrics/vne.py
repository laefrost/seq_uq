# Code for estimating the von neuman entropy
from sklearn import metrics
import numpy as np
# Sum (Eigen vlaues of 1/n * K) * log (Eigenvalues of 1/n K)

def vne(Y, kernel=lambda x, y: metrics.pairwise.rbf_kernel(x, y, gamma=None), type_mask = None):
    n = Y.shape[0]
    YY = kernel(Y, Y)
    YY.astype(np.float64)
    if type_mask is not None: 
        YY = np.where(type_mask == 1, 1, YY)
    YY = 1/n * YY 
    eigenvalues = np.linalg.eigvalsh(YY)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    
    # Muss ich die normieren?
    eigenvalues = eigenvalues / np.sum(eigenvalues)
    vne = - np.sum(eigenvalues * np.log(eigenvalues))
    
    return vne
    