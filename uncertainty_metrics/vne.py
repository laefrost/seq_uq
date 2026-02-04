# Code for estimating the von neuman entropy
from sklearn import metrics
import numpy as np
from sklearn.preprocessing import KernelCenterer
# Sum (Eigen vlaues of 1/n * K) * log (Eigenvalues of 1/n K)

def vne(Y, kernel=lambda x, y: metrics.pairwise.rbf_kernel(x, y, gamma=None), type_mask = None, Y2 = None):
    n = Y.shape[0]
    YY = kernel(Y, Y)
    
    if Y2 is not None: 
        Y2Y2 =  kernel(Y2, Y2)   
        YY = np.multiply(YY, Y2Y2)
        
    transformer = KernelCenterer().fit(YY)
    YY_centered = transformer.transform(YY)
    YY_centered.astype(np.float64)
    
    if type_mask is not None: 
        YY_centered = np.where(type_mask == 0, 1, YY_centered)
    
    YY_centered = 1/n * YY_centered 
    eigenvalues = np.linalg.eigvalsh(YY_centered)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    
    # Muss ich die normieren? --- eigentlich sollte das nicht nötig sein
    eigenvalues = eigenvalues / np.sum(eigenvalues)
    vne = - np.sum(eigenvalues * np.log(eigenvalues))
    
    return vne
    