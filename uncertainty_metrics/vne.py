from sklearn import metrics
import numpy as np
from sklearn.preprocessing import KernelCenterer

def vne(Y, kernel=lambda x, y: metrics.pairwise.rbf_kernel(x, y, gamma=1), type_mask = None, mode = 'sampling', probs = None, eps=1e-10, Y2 = None, combination_mode = "additive"): 
    """
    Computes the Von Neumann Entropy (VNE) of a dataset using a kernel matrix.

    Parameters
    ----------
    Y : np.ndarray
        Embedding matrix of shape (n_samples, d).
    kernel : callable, optional
        Kernel function taking two arrays and returning a kernel matrix.
    type_mask : np.ndarray or None, optional
        Deprecated. Binary mask applied to the kernel matrix; entries where mask == 1 are set to 1.0.
    mode : str, optional
        Normalization mode for the kernel matrix. 'sampling' divides by n; 'prob' applies a probability-weighted diagonal scaling. Default is 'sampling'.
    probs : array-like or None, optional
        Sample probabilities used when mode='prob'. Will be normalized to sum to 1.
    eps : float, optional
        Threshold below which eigenvalues are discarded. Default is 1e-10.
    Y2 : np.ndarray or None, optional
        Experimental. Optional second dataset. If provided, computes a combined entropy between Y and Y2 using the specified combination_mode.
    combination_mode : str, optional
        How to combine kernels when Y2 is provided

    Returns
    -------
    vne : float
        The Von Neumann Entropy value (or conditional entropy if Y2 is provided).
    std : float
        Standard deviation of the kernel matrix used in the entropy computation.
    """
    def entropy_from(K):
        """Computes VNE and kernel std from a raw kernel matrix K."""
        std = K.std()
        if mode == 'sampling':
            K = K / n
        else:
            p = np.array(probs, dtype=np.float64)
            p = p / p.sum()
            D = np.diag(np.sqrt(p))
            K = D @ K @ D
            
        evals = np.linalg.eigvalsh(K)
        evals = evals[evals > eps]
        
        if evals.size == 0:
            return 0.0
        if n != 1:
            vne = -np.sum(evals * np.log(evals))
        else: 
            vne = 0
            
        return vne, std
    
    YY = kernel(Y, Y).astype(np.float64)
    n = Y.shape[0]
    
    # Experimental for combining kernels
    if Y2 is not None:
        Y2Y2 = kernel(Y2, Y2).astype(np.float64)
        if combination_mode == "multiplicative": 
            YY2 = YY * Y2Y2
            HXY, STDXY = entropy_from(YY2)
            HY, STDY = entropy_from(Y2Y2)
            HX, STDX = entropy_from(YY)
            return HXY - HY, STDXY
        else: 
            YY2 = YY * Y2Y2
            HXY, STDXY = entropy_from(YY2)
            HY, STDY = entropy_from(Y2Y2)
            HX, STDX = entropy_from(YY)
            return HXY - HX, STDXY
    
    # Deprecated: 
    if type_mask is not None:
        YY = np.where(type_mask == 1, 1.0, YY)
    
    return entropy_from(YY)