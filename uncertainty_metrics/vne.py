# Code for estimating the von neuman entropy
from sklearn import metrics
import numpy as np
from sklearn.preprocessing import KernelCenterer
# Sum (Eigen vlaues of 1/n * K) * log (Eigenvalues of 1/n K)

import numpy as np

def build_Q(word_pos):
    word_pos = np.asarray(word_pos)
    n = len(word_pos)

    tags, inv = np.unique(word_pos, return_inverse=True)
    G = len(tags)

    # Z bauen
    Z = np.zeros((n, G))
    Z[np.arange(n), inv] = 1.0

    # Gruppengrößen
    counts = Z.sum(axis=0)  # (G,)
    inv_counts = np.where(counts > 0, 1.0 / counts, 0.0)

    # P = Z (Z^T Z)^{-1} Z^T
    P = (Z * inv_counts) @ Z.T

    Q = np.eye(n) - P
    return Q


# def vne(Y, kernel=lambda x, y: metrics.pairwise.rbf_kernel(x, y, gamma=None), type_mask = None, Y2 = None, word_pos = None):
#     n = Y.shape[0]
#     YY = kernel(Y, Y)
    
#     if Y2 is not None: 
#         Y2Y2 =  kernel(Y2, Y2)   
#         YY = np.multiply(YY, Y2Y2)
        
#     if type_mask is not None: 
#         YY = np.where(type_mask == 1, 0, YY)
    
#     if word_pos is not None: 
#         Q = build_Q(word_pos)
#         YY_centered = Q @ YY @ Q
#     else:     
#         transformer = KernelCenterer().fit(YY)
#         YY_centered = transformer.transform(YY)
#         YY_centered.astype(np.float64)
    
#     YY_centered = 1/n * YY_centered 
#     eigenvalues = np.linalg.eigvalsh(YY_centered)
#     eigenvalues = eigenvalues[eigenvalues > 1e-10]
    
#     # Muss ich die normieren? --- eigentlich sollte das nicht nötig sein
#     eigenvalues = eigenvalues / np.sum(eigenvalues)
#     vne = - np.sum(eigenvalues * np.log(eigenvalues))
    
#     return vne

# def gamma_median_heuristic(X, eps=1e-12):
#     D2 = pairwise_distances(X, metric="euclidean", squared=True)
#     # take upper triangle without diagonal
#     tri = D2[np.triu_indices_from(D2, k=1)]
#     med = np.median(tri[tri > 0])
#     return 1.0 / (2.0 * (med + eps))
    
    
# def vne(Y, kernel=lambda x, y: metrics.pairwise.rbf_kernel(x, y, gamma=None),
#         type_mask=None, Y2=None, word_pos=None, center=True, eps=1e-10, combination_mode = None):

#     YY = kernel(Y, Y).astype(np.float64)
#     n = Y.shape[0]
#     if Y2 is not None:
#         Y2Y2 = kernel(Y2, Y2).astype(np.float64)
#         if combination_mode == "multiplicative": 
#             YY = YY * Y2Y2
#         else: 
#             YY = 0.5 * YY + 0.5 * Y2Y2

#     if word_pos is not None:
#         Q = build_Q(word_pos)
#         YY = Q @ YY @ Q

#     # if center:
#     #     YY = KernelCenterer().fit_transform(YY)
        
#     if type_mask is not None:
#         YY = np.where(type_mask == 1, 1.0, YY)

#     # tr = np.trace(YY)
#     # if tr <= eps:
#     #     return 0.0

#     # R = YY / tr
#     YY = YY / n
#     eigenvalues = np.linalg.eigvalsh(YY)
#     eigenvalues = eigenvalues[eigenvalues > eps]
#     if eigenvalues.size == 0:
#         return 0.0

#     # eigenvalues = eigenvalues / eigenvalues.sum()
#     return float(-np.sum(eigenvalues * np.log(eigenvalues)))



def vne(Y, kernel=lambda x, y: metrics.pairwise.rbf_kernel(x, y, gamma=None), type_mask = None, mode = 'sampling', probs = None, eps=1e-10, Y2 = None, combination_mode = "additive"): 
    YY = kernel(Y, Y).astype(np.float64)
    n = Y.shape[0]
    print("kernel shape: ", YY.shape)
    if Y2 is not None:
        Y2Y2 = kernel(Y2, Y2).astype(np.float64)
        if combination_mode == "multiplicative": 
            YY = YY * Y2Y2
        else: 
            YY = 0.5 * YY + 0.5 * Y2Y2
    
    if type_mask is not None:
        YY = np.where(type_mask == 1, 1.0, YY)
        
    if mode == 'sampling': 
        YY = YY / n
        
    else: 
        YY = kernel(Y, Y).astype(np.float64)
        D = np.diag(np.sqrt(probs))
        YY = D @ YY @ D

    
    eigenvalues = np.linalg.eigvalsh(YY)
    eigenvalues = eigenvalues[eigenvalues > eps]
    if eigenvalues.size == 0:
        return 0.0

    # eigenvalues = eigenvalues / eigenvalues.sum()
    return float(-np.sum(eigenvalues * np.log(eigenvalues)))