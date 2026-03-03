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

def normalize_kernel(K):
    """Normalize a kernel matrix K."""
    diag = np.sqrt(np.diag(K))
    return K / np.outer(diag, diag)

def vne(Y, kernel=lambda x, y: metrics.pairwise.rbf_kernel(x, y, gamma=1), type_mask = None, mode = 'sampling', probs = None, eps=1e-10, Y2 = None, combination_mode = "additive"): 
    
    def entropy_from(K):
        std = K.std()
        if mode == 'sampling':
            K = K / n
        else:
            print("probs before: ", probs)
            p = np.array(probs, dtype=np.float64)
            p = p / p.sum()
            print(np.sum(probs))
            print("Probs: ", probs)
            D = np.diag(np.sqrt(p))
            K = D @ K @ D
            
        evals = np.linalg.eigvalsh(K)
        evals = evals[evals > eps]
        
        if evals.size == 0:
            return 0.0
        # if len(probs) > 1 and mode != 'sampling': 
        #     print(probs)
        #     print(-np.sum(evals * np.log(evals)), evals)
        #     print(-np.sum(evals_tmp * np.log(evals_tmp)), evals_tmp)
        if n != 1:
            vne = -np.sum(evals * np.log(evals)) / np.log(n)
        else: 
            vne = 0#-np.sum(evals * np.log(evals))
            
        return vne, std #float(-np.sum(evals * np.log(evals)))
    
    YY = kernel(Y, Y).astype(np.float64)
    
    n = Y.shape[0]
    
    if Y2 is not None:
        Y2Y2 = kernel(Y2, Y2).astype(np.float64)
        if combination_mode == "multiplicative": 
            YY2 = YY * Y2Y2
            HXY, STDXY = entropy_from(YY2)
            HY, STDY = entropy_from(Y2Y2)
            HX, STDX = entropy_from(YY)
            print("HXY", HXY,  "HY", HY, "HX", HX, "HX+HY", HX + HY, "HXY- HY", HXY - HY)
            return HXY - HY, STDXY
        else: 
            #YY = 0.5 * YY + 0.5 * Y2Y2
            YY2 = YY * Y2Y2
            HXY, STDXY = entropy_from(YY2)
            HY, STDY = entropy_from(Y2Y2)
            HX, STDX = entropy_from(YY)
            print("HXY", HXY,  "HY", HY, "HX", HX, "HX+HY", HX + HY, "HXY- HY", HXY - HY)
            return HXY - HX, STDXY
    
    if type_mask is not None:
        YY = np.where(type_mask == 1, 1.0, YY)
    
    return entropy_from(YY)
        
    # if mode == 'sampling': 
    #     # normalize the kernel
    #     YY = YY / n
        
    # else: 
    #     probs = np.array(probs)
    #     probs = probs / probs.sum()
    #     # normalize the kernel
    #     D = np.diag(np.sqrt(probs))
    #     YY = D @ YY @ D

    # eigenvalues = np.linalg.eigvalsh(YY)
    # eigenvalues = eigenvalues[eigenvalues > eps]
    # if eigenvalues.size == 0:
    #     return 0.0

    # # eigenvalues = eigenvalues / eigenvalues.sum()
    # return float(-np.sum(eigenvalues * np.log(eigenvalues)))