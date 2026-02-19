import numpy as np
from sklearn import metrics

def rao_entropy(Y, probs, kernel=lambda x, y: metrics.pairwise.cosine_distances(x, y)):
    probs = np.array(probs)
    s = probs.sum()
    if s <= 0:
        raise ValueError("probs must sum to a positive value")
    probs /= s
    probs = np.asarray(probs, dtype=float).reshape(-1, 1)  # list -> (n,1)
    YY = kernel(Y, Y)              # (n,n)
    probs_matrix = probs @ probs.T # (n,n)

    return float(np.sum(probs_matrix * YY))

def avg_conflict(Y, probs, kernel=lambda x, y: metrics.pairwise.cosine_distances(x, y), type_mask=None): 
    D = kernel(Y, Y)  # Shape: (S, S)
    D = D / 2
    probs = np.array(probs)
    s = probs.sum()
    if s <= 0:
        raise ValueError("probs must sum to a positive value")
    probs /= s
    
    # Setze Diagonale auf 0 für i != j Bedingung
    D_no_diag = D.copy()
    np.fill_diagonal(D_no_diag, 0)
    
    if type_mask is not None: 
        D_no_diag = np.where(type_mask == 1, 0, D_no_diag)
    
    # Berechne innere Summe für alle j gleichzeitig: sum_{i != j} d_ij * p_i
    inner_sums = D_no_diag @ probs  # Matrix-Vektor Multiplikation
    
    # Berechne H_d
    args = np.clip(1 - inner_sums, 1e-10, None)
    H_d = -np.sum(probs * np.log(args))
    
    return H_d

