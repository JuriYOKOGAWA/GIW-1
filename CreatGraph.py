"""
Create Graph by selecting representative points based on k-nearest neighbors

Author: Tahereh Bahraini
Converted to Python: 2025/11/12
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


def CreatGraph(Data):
    """
    Create graph by selecting representative data points
    
    Parameters:
    -----------
    Data : ndarray
        Input data matrix (n_samples, n_features)
        
    Returns:
    --------
    G : ndarray
        Selected representative points (n_features, n_selected)
    """
    n = Data.shape[0]
    k = int(n / 10)
    
    # Find k+1 nearest neighbors (including the point itself)
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean')
    nbrs.fit(Data)
    distances, idx = nbrs.kneighbors(Data)
    
    # Create adjacency matrix
    w = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            p = np.where(idx[i, :] == j)[0]
            if len(p) > 0 and i != j:
                w[i, j] = 1
            else:
                w[i, j] = 0
    
    # Calculate degree for each node
    F = np.sum(w, axis=0)
    
    mi = np.min(F)
    mx = np.max(F)
    
    B = (mi + mx) / mx + 1
    D = int((mi + mx) / B)
    
    # Select points with degree >= D
    dx = np.where(F >= D)[0]
    G = Data[dx, :].T
    
    return G
