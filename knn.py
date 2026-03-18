"""
K-Nearest Neighbors classifier

Author: Tahereh Bahraini
Converted to Python: 2025/11/12
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import mode


def knn(trdata, trl, tedata, k=1):
    """
    K-Nearest Neighbors classifier
    
    Parameters:
    -----------
    trdata : ndarray
        Training data (n_samples, n_features)
    trl : ndarray
        Training labels (n_samples,)
    tedata : ndarray
        Test data (n_test_samples, n_features)
    k : int, optional
        Number of neighbors (default=1)
        
    Returns:
    --------
    predl : ndarray
        Predicted labels (n_test_samples,)
    neighbor_dist : ndarray
        Distances to k nearest neighbors (n_test_samples, k)
    """
    # Calculate distances
    dist = cdist(np.abs(tedata), np.abs(trdata))
    
    # Sort distances and get indices
    ind = np.argsort(dist, axis=1)
    sdist = np.sort(dist, axis=1)
    
    # Get labels of k nearest neighbors
    neighbor_labels = trl[ind[:, :k]]
    
    # Get mode (most frequent label)
    predl = mode(neighbor_labels, axis=1, keepdims=False)[0]
    
    # Get distances to k nearest neighbors
    neighbor_dist = sdist[:, :k]
    
    return predl, neighbor_dist
