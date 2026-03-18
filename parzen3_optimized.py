"""
Parzen window density estimation - Optimized CPU Version

Author: Tahereh Bahraini
Converted to Python: 2025/11/12
Optimized (Vectorized): 2025/11/23
"""

import numpy as np
from scipy.spatial.distance import cdist


def parzen3_vectorized(D, X_query, h):
    """
    ベクトル化されたParzen窓密度推定（CPU版）
    
    cdistを使用して全距離を一括計算し、高速化を実現
    
    Parameters:
    -----------
    D : ndarray
        クラスデータ (d x n)
    X_query : ndarray
        クエリポイント (d x n_query)
    h : float
        帯域幅パラメータ
        
    Returns:
    --------
    p : ndarray
        各クエリポイントの密度推定値 (n_query,)
    """
    d = D.shape[0]  # 次元数
    n = D.shape[1]  # クラスのサンプル数
    v = h ** d
    
    # cdistは (n_samples, n_features) 形式を期待するため転置
    # distances: (n_query, n)
    distances = cdist(X_query.T, D.T, metric='euclidean')
    
    # ガウスカーネル: (n_query, n)
    k = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (distances / h)**2)
    
    # 各クエリポイントの密度推定: (n_query,)
    p = np.sum(k, axis=1) / (n * v)
    
    return p


def parzen3_vectorized_manual(D, X_query, h):
    """
    ベクトル化されたParzen窓密度推定（CPU版・手動ブロードキャスト）
    
    GPU版と同じ実装（NumPyのブロードキャストを使用）
    cdistを使わない代替実装
    
    Parameters:
    -----------
    D : ndarray
        クラスデータ (d x n)
    X_query : ndarray
        クエリポイント (d x n_query)
    h : float
        帯域幅パラメータ
        
    Returns:
    --------
    p : ndarray
        各クエリポイントの密度推定値 (n_query,)
    """
    d = D.shape[0]  # 次元数
    n = D.shape[1]  # クラスのサンプル数
    n_query = X_query.shape[1]  # クエリポイント数
    v = h ** d
    
    # ブロードキャストで差分を計算
    # X_query[:, :, None]: (d, n_query, 1)
    # D[:, None, :]: (d, 1, n)
    # diff: (d, n_query, n)
    diff = (X_query[:, :, np.newaxis] - D[:, np.newaxis, :]) / h
    
    # 各次元での差の二乗和: (n_query, n)
    sq_dist = np.sum(diff**2, axis=0)
    
    # ガウスカーネル: (n_query, n)
    k = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * sq_dist)
    
    # 各クエリポイントの密度推定: (n_query,)
    p = np.sum(k, axis=1) / (n * v)
    
    return p
