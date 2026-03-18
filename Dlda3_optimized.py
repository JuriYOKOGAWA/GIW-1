"""
Density-oriented Linear Discriminant Analysis (DLDA) - Optimized CPU Version

Author: Tahereh Bahraini
Converted to Python: 2025/11/12
Optimized (Vectorized Parzen): 2025/11/23
"""

import numpy as np
from parzen3_optimized import parzen3_vectorized


def Dlda3_optimized(data, lambda_param, h):
    """
    Density-oriented Linear Discriminant Analysis (最適化版)
    
    Parzen窓の密度推定をベクトル化して高速化
    元のDlda3と数学的に同一の計算を行うが、ループを削除
    
    Parameters:
    -----------
    data : ndarray
        Data matrix (features x samples), last row contains labels
    lambda_param : float
        Regularization parameter
    h : float
        Bandwidth for Parzen window
        
    Returns:
    --------
    W : ndarray
        Projection vector (d, 1)
    J : float
        Cost function value
    """
    # Find indices for each class
    d1 = np.where(data[-1, :] == 1)[0]
    d2 = np.where(data[-1, :] == 2)[0]
    
    # Extract class data (excluding labels)
    c1 = data[:-1, d1]
    c2 = data[:-1, d2]
    
    # Combine classes
    X = np.hstack([c1, c2])
    D, n = X.shape
    
    # Center data
    m = np.mean(X, axis=1, keepdims=True)
    X = X - m
    
    n1 = c1.shape[1]
    n2 = c2.shape[1]
    
    # Create target vector y
    y1 = np.full(n1, -2 * n2 / n)
    y2 = np.full(n2, 2 * n1 / n)
    y = np.hstack([y1, y2])
    
    # ===== ベクトル化されたParzen窓密度推定 =====
    # クラス1の密度推定（全サンプルを一度に計算）
    X_class1 = X[:, :n1]
    p1 = parzen3_vectorized(X_class1, X_class1, h)  # (n1,)
    
    # クラス2の密度推定（全サンプルを一度に計算）
    X_class2 = X[:, n1:n]
    p2 = parzen3_vectorized(X_class2, X_class2, h)  # (n2,)
    
    # 密度ベクトルを結合
    p = np.hstack([p1, p2])  # (n,)
    
    # 重み付き平均の計算（ベクトル化）
    # m1 = (1/n1) * Σ X[:, i] * p[i] for i in class1
    m1 = (X_class1 * p1[np.newaxis, :]).sum(axis=1, keepdims=True) / n1  # (D, 1)
    m2 = (X_class2 * p2[np.newaxis, :]).sum(axis=1, keepdims=True) / n2  # (D, 1)
    
    # Create diagonal matrix P
    P = np.diag(p)
    
    # Calculate scatter matrix
    st = X @ P @ X.T
    
    # Calculate projection vector W
    I = np.eye(D)
    W = (2 * n1 * n2 / n**2) * np.linalg.pinv(st + lambda_param * I) @ (m2 - m1)
    
    # Calculate cost function J
    XtW_minus_y = X.T @ W - y.reshape(-1, 1)
    J = (1/n) * (XtW_minus_y.T @ P @ XtW_minus_y) - lambda_param * (W.T @ W)
    J = J[0, 0]  # Extract scalar value
    
    return W, J
