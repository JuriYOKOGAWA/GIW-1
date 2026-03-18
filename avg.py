"""
Calculate mean of finite values

Author: Tahereh Bahraini
Converted to Python: 2025/11/12
"""

import numpy as np


def avg(x):
    """
    Calculate mean of finite values
    
    Parameters:
    -----------
    x : array-like
        Input array
        
    Returns:
    --------
    float
        Mean of finite values
    """
    x = np.array(x)
    x = x[np.isfinite(x)]
    return np.mean(x) if len(x) > 0 else 0
