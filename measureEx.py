"""
Classification performance measurements

Author: Amin Allahyar (amin.allahyar@gmail.com)
Converted to Python: 2025/11/12
"""

import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score


def measureEx(true_label, pred_label, measurement=8, confusion_mat=None):
    """
    Calculate various classification performance metrics
    
    Parameters:
    -----------
    true_label : array-like
        True labels
    pred_label : array-like
        Predicted labels
    measurement : int or str, optional
        Type of measurement (default=8):
        0, 'f', 'f-measure' => F-Measure
        1, 'p', 'precision' => Precision
        2, 'r', 'recall' => Recall
        3, 'tp', 'true positive rate' => True Positive Rate
        4, 'fp', 'false positive rate' => False Positive Rate
        5, 'se', 'sensitivity' => Sensitivity
        6, 'sp', 'specificity' => Specificity
        7, 'tn', 'true negative rate' => True Negative Rate
        8, 'a', 'acc' => Accuracy
        12, 'c', 'cm' => Confusion Matrix
    confusion_mat : ndarray, optional
        Pre-computed confusion matrix
        
    Returns:
    --------
    res : float or ndarray
        Measurement result
    """
    measurement = str(measurement).lower()
    
    if confusion_mat is None:
        confusion_mat = confusion_matrix(true_label, pred_label)
    
    K = confusion_mat.shape[0]
    
    if measurement in ['0', 'f', 'f-measure']:
        # F-Measure
        p = np.diag(confusion_mat) / confusion_mat.sum(axis=0)
        r = np.diag(confusion_mat) / confusion_mat.sum(axis=1)
        res = 2 * (p * r) / (p + r)
        
    elif measurement in ['1', 'p', 'precision']:
        # Precision
        res = np.diag(confusion_mat) / confusion_mat.sum(axis=0)
        
    elif measurement in ['2', 'r', 'recall', '3', 'tp', 'true positive rate', 
                          '5', 'se', 'sensitivity']:
        # Recall / True Positive Rate / Sensitivity
        res = np.diag(confusion_mat) / confusion_mat.sum(axis=1)
        
    elif measurement in ['4', 'fp', 'false positive rate']:
        # False Positive Rate
        res = np.zeros(K)
        for i in range(K):
            fp = confusion_mat[:, i].sum() - confusion_mat[i, i]
            tn_fp = np.delete(confusion_mat, i, axis=0).sum()
            res[i] = fp / tn_fp if tn_fp > 0 else 0
            
    elif measurement in ['6', 'sp', 'specificity', '7', 'tn', 'true negative rate']:
        # Specificity / True Negative Rate
        res = np.zeros(K)
        for i in range(K):
            tmp_mat = np.delete(confusion_mat, i, axis=0)
            tn = np.delete(tmp_mat, i, axis=1).sum()
            tn_fp = tmp_mat.sum()
            res[i] = tn / tn_fp if tn_fp > 0 else 0
            
    elif measurement in ['8', 'a', 'acc']:
        # Accuracy
        res = accuracy_score(true_label, pred_label)
        
    elif measurement in ['12', 'c', 'cm']:
        # Confusion Matrix
        res = confusion_mat
        
    else:
        # Default to accuracy
        res = accuracy_score(true_label, pred_label)
    
    return res
