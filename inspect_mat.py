import scipy.io as sio
import numpy as np

mat = sio.loadmat("data/data_TwoDiamonds.mat")

# 全キーを表示
for key, value in mat.items():
    if not key.startswith("__"):
        print(f"Key: {key}, Shape: {value.shape}, Type: {type(value).__name__}")
        if value.ndim == 2 and min(value.shape) <= 50:
            print(f"  First few values:\n{value[:5, :5]}")
