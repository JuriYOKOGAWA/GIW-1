import os
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import KFold, train_test_split

from Dlda3_optimized import Dlda3_optimized
from knn import knn
from measureEx import measureEx
from avg import avg


def _is_label_like(vec):
    vec = np.asarray(vec).reshape(-1)
    if vec.size < 2:
        return False
    if np.any(~np.isfinite(vec)):
        return False
    uniq = np.unique(vec)
    if len(uniq) < 2 or len(uniq) > 20:
        return False
    # ラベル想定: ほぼ整数
    return np.allclose(vec, np.round(vec), atol=1e-6)


def load_twodiamonds(path="data/data_TwoDiamonds.mat"):
    """
    TwoDiamonds データセットを読み込む
    
    Returns:
        X: (n_samples, n_features)
        y: (n_samples,)
    """
    mat = loadmat(path)
    
    # D: (800, 2), L: (800, 1) を抽出
    X = mat['D'].astype(float)  # (800, 2)
    y = mat['L'].astype(int).reshape(-1)  # (800,)
    
    print(f"Raw loaded: X={X.shape}, y={y.shape}, unique labels={np.unique(y)}")
    
    # 2値分類チェック
    uniq = np.unique(y)
    if len(uniq) != 2:
        raise ValueError(f"2値分類を想定していますが、クラス数={len(uniq)} です")
    
    # Dlda3_optimized 用に {1, 2} へマップ
    label_map = {uniq[0]: 1, uniq[1]: 2}
    y12 = np.vectorize(label_map.get)(y).astype(int)
    
    return X, y12


def run_dlda_grid_search(
    train_features,
    train_labels,
    h_values,
    lambda_values,
    k_neighbors=3
):
    train_features = np.asarray(train_features)
    train_labels = np.asarray(train_labels).astype(int).reshape(-1)

    # Dlda3_optimized の入力形式: (features x samples), 最終行がラベル
    dlda_input = np.vstack([train_features.T, train_labels.reshape(1, -1)])

    n_features = train_features.shape[1]
    n_h = len(h_values)
    n_lam = len(lambda_values)

    WW = np.zeros((n_features, n_lam, n_h))
    JJ = np.zeros((n_h, n_lam))
    ACC = np.zeros((n_h, n_lam))

    for ii, h in enumerate(h_values):
        print(f"Progress: h iteration {ii + 1}/{n_h} (h={h:.4f})")

        for jj, lambda_param in enumerate(lambda_values):
            W, J = Dlda3_optimized(dlda_input, lambda_param, h)
            WW[:, jj, ii] = W.flatten()
            JJ[ii, jj] = J

            projected = (dlda_input[:-1, :].T @ W).reshape(-1, 1)

            kf = KFold(n_splits=10, shuffle=True, random_state=42)
            fold_acc = []

            for tr_idx, va_idx in kf.split(projected):
                x_tr = projected[tr_idx]
                y_tr = train_labels[tr_idx]
                x_va = projected[va_idx]
                y_va = train_labels[va_idx]

                pred, _ = knn(x_tr, y_tr, x_va, k=k_neighbors)
                acc = measureEx(y_va, pred, 8) * 100
                fold_acc.append(acc)

            ACC[ii, jj] = avg(fold_acc)

    print("\nGrid search completed!")

    best_ii, best_jj = np.unravel_index(np.argmax(ACC), ACC.shape)
    best_result = {
        "best_h": h_values[best_ii],
        "best_lambda": lambda_values[best_jj],
        "best_acc_cv": ACC[best_ii, best_jj],
        "best_W_cv": WW[:, best_jj, best_ii]
    }

    return WW, JJ, ACC, best_result


def main():
    # 1) データ読み込み
    X, y = load_twodiamonds("data/data_TwoDiamonds.mat")
    print(f"Loaded: X={X.shape}, y={y.shape}, classes={np.unique(y)}")

    # 2) 学習/テスト分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3) ハイパーパラメータ範囲
    h_values = np.linspace(0.1, 2.0, 20)
    lambda_values = np.logspace(-4, 1, 20)

    # 4) 学習データでグリッドサーチ
    WW, JJ, ACC, best = run_dlda_grid_search(
        train_features=X_train,
        train_labels=y_train,
        h_values=h_values,
        lambda_values=lambda_values,
        k_neighbors=3
    )
    print("Best from CV:", best)

    # 5) 各 h ごとに最良の lambda でのテスト精度を計算
    test_acc_per_h = []
    
    for ii, h in enumerate(h_values):
        # この h での最良の lambda を取得
        best_jj_for_h = np.argmax(ACC[ii, :])
        best_lambda_for_h = lambda_values[best_jj_for_h]
        
        # 最良パラメータで再学習
        train_data_for_dlda = np.vstack([X_train.T, y_train.reshape(1, -1)])
        best_W, _ = Dlda3_optimized(
            train_data_for_dlda,
            lambda_param=best_lambda_for_h,
            h=h
        )
        
        # テスト予測
        train_proj = (X_train @ best_W).reshape(-1, 1)
        test_proj = (X_test @ best_W).reshape(-1, 1)
        
        y_pred_test, _ = knn(train_proj, y_train, test_proj, k=3)
        test_acc = measureEx(y_test, y_pred_test, 8) * 100
        
        test_acc_per_h.append(test_acc)
        print(f"h={h:.4f}, best_lambda={best_lambda_for_h:.6f}, test_acc={test_acc:.4f}%")

    # 6) 保存
    os.makedirs("output", exist_ok=True)
    np.savetxt("output/test_acc_DLDA.txt", np.array(test_acc_per_h), fmt="%.6f")
    print(f"\nSaved {len(test_acc_per_h)} values to output/test_acc_DLDA.txt")


if __name__ == "__main__":
    main()
