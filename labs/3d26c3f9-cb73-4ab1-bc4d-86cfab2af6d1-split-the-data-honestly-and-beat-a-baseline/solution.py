# --- cell 1 ---
import numpy as np

def split_and_baseline(X, y, train_frac, val_frac, test_frac, seed):
    """Split rows into train/val/test folds, fit something on the training fold, predict the test fold.

    Returns
    -------
    test_predictions : np.ndarray, shape (len(test_idx),)
    train_idx, val_idx, test_idx : 1-D integer arrays forming a partition of range(len(y))
    """
    n_samples, n_features = X.shape
    
    # 1. 使用指定的 seed 正确生成随机排列
    rng = np.random.default_rng(seed=seed)
    indices = rng.permutation(n_samples)
    
    # 2. 按比例切分索引
    train_end = int(round(n_samples * train_frac))
    val_end = train_end + int(round(n_samples * val_frac))
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    # 3. 仅使用训练集拟合带截距项的线性回归 (OLS)
    X_train = np.column_stack([np.ones(len(train_idx)), X[train_idx]])
    y_train = y[train_idx]
    weights, _, _, _ = np.linalg.lstsq(X_train, y_train, rcond=None)
    
    # 4. 在测试集上进行预测
    X_test = np.column_stack([np.ones(len(test_idx)), X[test_idx]])
    test_predictions = X_test @ weights
    
    # 5. 按题目要求的顺序返回 4 个值
    return test_predictions, train_idx, val_idx, test_idx