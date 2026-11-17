# --- cell 1 ---
import numpy as np

def impute(X: np.ndarray) -> np.ndarray:
    '''
    Fill in missing values (NaN) in the input array.
    
    Args:
        X: Array with possible NaN values, shape (n_samples, n_features)
    
    Returns:
        X_clean: Array with no NaN values, same shape as X
    '''
    X_clean = X.copy()
    
    # TODO: Fill in NaN values
    n_samples, n_features = X_clean.shape
    
    col_medians = np.nanmedian(X_clean, axis=0)
    
    col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)
    
    nan_rows, nan_cols = np.where(np.isnan(X_clean))
    X_clean[nan_rows, nan_cols] = col_medians[nan_cols]
    
    
    return X_clean