import numpy as np

def svd_2x2(A: np.ndarray) -> tuple:
    """
    Compute SVD of a 2x2 matrix.
    
    Args:
        A: 2x2 numpy array
    
    Returns:
        U: 2x2 orthogonal matrix (left singular vectors)
        s: 1D array of singular values
        V: 2x2 matrix (right singular vectors)
    """
    # Your code here
    A = np.asarray(A, dtype=float)
    
    # 1. Compute eigenvalues and eigenvectors of the symmetric matrix A.T @ A
    ATA = A.T @ A
    eigvals, eigvecs = np.linalg.eigh(ATA)
    
    # Sort in descending order
    idx = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[idx], 0.0)  # Clamp tiny negative float errors to 0
    V_cols = eigvecs[:, idx]                # Columns are right singular vectors v_1, v_2
    
    # Singular values are the square roots of the eigenvalues
    s = np.sqrt(eigvals)
    
    # 2. Compute left singular vectors U
    U = np.zeros((2, 2), dtype=float)
    
    if s[0] > 1e-12:
        U[:, 0] = (A @ V_cols[:, 0]) / s[0]
    else:
        # Matrix is effectively all zeros; any orthogonal matrices work
        return np.eye(2), np.zeros(2), np.eye(2)
        
    if s[1] > 1e-12:
        U[:, 1] = (A @ V_cols[:, 1]) / s[1]
    else:
        # Rank-1 case: construct u_2 perpendicular to u_1 so U remains orthogonal
        U[:, 1] = np.array([-U[1, 0], U[0, 0]])
        
    # Return V with right singular vectors as rows (V^T)
    V = V_cols.T
    
    return U, s, V