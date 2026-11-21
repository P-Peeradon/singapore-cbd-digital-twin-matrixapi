import numpy as np


def normal_matrix_from_4x4(transform_4x4: np.ndarray) -> np.ndarray:
    """
    Compute the 3x3 normal transformation matrix (inverse transpose of the
    upper-left 3x3 linear part) from a 4x4 affine transformation matrix.
    """
    m = np.asarray(transform_4x4, dtype=np.float64)
    if m.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 matrix, got shape {m.shape}.")

    linear_3x3 = m[:3, :3]
    # (A^-1)^T == (A^T)^-1
    return np.linalg.inv(linear_3x3).T


def transform_normals(normals: np.ndarray, transform_4x4: np.ndarray) -> np.ndarray:
    """
    Transform a normal vector (3,) or array of normals (N, 3) and re-normalize.
    """
    n_arr = np.asarray(normals, dtype=np.float64)
    n_mat = normal_matrix_from_4x4(transform_4x4)

    transformed = n_arr @ n_mat.T
    norms = np.linalg.norm(transformed, axis=-1, keepdims=True)
    # Guard against zero-length vectors
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    return transformed / safe_norms