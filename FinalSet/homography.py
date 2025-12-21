"""
Homography 계산 구현 (DLT 알고리즘)
Homography.pdf 강의자료 참고
"""

import numpy as np
from typing import Tuple


def normalize_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalized DLT: 평균 거리를 sqrt(2)로 만들도록 정규화
    """
    N = len(points)
    
    mean = np.mean(points, axis=0)
    
    centered = points - mean
    distances = np.sqrt(np.sum(centered ** 2, axis=1))
    mean_distance = np.mean(distances)
    
    if mean_distance < 1e-10:
        scale = 1.0
    else:
        scale = np.sqrt(2.0) / mean_distance
    
    T = np.array([
        [scale, 0, -scale * mean[0]],
        [0, scale, -scale * mean[1]],
        [0, 0, 1]
    ], dtype=np.float32)
    
    points_homogeneous = np.column_stack([points, np.ones(N)])
    points_norm_homogeneous = (T @ points_homogeneous.T).T
    points_norm = points_norm_homogeneous[:, :2]
    
    return points_norm, T


def compute_homography_dlt(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    """
    Normalized DLT: x' = H * x, A * h = 0 형태로 구성하여 SVD로 해 구하기
    """
    N = len(points1)
    
    if N < 4:
        return np.eye(3, dtype=np.float32)
    
    p1_norm, T1 = normalize_points(points1)
    p2_norm, T2 = normalize_points(points2)
    
    A = np.zeros((2 * N, 9), dtype=np.float32)
    
    for i in range(N):
        x, y = p1_norm[i]
        xp, yp = p2_norm[i]
        
        A[2*i] = [-x, -y, -1, 0, 0, 0, x*xp, y*xp, xp]
        A[2*i+1] = [0, 0, 0, -x, -y, -1, x*yp, y*yp, yp]
    
    U, S, Vt = np.linalg.svd(A)
    
    h = Vt[-1]
    
    H_norm = h.reshape(3, 3)
    
    T1_inv = np.linalg.inv(T1)
    T2_inv = np.linalg.inv(T2)
    
    H = T2_inv @ H_norm @ T1
    
    if abs(H[2, 2]) > 1e-10:
        H = H / H[2, 2]
    
    return H.astype(np.float32)


def apply_homography(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    N = len(points)
    
    points_homogeneous = np.column_stack([points, np.ones(N)])
    
    transformed_homogeneous = (H @ points_homogeneous.T).T
    
    w = transformed_homogeneous[:, 2]
    w_safe = np.where(np.abs(w) > 1e-10, w, np.nan)
    transformed = transformed_homogeneous[:, :2] / w_safe[:, np.newaxis]
    
    return transformed


def compute_homography_affine(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    """
    Affine 변환: [x']   [a  b  tx] [x]
                 [y'] = [c  d  ty] [y]
                 [1 ]   [0  0  1 ] [1]
    A * h = b 형태로 구성하여 최소제곱법으로 해 구하기
    """
    N = len(points1)
    
    if N < 3:
        return np.eye(3, dtype=np.float32)
    
    p1_norm, T1 = normalize_points(points1)
    p2_norm, T2 = normalize_points(points2)
    
    A = np.zeros((2 * N, 6), dtype=np.float32)
    b = np.zeros(2 * N, dtype=np.float32)
    
    for i in range(N):
        x, y = p1_norm[i]
        xp, yp = p2_norm[i]
        
        A[2*i] = [x, y, 1, 0, 0, 0]
        b[2*i] = xp
        
        A[2*i+1] = [0, 0, 0, x, y, 1]
        b[2*i+1] = yp
    
    try:
        h = np.linalg.lstsq(A, b, rcond=None)[0]
    except:
        return np.eye(3, dtype=np.float32)
    
    H_norm = np.array([
        [h[0], h[1], h[2]],
        [h[3], h[4], h[5]],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    T1_inv = np.linalg.inv(T1)
    T2_inv = np.linalg.inv(T2)
    
    H = T2_inv @ H_norm @ T1
    
    if abs(H[2, 2]) > 1e-10:
        H = H / H[2, 2]
    
    H[2, 0] = 0.0
    H[2, 1] = 0.0
    H[2, 2] = 1.0
    
    return H.astype(np.float32)


def interpolate_homography(H_proj: np.ndarray, H_aff: np.ndarray, alpha: float) -> np.ndarray:
    """
    H_interp = (1 - alpha) * H_proj + alpha * H_aff
    """
    alpha = np.clip(alpha, 0.0, 1.0)
    
    H_interp = (1.0 - alpha) * H_proj + alpha * H_aff
    
    if abs(H_interp[2, 2]) > 1e-10:
        H_interp = H_interp / H_interp[2, 2]
    
    return H_interp.astype(np.float32)