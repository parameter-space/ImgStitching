"""
기하학적 변환 모듈
RANSAC과 DLT(Direct Linear Transform)를 사용한 Homography 계산을 구현합니다.
"""

import numpy as np
from typing import Tuple


def normalize_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    점들을 정규화합니다 (평균을 0으로, 평균 거리를 sqrt(2)로).
    
    Args:
        points: 점들 (N, 2) - 각 행은 [x, y] - float32
    
    Returns:
        normalized_points: 정규화된 점들 (N, 2) - float32
        T: 정규화 변환 행렬 (3, 3) - float32
    """
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    mean_dist = np.mean(np.sqrt(np.sum(centered**2, axis=1)))
    scale = np.sqrt(2) / mean_dist if mean_dist > 0 else 1.0
    
    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1]
    ], dtype=np.float32)
    
    points_norm = (T @ np.column_stack([points, np.ones(len(points))]).T).T
    return points_norm[:, :2], T


def compute_homography_dlt(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    """
    정규화된 DLT를 사용하여 Homography 행렬을 계산합니다.
    
    Args:
        points1: 첫 번째 이미지의 점들 (N, 2) - 각 행은 [x, y] - float32, N >= 4
        points2: 두 번째 이미지의 점들 (N, 2) - 각 행은 [x, y] - float32, N >= 4
    
    Returns:
        H: Homography 행렬 (3, 3) - float32
    """
    # 1. Normalize points
    p1_norm, T1 = normalize_points(points1)
    p2_norm, T2 = normalize_points(points2)
    
    # 2. Build A matrix
    N = len(points1)
    A = np.zeros((2 * N, 9), dtype=np.float32)
    for i in range(N):
        x, y = p1_norm[i]
        xp, yp = p2_norm[i]
        A[2*i] = [-x, -y, -1, 0, 0, 0, x*xp, y*xp, xp]
        A[2*i+1] = [0, 0, 0, -x, -y, -1, x*yp, y*yp, yp]
        
    # 3. SVD
    U, S, Vt = np.linalg.svd(A)
    H_norm = Vt[-1].reshape(3, 3)
    
    # 4. Denormalize: H = inv(T2) * H_norm * T1
    H = np.linalg.inv(T2) @ H_norm @ T1
    return H / H[2, 2]  # Normalize scale


def ransac_homography(p1: np.ndarray, p2: np.ndarray, 
                     max_iterations: int = 2000, 
                     threshold: float = 5.0,
                     min_inliers: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    RANSAC 알고리즘을 사용하여 최적의 Homography 행렬을 찾습니다.
    
    Args:
        p1: 첫 번째 이미지의 점들 (N, 2) - 각 행은 [x, y] - float32
        p2: 두 번째 이미지의 점들 (N, 2) - 각 행은 [x, y] - float32
        max_iterations: 최대 반복 횟수 (int, 기본값: 2000)
        threshold: 인라이어 임계값 (픽셀 단위, float, 기본값: 5.0)
        min_inliers: 최소 인라이어 개수 (int, 기본값: 10)
    
    Returns:
        best_H: 최적의 Homography 행렬 (3, 3) - float32
        inlier_mask: 인라이어 마스크 (N,) - bool, True인 경우 인라이어
    """
    best_inliers = []
    best_H = np.eye(3, dtype=np.float32)
    N = len(p1)
    if N < 4:
        return best_H, np.zeros(N, dtype=bool)
    
    for _ in range(max_iterations):
        idx = np.random.choice(N, 4, replace=False)
        try:
            H = compute_homography_dlt(p1[idx], p2[idx])
            # Compute error
            p1_h = np.column_stack([p1, np.ones(N)])
            p2_proj = (H @ p1_h.T).T
            p2_proj = p2_proj[:, :2] / (p2_proj[:, 2:] + 1e-10)
            dist = np.linalg.norm(p2 - p2_proj, axis=1)
            
            inliers = dist < threshold
            
            if np.sum(inliers) > np.sum(best_inliers):
                best_inliers = inliers
                best_H = H
        except:
            continue
            
    # Refine with all inliers
    if np.sum(best_inliers) > 4:
        try:
            best_H = compute_homography_dlt(p1[best_inliers], p2[best_inliers])
        except:
            pass
            
    return best_H.astype(np.float32), np.array(best_inliers, dtype=bool)


def apply_homography(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Homography 행렬을 사용하여 점들을 변환합니다.
    
    Args:
        points: 점들 (N, 2) - 각 행은 [x, y] - float32
        H: Homography 행렬 (3, 3) - float32
    
    Returns:
        transformed_points: 변환된 점들 (N, 2) - 각 행은 [x, y] - float32
    """
    N = len(points)
    homogeneous_points = np.ones((N, 3), dtype=np.float32)
    homogeneous_points[:, :2] = points
    
    transformed_homogeneous = (H @ homogeneous_points.T).T
    w = transformed_homogeneous[:, 2]
    w = np.where(w == 0, 1e-10, w)
    
    transformed_points = transformed_homogeneous[:, :2] / w[:, np.newaxis]
    return transformed_points.astype(np.float32)


def compute_reprojection_error(points1: np.ndarray, points2: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Homography를 사용한 재투영 오차를 계산합니다.
    
    Args:
        points1: 첫 번째 이미지의 점들 (N, 2) - 각 행은 [x, y] - float32
        points2: 두 번째 이미지의 점들 (N, 2) - 각 행은 [x, y] - float32
        H: Homography 행렬 (3, 3) - float32
    
    Returns:
        errors: 각 점의 재투영 오차 (N,) - float32
    """
    transformed_points = apply_homography(points1, H)
    diff = transformed_points - points2
    errors = np.sqrt(np.sum(diff ** 2, axis=1))
    return errors.astype(np.float32)


def validate_homography(H: np.ndarray, points1: np.ndarray, points2: np.ndarray, 
                        max_reprojection_error: float = 5.0) -> Tuple[bool, float]:
    """
    Homography 행렬의 유효성을 검증합니다.
    
    Args:
        H: Homography 행렬 (3, 3) - float32
        points1: 첫 번째 이미지의 점들 (N, 2) - float32
        points2: 두 번째 이미지의 점들 (N, 2) - float32
        max_reprojection_error: 최대 허용 재투영 오차 (float, 기본값: 5.0)
    
    Returns:
        is_valid: Homography가 유효한지 여부 (bool)
        mean_error: 평균 재투영 오차 (float)
    """
    det = np.linalg.det(H)
    if abs(det) < 1e-6:
        return False, float('inf')
    
    scale_x = np.sqrt(H[0, 0]**2 + H[0, 1]**2)
    scale_y = np.sqrt(H[1, 0]**2 + H[1, 1]**2)
    
    if scale_x < 0.1 or scale_x > 10.0 or scale_y < 0.1 or scale_y > 10.0:
        return False, float('inf')
    
    if len(points1) > 0:
        errors = compute_reprojection_error(points1, points2, H)
        mean_error = np.mean(errors)
        
        if mean_error > max_reprojection_error:
            return False, mean_error
        
        return True, mean_error
    
    return True, 0.0
