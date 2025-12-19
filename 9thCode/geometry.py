"""
기하학적 변환 모듈
Normalized DLT (Direct Linear Transform)를 사용한 8-DoF Homography 계산을 구현합니다.
Hartley & Zisserman 알고리즘을 정확히 따릅니다.
"""

import numpy as np
from typing import Tuple


def normalize_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    점들을 정규화합니다 (평균을 0으로, 평균 거리를 sqrt(2)로).
    Hartley & Zisserman의 Normalized DLT 알고리즘에 따라 구현됩니다.
    
    Args:
        points: 점들 (N, 2) - 각 행은 [x, y] - float32
    
    Returns:
        normalized_points: 정규화된 점들 (N, 2) - float32
        T: 정규화 변환 행렬 (3, 3) - float32
    """
    # Compute centroid
    centroid = np.mean(points, axis=0)
    
    # Center points
    centered = points - centroid
    
    # Compute mean distance from origin
    mean_dist = np.mean(np.sqrt(np.sum(centered**2, axis=1)))
    
    # Scale so average distance is sqrt(2)
    scale = np.sqrt(2) / mean_dist if mean_dist > 0 else 1.0
    
    # Construct transformation matrix T
    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Apply transformation
    points_homogeneous = np.column_stack([points, np.ones(len(points))])
    points_norm_homogeneous = (T @ points_homogeneous.T).T
    points_norm = points_norm_homogeneous[:, :2]
    
    return points_norm, T


def compute_homography_dlt(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    """
    Normalized DLT 알고리즘 (Hartley & Zisserman)을 사용하여 8-DoF Homography 행렬을 계산합니다.
    
    단계:
    1. Normalization: points1과 points2를 정규화
    2. Construct Matrix A: 각 correspondence에 대해 2개 행 생성
    3. Solve via SVD: 최소 singular value에 해당하는 해 선택
    4. Denormalization: 원본 좌표계로 변환
    
    Args:
        points1: 첫 번째 이미지의 점들 (N, 2) - 각 행은 [x, y] - float32, N >= 4
        points2: 두 번째 이미지의 점들 (N, 2) - 각 행은 [x, y] - float32, N >= 4
    
    Returns:
        H: Homography 행렬 (3, 3) - float32, H[2,2] = 1.0으로 정규화됨
    """
    N = len(points1)
    if N < 4:
        return np.eye(3, dtype=np.float32)
    
    # 1. Normalization: Compute centroid and scale for both point sets
    p1_norm, T1 = normalize_points(points1)
    p2_norm, T2 = normalize_points(points2)
    
    # 2. Construct Matrix A (2N x 9)
    # For each correspondence (x, y) -> (x', y') (normalized):
    #   [-x  -y  -1   0   0   0   x x'  y x'  x']
    #   [ 0   0   0  -x  -y  -1   x y'  y y'  y']
    A = np.zeros((2 * N, 9), dtype=np.float32)
    
    for i in range(N):
        x, y = p1_norm[i]
        xp, yp = p2_norm[i]
        
        # First row for x' equation
        A[2*i] = [-x, -y, -1, 0, 0, 0, x*xp, y*xp, xp]
        
        # Second row for y' equation
        A[2*i+1] = [0, 0, 0, -x, -y, -1, x*yp, y*yp, yp]
    
    # 3. Solve via SVD
    U, S, Vt = np.linalg.svd(A)
    
    # The solution h is the last row of V^T (corresponding to smallest singular value)
    # Vt[-1] is the last row (corresponding to smallest singular value)
    h = Vt[-1]  # Shape: (9,)
    
    # Reshape h to 3x3 matrix
    H_norm = h.reshape(3, 3)
    
    # 4. Denormalization: H = T2^{-1} @ H_norm @ T1
    H = np.linalg.inv(T2) @ H_norm @ T1
    
    # Normalize so that H[2, 2] = 1.0
    if abs(H[2, 2]) > 1e-10:
        H = H / H[2, 2]
    
    return H.astype(np.float32)


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
    w = np.where(np.abs(w) < 1e-10, 1e-10 * np.sign(w) + (w == 0), w)  # Avoid division by zero
    
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


def ransac_homography(p1: np.ndarray, p2: np.ndarray, 
                     max_iterations: int = 2000, 
                     threshold: float = 5.0,
                     min_inliers: int = 10,
                     image_width: float = None,
                     image_height: float = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    RANSAC 알고리즘을 사용하여 최적의 8-DoF Homography 행렬을 찾습니다.
    
    단계:
    1. 4개의 무작위 점을 선택하여 compute_homography_dlt로 Homography 계산
    2. 모든 점에 대해 재투영 오차 계산하여 inlier 결정
    3. 가장 많은 inlier를 가진 Homography 선택
    4. 모든 inlier에 대해 Refinement 수행 (compute_homography_dlt 재실행)
    
    Args:
        p1: 첫 번째 이미지의 점들 (N, 2) - 각 행은 [x, y] - float32
        p2: 두 번째 이미지의 점들 (N, 2) - 각 행은 [x, y] - float32
        max_iterations: 최대 반복 횟수 (int, 기본값: 2000)
        threshold: 인라이어 임계값 (픽셀 단위, float, 기본값: 5.0)
        min_inliers: 최소 인라이어 개수 (int, 기본값: 10)
        image_width: 이미지 너비 (사용되지 않음, 하위 호환성 유지)
        image_height: 이미지 높이 (사용되지 않음, 하위 호환성 유지)
    
    Returns:
        best_H: 최적의 Homography 행렬 (3, 3) - float32
        inlier_mask: 인라이어 마스크 (N,) - bool, True인 경우 인라이어
    """
    best_inliers_count = 0
    best_H = np.eye(3, dtype=np.float32)
    best_inliers_mask = np.zeros(len(p1), dtype=bool)
    
    N = len(p1)
    if N < 4:
        return best_H, best_inliers_mask
    
    for _ in range(max_iterations):
        # Randomly select 4 points for Homography calculation
        idx = np.random.choice(N, 4, replace=False)
        
        try:
            # Compute Homography using Normalized DLT
            H = compute_homography_dlt(p1[idx], p2[idx])
            
            # Compute reprojection error for all points
            errors = compute_reprojection_error(p1, p2, H)
            
            # Determine inliers
            inliers_mask = errors < threshold
            inliers_count = np.sum(inliers_mask)
            
            # Update best model if this has more inliers
            if inliers_count > best_inliers_count:
                best_inliers_count = inliers_count
                best_inliers_mask = inliers_mask
                best_H = H
                
        except Exception:
            # Skip if computation fails (e.g., singular matrix)
            continue
    
    # Refinement: Re-run compute_homography_dlt on ALL inliers
    # This stabilizes the 8-DoF solution and prevents drift
    if best_inliers_count >= 4:
        try:
            refined_H = compute_homography_dlt(p1[best_inliers_mask], p2[best_inliers_mask])
            
            # Validate refined homography (basic sanity checks)
            det = np.linalg.det(refined_H)
            if det > 0.1:  # No reflection
                # Check scale (eigenvalues of 2x2 submatrix)
                scale_x = np.sqrt(refined_H[0, 0]**2 + refined_H[0, 1]**2)
                scale_y = np.sqrt(refined_H[1, 0]**2 + refined_H[1, 1]**2)
                
                if 0.5 <= scale_x <= 2.0 and 0.5 <= scale_y <= 2.0:
                    # Check reprojection error
                    errors_refined = compute_reprojection_error(p1[best_inliers_mask], p2[best_inliers_mask], refined_H)
                    mean_error = np.mean(errors_refined)
                    
                    if mean_error < 10.0:
                        best_H = refined_H
        except Exception:
            # Keep original best_H if refinement fails
            pass
    
    return best_H.astype(np.float32), best_inliers_mask


def validate_homography(H: np.ndarray, points1: np.ndarray, points2: np.ndarray, 
                        max_reprojection_error: float = 10.0,
                        image_width: float = None,
                        image_height: float = None) -> Tuple[bool, float]:
    """
    Homography 행렬의 유효성을 검증합니다.
    2D 자유 형태 파노라마 스티칭을 지원합니다 (Left, Right, Up, Down).
    
    기본적인 Sanity Check만 수행:
    - Determinant > 0.1 (No reflection)
    - Scale (0.5 ~ 2.0)
    - Reprojection Error < 10.0
    
    Args:
        H: Homography 행렬 (3, 3) - float32
        points1: 첫 번째 이미지의 점들 (N, 2) - float32
        points2: 두 번째 이미지의 점들 (N, 2) - float32
        max_reprojection_error: 최대 허용 재투영 오차 (float, 기본값: 10.0)
        image_width: 이미지 너비 (사용되지 않음, 하위 호환성 유지)
        image_height: 이미지 높이 (사용되지 않음, 하위 호환성 유지)
    
    Returns:
        is_valid: Homography가 유효한지 여부 (bool)
        mean_error: 평균 재투영 오차 (float)
    """
    # 1. Determinant Check (Must be positive and not too small)
    det = np.linalg.det(H)
    if det <= 0.1:  # No reflection, and not too close to singular
        return False, float('inf')
    
    # 2. Scale Check (0.5 ~ 2.0) - Eigenvalues of 2x2 submatrix
    scale_x = np.sqrt(H[0, 0]**2 + H[0, 1]**2)
    scale_y = np.sqrt(H[1, 0]**2 + H[1, 1]**2)
    
    if scale_x < 0.5 or scale_x > 2.0 or scale_y < 0.5 or scale_y > 2.0:
        return False, float('inf')
    
    # Note: Translation checks (tx, ty) are REMOVED to allow 2D free movement
    # Note: Rotation checks are REMOVED to allow arbitrary directions
    
    # 3. Reprojection Error Check
    if len(points1) > 0:
        errors = compute_reprojection_error(points1, points2, H)
        mean_error = np.mean(errors)
        
        if mean_error > max_reprojection_error:
            return False, mean_error
        
        return True, mean_error
    
    return True, 0.0
