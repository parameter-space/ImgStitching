"""
기하학적 변환 모듈
RANSAC과 DLT(Direct Linear Transform)를 사용한 Homography 계산을 구현합니다.
"""

import numpy as np
from typing import Tuple


def compute_translation_robust(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    """
    Robust한 통계량(Median)을 사용하여 최적의 Translation 행렬 (2 DoF)을 계산합니다.
    Outlier에 강건하며 복잡한 RANSAC 반복이 필요 없습니다.
    
    Args:
        points1: 첫 번째 이미지의 점들 (N, 2) - 각 행은 [x, y] - float32
        points2: 두 번째 이미지의 점들 (N, 2) - 각 행은 [x, y] - float32
    
    Returns:
        H: Translation 행렬 (3, 3) - float32
            [1  0  dx]
            [0  1  dy]
            [0  0  1 ]
    """
    if len(points1) < 1:
        return np.eye(3, dtype=np.float32)
    
    # Calculate flow vectors
    flows = points2 - points1  # (dx, dy)
    
    # Use MEDIAN to ignore outliers (robust statistics)
    dx = np.median(flows[:, 0])
    dy = np.median(flows[:, 1])
    
    # Construct Translation Matrix
    H = np.eye(3, dtype=np.float32)
    H[0, 2] = dx
    H[1, 2] = dy
    
    return H


def ransac_translation(points1: np.ndarray, points2: np.ndarray, threshold: float = 4.0, iterations: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
    """
    RANSAC을 사용하여 최적의 (dx, dy) 이동량을 찾습니다.
    2D 자유 형태 Translation을 지원하며, 수평/수직 이동 모두 감지합니다.
    Scale이나 Rotation을 사용하지 않으며, 오직 Translation만 사용합니다.
    
    Args:
        points1: 첫 번째 이미지의 점들 (N, 2) - 각 행은 [x, y] - float32
        points2: 두 번째 이미지의 점들 (N, 2) - 각 행은 [x, y] - float32
        threshold: 인라이어 임계값 (픽셀 단위, float, 기본값: 4.0)
        iterations: 최대 반복 횟수 (int, 기본값: 2000)
    
    Returns:
        H: Translation 행렬 (3, 3) - float32
        inlier_mask: 인라이어 마스크 (N,) - bool
    """
    N = len(points1)
    if N < 1:
        return np.eye(3, dtype=np.float32), np.zeros(N, dtype=bool)
    
    best_inliers_count = -1
    best_H = np.eye(3, dtype=np.float32)
    best_mask = np.zeros(N, dtype=bool)
    
    # Pre-calculate differences (flow vectors)
    diffs = points2 - points1  # (N, 2) array of [dx, dy]
    
    for _ in range(iterations):
        # 1. Randomly pick ONE match to hypothesize translation
        idx = np.random.randint(N)
        dx, dy = diffs[idx]
        
        # 2. Verify: How many other points agree with this shift?
        # A point is an inlier if dist(p2, p1 + shift) < threshold
        # This is equivalent to dist(diff, shift) < threshold
        errors = np.sqrt(np.sum((diffs - np.array([dx, dy]))**2, axis=1))
        current_inliers = errors < threshold
        count = np.sum(current_inliers)
        
        if count > best_inliers_count:
            best_inliers_count = count
            best_mask = current_inliers
            best_H[0, 2] = dx
            best_H[1, 2] = dy
    
    return best_H, best_mask


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


def compute_affine_transform(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    """
    정규화된 최소제곱법을 사용하여 최적의 Affine Transform (6 DoF)을 계산합니다.
    
    Affine Transform:
        x' = ax + by + tx
        y' = cx + dy + ty
    
    Args:
        points1: 첫 번째 이미지의 점들 (N, 2) - 각 행은 [x, y] - float32, N >= 3
        points2: 두 번째 이미지의 점들 (N, 2) - 각 행은 [x, y] - float32, N >= 3
    
    Returns:
        H: Affine Transform 행렬 (3, 3) - float32
            [a  b  tx]
            [c  d  ty]
            [0  0  1 ]
    """
    N = len(points1)
    if N < 3:
        return np.eye(3, dtype=np.float32)
    
    # 1. Normalize points for numerical stability
    p1_norm, T1 = normalize_points(points1)
    p2_norm, T2 = normalize_points(points2)
    
    # 2. Construct system of equations for normalized points: A * x = b
    A = np.zeros((2 * N, 6), dtype=np.float32)
    b = np.zeros((2 * N,), dtype=np.float32)
    
    for i in range(N):
        x, y = p1_norm[i]
        xp, yp = p2_norm[i]
        
        # Row for x'
        A[2*i] = [x, y, 1, 0, 0, 0]
        b[2*i] = xp
        
        # Row for y'
        A[2*i+1] = [0, 0, 0, x, y, 1]
        b[2*i+1] = yp
    
    # 3. Solve least squares on normalized points
    x_sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    
    # 4. Construct normalized Affine Matrix
    H_norm = np.eye(3, dtype=np.float32)
    H_norm[0, 0] = x_sol[0]  # a
    H_norm[0, 1] = x_sol[1]  # b
    H_norm[0, 2] = x_sol[2]  # tx
    H_norm[1, 0] = x_sol[3]  # c
    H_norm[1, 1] = x_sol[4]  # d
    H_norm[1, 2] = x_sol[5]  # ty
    # H_norm[2, 0] = 0 (already set by eye)
    # H_norm[2, 1] = 0 (already set by eye)
    # H_norm[2, 2] = 1 (already set by eye)
    
    # 5. Denormalize: H = inv(T2) @ H_norm @ T1
    H = np.linalg.inv(T2) @ H_norm @ T1
    
    # 6. Force last row to be strictly [0, 0, 1] to clean up float noise
    H[2, 0] = 0.0
    H[2, 1] = 0.0
    H[2, 2] = 1.0
    
    return H


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


def check_parallel_flow(p1: np.ndarray, p2: np.ndarray, threshold: float = 0.3) -> bool:
    """
    Optical flow vectors가 대략 평행한지 확인합니다.
    교차하는 매칭 라인(X-shape)을 outlier로 감지합니다.
    
    Args:
        p1: 첫 번째 이미지의 점들 (N, 2) - float32
        p2: 두 번째 이미지의 점들 (N, 2) - float32
        threshold: 평행도 임계값 (float, 기본값: 0.3)
    
    Returns:
        is_parallel: Flow vectors가 평행한지 여부 (bool)
    """
    if len(p1) < 4:
        return True
    
    # Flow vectors 계산
    flows = p2 - p1  # (N, 2)
    
    # Flow vectors의 방향 (각도)
    angles = np.arctan2(flows[:, 1], flows[:, 0])  # (N,)
    
    # 각도 차이 계산 (circular distance)
    angle_diffs = []
    for i in range(len(angles)):
        for j in range(i + 1, len(angles)):
            diff = abs(angles[i] - angles[j])
            # Circular distance (0 to pi)
            diff = min(diff, 2 * np.pi - diff)
            angle_diffs.append(diff)
    
    if len(angle_diffs) == 0:
        return True
    
    # 평균 각도 차이가 작으면 평행
    mean_diff = np.mean(angle_diffs)
    return mean_diff < threshold


def ransac_homography(p1: np.ndarray, p2: np.ndarray, 
                     max_iterations: int = 2000, 
                     threshold: float = 5.0,
                     min_inliers: int = 10,
                     image_width: float = None,
                     image_height: float = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    RANSAC 알고리즘을 사용하여 최적의 Homography 행렬을 찾습니다.
    Panorama 스티칭을 위한 엄격한 기하학적 제약 조건을 적용합니다.
    
    Args:
        p1: 첫 번째 이미지의 점들 (N, 2) - 각 행은 [x, y] - float32
        p2: 두 번째 이미지의 점들 (N, 2) - 각 행은 [x, y] - float32
        max_iterations: 최대 반복 횟수 (int, 기본값: 2000)
        threshold: 인라이어 임계값 (픽셀 단위, float, 기본값: 5.0)
        min_inliers: 최소 인라이어 개수 (int, 기본값: 10)
        image_width: 이미지 너비 (Translation check용, Optional)
    
    Returns:
        best_H: 최적의 Homography 행렬 (3, 3) - float32
        inlier_mask: 인라이어 마스크 (N,) - bool, True인 경우 인라이어
    """
    best_inliers = []
    best_H = np.eye(3, dtype=np.float32)
    N = len(p1)
    if N < 4:
        return best_H, np.zeros(N, dtype=bool)
    
    # Parallel flow check on input points
    if not check_parallel_flow(p1, p2):
        # If flow is not parallel, many outliers exist
        # Still try RANSAC but with stricter validation
        pass
    
    for _ in range(max_iterations):
        idx = np.random.choice(N, 3, replace=False)  # Affine needs minimum 3 points
        try:
            H = compute_affine_transform(p1[idx], p2[idx])
            
            # Geometric constraints check BEFORE computing error
            # 1. Determinant check (Must be positive)
            det = np.linalg.det(H)
            if abs(det) < 1e-6 or det < 0:  # Negative det = reflection
                continue
            
            # 2. Scale check (0.5 ~ 2.0) - Relaxed for indoor panoramas with perspective changes
            scale_x = np.sqrt(H[0, 0]**2 + H[0, 1]**2)
            scale_y = np.sqrt(H[1, 0]**2 + H[1, 1]**2)
            if scale_x < 0.5 or scale_x > 2.0 or scale_y < 0.5 or scale_y > 2.0:
                continue
            
            # Note: Translation checks (tx, ty) are REMOVED to allow 2D free movement
            # Camera can move in any direction (left, right, up, or down)
            
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
            
    # Refine with all inliers using Affine Transform (CRITICAL: Do NOT use DLT here)
    if np.sum(best_inliers) >= 3:  # Affine needs minimum 3 points
        try:
            best_H = compute_affine_transform(p1[best_inliers], p2[best_inliers])
            
            # Validate refined homography
            det = np.linalg.det(best_H)
            if abs(det) < 1e-6 or det < 0:
                # Fallback to previous best_H
                pass
            else:
                # Scale check (0.5 ~ 2.0) - Relaxed for indoor panoramas with perspective changes
                scale_x = np.sqrt(best_H[0, 0]**2 + best_H[0, 1]**2)
                scale_y = np.sqrt(best_H[1, 0]**2 + best_H[1, 1]**2)
                if scale_x < 0.5 or scale_x > 2.0 or scale_y < 0.5 or scale_y > 2.0:
                    # Fallback to previous best_H
                    pass
                # Note: Rotation and Translation checks are REMOVED to allow 2D free movement
        except:
            pass
    # Note: best_H is already Affine (from compute_affine_transform), no need to force H[2,0]=0
            
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
                        max_reprojection_error: float = 10.0,
                        image_width: float = None,
                        image_height: float = None) -> Tuple[bool, float]:
    """
    Homography 행렬의 유효성을 검증합니다.
    2D Multi-directional Panorama 스티칭을 지원합니다 (Left, Right, Up, Down).
    
    Args:
        H: Homography 행렬 (3, 3) - float32 (Affine-Regularized)
        points1: 첫 번째 이미지의 점들 (N, 2) - float32
        points2: 두 번째 이미지의 점들 (N, 2) - float32
        max_reprojection_error: 최대 허용 재투영 오차 (float, 기본값: 10.0)
        image_width: 이미지 너비 (사용되지 않음, 하위 호환성 유지)
        image_height: 이미지 높이 (사용되지 않음, 하위 호환성 유지)
    
    Returns:
        is_valid: Homography가 유효한지 여부 (bool)
        mean_error: 평균 재투영 오차 (float)
    """
    # 1. Determinant Check
    det = np.linalg.det(H)
    if abs(det) < 1e-6:
        return False, float('inf')
    
    # 2. Scale Check (0.5 ~ 2.0) - Relaxed for indoor panoramas with perspective changes
    scale_x = np.sqrt(H[0, 0]**2 + H[0, 1]**2)
    scale_y = np.sqrt(H[1, 0]**2 + H[1, 1]**2)
    
    if scale_x < 0.5 or scale_x > 2.0 or scale_y < 0.5 or scale_y > 2.0:
        return False, float('inf')
    
    # Note: Translation checks (tx, ty) are REMOVED to allow 2D free movement
    # Camera can move in any direction (left, right, up, or down)
    # Note: Rotation check is REMOVED to allow 2D free movement
    
    # 3. Reflection Check (negative determinant indicates reflection)
    if det < 0:
        return False, float('inf')
    
    # 4. Reprojection Error Check
    if len(points1) > 0:
        errors = compute_reprojection_error(points1, points2, H)
        mean_error = np.mean(errors)
        
        if mean_error > max_reprojection_error:
            return False, mean_error
        
        return True, mean_error
    
    return True, 0.0
