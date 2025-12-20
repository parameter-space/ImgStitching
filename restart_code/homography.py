"""
Homography 계산 구현 (DLT 알고리즘)
Homography.pdf 강의자료 참고
"""

import numpy as np
from typing import Tuple


def normalize_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    점들을 정규화합니다 (Normalized DLT).
    
    Args:
        points: 점들 (N, 2) - 각 행은 [x, y] - float32
    
    Returns:
        points_norm: 정규화된 점들 (N, 2) - float32
        T: 정규화 변환 행렬 (3, 3) - float32
    """
    N = len(points)
    
    # 중심 계산
    mean = np.mean(points, axis=0)
    
    # 평균 거리 계산
    centered = points - mean
    distances = np.sqrt(np.sum(centered ** 2, axis=1))
    mean_distance = np.mean(distances)
    
    if mean_distance < 1e-10:
        # 모든 점이 같은 위치에 있는 경우
        scale = 1.0
    else:
        # 평균 거리를 sqrt(2)로 만들도록 scale 계산
        scale = np.sqrt(2.0) / mean_distance
    
    # 정규화 변환 행렬 구성
    T = np.array([
        [scale, 0, -scale * mean[0]],
        [0, scale, -scale * mean[1]],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # 정규화
    points_homogeneous = np.column_stack([points, np.ones(N)])
    points_norm_homogeneous = (T @ points_homogeneous.T).T
    points_norm = points_norm_homogeneous[:, :2]
    
    return points_norm, T


def compute_homography_dlt(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    """
    Normalized DLT 알고리즘을 사용하여 Homography 행렬을 계산합니다.
    
    수식:
    - x' = H * x (동차 좌표)
    - 각 correspondence는 2개의 선형 방정식을 만듦
    - A * h = 0 형태로 구성하여 SVD로 해 구하기
    
    Args:
        points1: 첫 번째 이미지의 점들 (N, 2) - 각 행은 [x, y] - float32, N >= 4
        points2: 두 번째 이미지의 점들 (N, 2) - 각 행은 [x, y] - float32, N >= 4
    
    Returns:
        H: Homography 행렬 (3, 3) - float32, H[2,2] = 1.0으로 정규화됨
    """
    N = len(points1)
    
    if N < 4:
        # 최소 4개의 점이 필요
        return np.eye(3, dtype=np.float32)
    
    # 1. Normalization
    p1_norm, T1 = normalize_points(points1)
    p2_norm, T2 = normalize_points(points2)
    
    # 2. Matrix A 구성 (2N x 9)
    A = np.zeros((2 * N, 9), dtype=np.float32)
    
    for i in range(N):
        x, y = p1_norm[i]      # 정규화된 source 점
        xp, yp = p2_norm[i]    # 정규화된 target 점
        
        # First row: x' 방정식
        # x'*(h31*x + h32*y + h33) = h11*x + h12*y + h13
        # 선형화: -h11*x - h12*y - h13 + x'*h31*x + x'*h32*y + x'*h33 = 0
        A[2*i] = [-x, -y, -1, 0, 0, 0, x*xp, y*xp, xp]
        
        # Second row: y' 방정식
        # y'*(h31*x + h32*y + h33) = h21*x + h22*y + h23
        # 선형화: -h21*x - h22*y - h23 + y'*h31*x + y'*h32*y + y'*h33 = 0
        A[2*i+1] = [0, 0, 0, -x, -y, -1, x*yp, y*yp, yp]
    
    # 3. SVD를 사용하여 해 구하기
    # A * h = 0 형태이므로, 가장 작은 singular value에 해당하는 right singular vector가 해
    U, S, Vt = np.linalg.svd(A)
    
    # Vt의 마지막 행 (가장 작은 singular value에 해당)
    h = Vt[-1]  # Shape: (9,)
    
    # 4. Homography 행렬로 재구성
    H_norm = h.reshape(3, 3)
    
    # 5. Denormalization
    # H = T2^(-1) * H_norm * T1
    T1_inv = np.linalg.inv(T1)
    T2_inv = np.linalg.inv(T2)
    
    H = T2_inv @ H_norm @ T1
    
    # 6. 정규화 (H[2,2] = 1)
    if abs(H[2, 2]) > 1e-10:
        H = H / H[2, 2]
    
    return H.astype(np.float32)


def apply_homography(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Homography를 사용하여 점들을 변환합니다.
    
    Args:
        points: 점들 (N, 2) - 각 행은 [x, y] - float32
        H: Homography 행렬 (3, 3) - float32
    
    Returns:
        transformed: 변환된 점들 (N, 2) - float32
    """
    N = len(points)
    
    # 동차 좌표로 변환
    points_homogeneous = np.column_stack([points, np.ones(N)])
    
    # 변환
    transformed_homogeneous = (H @ points_homogeneous.T).T
    
    # 일반 좌표로 변환
    w = transformed_homogeneous[:, 2]
    # w ≈ 0인 경우는 invalid하므로 NaN으로 표시 (stitching.py와 일관성 유지)
    # 실제 warping에서는 valid_mask로 필터링하지만, 여기서는 NaN 반환하여 명확히 표시
    w_safe = np.where(np.abs(w) > 1e-10, w, np.nan)
    transformed = transformed_homogeneous[:, :2] / w_safe[:, np.newaxis]
    
    return transformed


def compute_homography_affine(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    """
    Normalized DLT 알고리즘을 사용하여 Affine Homography 행렬을 계산합니다.
    
    Affine 변환은 Perspective 성분이 없는 변환으로, Scale/Shear/Rotation/Translation만 허용합니다.
    Last row는 [0, 0, 1]로 고정됩니다.
    
    수식:
    - Affine 변환: [x']   [a  b  tx] [x]
                   [y'] = [c  d  ty] [y]
                   [1 ]   [0  0  1 ] [1]
    - 각 correspondence는 2개의 선형 방정식을 만듦
    - A * h = b 형태로 구성하여 최소제곱법으로 해 구하기
    
    Args:
        points1: 첫 번째 이미지의 점들 (N, 2) - 각 행은 [x, y] - float32, N >= 3
        points2: 두 번째 이미지의 점들 (N, 2) - 각 행은 [x, y] - float32, N >= 3
    
    Returns:
        H: Affine Homography 행렬 (3, 3) - float32, H[2,2] = 1.0, H[2,0] = H[2,1] = 0.0
    """
    N = len(points1)
    
    if N < 3:
        # 최소 3개 점이 필요 (Affine은 6 DoF)
        return np.eye(3, dtype=np.float32)
    
    # 1. Normalization
    p1_norm, T1 = normalize_points(points1)
    p2_norm, T2 = normalize_points(points2)
    
    # 2. Matrix A 구성 (2N x 6)
    # Affine 변환: [a, b, tx, c, d, ty] 6개 파라미터
    A = np.zeros((2 * N, 6), dtype=np.float32)
    b = np.zeros(2 * N, dtype=np.float32)
    
    for i in range(N):
        x, y = p1_norm[i]      # 정규화된 source 점
        xp, yp = p2_norm[i]    # 정규화된 target 점
        
        # First row: x' 방정식
        # x' = a*x + b*y + tx
        A[2*i] = [x, y, 1, 0, 0, 0]
        b[2*i] = xp
        
        # Second row: y' 방정식
        # y' = c*x + d*y + ty
        A[2*i+1] = [0, 0, 0, x, y, 1]
        b[2*i+1] = yp
    
    # 3. 최소제곱법으로 해 구하기
    # A * h = b 형태이므로, h = (A^T * A)^(-1) * A^T * b
    try:
        h = np.linalg.lstsq(A, b, rcond=None)[0]  # Shape: (6,)
    except:
        # 최소제곱법 실패 시 Identity 반환
        return np.eye(3, dtype=np.float32)
    
    # 4. Affine Homography 행렬로 재구성
    H_norm = np.array([
        [h[0], h[1], h[2]],  # a, b, tx
        [h[3], h[4], h[5]],  # c, d, ty
        [0.0, 0.0, 1.0]      # 0, 0, 1
    ], dtype=np.float32)
    
    # 5. Denormalization
    # H = T2^(-1) * H_norm * T1
    T1_inv = np.linalg.inv(T1)
    T2_inv = np.linalg.inv(T2)
    
    H = T2_inv @ H_norm @ T1
    
    # 6. 정규화 (H[2,2] = 1, H[2,0] = H[2,1] = 0)
    if abs(H[2, 2]) > 1e-10:
        H = H / H[2, 2]
    
    # Affine 변환이므로 마지막 행은 [0, 0, 1]이어야 함
    H[2, 0] = 0.0
    H[2, 1] = 0.0
    H[2, 2] = 1.0
    
    return H.astype(np.float32)


def interpolate_homography(H_proj: np.ndarray, H_aff: np.ndarray, alpha: float) -> np.ndarray:
    """
    Projective Homography와 Affine Homography를 보간하여 안정적인 Homography를 생성합니다.
    
    수식: H_interp = (1 - alpha) * H_proj + alpha * H_aff
    
    alpha가 0에 가까우면 Projective에 가깝고, 1에 가까우면 Affine에 가깝습니다.
    데이터가 불안정할 때 Affine 모델과 Projective 모델을 섞어서 안정성을 확보합니다.
    
    Args:
        H_proj: Projective Homography (3, 3) - float32
        H_aff: Affine Homography (3, 3) - float32
        alpha: 보간 계수 (0.0 = Projective, 1.0 = Affine) - float
    
    Returns:
        H_interp: 보간된 Homography (3, 3) - float32
    """
    # alpha를 [0, 1] 범위로 클램핑
    alpha = np.clip(alpha, 0.0, 1.0)
    
    # 두 행렬을 보간
    H_interp = (1.0 - alpha) * H_proj + alpha * H_aff
    
    # 정규화 (H[2,2] = 1)
    if abs(H_interp[2, 2]) > 1e-10:
        H_interp = H_interp / H_interp[2, 2]
    
    return H_interp.astype(np.float32)