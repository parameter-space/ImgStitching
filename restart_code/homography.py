"""
Homography 계산 모듈
Normalized DLT 알고리즘을 사용한 8-DoF Homography 계산 (필수 구현)
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
    # Compute centroid: 점들의 중심 (평균 좌표)
    # axis=0: 각 열(차원)에 대해 평균 계산, 결과: (2,) 배열 [mean_x, mean_y]
    centroid = np.mean(points, axis=0)
    
    # Center points: 중심을 원점으로 이동
    centered = points - centroid  # 브로드캐스팅: (N, 2) - (2,) = 각 행에서 centroid를 뺌
    
    # Compute mean distance from origin: 원점으로부터의 평균 거리
    # np.sum(centered**2, axis=1): 각 점의 거리 제곱 (x^2 + y^2)
    # np.sqrt(...): 거리 계산
    mean_dist = np.mean(np.sqrt(np.sum(centered**2, axis=1)))
    
    # Scale so average distance is sqrt(2)
    # Hartley & Zisserman의 정규화: 평균 거리를 sqrt(2)로 맞춤 (수치 안정성 향상)
    scale = np.sqrt(2) / mean_dist if mean_dist > 0 else 1.0
    
    # Construct transformation matrix T: 정규화 변환 행렬
    # [scale, 0, -scale*centroid_x]  : x를 스케일하고 평행이동
    # [0, scale, -scale*centroid_y]  : y를 스케일하고 평행이동
    # [0, 0, 1]                      : 동차 좌표 유지
    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Apply transformation: 동차 좌표계로 변환 후 행렬 곱셈
    # np.column_stack: [x, y, 1] 형태로 변환 (동차 좌표)
    points_homogeneous = np.column_stack([points, np.ones(len(points))])
    # T @ points_homogeneous.T: 행렬 곱셈, .T로 전치 후 다시 .T로 원래 형태로
    points_norm_homogeneous = (T @ points_homogeneous.T).T
    # 동차 좌표에서 일반 좌표로 변환 (마지막 차원 제거)
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
    
    # 1. Normalization
    p1_norm, T1 = normalize_points(points1)
    p2_norm, T2 = normalize_points(points2)
    
    # 2. Construct Matrix A (2N x 9): DLT 방정식 Ax = 0 구성
    # 각 correspondence (x, y) -> (x', y')는 2개의 방정식을 만듦
    A = np.zeros((2 * N, 9), dtype=np.float32)
    
    for i in range(N):
        x, y = p1_norm[i]      # 정규화된 source 점
        xp, yp = p2_norm[i]    # 정규화된 target 점
        
        # First row for x' equation: x' = (h11*x + h12*y + h13) / (h31*x + h32*y + h33)
        # 선형화: x'*(h31*x + h32*y + h33) = h11*x + h12*y + h13
        # 정리: -h11*x - h12*y - h13 + x'*h31*x + x'*h32*y + x'*h33 = 0
        A[2*i] = [-x, -y, -1, 0, 0, 0, x*xp, y*xp, xp]
        
        # Second row for y' equation: y' = (h21*x + h22*y + h23) / (h31*x + h32*y + h33)
        # 선형화: y'*(h31*x + h32*y + h33) = h21*x + h22*y + h23
        A[2*i+1] = [0, 0, 0, -x, -y, -1, x*yp, y*yp, yp]
    
    # 3. Solve via SVD: Ax = 0 형태의 과제약 시스템 풀기
    # np.linalg.svd: Singular Value Decomposition 수행
    # A = U @ diag(S) @ Vt 형태로 분해
    U, S, Vt = np.linalg.svd(A)
    
    # The solution h is the last row of V^T (corresponding to smallest singular value)
    # 가장 작은 singular value에 해당하는 Vt의 마지막 행이 해 (||h||=1, h[8]=1로 정규화)
    h = Vt[-1]  # Shape: (9,) - Homography 행렬의 9개 요소를 1D로 펼친 것
    
    # Reshape h to 3x3 matrix: [h11, h12, h13, h21, h22, h23, h31, h32, h33] -> 3x3
    H_norm = h.reshape(3, 3)
    
    # 4. Denormalization: 정규화된 좌표계에서 계산된 H를 원본 좌표계로 변환
    # H = T2^{-1} @ H_norm @ T1
    # T1: points1 정규화 행렬, T2: points2 정규화 행렬
    H = np.linalg.inv(T2) @ H_norm @ T1
    
    # Normalize so that H[2, 2] = 1.0: Homography는 스케일 불변이므로 정규화
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
    # 동차 좌표계로 변환: [x, y] -> [x, y, 1]
    homogeneous_points = np.ones((N, 3), dtype=np.float32)
    homogeneous_points[:, :2] = points  # [:2]: 처음 2개 열에 x, y 저장
    
    # Homography 적용: [x', y', w'] = H @ [x, y, 1]
    # .T로 전치하여 열 벡터로 변환 후 행렬 곱, 다시 .T로 원래 형태로
    transformed_homogeneous = (H @ homogeneous_points.T).T
    w = transformed_homogeneous[:, 2]  # 동차 좌표의 w 값
    
    # Division by zero 방지: w가 거의 0인 경우 작은 값으로 대체
    # np.where(condition, x, y): condition이 True면 x, False면 y
    # np.sign(w): 부호 유지, (w == 0): 0일 때만 1, 나머지는 0
    w = np.where(np.abs(w) < 1e-10, 1e-10 * np.sign(w) + (w == 0), w)
    
    # 동차 좌표를 일반 좌표로 변환: [x', y'] = [x'/w, y'/w]
    # w[:, np.newaxis]: (N,) -> (N, 1)로 변환하여 브로드캐스팅
    transformed_points = transformed_homogeneous[:, :2] / w[:, np.newaxis]
    return transformed_points.astype(np.float32)


def compute_reprojection_error(points1: np.ndarray, points2: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Homography를 사용한 재투영 오차를 계산합니다.
    
    Args:
        points1: 첫 번째 이미지의 점들 (N, 2) - float32
        points2: 두 번째 이미지의 점들 (N, 2) - float32
        H: Homography 행렬 (3, 3) - float32
    
    Returns:
        errors: 각 점의 재투영 오차 (N,) - float32
    """
    transformed_points = apply_homography(points1, H)
    diff = transformed_points - points2
    errors = np.sqrt(np.sum(diff ** 2, axis=1))
    return errors.astype(np.float32)

