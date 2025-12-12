"""
기하학적 변환 모듈
RANSAC과 DLT(Direct Linear Transform)를 사용한 Homography 계산을 구현합니다.
"""

import numpy as np
from typing import Tuple, List, Optional


def compute_homography_dlt(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    """
    DLT(Direct Linear Transform) 알고리즘을 사용하여 Homography 행렬을 계산합니다.
    
    최소 4개의 점이 필요하며, 4개보다 많은 경우 과도 결정 시스템을 풉니다.
    SVD를 사용하여 최소 제곱 해를 구합니다.
    
    Args:
        points1: 첫 번째 이미지의 점들 (N, 2) - 각 행은 [x, y] - float32, N >= 4
        points2: 두 번째 이미지의 점들 (N, 2) - 각 행은 [x, y] - float32, N >= 4
    
    Returns:
        H: Homography 행렬 (3, 3) - float32
    """
    if len(points1) < 4 or len(points2) < 4:
        raise ValueError("최소 4개의 점이 필요합니다.")
    
    if len(points1) != len(points2):
        raise ValueError("두 점 집합의 크기가 일치해야 합니다.")
    
    N = len(points1)
    
    # A 행렬 구성 (2N x 9)
    A = np.zeros((2*N, 9), dtype=np.float32)
    
    for i in range(N):
        x, y = points1[i, 0], points1[i, 1]
        xp, yp = points2[i, 0], points2[i, 1]
        
        # 첫 번째 방정식: x' 관련
        A[2*i, :] = [x, y, 1, 0, 0, 0, -xp*x, -xp*y, -xp]
        
        # 두 번째 방정식: y' 관련
        A[2*i+1, :] = [0, 0, 0, x, y, 1, -yp*x, -yp*y, -yp]
    
    # SVD를 사용하여 Ax = 0의 해 구하기
    # A의 가장 작은 특이값에 해당하는 오른쪽 특이벡터가 해
    U, S, Vt = np.linalg.svd(A)
    
    # Vt의 마지막 행 (가장 작은 특이값에 해당)이 해
    h = Vt[-1, :]
    
    # Homography 행렬로 재구성 (3x3)
    H = h.reshape(3, 3)
    
    # 정규화 (H[2, 2] = 1로 만들기)
    if H[2, 2] != 0:
        H = H / H[2, 2]
    
    return H.astype(np.float32)


def normalize_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    점들을 정규화합니다 (평균을 0으로, 평균 거리를 sqrt(2)로).
    
    정규화는 수치적 안정성을 향상시킵니다.
    
    Args:
        points: 점들 (N, 2) - 각 행은 [x, y] - float32
    
    Returns:
        normalized_points: 정규화된 점들 (N, 2) - float32
        T: 정규화 변환 행렬 (3, 3) - float32
    """
    # 중심 계산
    mean = np.mean(points, axis=0)  # (2,)
    
    # 중심을 원점으로 이동
    centered_points = points - mean
    
    # 평균 거리 계산
    distances = np.sqrt(np.sum(centered_points ** 2, axis=1))
    mean_distance = np.mean(distances)
    
    # sqrt(2)로 스케일링
    if mean_distance > 0:
        scale = np.sqrt(2) / mean_distance
    else:
        scale = 1.0
    
    # 정규화 변환 행렬 구성
    T = np.array([
        [scale, 0, -scale * mean[0]],
        [0, scale, -scale * mean[1]],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # 정규화된 점 계산
    normalized_points = centered_points * scale
    
    return normalized_points.astype(np.float32), T.astype(np.float32)


def compute_homography_dlt_normalized(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    """
    정규화된 DLT를 사용하여 Homography 행렬을 계산합니다.
    
    정규화를 통해 수치적 안정성을 향상시킵니다.
    
    Args:
        points1: 첫 번째 이미지의 점들 (N, 2) - 각 행은 [x, y] - float32
        points2: 두 번째 이미지의 점들 (N, 2) - 각 행은 [x, y] - float32
    
    Returns:
        H: Homography 행렬 (3, 3) - float32
    """
    # 점 정규화
    norm_points1, T1 = normalize_points(points1)
    norm_points2, T2 = normalize_points(points2)
    
    # 정규화된 점으로 Homography 계산
    H_norm = compute_homography_dlt(norm_points1, norm_points2)
    
    # 정규화 해제: H = T2^(-1) * H_norm * T1
    T2_inv = np.linalg.inv(T2)
    H = T2_inv @ H_norm @ T1
    
    # 정규화
    if H[2, 2] != 0:
        H = H / H[2, 2]
    
    return H.astype(np.float32)


def apply_homography(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Homography 행렬을 사용하여 점들을 변환합니다.
    
    [x', y', w']^T = H * [x, y, 1]^T
    [x', y'] = [x'/w', y'/w']
    
    Args:
        points: 점들 (N, 2) - 각 행은 [x, y] - float32
        H: Homography 행렬 (3, 3) - float32
    
    Returns:
        transformed_points: 변환된 점들 (N, 2) - 각 행은 [x, y] - float32
    """
    N = len(points)
    
    # 동차 좌표로 변환 (N, 3)
    homogeneous_points = np.ones((N, 3), dtype=np.float32)
    homogeneous_points[:, :2] = points
    
    # Homography 적용
    transformed_homogeneous = (H @ homogeneous_points.T).T  # (N, 3)
    
    # 동차 좌표를 일반 좌표로 변환
    w = transformed_homogeneous[:, 2]
    # w가 0인 경우 처리
    w = np.where(w == 0, 1e-10, w)
    
    transformed_points = transformed_homogeneous[:, :2] / w[:, np.newaxis]
    
    return transformed_points.astype(np.float32)


def validate_homography(H: np.ndarray, points1: np.ndarray, points2: np.ndarray, 
                        max_reprojection_error: float = 5.0) -> Tuple[bool, float]:
    """
    Homography 행렬의 유효성을 검증합니다.
    
    Args:
        H: Homography 행렬 (3, 3) - float32
        points1: 첫 번째 이미지의 점들 (N, 2) - float32
        points2: 두 번째 이미지의 점들 (N, 2) - float32
        max_reprojection_error: 최대 허용 재투영 오차 (float, 기본값: 10.0)
    
    Returns:
        is_valid: Homography가 유효한지 여부 (bool)
        mean_error: 평균 재투영 오차 (float)
    """
    # 1. 행렬식(determinant) 확인
    det = np.linalg.det(H)
    if abs(det) < 1e-6:
        return False, float('inf')
    
    # 2. 스케일 확인 (대각선 요소가 비정상적으로 크거나 작지 않은지)
    scale_x = np.sqrt(H[0, 0]**2 + H[0, 1]**2)
    scale_y = np.sqrt(H[1, 0]**2 + H[1, 1]**2)
    
    if scale_x < 0.1 or scale_x > 10.0 or scale_y < 0.1 or scale_y > 10.0:
        return False, float('inf')
    
    # 3. 재투영 오차 확인
    if len(points1) > 0:
        errors = compute_reprojection_error(points1, points2, H)
        mean_error = np.mean(errors)
        
        if mean_error > max_reprojection_error:
            return False, mean_error
        
        return True, mean_error
    
    return True, 0.0


def compute_reprojection_error(points1: np.ndarray, points2: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Homography를 사용한 재투영 오차를 계산합니다.
    
    points1을 H로 변환한 결과와 points2 간의 유클리드 거리를 계산합니다.
    
    Args:
        points1: 첫 번째 이미지의 점들 (N, 2) - 각 행은 [x, y] - float32
        points2: 두 번째 이미지의 점들 (N, 2) - 각 행은 [x, y] - float32
        H: Homography 행렬 (3, 3) - float32
    
    Returns:
        errors: 각 점의 재투영 오차 (N,) - float32 (유클리드 거리)
    """
    # points1을 H로 변환
    transformed_points = apply_homography(points1, H)
    
    # 유클리드 거리 계산
    diff = transformed_points - points2
    errors = np.sqrt(np.sum(diff ** 2, axis=1))
    
    return errors.astype(np.float32)


def ransac_homography(points1: np.ndarray, points2: np.ndarray, 
                     max_iterations: int = 1000, 
                     threshold: float = 5.0, 
                     min_inliers: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    RANSAC 알고리즘을 사용하여 최적의 Homography 행렬을 찾습니다.
    
    알고리즘:
    1. 임의로 4개의 점 쌍을 선택
    2. DLT를 사용하여 Homography H 계산
    3. 나머지 모든 점에 대해 재투영 오차 계산
    4. 임계값 이하인 점들을 인라이어로 카운트
    5. 최대 인라이어 개수를 가진 H를 선택
    6. 최종적으로 모든 인라이어를 사용하여 H를 재계산
    
    Args:
        points1: 첫 번째 이미지의 점들 (N, 2) - 각 행은 [x, y] - float32
        points2: 두 번째 이미지의 점들 (N, 2) - 각 행은 [x, y] - float32
        max_iterations: 최대 반복 횟수 (int, 기본값: 1000)
        threshold: 인라이어 임계값 (픽셀 단위, float, 기본값: 5.0)
        min_inliers: 최소 인라이어 개수 (int, 기본값: 10)
    
    Returns:
        best_H: 최적의 Homography 행렬 (3, 3) - float32
        inlier_mask: 인라이어 마스크 (N,) - bool, True인 경우 인라이어
    """
    if len(points1) < 4 or len(points2) < 4:
        raise ValueError("최소 4개의 점이 필요합니다.")
    
    if len(points1) != len(points2):
        raise ValueError("두 점 집합의 크기가 일치해야 합니다.")
    
    N = len(points1)
    
    if N < min_inliers:
        min_inliers = N
    
    best_H = None
    best_inlier_count = 0
    best_inlier_mask = None
    
    # RANSAC 반복
    for iteration in range(max_iterations):
        # 1. 임의로 4개의 점 쌍 선택
        indices = np.random.choice(N, size=4, replace=False)
        sample_p1 = points1[indices]
        sample_p2 = points2[indices]
        
        try:
            # 2. DLT를 사용하여 Homography 계산
            H = compute_homography_4points(sample_p1, sample_p2)
            
            # 3. 모든 점에 대해 재투영 오차 계산
            # H는 points1을 points2로 변환하는 Homography이므로
            # compute_reprojection_error(points1, points2, H)로 호출
            errors = compute_reprojection_error(points1, points2, H)
            
            # 4. 인라이어 마스크 생성
            inlier_mask = errors < threshold
            inlier_count = np.sum(inlier_mask)
            
            # 5. 최적의 H 업데이트
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_H = H.copy()
                best_inlier_mask = inlier_mask.copy()
                
                # 충분한 인라이어를 찾았으면 조기 종료 (선택적)
                if inlier_count >= min_inliers and inlier_count >= N * 0.8:
                    break
                    
        except (np.linalg.LinAlgError, ValueError):
            # SVD 실패 또는 기타 오류 시 다음 반복으로
            continue
    
    # 최적의 H를 찾지 못한 경우
    if best_H is None:
        # 기본 Homography (단위 행렬)
        best_H = np.eye(3, dtype=np.float32)
        best_inlier_mask = np.zeros(N, dtype=bool)
        return best_H, best_inlier_mask
    
    # 6. 최종적으로 모든 인라이어를 사용하여 H를 재계산 (선택적, 더 정확함)
    if best_inlier_count >= 4:
        inlier_p1 = points1[best_inlier_mask]
        inlier_p2 = points2[best_inlier_mask]
        
        if len(inlier_p1) >= 4:
            try:
                # 모든 인라이어를 사용하여 최종 H 계산
                best_H = compute_homography_dlt(inlier_p1, inlier_p2)
            except:
                # 재계산 실패 시 이전 H 사용
                pass
    
    return best_H.astype(np.float32), best_inlier_mask


def compute_homography_4points(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """
    4개의 대응점으로부터 Homography를 계산합니다.
    DLT (Direct Linear Transform) 알고리즘을 사용합니다.
    
    Homography 변환: [x', y', 1]^T = H * [x, y, 1]^T
    각 점 쌍에 대해 2개의 선형 방정식이 생성되어 총 8개의 방정식으로 8개의 미지수를 구합니다.
    
    Args:
        p1: 첫 번째 이미지의 4개 점 (4, 2) - 각 행은 [x, y] - float32
        p2: 두 번째 이미지의 4개 점 (4, 2) - 각 행은 [x, y] - float32
    
    Returns:
        H: Homography 행렬 (3, 3) - float32
    """
    if len(p1) != 4 or len(p2) != 4:
        raise ValueError("정확히 4개의 점이 필요합니다.")
    
    # A 행렬 구성 (8 x 9)
    A = np.zeros((8, 9), dtype=np.float32)
    
    for i in range(4):
        x, y = p1[i, 0], p1[i, 1]
        xp, yp = p2[i, 0], p2[i, 1]
        
        # 첫 번째 방정식: x' 관련
        A[2*i, :] = [x, y, 1, 0, 0, 0, -xp*x, -xp*y, -xp]
        
        # 두 번째 방정식: y' 관련
        A[2*i+1, :] = [0, 0, 0, x, y, 1, -yp*x, -yp*y, -yp]
    
    # SVD를 사용하여 Ax = 0의 해 구하기
    # A의 가장 작은 특이값에 해당하는 오른쪽 특이벡터가 해
    U, S, Vt = np.linalg.svd(A)
    
    # Vt의 마지막 행 (가장 작은 특이값에 해당)이 해
    h = Vt[-1, :]
    
    # Homography 행렬로 재구성 (3x3)
    H = h.reshape(3, 3)
    
    # 정규화 (H[2, 2] = 1로 만들기)
    if H[2, 2] != 0:
        H = H / H[2, 2]
    
    return H.astype(np.float32)

