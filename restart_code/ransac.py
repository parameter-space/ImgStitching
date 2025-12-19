"""
RANSAC (RANdom SAmple Consensus) 구현
Homography 계산 시 Outlier를 제거하여 안정적인 결과를 얻습니다.
"""

import numpy as np
from typing import Tuple
from homography import compute_homography_dlt, apply_homography


def compute_reprojection_error(points1: np.ndarray, points2: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Reprojection error를 계산합니다.
    
    Args:
        points1: 첫 번째 이미지의 점들 (N, 2) - float32
        points2: 두 번째 이미지의 점들 (N, 2) - float32
        H: Homography 행렬 (3, 3) - float32, points1을 points2로 변환
    
    Returns:
        errors: 각 점의 reprojection error (픽셀 단위) (N,) - float32
    """
    # points1을 H로 변환
    points1_transformed = apply_homography(points1, H)
    
    # 점들 간의 차이 계산
    diff = points1_transformed - points2
    
    # 유클리드 거리 계산
    errors = np.sqrt(np.sum(diff ** 2, axis=1))
    
    return errors.astype(np.float32)


def ransac_homography(points1: np.ndarray, 
                     points2: np.ndarray,
                     max_iterations: int = 2000,
                     threshold: float = 3.0,
                     min_inliers: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    RANSAC을 사용하여 Outlier를 제거하고 안정적인 Homography를 계산합니다.
    
    알고리즘:
    1. 4개 점을 랜덤하게 선택
    2. 그 4개 점으로 Homography 계산
    3. 모든 점에 대해 reprojection error 계산
    4. Error가 threshold보다 작은 점들을 Inlier로 간주
    5. Inlier 개수가 가장 많은 Homography 선택
    6. 최종적으로 모든 Inlier로 Homography 재계산
    
    Args:
        points1: 첫 번째 이미지의 점들 (N, 2) - float32, N >= 4
        points2: 두 번째 이미지의 점들 (N, 2) - float32, N >= 4
        max_iterations: 최대 반복 횟수 (int, 기본값: 2000)
        threshold: Inlier 판단 임계값 (픽셀 단위, float, 기본값: 3.0)
        min_inliers: 최소 Inlier 개수 (int, 기본값: 10)
    
    Returns:
        H: 최적 Homography 행렬 (3, 3) - float32
        inlier_mask: Inlier 마스크 (N,) - bool, True인 인덱스가 Inlier
    """
    N = len(points1)
    
    if N < 4:
        # 최소 4개 점이 필요
        return np.eye(3, dtype=np.float32), np.zeros(N, dtype=bool)
    
    if N == 4:
        # 정확히 4개면 바로 계산
        H = compute_homography_dlt(points1, points2)
        errors = compute_reprojection_error(points1, points2, H)
        inlier_mask = errors < threshold
        return H, inlier_mask
    
    best_H = None
    best_inlier_count = -1
    best_inlier_mask = None
    
    # RANSAC 반복
    for iteration in range(max_iterations):
        # 1. 4개 점을 랜덤하게 선택
        indices = np.random.choice(N, size=4, replace=False)
        sample1 = points1[indices]
        sample2 = points2[indices]
        
        # 2. 4개 점으로 Homography 계산
        try:
            H_candidate = compute_homography_dlt(sample1, sample2)
        except:
            continue
        
        # 3. 모든 점에 대해 reprojection error 계산
        errors = compute_reprojection_error(points1, points2, H_candidate)
        
        # 4. Inlier 개수 계산
        inlier_mask = errors < threshold
        inlier_count = np.sum(inlier_mask)
        
        # 5. 더 좋은 결과면 저장
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_H = H_candidate
            best_inlier_mask = inlier_mask
            
            # 충분한 Inlier를 찾으면 조기 종료
            if inlier_count >= max(min_inliers, int(N * 0.5)):
                break
    
    # 최소 Inlier 개수 미달 시
    if best_inlier_count < min_inliers:
        # Fallback: RANSAC이 실패한 경우
        # 모든 점을 사용하는 것은 outlier가 많을 때 위험하므로,
        # 최선의 RANSAC 결과를 반환 (best_inlier_count가 0인 경우만 Identity)
        if best_inlier_count == 0:
            print(f"  Warning: RANSAC failed completely (0 inliers). Returning identity matrix.")
            return np.eye(3, dtype=np.float32), np.zeros(N, dtype=bool)
        else:
            print(f"  Warning: RANSAC failed to find enough inliers ({best_inlier_count} < {min_inliers}).")
            print(f"    Using best RANSAC result with {best_inlier_count} inliers.")
            # 최선의 RANSAC 결과 반환 (비정상적일 수 있지만 Identity보다는 나음)
            return best_H, best_inlier_mask
    
    # 6. 모든 Inlier로 Homography 재계산 (더 정확한 결과)
    if best_inlier_count > 4:
        points1_inliers = points1[best_inlier_mask]
        points2_inliers = points2[best_inlier_mask]
        
        try:
            H_refined = compute_homography_dlt(points1_inliers, points2_inliers)
            
            # 재계산된 Homography 검증
            errors_refined = compute_reprojection_error(points1, points2, H_refined)
            inlier_mask_refined = errors_refined < threshold
            inlier_count_refined = np.sum(inlier_mask_refined)
            
            # 재계산된 것이 더 좋으면 사용
            if inlier_count_refined >= best_inlier_count:
                return H_refined, inlier_mask_refined
            else:
                return best_H, best_inlier_mask
        except:
            # 재계산 실패 시 원래 결과 반환
            return best_H, best_inlier_mask
    
    return best_H, best_inlier_mask

