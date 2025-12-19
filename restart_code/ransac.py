"""
RANSAC 알고리즘 모듈
선택적 구현 (추가 점수)
"""

import numpy as np
from typing import Tuple

from homography import compute_homography_dlt, compute_reprojection_error


def ransac_homography(p1: np.ndarray, p2: np.ndarray, 
                     max_iterations: int = 2000, 
                     threshold: float = 5.0,
                     min_inliers: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    RANSAC 알고리즘을 사용하여 최적의 8-DoF Homography 행렬을 찾습니다.
    
    단계:
    1. 4개의 무작위 점을 선택하여 compute_homography_dlt로 Homography 계산
    2. 모든 점에 대해 재투영 오차를 계산하여 inlier 결정
    3. 가장 많은 inlier를 가진 Homography 선택
    4. 모든 inlier에 대해 Refinement 수행 (compute_homography_dlt 재실행)
    
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
    best_inliers_count = 0
    best_H = np.eye(3, dtype=np.float32)
    best_inliers_mask = np.zeros(len(p1), dtype=bool)
    
    N = len(p1)
    if N < 4:
        return best_H, best_inliers_mask
    
    # RANSAC 메인 루프: 최대 max_iterations 번 반복
    for _ in range(max_iterations):
        # Randomly select 4 points for Homography calculation
        # Homography는 8-DoF이므로 최소 4개의 correspondence 필요 (각 점은 2개의 제약)
        # np.random.choice(N, 4, replace=False): N개 중 4개를 중복 없이 랜덤 선택
        idx = np.random.choice(N, 4, replace=False)
        
        try:
            # Compute Homography using Normalized DLT
            # 선택된 4개 점으로 Homography 계산
            H = compute_homography_dlt(p1[idx], p2[idx])
            
            # Compute reprojection error for all points
            # 모든 점에 대해 계산된 H로 재투영 오차 계산
            errors = compute_reprojection_error(p1, p2, H)
            
            # Determine inliers: 오차가 threshold보다 작은 점들
            inliers_mask = errors < threshold  # Boolean 배열
            inliers_count = np.sum(inliers_mask)  # True의 개수 = inlier 개수
            
            # Update best model if this has more inliers
            # 더 많은 inlier를 가진 모델로 업데이트
            if inliers_count > best_inliers_count:
                best_inliers_count = inliers_count
                best_inliers_mask = inliers_mask
                best_H = H
                
        except Exception:
            # Skip if computation fails (예: 4개 점이 collinear인 경우)
            continue
    
    # Refinement: Re-run compute_homography_dlt on ALL inliers
    # RANSAC으로 찾은 inlier만 사용하여 더 정확한 Homography 계산
    if best_inliers_count >= 4:
        try:
            # Boolean indexing: best_inliers_mask가 True인 점들만 선택
            refined_H = compute_homography_dlt(p1[best_inliers_mask], p2[best_inliers_mask])
            
            # Validate refined homography: 기하학적 제약 확인
            det = np.linalg.det(refined_H)  # 행렬식 계산
            if det > 0.1:  # No reflection (반사가 없어야 함, det > 0)
                # Check scale: 이미지 크기 변화가 너무 크지 않아야 함
                # Homography의 상단 2x2 행렬의 스케일 계산 (회전 행렬의 norm)
                scale_x = np.sqrt(refined_H[0, 0]**2 + refined_H[0, 1]**2)
                scale_y = np.sqrt(refined_H[1, 0]**2 + refined_H[1, 1]**2)
                
                if 0.5 <= scale_x <= 2.0 and 0.5 <= scale_y <= 2.0:
                    # Check reprojection error: 평균 재투영 오차 확인
                    errors_refined = compute_reprojection_error(p1[best_inliers_mask], p2[best_inliers_mask], refined_H)
                    mean_error = np.mean(errors_refined)
                    
                    if mean_error < 10.0:  # 평균 오차가 10픽셀 미만이면 사용
                        best_H = refined_H
        except Exception:
            # Keep original best_H if refinement fails
            # Refinement 실패 시 RANSAC으로 찾은 원본 H 사용
            pass
    
    return best_H.astype(np.float32), best_inliers_mask

