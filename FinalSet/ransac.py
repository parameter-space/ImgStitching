"""
RANSAC (RANdom SAmple Consensus) 구현
Homography 계산 시 Outlier를 제거하여 안정적인 결과를 얻습니다.
"""

import numpy as np
from typing import Tuple
from homography import compute_homography_dlt, apply_homography


def compute_reprojection_error(points1: np.ndarray, points2: np.ndarray, H: np.ndarray) -> np.ndarray:
    points1_transformed = apply_homography(points1, H)
    
    diff = points1_transformed - points2
    
    errors = np.sqrt(np.sum(diff ** 2, axis=1))
    
    return errors.astype(np.float32)


def ransac_homography(points1: np.ndarray, 
                     points2: np.ndarray,
                     max_iterations: int = 2000,
                     threshold: float = 3.0,
                     min_inliers: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    RANSAC 알고리즘:
    1. 4개 점을 랜덤하게 선택
    2. 그 4개 점으로 Homography 계산
    3. 모든 점에 대해 reprojection error 계산
    4. Error가 threshold보다 작은 점들을 Inlier로 간주
    5. Inlier 개수가 가장 많은 Homography 선택
    6. 최종적으로 모든 Inlier로 Homography 재계산
    """
    N = len(points1)
    
    if N < 4:
        return np.eye(3, dtype=np.float32), np.zeros(N, dtype=bool)
    
    if N == 4:
        H = compute_homography_dlt(points1, points2)
        errors = compute_reprojection_error(points1, points2, H)
        inlier_mask = errors < threshold
        return H, inlier_mask
    
    best_H = None
    best_inlier_count = -1
    best_inlier_mask = None
    
    for iteration in range(max_iterations):
        indices = np.random.choice(N, size=4, replace=False)
        sample1 = points1[indices]
        sample2 = points2[indices]
        
        try:
            H_candidate = compute_homography_dlt(sample1, sample2)
        except:
            continue
        
        errors = compute_reprojection_error(points1, points2, H_candidate)
        
        inlier_mask = errors < threshold
        inlier_count = np.sum(inlier_mask)
        
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_H = H_candidate
            best_inlier_mask = inlier_mask
            
            if inlier_count >= max(min_inliers, int(N * 0.5)):
                break
    
    if best_inlier_count < min_inliers:
        if best_inlier_count == 0 or best_H is None:
            print(f"  Warning: RANSAC failed completely (0 inliers or no valid model). Returning identity matrix.")
            return np.eye(3, dtype=np.float32), np.zeros(N, dtype=bool)
        else:
            print(f"  Warning: RANSAC failed to find enough inliers ({best_inlier_count} < {min_inliers}).")
            print(f"    Using best RANSAC result with {best_inlier_count} inliers.")
            return best_H, best_inlier_mask
    
    if best_inlier_count > 4:
        points1_inliers = points1[best_inlier_mask]
        points2_inliers = points2[best_inlier_mask]
        
        try:
            H_refined = compute_homography_dlt(points1_inliers, points2_inliers)
            
            errors_refined = compute_reprojection_error(points1, points2, H_refined)
            inlier_mask_refined = errors_refined < threshold
            inlier_count_refined = np.sum(inlier_mask_refined)
            
            if inlier_count_refined >= best_inlier_count:
                return H_refined, inlier_mask_refined
            else:
                return best_H, best_inlier_mask
        except:
            return best_H, best_inlier_mask
    
    if best_H is None:
        print(f"  Warning: RANSAC internal error (best_H is None despite {best_inlier_count} inliers). Returning identity.")
        return np.eye(3, dtype=np.float32), np.zeros(N, dtype=bool)
    return best_H, best_inlier_mask

