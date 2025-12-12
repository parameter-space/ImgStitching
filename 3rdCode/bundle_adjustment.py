"""
Bundle Adjustment (Group Adjustment) 모듈
전역 Homography 최적화를 구현합니다.
"""

import numpy as np
from typing import List, Tuple
from geometry import apply_homography, compute_reprojection_error, compute_homography_dlt


def compute_global_homographies(pairwise_homographies: List[np.ndarray]) -> List[np.ndarray]:
    """
    누적 Homography를 사용하여 전역 Homography를 계산합니다.
    
    첫 번째 이미지를 기준 좌표계로 설정합니다.
    H_global[i]는 images[i]를 images[0] 좌표계로 변환합니다.
    
    Args:
        pairwise_homographies: 인접 이미지 쌍의 Homography 리스트 [H1, H2, ...]
                              H[i]는 images[i+1]을 images[i] 좌표계로 변환
    
    Returns:
        global_homographies: 전역 Homography 리스트 [H0, H1, H2, ...]
                           H0 = I (첫 번째 이미지는 기준)
                           H[i]는 images[i]를 images[0] 좌표계로 변환
    """
    N = len(pairwise_homographies) + 1
    global_homographies = [np.eye(3, dtype=np.float32)]  # H0 = I
    
    cumulative_H = np.eye(3, dtype=np.float32)
    
    for i, H_pairwise in enumerate(pairwise_homographies):
        # 누적 변환: H_global[i+1] = H_pairwise[i] * H_global[i]
        cumulative_H = H_pairwise @ cumulative_H
        global_homographies.append(cumulative_H.copy())
    
    return global_homographies


def optimize_homographies_simple(global_homographies: List[np.ndarray],
                                 images: List[np.ndarray],
                                 compute_pairwise_homography_func,
                                 max_iterations: int = 3) -> List[np.ndarray]:
    """
    간단한 Bundle Adjustment: 각 인접 쌍의 Homography를 재계산하여 개선합니다.
    
    Args:
        global_homographies: 초기 전역 Homography 리스트 [H0, H1, ...]
        images: 이미지 리스트
        compute_pairwise_homography_func: Homography 계산 함수
        max_iterations: 최대 반복 횟수 (int, 기본값: 3)
    
    Returns:
        optimized_homographies: 최적화된 전역 Homography 리스트
    """
    N = len(images)
    optimized = global_homographies.copy()
    
    # 반복적으로 개선
    for iteration in range(max_iterations):
        print(f"  Bundle Adjustment 반복 {iteration + 1}/{max_iterations}...")
        
        # 각 인접 쌍에 대해 Homography 재계산
        new_pairwise = []
        
        for i in range(N - 1):
            j = i + 1
            
            try:
                # 실제 매칭을 다시 계산하여 개선
                H_pairwise_new, _ = compute_pairwise_homography_func(images[i], images[j])
                
                # 정규화
                if H_pairwise_new[2, 2] != 0:
                    H_pairwise_new = H_pairwise_new / H_pairwise_new[2, 2]
                
                new_pairwise.append(H_pairwise_new)
                
            except Exception as e:
                # 실패 시 현재 전역 Homography에서 역변환
                H_i = optimized[i]
                H_j = optimized[j]
                H_i_inv = np.linalg.inv(H_i)
                H_pairwise_current = H_j @ H_i_inv
                if H_pairwise_current[2, 2] != 0:
                    H_pairwise_current = H_pairwise_current / H_pairwise_current[2, 2]
                new_pairwise.append(H_pairwise_current)
        
        # 새로운 전역 Homography 계산
        optimized = compute_global_homographies(new_pairwise)
    
    return optimized


def bundle_adjustment(images: List[np.ndarray],
                     pairwise_homographies: List[np.ndarray],
                     compute_pairwise_homography_func,
                     max_iterations: int = 3) -> List[np.ndarray]:
    """
    Bundle Adjustment를 수행하여 전역 Homography를 최적화합니다.
    
    Args:
        images: 이미지 리스트 [img1, img2, ...]
        pairwise_homographies: 초기 인접 쌍 Homography 리스트 [H1, H2, ...]
        compute_pairwise_homography_func: Homography 계산 함수
        max_iterations: 최대 반복 횟수 (int, 기본값: 3)
    
    Returns:
        optimized_global_homographies: 최적화된 전역 Homography 리스트 [H0, H1, ...]
                                      H0 = I, H[i]는 images[i]를 images[0] 좌표계로 변환
    """
    print("Bundle Adjustment 시작...")
    
    # 1. 초기 전역 Homography 계산
    global_homographies = compute_global_homographies(pairwise_homographies)
    
    # 2. 최적화
    optimized = optimize_homographies_simple(
        global_homographies,
        images,
        compute_pairwise_homography_func,
        max_iterations=max_iterations
    )
    
    print("Bundle Adjustment 완료")
    
    return optimized

