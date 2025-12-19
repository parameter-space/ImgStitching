"""
Group Adjustment (Bundle Adjustment) 모듈
전역 Homography 최적화 (선택적 구현, 추가 점수)
"""

import numpy as np
from typing import List
from homography import apply_homography


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
    # 전역 Homography 계산: 첫 번째 이미지를 기준(Identity)으로 설정
    N = len(pairwise_homographies) + 1  # 이미지 개수
    global_homographies = [np.eye(3, dtype=np.float32)]  # H0 = I (첫 번째 이미지는 기준)
    
    # 누적 변환 행렬: 첫 번째 이미지부터 순차적으로 누적
    cumulative_H = np.eye(3, dtype=np.float32)
    
    for i, H_pairwise in enumerate(pairwise_homographies):
        # 누적 변환: cumulative_H = cumulative_H @ H_pairwise
        # H_pairwise는 images[i+1]을 images[i]로 변환
        # 누적하면 images[i+1]을 images[0]로 변환하는 행렬이 됨
        cumulative_H = cumulative_H @ H_pairwise
        # .copy(): 참조가 아닌 복사본 저장 (누적 행렬 변경 방지)
        global_homographies.append(cumulative_H.copy())
    
    return global_homographies

