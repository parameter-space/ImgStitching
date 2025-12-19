"""
Point Matching 구현
Point_Matching.pdf 강의자료 참고
"""

import numpy as np
from typing import List, Tuple


def compute_descriptors(image: np.ndarray, corners: np.ndarray, patch_size: int = 21) -> np.ndarray:
    """
    코너 점 주변의 patch를 descriptor로 사용합니다.
    
    Args:
        image: 그레이스케일 이미지 (H, W) - float32
        corners: 코너 점들 (N, 2) - 각 행은 [x, y] - float32
        patch_size: patch 크기 (int, 홀수, 기본값: 21)
    
    Returns:
        descriptors: descriptor 배열 (N, patch_size*patch_size) - float32
    """
    H, W = image.shape
    half_size = patch_size // 2
    
    # Padding 추가
    padded_image = np.pad(image, half_size, mode='constant', constant_values=0)
    
    descriptors = []
    
    for corner in corners:
        x, y = int(corner[0]), int(corner[1])
        
        # 경계 확인 (원본 이미지 기준)
        if x < half_size or x >= W - half_size or y < half_size or y >= H - half_size:
            # 경계에 가까운 점은 제외 (또는 zero padding 사용)
            descriptor = np.zeros(patch_size * patch_size, dtype=np.float32)
        else:
            # Patch 추출 (padded_image에서, padding을 고려하여 인덱스 조정)
            # Corner (x, y)를 중심으로 patch_size x patch_size patch 추출
            # 원본 이미지에서: [y-half_size : y+half_size+1, x-half_size : x+half_size+1]
            # padded_image에서: [(y-half_size)+half_size : (y+half_size+1)+half_size, (x-half_size)+half_size : (x+half_size+1)+half_size]
            # = [y : y+patch_size, x : x+patch_size]
            patch = padded_image[y:y+patch_size, x:x+patch_size]
            
            # Flatten
            descriptor = patch.flatten().astype(np.float32)
        
        descriptors.append(descriptor)
    
    return np.array(descriptors, dtype=np.float32)


def compute_ncc(descriptor1: np.ndarray, descriptor2: np.ndarray) -> float:
    """
    Normalized Cross Correlation (NCC)를 계산합니다.
    
    NCC = sum((d1 - mean1) * (d2 - mean2)) / (std1 * std2 * N)
    
    Args:
        descriptor1: 첫 번째 descriptor (D,) - float32
        descriptor2: 두 번째 descriptor (D,) - float32
    
    Returns:
        ncc: NCC 값 (-1 ~ 1) - float
    """
    mean1 = np.mean(descriptor1)
    mean2 = np.mean(descriptor2)
    
    d1_centered = descriptor1 - mean1
    d2_centered = descriptor2 - mean2
    
    std1 = np.std(descriptor1)
    std2 = np.std(descriptor2)
    
    if std1 < 1e-10 or std2 < 1e-10:
        return 0.0
    
    ncc = np.sum(d1_centered * d2_centered) / (std1 * std2 * len(descriptor1))
    
    return float(ncc)


def match_features(descriptors1: np.ndarray, 
                   descriptors2: np.ndarray,
                   threshold: float = 0.7) -> List[Tuple[int, int]]:
    """
    두 이미지의 descriptor를 매칭합니다.
    Ratio test를 사용하여 좋은 매칭만 선택합니다.
    
    Args:
        descriptors1: 첫 번째 이미지의 descriptor 배열 (N1, D) - float32
        descriptors2: 두 번째 이미지의 descriptor 배열 (N2, D) - float32
        threshold: ratio test threshold (float, 기본값: 0.7)
                   ratio = best2_ncc / best_ncc < threshold인 경우만 매칭
    
    Returns:
        matches: 매칭 결과 리스트 [(idx1, idx2), ...] - List[Tuple[int, int]]
    """
    N1 = len(descriptors1)
    N2 = len(descriptors2)
    
    matches = []
    
    for i in range(N1):
        # 각 descriptor1[i]에 대해 descriptors2와의 NCC 계산
        ncc_scores = []
        
        for j in range(N2):
            ncc = compute_ncc(descriptors1[i], descriptors2[j])
            ncc_scores.append(ncc)
        
        ncc_scores = np.array(ncc_scores)
        
        # 가장 큰 NCC와 두 번째로 큰 NCC 찾기
        sorted_indices = np.argsort(ncc_scores)[::-1]  # 내림차순 정렬
        
        if len(sorted_indices) < 2:
            continue
        
        best_idx = sorted_indices[0]
        second_best_idx = sorted_indices[1]
        
        best_ncc = ncc_scores[best_idx]
        second_best_ncc = ncc_scores[second_best_idx]
        
        # Ratio test: best_ncc가 충분히 크고, second_best_ncc와의 비율이 threshold보다 작아야 함
        if best_ncc > 0.5 and second_best_ncc > 0:
            ratio = second_best_ncc / best_ncc
            
            if ratio < threshold:
                matches.append((i, best_idx))
    
    return matches


def get_matched_points(corners1: np.ndarray, 
                       corners2: np.ndarray, 
                       matches: List[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    매칭 결과에서 점 좌표를 추출합니다.
    
    Args:
        corners1: 첫 번째 이미지의 코너 점들 (N1, 2) - float32
        corners2: 두 번째 이미지의 코너 점들 (N2, 2) - float32
        matches: 매칭 결과 [(idx1, idx2), ...]
    
    Returns:
        points1: 첫 번째 이미지의 매칭된 점들 (M, 2) - float32
        points2: 두 번째 이미지의 매칭된 점들 (M, 2) - float32
    """
    points1 = []
    points2 = []
    
    for idx1, idx2 in matches:
        points1.append(corners1[idx1])
        points2.append(corners2[idx2])
    
    if len(points1) == 0:
        return np.array([], dtype=np.float32).reshape(0, 2), np.array([], dtype=np.float32).reshape(0, 2)
    
    return np.array(points1, dtype=np.float32), np.array(points2, dtype=np.float32)

