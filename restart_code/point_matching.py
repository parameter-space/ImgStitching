"""
Point Matching (Correspondence) 모듈
특징점 매칭 구현 (필수 구현)
"""

import numpy as np
from typing import List, Tuple, Optional


def extract_patch(image: np.ndarray, center: Tuple[int, int], patch_size: int) -> np.ndarray:
    """
    이미지에서 패치를 추출합니다.
    
    Args:
        image: 이미지 배열 (H, W) - float32
        center: 패치 중심 좌표 (x, y)
        patch_size: 패치 크기 (int, 홀수여야 함)
    
    Returns:
        patch: 추출된 패치 (patch_size, patch_size) - float32
    """
    x, y = center
    half = patch_size // 2
    
    H, W = image.shape
    x_min = max(0, x - half)
    x_max = min(W, x + half + 1)
    y_min = max(0, y - half)
    y_max = min(H, y + half + 1)
    
    # 이미지에서 패치 추출 (좌표 범위 체크 후)
    patch = image[y_min:y_max, x_min:x_max]
    
    # 패딩: 경계 근처의 코너는 패치가 작아지므로 0으로 패딩하여 크기 맞춤
    if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
        # 각 방향으로 패딩 크기 계산
        pad_top = half - (y - y_min)
        pad_bottom = half - (y_max - y - 1)
        pad_left = half - (x - x_min)
        pad_right = half - (x_max - x - 1)
        # mode='constant', constant_values=0: 0으로 패딩
        patch = np.pad(patch, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
    
    return patch


def compute_patch_descriptor(image: np.ndarray, corner: Tuple[int, int], patch_size: int = 21) -> np.ndarray:
    """
    코너 주변의 패치를 디스크립터로 사용합니다.
    조명 변화에 강건하도록 정규화합니다.
    
    Args:
        image: 그레이스케일 이미지 배열 (H, W) - float32
        corner: 코너 좌표 (x, y)
        patch_size: 패치 크기 (int, 기본값: 21)
    
    Returns:
        descriptor: 패치 디스크립터 (patch_size * patch_size,) - float32
    """
    # 패치 추출 및 1D 배열로 변환
    patch = extract_patch(image, corner, patch_size)
    # flatten(): 2D 배열을 1D로 변환 (예: (21, 21) -> (441,))
    patch_flat = patch.flatten().astype(np.float32)
    
    # Normalize: 평균 제거 및 표준편차로 나누기
    # 목적: 조명 변화에 강건하게 만들기 (밝기와 대비 변화에 무관한 패턴만 비교)
    mean = np.mean(patch_flat)  # 평균 계산
    std = np.std(patch_flat)    # 표준편차 계산
    if std > 1e-6:
        # Z-score 정규화: (x - mean) / std
        descriptor = (patch_flat - mean) / std
    else:
        # 표준편차가 너무 작으면 (거의 균일한 패치) 0 벡터 반환
        descriptor = np.zeros_like(patch_flat)
    
    return descriptor


def compute_descriptors(image: np.ndarray, corners: np.ndarray, patch_size: int = 21) -> np.ndarray:
    """
    모든 코너에 대해 디스크립터를 계산합니다.
    
    Args:
        image: 그레이스케일 이미지 배열 (H, W) - float32
        corners: 코너 좌표 배열 (N, 2) - 각 행은 [x, y] - int32
        patch_size: 패치 크기 (int, 기본값: 21)
    
    Returns:
        descriptors: 디스크립터 배열 (N, patch_size * patch_size) - float32
    """
    N = len(corners)
    descriptors = np.zeros((N, patch_size * patch_size), dtype=np.float32)
    
    for i, corner in enumerate(corners):
        descriptors[i] = compute_patch_descriptor(image, (corner[0], corner[1]), patch_size)
    
    return descriptors


def compute_ncc(descriptor1: np.ndarray, descriptor2: np.ndarray) -> float:
    """
    두 디스크립터 간의 Normalized Cross Correlation을 계산합니다.
    
    Args:
        descriptor1: 첫 번째 디스크립터 (D,) - float32
        descriptor2: 두 번째 디스크립터 (D,) - float32
    
    Returns:
        ncc: NCC 값 (float, -1 ~ 1, 클수록 유사함)
    """
    d1_centered = descriptor1 - np.mean(descriptor1)
    d2_centered = descriptor2 - np.mean(descriptor2)
    
    std1 = np.std(descriptor1)
    std2 = np.std(descriptor2)
    
    if std1 == 0 or std2 == 0:
        return 0.0
    
    ncc = np.sum(d1_centered * d2_centered) / (std1 * std2 * len(descriptor1) + 1e-6)
    return float(ncc)


def match_features(descriptors1: np.ndarray, descriptors2: np.ndarray, 
                   method: str = 'ncc', threshold: float = 0.95) -> List[Tuple[int, int]]:
    """
    두 이미지의 디스크립터를 매칭합니다.
    Ratio Test를 사용합니다.
    
    Args:
        descriptors1: 첫 번째 이미지의 디스크립터 배열 (N1, D) - float32
        descriptors2: 두 번째 이미지의 디스크립터 배열 (N2, D) - float32
        method: 매칭 방법 ('ncc', 기본값: 'ncc')
        threshold: Ratio Test 임계값 (float, 기본값: 0.95)
                   NCC: ratio = corr2 / corr1 < threshold
    
    Returns:
        matches: 매칭 결과 리스트 [(idx1, idx2), ...]
    """
    N1, D = descriptors1.shape
    N2, D2 = descriptors2.shape
    
    if D != D2:
        raise ValueError(f"디스크립터 차원이 일치하지 않습니다: {D} vs {D2}")
    
    matches = []
    
    if method == 'ncc':
        # NCC 방식: 벡터화된 계산을 사용하여 효율성 향상
        # 진행 상황 출력
        print_interval = max(1, N1 // 10)  # 10번 정도만 출력
        
        # descriptors2의 통계량을 미리 계산 (중복 계산 방지)
        # 기존 방식: 루프 안에서 매번 계산 (5000번 × 5000번 = 25,000,000번 계산)
        # 개선 방식: 루프 밖에서 한 번만 계산 (5000번 계산) 후 재사용
        # axis=1: 각 descriptor(행)에 대해 계산, keepdims=True: 차원 유지 (N2, 1)
        desc2_means = np.mean(descriptors2, axis=1, keepdims=True)  # (N2, 1) - 모든 descriptor2의 평균을 한 번에 계산
        # 브로드캐스팅: (N2, D) - (N2, 1) = 각 행에서 평균을 빼기
        desc2_centered = descriptors2 - desc2_means  # (N2, D) - 평균을 뺀 값도 미리 계산
        desc2_stds = np.std(descriptors2, axis=1)  # (N2,) - 모든 descriptor2의 표준편차를 한 번에 계산
        valid_mask = desc2_stds > 1e-6  # 표준편차가 너무 작은 descriptor는 제외 (균일한 패치)
        
        # 벡터화된 NCC 계산
        for i in range(N1):
            if i % print_interval == 0:
                print(f"    Matching progress: {i}/{N1} ({100*i/N1:.1f}%)")
            
            desc1 = descriptors1[i]  # (D,)
            desc1_mean = np.mean(desc1)
            desc1_centered = desc1 - desc1_mean
            std1 = np.std(desc1)
            
            if std1 < 1e-6:
                continue  # Skip uniform descriptors
            
            # NCC 계산: 벡터화된 방식
            # NCC 공식: sum((desc1_centered * desc2_centered)) / (std1 * std2 * D)
            correlations = np.zeros(N2, dtype=np.float32)
            
            # 유효한 descriptor2만 계산 (표준편차가 충분한 것만)
            if np.any(valid_mask):
                # [None, :]: (1, D)로 확장하여 브로드캐스팅
                # valid_mask로 필터링된 descriptor2와 내적 계산
                dot_products = np.sum(desc1_centered[None, :] * desc2_centered[valid_mask], axis=1)  # (N_valid,)
                # NCC 계산: 내적 / (표준편차 * 차원), +1e-6은 division by zero 방지
                correlations[valid_mask] = dot_products / (std1 * desc2_stds[valid_mask] * D + 1e-6)
            
            # Sort and apply ratio test
            # np.argsort: 정렬된 인덱스 반환, [::-1]: 내림차순 (큰 값부터)
            sorted_indices = np.argsort(correlations)[::-1]  # 내림차순
            
            if len(sorted_indices) >= 2:
                corr1 = correlations[sorted_indices[0]]  # 가장 큰 상관계수 (1순위 매치)
                corr2 = correlations[sorted_indices[1]]   # 두 번째로 큰 상관계수 (2순위 매치)
                
                # Ratio Test: 두 번째 매치가 첫 번째 매치와 비슷하면 모호하므로 제외
                # corr2 / corr1 < threshold: 두 번째가 첫 번째보다 충분히 작으면 유일한 매치로 인정
                if corr1 > 0 and corr2 / corr1 < threshold:
                    matches.append((i, sorted_indices[0]))
    
    return matches


def get_matched_points(corners1: np.ndarray, corners2: np.ndarray, 
                       matches: List[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    매칭 결과를 좌표 배열로 변환합니다.
    
    Args:
        corners1: 첫 번째 이미지의 코너 좌표 (N1, 2) - int32
        corners2: 두 번째 이미지의 코너 좌표 (N2, 2) - int32
        matches: 매칭 결과 리스트 [(idx1, idx2), ...]
    
    Returns:
        points1: 첫 번째 이미지의 매칭된 점들 (M, 2) - float32
        points2: 두 번째 이미지의 매칭된 점들 (M, 2) - float32
    """
    if len(matches) == 0:
        return np.array([], dtype=np.float32).reshape(0, 2), np.array([], dtype=np.float32).reshape(0, 2)
    
    points1 = np.array([corners1[idx1] for idx1, _ in matches], dtype=np.float32)
    points2 = np.array([corners2[idx2] for _, idx2 in matches], dtype=np.float32)
    
    return points1, points2

