"""
특징점 검출 및 매칭 모듈
Harris Corner Detection과 Feature Matching을 구현합니다.
"""

import numpy as np
from typing import List, Tuple, Optional

try:
    from numpy.lib.stride_tricks import sliding_window_view
    HAS_SLIDING_WINDOW = True
except ImportError:
    HAS_SLIDING_WINDOW = False


def compute_image_gradients(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    이미지의 x, y 방향 그래디언트를 계산합니다.
    Sobel Filter를 직접 정의하여 사용합니다.
    
    Sobel Filter:
    - x 방향: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    - y 방향: [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    
    Args:
        image: 그레이스케일 이미지 배열 (H, W) - float32
    
    Returns:
        Ix: x 방향 그래디언트 (H, W) - float32
        Iy: y 방향 그래디언트 (H, W) - float32
    """
    # Sobel Filter 정의
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)
    
    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=np.float32)
    
    # 이미지 패딩 (경계 처리)
    padded = np.pad(image, ((1, 1), (1, 1)), mode='reflect')
    
    H, W = image.shape
    
    # 벡터화된 Convolution 수행
    if HAS_SLIDING_WINDOW:
        # NumPy 1.20.0+ sliding_window_view 사용 (더 빠름)
        windows = sliding_window_view(padded, (3, 3))
        Ix = np.sum(windows * sobel_x, axis=(2, 3))
        Iy = np.sum(windows * sobel_y, axis=(2, 3))
    else:
        # Fallback: 기본 구현
        Ix = np.zeros_like(image, dtype=np.float32)
        Iy = np.zeros_like(image, dtype=np.float32)
        
        for i in range(H):
            for j in range(W):
                patch = padded[i:i+3, j:j+3]
                Ix[i, j] = np.sum(patch * sobel_x)
                Iy[i, j] = np.sum(patch * sobel_y)
    
    return Ix, Iy


def compute_structure_tensor(Ix: np.ndarray, Iy: np.ndarray, window_size: int = 3, use_gaussian: bool = True, sigma: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Structure Tensor (M 행렬)의 구성 요소를 계산합니다.
    M = [[Ix^2, Ix*Iy], [Ix*Iy, Iy^2]]
    
    Window function으로 Gaussian 또는 Box filter를 적용합니다.
    
    Args:
        Ix: x 방향 그래디언트 (H, W) - float32
        Iy: y 방향 그래디언트 (H, W) - float32
        window_size: 윈도우 크기 (int, 기본값: 3, 홀수여야 함)
        use_gaussian: True면 Gaussian window, False면 Box window 사용 (bool, 기본값: True)
        sigma: Gaussian window의 표준편차 (float, 기본값: 1.0)
    
    Returns:
        Ixx: Ix^2의 윈도우 합 (H, W) - float32
        Ixy: Ix*Iy의 윈도우 합 (H, W) - float32
        Iyy: Iy^2의 윈도우 합 (H, W) - float32
    """
    # Structure Tensor 구성 요소 계산
    Ixx = Ix * Ix  # (H, W)
    Ixy = Ix * Iy  # (H, W)
    Iyy = Iy * Iy  # (H, W)
    
    # Window function 생성
    if use_gaussian:
        # Gaussian window 생성
        half = window_size // 2
        x = np.arange(-half, half + 1, dtype=np.float32)
        y = np.arange(-half, half + 1, dtype=np.float32)
        X, Y = np.meshgrid(x, y)
        window = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
        window = window / np.sum(window)  # 정규화
    else:
        # Box window (균일 가중치)
        window = np.ones((window_size, window_size), dtype=np.float32)
        window = window / np.sum(window)  # 정규화
    
    # Convolution을 통한 윈도우 합 계산
    # 패딩 적용
    pad = window_size // 2
    Ixx_padded = np.pad(Ixx, ((pad, pad), (pad, pad)), mode='reflect')
    Ixy_padded = np.pad(Ixy, ((pad, pad), (pad, pad)), mode='reflect')
    Iyy_padded = np.pad(Iyy, ((pad, pad), (pad, pad)), mode='reflect')
    
    H, W = Ix.shape
    
    # 벡터화된 Convolution 수행
    if HAS_SLIDING_WINDOW:
        # NumPy 1.20.0+ sliding_window_view 사용 (더 빠름)
        windows_xx = sliding_window_view(Ixx_padded, (window_size, window_size))
        windows_xy = sliding_window_view(Ixy_padded, (window_size, window_size))
        windows_yy = sliding_window_view(Iyy_padded, (window_size, window_size))
        
        Ixx_windowed = np.sum(windows_xx * window, axis=(2, 3))
        Ixy_windowed = np.sum(windows_xy * window, axis=(2, 3))
        Iyy_windowed = np.sum(windows_yy * window, axis=(2, 3))
    else:
        # Fallback: 기본 구현
        Ixx_windowed = np.zeros_like(Ixx, dtype=np.float32)
        Ixy_windowed = np.zeros_like(Ixy, dtype=np.float32)
        Iyy_windowed = np.zeros_like(Iyy, dtype=np.float32)
        
        for i in range(H):
            for j in range(W):
                patch_xx = Ixx_padded[i:i+window_size, j:j+window_size]
                patch_xy = Ixy_padded[i:i+window_size, j:j+window_size]
                patch_yy = Iyy_padded[i:i+window_size, j:j+window_size]
                
                Ixx_windowed[i, j] = np.sum(patch_xx * window)
                Ixy_windowed[i, j] = np.sum(patch_xy * window)
                Iyy_windowed[i, j] = np.sum(patch_yy * window)
    
    return Ixx_windowed, Ixy_windowed, Iyy_windowed


def compute_harris_response(Ixx: np.ndarray, Ixy: np.ndarray, Iyy: np.ndarray, k: float = 0.04) -> np.ndarray:
    """
    Harris Corner Response를 계산합니다.
    
    수식: R = det(M) - k * trace(M)^2
    
    여기서 M = [[Ixx, Ixy], [Ixy, Iyy]]이므로:
    - det(M) = Ixx * Iyy - Ixy^2
    - trace(M) = Ixx + Iyy
    
    Args:
        Ixx: Ix^2의 윈도우 합 (H, W) - float32
        Ixy: Ix*Iy의 윈도우 합 (H, W) - float32
        Iyy: Iy^2의 윈도우 합 (H, W) - float32
        k: Harris 상수 (float, 기본값: 0.04)
    
    Returns:
        response: Harris Response 맵 (H, W) - float32
    """
    # det(M) = Ixx * Iyy - Ixy^2
    det_M = Ixx * Iyy - Ixy * Ixy
    
    # trace(M) = Ixx + Iyy
    trace_M = Ixx + Iyy
    
    # R = det(M) - k * trace(M)^2
    response = det_M - k * (trace_M ** 2)
    
    return response


def detect_corners(response: np.ndarray, threshold: float, nms_window_size: int = 3, max_corners: int = 5000) -> np.ndarray:
    """
    Harris Response에서 코너를 검출하고 Non-Maximum Suppression을 적용합니다.
    
    Args:
        response: Harris Response 맵 (H, W) - float32
        threshold: 코너 검출 임계값 (float)
        nms_window_size: NMS 윈도우 크기 (int, 기본값: 3, 홀수여야 함)
        max_corners: 최대 코너 개수 (int, 기본값: 5000)
    
    Returns:
        corners: 코너 좌표 배열 (N, 2) - 각 행은 [x, y] 좌표 - int32
    """
    # Threshold 적용
    corner_mask = response > threshold
    
    if not np.any(corner_mask):
        return np.empty((0, 2), dtype=np.int32)
    
    # Non-Maximum Suppression (NMS)
    # 각 픽셀의 주변 윈도우에서 최대값인지 확인
    H, W = response.shape
    pad = nms_window_size // 2
    padded_response = np.pad(response, ((pad, pad), (pad, pad)), mode='constant', constant_values=-np.inf)
    
    # 벡터화된 NMS 수행
    if HAS_SLIDING_WINDOW:
        # 모든 윈도우의 최대값 계산
        windows = sliding_window_view(padded_response, (nms_window_size, nms_window_size))
        max_values = np.max(windows, axis=(2, 3))
        
        # 각 픽셀이 자신의 윈도우에서 최대값인지 확인
        nms_mask = (response >= max_values) & corner_mask
    else:
        # Fallback: 기본 구현
        nms_mask = np.zeros_like(corner_mask, dtype=bool)
        
        for i in range(H):
            for j in range(W):
                if corner_mask[i, j]:
                    # 주변 윈도우에서 최대값인지 확인
                    window = padded_response[i:i+nms_window_size, j:j+nms_window_size]
                    if response[i, j] >= np.max(window):
                        nms_mask[i, j] = True
    
    # 코너 좌표 추출 (x, y) 형태로 반환
    y_coords, x_coords = np.where(nms_mask)
    corners = np.column_stack([x_coords, y_coords]).astype(np.int32)
    
    # 코너가 너무 많으면 Response 값이 큰 순서로 제한
    if len(corners) > max_corners:
        corner_responses = response[y_coords, x_coords]
        top_indices = np.argsort(corner_responses)[::-1][:max_corners]
        corners = corners[top_indices]
    
    return corners


def harris_corner_detection(image: np.ndarray, threshold: float = 0.01, k: float = 0.04, window_size: int = 3, nms_window_size: int = 3, use_gaussian: bool = True, sigma: float = 1.0, max_corners: int = 5000) -> np.ndarray:
    """
    전체 Harris Corner Detection 파이프라인을 실행합니다.
    
    알고리즘 단계:
    1. Sobel Filter로 이미지 그래디언트 계산 (Ix, Iy)
    2. Structure Tensor 구성 요소 계산 (Ixx, Ixy, Iyy) - Window function 적용
    3. Harris Response 계산: R = det(M) - k * trace(M)^2
    4. Threshold 및 NMS를 통한 코너 검출
    
    Args:
        image: 그레이스케일 이미지 배열 (H, W) - float32
        threshold: 코너 검출 임계값 (float, 기본값: 0.01)
        k: Harris 상수 (float, 기본값: 0.04)
        window_size: Structure Tensor 윈도우 크기 (int, 기본값: 3, 홀수여야 함)
        nms_window_size: NMS 윈도우 크기 (int, 기본값: 3, 홀수여야 함)
        use_gaussian: True면 Gaussian window, False면 Box window 사용 (bool, 기본값: True)
        sigma: Gaussian window의 표준편차 (float, 기본값: 1.0)
        max_corners: 최대 코너 개수 (int, 기본값: 5000)
    
    Returns:
        corners: 코너 좌표 배열 (N, 2) - 각 행은 [x, y] 좌표 - int32
    """
    # 1. 이미지 그래디언트 계산 (Sobel Filter)
    Ix, Iy = compute_image_gradients(image)
    
    # 2. Structure Tensor 구성 요소 계산 (Window function 적용)
    Ixx, Ixy, Iyy = compute_structure_tensor(Ix, Iy, window_size, use_gaussian, sigma)
    
    # 3. Harris Response 계산
    response = compute_harris_response(Ixx, Ixy, Iyy, k)
    
    # 4. 코너 검출 (Threshold + NMS)
    corners = detect_corners(response, threshold, nms_window_size, max_corners)
    
    return corners


def extract_patch(image: np.ndarray, center: Tuple[int, int], patch_size: int) -> np.ndarray:
    """
    이미지에서 패치를 추출합니다.
    
    Args:
        image: 이미지 배열 (H, W) 또는 (H, W, C) - float32 또는 uint8
        center: 패치 중심 좌표 (x, y) - Tuple[int, int]
        patch_size: 패치 크기 (int, 홀수여야 함)
    
    Returns:
        patch: 추출된 패치 (patch_size, patch_size) 또는 (patch_size, patch_size, C) - 입력과 동일한 dtype
    """
    x, y = center
    half = patch_size // 2
    
    H, W = image.shape[:2]
    
    # 경계 체크 및 패딩
    x_min = max(0, x - half)
    x_max = min(W, x + half + 1)
    y_min = max(0, y - half)
    y_max = min(H, y + half + 1)
    
    # 패치 추출
    if len(image.shape) == 2:
        patch = image[y_min:y_max, x_min:x_max]
    else:
        patch = image[y_min:y_max, x_min:x_max, :]
    
    # 패치가 경계에 걸린 경우 패딩
    if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
        pad_top = half - (y - y_min)
        pad_bottom = half - (y_max - y - 1)
        pad_left = half - (x - x_min)
        pad_right = half - (x_max - x - 1)
        
        if len(image.shape) == 2:
            patch = np.pad(patch, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
        else:
            patch = np.pad(patch, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=0)
    
    return patch


def compute_patch_descriptor(image: np.ndarray, corner: Tuple[int, int], patch_size: int = 7) -> np.ndarray:
    """
    코너 주변의 패치를 디스크립터로 사용합니다.
    패치를 1D 벡터로 평탄화(flatten)하여 반환합니다.
    
    Args:
        image: 그레이스케일 이미지 배열 (H, W) - float32
        corner: 코너 좌표 (x, y) - Tuple[int, int]
        patch_size: 패치 크기 (int, 기본값: 7, 홀수여야 함)
    
    Returns:
        descriptor: 패치 디스크립터 (patch_size * patch_size,) - float32
    """
    patch = extract_patch(image, corner, patch_size)
    descriptor = patch.flatten().astype(np.float32)
    return descriptor


def compute_descriptors(image: np.ndarray, corners: np.ndarray, patch_size: int = 7) -> np.ndarray:
    """
    모든 코너에 대해 디스크립터를 계산합니다.
    
    Args:
        image: 그레이스케일 이미지 배열 (H, W) - float32
        corners: 코너 좌표 배열 (N, 2) - 각 행은 [x, y] - int32
        patch_size: 패치 크기 (int, 기본값: 7, 홀수여야 함)
    
    Returns:
        descriptors: 디스크립터 배열 (N, patch_size * patch_size) - float32
    """
    N = len(corners)
    descriptors = np.zeros((N, patch_size * patch_size), dtype=np.float32)
    
    for i, corner in enumerate(corners):
        x, y = corner[0], corner[1]
        descriptors[i] = compute_patch_descriptor(image, (x, y), patch_size)
    
    return descriptors


def compute_ssd(descriptor1: np.ndarray, descriptor2: np.ndarray) -> float:
    """
    두 디스크립터 간의 Sum of Squared Differences를 계산합니다.
    
    수식: SSD = Σ(descriptor1 - descriptor2)²
    
    Args:
        descriptor1: 첫 번째 디스크립터 (D,) - float32
        descriptor2: 두 번째 디스크립터 (D,) - float32
    
    Returns:
        ssd: SSD 값 (float, 작을수록 유사함)
    """
    diff = descriptor1 - descriptor2
    ssd = np.sum(diff ** 2)
    return float(ssd)


def compute_ncc(descriptor1: np.ndarray, descriptor2: np.ndarray) -> float:
    """
    두 디스크립터 간의 Normalized Cross Correlation을 계산합니다.
    
    수식: NCC = (descriptor1 - μ1) · (descriptor2 - μ2) / (σ1 * σ2)
    
    Args:
        descriptor1: 첫 번째 디스크립터 (D,) - float32
        descriptor2: 두 번째 디스크립터 (D,) - float32
    
    Returns:
        ncc: NCC 값 (float, -1 ~ 1, 클수록 유사함)
    """
    # 평균 제거 (zero-mean)
    d1_centered = descriptor1 - np.mean(descriptor1)
    d2_centered = descriptor2 - np.mean(descriptor2)
    
    # 표준편차 계산
    std1 = np.std(descriptor1)
    std2 = np.std(descriptor2)
    
    # 분모가 0인 경우 처리
    if std1 == 0 or std2 == 0:
        return 0.0
    
    # NCC 계산 (division by zero 방지를 위한 epsilon 추가)
    ncc = np.sum(d1_centered * d2_centered) / (std1 * std2 * len(descriptor1) + 1e-6)
    return float(ncc)


def match_features(descriptors1: np.ndarray, descriptors2: np.ndarray, 
                   method: str = 'ssd', threshold: float = 0.7, 
                   max_distance: Optional[float] = None) -> List[Tuple[int, int]]:
    """
    두 이미지의 디스크립터를 매칭합니다.
    
    SSD 방식: 가장 작은 거리와 두 번째로 작은 거리의 비율이 threshold보다 작으면 매칭
    NCC 방식: 가장 큰 상관계수와 두 번째로 큰 상관계수의 비율이 threshold보다 크면 매칭
    
    Args:
        descriptors1: 첫 번째 이미지의 디스크립터 배열 (N1, D) - float32
        descriptors2: 두 번째 이미지의 디스크립터 배열 (N2, D) - float32
        method: 매칭 방법 ('ssd' 또는 'ncc', 기본값: 'ssd')
        threshold: Ratio Test 임계값 (float, 기본값: 0.7)
                   SSD: ratio = dist1 / dist2 < threshold
                   NCC: ratio = corr2 / corr1 < threshold
        max_distance: 최대 거리 제한 (SSD만 사용, Optional, 기본값: None)
    
    Returns:
        matches: 매칭 결과 리스트 [(idx1, idx2), ...] - List[Tuple[int, int]]
    """
    N1, D = descriptors1.shape
    N2, D2 = descriptors2.shape
    
    if D != D2:
        raise ValueError(f"디스크립터 차원이 일치하지 않습니다: {D} vs {D2}")
    
    matches = []
    
    if method == 'ssd':
        # SSD 방식: 작을수록 유사
        # 메모리 효율적인 방식: 배치 처리 또는 루프 사용
        # 큰 배열을 한 번에 생성하지 않고 배치로 나눠서 처리
        
        # 메모리 사용량이 너무 크면 배치 처리
        max_batch_size = 1000  # 한 번에 처리할 최대 descriptor 개수
        
        if N1 * N2 > 1000000:  # 1M 이상이면 배치 처리
            # 배치 단위로 처리
            for i in range(N1):
                desc1 = descriptors1[i:i+1, :]  # (1, D)
                # 각 descriptor1에 대해 descriptors2와의 거리 계산
                diff = desc1 - descriptors2  # (1, N2, D) -> 브로드캐스팅으로 (N2, D)
                distances = np.sum(diff ** 2, axis=1)  # (N2,)
                
                sorted_indices = np.argsort(distances)
                
                if len(sorted_indices) >= 2:
                    dist1 = distances[sorted_indices[0]]  # 가장 작은 거리
                    dist2 = distances[sorted_indices[1]]   # 두 번째로 작은 거리
                    
                    # Ratio Test: dist1 / dist2 < threshold
                    if dist2 > 0 and dist1 / dist2 < threshold:
                        # 최대 거리 제한 확인
                        if max_distance is None or dist1 < max_distance:
                            matches.append((i, sorted_indices[0]))
                elif len(sorted_indices) == 1:
                    # 매칭 후보가 1개만 있는 경우
                    dist1 = distances[sorted_indices[0]]
                    if max_distance is None or dist1 < max_distance:
                        matches.append((i, sorted_indices[0]))
        else:
            # 메모리가 충분하면 벡터화된 방식 사용
            diff = descriptors1[:, np.newaxis, :] - descriptors2[np.newaxis, :, :]  # (N1, N2, D)
            distances_matrix = np.sum(diff ** 2, axis=2)  # (N1, N2)
            
            for i in range(N1):
                distances = distances_matrix[i]  # (N2,)
                sorted_indices = np.argsort(distances)
                
                if len(sorted_indices) >= 2:
                    dist1 = distances[sorted_indices[0]]  # 가장 작은 거리
                    dist2 = distances[sorted_indices[1]]   # 두 번째로 작은 거리
                    
                    # Ratio Test: dist1 / dist2 < threshold
                    if dist2 > 0 and dist1 / dist2 < threshold:
                        # 최대 거리 제한 확인
                        if max_distance is None or dist1 < max_distance:
                            matches.append((i, sorted_indices[0]))
                elif len(sorted_indices) == 1:
                    # 매칭 후보가 1개만 있는 경우
                    dist1 = distances[sorted_indices[0]]
                    if max_distance is None or dist1 < max_distance:
                        matches.append((i, sorted_indices[0]))
    
    elif method == 'ncc':
        # NCC 방식: 클수록 유사
        # 메모리 효율적인 방식: 각 descriptor1에 대해 개별 계산
        
        # 각 디스크립터의 평균과 표준편차를 미리 계산
        mean1 = np.mean(descriptors1, axis=1, keepdims=True)  # (N1, 1)
        mean2 = np.mean(descriptors2, axis=1, keepdims=True)   # (N2, 1)
        std1 = np.std(descriptors1, axis=1, keepdims=True)    # (N1, 1)
        std2 = np.std(descriptors2, axis=1, keepdims=True)     # (N2, 1)
        
        # Zero-mean
        d1_centered = descriptors1 - mean1  # (N1, D)
        d2_centered = descriptors2 - mean2  # (N2, D)
        
        # 표준편차가 0인 경우 처리 (1로 대체하여 나눗셈 오류 방지)
        std1 = np.where(std1 == 0, 1.0, std1)
        std2 = np.where(std2 == 0, 1.0, std2)
        
        # 메모리 사용량이 너무 크면 배치 처리
        if N1 * N2 > 1000000:  # 1M 이상이면 배치 처리
            for i in range(N1):
                d1_i = d1_centered[i:i+1, :]  # (1, D)
                std1_i = std1[i, 0]
                
                # 상관계수 계산: (1, N2) (division by zero 방지를 위한 epsilon 추가)
                correlations = np.dot(d1_i, d2_centered.T) / (std1_i * std2.T * D + 1e-6)  # (1, N2)
                correlations = correlations.flatten()  # (N2,)
                
                sorted_indices = np.argsort(correlations)[::-1]  # 내림차순
                
                if len(sorted_indices) >= 2:
                    corr1 = correlations[sorted_indices[0]]  # 가장 큰 상관계수
                    corr2 = correlations[sorted_indices[1]]   # 두 번째로 큰 상관계수
                    
                    # Ratio Test: corr2 / corr1 < threshold (NCC는 클수록 좋으므로 역순)
                    if corr1 > 0 and corr2 / corr1 < threshold:
                        matches.append((i, sorted_indices[0]))
                elif len(sorted_indices) == 1:
                    # 매칭 후보가 1개만 있는 경우
                    if correlations[sorted_indices[0]] > 0:  # 양수 상관계수만 허용
                        matches.append((i, sorted_indices[0]))
        else:
            # 메모리가 충분하면 벡터화된 방식 사용
            # 상관계수 행렬 계산: (N1, N2)
            # NCC = sum((d1 - μ1) * (d2 - μ2)) / (σ1 * σ2 * D) (division by zero 방지를 위한 epsilon 추가)
            correlations_matrix = np.dot(d1_centered, d2_centered.T) / (std1 * std2.T * D + 1e-6)
            
            for i in range(N1):
                correlations = correlations_matrix[i]  # (N2,)
                sorted_indices = np.argsort(correlations)[::-1]  # 내림차순
                
                if len(sorted_indices) >= 2:
                    corr1 = correlations[sorted_indices[0]]  # 가장 큰 상관계수
                    corr2 = correlations[sorted_indices[1]]   # 두 번째로 큰 상관계수
                    
                    # Ratio Test: corr2 / corr1 < threshold (NCC는 클수록 좋으므로 역순)
                    if corr1 > 0 and corr2 / corr1 < threshold:
                        matches.append((i, sorted_indices[0]))
                elif len(sorted_indices) == 1:
                    # 매칭 후보가 1개만 있는 경우
                    if correlations[sorted_indices[0]] > 0:  # 양수 상관계수만 허용
                        matches.append((i, sorted_indices[0]))
    
    else:
        raise ValueError(f"지원하지 않는 매칭 방법: {method}. 'ssd' 또는 'ncc'를 사용하세요.")
    
    return matches


def get_matched_points(corners1: np.ndarray, corners2: np.ndarray, matches: List[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    매칭 결과로부터 대응점 쌍을 추출합니다.
    
    Args:
        corners1: 첫 번째 이미지의 코너 좌표 (N1, 2) - 각 행은 [x, y] - int32
        corners2: 두 번째 이미지의 코너 좌표 (N2, 2) - 각 행은 [x, y] - int32
        matches: 매칭 결과 리스트 [(idx1, idx2), ...] - List[Tuple[int, int]]
    
    Returns:
        points1: 첫 번째 이미지의 대응점 (M, 2) - 각 행은 [x, y] - int32
        points2: 두 번째 이미지의 대응점 (M, 2) - 각 행은 [x, y] - int32
    """
    if len(matches) == 0:
        return np.empty((0, 2), dtype=np.int32), np.empty((0, 2), dtype=np.int32)
    
    points1 = np.array([corners1[idx1] for idx1, _ in matches], dtype=np.int32)
    points2 = np.array([corners2[idx2] for _, idx2 in matches], dtype=np.int32)
    
    return points1, points2

