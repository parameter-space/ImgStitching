"""
코너 포인트 찾기 모듈
Harris Corner Detection 구현 (필수 구현)
"""

import numpy as np
from typing import Tuple

try:
    from numpy.lib.stride_tricks import sliding_window_view
    HAS_SLIDING_WINDOW = True
except ImportError:
    HAS_SLIDING_WINDOW = False


def compute_image_gradients(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    이미지의 x, y 방향 그래디언트를 계산합니다.
    Sobel Filter를 직접 정의하여 사용합니다.
    
    Args:
        image: 그레이스케일 이미지 배열 (H, W) - float32
    
    Returns:
        Ix: x 방향 그래디언트 (H, W) - float32
        Iy: y 방향 그래디언트 (H, W) - float32
    """
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)
    
    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=np.float32)
    
    # 패딩
    padded = np.pad(image, ((1, 1), (1, 1)), mode='reflect')
    H, W = image.shape
    
    if HAS_SLIDING_WINDOW:
        windows = sliding_window_view(padded, (3, 3))
        Ix = np.sum(windows * sobel_x, axis=(2, 3))
        Iy = np.sum(windows * sobel_y, axis=(2, 3))
    else:
        Ix = np.zeros_like(image, dtype=np.float32)
        Iy = np.zeros_like(image, dtype=np.float32)
        
        for i in range(H):
            for j in range(W):
                patch = padded[i:i+3, j:j+3]
                Ix[i, j] = np.sum(patch * sobel_x)
                Iy[i, j] = np.sum(patch * sobel_y)
    
    return Ix, Iy


def compute_structure_tensor(Ix: np.ndarray, Iy: np.ndarray, window_size: int = 3, 
                             use_gaussian: bool = True, sigma: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Structure Tensor (M 행렬)의 구성 요소를 계산합니다.
    
    Args:
        Ix: x 방향 그래디언트 (H, W) - float32
        Iy: y 방향 그래디언트 (H, W) - float32
        window_size: 윈도우 크기 (int, 기본값: 3, 홀수여야 함)
        use_gaussian: True면 Gaussian window, False면 Box window 사용
        sigma: Gaussian window의 표준편차 (float, 기본값: 1.0)
    
    Returns:
        Ixx: Ix^2의 윈도우 합 (H, W) - float32
        Ixy: Ix*Iy의 윈도우 합 (H, W) - float32
        Iyy: Iy^2의 윈도우 합 (H, W) - float32
    """
    Ixx = Ix * Ix
    Ixy = Ix * Iy
    Iyy = Iy * Iy
    
    # Window function 생성
    if use_gaussian:
        half = window_size // 2
        x = np.arange(-half, half + 1, dtype=np.float32)
        y = np.arange(-half, half + 1, dtype=np.float32)
        X, Y = np.meshgrid(x, y)
        window = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
        window = window / np.sum(window)
    else:
        window = np.ones((window_size, window_size), dtype=np.float32)
        window = window / np.sum(window)
    
    # Convolution
    H, W = Ixx.shape
    half = window_size // 2
    padded_xx = np.pad(Ixx, ((half, half), (half, half)), mode='reflect')
    padded_xy = np.pad(Ixy, ((half, half), (half, half)), mode='reflect')
    padded_yy = np.pad(Iyy, ((half, half), (half, half)), mode='reflect')
    
    Ixx_smooth = np.zeros_like(Ixx, dtype=np.float32)
    Ixy_smooth = np.zeros_like(Ixy, dtype=np.float32)
    Iyy_smooth = np.zeros_like(Iyy, dtype=np.float32)
    
    for i in range(H):
        for j in range(W):
            patch_xx = padded_xx[i:i+window_size, j:j+window_size]
            patch_xy = padded_xy[i:i+window_size, j:j+window_size]
            patch_yy = padded_yy[i:i+window_size, j:j+window_size]
            
            Ixx_smooth[i, j] = np.sum(patch_xx * window)
            Ixy_smooth[i, j] = np.sum(patch_xy * window)
            Iyy_smooth[i, j] = np.sum(patch_yy * window)
    
    return Ixx_smooth, Ixy_smooth, Iyy_smooth


def compute_harris_response(Ixx: np.ndarray, Ixy: np.ndarray, Iyy: np.ndarray, k: float = 0.04) -> np.ndarray:
    """
    Harris Response를 계산합니다.
    R = det(M) - k * trace(M)^2
    
    Args:
        Ixx: Ix^2의 윈도우 합 (H, W) - float32
        Ixy: Ix*Iy의 윈도우 합 (H, W) - float32
        Iyy: Iy^2의 윈도우 합 (H, W) - float32
        k: Harris 상수 (float, 기본값: 0.04)
    
    Returns:
        response: Harris Response (H, W) - float32
    """
    det_M = Ixx * Iyy - Ixy * Ixy
    trace_M = Ixx + Iyy
    response = det_M - k * (trace_M ** 2)
    
    return response


def detect_corners(response: np.ndarray, threshold: float, nms_window_size: int, max_corners: int) -> np.ndarray:
    """
    Harris Response에서 코너를 검출합니다.
    Threshold와 NMS (Non-Maximum Suppression)를 적용합니다.
    
    Args:
        response: Harris Response (H, W) - float32
        threshold: 코너 검출 임계값 (float)
        nms_window_size: NMS 윈도우 크기 (int, 홀수여야 함)
        max_corners: 최대 코너 개수 (int)
    
    Returns:
        corners: 코너 좌표 배열 (N, 2) - 각 행은 [x, y] - int32
    """
    # Threshold
    corner_mask = response > threshold
    
    if np.sum(corner_mask) == 0:
        return np.array([], dtype=np.int32).reshape(0, 2)
    
    # NMS
    H, W = response.shape
    half = nms_window_size // 2
    padded_response = np.pad(response, ((half, half), (half, half)), mode='constant', constant_values=-np.inf)
    
    if HAS_SLIDING_WINDOW:
        windows = sliding_window_view(padded_response, (nms_window_size, nms_window_size))
        max_values = np.max(windows, axis=(2, 3))
        nms_mask = (response >= max_values) & corner_mask
    else:
        nms_mask = np.zeros_like(corner_mask, dtype=bool)
        for i in range(H):
            for j in range(W):
                if corner_mask[i, j]:
                    window = padded_response[i:i+nms_window_size, j:j+nms_window_size]
                    if response[i, j] >= np.max(window):
                        nms_mask[i, j] = True
    
    # 코너 좌표 추출
    y_coords, x_coords = np.where(nms_mask)
    corners = np.column_stack([x_coords, y_coords]).astype(np.int32)
    
    # 최대 개수 제한
    if len(corners) > max_corners:
        corner_responses = response[y_coords, x_coords]
        top_indices = np.argsort(corner_responses)[::-1][:max_corners]
        corners = corners[top_indices]
    
    return corners


def harris_corner_detection(image: np.ndarray, threshold: float = 0.01, k: float = 0.04, 
                           window_size: int = 3, nms_window_size: int = 3, 
                           use_gaussian: bool = True, sigma: float = 1.0, 
                           max_corners: int = 5000) -> np.ndarray:
    """
    전체 Harris Corner Detection 파이프라인을 실행합니다.
    
    Args:
        image: 그레이스케일 이미지 배열 (H, W) - float32
        threshold: 코너 검출 임계값 (float, 기본값: 0.01)
        k: Harris 상수 (float, 기본값: 0.04)
        window_size: Structure Tensor 윈도우 크기 (int, 기본값: 3)
        nms_window_size: NMS 윈도우 크기 (int, 기본값: 3)
        use_gaussian: True면 Gaussian window 사용 (bool, 기본값: True)
        sigma: Gaussian 표준편차 (float, 기본값: 1.0)
        max_corners: 최대 코너 개수 (int, 기본값: 5000)
    
    Returns:
        corners: 코너 좌표 배열 (N, 2) - 각 행은 [x, y] - int32
    """
    # 1. 그래디언트 계산
    Ix, Iy = compute_image_gradients(image)
    
    # 2. Structure Tensor 계산
    Ixx, Ixy, Iyy = compute_structure_tensor(Ix, Iy, window_size, use_gaussian, sigma)
    
    # 3. Harris Response 계산
    response = compute_harris_response(Ixx, Ixy, Iyy, k)
    
    # 4. 코너 검출
    corners = detect_corners(response, threshold, nms_window_size, max_corners)
    
    return corners

