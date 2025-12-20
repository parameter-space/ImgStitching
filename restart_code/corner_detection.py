"""
Harris Corner Detection 구현
Corner.pdf 강의자료 참고
"""

import numpy as np


def compute_image_gradients(image: np.ndarray) -> tuple:
    """
    이미지의 x, y 방향 gradient를 계산합니다.
    
    Args:
        image: 그레이스케일 이미지 (H, W) - float32
    
    Returns:
        Ix: x 방향 gradient (H, W) - float32
        Iy: y 방향 gradient (H, W) - float32
    """
    H, W = image.shape
    
    # Sobel operator를 사용한 gradient 계산
    # x 방향: [-1, 0, 1; -2, 0, 2; -1, 0, 1]
    # y 방향: [-1, -2, -1; 0, 0, 0; 1, 2, 1]
    
    # Padding 추가
    padded = np.pad(image, ((1, 1), (1, 1)), mode='edge')
    
    # x 방향 gradient (Sobel operator)
    # 원본 image의 (i, j) 위치는 padded의 (i+1, j+1) 위치
    # Sobel x: [-1, 0, 1; -2, 0, 2; -1, 0, 1]
    # 각 픽셀 (i, j)에서: 왼쪽(j-1), 중앙(j), 오른쪽(j+1) 필요
    Ix = -1 * padded[0:H, 0:W] + 1 * padded[0:H, 2:W+2] + \
         -2 * padded[1:H+1, 0:W] + 2 * padded[1:H+1, 2:W+2] + \
         -1 * padded[2:H+2, 0:W] + 1 * padded[2:H+2, 2:W+2]
    Ix = Ix / 4.0
    
    # y 방향 gradient (Sobel operator)
    # Sobel y: [-1, -2, -1; 0, 0, 0; 1, 2, 1]
    # 각 픽셀 (i, j)에서: 위(i-1), 중앙(i), 아래(i+1) 필요
    Iy = -1 * padded[0:H, 0:W] - 2 * padded[0:H, 1:W+1] - 1 * padded[0:H, 2:W+2] + \
          1 * padded[2:H+2, 0:W] + 2 * padded[2:H+2, 1:W+1] + 1 * padded[2:H+2, 2:W+2]
    Iy = Iy / 4.0
    
    return Ix, Iy


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """
    Gaussian kernel을 생성합니다.
    
    Args:
        size: kernel 크기 (int, 홀수)
        sigma: 표준편차 (float)
    
    Returns:
        kernel: Gaussian kernel (size, size) - float32
    """
    center = size // 2
    x, y = np.meshgrid(np.arange(size) - center, np.arange(size) - center)
    
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = kernel / np.sum(kernel)  # 정규화
    
    return kernel.astype(np.float32)


def harris_corner_detection(image: np.ndarray, 
                            threshold: float = 0.01,
                            k: float = 0.04,
                            window_size: int = 3,
                            sigma: float = 1.0) -> np.ndarray:
    """
    Harris Corner Detection 알고리즘을 구현합니다.
    
    수식:
    - M = sum(w(x,y) * [Ix^2, IxIy; IxIy, Iy^2])
    - R = det(M) - k * trace(M)^2
    - R > threshold 인 점을 corner로 판단
    
    Args:
        image: 그레이스케일 이미지 (H, W) - float32
        threshold: corner response threshold (float, 기본값: 0.01)
        k: Harris 상수 (float, 기본값: 0.04)
        window_size: window 크기 (int, 기본값: 3)
        sigma: Gaussian smoothing 표준편차 (float, 기본값: 1.0)
    
    Returns:
        corners: 코너 점들의 좌표 (N, 2) - 각 행은 [x, y] - float32
    """
    # 1. 이미지 gradient 계산
    Ix, Iy = compute_image_gradients(image)
    
    # 2. Structure Tensor 요소 계산
    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    Ixy = Ix * Iy
    
    # 3. Gaussian window를 사용한 weighted sum
    gaussian = gaussian_kernel(window_size, sigma)
    pad_size = window_size // 2
    
    # Padding 추가
    Ix2_padded = np.pad(Ix2, pad_size, mode='constant')
    Iy2_padded = np.pad(Iy2, pad_size, mode='constant')
    Ixy_padded = np.pad(Ixy, pad_size, mode='constant')
    
    # Convolution 수행 (2D convolution)
    H, W = image.shape
    M11 = np.zeros((H, W), dtype=np.float32)
    M12 = np.zeros((H, W), dtype=np.float32)
    M22 = np.zeros((H, W), dtype=np.float32)
    
    for i in range(H):
        for j in range(W):
            patch_x2 = Ix2_padded[i:i+window_size, j:j+window_size]
            patch_y2 = Iy2_padded[i:i+window_size, j:j+window_size]
            patch_xy = Ixy_padded[i:i+window_size, j:j+window_size]
            
            M11[i, j] = np.sum(patch_x2 * gaussian)
            M12[i, j] = np.sum(patch_xy * gaussian)
            M22[i, j] = np.sum(patch_y2 * gaussian)
    
    # 4. Harris Response 계산
    # R = det(M) - k * trace(M)^2
    # det(M) = M11 * M22 - M12^2
    # trace(M) = M11 + M22
    det_M = M11 * M22 - M12 * M12
    trace_M = M11 + M22
    R = det_M - k * trace_M * trace_M
    
    # 5. Thresholding
    corner_mask = R > threshold
    
    # 6. Non-Maximum Suppression (NMS)
    # Local maximum만 선택
    nms_size = 3
    nms_pad = nms_size // 2
    R_padded = np.pad(R, nms_pad, mode='constant', constant_values=-np.inf)
    
    corners_list = []
    corner_responses = []  # Response 값도 저장
    
    for i in range(H):
        for j in range(W):
            if corner_mask[i, j]:
                # Local neighborhood 확인
                neighborhood = R_padded[i:i+nms_size, j:j+nms_size]
                max_val = np.max(neighborhood)
                
                # 현재 점이 local maximum인지 확인
                if R[i, j] >= max_val - 1e-6:  # 부동소수점 오차 고려
                    corners_list.append([j, i])  # [x, y] 형식
                    corner_responses.append(R[i, j])
    
    if len(corners_list) == 0:
        return np.array([], dtype=np.float32).reshape(0, 2)
    
    # 7. 최대 코너 개수 제한 (성능 개선)
    # Harris Response 값이 높은 순으로 정렬하여 상위 N개만 선택
    MAX_CORNERS = 1000  # 최대 코너 개수 제한
    
    if len(corners_list) > MAX_CORNERS:
        # Response 값으로 정렬 (내림차순)
        corner_responses = np.array(corner_responses, dtype=np.float32)
        sorted_indices = np.argsort(corner_responses)[::-1]  # 내림차순
        selected_indices = sorted_indices[:MAX_CORNERS]
        
        # 상위 N개만 선택
        corners_list = [corners_list[i] for i in selected_indices]
    
    corners = np.array(corners_list, dtype=np.float32)
    
    return corners

