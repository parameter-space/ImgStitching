"""
이미지 전처리 모듈
가우시안 블러를 통한 노이즈 제거
NoiseRemoveWithGaussian.pdf 강의자료 참고
"""

import numpy as np
from typing import Union


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """
    Gaussian kernel을 생성합니다.
    
    수식:
    - G(x, y) = (1 / (2 * pi * sigma^2)) * exp(-(x^2 + y^2) / (2 * sigma^2))
    - 정규화: sum(kernel) = 1
    
    Args:
        size: kernel 크기 (int, 홀수 권장)
        sigma: 표준편차 (float)
    
    Returns:
        kernel: Gaussian kernel (size, size) - float32, 정규화됨
    """
    if size % 2 == 0:
        # 짝수 크기는 홀수로 조정
        size = size + 1
    
    center = size // 2
    
    # Meshgrid로 좌표 생성
    x, y = np.meshgrid(
        np.arange(size, dtype=np.float32) - center,
        np.arange(size, dtype=np.float32) - center,
        indexing='ij'
    )
    
    # 가우시안 함수 계산
    # G(x, y) = exp(-(x^2 + y^2) / (2 * sigma^2))
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    # 정규화: 합이 1이 되도록
    kernel = kernel / np.sum(kernel)
    
    return kernel.astype(np.float32)


def apply_gaussian_filter(image: np.ndarray, 
                          kernel_size: int = 5, 
                          sigma: float = 1.0,
                          padding_mode: str = 'edge') -> np.ndarray:
    """
    가우시안 필터를 적용하여 이미지의 노이즈를 제거합니다.
    
    NumPy를 사용하여 2D 컨볼루션을 직접 구현합니다.
    RGB 이미지인 경우 각 채널별로 필터링을 수행합니다.
    
    Args:
        image: 입력 이미지 (H, W) 또는 (H, W, 3) - uint8 또는 float32
        kernel_size: 가우시안 커널 크기 (int, 기본값: 5)
        sigma: 가우시안 표준편차 (float, 기본값: 1.0)
        padding_mode: 패딩 모드 ('edge' 또는 'zero', 기본값: 'edge')
    
    Returns:
        filtered: 필터링된 이미지 (H, W) 또는 (H, W, 3) - float32
    """
    # 입력 이미지 타입 및 형태 확인
    is_uint8 = image.dtype == np.uint8
    is_rgb = len(image.shape) == 3
    
    # float32로 변환 (0~1 범위로 정규화)
    if is_uint8:
        image_float = image.astype(np.float32) / 255.0
    else:
        image_float = image.astype(np.float32)
    
    # 가우시안 커널 생성
    kernel = gaussian_kernel(kernel_size, sigma)
    pad_size = kernel_size // 2
    
    # RGB 이미지인 경우 각 채널별로 처리
    if is_rgb:
        H, W, C = image_float.shape
        filtered = np.zeros_like(image_float, dtype=np.float32)
        
        for c in range(C):
            channel = image_float[:, :, c]
            filtered[:, :, c] = _convolve_2d(channel, kernel, pad_size, padding_mode)
    else:
        # 그레이스케일 이미지
        filtered = _convolve_2d(image_float, kernel, pad_size, padding_mode)
    
    return filtered.astype(np.float32)


def _convolve_2d(image: np.ndarray, 
                  kernel: np.ndarray, 
                  pad_size: int,
                  padding_mode: str) -> np.ndarray:
    """
    2D 컨볼루션 연산을 NumPy로 구현합니다.
    
    벡터화를 사용하여 성능을 최적화합니다.
    
    Args:
        image: 입력 이미지 (H, W) - float32
        kernel: 컨볼루션 커널 (K, K) - float32
        pad_size: 패딩 크기 (int)
        padding_mode: 패딩 모드 ('edge' 또는 'zero')
    
    Returns:
        convolved: 컨볼루션 결과 (H, W) - float32
    """
    H, W = image.shape
    K = kernel.shape[0]
    
    # Padding 추가
    if padding_mode == 'edge':
        padded = np.pad(image, pad_size, mode='edge')
    else:  # 'zero'
        padded = np.pad(image, pad_size, mode='constant', constant_values=0.0)
    
    # 벡터화된 컨볼루션 구현
    # 각 픽셀 (i, j)에 대해 kernel 크기의 패치를 추출하고 곱셈 후 합산
    # NumPy의 슬라이싱과 broadcasting을 활용하여 for loop 최소화
    
    # 결과 배열 초기화
    convolved = np.zeros((H, W), dtype=np.float32)
    
    # 벡터화: 행 단위로 처리하여 메모리 효율성과 성능의 균형 유지
    # 각 행의 모든 열에 대해 한 번에 처리
    for i in range(H):
        # i번째 행의 모든 열에 대해 패치 추출 (벡터화)
        # padded[i:i+K, pad_size:pad_size+W]에서 각 열 j에 대해 패치 [i:i+K, j:j+K] 추출
        # 슬라이싱을 사용하여 각 열의 패치를 한 번에 추출
        row_slice = padded[i:i+K, :]  # (K, padded_W)
        
        # 각 열 j에 대해 패치 [i:i+K, j:j+K] 추출
        # 벡터화: 모든 열에 대해 한 번에 처리
        patches = np.zeros((W, K, K), dtype=np.float32)
        for j in range(W):
            patches[j] = row_slice[:, j:j+K]
        
        # 각 패치와 kernel을 곱하고 합산 (벡터화)
        # patches: (W, K, K), kernel: (K, K)
        # broadcasting을 사용하여 모든 패치에 대해 한 번에 계산
        convolved[i, :] = np.sum(patches * kernel[np.newaxis, :, :], axis=(1, 2))
    
    return convolved


def preprocess_image(image: np.ndarray, 
                     kernel_size: int = 5, 
                     sigma: float = 1.0) -> np.ndarray:
    """
    파노라마 스티칭을 위한 이미지 전처리 함수.
    
    가우시안 블러를 적용하여 노이즈를 제거하고, 
    코너 검출 및 특징점 매칭의 정확도를 향상시킵니다.
    
    Args:
        image: 입력 이미지 (H, W, 3) - uint8
        kernel_size: 가우시안 커널 크기 (int, 기본값: 5)
        sigma: 가우시안 표준편차 (float, 기본값: 1.0)
    
    Returns:
        preprocessed: 전처리된 이미지 (H, W, 3) - float32, 0~1 범위
    """
    # 가우시안 필터 적용
    filtered = apply_gaussian_filter(image, kernel_size, sigma, padding_mode='edge')
    
    return filtered

