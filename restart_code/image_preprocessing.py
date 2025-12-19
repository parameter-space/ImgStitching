"""
이미지 전처리 및 노이즈 제거 모듈
선택적 구현 (추가 점수)
"""

import numpy as np
from typing import Tuple


def apply_gaussian_blur(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    Gaussian Blur를 적용하여 노이즈를 제거합니다.
    NumPy를 사용하여 직접 구현합니다.
    
    Args:
        image: 그레이스케일 이미지 배열 (H, W) - float32
        kernel_size: 커널 크기 (int, 홀수여야 함, 기본값: 5)
        sigma: Gaussian 표준편차 (float, 기본값: 1.0)
    
    Returns:
        blurred: 블러 처리된 이미지 (H, W) - float32
    """
    if kernel_size % 2 == 0:
        kernel_size += 1  # 홀수로 만들기
    
    half = kernel_size // 2
    H, W = image.shape
    
    # Gaussian kernel 생성
    # np.arange: 일정한 간격의 배열 생성
    # 예: np.arange(-2, 3) = [-2, -1, 0, 1, 2]
    x = np.arange(-half, half + 1, dtype=np.float32)
    y = np.arange(-half, half + 1, dtype=np.float32)
    # np.meshgrid: 1D 배열을 2D 좌표 격자로 변환
    # X, Y: 각 픽셀의 (x, y) 좌표를 담은 2D 배열
    X, Y = np.meshgrid(x, y)
    # Gaussian 함수: exp(-(x^2 + y^2) / (2*sigma^2))
    kernel = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    kernel = kernel / np.sum(kernel)  # 정규화: 합이 1이 되도록 (컨볼루션 시 밝기 유지)
    
    # 패딩: 이미지 경계를 확장하여 컨볼루션 시 경계 처리
    # np.pad: 배열에 패딩 추가
    # ((half, half), (half, half)): (위, 아래), (왼쪽, 오른쪽) 패딩 크기
    # mode='reflect': 경계를 반사 (거울 효과)
    padded = np.pad(image, ((half, half), (half, half)), mode='reflect')
    
    # Convolution: 각 픽셀에 대해 kernel을 적용
    # np.zeros_like: 입력 배열과 동일한 크기의 0 배열 생성
    blurred = np.zeros_like(image, dtype=np.float32)
    for i in range(H):
        for j in range(W):
            # 패치 추출: padded[i:i+kernel_size, j:j+kernel_size]는 (kernel_size, kernel_size) 크기의 윈도우
            patch = padded[i:i+kernel_size, j:j+kernel_size]
            # Element-wise 곱셈 후 합산: np.sum(patch * kernel)은 내적과 유사
            blurred[i, j] = np.sum(patch * kernel)
    
    return blurred


def preprocess_image(image: np.ndarray, denoise: bool = True) -> np.ndarray:
    """
    이미지 전처리 파이프라인.
    
    Args:
        image: 그레이스케일 이미지 배열 (H, W) - float32
        denoise: 노이즈 제거 여부 (bool, 기본값: True)
    
    Returns:
        processed: 전처리된 이미지 (H, W) - float32
    """
    processed = image.copy()
    
    if denoise:
        # 가벼운 Gaussian Blur로 노이즈 제거
        processed = apply_gaussian_blur(processed, kernel_size=3, sigma=0.5)
    
    return processed

