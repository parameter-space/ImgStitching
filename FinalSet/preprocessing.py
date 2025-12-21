"""
이미지 전처리 모듈
가우시안 블러를 통한 노이즈 제거
NoiseRemoveWithGaussian.pdf 강의자료 참고
"""

import numpy as np
from typing import Union


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """
    G(x, y) = (1 / (2 * pi * sigma^2)) * exp(-(x^2 + y^2) / (2 * sigma^2))
    """
    if size % 2 == 0:
        size = size + 1
    
    center = size // 2
    
    x, y = np.meshgrid(
        np.arange(size, dtype=np.float32) - center,
        np.arange(size, dtype=np.float32) - center,
        indexing='ij'
    )
    
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = kernel / np.sum(kernel)
    
    return kernel.astype(np.float32)


def apply_gaussian_filter(image: np.ndarray, 
                          kernel_size: int = 5, 
                          sigma: float = 1.0,
                          padding_mode: str = 'edge') -> np.ndarray:
    is_uint8 = image.dtype == np.uint8
    is_rgb = len(image.shape) == 3
    
    if is_uint8:
        image_float = image.astype(np.float32) / 255.0
    else:
        image_float = image.astype(np.float32)
    
    kernel = gaussian_kernel(kernel_size, sigma)
    pad_size = kernel_size // 2
    
    if is_rgb:
        H, W, C = image_float.shape
        filtered = np.zeros_like(image_float, dtype=np.float32)
        
        for c in range(C):
            channel = image_float[:, :, c]
            filtered[:, :, c] = _convolve_2d(channel, kernel, pad_size, padding_mode)
    else:
        filtered = _convolve_2d(image_float, kernel, pad_size, padding_mode)
    
    return filtered.astype(np.float32)


def _convolve_2d(image: np.ndarray, 
                  kernel: np.ndarray, 
                  pad_size: int,
                  padding_mode: str) -> np.ndarray:
    H, W = image.shape
    K = kernel.shape[0]
    
    if padding_mode == 'edge':
        padded = np.pad(image, pad_size, mode='edge')
    else:
        padded = np.pad(image, pad_size, mode='constant', constant_values=0.0)
    
    convolved = np.zeros((H, W), dtype=np.float32)
    
    for i in range(H):
        row_slice = padded[i:i+K, :]
        
        patches = np.zeros((W, K, K), dtype=np.float32)
        for j in range(W):
            patches[j] = row_slice[:, j:j+K]
        
        convolved[i, :] = np.sum(patches * kernel[np.newaxis, :, :], axis=(1, 2))
    
    return convolved


def preprocess_image(image: np.ndarray, 
                     kernel_size: int = 5, 
                     sigma: float = 1.0) -> np.ndarray:
    filtered = apply_gaussian_filter(image, kernel_size, sigma, padding_mode='edge')
    
    return filtered

