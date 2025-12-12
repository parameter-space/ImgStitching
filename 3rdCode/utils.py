"""
유틸리티 함수 모듈
이미지 로드, 그레이스케일 변환, 기본 이미지 처리 함수들을 포함합니다.
"""

import numpy as np
import cv2
import os
from typing import List, Tuple, Union


def load_image(image_path: str) -> np.ndarray:
    """
    이미지 파일을 로드합니다.
    
    Args:
        image_path: 이미지 파일 경로 (str)
    
    Returns:
        image: BGR 형식의 이미지 배열 (H, W, 3) - uint8
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
    return image


def load_images_from_folder(folder_path: str) -> List[np.ndarray]:
    """
    폴더에서 모든 이미지를 로드합니다.
    
    Args:
        folder_path: 이미지가 있는 폴더 경로 (str)
    
    Returns:
        images: 이미지 배열 리스트 [image1, image2, ...]
                각 이미지는 (H, W, 3) 형태 - uint8
    """
    import glob
    
    # 지원하는 이미지 확장자
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    image_paths = []
    
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
        image_paths.extend(glob.glob(os.path.join(folder_path, ext.upper())))
    
    # 파일명으로 정렬
    image_paths.sort()
    
    images = []
    for path in image_paths:
        try:
            img = load_image(path)
            images.append(img)
        except Exception as e:
            print(f"경고: {path} 로드 실패: {e}")
    
    return images


def rgb_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    RGB/BGR 이미지를 그레이스케일로 변환합니다.
    
    Args:
        image: BGR 형식의 이미지 배열 (H, W, 3) - uint8
    
    Returns:
        gray: 그레이스케일 이미지 배열 (H, W) - float32
    """
    if len(image.shape) == 3:
        # BGR to Grayscale: 0.299*R + 0.587*G + 0.114*B
        # OpenCV는 BGR 순서이므로 B, G, R 순서
        gray = 0.114 * image[:, :, 0].astype(np.float32) + \
               0.587 * image[:, :, 1].astype(np.float32) + \
               0.299 * image[:, :, 2].astype(np.float32)
    else:
        gray = image.astype(np.float32)
    
    return gray


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    이미지를 0-1 범위로 정규화합니다.
    
    Args:
        image: 이미지 배열 (H, W) 또는 (H, W, C) - uint8 또는 float
    
    Returns:
        normalized: 정규화된 이미지 배열 (H, W) 또는 (H, W, C) - float32
    """
    if image.dtype == np.uint8:
        normalized = image.astype(np.float32) / 255.0
    else:
        # 이미 float인 경우 최대값으로 정규화
        max_val = np.max(image)
        if max_val > 1.0:
            normalized = image.astype(np.float32) / max_val
        else:
            normalized = image.astype(np.float32)
    
    return normalized


def denormalize_image(image: np.ndarray) -> np.ndarray:
    """
    정규화된 이미지를 0-255 범위로 되돌립니다.
    
    Args:
        image: 정규화된 이미지 배열 (H, W) 또는 (H, W, C) - float32
    
    Returns:
        denormalized: 원래 범위의 이미지 배열 (H, W) 또는 (H, W, C) - uint8
    """
    denormalized = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return denormalized


def show_image(image: np.ndarray, window_name: str = "Image") -> None:
    """
    이미지를 화면에 표시합니다.
    
    Args:
        image: 이미지 배열 (H, W) 또는 (H, W, 3) - uint8
        window_name: 윈도우 이름 (str)
    """
    cv2.imshow(window_name, image)


def save_image(image: np.ndarray, output_path: str) -> None:
    """
    이미지를 파일로 저장합니다.
    
    Args:
        image: 이미지 배열 (H, W) 또는 (H, W, 3) - uint8
        output_path: 저장할 파일 경로 (str)
    """
    cv2.imwrite(output_path, image)

