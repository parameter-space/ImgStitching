"""
기본 유틸리티 함수들
이미지 I/O, 그레이스케일 변환, 정규화 등
"""

import numpy as np
import cv2
import os
from typing import List


def load_images_from_folder(folder_path: str) -> List[np.ndarray]:
    """
    폴더에서 이미지를 로드합니다.
    지원하는 확장자: jpg, jpeg, png, bmp, tiff, tif (대소문자 구분 없음)
    
    Args:
        folder_path: 이미지가 있는 폴더 경로 (str)
    
    Returns:
        images: 이미지 리스트 (List[np.ndarray]), 각 이미지는 (H, W, 3) 형태 - uint8
    """
    images = []
    
    # 폴더 안의 모든 이미지 파일 찾기 (다양한 확장자 지원)
    import glob
    # 지원하는 이미지 확장자 목록 (대소문자 모두)
    image_extensions = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', 
                       '*.png', '*.PNG', 
                       '*.bmp', '*.BMP',
                       '*.tiff', '*.TIFF', '*.tif', '*.TIF']
    
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
    
    if len(image_files) == 0:
        return images
    
    # Windows에서는 대소문자 구분 없이 중복 제거
    # 정규화된 경로를 사용하여 중복 확인
    seen_paths = set()
    unique_files = []
    for path in image_files:
        normalized_path = os.path.normpath(os.path.normcase(path))
        if normalized_path not in seen_paths:
            seen_paths.add(normalized_path)
            unique_files.append(path)
    
    image_files = unique_files
    
    # 파일명 기준으로 정렬 (testimg1.jpg, testimg2.png, ... 순서)
    def natural_sort_key(path: str):
        import re
        filename = os.path.basename(path)
        parts = re.split(r'(\d+)', filename)
        return [int(c) if c.isdigit() else c.lower() for c in parts]
    
    image_files.sort(key=natural_sort_key)
    
    # 이미지 로드
    for filepath in image_files:
        img = cv2.imread(filepath)
        if img is not None:
            # BGR -> RGB 변환
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
    
    return images


def rgb_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    RGB 이미지를 그레이스케일로 변환합니다.
    
    Args:
        image: RGB 이미지 (H, W, 3) - uint8
    
    Returns:
        gray: 그레이스케일 이미지 (H, W) - float32, 0~1 범위로 정규화
    """
    # RGB -> Grayscale 변환 공식: Y = 0.299*R + 0.587*G + 0.114*B
    if len(image.shape) == 3:
        gray = 0.299 * image[:, :, 0].astype(np.float32) + \
               0.587 * image[:, :, 1].astype(np.float32) + \
               0.114 * image[:, :, 2].astype(np.float32)
    else:
        gray = image.astype(np.float32)
    
    # 0~255 범위를 0~1로 정규화
    gray = gray / 255.0
    
    return gray


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    이미지 정규화: 밝기 차이를 극복하기 위해 평균을 0, 표준편차를 1로 정규화합니다.
    
    Args:
        image: 그레이스케일 이미지 (H, W) - float32
    
    Returns:
        normalized: 정규화된 이미지 (H, W) - float32
    """
    mean = np.mean(image)
    std = np.std(image)
    
    if std < 1e-10:  # 표준편차가 거의 0인 경우 (단색 이미지)
        return image - mean
    
    normalized = (image - mean) / std
    
    return normalized


def save_image(image: np.ndarray, filepath: str):
    """
    이미지를 저장합니다.
    
    Args:
        image: 이미지 (H, W, 3) 또는 (H, W) - uint8 또는 float32
        filepath: 저장 경로 (str)
    """
    # float32이고 0~1 범위면 uint8 0~255로 변환
    if image.dtype == np.float32 or image.dtype == np.float64:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)
    
    # RGB -> BGR 변환 (OpenCV는 BGR 형식 사용)
    if len(image.shape) == 3:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image
    
    cv2.imwrite(filepath, image_bgr)


def show_image(image: np.ndarray, window_name: str = "Image"):
    """
    이미지를 화면에 표시합니다.
    
    Args:
        image: 이미지 (H, W, 3) 또는 (H, W) - uint8 또는 float32
        window_name: 창 이름 (str)
    """
    # float32이고 0~1 범위면 uint8 0~255로 변환
    if image.dtype == np.float32 or image.dtype == np.float64:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)
    
    # RGB -> BGR 변환
    if len(image.shape) == 3:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image
    
    cv2.imshow(window_name, image_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

