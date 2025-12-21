"""
기본 유틸리티 함수들
이미지 I/O, 그레이스케일 변환, 정규화 등
"""

import numpy as np
import cv2
import os
from typing import List


def load_images_from_folder(folder_path: str) -> List[np.ndarray]:
    images = []
    
    import glob
    image_extensions = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', 
                       '*.png', '*.PNG', 
                       '*.bmp', '*.BMP',
                       '*.tiff', '*.TIFF', '*.tif', '*.TIF']
    
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
    
    if len(image_files) == 0:
        return images
    
    seen_paths = set()
    unique_files = []
    for path in image_files:
        normalized_path = os.path.normpath(os.path.normcase(path))
        if normalized_path not in seen_paths:
            seen_paths.add(normalized_path)
            unique_files.append(path)
    
    image_files = unique_files
    
    def natural_sort_key(path: str):
        import re
        filename = os.path.basename(path)
        parts = re.split(r'(\d+)', filename)
        return [int(c) if c.isdigit() else c.lower() for c in parts]
    
    image_files.sort(key=natural_sort_key)
    
    for filepath in image_files:
        img = cv2.imread(filepath)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
    
    return images


def rgb_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Y = 0.299*R + 0.587*G + 0.114*B
    """
    if len(image.shape) == 3:
        gray = 0.299 * image[:, :, 0].astype(np.float32) + \
               0.587 * image[:, :, 1].astype(np.float32) + \
               0.114 * image[:, :, 2].astype(np.float32)
    else:
        gray = image.astype(np.float32)
    
    gray = gray / 255.0
    
    return gray


def normalize_image(image: np.ndarray) -> np.ndarray:
    mean = np.mean(image)
    std = np.std(image)
    
    if std < 1e-10:
        return image - mean
    
    normalized = (image - mean) / std
    
    return normalized


def save_image(image: np.ndarray, filepath: str):
    if image.dtype == np.float32 or image.dtype == np.float64:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)
    
    if len(image.shape) == 3:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image
    
    cv2.imwrite(filepath, image_bgr)


def show_image(image: np.ndarray, window_name: str = "Image"):
    if image.dtype == np.float32 or image.dtype == np.float64:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)
    
    if len(image.shape) == 3:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image
    
    cv2.imshow(window_name, image_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

