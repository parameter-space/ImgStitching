"""
유틸리티 함수 모듈
이미지 로드, 저장, 그레이스케일 변환 등의 기본 기능
"""

import numpy as np
import cv2
import os
import re
from typing import List, Union


def load_image(image_path: str) -> np.ndarray:
    """
    이미지 파일을 로드합니다.
    
    Args:
        image_path: 이미지 파일 경로 (str)
    
    Returns:
        image: BGR 형식의 이미지 배열 (H, W, 3) - uint8
                주의: OpenCV는 BGR 순서로 이미지를 로드합니다 (RGB가 아님)
    """
    # cv2.imread: OpenCV의 이미지 로드 함수
    # - 성공 시: (H, W, 3) numpy 배열 반환 (BGR 형식, uint8)
    # - 실패 시: None 반환
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
    return image


def load_images_from_folder(folder_path: str) -> List[np.ndarray]:
    """
    폴더에서 모든 이미지를 로드합니다.
    Natural Sort를 사용하여 올바른 순서로 정렬합니다.
    
    Args:
        folder_path: 이미지가 있는 폴더 경로 (str)
    
    Returns:
        images: 이미지 배열 리스트 [image1, image2, ...]
                각 이미지는 (H, W, 3) 형태 - uint8
    """
    import glob
    
    # 지원하는 이미지 확장자
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif', 
                  '*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.TIFF', '*.TIF']
    image_paths = []
    
    # 모든 확장자로 검색
    for ext in extensions:
        paths = glob.glob(os.path.join(folder_path, ext))
        image_paths.extend(paths)
    
    # 중복 제거
    unique_paths = []
    seen_paths = set()
    for path in image_paths:
        path_normalized = os.path.normpath(os.path.normcase(path))
        if path_normalized not in seen_paths:
            seen_paths.add(path_normalized)
            unique_paths.append(path)
    
    # Natural Sort: 숫자가 포함된 파일명을 올바르게 정렬
    # 예: "testimg1.jpg", "testimg2.jpg", "testimg10.jpg" 순서로 정렬
    # ASCII 정렬은 "testimg10.jpg"가 "testimg2.jpg"보다 먼저 나오는 문제가 있음
    def natural_sort_key(path: str) -> List[Union[int, str]]:
        filename = os.path.basename(path)
        # re.split(r'(\d+)', ...): 숫자와 비숫자를 분리
        # 예: "testimg10.jpg" -> ["testimg", "10", ".jpg"]
        parts = re.split(r'(\d+)', filename)
        # 숫자는 int로, 문자는 소문자로 변환하여 정렬 키 생성
        # 예: ["testimg", 10, ".jpg"] -> 정렬 시 숫자가 올바르게 비교됨
        return [int(c) if c.isdigit() else c.lower() for c in parts]
    
    # sort(key=...): 정렬 키 함수를 사용하여 정렬
    unique_paths.sort(key=natural_sort_key)
    
    images = []
    for path in unique_paths:
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
        # BGR to Grayscale 변환 (ITU-R BT.601 표준)
        # 공식: Y = 0.299*R + 0.587*G + 0.114*B
        # OpenCV는 BGR 순서이므로:
        # - image[:, :, 0] = Blue 채널
        # - image[:, :, 1] = Green 채널
        # - image[:, :, 2] = Red 채널
        # astype(np.float32): uint8 (0-255)를 float32로 변환 (계산 정확도 향상)
        gray = 0.114 * image[:, :, 0].astype(np.float32) + \
               0.587 * image[:, :, 1].astype(np.float32) + \
               0.299 * image[:, :, 2].astype(np.float32)
    else:
        # 이미 그레이스케일인 경우
        gray = image.astype(np.float32)
    
    return gray


def save_image(image: np.ndarray, output_path: str) -> None:
    """
    이미지를 파일로 저장합니다.
    
    Args:
        image: 이미지 배열 (H, W) 또는 (H, W, 3) - uint8
        output_path: 저장할 파일 경로 (str)
    """
    cv2.imwrite(output_path, image)


def show_image(image: np.ndarray, window_name: str = "Image") -> None:
    """
    이미지를 화면에 표시합니다.
    
    Args:
        image: 이미지 배열 (H, W) 또는 (H, W, 3) - uint8
        window_name: 윈도우 이름 (str)
    """
    # 유효성 검사: 이미지가 None이거나 크기가 0이면 표시하지 않음
    if image is None:
        print(f"경고: {window_name} 이미지가 None입니다.")
        return
    
    # 이미지 크기 확인
    if image.size == 0:
        print(f"경고: {window_name} 이미지의 크기가 0입니다.")
        return
    
    # 데이터 타입 확인 및 변환
    if image.dtype != np.uint8:
        print(f"경고: {window_name} 이미지의 데이터 타입이 uint8이 아닙니다 ({image.dtype}). 변환합니다.")
        # float 타입인 경우 0-1 범위 또는 0-255 범위로 가정
        if image.dtype in [np.float32, np.float64]:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = np.clip(image, 0, 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    # 이미지 shape 확인
    if len(image.shape) not in [2, 3]:
        print(f"경고: {window_name} 이미지의 차원이 올바르지 않습니다 (shape: {image.shape})")
        return
    
    # 높이 또는 너비가 0이면 표시하지 않음
    if image.shape[0] == 0 or image.shape[1] == 0:
        print(f"경고: {window_name} 이미지의 크기가 0입니다 (shape: {image.shape})")
        return
    
    try:
        cv2.imshow(window_name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error as e:
        print(f"경고: {window_name} 이미지를 표시하는 중 오류가 발생했습니다: {e}")
        print(f"  이미지 shape: {image.shape}, dtype: {image.dtype}")

