"""
이미지 스티칭 모듈
Center-Reference Stitching 구현 (필수 구현)
"""

import numpy as np
import cv2
from typing import Tuple, List
from homography import apply_homography


def compute_canvas_size_center_ref(images: List[np.ndarray], global_homographies: List[np.ndarray]) -> Tuple[int, int, np.ndarray]:
    """
    Center-Reference 방식으로 Canvas 크기를 계산합니다.
    
    Args:
        images: 이미지 리스트
        global_homographies: 전역 Homography 리스트 (각 이미지를 center 이미지 좌표계로 변환)
    
    Returns:
        width: Canvas 너비 (int)
        height: Canvas 높이 (int)
        offset: 오프셋 [offset_x, offset_y] (np.ndarray)
    """
    all_corners = []
    
    for i, (image, H_global) in enumerate(zip(images, global_homographies)):
        H, W = image.shape[:2]
        corners = np.array([[0, 0], [W, 0], [W, H], [0, H]], dtype=np.float32)
        transformed_corners = apply_homography(corners, H_global)
        all_corners.append(transformed_corners)
        
        # Safety check
        pair_min_x = np.min(transformed_corners[:, 0])
        pair_max_x = np.max(transformed_corners[:, 0])
        pair_min_y = np.min(transformed_corners[:, 1])
        pair_max_y = np.max(transformed_corners[:, 1])
        pair_width = pair_max_x - pair_min_x
        pair_height = pair_max_y - pair_min_y
        
        if pair_width > 30000 or pair_height > 30000:
            print(f"  *** WARNING: Image {i+1} would create huge canvas ({pair_width:.0f}x{pair_height:.0f}px). Clamping offset. ***")
            transformed_corners[:, 0] = np.clip(transformed_corners[:, 0], -30000, 30000)
            transformed_corners[:, 1] = np.clip(transformed_corners[:, 1], -30000, 30000)
            all_corners[-1] = transformed_corners
    
    all_corners = np.vstack(all_corners)
    
    min_x = np.min(all_corners[:, 0])
    max_x = np.max(all_corners[:, 0])
    min_y = np.min(all_corners[:, 1])
    max_y = np.max(all_corners[:, 1])
    
    width = int(np.ceil(max_x - min_x))
    height = int(np.ceil(max_y - min_y))
    
    # Safety check: NaN이나 Inf 값 확인
    if not np.isfinite(width) or not np.isfinite(height):
        print(f"  *** ERROR: Invalid canvas size computed (width: {width}, height: {height}). Using default size. ***")
        # 기본값: 첫 번째 이미지의 크기 사용
        if len(images) > 0:
            default_h, default_w = images[0].shape[:2]
            width = default_w * len(images)  # 이미지 개수만큼 가정
            height = default_h
        else:
            width = 1000
            height = 1000
    
    # Safety check: 0 또는 음수 크기 확인
    if width <= 0 or height <= 0:
        print(f"  *** ERROR: Canvas size is invalid (width: {width}, height: {height}). Using default size. ***")
        if len(images) > 0:
            default_h, default_w = images[0].shape[:2]
            width = default_w * len(images)
            height = default_h
        else:
            width = 1000
            height = 1000
    
    # Safety check: 너무 큰 크기 제한
    if width > 30000 or height > 30000:
        print(f"  *** ERROR: Total panorama size ({width}x{height}px) exceeds 30000px limit. ***")
        width = min(width, 30000)
        height = min(height, 30000)
    
    # Offset to shift negative coordinates to positive
    offset_x = int(np.floor(-min_x))
    offset_y = int(np.floor(-min_y))
    
    # Safety check: NaN이나 Inf 값 확인
    if not np.isfinite(offset_x) or not np.isfinite(offset_y):
        print(f"  *** ERROR: Invalid offset computed (offset_x: {offset_x}, offset_y: {offset_y}). Using default offset. ***")
        offset_x = 0
        offset_y = 0
    
    # Clamp offset
    offset_x = np.clip(offset_x, -30000, 30000)
    offset_y = np.clip(offset_y, -30000, 30000)
    
    return width, height, np.array([offset_x, offset_y], dtype=np.int32)


def inverse_warp(image: np.ndarray, H: np.ndarray, output_shape: Tuple[int, int], offset: np.ndarray) -> np.ndarray:
    """
    Homography를 사용하여 이미지를 역변환(역매핑)합니다.
    cv2.remap을 사용하여 효율적으로 구현합니다.
    
    Args:
        image: 입력 이미지 (H_src, W_src, C) - uint8
        H: Homography 행렬 (3, 3) - float32 (Image -> Canvas)
        output_shape: 출력 크기 (H_out, W_out)
        offset: 오프셋 [offset_x, offset_y]
    
    Returns:
        warped: 변환된 이미지 (H_out, W_out, C) - uint8
    """
    H_inv = np.linalg.inv(H)  # Canvas -> Image
    H_out, W_out = output_shape
    offset_x, offset_y = offset
    H_src, W_src = image.shape[:2]
    C = image.shape[2] if len(image.shape) == 3 else 1
    
    # Initialize output
    if C == 1:
        warped = np.zeros((H_out, W_out), dtype=np.uint8)
    else:
        warped = np.zeros((H_out, W_out, C), dtype=np.uint8)
    
    # Tile-based processing for large canvases
    tile_size = 1000
    
    for y_start in range(0, H_out, tile_size):
        y_end = min(y_start + tile_size, H_out)
        for x_start in range(0, W_out, tile_size):
            x_end = min(x_start + tile_size, W_out)
            
            tile_h = y_end - y_start
            tile_w = x_end - x_start
            
            # Generate grid for this tile in Canvas coordinates
            # np.meshgrid: 타일 내의 모든 픽셀 좌표 생성
            # indexing='ij': (i, j) 인덱싱 방식 (행, 열 순서)
            y_tile, x_tile = np.meshgrid(
                np.arange(y_start, y_end, dtype=np.float32),
                np.arange(x_start, x_end, dtype=np.float32),
                indexing='ij'
            )
            
            # Shift by offset (Canvas -> Global): Canvas 좌표를 전역 좌표로 변환
            x_tile = x_tile - offset_x
            y_tile = y_tile - offset_y
            
            # Apply H_inv (Global -> Source Image): 역변환으로 원본 이미지 좌표 계산
            # 동차 좌표로 변환: [x, y, 1]
            ones = np.ones_like(x_tile)
            # np.stack: 배열들을 쌓아서 (3, N) 형태로 만듦
            coords = np.stack([x_tile.flatten(), y_tile.flatten(), ones.flatten()])
            # Homography 역변환 적용
            src_coords = H_inv @ coords
            
            # Normalize: 동차 좌표를 일반 좌표로 변환
            w = src_coords[2, :]  # w 값
            w = np.where(np.abs(w) < 1e-10, 1e-10, w)  # Division by zero 방지
            # [x'/w, y'/w]로 변환 후 원래 타일 크기로 reshape
            x_src = (src_coords[0, :] / w).reshape(tile_h, tile_w)
            y_src = (src_coords[1, :] / w).reshape(tile_h, tile_w)
            
            # Use cv2.remap for fast bilinear interpolation
            # cv2.remap: 역매핑 방식으로 이미지 변환 (효율적)
            # map_x, map_y: 각 출력 픽셀이 가져올 원본 이미지의 좌표
            map_x = x_src.astype(np.float32)
            map_y = y_src.astype(np.float32)
            
            # cv2.remap: map_x, map_y에 따라 원본 이미지에서 픽셀 추출
            # cv2.INTER_LINEAR: 양선형 보간 (빠르고 부드러움)
            # borderMode=cv2.BORDER_CONSTANT: 경계 밖은 0으로 채움
            tile_warped = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, 
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            
            # Copy to output
            if C == 1:
                warped[y_start:y_end, x_start:x_end] = tile_warped
            else:
                warped[y_start:y_end, x_start:x_end] = tile_warped
    
    return warped


def stitch_multiple_images(images: List[np.ndarray], homographies: List[np.ndarray]) -> np.ndarray:
    """
    Center-Reference 방식으로 여러 이미지를 스티칭합니다.
    
    Args:
        images: 이미지 리스트 [img1, img2, ...]
        homographies: Homography 리스트 [H1, H2, ...]
                     H[i]는 images[i+1]을 images[i] 좌표계로 변환
    
    Returns:
        panorama: 스티칭된 파노라마 이미지 (H, W, 3) - uint8
    """
    if not images:
        return None
    
    # Center-Reference Stitching: 중간 이미지를 기준(Identity)으로 설정
    # 이렇게 하면 오차가 양쪽으로 분산되어 누적 오차 감소
    mid = len(images) // 2  # 중간 이미지 인덱스
    global_homos = [None] * len(images)
    global_homos[mid] = np.eye(3, dtype=np.float32)  # 중간 이미지는 Identity (기준)
    
    # Propagate Right (mid -> end): 오른쪽 이미지들에 대해 전역 Homography 계산
    # H[i]는 images[i+1]을 images[i]로 변환하므로, 누적하여 중간 이미지 좌표계로 변환
    for i in range(mid, len(images) - 1):
        # global_homos[i+1] = global_homos[i] @ homographies[i]
        # 현재까지의 전역 변환에 다음 변환을 누적
        global_homos[i+1] = global_homos[i] @ homographies[i]
    
    # Propagate Left (mid -> 0): 왼쪽 이미지들에 대해 전역 Homography 계산
    # 역방향이므로 homographies[i]의 역행렬 사용
    for i in range(mid - 1, -1, -1):
        # H[i]는 images[i+1]을 images[i]로 변환하므로,
        # 역행렬은 images[i]를 images[i+1]로 변환
        H_inv = np.linalg.inv(homographies[i])
        # 이미 계산된 i+1의 전역 변환에 역변환을 적용
        global_homos[i] = global_homos[i+1] @ H_inv
    
    # Compute Canvas
    W, H, offset = compute_canvas_size_center_ref(images, global_homos)
    print(f"Canvas Size: {W}x{H}, Offset: {offset}")
    
    # Warp and Blend
    panorama = np.zeros((H, W, 3), dtype=np.uint8)
    
    # Process center first, then outwards: 중간 이미지부터 양쪽으로 순차 처리
    # 이렇게 하면 중간 이미지를 먼저 배치하고 주변 이미지를 덮어씀
    indices = [mid]  # 중간 이미지 먼저
    for i in range(1, len(images)):
        if mid - i >= 0:  # 왼쪽 이미지
            indices.append(mid - i)
        if mid + i < len(images):  # 오른쪽 이미지
            indices.append(mid + i)
    
    for i in indices:
        print(f"Warping image {i+1}...")
        # 역변환으로 이미지를 Canvas에 투영
        warped = inverse_warp(images[i], global_homos[i], (H, W), offset)
        
        # Simple masking and overlay: 간단한 마스킹 및 덮어쓰기
        # np.any(warped > 0, axis=2): RGB 중 하나라도 0이 아니면 True (픽셀이 있음)
        mask = np.any(warped > 0, axis=2)
        # 마스크가 True인 위치에 warped 이미지 복사 (덮어쓰기)
        panorama[mask] = warped[mask]
    
    return panorama

