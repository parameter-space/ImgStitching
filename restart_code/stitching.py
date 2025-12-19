"""
이미지 스티칭 구현
"""

import numpy as np


def apply_homography(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Homography를 사용하여 점들을 변환합니다.
    
    Args:
        points: 점들 (N, 2) - 각 행은 [x, y] - float32
        H: Homography 행렬 (3, 3) - float32
    
    Returns:
        transformed: 변환된 점들 (N, 2) - float32
    """
    N = len(points)
    
    # 동차 좌표로 변환
    points_homogeneous = np.column_stack([points, np.ones(N)])
    
    # 변환
    transformed_homogeneous = (H @ points_homogeneous.T).T
    
    # 일반 좌표로 변환
    w = transformed_homogeneous[:, 2]
    w = np.where(np.abs(w) < 1e-10, 1.0, w)  # 0으로 나누기 방지
    transformed = transformed_homogeneous[:, :2] / w[:, np.newaxis]
    
    return transformed


def bilinear_interpolate(image: np.ndarray, x: float, y: float) -> np.ndarray:
    """
    Bilinear interpolation을 사용하여 이미지에서 값을 샘플링합니다.
    
    Args:
        image: 이미지 (H, W) 또는 (H, W, C) - float32 또는 uint8
        x: x 좌표 (float)
        y: y 좌표 (float)
    
    Returns:
        value: 샘플링된 값 (C,) 또는 스칼라
    """
    H, W = image.shape[:2]
    
    # 경계 확인
    if x < 0 or x >= W - 1 or y < 0 or y >= H - 1:
        if len(image.shape) == 3:
            return np.zeros(image.shape[2], dtype=image.dtype)
        else:
            return image.dtype.type(0)
    
    # 정수 좌표
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = x0 + 1
    y1 = y0 + 1
    
    # 경계 처리
    x1 = min(x1, W - 1)
    y1 = min(y1, H - 1)
    
    # 가중치
    wx = x - x0
    wy = y - y0
    
    # Bilinear interpolation
    if len(image.shape) == 3:
        value = (1 - wx) * (1 - wy) * image[y0, x0] + \
                wx * (1 - wy) * image[y0, x1] + \
                (1 - wx) * wy * image[y1, x0] + \
                wx * wy * image[y1, x1]
    else:
        value = (1 - wx) * (1 - wy) * image[y0, x0] + \
                wx * (1 - wy) * image[y0, x1] + \
                (1 - wx) * wy * image[y1, x0] + \
                wx * wy * image[y1, x1]
    
    return value


def warp_image(image: np.ndarray, H: np.ndarray, output_shape: tuple) -> np.ndarray:
    """
    Homography를 사용하여 이미지를 변환합니다.
    
    Args:
        image: 입력 이미지 (H, W, 3) - uint8
        H: Homography 행렬 (3, 3) - float32
        output_shape: 출력 이미지 크기 (height, width) - tuple
    
    Returns:
        warped: 변환된 이미지 (output_shape[0], output_shape[1], 3) - uint8
    """
    H_out, W_out = output_shape
    
    # 역변환 (출력 좌표 -> 입력 좌표)
    H_inv = np.linalg.inv(H)
    
    warped = np.zeros((H_out, W_out, 3), dtype=np.float32)
    
    # 출력 이미지의 각 픽셀에 대해
    for y_out in range(H_out):
        for x_out in range(W_out):
            # 입력 이미지 좌표로 변환
            point_out = np.array([x_out, y_out, 1.0])
            point_in_homogeneous = H_inv @ point_out
            
            # 일반 좌표로 변환
            w = point_in_homogeneous[2]
            if abs(w) < 1e-10:
                continue
            
            x_in = point_in_homogeneous[0] / w
            y_in = point_in_homogeneous[1] / w
            
            # 입력 이미지 경계 확인
            if x_in < 0 or x_in >= image.shape[1] - 1 or \
               y_in < 0 or y_in >= image.shape[0] - 1:
                continue
            
            # Bilinear interpolation
            value = bilinear_interpolate(image, x_in, y_in)
            warped[y_out, x_out] = value
    
    return warped.astype(np.uint8)


def compute_canvas_size(images: list, homographies: list) -> tuple:
    """
    모든 이미지를 포함할 수 있는 canvas 크기를 계산합니다.
    
    Args:
        images: 이미지 리스트
        homographies: Homography 행렬 리스트
    
    Returns:
        canvas_size: (height, width) - tuple
        offset: (y_offset, x_offset) - tuple
    """
    # 첫 번째 이미지를 기준으로 설정
    H0, W0 = images[0].shape[:2]
    
    # 첫 번째 이미지의 네 모서리
    corners = np.array([
        [0, 0],           # 좌상단
        [W0 - 1, 0],      # 우상단
        [0, H0 - 1],      # 좌하단
        [W0 - 1, H0 - 1]  # 우하단
    ], dtype=np.float32)
    
    # 모든 이미지의 변환된 모서리 계산
    all_corners = [corners]
    
    # 누적 Homography 계산 (첫 번째 이미지 기준)
    cumulative_H = np.eye(3, dtype=np.float32)
    
    for i, H in enumerate(homographies):
        # H[i]는 images[i+1]을 images[i]로 변환
        # 첫 번째 이미지 기준으로 변환하려면 누적해야 함
        cumulative_H = cumulative_H @ H
        H_inv = np.linalg.inv(cumulative_H)
        
        H_img, W_img = images[i + 1].shape[:2]
        img_corners = np.array([
            [0, 0],
            [W_img - 1, 0],
            [0, H_img - 1],
            [W_img - 1, H_img - 1]
        ], dtype=np.float32)
        
        transformed_corners = apply_homography(img_corners, H_inv)
        all_corners.append(transformed_corners)
    
    # 모든 모서리 결합
    all_corners = np.vstack(all_corners)
    
    # Bounding box 계산
    min_x = np.min(all_corners[:, 0])
    max_x = np.max(all_corners[:, 0])
    min_y = np.min(all_corners[:, 1])
    max_y = np.max(all_corners[:, 1])
    
    # Offset 계산 (음수 좌표를 양수로 만들기)
    x_offset = int(np.floor(min_x)) if min_x < 0 else 0
    y_offset = int(np.floor(min_y)) if min_y < 0 else 0
    
    # Canvas 크기
    width = int(np.ceil(max_x)) - x_offset + 1
    height = int(np.ceil(max_y)) - y_offset + 1
    
    return (height, width), (y_offset, x_offset)


def stitch_multiple_images(images: list, homographies: list) -> np.ndarray:
    """
    여러 이미지를 스티칭합니다.
    
    Args:
        images: 이미지 리스트 [img1, img2, ...], 각 이미지는 (H, W, 3) - uint8
        homographies: Homography 행렬 리스트 [H1, H2, ...], 각 H는 (3, 3) - float32
                     H[i]는 images[i+1]을 images[i] 좌표계로 변환
    
    Returns:
        panorama: 파노라마 이미지 (H, W, 3) - uint8
    """
    if len(images) < 1:
        return None
    
    if len(images) == 1:
        return images[0]
    
    # Canvas 크기 계산
    canvas_size, offset = compute_canvas_size(images, homographies)
    H_canvas, W_canvas = canvas_size
    y_offset, x_offset = offset
    
    # Canvas 초기화
    panorama = np.zeros((H_canvas, W_canvas, 3), dtype=np.float32)
    count = np.zeros((H_canvas, W_canvas), dtype=np.float32)
    
    # 첫 번째 이미지 추가
    y1_start = y_offset
    y1_end = y_offset + images[0].shape[0]
    x1_start = x_offset
    x1_end = x_offset + images[0].shape[1]
    
    panorama[y1_start:y1_end, x1_start:x1_end] = images[0].astype(np.float32)
    count[y1_start:y1_end, x1_start:x1_end] = 1.0
    
    # 나머지 이미지들 추가
    cumulative_H = np.eye(3, dtype=np.float32)
    
    for i, H in enumerate(homographies):
        # 누적 Homography 계산
        cumulative_H = cumulative_H @ H
        
        # 첫 번째 이미지 좌표계로 변환하는 Homography (역변환)
        H_to_first = np.linalg.inv(cumulative_H)
        
        # Translation 추가 (offset 고려)
        T = np.array([
            [1, 0, x_offset],
            [0, 1, y_offset],
            [0, 0, 1]
        ], dtype=np.float32)
        
        H_final = T @ H_to_first
        
        # 이미지 변환
        H_img, W_img = images[i + 1].shape[:2]
        
        for y_out in range(H_canvas):
            for x_out in range(W_canvas):
                # Canvas 좌표를 offset을 뺀 좌표로 변환
                x_canvas = x_out - x_offset
                y_canvas = y_out - y_offset
                
                # 입력 이미지 좌표로 변환
                point_canvas = np.array([x_canvas, y_canvas, 1.0])
                point_img_homogeneous = H_final @ point_canvas
                
                w = point_img_homogeneous[2]
                if abs(w) < 1e-10:
                    continue
                
                x_img = point_img_homogeneous[0] / w
                y_img = point_img_homogeneous[1] / w
                
                # 입력 이미지 경계 확인
                if x_img >= 0 and x_img < W_img - 1 and \
                   y_img >= 0 and y_img < H_img - 1:
                    # Bilinear interpolation
                    value = bilinear_interpolate(images[i + 1], x_img, y_img)
                    panorama[y_out, x_out] += value
                    count[y_out, x_out] += 1.0
    
    # 평균 계산 (blending)
    mask = count > 0
    panorama[mask] = panorama[mask] / count[mask, np.newaxis]
    
    return panorama.astype(np.uint8)

