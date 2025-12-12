"""
이미지 스티칭 모듈
Image Warping (Inverse Mapping)과 Blending을 구현합니다.
"""

import numpy as np
from typing import Tuple, Optional


def compute_canvas_size(images: list, homographies: list) -> Tuple[int, int, np.ndarray]:
    """
    모든 이미지를 포함할 수 있는 캔버스 크기를 계산합니다.
    
    각 이미지의 4개 코너를 Homography로 변환하여 전체 영역을 계산합니다.
    
    Args:
        images: 이미지 리스트 [img1, img2, ...]
                각 이미지는 (H, W, C) 형태 - uint8
        homographies: Homography 행렬 리스트 [H1, H2, ...]
                     각 H는 (3, 3) 형태 - float32
                     H[i]는 images[i+1]을 images[0] 좌표계로 변환
    
    Returns:
        canvas_width: 캔버스 너비 (int)
        canvas_height: 캔버스 높이 (int)
        offset: 오프셋 배열 (2,) - [offset_x, offset_y] - int32
    """
    from geometry import apply_homography
    
    # 첫 번째 이미지는 변환 없음 (기준 좌표계)
    H1, W1 = images[0].shape[:2]
    corners = np.array([[0, 0], [W1, 0], [W1, H1], [0, H1]], dtype=np.float32)
    
    all_corners = [corners]
    
    # 누적 Homography 계산 (각 이미지를 첫 번째 이미지 좌표계로 변환)
    cumulative_H = np.eye(3, dtype=np.float32)
    
    for i, H in enumerate(homographies):
        # 누적 변환: H_total = H * H_prev
        cumulative_H = H @ cumulative_H
        
        # 다음 이미지의 코너
        H_img, W_img = images[i+1].shape[:2]
        img_corners = np.array([[0, 0], [W_img, 0], [W_img, H_img], [0, H_img]], dtype=np.float32)
        
        # Homography 적용
        transformed_corners = apply_homography(img_corners, cumulative_H)
        all_corners.append(transformed_corners)
    
    # 모든 코너를 하나로 합치기
    all_corners = np.vstack(all_corners)  # (N*4, 2)
    
    # 최소/최대 좌표 계산
    min_x = np.min(all_corners[:, 0])
    max_x = np.max(all_corners[:, 0])
    min_y = np.min(all_corners[:, 1])
    max_y = np.max(all_corners[:, 1])
    
    # 오프셋 계산 (음수 좌표를 양수로 만들기)
    offset_x = int(-min_x) if min_x < 0 else 0
    offset_y = int(-min_y) if min_y < 0 else 0
    
    # 캔버스 크기 계산
    canvas_width = int(max_x - min_x) + 1
    canvas_height = int(max_y - min_y) + 1
    
    offset = np.array([offset_x, offset_y], dtype=np.int32)
    
    return canvas_width, canvas_height, offset


def inverse_warp(image: np.ndarray, H: np.ndarray, output_shape: Tuple[int, int], offset: np.ndarray) -> np.ndarray:
    """
    Inverse Mapping을 사용하여 이미지를 워핑합니다.
    
    결과 이미지의 각 픽셀 좌표 (x', y')에 H^-1를 곱하여 원본 이미지의 좌표 (x, y)를 찾습니다.
    Bilinear Interpolation을 사용하여 서브픽셀 값을 보간합니다.
    
    Args:
        image: 입력 이미지 (H, W, C) - uint8
        H: Homography 행렬 (3, 3) - float32
        output_shape: 출력 이미지 크기 (height, width) - Tuple[int, int]
        offset: 오프셋 [offset_x, offset_y] - np.ndarray (2,) - int32
    
    Returns:
        warped: 워핑된 이미지 (output_shape[0], output_shape[1], C) - uint8
    """
    H_inv = np.linalg.inv(H)
    
    H_out, W_out = output_shape
    offset_x, offset_y = offset[0], offset[1]
    
    # 출력 이미지의 모든 픽셀 좌표 생성 (메시그리드)
    x_out, y_out = np.meshgrid(np.arange(W_out, dtype=np.float32), 
                               np.arange(H_out, dtype=np.float32))
    
    # 오프셋 적용 (캔버스 좌표계에서 원본 좌표계로)
    x_out = x_out - offset_x
    y_out = y_out - offset_y
    
    # 동차 좌표로 변환
    ones = np.ones_like(x_out)
    coords_homogeneous = np.stack([x_out.flatten(), y_out.flatten(), ones.flatten()], axis=0)  # (3, H_out*W_out)
    
    # H^-1 적용하여 원본 좌표 계산
    source_coords_homogeneous = H_inv @ coords_homogeneous  # (3, H_out*W_out)
    
    # 동차 좌표를 일반 좌표로 변환
    w = source_coords_homogeneous[2, :]
    w = np.where(np.abs(w) < 1e-10, np.sign(w) * 1e-10, w)  # 0으로 나누기 방지
    
    x_source = source_coords_homogeneous[0, :] / w
    y_source = source_coords_homogeneous[1, :] / w
    
    # 원본 이미지 크기로 재구성
    x_source = x_source.reshape(H_out, W_out)
    y_source = y_source.reshape(H_out, W_out)
    
    # NaN/Inf 값 필터링 (유효하지 않은 좌표는 이미지 범위 밖으로 설정)
    valid_mask = np.isfinite(x_source) & np.isfinite(y_source)
    x_source = np.where(valid_mask, x_source, -1)
    y_source = np.where(valid_mask, y_source, -1)
    
    # Bilinear Interpolation으로 샘플링
    warped = bilinear_interpolation(image, x_source, y_source)
    
    # 유효하지 않은 픽셀은 0으로 설정
    if len(image.shape) == 3:
        warped = np.where(valid_mask[:, :, np.newaxis], warped, 0)
    else:
        warped = np.where(valid_mask, warped, 0)
    
    return warped


def bilinear_interpolation(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Bilinear Interpolation을 사용하여 이미지에서 값을 샘플링합니다.
    
    주변 4개 픽셀 (x0, y0), (x1, y0), (x0, y1), (x1, y1)의 가중 평균을 계산합니다.
    
    Args:
        image: 입력 이미지 (H, W, C) - uint8 또는 float32
        x: x 좌표 배열 (H_out, W_out) - float32
        y: y 좌표 배열 (H_out, W_out) - float32
    
    Returns:
        interpolated: 보간된 값 (H_out, W_out, C) - 입력과 동일한 dtype
    """
    H, W = image.shape[:2]
    H_out, W_out = x.shape
    
    # NaN/Inf 값 처리 (이미지 범위 밖으로 설정)
    x = np.where(np.isfinite(x), x, -1)
    y = np.where(np.isfinite(y), y, -1)
    
    # 정수 부분과 소수 부분 분리
    x0 = np.floor(x)
    y0 = np.floor(y)
    
    # 유효한 좌표인지 확인 (이미지 범위 내)
    valid_mask = (x0 >= 0) & (x0 < W) & (y0 >= 0) & (y0 < H)
    
    # 정수로 변환 (NaN이 아닌 경우만)
    x0 = np.where(valid_mask, x0, 0).astype(np.int32)
    y0 = np.where(valid_mask, y0, 0).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    
    # 소수 부분 (가중치)
    dx = x - x0.astype(np.float32)
    dy = y - y0.astype(np.float32)
    
    # 경계 체크 및 클리핑
    x0 = np.clip(x0, 0, W - 1)
    y0 = np.clip(y0, 0, H - 1)
    x1 = np.clip(x1, 0, W - 1)
    y1 = np.clip(y1, 0, H - 1)
    
    # 이미지가 컬러인지 그레이스케일인지 확인
    if len(image.shape) == 3:
        C = image.shape[2]
        # 4개 코너 픽셀 값 추출
        I00 = image[y0, x0, :]  # (H_out, W_out, C)
        I10 = image[y1, x0, :]  # (H_out, W_out, C)
        I01 = image[y0, x1, :]  # (H_out, W_out, C)
        I11 = image[y1, x1, :]  # (H_out, W_out, C)
    else:
        # 그레이스케일
        I00 = image[y0, x0]  # (H_out, W_out)
        I10 = image[y1, x0]  # (H_out, W_out)
        I01 = image[y0, x1]  # (H_out, W_out)
        I11 = image[y1, x1]  # (H_out, W_out)
    
    # Bilinear interpolation
    # I = (1-dx)(1-dy)*I00 + (1-dx)*dy*I10 + dx*(1-dy)*I01 + dx*dy*I11
    if len(image.shape) == 3:
        dx_expanded = dx[:, :, np.newaxis]  # (H_out, W_out, 1)
        dy_expanded = dy[:, :, np.newaxis]  # (H_out, W_out, 1)
        valid_mask_expanded = valid_mask[:, :, np.newaxis]  # (H_out, W_out, 1)
    else:
        dx_expanded = dx
        dy_expanded = dy
        valid_mask_expanded = valid_mask
    
    interpolated = (1 - dx_expanded) * (1 - dy_expanded) * I00 + \
                   (1 - dx_expanded) * dy_expanded * I10 + \
                   dx_expanded * (1 - dy_expanded) * I01 + \
                   dx_expanded * dy_expanded * I11
    
    # 원본 dtype으로 변환
    if image.dtype == np.uint8:
        interpolated = np.clip(interpolated, 0, 255).astype(np.uint8)
    else:
        interpolated = interpolated.astype(image.dtype)
    
    # 유효하지 않은 픽셀은 0으로 설정
    interpolated = np.where(valid_mask_expanded, interpolated, 0)
    
    return interpolated


def simple_blend(image1: np.ndarray, image2: np.ndarray, mask1: Optional[np.ndarray] = None, mask2: Optional[np.ndarray] = None) -> np.ndarray:
    """
    두 이미지를 개선된 블렌딩으로 합성합니다.
    
    오버랩 영역에서는 선형 블렌딩을 사용하고, 비오버랩 영역에서는 원본 이미지를 사용합니다.
    고스팅 현상을 줄이기 위해 더 정교한 블렌딩을 수행합니다.
    
    Args:
        image1: 첫 번째 이미지 (H, W, C) - uint8
        image2: 두 번째 이미지 (H, W, C) - uint8
        mask1: 첫 번째 이미지의 마스크 (H, W) - bool, Optional
        mask2: 두 번째 이미지의 마스크 (H, W) - bool, Optional
    
    Returns:
        blended: 블렌딩된 이미지 (H, W, C) - uint8
    """
    if mask1 is None:
        mask1 = create_mask(image1)
    if mask2 is None:
        mask2 = create_mask(image2)
    
    # 마스크를 float로 변환
    mask1_float = mask1.astype(np.float32)
    mask2_float = mask2.astype(np.float32)
    
    # 오버랩 영역 식별
    overlap = mask1_float * mask2_float
    
    # 오버랩 영역에서 선형 블렌딩 (거리 기반 가중치)
    # 오버랩 영역의 중심에서 멀어질수록 해당 이미지의 가중치가 높아짐
    # 간단한 버전: 균등 가중치 사용
    total_weight = mask1_float + mask2_float
    total_weight = np.where(total_weight == 0, 1.0, total_weight)  # 0으로 나누기 방지
    
    # 가중치 정규화
    w1 = mask1_float / total_weight
    w2 = mask2_float / total_weight
    
    # 오버랩 영역에서만 블렌딩, 비오버랩 영역에서는 원본 사용
    if len(image1.shape) == 3:
        w1 = w1[:, :, np.newaxis]
        w2 = w2[:, :, np.newaxis]
        overlap = overlap[:, :, np.newaxis]
    
    # 블렌딩
    blended = np.zeros_like(image1, dtype=np.float32)
    
    # image1만 있는 영역
    only1 = (mask1_float > 0) & (mask2_float == 0)
    if len(image1.shape) == 3:
        only1 = only1[:, :, np.newaxis]
    blended = np.where(only1, image1.astype(np.float32), blended)
    
    # image2만 있는 영역
    only2 = (mask1_float == 0) & (mask2_float > 0)
    if len(image1.shape) == 3:
        only2 = only2[:, :, np.newaxis]
    blended = np.where(only2, image2.astype(np.float32), blended)
    
    # 오버랩 영역 (블렌딩)
    overlap_mask = overlap > 0
    blended_overlap = (w1 * image1.astype(np.float32) + w2 * image2.astype(np.float32))
    blended = np.where(overlap_mask, blended_overlap, blended)
    
    return blended.astype(np.uint8)


def linear_blend(image1: np.ndarray, image2: np.ndarray, overlap_region: Tuple[int, int, int, int]) -> np.ndarray:
    """
    오버랩 영역에서 선형 블렌딩을 수행합니다.
    
    Args:
        image1: 첫 번째 이미지 (H, W, C) - uint8
        image2: 두 번째 이미지 (H, W, C) - uint8
        overlap_region: 오버랩 영역 (x_min, x_max, y_min, y_max) - Tuple[int, int, int, int]
    
    Returns:
        blended: 블렌딩된 이미지 (H, W, C) - uint8
    """
    pass


def create_mask(image: np.ndarray) -> np.ndarray:
    """
    이미지의 유효 영역 마스크를 생성합니다 (0이 아닌 픽셀).
    
    Args:
        image: 이미지 (H, W, C) - uint8
    
    Returns:
        mask: 마스크 (H, W) - bool, True인 경우 유효 픽셀
    """
    if len(image.shape) == 3:
        # 컬러 이미지: 모든 채널이 0이 아닌 경우만 유효
        mask = np.any(image > 0, axis=2)
    else:
        # 그레이스케일 이미지
        mask = image > 0
    
    return mask.astype(bool)


def stitch_two_images(image1: np.ndarray, image2: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    두 이미지를 Homography를 사용하여 스티칭합니다.
    
    Args:
        image1: 첫 번째 이미지 (H1, W1, C) - uint8
        image2: 두 번째 이미지 (H2, W2, C) - uint8
        H: image2를 image1 좌표계로 변환하는 Homography (3, 3) - float32
    
    Returns:
        panorama: 스티칭된 파노라마 이미지 (H_out, W_out, C) - uint8
    """
    # 캔버스 크기 계산
    H1, W1 = image1.shape[:2]
    H2, W2 = image2.shape[:2]
    
    # 첫 번째 이미지의 코너
    corners1 = np.array([[0, 0], [W1, 0], [W1, H1], [0, H1]], dtype=np.float32)
    
    # 두 번째 이미지의 코너를 H로 변환
    from geometry import apply_homography
    corners2 = np.array([[0, 0], [W2, 0], [W2, H2], [0, H2]], dtype=np.float32)
    corners2_transformed = apply_homography(corners2, H)
    
    # 모든 코너를 합쳐서 캔버스 크기 계산
    all_corners = np.vstack([corners1, corners2_transformed])
    min_x = int(np.min(all_corners[:, 0]))
    max_x = int(np.max(all_corners[:, 0]))
    min_y = int(np.min(all_corners[:, 1]))
    max_y = int(np.max(all_corners[:, 1]))
    
    # 오프셋 계산
    offset_x = -min_x if min_x < 0 else 0
    offset_y = -min_y if min_y < 0 else 0
    
    # 캔버스 크기
    canvas_width = max_x - min_x + 1
    canvas_height = max_y - min_y + 1
    
    # 첫 번째 이미지를 캔버스에 배치
    panorama = np.zeros((canvas_height, canvas_width, image1.shape[2]), dtype=np.uint8)
    panorama[offset_y:offset_y+H1, offset_x:offset_x+W1] = image1
    
    # 두 번째 이미지를 워핑하여 캔버스에 배치
    # 오프셋을 고려한 Homography
    T = np.array([[1, 0, offset_x],
                  [0, 1, offset_y],
                  [0, 0, 1]], dtype=np.float32)
    H_offset = T @ H
    
    warped2 = inverse_warp(image2, H_offset, (canvas_height, canvas_width), 
                          np.array([offset_x, offset_y], dtype=np.int32))
    
    # 블렌딩
    mask1 = create_mask(panorama)
    mask2 = create_mask(warped2)
    panorama = simple_blend(panorama, warped2, mask1, mask2)
    
    return panorama


def stitch_multiple_images(images: list, homographies: list) -> np.ndarray:
    """
    여러 이미지를 순차적으로 스티칭합니다.
    
    Args:
        images: 이미지 리스트 [img1, img2, ...]
                각 이미지는 (H, W, C) 형태 - uint8
        homographies: Homography 행렬 리스트 [H1, H2, ...]
                     각 H는 (3, 3) 형태 - float32
                     H[i]는 images[i+1]을 images[i] 좌표계로 변환
    
    Returns:
        panorama: 스티칭된 파노라마 이미지 (H_out, W_out, C) - uint8
    """
    if len(images) == 0:
        raise ValueError("이미지가 없습니다.")
    
    if len(images) == 1:
        return images[0]
    
    # 캔버스 크기 계산
    canvas_width, canvas_height, offset = compute_canvas_size(images, homographies)
    
    # 캔버스 초기화
    C = images[0].shape[2] if len(images[0].shape) == 3 else 1
    if C == 1:
        panorama = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
    else:
        panorama = np.zeros((canvas_height, canvas_width, C), dtype=np.uint8)
    
    # 첫 번째 이미지를 캔버스에 배치
    H1, W1 = images[0].shape[:2]
    offset_x, offset_y = offset[0], offset[1]
    
    if C == 1:
        panorama[offset_y:offset_y+H1, offset_x:offset_x+W1] = images[0]
    else:
        panorama[offset_y:offset_y+H1, offset_x:offset_x+W1, :] = images[0]
    
    # 누적 Homography 계산
    cumulative_H = np.eye(3, dtype=np.float32)
    
    # 나머지 이미지들을 순차적으로 스티칭
    for i, (image, H) in enumerate(zip(images[1:], homographies)):
        # 누적 변환: H_total = H * H_prev
        cumulative_H = H @ cumulative_H
        
        # 오프셋을 고려한 Homography
        T = np.array([[1, 0, offset_x],
                      [0, 1, offset_y],
                      [0, 0, 1]], dtype=np.float32)
        H_offset = T @ cumulative_H
        
        # 이미지 워핑
        warped = inverse_warp(image, H_offset, (canvas_height, canvas_width), offset)
        
        # 블렌딩
        mask1 = create_mask(panorama)
        mask2 = create_mask(warped)
        panorama = simple_blend(panorama, warped, mask1, mask2)
    
    # 검은색 배경 제거 (크롭)
    mask = create_mask(panorama)
    
    # 유효 영역의 경계 찾기
    if np.any(mask):
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        # 크롭
        panorama = panorama[y_min:y_max+1, x_min:x_max+1]
    
    return panorama

