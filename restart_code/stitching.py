"""
이미지 스티칭 구현
"""

import numpy as np
from homography import apply_homography


def bilinear_interpolate(image: np.ndarray, x: float, y: float) -> np.ndarray:
    """
    Bilinear interpolation을 사용하여 이미지에서 값을 샘플링합니다.
    경계 밖에서는 가장 가까운 경계 픽셀 값을 반환합니다 (edge clamping).
    
    Args:
        image: 이미지 (H, W) 또는 (H, W, C) - float32 또는 uint8
        x: x 좌표 (float)
        y: y 좌표 (float)
    
    Returns:
        value: 샘플링된 값 (C,) 또는 스칼라
    """
    H, W = image.shape[:2]
    
    # 경계 밖이면 가장 가까운 경계로 제한 (edge clamping)
    x = max(0.0, min(float(W - 1), x))
    y = max(0.0, min(float(H - 1), y))
    
    # 정수 좌표
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, W - 1)
    y1 = min(y0 + 1, H - 1)
    
    # 가중치
    wx = x - x0
    wy = y - y0
    
    # 가중치 클램핑 (x1 == x0 또는 y1 == y0인 경우 대비)
    if x1 == x0:
        wx = 0.0
    if y1 == y0:
        wy = 0.0
    
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


def compute_global_homographies_center_ref(homographies: list) -> list:
    """
    Center-Reference 방식을 사용하여 전역 Homography를 계산합니다.
    중앙 이미지를 기준으로 양방향으로 누적하여 오차 분산.
    
    Args:
        homographies: 인접 쌍 Homography 리스트 [H1, H2, ...]
                     H[i]는 images[i+1]을 images[i]로 변환
    
    Returns:
        global_homographies: 전역 Homography 리스트 [H0, H1, H2, ...]
                           H[i]는 images[i]를 중앙 이미지 좌표계로 변환
    """
    N = len(homographies)
    if N == 0:
        return [np.eye(3, dtype=np.float32)]
    
    # 중앙 이미지 인덱스 (0-based)
    center_idx = N // 2  # 중앙 또는 중앙 근처
    
    global_homographies = []
    
    # 각 이미지를 중앙 이미지 좌표계로 변환하는 Homography 계산
    for img_idx in range(N + 1):
        if img_idx == center_idx:
            # 중앙 이미지는 Identity
            global_homographies.append(np.eye(3, dtype=np.float32))
        elif img_idx < center_idx:
            # 중앙보다 왼쪽: 역방향 누적
            # images[img_idx] → images[img_idx+1] → ... → images[center_idx]
            # H_global = H[img_idx]^-1 @ H[img_idx+1]^-1 @ ... @ H[center_idx-1]^-1
            H_to_center = np.eye(3, dtype=np.float32)
            for i in range(img_idx, center_idx):
                H_inv = np.linalg.inv(homographies[i])
                # 정규화
                if abs(H_inv[2, 2]) > 1e-10:
                    H_inv = H_inv / H_inv[2, 2]
                H_to_center = H_to_center @ H_inv
            global_homographies.append(H_to_center)
        else:
            # 중앙보다 오른쪽: 정방향 누적
            # images[center_idx] → images[center_idx+1] → ... → images[img_idx]
            # H_global = H[center_idx] @ H[center_idx+1] @ ... @ H[img_idx-1]
            H_to_center = np.eye(3, dtype=np.float32)
            for i in range(center_idx, img_idx):
                H = homographies[i]
                H_to_center = H_to_center @ H
            global_homographies.append(H_to_center)
    
    return global_homographies


def compute_canvas_size(images: list, homographies: list, H_center_to_first: np.ndarray = None) -> tuple:
    """
    모든 이미지를 포함할 수 있는 canvas 크기를 계산합니다.
    Center-Reference 방식을 사용하여 오차 분산.
    
    Args:
        images: 이미지 리스트
        homographies: Homography 행렬 리스트
        H_center_to_first: 중앙 이미지를 첫 번째 이미지 좌표계로 변환하는 Homography (선택적, 재사용 가능)
    
    Returns:
        canvas_size: (height, width) - tuple
        offset: (y_offset, x_offset) - tuple
        adjustments: (height_adjustment, width_adjustment) - tuple
        bounds: (min_x, max_x, min_y, max_y) - tuple
    """
    # 합리적인 최대 canvas 크기 (메모리 보호용 체크)
    MAX_REASONABLE_SIZE = 100000  # 경고만 출력, 실제로는 제한하지 않음
    
    # 중앙 이미지 기준 전역 Homography 계산
    global_homographies = compute_global_homographies_center_ref(homographies)
    center_idx = len(homographies) // 2
    
    # 중앙 이미지를 첫 번째 이미지 좌표계로 변환하는 Homography
    if H_center_to_first is None:
        if center_idx > 0:
            H_center_to_first = np.eye(3, dtype=np.float32)
            for j in range(center_idx):
                H_inv = np.linalg.inv(homographies[j])
                if abs(H_inv[2, 2]) > 1e-10:
                    H_inv = H_inv / H_inv[2, 2]
                H_center_to_first = H_center_to_first @ H_inv
        else:
            H_center_to_first = np.eye(3, dtype=np.float32)
    
    # 첫 번째 이미지를 기준으로 설정
    H0, W0 = images[0].shape[:2]
    total_width = W0
    total_height = H0
    
    # 모든 이미지의 변환된 모서리 계산
    all_corners = []
    
    for i, img in enumerate(images):
        H_img, W_img = img.shape[:2]
        total_width += W_img
        total_height += H_img
        
        # 이미지의 네 모서리
        img_corners = np.array([
            [0, 0],
            [W_img - 1, 0],
            [0, H_img - 1],
            [W_img - 1, H_img - 1]
        ], dtype=np.float32)
        
        # images[i]를 첫 번째 이미지 좌표계로 변환
        if i == 0:
            # 첫 번째 이미지는 Identity (자기 자신의 좌표계)
            H_to_first = np.eye(3, dtype=np.float32)
        else:
            # global_homographies[i]: images[i] → 중앙 좌표계
            # H_center_to_first: 중앙 좌표계 → 첫 번째 이미지 좌표계
            H_to_first = H_center_to_first @ global_homographies[i]
            
            # Homography 정규화
            if abs(H_to_first[2, 2]) > 1e-10:
                H_to_first = H_to_first / H_to_first[2, 2]
            
            # 비정상적인 Homography 검증 (Image 10 같은 경우 방지)
            scale_x = np.sqrt(H_to_first[0, 0]**2 + H_to_first[0, 1]**2)
            scale_y = np.sqrt(H_to_first[1, 0]**2 + H_to_first[1, 1]**2)
            
            # 매우 비정상적인 경우 (Image 10 같은 극단적인 경우)
            if scale_x > 1000.0 or scale_y > 1000.0 or scale_x < 0.001 or scale_y < 0.001:
                print(f"  Warning: Image {i+1}의 전역 Homography가 극도로 비정상적 (scale: x={scale_x:.2f}, y={scale_y:.2f}).")
                print(f"    이전 이미지의 Homography를 추정하여 사용합니다.")
                # 이전 이미지의 Homography 기반으로 추정 (간단한 평행이동 가정)
                # 또는 Identity 사용 (더 안전)
                H_to_first = np.eye(3, dtype=np.float32)
        
        transformed_corners = apply_homography(img_corners, H_to_first)
        
        # 비정상적인 corner 값 필터링 (Image 10 같은 경우 방지)
        valid_corners_mask = ~(np.isnan(transformed_corners).any(axis=1) | np.isinf(transformed_corners).any(axis=1))
        # 극단적으로 큰 값도 필터링
        valid_corners_mask = valid_corners_mask & (np.abs(transformed_corners[:, 0]) < 1e6) & (np.abs(transformed_corners[:, 1]) < 1e6)
        
        if not np.any(valid_corners_mask):
            print(f"  Warning: Image {i+1}의 변환된 모서리가 모두 비정상적. 이 이미지를 건너뜁니다.")
            continue
        
        valid_transformed_corners = transformed_corners[valid_corners_mask]
        
        # 이상값 필터링
        max_reasonable_distance = max(total_width, total_height) * 3
        valid_corners = []
        for corner in transformed_corners:
            if abs(corner[0]) < max_reasonable_distance and abs(corner[1]) < max_reasonable_distance:
                valid_corners.append(corner)
        
        if len(valid_corners) > 0:
            all_corners.append(np.array(valid_corners, dtype=np.float32))
    
    # 모든 모서리 결합
    if len(all_corners) == 0:
        H0, W0 = images[0].shape[:2]
        print(f"  Warning: all_corners가 비어있음. Fallback 사용: ({H0}, {W0})")
        return (H0, W0), (0, 0), (0, 0), (0.0, float(W0-1), 0.0, float(H0-1))
    
    all_corners = np.vstack(all_corners)
    
    # NaN/Inf 필터링
    valid_mask = ~(np.isnan(all_corners).any(axis=1) | np.isinf(all_corners).any(axis=1))
    if not np.any(valid_mask):
        H0, W0 = images[0].shape[:2]
        print(f"  Warning: 모든 모서리가 NaN/Inf. Fallback 사용: ({H0}, {W0})")
        return (H0, W0), (0, 0), (0, 0), (0.0, float(W0-1), 0.0, float(H0-1))
    
    all_corners = all_corners[valid_mask]
    
    # Bounding box 계산 (모든 이미지의 모서리를 첫 번째 이미지 좌표계 기준으로 계산)
    # 변수명: bounds_min_x, bounds_max_x 등으로 명확하게 구분
    bounds_min_x = np.min(all_corners[:, 0])
    bounds_max_x = np.max(all_corners[:, 0])
    bounds_min_y = np.min(all_corners[:, 1])
    bounds_max_y = np.max(all_corners[:, 1])
    
    # 첫 번째 이미지는 항상 첫 번째 이미지 좌표계에서 (0, 0)에 위치
    # 다른 이미지가 첫 번째 이미지보다 위나 왼쪽에 있으면 bounds_min_x, bounds_min_y < 0일 수 있음
    # 이 경우 첫 번째 이미지를 (0, 0)에 고정하고 Canvas를 확장해야 함
    
    # Canvas 좌표계 설정:
    # - 첫 번째 이미지 좌표계의 (0, 0) = Canvas 좌표계의 (height_adjustment, width_adjustment)
    # - bounds_min_x, bounds_min_y가 음수면 Canvas를 확장
    x_offset = 0  # 첫 번째 이미지는 항상 첫 번째 이미지 좌표계에서 x=0
    y_offset = 0  # 첫 번째 이미지는 항상 첫 번째 이미지 좌표계에서 y=0
    
    if bounds_min_x < 0:
        width_adjustment = -int(np.floor(bounds_min_x))  # 음수 부분만큼 Canvas 확장
    else:
        width_adjustment = 0
    
    if bounds_min_y < 0:
        height_adjustment = -int(np.floor(bounds_min_y))  # 음수 부분만큼 Canvas 확장
    else:
        height_adjustment = 0
    
    # Canvas 크기 계산
    # bounds_max_x, bounds_max_y는 항상 >= 0 (첫 번째 이미지 크기 이상)
    width = int(np.ceil(bounds_max_x)) + 1 + width_adjustment
    height = int(np.ceil(bounds_max_y)) + 1 + height_adjustment
    
    # Canvas 크기 검증
    if width <= 0 or height <= 0:
        print(f"  Warning: 계산된 Canvas 크기가 유효하지 않음 (width={width}, height={height})")
        H0, W0 = images[0].shape[:2]
        return (H0, W0), (0, 0), (0, 0), (0.0, float(W0-1), 0.0, float(H0-1))
    
    # 최소 크기 보장 (첫 번째 이미지가 들어갈 수 있어야 함)
    H0, W0 = images[0].shape[:2]
    min_required_width = W0 + width_adjustment
    min_required_height = H0 + height_adjustment
    if width < min_required_width:
        width = min_required_width
    if height < min_required_height:
        height = min_required_height
    
    # 크기가 비정상적으로 크면 경고
    if width > MAX_REASONABLE_SIZE or height > MAX_REASONABLE_SIZE:
        print(f"  ERROR: Canvas 크기가 비정상적으로 큼 ({width}x{height}).")
        print(f"    이는 Homography 계산 오류를 의미할 수 있습니다.")
        raise ValueError(f"Canvas 크기가 너무 큼: {width}x{height}. Homography 계산을 확인해주세요.")
    
    return (height, width), (y_offset, x_offset), (height_adjustment, width_adjustment), (bounds_min_x, bounds_max_x, bounds_min_y, bounds_max_y)


def stitch_multiple_images(images: list, homographies: list) -> np.ndarray:
    """
    여러 이미지를 스티칭합니다.
    Center-Reference 방식을 사용하여 오차 분산.
    
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
    
    # 중앙 이미지 기준 전역 Homography 계산
    global_homographies = compute_global_homographies_center_ref(homographies)
    center_idx = len(homographies) // 2
    
    # 첫 번째 이미지 좌표계로 변환하기 위한 Homography 계산
    # 중앙 이미지를 첫 번째 이미지 좌표계로 변환하는 Homography
    if center_idx > 0:
        H_center_to_first = np.eye(3, dtype=np.float32)
        for j in range(center_idx):
            H_inv = np.linalg.inv(homographies[j])
            if abs(H_inv[2, 2]) > 1e-10:
                H_inv = H_inv / H_inv[2, 2]
            H_center_to_first = H_center_to_first @ H_inv
    else:
        H_center_to_first = np.eye(3, dtype=np.float32)
    
    # Canvas 크기 계산 (전역 Homography 사용, H_center_to_first 재사용)
    canvas_size, offset, adjustments, bounds = compute_canvas_size(images, homographies, H_center_to_first)
    H_canvas, W_canvas = canvas_size
    y_offset, x_offset = offset
    height_adjustment, width_adjustment = adjustments
    bounds_min_x, bounds_max_x, bounds_min_y, bounds_max_y = bounds
    
    print(f"Canvas 크기: {W_canvas}x{H_canvas} 픽셀")
    print(f"Offset: ({x_offset}, {y_offset})")
    print(f"중앙 이미지 인덱스: {center_idx + 1} (0-based: {center_idx})")
    print()
    
    # Canvas 초기화
    panorama = np.zeros((H_canvas, W_canvas, 3), dtype=np.float32)
    count = np.zeros((H_canvas, W_canvas), dtype=np.float32)
    
    # 첫 번째 이미지 추가
    # 첫 번째 이미지는 첫 번째 이미지 좌표계에서 (0, 0)에 위치
    # Canvas 좌표계로 변환: 첫 번째 이미지 좌표계의 (0, 0) = Canvas 좌표계의 (height_adjustment, width_adjustment)
    H0, W0 = images[0].shape[:2]
    
    # Canvas 좌표계에서 첫 번째 이미지 위치 (항상 adjustment 위치에 배치)
    canvas_y_start = height_adjustment
    canvas_y_end = canvas_y_start + H0
    canvas_x_start = width_adjustment
    canvas_x_end = canvas_x_start + W0
    
    # Canvas 범위 내에서만 복사
    copy_y_start = max(0, canvas_y_start)
    copy_y_end = min(H_canvas, canvas_y_end)
    copy_x_start = max(0, canvas_x_start)
    copy_x_end = min(W_canvas, canvas_x_end)
    
    # 원본 이미지에서 복사할 영역 계산
    src_y_start = copy_y_start - canvas_y_start
    src_y_end = src_y_start + (copy_y_end - copy_y_start)
    src_x_start = copy_x_start - canvas_x_start
    src_x_end = src_x_start + (copy_x_end - copy_x_start)
    
    # 유효한 범위인지 확인 후 복사
    # 첫 번째 이미지는 항상 Canvas 내에 있어야 하므로 추가 검증
    if copy_y_end > copy_y_start and copy_x_end > copy_x_start and \
       src_y_end <= H0 and src_x_end <= W0 and \
       src_y_start >= 0 and src_x_start >= 0 and \
       canvas_y_start >= 0 and canvas_x_start >= 0:
        panorama[copy_y_start:copy_y_end, copy_x_start:copy_x_end] = \
            images[0][src_y_start:src_y_end, src_x_start:src_x_end].astype(np.float32)
        count[copy_y_start:copy_y_end, copy_x_start:copy_x_end] = 1.0
    else:
        print(f"  Warning: 첫 번째 이미지 배치 실패")
        print(f"    Canvas 좌표: y=[{copy_y_start}, {copy_y_end}), x=[{copy_x_start}, {copy_x_end})")
        print(f"    Source 좌표: y=[{src_y_start}, {src_y_end}), x=[{src_x_start}, {src_x_end})")
        print(f"    Image shape: {images[0].shape}")
        print(f"    Adjustments: h={height_adjustment}, w={width_adjustment}")
        print(f"    Canvas 크기: {H_canvas}x{W_canvas}")
        print(f"    canvas_y_start={canvas_y_start}, canvas_x_start={canvas_x_start}")
    
    # 나머지 이미지들 추가
    for i in range(1, len(images)):
        # images[i]를 첫 번째 이미지 좌표계로 변환하는 Homography
        # (i >= 1이므로 첫 번째 이미지 체크 불필요)
        # global_homographies[i]: images[i] → 중앙
        # H_center_to_first: 중앙 → 첫 번째
        if center_idx > 0:
            H_to_first = H_center_to_first @ global_homographies[i]
        else:
            H_to_first = global_homographies[i]
        
        # Homography 정규화
        if abs(H_to_first[2, 2]) > 1e-10:
            H_to_first = H_to_first / H_to_first[2, 2]
        
        # 비정상적인 Homography 검증 (Image 10 같은 경우 방지)
        scale_x = np.sqrt(H_to_first[0, 0]**2 + H_to_first[0, 1]**2)
        scale_y = np.sqrt(H_to_first[1, 0]**2 + H_to_first[1, 1]**2)
        
        # 매우 비정상적인 경우 (Image 10 같은 극단적인 경우)
        if scale_x > 1000.0 or scale_y > 1000.0 or scale_x < 0.001 or scale_y < 0.001:
            print(f"  Warning: Image {i+1}의 전역 Homography가 극도로 비정상적 (scale: x={scale_x:.2f}, y={scale_y:.2f}).")
            print(f"    이 이미지를 건너뜁니다.")
            continue
        
        H_img, W_img = images[i].shape[:2]
        
        # 디버깅: 이미지 변환 위치 출력
        img_corners = np.array([[0, 0], [W_img-1, 0], [0, H_img-1], [W_img-1, H_img-1]], dtype=np.float32)
        transformed_corners = apply_homography(img_corners, H_to_first)
        
        # 비정상적인 corner 값 사전 검증
        if np.any(np.isnan(transformed_corners)) or np.any(np.isinf(transformed_corners)):
            print(f"  Warning: Image {i+1}의 변환된 모서리에 NaN/Inf가 있습니다. 이 이미지를 건너뜁니다.")
            continue
        
        if np.any(np.abs(transformed_corners) > 1e6):
            print(f"  Warning: Image {i+1}의 변환된 모서리가 극도로 비정상적입니다. 이 이미지를 건너뜁니다.")
            continue
        
        print(f"  Image {i+1} 변환 위치 (첫 번째 이미지 좌표계):")
        print(f"    좌상단: ({transformed_corners[0][0]:.1f}, {transformed_corners[0][1]:.1f})")
        print(f"    우하단: ({transformed_corners[3][0]:.1f}, {transformed_corners[3][1]:.1f})")
        
        # 역변환: 첫 번째 이미지 좌표계 → images[i] 좌표계
        H_inv = np.linalg.inv(H_to_first)
        if abs(H_inv[2, 2]) > 1e-10:
            H_inv = H_inv / H_inv[2, 2]
        
        # 최적화: 이미지가 실제로 차지하는 영역만 처리
        # transformed_corners는 첫 번째 이미지 좌표계 기준
        # Canvas 좌표계로 변환하려면 adjustment를 더해야 함
        # 첫 번째 이미지 좌표계의 (0, 0)은 Canvas 좌표계의 (height_adjustment, width_adjustment)에 해당
        
        # 비정상적인 corner 값 필터링 (Image 10 같은 경우 방지)
        valid_corners_mask = ~(np.isnan(transformed_corners).any(axis=1) | np.isinf(transformed_corners).any(axis=1))
        # 극단적으로 큰 값도 필터링
        valid_corners_mask = valid_corners_mask & (np.abs(transformed_corners[:, 0]) < 1e6) & (np.abs(transformed_corners[:, 1]) < 1e6)
        
        if not np.any(valid_corners_mask):
            print(f"  Warning: Image {i+1}의 변환된 모서리가 모두 비정상적. 이 이미지를 건너뜁니다.")
            continue
        
        valid_corners = transformed_corners[valid_corners_mask]
        min_x_first_coord = np.min(valid_corners[:, 0])
        max_x_first_coord = np.max(valid_corners[:, 0])
        min_y_first_coord = np.min(valid_corners[:, 1])
        max_y_first_coord = np.max(valid_corners[:, 1])
        
        # Canvas 좌표계로 변환 (첫 번째 이미지 좌표계 → Canvas 좌표계)
        min_x_canvas = min_x_first_coord + width_adjustment
        max_x_canvas = max_x_first_coord + width_adjustment
        min_y_canvas = min_y_first_coord + height_adjustment
        max_y_canvas = max_y_first_coord + height_adjustment
        
        # Canvas 범위 내로 클램핑
        y_start = max(0, int(np.floor(min_y_canvas)) - 5)  # 여유 5픽셀
        y_end = min(H_canvas, int(np.ceil(max_y_canvas)) + 5)
        x_start = max(0, int(np.floor(min_x_canvas)) - 5)
        x_end = min(W_canvas, int(np.ceil(max_x_canvas)) + 5)
        
        print(f"    처리 영역: y=[{y_start}, {y_end}), x=[{x_start}, {x_end})")
        
        # 진행 상황 표시를 위한 계산
        total_pixels = (y_end - y_start) * (x_end - x_start)
        total_rows = y_end - y_start
        
        # 처리 영역만 루프
        # 진행률 출력 빈도: 총 행의 10% 또는 최소 10줄마다
        progress_update_interval = max(10, total_rows // 10) if total_rows > 0 else 10
        
        for y_idx, y_out in enumerate(range(y_start, y_end)):
            if y_idx % progress_update_interval == 0:
                progress = (y_idx / total_rows * 100) if total_rows > 0 else 0
                print(f"      진행 중: {progress:.1f}%", end='\r')
            
            for x_out in range(x_start, x_end):
                # Canvas 좌표를 첫 번째 이미지 좌표계로 변환
                # Canvas 좌표계: (0, 0)이 Canvas의 좌상단
                # 첫 번째 이미지 좌표계: (0, 0)이 첫 번째 이미지의 좌상단
                # Canvas의 (height_adjustment, width_adjustment) = 첫 번째 이미지 좌표계의 (0, 0)
                x_canvas = x_out - width_adjustment
                y_canvas = y_out - height_adjustment
                
                # 첫 번째 이미지 좌표계 → images[i] 좌표계로 변환 (역변환)
                point_canvas = np.array([x_canvas, y_canvas, 1.0])
                point_img_homogeneous = H_inv @ point_canvas
                
                w = point_img_homogeneous[2]
                if abs(w) < 1e-10:
                    continue
                
                x_img = point_img_homogeneous[0] / w
                y_img = point_img_homogeneous[1] / w
                
                # Bilinear interpolation (경계 처리 포함)
                value = bilinear_interpolate(images[i], x_img, y_img)
                
                # 이미지 경계 내부에 있는 경우에만 추가
                margin = 1.0
                if x_img >= -margin and x_img < W_img + margin and \
                   y_img >= -margin and y_img < H_img + margin:
                    panorama[y_out, x_out] += value
                    count[y_out, x_out] += 1.0
        
        print(f"      완료: Image {i+1} 처리 완료" + " " * 30)  # 공백으로 이전 출력 지움
    
    # 평균 계산 (blending)
    mask = count > 0
    panorama[mask] = panorama[mask] / count[mask, np.newaxis]
    
    return panorama.astype(np.uint8)

