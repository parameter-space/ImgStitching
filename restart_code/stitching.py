"""
이미지 스티칭 구현
"""

import numpy as np
from homography import apply_homography


def bilinear_interpolate(image: np.ndarray, x: float, y: float) -> np.ndarray:
    """
    Bilinear interpolation을 사용하여 이미지에서 값을 샘플링합니다.
    경계 밖이면 0을 반환합니다 (edge clamping 제거로 Streaking 방지).
    
    Args:
        image: 이미지 (H, W) 또는 (H, W, C) - float32 또는 uint8
        x: x 좌표 (float)
        y: y 좌표 (float)
    
    Returns:
        value: 샘플링된 값 (C,) 또는 스칼라, 경계 밖이면 0
    """
    H, W = image.shape[:2]
    
    # 경계 밖이면 0 반환 (Streaking 방지)
    if x < 0.0 or x >= float(W - 1) or y < 0.0 or y >= float(H - 1):
        if len(image.shape) == 3:
            return np.zeros(image.shape[2], dtype=image.dtype)
        else:
            return 0.0
    
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
    
    print(f"Global Homography 계산: 중앙 이미지 인덱스 = {center_idx} (이미지 {center_idx+1}번)")
    
    global_homographies = []
    
    # 각 이미지를 중앙 이미지 좌표계로 변환하는 Homography 계산
    for img_idx in range(N + 1):
        if img_idx == center_idx:
            # 중앙 이미지는 Identity
            global_homographies.append(np.eye(3, dtype=np.float32))
            print(f"  Image {img_idx+1}: Identity (중앙 이미지)")
        elif img_idx < center_idx:
            # 중앙보다 왼쪽: 역방향 누적
            # images[img_idx] → images[img_idx+1] → ... → images[center_idx]
            # H_global = H[img_idx]^-1 @ H[img_idx+1]^-1 @ ... @ H[center_idx-1]^-1
            H_to_center = np.eye(3, dtype=np.float32)
            accumulated_indices = []
            for i in range(img_idx, center_idx):
                # Identity Homography 체크
                if np.allclose(homographies[i], np.eye(3, dtype=np.float32), atol=1e-6):
                    # Identity는 누적에서 스킵 (역행렬도 Identity이므로)
                    continue
                accumulated_indices.append(i)
                H_inv = np.linalg.inv(homographies[i])
                # 정규화
                if abs(H_inv[2, 2]) > 1e-10:
                    H_inv = H_inv / H_inv[2, 2]
                H_to_center = H_to_center @ H_inv
            
            # 검증: H_to_center가 올바른지 확인
            scale_x = np.sqrt(H_to_center[0, 0]**2 + H_to_center[0, 1]**2)
            scale_y = np.sqrt(H_to_center[1, 0]**2 + H_to_center[1, 1]**2)
            print(f"  Image {img_idx+1}: 왼쪽 누적 (indices: {accumulated_indices}), scale: ({scale_x:.2f}, {scale_y:.2f})")
            
            global_homographies.append(H_to_center)
        else:
            # 중앙보다 오른쪽: 순방향 누적
            # images[img_idx] → images[img_idx-1] → ... → images[center_idx]
            # H[i]는 images[i+1] → images[i]이므로,
            # images[img_idx] → images[img_idx-1]은 H[img_idx-1]
            # images[img_idx-1] → images[img_idx-2]는 H[img_idx-2]
            # ...
            # images[center_idx+1] → images[center_idx]는 H[center_idx]
            # H_global = H[img_idx-1] @ H[img_idx-2] @ ... @ H[center_idx]
            # 역순으로 누적해야 하므로 range를 역순으로
            H_to_center = np.eye(3, dtype=np.float32)
            accumulated_indices = []
            for i in range(img_idx - 1, center_idx - 1, -1):  # img_idx-1부터 center_idx까지 역순
                # Identity Homography 체크
                if np.allclose(homographies[i], np.eye(3, dtype=np.float32), atol=1e-6):
                    # Identity는 누적에서 스킵
                    continue
                accumulated_indices.append(i)
                H = homographies[i]
                # 정규화
                if abs(H[2, 2]) > 1e-10:
                    H = H / H[2, 2]
                H_to_center = H_to_center @ H
            
            # 검증: H_to_center가 올바른지 확인
            scale_x = np.sqrt(H_to_center[0, 0]**2 + H_to_center[0, 1]**2)
            scale_y = np.sqrt(H_to_center[1, 0]**2 + H_to_center[1, 1]**2)
            print(f"  Image {img_idx+1}: 오른쪽 누적 (indices: {accumulated_indices}), scale: ({scale_x:.2f}, {scale_y:.2f})")
            
            global_homographies.append(H_to_center)
    
    print()
    return global_homographies


def compute_canvas_size(images: list, homographies: list, H_center_to_first: np.ndarray = None) -> tuple:
    """
    모든 이미지를 포함할 수 있는 canvas 크기를 계산합니다.
    Center-Reference 방식을 사용하여 오차 분산.
    중앙 이미지 좌표계를 기준으로 Canvas 크기를 계산합니다.
    
    Args:
        images: 이미지 리스트
        homographies: Homography 행렬 리스트
        H_center_to_first: 사용하지 않음 (하위 호환성을 위해 유지, 무시됨)
    
    Returns:
        canvas_size: (height, width) - tuple
        offset: (y_offset, x_offset) - tuple (항상 (0, 0))
        adjustments: (height_adjustment, width_adjustment) - tuple
                     중앙 이미지 좌표계의 (0, 0)이 Canvas 좌표계의 (height_adjustment, width_adjustment)에 위치
        bounds: (min_x, max_x, min_y, max_y) - tuple (중앙 이미지 좌표계 기준)
    """
    # 합리적인 최대 canvas 크기 (메모리 보호용 체크)
    # 10만 픽셀 = 약 111GB RAM 필요 (float32, 3채널)
    # 현실적인 한계로 25000으로 설정 (약 7GB RAM)
    MAX_REASONABLE_SIZE = 25000
    
    # 중앙 이미지 기준 전역 Homography 계산
    global_homographies = compute_global_homographies_center_ref(homographies)
    center_idx = len(homographies) // 2
    
    # 중앙 이미지 크기 (Canvas 크기 계산용)
    H_center, W_center = images[center_idx].shape[:2]
    
    # max_reasonable_distance 계산용: 중앙 이미지 크기를 기준으로 사용
    base_width = W_center
    base_height = H_center
    
    # 모든 이미지의 변환된 모서리 계산 (중앙 이미지 좌표계 기준)
    all_corners = []
    
    for i, img in enumerate(images):
        H_img, W_img = img.shape[:2]
        
        # 이미지의 네 모서리
        img_corners = np.array([
            [0, 0],
            [W_img - 1, 0],
            [0, H_img - 1],
            [W_img - 1, H_img - 1]
        ], dtype=np.float32)
        
        # images[i]를 중앙 이미지 좌표계로 변환
        # global_homographies[i]: images[i] → 중앙 이미지 좌표계
        H_to_center = global_homographies[i]
        
        # Homography 정규화
        if abs(H_to_center[2, 2]) > 1e-10:
            H_to_center = H_to_center / H_to_center[2, 2]
        
        # 비정상적인 Homography 검증 (극단적인 경우만 제외)
        scale_x = np.sqrt(H_to_center[0, 0]**2 + H_to_center[0, 1]**2)
        scale_y = np.sqrt(H_to_center[1, 0]**2 + H_to_center[1, 1]**2)
        
        # 매우 비정상적인 경우만 제외 (범위 완화: 파노라마는 확대/축소될 수 있음)
        # 100배 이상 확대/축소는 비정상적
        if scale_x > 100.0 or scale_y > 100.0 or scale_x < 0.001 or scale_y < 0.001:
            print(f"  Warning: Image {i+1}의 전역 Homography가 극도로 비정상적 (scale: x={scale_x:.2f}, y={scale_y:.2f}).")
            print(f"    이 이미지를 Canvas 계산에서 제외합니다.")
            continue
        
        # Identity Homography 체크 (중앙 이미지가 아닌 경우만, Canvas 크기 계산에서 제외)
        if i != center_idx and np.allclose(H_to_center, np.eye(3, dtype=np.float32), atol=1e-6):
            print(f"  Warning: Image {i+1}의 전역 Homography가 Identity입니다.")
            print(f"    Canvas 계산에서 제외합니다.")
            continue
        
        transformed_corners = apply_homography(img_corners, H_to_center)
        
        # 중앙 이미지는 항상 포함 (Identity이므로 원본 corner와 동일)
        if i == center_idx:
            # 중앙 이미지는 항상 추가 (검증 불필요)
            all_corners.append(transformed_corners)
        else:
            # 비정상적인 corner 값 필터링 (Image 10 같은 경우 방지)
            valid_corners_mask = ~(np.isnan(transformed_corners).any(axis=1) | np.isinf(transformed_corners).any(axis=1))
            # 극단적으로 큰 값도 필터링
            valid_corners_mask = valid_corners_mask & (np.abs(transformed_corners[:, 0]) < 1e6) & (np.abs(transformed_corners[:, 1]) < 1e6)
            
            if not np.any(valid_corners_mask):
                print(f"  Warning: Image {i+1}의 변환된 모서리가 모두 비정상적. 이 이미지를 건너뜁니다.")
                continue
            
            valid_transformed_corners = transformed_corners[valid_corners_mask]
            
            # 뒤집힌 이미지 체크 (Canvas 크기 계산 전에 제외)
            min_x = np.min(valid_transformed_corners[:, 0])
            max_x = np.max(valid_transformed_corners[:, 0])
            min_y = np.min(valid_transformed_corners[:, 1])
            max_y = np.max(valid_transformed_corners[:, 1])
            
            if min_x > max_x or min_y > max_y:
                print(f"  Warning: Image {i+1}의 변환된 모서리가 뒤바뀐 것 같습니다 (이미지가 뒤집힘). Canvas 계산에서 제외합니다.")
                continue
            
            # Bounding box 크기 체크 (Canvas 크기 계산 시)
            bbox_width = max_x - min_x
            bbox_height = max_y - min_y
            # Corner distance 검증이 이미 있으므로, bounding box 검증은 매우 관대하게 설정
            # 단순히 비정상적으로 큰 경우만 제외 (예: 이미지 크기의 50배 이상)
            if bbox_width > W_img * 50 or bbox_height > H_img * 50:
                print(f"  Warning: Image {i+1}의 변환된 corner bounding box가 극도로 큼 (w: {bbox_width:.1f}, h: {bbox_height:.1f}). Canvas 계산에서 제외합니다.")
                continue
            
            # 추가 이상값 필터링: 더 관대한 기준 사용
            # 파노라마는 이미지들이 옆으로 확장될 수 있으므로, 더 넓은 범위 허용
            # 중앙 이미지 크기를 기준으로 사용 (모든 이미지 크기 누적은 실제 크기보다 훨씬 큼)
            max_reasonable_distance = max(base_width, base_height) * 5  # 3 → 5로 완화
            # 벡터화된 필터링
            distance_mask = (np.abs(valid_transformed_corners[:, 0]) < max_reasonable_distance) & \
                           (np.abs(valid_transformed_corners[:, 1]) < max_reasonable_distance)
            
            if np.any(distance_mask):
                final_valid_corners = valid_transformed_corners[distance_mask]
                all_corners.append(final_valid_corners)
            else:
                print(f"  Warning: Image {i+1}의 변환된 모서리가 distance_mask에서 제외됨. Canvas 계산에서 제외합니다.")
    
    # 모든 모서리 결합
    if len(all_corners) == 0:
        print(f"  Warning: all_corners가 비어있음. Fallback 사용: ({H_center}, {W_center})")
        return (H_center, W_center), (0, 0), (0, 0), (0.0, float(W_center-1), 0.0, float(H_center-1))
    
    all_corners = np.vstack(all_corners)
    
    # NaN/Inf 필터링
    valid_mask = ~(np.isnan(all_corners).any(axis=1) | np.isinf(all_corners).any(axis=1))
    if not np.any(valid_mask):
        print(f"  Warning: 모든 모서리가 NaN/Inf. Fallback 사용: ({H_center}, {W_center})")
        return (H_center, W_center), (0, 0), (0, 0), (0.0, float(W_center-1), 0.0, float(H_center-1))
    
    all_corners = all_corners[valid_mask]
    
    # Bounding box 계산 (모든 이미지의 모서리를 중앙 이미지 좌표계 기준으로 계산)
    # 변수명: bounds_min_x, bounds_max_x 등으로 명확하게 구분
    bounds_min_x = np.min(all_corners[:, 0])
    bounds_max_x = np.max(all_corners[:, 0])
    bounds_min_y = np.min(all_corners[:, 1])
    bounds_max_y = np.max(all_corners[:, 1])
    
    # 중앙 이미지는 항상 중앙 이미지 좌표계에서 (0, 0)에 위치
    # 다른 이미지가 중앙 이미지보다 위나 왼쪽에 있으면 bounds_min_x, bounds_min_y < 0일 수 있음
    # 이 경우 중앙 이미지를 Canvas 중앙에 배치하고 Canvas를 확장해야 함
    
    # Canvas 좌표계 설정:
    # - 중앙 이미지 좌표계의 (0, 0) = Canvas 좌표계의 (center_y_pos, center_x_pos)
    # - 중앙 이미지를 Canvas 중앙에 배치하기 위해 조정
    # - bounds_min_x, bounds_min_y가 음수면 Canvas를 확장
    x_offset = 0  # 중앙 이미지는 항상 중앙 이미지 좌표계에서 x=0
    y_offset = 0  # 중앙 이미지는 항상 중앙 이미지 좌표계에서 y=0
    
    # Padding 추가: 각 방향에 여유 공간 추가 (Ghost 현상 방지 및 안전한 배치)
    # 실제 이미지 범위를 기반으로 적응적 padding 적용
    actual_range_width = bounds_max_x - bounds_min_x
    actual_range_height = bounds_max_y - bounds_min_y
    # 작은 범위는 작은 padding, 큰 범위는 큰 padding (최소 20, 최대 100)
    padding = max(20, min(100, int(max(actual_range_width, actual_range_height) / 100)))
    
    # Canvas 크기 계산 (중앙 이미지 좌표계 기준)
    # 실제 필요한 크기: (bounds_max_x - bounds_min_x) + 1
    actual_width = int(np.ceil(bounds_max_x - bounds_min_x)) + 1
    actual_height = int(np.ceil(bounds_max_y - bounds_min_y)) + 1
    
    # Canvas 크기 = 실제 필요한 크기 + padding * 2 (양쪽 끝)
    width = actual_width + padding * 2
    height = actual_height + padding * 2
    
    # 중앙 이미지를 Canvas 중앙에 배치하기 위한 adjustment 계산
    # 중앙 이미지 좌표계의 (0, 0) = Canvas 좌표계의 (height_adjustment, width_adjustment)
    # 중앙 이미지 중심이 Canvas 중심에 오도록 배치:
    # - 중앙 이미지 중심: (W_center/2, H_center/2) (중앙 이미지 좌표계)
    # - Canvas 중심: (width/2, height/2) (Canvas 좌표계)
    # - 중앙 이미지 중심이 Canvas 중심에 오려면:
    #   width_adjustment = width/2 - W_center/2 = (width - W_center) / 2
    #   height_adjustment = height/2 - H_center/2 = (height - H_center) / 2
    
    # 하지만 bounds를 모두 포함하면서 중앙 이미지 중심을 Canvas 중심에 배치하려면:
    # - bounds_min_x < 0인 경우: 왼쪽으로 확장 필요
    # - 중앙 이미지 좌표계의 (bounds_min_x, ...)가 Canvas의 (padding, ...)에 오려면:
    #   width_adjustment = padding - bounds_min_x
    # - 중앙 이미지 중심 기준: width_adjustment = (width - W_center) / 2
    # - 두 조건을 모두 만족하려면 더 큰 값을 선택
    
    # 중앙 이미지 중심을 Canvas 중심에 최대한 가깝게 배치하면서 bounds를 모두 포함
    ideal_center_x = (width - W_center) / 2
    ideal_center_y = (height - H_center) / 2
    
    # bounds를 고려한 최소값
    # 중앙 이미지 좌표계의 (bounds_min_x)가 Canvas 좌표계의 (padding)에 오려면:
    # padding = bounds_min_x + width_adjustment
    # 따라서 width_adjustment = padding - bounds_min_x (bounds_min_x의 부호와 무관)
    min_width_adjustment = padding - bounds_min_x
    min_height_adjustment = padding - bounds_min_y
    
    # 디버그 출력
    print(f"  ideal_center: x={ideal_center_x:.1f}, y={ideal_center_y:.1f}")
    print(f"  min_adjustment: x={min_width_adjustment:.1f}, y={min_height_adjustment:.1f}")
    print(f"  bounds: x=[{bounds_min_x:.1f}, {bounds_max_x:.1f}], y=[{bounds_min_y:.1f}, {bounds_max_y:.1f}]")
    
    # 두 조건 중 더 큰 값 선택 (모든 이미지를 포함하면서 중앙 이미지를 가능한 한 중앙에)
    width_adjustment = max(int(ideal_center_x), int(min_width_adjustment))
    height_adjustment = max(int(ideal_center_y), int(min_height_adjustment))
    
    print(f"  최종 adjustment: x={width_adjustment}, y={height_adjustment}")
    
    # Canvas 크기 검증
    if width <= 0 or height <= 0:
        print(f"  Warning: 계산된 Canvas 크기가 유효하지 않음 (width={width}, height={height})")
        return (H_center, W_center), (0, 0), (0, 0), (0.0, float(W_center-1), 0.0, float(H_center-1))
    
    # 최소 크기 보장 (중앙 이미지가 들어갈 수 있어야 함)
    min_required_width = W_center + width_adjustment
    min_required_height = H_center + height_adjustment
    if width < min_required_width:
        width = min_required_width
    if height < min_required_height:
        height = min_required_height
    
    # 크기가 비정상적으로 크면 제한 및 경고
    if width > MAX_REASONABLE_SIZE or height > MAX_REASONABLE_SIZE:
        print(f"  ERROR: Canvas 크기가 비정상적으로 큼 ({width}x{height}).")
        print(f"    이는 Homography 계산 오류를 의미할 수 있습니다.")
        print(f"    최대 크기로 제한합니다: {MAX_REASONABLE_SIZE}")
        # 메모리 보호를 위해 최대 크기로 제한
        width = min(width, MAX_REASONABLE_SIZE)
        height = min(height, MAX_REASONABLE_SIZE)
        # bounds도 조정
        bounds_max_x = min(bounds_max_x, float(width - width_adjustment - 1))
        bounds_max_y = min(bounds_max_y, float(height - height_adjustment - 1))
    
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
    
    # Canvas 크기 계산 (중앙 이미지 좌표계 기준)
    canvas_size, offset, adjustments, bounds = compute_canvas_size(images, homographies, None)
    H_canvas, W_canvas = canvas_size
    y_offset, x_offset = offset
    height_adjustment, width_adjustment = adjustments
    bounds_min_x, bounds_max_x, bounds_min_y, bounds_max_y = bounds
    
    print(f"Canvas 크기: {W_canvas}x{H_canvas} 픽셀")
    print(f"  실제 이미지 범위 (중앙 이미지 좌표계): x=[{bounds_min_x:.1f}, {bounds_max_x:.1f}], y=[{bounds_min_y:.1f}, {bounds_max_y:.1f}]")
    print(f"  실제 필요한 크기: {int(bounds_max_x - bounds_min_x) + 1}x{int(bounds_max_y - bounds_min_y) + 1}")
    print(f"  중앙 이미지 Canvas 위치: ({width_adjustment}, {height_adjustment})")
    # padding 값을 다시 읽어서 출력 (적응적 padding 적용됨)
    actual_range_width = bounds_max_x - bounds_min_x
    actual_range_height = bounds_max_y - bounds_min_y
    adaptive_padding = max(20, min(100, int(max(actual_range_width, actual_range_height) / 100)))
    print(f"  Padding: {adaptive_padding}픽셀 (각 방향, 적응적)")
    print(f"Offset: ({x_offset}, {y_offset})")
    print(f"중앙 이미지 인덱스: {center_idx + 1} (0-based: {center_idx})")
    print()
    
    # Canvas 초기화 (메모리 할당 실패 감지)
    try:
        panorama = np.zeros((H_canvas, W_canvas, 3), dtype=np.float32)
        count = np.zeros((H_canvas, W_canvas), dtype=np.float32)
        # Weighted blending을 위한 가중치 합 배열
        weight_sum = np.zeros((H_canvas, W_canvas), dtype=np.float32)
    except MemoryError:
        print(f"  ERROR: 메모리 부족 - Canvas 크기가 너무 큼 ({W_canvas}x{H_canvas})")
        print(f"    필요한 메모리: 약 {W_canvas * H_canvas * 3 * 4 / (1024**3):.2f} GB")
        raise ValueError(f"Canvas 크기가 너무 커서 메모리 할당 실패: {W_canvas}x{H_canvas}")
    
    # 중앙 이미지 추가
    # 중앙 이미지는 중앙 이미지 좌표계에서 (0, 0)에 위치
    # Canvas 좌표계로 변환: 중앙 이미지 좌표계의 (0, 0) = Canvas 좌표계의 (height_adjustment, width_adjustment)
    H_center, W_center = images[center_idx].shape[:2]
    
    # Canvas 좌표계에서 중앙 이미지 위치 (adjustment 위치에 배치)
    canvas_y_start = height_adjustment
    canvas_y_end = canvas_y_start + H_center
    canvas_x_start = width_adjustment
    canvas_x_end = canvas_x_start + W_center
    
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
    # 중앙 이미지는 항상 Canvas 내에 있어야 하므로 추가 검증
    if copy_y_end > copy_y_start and copy_x_end > copy_x_start and \
       src_y_end <= H_center and src_x_end <= W_center and \
       src_y_start >= 0 and src_x_start >= 0 and \
       canvas_y_start >= 0 and canvas_x_start >= 0:
        # 중앙 이미지 복사 (가중치 1.0으로 처리)
        panorama[copy_y_start:copy_y_end, copy_x_start:copy_x_end] = \
            images[center_idx][src_y_start:src_y_end, src_x_start:src_x_end].astype(np.float32)
        count[copy_y_start:copy_y_end, copy_x_start:copy_x_end] = 1.0
        # Weighted blending을 위한 가중치 합 (이미지 중심에서의 거리 기반)
        # 간단하게 중앙 이미지는 가중치 1.0
        weight_sum[copy_y_start:copy_y_end, copy_x_start:copy_x_end] = 1.0
        print(f"  중앙 이미지 (Image {center_idx+1}) 배치 완료: Canvas 좌표 y=[{copy_y_start}, {copy_y_end}), x=[{copy_x_start}, {copy_x_end})")
    else:
        print(f"  ERROR: 중앙 이미지 배치 실패")
        print(f"    Canvas 좌표: y=[{copy_y_start}, {copy_y_end}), x=[{copy_x_start}, {copy_x_end})")
        print(f"    Source 좌표: y=[{src_y_start}, {src_y_end}), x=[{src_x_start}, {src_x_end})")
        print(f"    Image shape: {images[center_idx].shape}")
        print(f"    Adjustments: h={height_adjustment}, w={width_adjustment}")
        print(f"    Canvas 크기: {H_canvas}x{W_canvas}")
        print(f"    canvas_y_start={canvas_y_start}, canvas_x_start={canvas_x_start}")
        return None
    
    # 나머지 이미지들 추가 (중앙 이미지 제외)
    for i in range(len(images)):
        if i == center_idx:
            continue  # 중앙 이미지는 이미 추가했으므로 건너뜀
        # images[i]를 중앙 이미지 좌표계로 변환하는 Homography
        # global_homographies[i]: images[i] → 중앙 이미지 좌표계
        H_to_center = global_homographies[i]
        
        # Homography 정규화
        if abs(H_to_center[2, 2]) > 1e-10:
            H_to_center = H_to_center / H_to_center[2, 2]
        
        # Identity Homography 체크 (같은 위치에 stacking 방지)
        if np.allclose(H_to_center, np.eye(3, dtype=np.float32), atol=1e-6):
            print(f"  Warning: Image {i+1}의 전역 Homography가 Identity입니다.")
            print(f"    중앙 이미지와 같은 위치에 배치되어 stacking이 발생할 수 있습니다.")
            print(f"    이 이미지를 건너뜁니다.")
            continue
        
        # 비정상적인 Homography 검증 (극단적인 경우만 제외)
        scale_x = np.sqrt(H_to_center[0, 0]**2 + H_to_center[0, 1]**2)
        scale_y = np.sqrt(H_to_center[1, 0]**2 + H_to_center[1, 1]**2)
        
        # 매우 비정상적인 경우만 제외 (범위 완화: 파노라마는 확대/축소될 수 있음)
        # 100배 이상 확대/축소는 비정상적
        if scale_x > 100.0 or scale_y > 100.0 or scale_x < 0.001 or scale_y < 0.001:  # 50 → 100, 0.01 → 0.001로 완화
            print(f"  Warning: Image {i+1}의 전역 Homography가 극도로 비정상적 (scale: x={scale_x:.2f}, y={scale_y:.2f}).")
            print(f"    이 이미지를 건너뜁니다.")
            continue
        
        H_img, W_img = images[i].shape[:2]
        
        # 디버깅: 이미지 변환 위치 계산 및 검증
        img_corners = np.array([[0, 0], [W_img-1, 0], [0, H_img-1], [W_img-1, H_img-1]], dtype=np.float32)
        transformed_corners = apply_homography(img_corners, H_to_center)
        
        # 비정상적인 corner 값 검증 (통합 검증)
        # 1. NaN/Inf 체크
        if np.any(np.isnan(transformed_corners)) or np.any(np.isinf(transformed_corners)):
            print(f"  Warning: Image {i+1}의 변환된 모서리에 NaN/Inf가 있습니다. 이 이미지를 건너뜁니다.")
            continue
        
        # 2. 극단적으로 큰 값 체크
        if np.any(np.abs(transformed_corners) > 1e6):
            print(f"  Warning: Image {i+1}의 변환된 모서리가 극도로 비정상적입니다. 이 이미지를 건너뜁니다.")
            continue
        
        # 3. 변환된 corner의 범위가 원본 이미지 크기의 배수를 넘는지 확인
        # 파노라마는 옆으로 확장될 수 있으므로 더 관대한 기준 사용
        max_reasonable_distance = max(W_img, H_img) * 10  # 5 → 10으로 완화
        corner_distances = np.sqrt(transformed_corners[:, 0]**2 + transformed_corners[:, 1]**2)
        if np.any(corner_distances > max_reasonable_distance):
            max_dist = np.max(corner_distances)
            print(f"  Warning: Image {i+1}의 변환된 모서리가 너무 멀리 떨어져 있음 (최대 거리: {max_dist:.1f} > {max_reasonable_distance:.1f}).")
            print(f"    이 이미지를 건너뜁니다.")
            continue
        
        # 4. 좌상단과 우하단이 뒤바뀐 경우 체크 (이미지가 뒤집힘) - 먼저 체크
        min_x = np.min(transformed_corners[:, 0])
        max_x = np.max(transformed_corners[:, 0])
        min_y = np.min(transformed_corners[:, 1])
        max_y = np.max(transformed_corners[:, 1])
        
        # 뒤집힌 이미지는 bounding box 크기 계산 전에 제외
        if min_x > max_x or min_y > max_y:
            print(f"  Warning: Image {i+1}의 변환된 모서리가 뒤바뀐 것 같습니다 (이미지가 뒤집힘).")
            print(f"    좌상단: ({min_x:.1f}, {min_y:.1f}), 우하단: ({max_x:.1f}, {max_y:.1f})")
            print(f"    이 이미지를 건너뜁니다.")
            continue
        
        # 5. Bounding box 크기 체크 (뒤집힘 체크 통과 후)
        bbox_width = max_x - min_x
        bbox_height = max_y - min_y
        
        # 파노라마는 이미지들이 옆으로 확장될 수 있으므로 매우 관대한 기준 사용
        # Corner distance 검증이 이미 있으므로, bounding box 검증은 매우 관대하게 설정
        # 단순히 비정상적으로 큰 경우만 제외 (예: 이미지 크기의 50배 이상)
        # 일반적으로는 corner distance 검증으로 충분함
        if bbox_width > W_img * 50 or bbox_height > H_img * 50:
            print(f"  Warning: Image {i+1}의 변환된 corner bounding box가 극도로 큼 (w: {bbox_width:.1f}, h: {bbox_height:.1f}).")
            print(f"    이는 Homography 계산 오류를 의미할 수 있습니다.")
            print(f"    이 이미지를 건너뜁니다.")
            continue
        
        # 모든 검증 통과
        print(f"  Image {i+1} 변환 위치 (중앙 이미지 좌표계):")
        print(f"    좌상단: ({transformed_corners[0][0]:.1f}, {transformed_corners[0][1]:.1f})")
        print(f"    우하단: ({transformed_corners[3][0]:.1f}, {transformed_corners[3][1]:.1f})")
        print(f"    범위: x=[{min_x:.1f}, {max_x:.1f}], y=[{min_y:.1f}, {max_y:.1f}]")
        
        # 역변환: 중앙 이미지 좌표계 → images[i] 좌표계
        # 검증: H_to_center의 determinant 확인
        det_H = np.linalg.det(H_to_center)
        if abs(det_H) < 1e-5:
            print(f"    Warning: Image {i+1}의 H_to_center determinant가 너무 작음 ({det_H:.2e}). 건너뜁니다.")
            continue
        
        H_inv = np.linalg.inv(H_to_center)
        if abs(H_inv[2, 2]) > 1e-10:
            H_inv = H_inv / H_inv[2, 2]
        
        # 역변환 검증: H_inv @ H_to_center가 Identity에 가까운지 확인
        H_check = H_inv @ H_to_center
        identity_error = np.max(np.abs(H_check - np.eye(3, dtype=np.float32)))
        if identity_error > 1e-4:
            print(f"    Warning: Image {i+1}의 H_inv 검증 실패 (identity_error={identity_error:.2e})")
        
        # 최적화: 이미지가 실제로 차지하는 영역만 처리
        # transformed_corners는 중앙 이미지 좌표계 기준
        # Canvas 좌표계로 변환하려면 adjustment를 더해야 함
        min_x_center_coord = min_x
        max_x_center_coord = max_x
        min_y_center_coord = min_y
        max_y_center_coord = max_y
        
        # Canvas 좌표계로 변환 (중앙 이미지 좌표계 → Canvas 좌표계)
        min_x_canvas = min_x_center_coord + width_adjustment
        max_x_canvas = max_x_center_coord + width_adjustment
        min_y_canvas = min_y_center_coord + height_adjustment
        max_y_canvas = max_y_center_coord + height_adjustment
        
        # Canvas 범위 내로 클램핑
        y_start = max(0, int(np.floor(min_y_canvas)) - 5)  # 여유 5픽셀
        y_end = min(H_canvas, int(np.ceil(max_y_canvas)) + 5)
        x_start = max(0, int(np.floor(min_x_canvas)) - 5)
        x_end = min(W_canvas, int(np.ceil(max_x_canvas)) + 5)
        
        print(f"    처리 영역: y=[{y_start}, {y_end}), x=[{x_start}, {x_end})")
        
        # 처리 영역 크기 검증
        if y_end <= y_start or x_end <= x_start:
            print(f"      Warning: Image {i+1}의 처리 영역이 유효하지 않음. 건너뜁니다.")
            continue
        
        # NumPy 벡터화로 전체 영역 일괄 처리
        total_pixels = (y_end - y_start) * (x_end - x_start)
        print(f"      벡터화 처리 중: {total_pixels} 픽셀...", end='\r')
        
        # 1. meshgrid로 모든 픽셀 좌표 생성
        y_coords, x_coords = np.meshgrid(
            np.arange(y_start, y_end, dtype=np.float32),
            np.arange(x_start, x_end, dtype=np.float32),
            indexing='ij'
        )
        
        # 2. Canvas 좌표 → 중앙 이미지 좌표계 변환 (벡터화)
        x_canvas = x_coords - width_adjustment
        y_canvas = y_coords - height_adjustment
        
        # 3. 동차 좌표 생성 (벡터화)
        # Flatten하여 (N, 3) 형태로 변환
        # C-order로 명시적으로 flatten (행 우선, meshgrid의 indexing='ij'와 호환)
        N = x_canvas.size
        # column_stack은 1D 배열들을 열 방향으로 스택하여 (N, 3) 형태 생성
        points_canvas = np.column_stack([
            x_canvas.flatten(order='C'),
            y_canvas.flatten(order='C'),
            np.ones(N, dtype=np.float32)
        ])  # (N, 3)
        
        # 4. Homography 역변환 (벡터화 - 한 번에 모든 점 변환)
        # H_inv는 (3, 3), points_canvas.T는 (3, N)
        # 결과는 (3, N) → 전치하여 (N, 3)
        points_img_homogeneous = (H_inv @ points_canvas.T).T  # (N, 3)
        
        # 5. 정규화 (벡터화)
        w = points_img_homogeneous[:, 2]
        valid_mask = np.abs(w) > 1e-10
        w_safe = np.where(valid_mask, w, 1.0)
        # C-order로 reshape (flatten과 동일한 순서)
        x_img_all = (points_img_homogeneous[:, 0] / w_safe).reshape(y_coords.shape, order='C')
        y_img_all = (points_img_homogeneous[:, 1] / w_safe).reshape(y_coords.shape, order='C')
        
        # 6. 경계 마스크 생성 (벡터화)
        # 경계를 엄격하게 체크하여 Streaking 방지
        # Bilinear interpolation을 위해서는 x0, x1, y0, y1 모두 유효한 인덱스여야 함
        # x1 = x0 + 1이므로, x0 <= W_img - 2까지 가능해야 x1 <= W_img - 1
        # 따라서 x_img_all < W_img - 1 (또는 x_img_all <= W_img - 2) 조건 필요
        valid_mask = valid_mask.reshape(y_coords.shape, order='C') & \
                     (x_img_all >= 0.0) & (x_img_all <= float(W_img - 2)) & \
                     (y_img_all >= 0.0) & (y_img_all <= float(H_img - 2))
        
        # 디버깅: valid_mask 통계 출력
        total_region_pixels = valid_mask.size
        valid_count = np.sum(valid_mask)
        valid_ratio = 100.0 * valid_count / total_region_pixels if total_region_pixels > 0 else 0.0
        print(f"      유효 픽셀: {valid_count}/{total_region_pixels} ({valid_ratio:.1f}%)", end='\r')
        
        # 유효 픽셀이 너무 적으면 경고
        if valid_count < total_region_pixels * 0.01:  # 1% 미만
            print(f"\n      Warning: 유효 픽셀이 너무 적음 ({valid_count}/{total_region_pixels}, {valid_ratio:.1f}%).")
            print(f"        Homography가 부정확하거나 이미지가 처리 영역 밖에 있을 수 있습니다.")
        
        # 7. 벡터화된 bilinear interpolation
        # valid_mask가 True인 픽셀만 처리 (경계 밖은 이미 필터링됨)
        if np.any(valid_mask):
            # valid_mask로 이미 경계 내부 픽셀만 선택되었으므로 clamp 불필요
            # 하지만 부동소수점 오차 방지를 위해 안전하게 클램핑
            x_img_clamped = np.clip(x_img_all[valid_mask], 0.0, float(W_img - 1))
            y_img_clamped = np.clip(y_img_all[valid_mask], 0.0, float(H_img - 1))
            
            # 정수 좌표 (경계 체크 강화)
            x0 = np.floor(x_img_clamped).astype(np.int32)
            y0 = np.floor(y_img_clamped).astype(np.int32)
            x0 = np.clip(x0, 0, W_img - 1)  # 추가 경계 체크
            y0 = np.clip(y0, 0, H_img - 1)  # 추가 경계 체크
            x1 = np.clip(x0 + 1, 0, W_img - 1)
            y1 = np.clip(y0 + 1, 0, H_img - 1)
            
            # 가중치 계산 (경계에서 1.0을 넘지 않도록)
            wx = np.clip(x_img_clamped - x0.astype(np.float32), 0.0, 1.0)
            wy = np.clip(y_img_clamped - y0.astype(np.float32), 0.0, 1.0)
            
            # x1 == x0 또는 y1 == y0인 경우 가중치 보정
            wx = np.where(x1 == x0, 0.0, wx)
            wy = np.where(y1 == y0, 0.0, wy)
            
            # Bilinear interpolation (벡터화)
            img_float = images[i].astype(np.float32)
            
            # Canvas 좌표 추출 (valid_mask가 True인 위치만)
            # 이 좌표들은 이미지 내부 픽셀에 해당하는 Canvas 위치
            valid_y_coords = y_coords[valid_mask].astype(int)
            valid_x_coords = x_coords[valid_mask].astype(int)
            
            # 중요: x0, y0, x1, y1, wx, wy는 valid_mask로 이미 필터링된 상태
            # 따라서 valid_y_coords, valid_x_coords와 동일한 길이와 순서를 가짐
            
            # Canvas 좌표 범위 체크 (추가 안전장치)
            valid_canvas_mask = (valid_y_coords >= 0) & (valid_y_coords < H_canvas) & \
                               (valid_x_coords >= 0) & (valid_x_coords < W_canvas)
            
            if not np.any(valid_canvas_mask):
                print(f"      Warning: Image {i+1}의 모든 유효 픽셀이 Canvas 범위를 벗어남")
                continue
            
            # Canvas 범위 내 픽셀만 처리
            # 모든 배열을 동일한 순서로 필터링하여 인덱스 일관성 유지
            valid_y_coords = valid_y_coords[valid_canvas_mask]
            valid_x_coords = valid_x_coords[valid_canvas_mask]
            x0 = x0[valid_canvas_mask]
            y0 = y0[valid_canvas_mask]
            x1 = x1[valid_canvas_mask]
            y1 = y1[valid_canvas_mask]
            wx = wx[valid_canvas_mask]
            wy = wy[valid_canvas_mask]
            
            if len(images[i].shape) == 3:
                # RGB 이미지: (H, W, 3) 형태
                # y0, x0는 (N,) 형태이므로 img_float[y0, x0]는 (N, 3) 형태로 올바름
                # 인덱스 범위는 이미 clip되었으므로 안전
                value = (1 - wx[:, np.newaxis]) * (1 - wy[:, np.newaxis]) * img_float[y0, x0] + \
                        wx[:, np.newaxis] * (1 - wy[:, np.newaxis]) * img_float[y0, x1] + \
                        (1 - wx[:, np.newaxis]) * wy[:, np.newaxis] * img_float[y1, x0] + \
                        wx[:, np.newaxis] * wy[:, np.newaxis] * img_float[y1, x1]
                # value는 (N, 3) 형태
            else:
                # 그레이스케일 이미지: (H, W) 형태
                value = (1 - wx) * (1 - wy) * img_float[y0, x0] + \
                        wx * (1 - wy) * img_float[y0, x1] + \
                        (1 - wx) * wy * img_float[y1, x0] + \
                        wx * wy * img_float[y1, x1]
                # value는 (N,) 형태
            
            # Weighted blending: 이미지 중심에서의 거리 기반 가중치 계산
            # 이미지 중심 좌표 (원본 이미지 기준)
            center_x = W_img / 2.0
            center_y = H_img / 2.0
            
            # 원본 이미지 좌표에서 중심까지의 거리 계산
            # x_img_clamped, y_img_clamped는 valid_mask로 이미 필터링된 상태
            # valid_canvas_mask로 다시 필터링하여 x0, y0, wx, wy와 동일한 길이로 맞춤
            # 인덱스 일관성 보장: valid_mask → valid_canvas_mask 순서로 필터링
            x_img_filtered = x_img_clamped  # 이미 valid_mask로 필터링됨
            y_img_filtered = y_img_clamped  # 이미 valid_mask로 필터링됨
            
            # valid_canvas_mask 적용 (x0, y0와 동일한 순서로 필터링)
            x_img_filtered = x_img_filtered[valid_canvas_mask]
            y_img_filtered = y_img_filtered[valid_canvas_mask]
            
            # 거리 계산 (정규화: 최대 거리 = 대각선 길이)
            max_dist = np.sqrt(center_x**2 + center_y**2)
            distances = np.sqrt((x_img_filtered - center_x)**2 + (y_img_filtered - center_y)**2)
            
            # 가중치: 중심에 가까울수록 높은 가중치 (선형 감쇠)
            # 거리가 멀수록 가중치 감소, 이미지 경계 근처에서 더 빠르게 감소
            # Ghost 현상 감소를 위해 경계 가중치를 더 강하게 감소
            normalized_distances = distances / (max_dist + 1e-6)
            # 거리 기반 가중치: 중심에서 멀어질수록 더 빠르게 감소 (제곱 함수 사용)
            weights = np.maximum((1.0 - normalized_distances) ** 2, 0.05)  # 최소값 0.05, 제곱으로 더 빠른 감소
            
            # 값 할당 (가중치 적용)
            # Weighted sum: panorama += value * weight
            if len(images[i].shape) == 3:
                panorama[valid_y_coords, valid_x_coords] += value * weights[:, np.newaxis]
            else:
                panorama[valid_y_coords, valid_x_coords] += value * weights
            count[valid_y_coords, valid_x_coords] += 1.0
            weight_sum[valid_y_coords, valid_x_coords] += weights
            
            # 실제 할당된 픽셀 수 디버깅
            assigned_pixels = len(valid_y_coords)
            print(f"      할당된 픽셀: {assigned_pixels}개" + " " * 20)
        else:
            print(f"      Warning: 유효 픽셀이 없음. Image {i+1}를 건너뜁니다." + " " * 20)
        
        print(f"      완료: Image {i+1} 처리 완료 (벡터화)" + " " * 30)  # 공백으로 이전 출력 지움
    
    # Weighted blending: 가중치 합으로 나누기
    mask = weight_sum > 1e-6  # 가중치 합이 0보다 큰 픽셀만
    panorama[mask] = panorama[mask] / weight_sum[mask, np.newaxis]
    
    # 가중치 합이 0인 픽셀은 검은색으로 유지 (이미 0으로 초기화됨)
    
    return panorama.astype(np.uint8)

