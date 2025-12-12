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
        # 누적 변환: H[i]는 images[i+1]을 images[i]로 변환
        # cumulative_H는 images[i]를 images[0]로 변환
        # images[i+1]을 images[0]로 변환하려면:
        # images[i+1] -> images[i] (H) -> images[0] (cumulative_H)
        # 따라서 cumulative_H = cumulative_H @ H (먼저 H 적용, 그 다음 cumulative_H 적용)
        cumulative_H = cumulative_H @ H
        
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


def find_seam(image1: np.ndarray, image2: np.ndarray, mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
    """
    Seam Finding: 오버랩 영역에서 최소 에너지 경로를 찾습니다.
    
    Dynamic Programming을 사용하여 각 행에서 최소 에너지 경로를 계산합니다.
    에너지는 두 이미지 간의 차이로 정의됩니다.
    
    Args:
        image1: 첫 번째 이미지 (H, W, C) - uint8
        image2: 두 번째 이미지 (H, W, C) - uint8
        mask1: 첫 번째 이미지의 마스크 (H, W) - bool
        mask2: 두 번째 이미지의 마스크 (H, W) - bool
    
    Returns:
        seam_mask: Seam 마스크 (H, W) - bool, True인 경우 image1 사용, False인 경우 image2 사용
    """
    H, W = image1.shape[:2]
    
    # 오버랩 영역 식별
    overlap = mask1 & mask2
    
    if not np.any(overlap):
        # 오버랩이 없으면 마스크 기반으로 반환
        return mask1
    
    # 그레이스케일 변환 (에너지 계산용)
    if len(image1.shape) == 3:
        gray1 = np.mean(image1.astype(np.float32), axis=2)
        gray2 = np.mean(image2.astype(np.float32), axis=2)
    else:
        gray1 = image1.astype(np.float32)
        gray2 = image2.astype(np.float32)
    
    # 에너지 맵 계산: 두 이미지 간의 차이
    energy = np.abs(gray1 - gray2)  # (H, W)
    
    # 오버랩 영역에서만 Seam 찾기
    # 각 행에서 오버랩 영역의 x 범위 찾기
    seam_mask = np.zeros((H, W), dtype=bool)
    
    # Dynamic Programming으로 각 행의 최소 에너지 경로 찾기
    # 간단한 버전: 각 행에서 오버랩 영역의 중심을 기준으로 Seam 결정
    for y in range(H):
        # 해당 행의 오버랩 영역 찾기
        row_overlap = overlap[y, :]
        if not np.any(row_overlap):
            # 오버랩이 없으면 마스크 기반으로 결정
            seam_mask[y, :] = mask1[y, :]
            continue
        
        # 오버랩 영역의 x 범위
        x_indices = np.where(row_overlap)[0]
        if len(x_indices) == 0:
            seam_mask[y, :] = mask1[y, :]
            continue
        
        x_min = x_indices[0]
        x_max = x_indices[-1]
        
        # 오버랩 영역에서만 Seam 찾기
        # 간단한 버전: 누적 에너지 최소 경로 찾기
        overlap_width = x_max - x_min + 1
        if overlap_width < 2:
            # 너무 좁으면 중심 기준
            center_x = (x_min + x_max) // 2
            seam_mask[y, :x_min] = mask1[y, :x_min]
            seam_mask[y, x_min:center_x+1] = True  # image1 사용
            seam_mask[y, center_x+1:x_max+1] = False  # image2 사용
            seam_mask[y, x_max+1:] = mask2[y, x_max+1:]
            continue
        
        # 오버랩 영역의 에너지 추출
        overlap_energy = energy[y, x_min:x_max+1]
        
        # Dynamic Programming: 누적 에너지 최소 경로
        # dp[i] = x_min + i 위치에서의 최소 누적 에너지
        dp = np.zeros(overlap_width, dtype=np.float32)
        parent = np.zeros(overlap_width, dtype=np.int32)  # 이전 위치 추적
        
        # 초기화: 첫 번째 열
        dp[0] = overlap_energy[0]
        parent[0] = -1
        
        # 각 열에 대해 최소 경로 찾기
        for i in range(1, overlap_width):
            # 이전 열의 3개 위치 중 최소값 선택 (상, 중, 하)
            candidates = []
            candidate_indices = []
            
            # 중앙 (같은 행)
            candidates.append(dp[i-1])
            candidate_indices.append(i-1)
            
            # 위쪽 (이전 행, 같은 열) - 간단화를 위해 같은 행만 고려
            # 실제로는 이전 행의 같은 열도 고려해야 하지만, 여기서는 단순화
            
            # 최소값 선택
            min_idx = np.argmin(candidates)
            dp[i] = candidates[min_idx] + overlap_energy[i]
            parent[i] = candidate_indices[min_idx]
        
        # 역추적: 최소 에너지 경로 찾기
        # 마지막 열에서 최소값 위치 찾기
        min_energy_idx = np.argmin(dp)
        
        # 경로 역추적
        path = []
        current_idx = min_energy_idx
        while current_idx >= 0:
            path.append(x_min + current_idx)
            current_idx = parent[current_idx]
        
        # 경로를 역순으로 정렬
        path = path[::-1]
        
        # Seam 마스크 생성: 경로 왼쪽은 image1, 오른쪽은 image2
        seam_x = path[-1] if len(path) > 0 else (x_min + x_max) // 2
        
        # 비오버랩 영역
        seam_mask[y, :x_min] = mask1[y, :x_min]
        seam_mask[y, x_max+1:] = mask2[y, x_max+1:]
        
        # 오버랩 영역: Seam 기준으로 분할
        seam_mask[y, x_min:seam_x+1] = True  # image1
        seam_mask[y, seam_x+1:x_max+1] = False  # image2
    
    return seam_mask


def compute_reprojection_error_map(image1: np.ndarray, image2: np.ndarray, 
                                   mask1: np.ndarray, mask2: np.ndarray,
                                   H: np.ndarray) -> np.ndarray:
    """
    오버랩 영역에서 재투영 오차 맵을 계산합니다.
    
    Args:
        image1: 첫 번째 이미지 (H, W, C) - uint8
        image2: 두 번째 이미지 (H, W, C) - uint8
        mask1: 첫 번째 이미지의 마스크 (H, W) - bool
        mask2: 두 번째 이미지의 마스크 (H, W) - bool
        H: image2를 image1 좌표계로 변환하는 Homography (3, 3) - float32
    
    Returns:
        error_map: 재투영 오차 맵 (H, W) - float32, 작을수록 정렬이 정확함
    """
    from geometry import apply_homography
    
    H_img, W_img = image2.shape[:2]
    
    # image2의 모든 픽셀 좌표
    y_coords, x_coords = np.meshgrid(np.arange(H_img), np.arange(W_img), indexing='ij')
    points2 = np.stack([x_coords.flatten(), y_coords.flatten()], axis=1).astype(np.float32)
    
    # Homography로 변환
    points2_transformed = apply_homography(points2, H)
    points2_transformed = points2_transformed.reshape(H_img, W_img, 2)
    
    # image1과 image2의 그레이스케일 변환
    if len(image1.shape) == 3:
        gray1 = np.mean(image1.astype(np.float32), axis=2)
        gray2 = np.mean(image2.astype(np.float32), axis=2)
    else:
        gray1 = image1.astype(np.float32)
        gray2 = image2.astype(np.float32)
    
    # 오버랩 영역에서만 오차 계산
    overlap = mask1 & mask2
    error_map = np.zeros((image1.shape[0], image1.shape[1]), dtype=np.float32)
    
    # image2의 각 픽셀이 image1의 어디에 매핑되는지 확인
    for y in range(H_img):
        for x in range(W_img):
            if not mask2[y, x]:
                continue
            
            # 변환된 좌표
            x_t, y_t = points2_transformed[y, x, 0], points2_transformed[y, x, 1]
            x_t_int = int(np.round(x_t))
            y_t_int = int(np.round(y_t))
            
            # image1 범위 내인지 확인
            if 0 <= y_t_int < image1.shape[0] and 0 <= x_t_int < image1.shape[1]:
                if mask1[y_t_int, x_t_int]:
                    # 픽셀 값 차이로 오차 계산
                    error = np.abs(gray1[y_t_int, x_t_int] - gray2[y, x])
                    error_map[y_t_int, x_t_int] = error
    
    return error_map


def gaussian_weight_blend(image1: np.ndarray, image2: np.ndarray, 
                         mask1: np.ndarray, mask2: np.ndarray,
                         sigma: float = 50.0) -> np.ndarray:
    """
    가우시안 가중 블렌딩을 수행합니다.
    
    오버랩 영역에서 각 이미지의 중심으로부터의 거리에 가우시안 가중치를 적용합니다.
    
    Args:
        image1: 첫 번째 이미지 (H, W, C) - uint8
        image2: 두 번째 이미지 (H, W, C) - uint8
        mask1: 첫 번째 이미지의 마스크 (H, W) - bool
        mask2: 두 번째 이미지의 마스크 (H, W) - bool
        sigma: 가우시안 표준편차 (float, 기본값: 50.0)
    
    Returns:
        blended: 블렌딩된 이미지 (H, W, C) - uint8
    """
    H_img, W_img = image1.shape[:2]
    
    # 오버랩 영역 식별
    overlap = mask1 & mask2
    
    if not np.any(overlap):
        # 오버랩이 없으면 마스크 기반으로 합성
        blended = np.zeros_like(image1, dtype=np.float32)
        blended[mask1] = image1[mask1].astype(np.float32)
        blended[mask2] = image2[mask2].astype(np.float32)
        return blended.astype(np.uint8)
    
    # 각 마스크의 중심 계산
    mask1_y, mask1_x = np.where(mask1)
    mask2_y, mask2_x = np.where(mask2)
    
    if len(mask1_y) == 0 or len(mask2_y) == 0:
        # 마스크가 비어있으면 기본 블렌딩
        blended = np.zeros_like(image1, dtype=np.float32)
        blended[mask1] = image1[mask1].astype(np.float32)
        blended[mask2] = image2[mask2].astype(np.float32)
        return blended.astype(np.uint8)
    
    center1_y = np.mean(mask1_y)
    center1_x = np.mean(mask1_x)
    center2_y = np.mean(mask2_y)
    center2_x = np.mean(mask2_x)
    
    # 각 픽셀에서 각 중심까지의 거리 계산
    y_coords, x_coords = np.meshgrid(np.arange(H_img), np.arange(W_img), indexing='ij')
    
    dist1 = np.sqrt((y_coords - center1_y)**2 + (x_coords - center1_x)**2)
    dist2 = np.sqrt((y_coords - center2_y)**2 + (x_coords - center2_x)**2)
    
    # 가우시안 가중치 계산
    weight1 = np.exp(-(dist1**2) / (2 * sigma**2))
    weight2 = np.exp(-(dist2**2) / (2 * sigma**2))
    
    # 오버랩 영역에서만 가중치 적용
    weight1 = np.where(overlap, weight1, mask1.astype(np.float32))
    weight2 = np.where(overlap, weight2, mask2.astype(np.float32))
    
    # 가중치 정규화
    total_weight = weight1 + weight2
    total_weight = np.where(total_weight == 0, 1.0, total_weight)
    
    w1 = weight1 / total_weight
    w2 = weight2 / total_weight
    
    # 블렌딩
    if len(image1.shape) == 3:
        w1 = w1[:, :, np.newaxis]
        w2 = w2[:, :, np.newaxis]
    
    blended = (w1 * image1.astype(np.float32) + w2 * image2.astype(np.float32))
    
    return blended.astype(np.uint8)


def simple_blend(image1: np.ndarray, image2: np.ndarray, mask1: Optional[np.ndarray] = None, mask2: Optional[np.ndarray] = None, H: Optional[np.ndarray] = None) -> np.ndarray:
    """
    개선된 블렌딩: Seam Finding + 재투영 오차 기반 선택적 블렌딩 + 가우시안 가중 블렌딩
    
    고스팅 현상을 줄이기 위해 다음 기법들을 통합합니다:
    1. Seam Finding: 최소 에너지 경로 찾기
    2. 재투영 오차 기반 선택적 블렌딩: 정렬이 정확한 영역은 한 이미지 선택
    3. 가우시안 가중 블렌딩: 경계에서 부드러운 전환
    
    Args:
        image1: 첫 번째 이미지 (H, W, C) - uint8
        image2: 두 번째 이미지 (H, W, C) - uint8
        mask1: 첫 번째 이미지의 마스크 (H, W) - bool, Optional
        mask2: 두 번째 이미지의 마스크 (H, W) - bool, Optional
        H: image2를 image1 좌표계로 변환하는 Homography (3, 3) - float32, Optional
    
    Returns:
        blended: 블렌딩된 이미지 (H, W, C) - uint8
    """
    if mask1 is None:
        mask1 = create_mask(image1)
    if mask2 is None:
        mask2 = create_mask(image2)
    
    # 오버랩 영역 식별
    overlap = mask1 & mask2
    
    if not np.any(overlap):
        # 오버랩이 없으면 마스크 기반으로 합성
        blended = np.zeros_like(image1, dtype=np.float32)
        if len(image1.shape) == 3:
            blended[mask1] = image1[mask1].astype(np.float32)
            blended[mask2] = image2[mask2].astype(np.float32)
        else:
            blended[mask1] = image1[mask1].astype(np.float32)
            blended[mask2] = image2[mask2].astype(np.float32)
        return blended.astype(np.uint8)
    
    # 1. 재투영 오차 기반 선택적 블렌딩 (H가 제공된 경우)
    error_threshold = 1.0  # 픽셀 단위 오차 임계값
    use_error_based = (H is not None)
    
    if use_error_based:
        error_map = compute_reprojection_error_map(image1, image2, mask1, mask2, H)
        # 오차가 작은 영역은 한 이미지 선택, 큰 영역은 블렌딩
        low_error_mask = error_map < error_threshold
    else:
        low_error_mask = np.zeros_like(overlap, dtype=bool)
    
    # 2. Seam Finding으로 최적 경계 찾기
    seam_mask = find_seam(image1, image2, mask1, mask2)
    
    # 3. 최종 블렌딩 전략 결정
    H_img, W_img = image1.shape[:2]
    blended = np.zeros_like(image1, dtype=np.float32)
    
    # 비오버랩 영역
    only1 = mask1 & (~mask2)
    only2 = mask2 & (~mask1)
    
    if len(image1.shape) == 3:
        blended[only1] = image1[only1].astype(np.float32)
        blended[only2] = image2[only2].astype(np.float32)
    else:
        blended[only1] = image1[only1].astype(np.float32)
        blended[only2] = image2[only2].astype(np.float32)
    
    # 오버랩 영역 처리
    if use_error_based and np.any(low_error_mask & overlap):
        # 재투영 오차가 작은 영역: Seam 기반으로 한 이미지 선택
        overlap_low_error = overlap & low_error_mask
        if len(image1.shape) == 3:
            blended[overlap_low_error & seam_mask] = image1[overlap_low_error & seam_mask].astype(np.float32)
            blended[overlap_low_error & (~seam_mask)] = image2[overlap_low_error & (~seam_mask)].astype(np.float32)
        else:
            blended[overlap_low_error & seam_mask] = image1[overlap_low_error & seam_mask].astype(np.float32)
            blended[overlap_low_error & (~seam_mask)] = image2[overlap_low_error & (~seam_mask)].astype(np.float32)
        
        # 재투영 오차가 큰 영역: 가우시안 가중 블렌딩
        overlap_high_error = overlap & (~low_error_mask)
        if np.any(overlap_high_error):
            blended_high_error = gaussian_weight_blend(image1, image2, mask1, mask2, sigma=50.0)
            if len(image1.shape) == 3:
                blended[overlap_high_error] = blended_high_error[overlap_high_error]
            else:
                blended[overlap_high_error] = blended_high_error[overlap_high_error]
    else:
        # H가 없거나 오차 기반 선택이 불가능한 경우: Seam 기반 선택 + 가우시안 블렌딩
        # Seam 근처에서만 블렌딩, 나머지는 한 이미지 선택
        blend_width = 10  # Seam 주변 블렌딩 영역 폭
        
        # Seam 근처 영역 찾기 (간단한 근사)
        # 각 픽셀에서 Seam까지의 거리 계산
        seam_distance = np.zeros_like(overlap, dtype=np.float32)
        
        for y in range(H_img):
            row_overlap = overlap[y, :]
            if not np.any(row_overlap):
                continue
            
            x_indices = np.where(row_overlap)[0]
            if len(x_indices) == 0:
                continue
            
            # Seam 위치 찾기 (seam_mask가 True에서 False로 바뀌는 지점)
            seam_x = None
            for x in range(x_indices[0], x_indices[-1] + 1):
                if x < W_img - 1 and seam_mask[y, x] and not seam_mask[y, x+1]:
                    seam_x = x
                    break
            
            if seam_x is None:
                # Seam을 찾지 못하면 중심 사용
                seam_x = (x_indices[0] + x_indices[-1]) // 2
            
            # 각 픽셀에서 Seam까지의 거리
            x_coords = np.arange(W_img)
            seam_distance[y, :] = np.abs(x_coords - seam_x)
        
        # Seam 근처에서만 블렌딩
        blend_region = overlap & (seam_distance < blend_width)
        select_region = overlap & (seam_distance >= blend_width)
        
        # 선택 영역: Seam 기반으로 한 이미지 선택
        if len(image1.shape) == 3:
            blended[select_region & seam_mask] = image1[select_region & seam_mask].astype(np.float32)
            blended[select_region & (~seam_mask)] = image2[select_region & (~seam_mask)].astype(np.float32)
        else:
            blended[select_region & seam_mask] = image1[select_region & seam_mask].astype(np.float32)
            blended[select_region & (~seam_mask)] = image2[select_region & (~seam_mask)].astype(np.float32)
        
        # 블렌딩 영역: 가우시안 가중 블렌딩
        if np.any(blend_region):
            blended_blend = gaussian_weight_blend(image1, image2, mask1, mask2, sigma=30.0)
            if len(image1.shape) == 3:
                blended[blend_region] = blended_blend[blend_region]
            else:
                blended[blend_region] = blended_blend[blend_region]
    
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
    개선된 버전: 경계를 더 정확하게 감지하고, 큰 이미지에서도 메모리 효율적으로 처리합니다.
    
    Args:
        image: 이미지 (H, W, C) - uint8
    
    Returns:
        mask: 마스크 (H, W) - bool, True인 경우 유효 픽셀
    """
    H, W = image.shape[:2]
    
    # 큰 이미지의 경우 타일 단위로 처리 (메모리 효율성)
    if H * W > 500000:  # 픽셀 수가 50만 개 이상이면 타일 처리
        mask = np.zeros((H, W), dtype=bool)
        tile_size = 2000
        
        for y_start in range(0, H, tile_size):
            y_end = min(y_start + tile_size, H)
            for x_start in range(0, W, tile_size):
                x_end = min(x_start + tile_size, W)
                
                tile = image[y_start:y_end, x_start:x_end]
                if len(image.shape) == 3:
                    tile_mask = np.any(tile > 1, axis=2)
                else:
                    tile_mask = tile > 1
                mask[y_start:y_end, x_start:x_end] = tile_mask
        
        return mask
    
    # 작은 이미지: 직접 처리
    if len(image.shape) == 3:
        # 컬러 이미지: 모든 채널이 0이 아닌 경우만 유효
        # 개선: 약간의 노이즈 허용 (1 이상의 값)
        mask = np.any(image > 1, axis=2)
    else:
        # 그레이스케일 이미지
        mask = image > 1
    
    return mask.astype(bool)


def compute_distance_to_boundary(mask: np.ndarray) -> np.ndarray:
    """
    마스크의 각 픽셀에서 가장 가까운 경계까지의 거리를 계산합니다.
    
    최적화된 버전: 큰 이미지에서도 빠르게 동작하도록 간소화된 거리 근사치를 사용합니다.
    
    Args:
        mask: 마스크 (H, W) - bool
    
    Returns:
        distance: 거리 맵 (H, W) - float32, 경계에서 멀수록 큰 값 (0-1 범위)
    """
    H, W = mask.shape
    
    if not np.any(mask):
        return np.zeros((H, W), dtype=np.float32)
    
    # 큰 이미지의 경우 거리 계산을 건너뛰고 균등 가중치 반환
    # 성능 최적화: 픽셀 수가 1백만 개 이상이면 거리 계산 생략
    if H * W > 1000000:
        # 큰 이미지: 균등 가중치 반환 (거리 계산 생략)
        distance = mask.astype(np.float32)
        return distance
    
    # 작은 이미지: 간단한 거리 근사치 계산
    # 각 픽셀에서 가장 가까운 경계까지의 거리를 근사치로 계산
    distance = np.zeros((H, W), dtype=np.float32)
    
    # 경계 픽셀 찾기 (마스크가 True이지만 상하좌우 중 하나라도 False인 픽셀)
    mask_padded = np.pad(mask, ((1, 1), (1, 1)), mode='constant', constant_values=False)
    
    # 상하좌우 이웃 확인
    up = mask_padded[:-2, 1:-1]
    down = mask_padded[2:, 1:-1]
    left = mask_padded[1:-1, :-2]
    right = mask_padded[1:-1, 2:]
    
    # 경계 픽셀: 자신은 True이지만 이웃 중 하나라도 False
    boundary = mask & (~up | ~down | ~left | ~right)
    
    # 경계 픽셀의 좌표
    boundary_y, boundary_x = np.where(boundary)
    
    if len(boundary_y) == 0:
        # 경계가 없으면 모든 픽셀에 최대 거리 부여
        distance[mask] = 1.0
        return distance
    
    # 각 마스크 픽셀의 좌표
    mask_y, mask_x = np.where(mask)
    
    # 간단한 거리 근사: 각 마스크 픽셀에서 가장 가까운 경계까지의 거리
    # 효율성을 위해 다운샘플링된 버전에서 거리 계산
    if len(mask_y) > 50000:  # 마스크 픽셀이 너무 많으면 다운샘플링
        # 다운샘플링 비율
        scale = 4
        mask_downsampled = mask[::scale, ::scale]
        distance_downsampled = compute_distance_to_boundary(mask_downsampled)
        
        # 업샘플링 (간단한 nearest neighbor - NumPy만 사용, 벡터화)
        # 원본 크기로 확대
        H_ds, W_ds = distance_downsampled.shape
        
        # 벡터화된 인덱스 계산
        y_indices = np.arange(H, dtype=np.int32) // scale
        x_indices = np.arange(W, dtype=np.int32) // scale
        y_indices = np.clip(y_indices, 0, H_ds - 1)
        x_indices = np.clip(x_indices, 0, W_ds - 1)
        
        # 메시그리드로 인덱스 생성
        Y, X = np.meshgrid(y_indices, x_indices, indexing='ij')
        distance = distance_downsampled[Y, X]
        
        return distance
    
    # 작은 마스크: 직접 거리 계산 (최적화된 버전)
    # 각 마스크 픽셀에 대해 가장 가까운 경계까지의 거리
    # 벡터화된 계산 (배치 처리)
    batch_size = 5000  # 배치 크기 증가
    for i in range(0, len(mask_y), batch_size):
        end_idx = min(i + batch_size, len(mask_y))
        batch_y = mask_y[i:end_idx]
        batch_x = mask_x[i:end_idx]
        
        # 배치 내 각 픽셀에서 모든 경계까지의 거리
        dy = boundary_y[:, np.newaxis] - batch_y[np.newaxis, :]  # (n_boundary, n_batch)
        dx = boundary_x[:, np.newaxis] - batch_x[np.newaxis, :]  # (n_boundary, n_batch)
        dists = np.sqrt(dy**2 + dx**2)  # (n_boundary, n_batch)
        min_dists = np.min(dists, axis=0)  # (n_batch,)
        
        distance[batch_y, batch_x] = min_dists
    
    # 정규화 (최대값으로 나누어 0-1 범위로)
    max_dist = np.max(distance)
    if max_dist > 0:
        distance = distance / max_dist
    else:
        distance[mask] = 1.0
    
    return distance


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
    
    # 블렌딩 (Homography 전달)
    mask1 = create_mask(panorama)
    mask2 = create_mask(warped2)
    panorama = simple_blend(panorama, warped2, mask1, mask2, H=H)
    
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
        print(f"  이미지 {i+2}/{len(images)} 스티칭 중...")
        
        # 누적 변환: H[i]는 images[i+1]을 images[i]로 변환
        # cumulative_H는 images[i]를 images[0]로 변환
        # images[i+1]을 images[0]로 변환하려면:
        # images[i+1] -> images[i] (H) -> images[0] (cumulative_H)
        # 따라서 cumulative_H = cumulative_H @ H (먼저 H 적용, 그 다음 cumulative_H 적용)
        cumulative_H = cumulative_H @ H
        
        # 오프셋을 고려한 Homography
        # 캔버스 좌표계로 변환하기 위해 오프셋 추가
        # T는 캔버스 좌표계로의 변환 (오프셋 추가)
        T = np.array([[1, 0, offset_x],
                      [0, 1, offset_y],
                      [0, 0, 1]], dtype=np.float32)
        # H_offset: images[i+1]의 픽셀을 캔버스 좌표계로 변환
        H_offset = T @ cumulative_H
        
        # 이미지 워핑
        print(f"    워핑 중... (cumulative_H scale: {np.sqrt(cumulative_H[0,0]**2 + cumulative_H[0,1]**2):.2f})")
        warped = inverse_warp(image, H_offset, (canvas_height, canvas_width), offset)
        
        # 블렌딩 (원본 Homography 전달 - 재투영 오차 계산용)
        print(f"    블렌딩 중...")
        mask1 = create_mask(panorama)
        mask2 = create_mask(warped)
        # 재투영 오차 계산을 위해 원본 cumulative_H 사용 (오프셋 없이)
        panorama = simple_blend(panorama, warped, mask1, mask2, H=cumulative_H)
        print(f"  이미지 {i+2} 스티칭 완료")
    
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

