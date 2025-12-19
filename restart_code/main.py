"""
파노라마 이미지 스티칭 메인 파이프라인
필수 구현 요소만 포함:
1. 코너 포인트 찾기 (Harris Corner Detection)
2. Point Matching (Feature Matching)
3. Homography 계산 (DLT 알고리즘)
4. Stitching
"""

import numpy as np
import os
from typing import List, Tuple

from utils import load_images_from_folder, rgb_to_grayscale, normalize_image, save_image, show_image
from corner_detection import harris_corner_detection
from point_matching import compute_descriptors, match_features, get_matched_points
from homography import compute_homography_dlt, apply_homography
from ransac import ransac_homography
from stitching import stitch_multiple_images


def filter_by_spatial_distribution(image1: np.ndarray,
                                   image2: np.ndarray,
                                   corners1: np.ndarray, 
                                   corners2: np.ndarray, 
                                   matches: List[Tuple[int, int]],
                                   grid_size: int = 3) -> List[Tuple[int, int]]:
    """
    매칭된 점들이 이미지 전체에 고르게 분포하도록 필터링합니다.
    
    Args:
        image1: 첫 번째 이미지 (H1, W1, 3) - 실제 이미지 크기 확인용
        image2: 두 번째 이미지 (H2, W2, 3) - 실제 이미지 크기 확인용
        corners1: 첫 번째 이미지의 코너 점들 (N1, 2)
        corners2: 두 번째 이미지의 코너 점들 (N2, 2)
        matches: 매칭 결과 리스트 [(idx1, idx2), ...]
        grid_size: 그리드 크기 (int, 기본값: 3, 3x3 = 9개 영역)
    
    Returns:
        filtered_matches: 필터링된 매칭 리스트
    """
    if len(matches) < grid_size * grid_size:
        return matches
    
    # 실제 이미지 크기 사용
    H1, W1 = image1.shape[:2]
    H2, W2 = image2.shape[:2]
    
    # 그리드 경계 계산
    cell_w1 = W1 / grid_size
    cell_h1 = H1 / grid_size
    cell_w2 = W2 / grid_size
    cell_h2 = H2 / grid_size
    
    # 각 매칭을 그리드 셀에 분류
    grid_cells = {}  # (cell_row, cell_col) -> [match indices]
    
    for i, (idx1, idx2) in enumerate(matches):
        corner1 = corners1[idx1]
        corner2 = corners2[idx2]
        
        # 각 이미지에서의 그리드 위치
        cell_col1 = int(corner1[0] / cell_w1)
        cell_row1 = int(corner1[1] / cell_h1)
        cell_col2 = int(corner2[0] / cell_w2)
        cell_row2 = int(corner2[1] / cell_h2)
        
        # 그리드 위치 클램핑
        cell_col1 = min(cell_col1, grid_size - 1)
        cell_row1 = min(cell_row1, grid_size - 1)
        cell_col2 = min(cell_col2, grid_size - 1)
        cell_row2 = min(cell_row2, grid_size - 1)
        
        # 두 이미지의 그리드 위치 평균 사용
        cell_key = (cell_row1, cell_col1)  # 첫 번째 이미지 기준
        
        if cell_key not in grid_cells:
            grid_cells[cell_key] = []
        grid_cells[cell_key].append(i)
    
    # 각 그리드 셀에서 최대 개수 제한 (균등 분산)
    max_per_cell = max(1, len(matches) // (grid_size * grid_size) + 2)
    
    filtered_indices = []
    for cell_key, match_indices in grid_cells.items():
        # 각 셀에서 최대 max_per_cell개만 선택
        selected = match_indices[:max_per_cell]
        filtered_indices.extend(selected)
    
    # 필터링된 매칭 반환
    filtered_matches = [matches[i] for i in filtered_indices]
    
    return filtered_matches


def compute_pairwise_homography(image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
    """
    두 이미지 간의 Homography를 계산합니다.
    
    Args:
        image1: 첫 번째 이미지 (H1, W1, 3) - uint8
        image2: 두 번째 이미지 (H2, W2, 3) - uint8
    
    Returns:
        H: Homography 행렬 (3, 3) - float32
           image2를 image1 좌표계로 변환하는 Homography
    """
    # 1. 그레이스케일 변환
    gray1 = rgb_to_grayscale(image1)
    gray2 = rgb_to_grayscale(image2)
    
    # 2. 정규화 (밝기 차이 극복)
    gray1_norm = normalize_image(gray1)
    gray2_norm = normalize_image(gray2)
    
    # 3. 코너 포인트 찾기
    corners1 = harris_corner_detection(
        gray1_norm,
        threshold=0.01,
        k=0.04,
        window_size=3,
        sigma=1.0
    )
    
    corners2 = harris_corner_detection(
        gray2_norm,
        threshold=0.01,
        k=0.04,
        window_size=3,
        sigma=1.0
    )
    
    print(f"  Image 1: {len(corners1)} corners detected")
    print(f"  Image 2: {len(corners2)} corners detected")
    
    if len(corners1) < 4 or len(corners2) < 4:
        print(f"  Warning: Too few corners. Using identity matrix.")
        return np.eye(3, dtype=np.float32)
    
    # 4. Descriptor 계산
    descriptors1 = compute_descriptors(gray1_norm, corners1, patch_size=21)
    descriptors2 = compute_descriptors(gray2_norm, corners2, patch_size=21)
    
    # 5. Feature Matching (threshold 완화하여 더 많은 후보 확보)
    # 공간 분산을 위해 약간 완화된 threshold 사용
    matches = match_features(descriptors1, descriptors2, threshold=0.75)
    print(f"  Matched features: {len(matches)}")
    
    if len(matches) < 4:
        print(f"  Warning: Too few matches ({len(matches)} < 4). Using identity matrix.")
        return np.eye(3, dtype=np.float32)
    
    # 6. 매칭된 점 좌표 추출
    points1, points2 = get_matched_points(corners1, corners2, matches)
    
    # 6.5. 공간 분산 필터링: 매칭이 이미지 전체에 고르게 분포하도록 필터링
    # 이미지를 3x3 그리드로 나누어 각 영역에서 균등하게 선택
    # 단, 충분한 매칭이 있을 때만 적용하고, 필터링 후에도 충분한 매칭이 남아있어야 함
    if len(matches) > 20:  # 충분한 매칭이 있을 때만 적용 (기준 완화: 12 → 20)
        matches_filtered = filter_by_spatial_distribution(image1, image2, corners1, corners2, matches, grid_size=3)
        # 필터링 후에도 최소 8개 이상의 매칭이 남아있어야 함 (RANSAC을 위해)
        if len(matches_filtered) >= max(8, len(matches) * 0.3):  # 원래 매칭의 30% 이상
            matches = matches_filtered
            points1, points2 = get_matched_points(corners1, corners2, matches)
            print(f"  After spatial filtering: {len(matches)} matches")
    
    # 7. RANSAC을 사용하여 Outlier 제거 및 Homography 계산
    # image1 -> image2 변환 Homography 계산
    # min_inliers를 동적으로 조정: 매칭 개수의 30% 또는 최소 4개
    min_inliers = max(4, int(len(matches) * 0.3))
    
    H_1to2, inlier_mask = ransac_homography(
        points1, points2,
        max_iterations=2000,
        threshold=3.0,  # 픽셀 단위 임계값
        min_inliers=min_inliers
    )
    
    inlier_count = np.sum(inlier_mask)
    print(f"  RANSAC inliers: {inlier_count}/{len(matches)} ({100*inlier_count/len(matches):.1f}%)")
    
    # RANSAC fallback은 이미 ransac_homography 내부에서 처리됨
    # 여기서는 추가 검증만 수행 (경고 출력)
    if inlier_count < 4:
        print(f"  Warning: Low inlier count ({inlier_count} < 4). Result may be unreliable.")
    
    # 8. 스티칭에는 image2를 image1로 변환하는 Homography가 필요하므로 역행렬 사용
    det = np.linalg.det(H_1to2)
    if abs(det) < 1e-5:
        print(f"  Warning: Homography determinant too small ({det:.2e}). Using identity matrix.")
        H = np.eye(3, dtype=np.float32)
    else:
        H = np.linalg.inv(H_1to2)
        if abs(H[2, 2]) > 1e-10:
            H = H / H[2, 2]  # 정규화
        
        # 9. Homography 검증: Scale factor 체크 (완화)
        scale_x = np.sqrt(H[0, 0]**2 + H[0, 1]**2)
        scale_y = np.sqrt(H[1, 0]**2 + H[1, 1]**2)
        
        # 비정상적인 scale factor 검증 (완화: 파노라마는 확대/축소될 수 있음)
        if scale_x < 0.01 or scale_x > 100.0 or scale_y < 0.01 or scale_y > 100.0:  # 0.05→0.01, 20→100으로 완화
            print(f"  Warning: Homography scale factor out of range (x: {scale_x:.2f}, y: {scale_y:.2f}).")
            print(f"    Identity matrix로 대체합니다.")
            return np.eye(3, dtype=np.float32)
        
        # 10. 변환된 corner 위치 검증: image2의 corner를 image1 좌표계로 변환하여 검증
        H1, W1 = image1.shape[:2]
        H2, W2 = image2.shape[:2]
        
        # image2의 네 모서리
        img2_corners = np.array([
            [0, 0],
            [W2 - 1, 0],
            [0, H2 - 1],
            [W2 - 1, H2 - 1]
        ], dtype=np.float32)
        
        # H를 사용하여 image2 corner를 image1 좌표계로 변환
        transformed_corners = apply_homography(img2_corners, H)
        
        # 변환된 corner의 범위가 합리적인지 검증 (완화)
        # 파노라마는 옆으로 확장될 수 있으므로 더 관대한 기준 사용
        max_reasonable_distance = max(W1, H1) * 10  # 5 → 10으로 완화
        
        corner_distances = np.sqrt(transformed_corners[:, 0]**2 + transformed_corners[:, 1]**2)
        if np.any(corner_distances > max_reasonable_distance):
            max_dist = np.max(corner_distances)
            print(f"  Warning: 변환된 corner가 너무 멀리 떨어져 있음 (최대 거리: {max_dist:.1f} > {max_reasonable_distance:.1f}).")
            print(f"    Identity matrix로 대체합니다.")
            return np.eye(3, dtype=np.float32)
        
        # 변환된 corner의 bounding box가 이미지 크기의 몇 배를 넘는지 확인 (완화)
        min_x = np.min(transformed_corners[:, 0])
        max_x = np.max(transformed_corners[:, 0])
        min_y = np.min(transformed_corners[:, 1])
        max_y = np.max(transformed_corners[:, 1])
        
        bbox_width = max_x - min_x
        bbox_height = max_y - min_y
        
        # Bounding box가 원본 이미지 크기의 5배를 넘으면 비정상적 (완화)
        if bbox_width > W1 * 5 or bbox_height > H1 * 5:  # 3 → 5로 완화
            print(f"  Warning: 변환된 corner의 bounding box가 너무 큼 (w: {bbox_width:.1f}, h: {bbox_height:.1f}).")
            print(f"    Identity matrix로 대체합니다.")
            return np.eye(3, dtype=np.float32)
        
        # 11. 추가 검증: Inlier 비율 확인
        if inlier_count < len(matches) * 0.3:  # Inlier 비율이 30% 미만
            print(f"  Warning: Low inlier ratio ({100*inlier_count/len(matches):.1f}% < 30%). Result may be unreliable.")
            # 낮은 inlier 비율은 경고만 출력 (Identity 대체하지 않음)
    
    return H.astype(np.float32)


def compute_all_homographies(images: List[np.ndarray]) -> List[np.ndarray]:
    """
    모든 인접 이미지 쌍에 대해 Homography를 계산합니다.
    
    Args:
        images: 이미지 리스트 [img1, img2, ...], 각 이미지는 (H, W, 3) - uint8
    
    Returns:
        homographies: Homography 행렬 리스트 [H1, H2, ...], 각 H는 (3, 3) - float32
                     H[i]는 images[i+1]을 images[i] 좌표계로 변환
    """
    homographies = []
    identity_count = 0
    
    for i in range(len(images) - 1):
        print(f"이미지 {i+1}와 {i+2} 간의 Homography 계산 중...")
        H = compute_pairwise_homography(images[i], images[i+1])
        
        # Identity Homography 체크
        if np.allclose(H, np.eye(3, dtype=np.float32), atol=1e-6):
            identity_count += 1
            print(f"  Warning: 이미지 {i+1}와 {i+2} 간의 Homography가 Identity입니다.")
            print(f"    이 이미지 쌍은 같은 위치에 배치되어 stacking이 발생할 수 있습니다.")
        
        homographies.append(H)
        print()
    
    if identity_count > 0:
        print(f"총 {identity_count}개의 Identity Homography가 발견되었습니다.")
        print(f"  이는 해당 이미지 쌍들이 제대로 매칭되지 않았음을 의미합니다.")
        print()
    
    return homographies


def main():
    """
    메인 파이프라인 실행 함수
    """
    # 1. 이미지 로드
    # 사용자가 사용할 sampleset 폴더 이름을 여기에 지정하세요
    sampleset_folder_name = "sampleset1"  # sampleset0, sampleset1, sampleset2 등으로 변경 가능
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    img_folder = os.path.join(project_root, "img", sampleset_folder_name)
    
    if not os.path.exists(img_folder):
        print(f"이미지 폴더를 찾을 수 없습니다: {img_folder}")
        print(f"코드에서 sampleset_folder_name을 확인해주세요: {sampleset_folder_name}")
        return
    
    images = load_images_from_folder(img_folder)
    
    if len(images) < 2:
        print("최소 2개의 이미지가 필요합니다.")
        print(f"현재 폴더: {img_folder}")
        return
    
    print(f"로드된 이미지 개수: {len(images)}")
    print()
    
    # 2. 모든 인접 이미지 쌍에 대해 Homography 계산
    homographies = compute_all_homographies(images)
    
    # 3. 이미지 스티칭
    print("파노라마 이미지 생성 중...")
    panorama = stitch_multiple_images(images, homographies)
    
    if panorama is None:
        print("스티칭 실패")
        return
    
    if panorama.size == 0 or panorama.shape[0] == 0 or panorama.shape[1] == 0:
        print("스티칭 실패: panorama 크기가 유효하지 않습니다")
        return
    
    # 4. 결과 저장 및 표시
    output_path = "result.jpg"
    try:
        save_image(panorama, output_path)
        print(f"\n파노라마 이미지가 저장되었습니다: {output_path}")
        print(f"  이미지 크기: {panorama.shape[1]}x{panorama.shape[0]} 픽셀")
    except Exception as e:
        print(f"이미지 저장 중 오류 발생: {e}")
        return
    
    # 이미지 표시
    show_image(panorama, "Panorama")


if __name__ == "__main__":
    main()

