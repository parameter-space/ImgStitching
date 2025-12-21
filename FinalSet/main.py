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
from homography import compute_homography_dlt, compute_homography_affine, interpolate_homography, apply_homography
from ransac import ransac_homography
from stitching import stitch_multiple_images
from preprocessing import preprocess_image
from tone_mapping import tone_map_images


def filter_by_spatial_distribution(image1: np.ndarray,
                                   image2: np.ndarray,
                                   corners1: np.ndarray, 
                                   corners2: np.ndarray, 
                                   matches: List[Tuple[int, int]],
                                   grid_size: int = 3) -> List[Tuple[int, int]]:
    if len(matches) < grid_size * grid_size:
        return matches
    
    H1, W1 = image1.shape[:2]
    H2, W2 = image2.shape[:2]
    
    cell_w1 = W1 / grid_size
    cell_h1 = H1 / grid_size
    cell_w2 = W2 / grid_size
    cell_h2 = H2 / grid_size
    
    grid_cells = {}
    
    for i, (idx1, idx2) in enumerate(matches):
        corner1 = corners1[idx1]
        corner2 = corners2[idx2]
        
        cell_col1 = int(corner1[0] / cell_w1)
        cell_row1 = int(corner1[1] / cell_h1)
        cell_col2 = int(corner2[0] / cell_w2)
        cell_row2 = int(corner2[1] / cell_h2)
        
        cell_col1 = min(cell_col1, grid_size - 1)
        cell_row1 = min(cell_row1, grid_size - 1)
        cell_col2 = min(cell_col2, grid_size - 1)
        cell_row2 = min(cell_row2, grid_size - 1)
        
        cell_key = (cell_row1, cell_col1)
        
        if cell_key not in grid_cells:
            grid_cells[cell_key] = []
        grid_cells[cell_key].append(i)
    
    max_per_cell = max(1, len(matches) // (grid_size * grid_size) + 2)
    
    filtered_indices = []
    for cell_key, match_indices in grid_cells.items():
        selected = match_indices[:max_per_cell]
        filtered_indices.extend(selected)
    
    filtered_matches = [matches[i] for i in filtered_indices]
    
    return filtered_matches


def compute_pairwise_homography(image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
    preprocessed1 = preprocess_image(image1, kernel_size=5, sigma=1.0)
    preprocessed2 = preprocess_image(image2, kernel_size=5, sigma=1.0)
    if len(preprocessed1.shape) == 3:
        gray1 = 0.299 * preprocessed1[:, :, 0] + 0.587 * preprocessed1[:, :, 1] + 0.114 * preprocessed1[:, :, 2]
    else:
        gray1 = preprocessed1
    
    if len(preprocessed2.shape) == 3:
        gray2 = 0.299 * preprocessed2[:, :, 0] + 0.587 * preprocessed2[:, :, 1] + 0.114 * preprocessed2[:, :, 2]
    else:
        gray2 = preprocessed2
    
    gray1_norm = normalize_image(gray1)
    gray2_norm = normalize_image(gray2)
    
    corners1 = harris_corner_detection(
        gray1_norm,
        threshold=0.001,
        k=0.04,
        window_size=3,
        sigma=1.0
    )
    
    corners2 = harris_corner_detection(
        gray2_norm,
        threshold=0.001,
        k=0.04,
        window_size=3,
        sigma=1.0
    )
    
    print(f"  Image 1: {len(corners1)} corners detected")
    print(f"  Image 2: {len(corners2)} corners detected")
    
    if len(corners1) < 4 or len(corners2) < 4:
        print(f"  Warning: Too few corners. Using identity matrix.")
        return np.eye(3, dtype=np.float32)
    
    descriptors1 = compute_descriptors(gray1_norm, corners1, patch_size=21)
    descriptors2 = compute_descriptors(gray2_norm, corners2, patch_size=21)
    
    matches = match_features(descriptors1, descriptors2, threshold=0.75)
    print(f"  Matched features: {len(matches)}")
    
    if len(matches) < 4:
        print(f"  Warning: Too few matches ({len(matches)} < 4). Using identity matrix.")
        return np.eye(3, dtype=np.float32)
    
    points1, points2 = get_matched_points(corners1, corners2, matches)
    
    min_inliers = max(4, int(len(matches) * 0.3))
    
    H_1to2, inlier_mask = ransac_homography(
        points1, points2,
        max_iterations=2000,
        threshold=4.0,
        min_inliers=min_inliers
    )
    
    inlier_count = np.sum(inlier_mask)
    print(f"  RANSAC inliers: {inlier_count}/{len(matches)} ({100*inlier_count/len(matches):.1f}%)")
    
    if inlier_count < 4:
        print(f"  Warning: Low inlier count ({inlier_count} < 4). Result may be unreliable.")
    
    # 적응형 하이브리드 전략: Scale과 Perspective에 따라 Projective와 Affine 보간
    det = H_1to2[0, 0] * H_1to2[1, 1] - H_1to2[0, 1] * H_1to2[1, 0]
    if det <= 1e-10:
        current_scale = 0.0
    else:
        current_scale = np.sqrt(det)
    
    if abs(H_1to2[2, 2]) > 1e-10:
        H_norm = H_1to2 / H_1to2[2, 2]
    else:
        H_norm = H_1to2
    persp_val = abs(H_norm[2, 0]) + abs(H_norm[2, 1])
    
    # Scale 1.1 이하면 Homography (0.0), Scale 2.0 이상이면 Affine (1.0)
    if current_scale <= 1.1:
        scale_alpha = 0.0
    elif current_scale >= 2.0:
        scale_alpha = 1.0
    else:
        scale_alpha = (current_scale - 1.1) / (2.0 - 1.1)
    
    if current_scale < 0.5:
        if current_scale <= 0.0:
            scale_alpha = 1.0
        else:
            scale_alpha = (0.5 - current_scale) / 0.5
    
    if persp_val <= 0.0005:
        persp_alpha = 0.0
    elif persp_val >= 0.001:
        persp_alpha = 1.0
    else:
        persp_alpha = (persp_val - 0.0005) / (0.001 - 0.0005)
    
    alpha = max(scale_alpha, persp_alpha)
    
    if inlier_count < 15:
        alpha = max(alpha, 0.8)
    if alpha >= 1.0:
        stage = "Critical"
        reason = f"Full Affine (scale={current_scale:.3f}, persp={persp_val:.5f})"
    elif alpha >= 0.5:
        stage = "Warning"
        reason = f"High Affine Mix (alpha={alpha:.2f}, scale={current_scale:.3f})"
    elif alpha > 0.0:
        stage = "Warning"
        reason = f"Moderate Affine Mix (alpha={alpha:.2f}, scale={current_scale:.3f})"
    else:
        stage = "Safe"
        reason = f"Pure Homography (scale={current_scale:.3f})"
    
    if alpha > 0.0:
        print(f"  Adaptive Strategy [{stage}]: {reason}. Alpha = {alpha:.2f}")
        
        points1_inliers = points1[inlier_mask]
        points2_inliers = points2[inlier_mask]
        
        if len(points1_inliers) >= 3:
            try:
                H_affine = compute_homography_affine(points1_inliers, points2_inliers)
                H_1to2 = interpolate_homography(H_1to2, H_affine, alpha)
                print(f"    -> Applied Hybrid Matrix (Projective + Affine, alpha={alpha:.2f})")
                
            except Exception as e:
                print(f"    Affine computation failed: {e}. Using original Projective H.")
        else:
            print(f"    Not enough inliers for Affine ({len(points1_inliers)} < 3). Using original Projective H.")
    det = np.linalg.det(H_1to2)
    if abs(det) < 1e-5:
        print(f"  Warning: Homography determinant too small ({det:.2e}). Using identity matrix.")
        H = np.eye(3, dtype=np.float32)
    else:
        H = np.linalg.inv(H_1to2)
        if abs(H[2, 2]) > 1e-10:
            H = H / H[2, 2]
        
        scale_x = np.sqrt(H[0, 0]**2 + H[0, 1]**2)
        scale_y = np.sqrt(H[1, 0]**2 + H[1, 1]**2)
        
        if scale_x < 0.01 or scale_x > 100.0 or scale_y < 0.01 or scale_y > 100.0:
            print(f"  Warning: Homography scale factor out of range (x: {scale_x:.2f}, y: {scale_y:.2f}).")
            print(f"    Identity matrix로 대체합니다.")
            return np.eye(3, dtype=np.float32)
        
        H1, W1 = image1.shape[:2]
        H2, W2 = image2.shape[:2]
        
        img2_corners = np.array([
            [0, 0],
            [W2 - 1, 0],
            [0, H2 - 1],
            [W2 - 1, H2 - 1]
        ], dtype=np.float32)
        
        transformed_corners = apply_homography(img2_corners, H)
        
        max_reasonable_distance = max(W1, H1) * 10
        
        corner_distances = np.sqrt(transformed_corners[:, 0]**2 + transformed_corners[:, 1]**2)
        if np.any(corner_distances > max_reasonable_distance):
            max_dist = np.max(corner_distances)
            print(f"  Warning: 변환된 corner가 너무 멀리 떨어져 있음 (최대 거리: {max_dist:.1f} > {max_reasonable_distance:.1f}).")
            print(f"    Identity matrix로 대체합니다.")
            return np.eye(3, dtype=np.float32)
        
        min_x = np.min(transformed_corners[:, 0])
        max_x = np.max(transformed_corners[:, 0])
        min_y = np.min(transformed_corners[:, 1])
        max_y = np.max(transformed_corners[:, 1])
        
        bbox_width = max_x - min_x
        bbox_height = max_y - min_y
        
        if bbox_width > W1 * 50 or bbox_height > H1 * 50:
            print(f"  Warning: 변환된 corner의 bounding box가 극도로 큼 (w: {bbox_width:.1f}, h: {bbox_height:.1f}).")
            print(f"    이는 Homography 계산 오류를 의미할 수 있습니다.")
            print(f"    Identity matrix로 대체합니다.")
            return np.eye(3, dtype=np.float32)
        
        if inlier_count < len(matches) * 0.3:
            print(f"  Warning: Low inlier ratio ({100*inlier_count/len(matches):.1f}% < 30%). Result may be unreliable.")
    
    return H.astype(np.float32)


def compute_all_homographies(images: List[np.ndarray]) -> List[np.ndarray]:
    homographies = []
    identity_count = 0
    
    for i in range(len(images) - 1):
        print(f"이미지 {i+1}와 {i+2} 간의 Homography 계산 중...")
        H = compute_pairwise_homography(images[i], images[i+1])
        
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
    sampleset_folder_name = "sampleset1"
    
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
    
    print("Tone Mapping 적용 중...")
    tone_mapping_method = "histogram"
    images_tone_mapped = tone_map_images(images, method=tone_mapping_method)
    
    images_for_processing = []
    for img_float in images_tone_mapped:
        img_uint8 = (np.clip(img_float, 0.0, 1.0) * 255.0).astype(np.uint8)
        images_for_processing.append(img_uint8)
    
    print(f"Tone Mapping 완료 (방법: {tone_mapping_method})")
    print()
    
    homographies = compute_all_homographies(images_for_processing)
    
    print("파노라마 이미지 생성 중...")
    panorama = stitch_multiple_images(images_for_processing, homographies)
    
    if panorama is None:
        print("스티칭 실패")
        return
    
    if panorama.size == 0 or panorama.shape[0] == 0 or panorama.shape[1] == 0:
        print("스티칭 실패: panorama 크기가 유효하지 않습니다")
        return
    output_path = "result.jpg"
    try:
        save_image(panorama, output_path)
        print(f"\n파노라마 이미지가 저장되었습니다: {output_path}")
        print(f"  이미지 크기: {panorama.shape[1]}x{panorama.shape[0]} 픽셀")
    except Exception as e:
        print(f"이미지 저장 중 오류 발생: {e}")
        return
    
    show_image(panorama, "Panorama")


if __name__ == "__main__":
    main()

