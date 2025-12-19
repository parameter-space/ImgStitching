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
from homography import compute_homography_dlt
from stitching import stitch_multiple_images


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
    
    # 5. Feature Matching
    matches = match_features(descriptors1, descriptors2, threshold=0.7)
    print(f"  Matched features: {len(matches)}")
    
    if len(matches) < 4:
        print(f"  Warning: Too few matches ({len(matches)} < 4). Using identity matrix.")
        return np.eye(3, dtype=np.float32)
    
    # 6. 매칭된 점 좌표 추출
    points1, points2 = get_matched_points(corners1, corners2, matches)
    
    # 7. Homography 계산 (image1 -> image2 변환)
    H_1to2 = compute_homography_dlt(points1, points2)
    
    # 8. 스티칭에는 image2를 image1로 변환하는 Homography가 필요하므로 역행렬 사용
    det = np.linalg.det(H_1to2)
    if abs(det) < 1e-5:
        print(f"  Warning: Homography determinant too small ({det:.2e}). Using identity matrix.")
        H = np.eye(3, dtype=np.float32)
    else:
        H = np.linalg.inv(H_1to2)
        if abs(H[2, 2]) > 1e-10:
            H = H / H[2, 2]  # 정규화
    
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
    
    for i in range(len(images) - 1):
        print(f"이미지 {i+1}와 {i+2} 간의 Homography 계산 중...")
        H = compute_pairwise_homography(images[i], images[i+1])
        homographies.append(H)
        print()
    
    return homographies


def main():
    """
    메인 파이프라인 실행 함수
    """
    # 1. 이미지 로드
    # testing1.jpg ~ testing10.jpg를 로컬 폴더에서 로드
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    img_folder = os.path.join(project_root, "img", "sampleset1")
    
    # 만약 sampleset1 폴더가 없으면 현재 디렉토리에서 찾기
    if not os.path.exists(img_folder):
        img_folder = current_dir
    
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

