"""
파노라마 이미지 스티칭 메인 파이프라인
전체 프로세스를 실행합니다.

필수 구현 요소:
1. 코너 포인트 찾기 (corner_detection.py)
2. Point Matching (point_matching.py)
3. Homography 계산 (homography.py) - DLT 알고리즘 사용
4. Stitching (stitching.py)
"""

import numpy as np
import cv2
import os
from typing import List, Tuple

from utils import load_images_from_folder, rgb_to_grayscale, save_image, show_image
from corner_detection import harris_corner_detection
from point_matching import compute_descriptors, match_features, get_matched_points
from homography import compute_homography_dlt
from stitching import stitch_multiple_images


def compute_pairwise_homography(image1: np.ndarray, image2: np.ndarray, pair_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    두 이미지 간의 Homography를 계산합니다.
    Multi-Scale Homography Estimation을 사용하여 고해상도 이미지에서도 안정적으로 동작합니다.
    DLT 알고리즘으로 직접 Homography를 계산합니다 (RANSAC 미사용 - 필수 구현만 포함).
    
    Args:
        image1: 첫 번째 이미지 (H1, W1, 3) - uint8
        image2: 두 번째 이미지 (H2, W2, 3) - uint8
        pair_idx: 이미지 쌍 인덱스 (기본값: 0) - int (호환성을 위해 유지)
    
    Returns:
        H: Homography 행렬 (3, 3) - float32 (원본 해상도에 호환)
        inlier_mask: 인라이어 마스크 (N,) - bool (호환성을 위해 반환, 모든 점이 inlier로 설정됨)
    """
    # 원본 이미지 크기 저장
    H1_orig, W1_orig = image1.shape[:2]
    H2_orig, W2_orig = image2.shape[:2]
    
    # 1. 다운스케일된 이미지 생성 (800px 너비로 고정)
    # Multi-Scale Homography Estimation: 고해상도 이미지에서도 빠르게 작업하기 위해
    # Feature Detection과 Matching은 저해상도에서, Homography 계산은 원본 해상도에서 수행
    target_width = 800
    scale_factor1 = target_width / W1_orig  # 첫 번째 이미지의 스케일 비율
    scale_factor2 = target_width / W2_orig  # 두 번째 이미지의 스케일 비율
    
    new_width1 = target_width
    new_height1 = int(H1_orig * scale_factor1)  # 가로세로 비율 유지
    new_width2 = target_width
    new_height2 = int(H2_orig * scale_factor2)
    
    # 다운스케일된 이미지 생성
    # cv2.INTER_AREA: 이미지 축소에 최적화된 보간 방법 (선명도 유지)
    image1_low = cv2.resize(image1, (new_width1, new_height1), interpolation=cv2.INTER_AREA)
    image2_low = cv2.resize(image2, (new_width2, new_height2), interpolation=cv2.INTER_AREA)
    
    # 2. 그레이스케일 변환
    gray1_low = rgb_to_grayscale(image1_low)
    gray2_low = rgb_to_grayscale(image2_low)
    
    # 3. 코너 포인트 찾기 (Harris Corner Detection) - 필수 구현
    corners1_low = harris_corner_detection(
        gray1_low,
        threshold=0.01,
        k=0.04,
        window_size=3,
        nms_window_size=3,
        use_gaussian=True,
        sigma=1.0,
        max_corners=5000
    )
    
    corners2_low = harris_corner_detection(
        gray2_low,
        threshold=0.01,
        k=0.04,
        window_size=3,
        nms_window_size=3,
        use_gaussian=True,
        sigma=1.0,
        max_corners=5000
    )
    
    print(f"  Image 1: {len(corners1_low)} corners detected")
    print(f"  Image 2: {len(corners2_low)} corners detected")
    
    # 4. 디스크립터 계산
    print(f"  Computing descriptors for Image 1...")
    descriptors1_low = compute_descriptors(gray1_low, corners1_low, patch_size=21)
    print(f"  Computing descriptors for Image 2...")
    descriptors2_low = compute_descriptors(gray2_low, corners2_low, patch_size=21)
    
    # 5. Feature Matching (Point Matching) - 필수 구현
    print(f"  Matching features (this may take a while for {len(corners1_low)} x {len(corners2_low)} = {len(corners1_low) * len(corners2_low)} comparisons)...")
    matches = match_features(descriptors1_low, descriptors2_low, method='ncc', threshold=0.95)
    print(f"  Matched features: {len(matches)}")
    
    if len(matches) < 4:
        print(f"  *** WARNING: Too few matches ({len(matches)} < 4). Cannot compute Homography. ***")
        return np.eye(3, dtype=np.float32), np.array([], dtype=bool)
    
    # 6. 매칭된 점 좌표 추출 (저해상도)
    points1_low, points2_low = get_matched_points(corners1_low, corners2_low, matches)
    
    # 원본 해상도로 스케일 업: 저해상도에서 찾은 매칭점을 원본 해상도로 변환
    # Homography 계산은 원본 해상도 좌표로 수행해야 정확함
    points1_orig = points1_low.astype(np.float32) / scale_factor1
    points2_orig = points2_low.astype(np.float32) / scale_factor2
    
    # 7. Homography 계산 (DLT 알고리즘) - 필수 구현
    # 모든 매칭된 점을 사용하여 직접 DLT로 Homography 계산
    # H_1to2: image1을 image2로 변환하는 Homography
    H_1to2 = compute_homography_dlt(points1_orig, points2_orig)
    
    # 8. 스티칭에는 image2를 image1로 변환하는 Homography가 필요하므로 역행렬 사용
    # compute_homography_dlt는 image1->image2 변환을 계산했지만, 스티칭은 image2->image1이 필요
    det = np.linalg.det(H_1to2)  # 행렬식 계산 (0에 가까우면 역행렬 계산 불가)
    if abs(det) < 1e-5:
        print(f"  *** WARNING: Homography determinant가 너무 작습니다 ({det:.2e}). Using Identity Matrix. ***")
        H = np.eye(3, dtype=np.float32)  # 역행렬 계산 불가 시 Identity 사용 (변환 없음)
    else:
        H = np.linalg.inv(H_1to2)  # 역행렬 계산
        if abs(H[2, 2]) > 1e-10:
            H = H / H[2, 2]  # 정규화 (H[2,2]=1로 만들어 스케일 불변)
    
    # inlier_mask는 사용하지 않으므로 빈 배열 반환 (호환성을 위해)
    inlier_mask = np.ones(len(points1_orig), dtype=bool)
    
    return H, inlier_mask


def compute_all_homographies(images: List[np.ndarray]) -> List[np.ndarray]:
    """
    모든 인접 이미지 쌍에 대해 Homography를 계산합니다.
    
    Args:
        images: 이미지 리스트 [img1, img2, ...]
                각 이미지는 (H, W, 3) 형태 - uint8
    
    Returns:
        homographies: Homography 행렬 리스트 [H1, H2, ...]
                     각 H는 (3, 3) 형태 - float32
                     H[i]는 images[i+1]을 images[i] 좌표계로 변환 (forward homography)
    """
    homographies = []
    
    # 인접한 이미지 쌍에 대해 Homography 계산
    for i in range(len(images) - 1):
        print(f"이미지 {i+1}와 {i+2} 간의 Homography 계산 중...")
        H, inlier_mask = compute_pairwise_homography(images[i], images[i+1], pair_idx=i)
        homographies.append(H)
    
    return homographies


def main():
    """
    메인 파이프라인 실행 함수
    """
    # 1. 이미지 로드
    img_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "img/sampleset1")
    images = load_images_from_folder(img_folder)
    
    if len(images) < 2:
        print("최소 2개의 이미지가 필요합니다.")
        return
    
    print(f"로드된 이미지 개수: {len(images)}")
    
    # 2. 모든 인접 이미지 쌍에 대해 Homography 계산 - 필수 구현
    homographies = compute_all_homographies(images)
    
    # 3. 이미지 스티칭 - 필수 구현
    panorama = stitch_multiple_images(images, homographies)
    
    if panorama is None:
        print("스티칭 실패")
        return
    
    # 추가 유효성 검사: panorama가 빈 배열이거나 크기가 0인지 확인
    if panorama.size == 0:
        print("스티칭 실패: panorama 크기가 0입니다")
        return
    
    if panorama.shape[0] == 0 or panorama.shape[1] == 0:
        print(f"스티칭 실패: panorama 크기가 유효하지 않습니다 (shape: {panorama.shape})")
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
    
    # 이미지 표시 (유효성 검사는 show_image 내부에서 수행)
    show_image(panorama, "Panorama")


if __name__ == "__main__":
    main()

