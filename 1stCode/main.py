"""
파노라마 이미지 스티칭 메인 파이프라인
전체 프로세스를 실행합니다.
"""

import numpy as np
import cv2
import os
from typing import List, Tuple

from utils import load_images_from_folder, rgb_to_grayscale, show_image, save_image
from features import harris_corner_detection, compute_descriptors, match_features, get_matched_points
from geometry import ransac_homography
from stitch import stitch_multiple_images


def compute_pairwise_homography(image1: np.ndarray, image2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    두 이미지 간의 Homography를 계산합니다.
    
    Args:
        image1: 첫 번째 이미지 (H1, W1, 3) - uint8
        image2: 두 번째 이미지 (H2, W2, 3) - uint8
    
    Returns:
        H: Homography 행렬 (3, 3) - float32
        inlier_mask: 인라이어 마스크 (N,) - bool
    """
    # 1. 그레이스케일 변환
    gray1 = rgb_to_grayscale(image1)  # (H1, W1) - float32
    gray2 = rgb_to_grayscale(image2)  # (H2, W2) - float32
    
    # 2. 코너 검출 (최대 개수 제한으로 메모리 사용량 제어)
    corners1 = harris_corner_detection(gray1, threshold=0.01, max_corners=5000)  # (N1, 2) - int32
    corners2 = harris_corner_detection(gray2, threshold=0.01, max_corners=5000)  # (N2, 2) - int32
    
    if len(corners1) < 4 or len(corners2) < 4:
        print(f"경고: 충분한 코너를 찾지 못했습니다. (corners1: {len(corners1)}, corners2: {len(corners2)})")
        # 기본 Homography 반환
        H = np.eye(3, dtype=np.float32)
        inlier_mask = np.array([], dtype=bool)
        return H, inlier_mask
    
    # 3. 디스크립터 계산
    descriptors1 = compute_descriptors(gray1, corners1, patch_size=7)  # (N1, D) - float32
    descriptors2 = compute_descriptors(gray2, corners2, patch_size=7)  # (N2, D) - float32
    
    # 4. 특징점 매칭
    matches = match_features(descriptors1, descriptors2, method='ssd', threshold=0.7)  # List[Tuple[int, int]]
    
    if len(matches) < 4:
        print(f"경고: 충분한 매칭을 찾지 못했습니다. (matches: {len(matches)})")
        # 기본 Homography 반환
        H = np.eye(3, dtype=np.float32)
        inlier_mask = np.array([], dtype=bool)
        return H, inlier_mask
    
    # 5. 대응점 추출
    points1, points2 = get_matched_points(corners1, corners2, matches)  # (M, 2), (M, 2) - int32
    
    # 6. RANSAC으로 Homography 계산
    H, inlier_mask = ransac_homography(
        points1.astype(np.float32), 
        points2.astype(np.float32),
        max_iterations=1000,
        threshold=5.0,
        min_inliers=10
    )  # (3, 3) - float32, (M,) - bool
    
    # 인라이어 개수 출력 (마스크 크기 확인)
    actual_inlier_count = np.sum(inlier_mask) if len(inlier_mask) > 0 else 0
    print(f"  매칭된 점: {len(points1)}, 인라이어: {actual_inlier_count}")
    
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
                     H[i]는 images[i+1]을 images[i] 좌표계로 변환
    """
    homographies = []
    
    # 인접한 이미지 쌍에 대해 Homography 계산
    for i in range(len(images) - 1):
        print(f"이미지 {i+1}와 {i+2} 간의 Homography 계산 중...")
        H, inlier_mask = compute_pairwise_homography(images[i], images[i+1])
        homographies.append(H)
    
    return homographies


def main():
    """
    메인 파이프라인 실행 함수
    """
    # 1. 이미지 로드
    # FirstImplementation 폴더에서 상위 폴더의 img 폴더 참조
    img_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "img/sampleset0")
    images = load_images_from_folder(img_folder)  # List[np.ndarray], 각 (H, W, 3) - uint8
    
    if len(images) < 2:
        print("최소 2개 이상의 이미지가 필요합니다.")
        return
    
    print(f"로드된 이미지 개수: {len(images)}")
    
    # 2. 모든 Homography 계산
    homographies = compute_all_homographies(images)  # List[np.ndarray], 각 (3, 3) - float32
    
    if len(homographies) != len(images) - 1:
        print("Homography 계산 실패")
        return
    
    print(f"계산된 Homography 개수: {len(homographies)}")
    
    # 3. 이미지 스티칭
    panorama = stitch_multiple_images(images, homographies)  # (H_out, W_out, 3) - uint8
    
    # 4. 결과 저장 및 표시
    output_path = "result.jpg"
    save_image(panorama, output_path)
    print(f"파노라마 이미지가 저장되었습니다: {output_path}")
    
    show_image(panorama, "Panorama")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

