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
from geometry import ransac_homography, validate_homography
from stitch import stitch_multiple_images, compute_canvas_size
from bundle_adjustment import bundle_adjustment, compute_global_homographies


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
    
    # 4. 특징점 매칭 (NCC 방식 사용 - 조명 차이에 강건함)
    # NCC는 -1에서 1 사이의 값이므로 threshold를 0.8로 설정 (높을수록 더 엄격한 매칭)
    matches = match_features(descriptors1, descriptors2, method='ncc', threshold=0.8)  # List[Tuple[int, int]]
    
    if len(matches) < 4:
        print(f"경고: 충분한 매칭을 찾지 못했습니다. (matches: {len(matches)})")
        # 기본 Homography 반환
        H = np.eye(3, dtype=np.float32)
        inlier_mask = np.array([], dtype=bool)
        return H, inlier_mask
    
    # 5. 대응점 추출
    points1, points2 = get_matched_points(corners1, corners2, matches)  # (M, 2), (M, 2) - int32
    
    # 6. RANSAC으로 Homography 계산 (개선된 파라미터)
    # ransac_homography(points1, points2)는 points1을 points2로 변환하는 Homography를 계산
    # 스티칭에는 image2를 image1 좌표계로 변환하는 Homography가 필요
    # 따라서 points1을 points2로 변환하는 Homography를 계산한 후 역행렬을 사용
    H_1to2, inlier_mask = ransac_homography(
        points1.astype(np.float32),  # source: image1의 점들
        points2.astype(np.float32),  # target: image2의 점들
        max_iterations=2000,  # 충분한 반복
        threshold=5.0,        # RANSAC threshold를 5.0으로 설정
        min_inliers=20        # 증가: 더 많은 인라이어 필요
    )  # (3, 3) - float32, (M,) - bool
    # H_1to2는 points1을 points2로 변환하는 Homography (즉, image1을 image2로 변환)
    
    # 스티칭에는 image2를 image1로 변환하는 Homography가 필요하므로 역행렬 사용
    # 역행렬 계산 전에 determinant 체크 (너무 작으면 Identity 반환)
    det = np.linalg.det(H_1to2)
    if abs(det) < 1e-5:
        print(f"  경고: Homography determinant가 너무 작습니다 ({det:.2e}). Identity 행렬 사용.")
        H = np.eye(3, dtype=np.float32)
    else:
        H = np.linalg.inv(H_1to2)
        if H[2, 2] != 0:
            H = H / H[2, 2]  # 정규화
    
    # Homography의 translation 성분 확인 (디버깅)
    translation_x = H[0, 2]
    translation_y = H[1, 2]
    scale_x = np.sqrt(H[0, 0]**2 + H[0, 1]**2)
    scale_y = np.sqrt(H[1, 0]**2 + H[1, 1]**2)
    print(f"  Homography - translation: ({translation_x:.2f}, {translation_y:.2f}), scale: ({scale_x:.2f}, {scale_y:.2f})")
    
    # 7. Homography 검증
    # H는 points2를 points1로 변환하므로 validate_homography(points2, points1, H)로 호출
    is_valid, mean_error = validate_homography(H, points2.astype(np.float32), points1.astype(np.float32))
    
    if not is_valid:
        print(f"  경고: Homography 검증 실패 (평균 오차: {mean_error:.2f} 픽셀)")
        # 검증 실패 시 단위 행렬 반환
        H = np.eye(3, dtype=np.float32)
        inlier_mask = np.zeros(len(points1), dtype=bool)
    
    # 인라이어 개수 및 재투영 오차 출력
    actual_inlier_count = np.sum(inlier_mask) if len(inlier_mask) > 0 else 0
    print(f"  매칭된 점: {len(points1)}, 인라이어: {actual_inlier_count}, 평균 재투영 오차: {mean_error:.2f} 픽셀")
    
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
    img_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "img/sampleset1")
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
    
    # 3. Bundle Adjustment 우회 (수치적 불안정성 문제로 인해 임시로 비활성화)
    # Bundle Adjustment 단계를 우회하고 직접 초기 Homography 사용
    # print("\n=== Bundle Adjustment 시작 ===")
    # global_homographies = bundle_adjustment(
    #     images,
    #     homographies,
    #     compute_pairwise_homography,
    #     max_iterations=3  # 반복 횟수
    # )
    # print("=== Bundle Adjustment 완료 ===\n")
    # 
    # # 4. 전역 Homography를 인접 쌍 Homography로 변환 (stitch_multiple_images 호환)
    # # global_homographies는 [H0, H1, H2, ...] 형태 (H0=I)
    # # H[i]는 images[i]를 images[0] 좌표계로 변환
    # # stitch_multiple_images는 인접 쌍 Homography [H1, H2, ...] 형태를 기대
    # # H[i]는 images[i+1]을 images[i] 좌표계로 변환해야 함
    # pairwise_from_global = []
    # for i in range(len(global_homographies) - 1):
    #     H_i = global_homographies[i]  # images[i] -> images[0]
    #     H_j = global_homographies[i + 1]  # images[i+1] -> images[0]
    #     
    #     # images[i+1]을 images[i]로 변환하는 Homography 계산
    #     # images[i+1] -> images[0] (H_j) -> images[i] (H_i^-1)
    #     # 따라서 H_pairwise = H_i^-1 @ H_j
    #     H_i_inv = np.linalg.inv(H_i)
    #     H_pairwise = H_i_inv @ H_j
    #     
    #     # 정규화
    #     if H_pairwise[2, 2] != 0:
    #         H_pairwise = H_pairwise / H_pairwise[2, 2]
    #     
    #     # Homography 검증 (너무 큰 변환이면 원본 사용)
    #     det = np.linalg.det(H_pairwise)
    #     scale_x = np.sqrt(H_pairwise[0, 0]**2 + H_pairwise[0, 1]**2)
    #     scale_y = np.sqrt(H_pairwise[1, 0]**2 + H_pairwise[1, 1]**2)
    #     
    #     if abs(det) < 1e-6 or scale_x < 0.1 or scale_x > 10.0 or scale_y < 0.1 or scale_y > 10.0:
    #         print(f"  경고: 변환된 Homography {i}가 유효하지 않음. 원본 사용.")
    #         # 원본 Homography 사용
    #         if i < len(homographies):
    #             H_pairwise = homographies[i]
    #         else:
    #             H_pairwise = np.eye(3, dtype=np.float32)
    #     
    #     pairwise_from_global.append(H_pairwise)
    
    # 4. 이미지 스티칭 (초기 Homography 직접 사용)
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

