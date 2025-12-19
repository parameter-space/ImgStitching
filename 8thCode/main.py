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
from geometry import ransac_translation
from stitch import stitch_multiple_images, compute_canvas_size
from bundle_adjustment import bundle_adjustment, compute_global_homographies


def compute_pairwise_homography(image1: np.ndarray, image2: np.ndarray, pair_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    두 이미지 간의 Homography를 계산합니다.
    Multi-Scale Homography Estimation을 사용하여 고해상도 이미지에서도 안정적으로 동작합니다.
    
    Args:
        image1: 첫 번째 이미지 (H1, W1, 3) - uint8
        image2: 두 번째 이미지 (H2, W2, 3) - uint8
        pair_idx: 이미지 쌍 인덱스 (디버그 파일명에 사용, 기본값: 0) - int
    
    Returns:
        H: Homography 행렬 (3, 3) - float32 (원본 해상도에 호환)
        inlier_mask: 인라이어 마스크 (N,) - bool
    """
    # 원본 이미지 크기 저장
    H1_orig, W1_orig = image1.shape[:2]
    H2_orig, W2_orig = image2.shape[:2]
    
    # 1. 다운스케일된 이미지 생성 (800px 너비로 고정)
    target_width = 800
    scale_factor1 = target_width / W1_orig
    scale_factor2 = target_width / W2_orig
    
    # 다운스케일된 이미지 크기 계산
    new_width1 = target_width
    new_height1 = int(H1_orig * scale_factor1)
    new_width2 = target_width
    new_height2 = int(H2_orig * scale_factor2)
    
    # 다운스케일된 이미지 생성 (cv2.resize는 I/O가 아닌 이미지 처리이므로 사용 가능)
    image1_low = cv2.resize(image1, (new_width1, new_height1), interpolation=cv2.INTER_AREA)
    image2_low = cv2.resize(image2, (new_width2, new_height2), interpolation=cv2.INTER_AREA)
    
    # 2. 다운스케일된 이미지에서 그레이스케일 변환
    gray1_low = rgb_to_grayscale(image1_low)  # (new_height1, new_width1) - float32
    gray2_low = rgb_to_grayscale(image2_low)  # (new_height2, new_width2) - float32
    
    # 3. 다운스케일된 이미지에서 코너 검출 (저해상도에서 노이즈 감소 및 안정성 향상)
    # threshold를 0.01로 설정하여 노이즈를 필터링하고 강한 코너만 검출
    corners1_low = harris_corner_detection(gray1_low, threshold=0.01, max_corners=5000)  # (N1, 2) - int32
    corners2_low = harris_corner_detection(gray2_low, threshold=0.01, max_corners=5000)  # (N2, 2) - int32
    
    if len(corners1_low) < 4 or len(corners2_low) < 4:
        print(f"경고: 충분한 코너를 찾지 못했습니다. (corners1: {len(corners1_low)}, corners2: {len(corners2_low)})")
        # 기본 Homography 반환
        H = np.eye(3, dtype=np.float32)
        inlier_mask = np.array([], dtype=bool)
        return H, inlier_mask
    
    # 4. 다운스케일된 이미지에서 디스크립터 계산
    # White Wall 문제 해결: patch_size를 7에서 21로 증가
    # 21x21 패치는 흰색 페인트뿐만 아니라 손잡이 가장자리나 문 틈새도 포착하여 특징을 고유하게 만듦
    descriptors1_low = compute_descriptors(gray1_low, corners1_low, patch_size=21)  # (N1, D) - float32
    descriptors2_low = compute_descriptors(gray2_low, corners2_low, patch_size=21)  # (N2, D) - float32
    
    # 5. 다운스케일된 이미지에서 특징점 매칭 (NCC 방식 사용 - 조명 차이에 강건함)
    # White Wall 문제 해결: threshold를 0.95로 증가하여 약간 모호한 매칭도 허용
    # RANSAC이 outlier를 필터링하므로 더 많은 매칭을 허용하고 정확한 매칭만 선택
    matches = match_features(descriptors1_low, descriptors2_low, method='ncc', threshold=0.95)  # List[Tuple[int, int]]
    
    print(f"  Debug: Found {len(matches)} raw matches before RANSAC")
    
    # --- DEBUG VISUALIZATION START ---
    # Visualize matches on the low-res images
    h1, w1 = gray1_low.shape
    h2, w2 = gray2_low.shape
    vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    vis[:h1, :w1] = image1_low
    vis[:h2, w1:w1+w2] = image2_low
    
    # Draw lines for a subset of matches (first 100)
    for idx1, idx2 in matches[:100]:
        pt1 = corners1_low[idx1]
        pt2 = corners2_low[idx2]
        pt2_shifted = (pt2[0] + w1, pt2[1])
        cv2.line(vis, tuple(pt1), tuple(pt2_shifted), (0, 255, 0), 1)
        cv2.circle(vis, tuple(pt1), 4, (0, 0, 255), 1)
        cv2.circle(vis, tuple(pt2_shifted), 4, (0, 0, 255), 1)
    
    cv2.imwrite(f"debug_matches_{pair_idx}_{pair_idx+1}.jpg", vis)
    print(f"  Debug: Saved match visualization to debug_matches_{pair_idx}_{pair_idx+1}.jpg")
    # --- DEBUG VISUALIZATION END ---
    
    if len(matches) < 4:
        print(f"경고: 충분한 매칭을 찾지 못했습니다. (matches: {len(matches)})")
        # 기본 Homography 반환
        H = np.eye(3, dtype=np.float32)
        inlier_mask = np.array([], dtype=bool)
        return H, inlier_mask
    
    # 6. 다운스케일된 이미지에서 대응점 추출
    points1_low, points2_low = get_matched_points(corners1_low, corners2_low, matches)  # (M, 2), (M, 2) - int32
    
    # 7. 대응점을 원본 해상도로 스케일 업
    # scale_factor = target_width / original_width이므로
    # 원본 좌표 = 저해상도 좌표 / scale_factor = 저해상도 좌표 * (original_width / target_width)
    points1_orig = points1_low.astype(np.float32) / scale_factor1  # (M, 2) - float32
    points2_orig = points2_low.astype(np.float32) / scale_factor2  # (M, 2) - float32
    
    # 8. 원본 해상도 좌표로 RANSAC Translation 계산 (Translation-Only 모델, 2-DoF)
    # Translation-only 모델은 매우 robust하며 이미지 왜곡이 없습니다
    # 800px 이미지에서 검출한 특징을 ~4000px로 스케일 업하므로 작은 픽셀 오차가 증폭됨
    # 따라서 threshold를 10.0으로 증가하여 더 큰 허용 오차 설정
    H_1to2, inlier_mask = ransac_translation(
        points1_orig,  # source: image1의 점들 (원본 해상도)
        points2_orig,  # target: image2의 점들 (원본 해상도)
        threshold=10.0  # 증가: 스케일 업으로 인한 픽셀 오차 증폭 고려
    )  # (3, 3) - float32, (M,) - bool
    # H_1to2는 points1_orig을 points2_orig로 변환하는 Translation 행렬 (즉, image1을 image2로 변환)
    
    # --- DEBUG: Visualize RANSAC Inliers ---
    # Draw inliers on the low-res images
    h1, w1 = gray1_low.shape
    h2, w2 = gray2_low.shape
    vis_inliers = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    vis_inliers[:h1, :w1] = image1_low
    vis_inliers[:h2, w1:w1+w2] = image2_low
    
    # Draw lines only for inliers
    # inlier_mask corresponds to points1_orig/points2_orig which has same length as matches
    inlier_count = 0
    if len(inlier_mask) == len(matches):
        for match_idx, (idx1, idx2) in enumerate(matches):
            if inlier_mask[match_idx]:
                pt1 = corners1_low[idx1]
                pt2 = corners2_low[idx2]
                pt2_shifted = (pt2[0] + w1, pt2[1])
                cv2.line(vis_inliers, tuple(pt1), tuple(pt2_shifted), (0, 255, 0), 2)
                cv2.circle(vis_inliers, tuple(pt1), 4, (0, 0, 255), -1)
                cv2.circle(vis_inliers, tuple(pt2_shifted), 4, (0, 0, 255), -1)
                inlier_count += 1
    else:
        print(f"  Warning: inlier_mask length ({len(inlier_mask)}) != matches length ({len(matches)})")
    
    cv2.imwrite(f"debug_inliers_{pair_idx}_{pair_idx+1}.jpg", vis_inliers)
    print(f"  Debug: Saved RANSAC inliers visualization ({inlier_count} inliers) to debug_inliers_{pair_idx}_{pair_idx+1}.jpg")
    
    # Debug: Print number of inliers found
    if inlier_count == 0:
        print(f"  *** RANSAC failed to find any model! ***")
    else:
        print(f"  RANSAC found {inlier_count} inliers out of {len(matches)} matches")
    # --- DEBUG VISUALIZATION END ---
    
    # 9. 스티칭에는 image2를 image1로 변환하는 Translation 행렬이 필요하므로 역행렬 사용
    # Translation 행렬의 역행렬은 간단히 translation을 반대로 한 것입니다
    H = np.linalg.inv(H_1to2)
    # Translation 행렬이므로 항상 정규화되어 있음 (H[2,2] = 1)
    
    # Translation 성분 확인 (디버깅)
    translation_x = H[0, 2]
    translation_y = H[1, 2]
    print(f"  Translation - dx: {translation_x:.2f}, dy: {translation_y:.2f}")
    
    # 10. Translation 검증: Inlier 개수만 확인 (Translation-only 모델이므로 scale/rotation/perspective 체크 불필요)
    actual_inlier_count = np.sum(inlier_mask) if len(inlier_mask) > 0 else 0
    
    if actual_inlier_count < 4:
        # Matching failed completely (white wall 등으로 인해)
        # Better to return Identity Matrix (stack on top) than to guess a direction
        # Guessing ruins the panorama map
        print(f"  *** MATCHING FAILED. Too few inliers ({actual_inlier_count} < 4). Using Identity Matrix. ***")
        H = np.eye(3, dtype=np.float32)
        inlier_mask = np.zeros(len(points1_orig), dtype=bool)
    else:
        # Valid translation found - TRUST IT even if shift is vertical or any direction
        # Do NOT filter based on direction - features know best
        print(f"  Translation validated: {actual_inlier_count} inliers (>= 4)")
        print(f"  Translation direction: dx={translation_x:.2f}, dy={translation_y:.2f}")
    
    # 인라이어 개수 출력
    print(f"  매칭된 점: {len(points1_orig)}, 인라이어: {actual_inlier_count}")
    
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
        H, inlier_mask = compute_pairwise_homography(images[i], images[i+1], pair_idx=i + 1)
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

