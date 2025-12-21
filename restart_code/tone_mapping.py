"""
Tone Mapping 구현
RGB 채널별 독립 정규화를 통한 밝기 통일
"""

import numpy as np
from typing import List


def normalize_image_brightness_rgb(images: List[np.ndarray]) -> List[np.ndarray]:
    """
    RGB 채널별 독립 정규화 (평균 + 분산)를 통해 모든 이미지의 tone을 통일합니다.
    
    알고리즘:
    1. 각 이미지의 각 채널(R, G, B) 평균과 표준편차 계산
    2. 전체 이미지들의 각 채널 평균과 평균 분산 계산
    3. Z-score 정규화를 통해 평균과 분산 모두 맞춤
    4. 모든 이미지의 tone이 통일됨
    
    수식 (Z-score 정규화):
    - 각 이미지 i의 채널별 통계: μ_R_i, σ_R_i, μ_G_i, σ_G_i, μ_B_i, σ_B_i
    - 전체 평균 및 평균 분산: μ_R_global, σ_R_global, μ_G_global, σ_G_global, μ_B_global, σ_B_global
    - 정규화: image_i_new[:, :, 0] = (image_i[:, :, 0] - μ_R_i) * (σ_R_global / σ_R_i) + μ_R_global
              image_i_new[:, :, 1] = (image_i[:, :, 1] - μ_G_i) * (σ_G_global / σ_G_i) + μ_G_global
              image_i_new[:, :, 2] = (image_i[:, :, 2] - μ_B_i) * (σ_B_global / σ_B_i) + μ_B_global
    
    Args:
        images: 이미지 리스트 [(H, W, 3), ...] - uint8
    
    Returns:
        normalized_images: 정규화된 이미지 리스트 - uint8
    """
    N = len(images)
    if N == 0:
        return images
    
    print("=" * 60)
    print("Tone Mapping: RGB 채널별 독립 정규화 (평균 + 분산)")
    print("=" * 60)
    
    # Step 1: 각 이미지의 채널별 평균과 표준편차 계산
    channel_stats = []  # [(μ_R, σ_R, μ_G, σ_G, μ_B, σ_B), ...]
    
    for i, img in enumerate(images):
        # uint8 -> float32 변환 (정밀도 향상)
        img_float = img.astype(np.float32)
        
        # 각 채널의 평균과 표준편차
        mean_R = np.mean(img_float[:, :, 0])
        std_R = np.std(img_float[:, :, 0])
        mean_G = np.mean(img_float[:, :, 1])
        std_G = np.std(img_float[:, :, 1])
        mean_B = np.mean(img_float[:, :, 2])
        std_B = np.std(img_float[:, :, 2])
        
        channel_stats.append((mean_R, std_R, mean_G, std_G, mean_B, std_B))
        print(f"  Image {i+1}: R(μ={mean_R:.1f}, σ={std_R:.1f}), "
              f"G(μ={mean_G:.1f}, σ={std_G:.1f}), B(μ={mean_B:.1f}, σ={std_B:.1f})")
    
    # Step 2: 전체 이미지들의 채널별 평균과 평균 분산 계산
    mean_R_global = np.mean([s[0] for s in channel_stats])
    std_R_global = np.mean([s[1] for s in channel_stats])
    mean_G_global = np.mean([s[2] for s in channel_stats])
    std_G_global = np.mean([s[3] for s in channel_stats])
    mean_B_global = np.mean([s[4] for s in channel_stats])
    std_B_global = np.mean([s[5] for s in channel_stats])
    
    print(f"  Global stats: R(μ={mean_R_global:.1f}, σ={std_R_global:.1f}), "
          f"G(μ={mean_G_global:.1f}, σ={std_G_global:.1f}), "
          f"B(μ={mean_B_global:.1f}, σ={std_B_global:.1f})")
    print()
    
    # Step 3: Z-score 정규화 (평균 + 분산)
    normalized_images = []
    
    for i, img in enumerate(images):
        img_float = img.astype(np.float32)
        mean_R_i, std_R_i, mean_G_i, std_G_i, mean_B_i, std_B_i = channel_stats[i]
        
        # Z-score 정규화 계수 (0으로 나누기 방지)
        epsilon = 1e-6
        scale_R = std_R_global / max(std_R_i, epsilon)
        scale_G = std_G_global / max(std_G_i, epsilon)
        scale_B = std_B_global / max(std_B_i, epsilon)
        
        # Z-score 정규화: (x - μ) * (σ_global / σ) + μ_global
        img_normalized = img_float.copy()
        
        # R 채널
        img_normalized[:, :, 0] = (img_float[:, :, 0] - mean_R_i) * scale_R + mean_R_global
        
        # G 채널
        img_normalized[:, :, 1] = (img_float[:, :, 1] - mean_G_i) * scale_G + mean_G_global
        
        # B 채널
        img_normalized[:, :, 2] = (img_float[:, :, 2] - mean_B_i) * scale_B + mean_B_global
        
        # 클램핑 (0~255 범위 유지)
        img_normalized = np.clip(img_normalized, 0, 255)
        
        # uint8로 변환
        normalized_images.append(img_normalized.astype(np.uint8))
        
        # 정규화 후 통계 확인 (검증용)
        mean_R_new = np.mean(normalized_images[i][:, :, 0].astype(np.float32))
        std_R_new = np.std(normalized_images[i][:, :, 0].astype(np.float32))
        mean_G_new = np.mean(normalized_images[i][:, :, 1].astype(np.float32))
        std_G_new = np.std(normalized_images[i][:, :, 1].astype(np.float32))
        mean_B_new = np.mean(normalized_images[i][:, :, 2].astype(np.float32))
        std_B_new = np.std(normalized_images[i][:, :, 2].astype(np.float32))
        
        print(f"  Image {i+1} normalized: scale_R={scale_R:.3f}, scale_G={scale_G:.3f}, scale_B={scale_B:.3f}")
        print(f"    After: R(μ={mean_R_new:.1f}, σ={std_R_new:.1f}), "
              f"G(μ={mean_G_new:.1f}, σ={std_G_new:.1f}), "
              f"B(μ={mean_B_new:.1f}, σ={std_B_new:.1f})")
    
    print("=" * 60)
    print()
    
    return normalized_images

