"""
Tone Mapping 구현
파노라마 스티칭에서 여러 이미지의 노출 차이를 조정하여 자연스러운 블렌딩을 만듭니다.

구현 의도: 서로 다른 시점에서 촬영된 이미지들은 조명 차이나 노출 차이 등으로 인해 
각기 다른 밝기와 색조를 가집니다. 이를 보정하지 않고 스티칭할 경우 연결 부위에서 
급격한 밝기 차이가 발생하므로, 전체 이미지의 톤을 통일하는 전처리가 필수적입니다.

알고리즘: RGB 채널별 독립 통계 분석 + Z-score 정규화 기반 선형 매핑
"""

import numpy as np
from typing import List, Tuple, Optional


def compute_channel_statistics(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    이미지의 RGB 채널별 평균과 표준편차를 계산합니다.
    
    RGB 채널별 독립 통계 분석: 각 이미지의 R, G, B 채널을 분리하여 
    개별적으로 평균과 표준편차를 산출합니다. 흑백 이미지의 단순 밝기 보정과 달리, 
    채널별 독립 처리를 통해 화이트 밸런스의 불일치까지 동시에 교정하도록 설계하였습니다.
    
    Args:
        image: 이미지 (H, W, 3) 또는 (H, W) - uint8 또는 float32
    
    Returns:
        means: 채널별 평균 (3,) 또는 스칼라 - float32
        stds: 채널별 표준편차 (3,) 또는 스칼라 - float32
    """
    # float32 0~1 범위로 변환
    if image.dtype == np.uint8:
        img_float = image.astype(np.float32) / 255.0
    else:
        img_float = np.clip(image.astype(np.float32), 0.0, 1.0)
    
    if len(img_float.shape) == 3:
        # RGB 이미지: 각 채널별로 독립적으로 통계 계산
        H, W, C = img_float.shape
        means = np.zeros(C, dtype=np.float32)
        stds = np.zeros(C, dtype=np.float32)
        
        for c in range(C):
            channel = img_float[:, :, c]
            means[c] = np.mean(channel)
            stds[c] = np.std(channel)
    else:
        # 그레이스케일 이미지
        means = np.array([np.mean(img_float)], dtype=np.float32)
        stds = np.array([np.std(img_float)], dtype=np.float32)
    
    return means, stds


def compute_global_statistics(images: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    전체 이미지 시퀀스의 채널별 전역 평균과 평균 표준편차를 계산합니다.
    
    Args:
        images: 이미지 리스트 [img1, img2, ...], 각 이미지는 (H, W, 3) - uint8
    
    Returns:
        global_means: 채널별 전역 평균 (3,) - float32
        global_stds: 채널별 평균 표준편차 (3,) - float32
    """
    if len(images) == 0:
        return np.array([0.5, 0.5, 0.5], dtype=np.float32), np.array([0.2, 0.2, 0.2], dtype=np.float32)
    
    # 모든 이미지의 통계량 수집
    all_means = []
    all_stds = []
    
    for img in images:
        means, stds = compute_channel_statistics(img)
        all_means.append(means)
        all_stds.append(stds)
    
    # 전역 평균: 모든 이미지의 평균값들의 평균
    all_means = np.array(all_means)  # (N, C)
    global_means = np.mean(all_means, axis=0)  # (C,)
    
    # 평균 표준편차: 모든 이미지의 표준편차들의 평균
    all_stds = np.array(all_stds)  # (N, C)
    global_stds = np.mean(all_stds, axis=0)  # (C,)
    
    return global_means.astype(np.float32), global_stds.astype(np.float32)


def apply_zscore_normalization(image: np.ndarray, 
                                local_means: np.ndarray, 
                                local_stds: np.ndarray,
                                global_means: np.ndarray, 
                                global_stds: np.ndarray,
                                epsilon: float = 1e-10) -> np.ndarray:
    """
    Z-score 정규화 기반 선형 매핑을 적용합니다.
    
    전체 이미지 시퀀스의 채널별 전역 평균(μ_global)과 평균 표준편차(σ_global)를 
    목표치로 설정하고, 각 이미지의 픽셀값(x)을 다음과 같은 선형 변환 수식을 통해 정규화합니다:
    
    normalized = (x - μ_local) / σ_local * σ_global + μ_global
    
    구현 로직: 각 이미지의 통계량을 전역 통계량에 맞춤으로써, 어두운 이미지는 밝게, 
    대비가 낮은 이미지는 선명하게 조정되어 전체 파노라마의 일관성을 확보합니다.
    
    Args:
        image: 이미지 (H, W, 3) 또는 (H, W) - uint8 또는 float32
        local_means: 로컬 평균 (3,) 또는 스칼라 - float32
        local_stds: 로컬 표준편차 (3,) 또는 스칼라 - float32
        global_means: 전역 평균 (3,) 또는 스칼라 - float32
        global_stds: 전역 표준편차 (3,) 또는 스칼라 - float32
        epsilon: Zero-division 방지를 위한 작은 상수 (float, 기본값: 1e-10)
    
    Returns:
        normalized: 정규화된 이미지 (H, W, 3) 또는 (H, W) - float32, 0~1 범위
    """
    # float32 0~1 범위로 변환
    if image.dtype == np.uint8:
        img_float = image.astype(np.float32) / 255.0
    else:
        img_float = np.clip(image.astype(np.float32), 0.0, 1.0)
    
    # Zero-division 방지: 표준편차가 0에 가까운 단색 영역의 경우 작은 상수 추가
    local_stds_safe = np.maximum(local_stds, epsilon)
    
    if len(img_float.shape) == 3:
        # RGB 이미지: 각 채널별로 독립적으로 처리
        H, W, C = img_float.shape
        normalized = np.zeros_like(img_float, dtype=np.float32)
        
        for c in range(C):
            channel = img_float[:, :, c]
            local_mean = local_means[c] if len(local_means.shape) > 0 else local_means[0]
            local_std = local_stds_safe[c] if len(local_stds_safe.shape) > 0 else local_stds_safe[0]
            global_mean = global_means[c] if len(global_means.shape) > 0 else global_means[0]
            global_std = global_stds[c] if len(global_stds.shape) > 0 else global_stds[0]
            
            # Z-score 정규화: normalized = (x - μ_local) / σ_local * σ_global + μ_global
            normalized[:, :, c] = (channel - local_mean) / local_std * global_std + global_mean
    else:
        # 그레이스케일 이미지
        local_mean = local_means[0] if len(local_means.shape) > 0 else local_means
        local_std = local_stds_safe[0] if len(local_stds_safe.shape) > 0 else local_stds_safe
        global_mean = global_means[0] if len(global_means.shape) > 0 else global_means
        global_std = global_stds[0] if len(global_stds.shape) > 0 else global_stds
        
        # Z-score 정규화
        normalized = (img_float - local_mean) / local_std * global_std + global_mean
    
    # Clamping: 정규화 과정에서 0~1 범위를 벗어나는 데이터가 발생하므로, 
    # np.clip을 통해 유효 범위 내로 고정
    normalized = np.clip(normalized, 0.0, 1.0)
    
    return normalized.astype(np.float32)


def compute_channel_ratios(image: np.ndarray) -> np.ndarray:
    """
    이미지의 RGB 채널 간 상대적 비율을 계산합니다.
    
    색온도 차이를 보정하기 위해 R/G, B/G 비율을 계산합니다.
    G 채널을 기준으로 R, B 채널의 상대적 비율을 구합니다.
    
    Args:
        image: 이미지 (H, W, 3) - uint8 또는 float32
    
    Returns:
        ratios: 채널 비율 (2,) - [R/G, B/G] - float32
    """
    # float32 0~1 범위로 변환
    if image.dtype == np.uint8:
        img_float = image.astype(np.float32) / 255.0
    else:
        img_float = np.clip(image.astype(np.float32), 0.0, 1.0)
    
    if len(img_float.shape) != 3:
        # 그레이스케일 이미지는 비율 계산 불가
        return np.array([1.0, 1.0], dtype=np.float32)
    
    # 각 채널의 평균 계산
    mean_r = np.mean(img_float[:, :, 0])
    mean_g = np.mean(img_float[:, :, 1])
    mean_b = np.mean(img_float[:, :, 2])
    
    # R/G, B/G 비율 계산 (Zero-division 방지)
    epsilon = 1e-10
    ratio_rg = mean_r / (mean_g + epsilon)
    ratio_bg = mean_b / (mean_g + epsilon)
    
    return np.array([ratio_rg, ratio_bg], dtype=np.float32)


def compute_global_channel_ratios(images: List[np.ndarray]) -> np.ndarray:
    """
    전체 이미지 시퀀스의 전역 채널 비율을 계산합니다.
    
    Args:
        images: 이미지 리스트 [img1, img2, ...], 각 이미지는 (H, W, 3) - uint8
    
    Returns:
        global_ratios: 전역 채널 비율 (2,) - [R/G, B/G] - float32
    """
    if len(images) == 0:
        return np.array([1.0, 1.0], dtype=np.float32)
    
    # 모든 이미지의 채널 비율 수집
    all_ratios = []
    
    for img in images:
        ratios = compute_channel_ratios(img)
        all_ratios.append(ratios)
    
    # 전역 비율: 모든 이미지의 비율들의 평균
    all_ratios = np.array(all_ratios)  # (N, 2)
    global_ratios = np.mean(all_ratios, axis=0)  # (2,)
    
    return global_ratios.astype(np.float32)


def apply_channel_ratio_correction(image: np.ndarray,
                                    local_ratios: np.ndarray,
                                    global_ratios: np.ndarray) -> np.ndarray:
    """
    채널 간 상대적 비율 보정을 적용합니다.
    
    색온도 차이를 보정하기 위해 R/G, B/G 비율을 전역 비율로 맞춥니다.
    
    수식:
    - G 채널은 유지
    - R 채널: R_new = R * (global_RG / local_RG)
    - B 채널: B_new = B * (global_BG / local_BG)
    
    Args:
        image: 이미지 (H, W, 3) - uint8 또는 float32
        local_ratios: 로컬 채널 비율 (2,) - [R/G, B/G] - float32
        global_ratios: 전역 채널 비율 (2,) - [R/G, B/G] - float32
    
    Returns:
        corrected: 보정된 이미지 (H, W, 3) - float32, 0~1 범위
    """
    # float32 0~1 범위로 변환
    if image.dtype == np.uint8:
        img_float = image.astype(np.float32) / 255.0
    else:
        img_float = np.clip(image.astype(np.float32), 0.0, 1.0)
    
    if len(img_float.shape) != 3:
        # 그레이스케일 이미지는 보정 불가
        return img_float
    
    corrected = img_float.copy()
    
    # Zero-division 방지
    epsilon = 1e-10
    local_rg = local_ratios[0] if len(local_ratios.shape) > 0 else local_ratios
    local_bg = local_ratios[1] if len(local_ratios.shape) > 0 else local_ratios
    global_rg = global_ratios[0] if len(global_ratios.shape) > 0 else global_ratios
    global_bg = global_ratios[1] if len(global_ratios.shape) > 0 else global_ratios
    
    # R 채널 보정: R_new = R * (global_RG / local_RG)
    if local_rg > epsilon:
        r_scale = global_rg / local_rg
        corrected[:, :, 0] = corrected[:, :, 0] * r_scale
    
    # B 채널 보정: B_new = B * (global_BG / local_BG)
    if local_bg > epsilon:
        b_scale = global_bg / local_bg
        corrected[:, :, 2] = corrected[:, :, 2] * b_scale
    
    # G 채널은 유지 (기준 채널)
    
    # Clamping
    corrected = np.clip(corrected, 0.0, 1.0)
    
    return corrected.astype(np.float32)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """
    RGB 색공간을 LAB 색공간으로 변환합니다.
    
    LAB 색공간은 색조와 채도를 독립적으로 처리할 수 있어 색온도 보정에 유리합니다.
    - L: 밝기 (Lightness)
    - a: 녹색-빨강 축
    - b: 파랑-노랑 축
    
    Args:
        rgb: RGB 이미지 (H, W, 3) - float32, 0~1 범위
    
    Returns:
        lab: LAB 이미지 (H, W, 3) - float32
    """
    # RGB를 0~1 범위로 확보
    rgb = np.clip(rgb, 0.0, 1.0)
    
    # sRGB to Linear RGB (Gamma correction)
    def f(t):
        return np.where(t > 0.04045,
                        np.power((t + 0.055) / 1.055, 2.4),
                        t / 12.92)
    
    r_linear = f(rgb[:, :, 0])
    g_linear = f(rgb[:, :, 1])
    b_linear = f(rgb[:, :, 2])
    
    # Linear RGB to XYZ
    # sRGB D65 white point
    x = 0.4124564 * r_linear + 0.3575761 * g_linear + 0.1804375 * b_linear
    y = 0.2126729 * r_linear + 0.7151522 * g_linear + 0.0721750 * b_linear
    z = 0.0193339 * r_linear + 0.1191920 * g_linear + 0.9503041 * b_linear
    
    # XYZ to LAB
    # D65 white point: Xn=0.95047, Yn=1.00000, Zn=1.08883
    xn, yn, zn = 0.95047, 1.00000, 1.08883
    
    x_norm = x / xn
    y_norm = y / yn
    z_norm = z / zn
    
    def f_inv(t):
        return np.where(t > 0.008856,
                        np.power(t, 1.0/3.0),
                        (7.787 * t + 16.0/116.0))
    
    fx = f_inv(x_norm)
    fy = f_inv(y_norm)
    fz = f_inv(z_norm)
    
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    
    # LAB 값 정규화 (일반적인 범위: L: 0~100, a: -128~127, b: -128~127)
    # 여기서는 0~1 범위로 정규화하지 않고 원본 값 사용
    lab = np.stack([L, a, b], axis=2)
    
    return lab.astype(np.float32)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """
    LAB 색공간을 RGB 색공간으로 변환합니다.
    
    Args:
        lab: LAB 이미지 (H, W, 3) - float32
    
    Returns:
        rgb: RGB 이미지 (H, W, 3) - float32, 0~1 범위
    """
    L = lab[:, :, 0]
    a = lab[:, :, 1]
    b = lab[:, :, 2]
    
    # LAB to XYZ
    # D65 white point
    xn, yn, zn = 0.95047, 1.00000, 1.08883
    
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0
    
    def f_inv(t):
        return np.where(t > 0.206897,
                        np.power(t, 3.0),
                        3.0 * 0.206897**2 * (t - 16.0/116.0))
    
    x_norm = f_inv(fx)
    y_norm = f_inv(fy)
    z_norm = f_inv(fz)
    
    x = x_norm * xn
    y = y_norm * yn
    z = z_norm * zn
    
    # XYZ to Linear RGB
    r_linear = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
    g_linear = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
    b_linear = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z
    
    # Linear RGB to sRGB (Gamma correction)
    def f(t):
        return np.where(t > 0.0031308,
                        1.055 * np.power(t, 1.0/2.4) - 0.055,
                        12.92 * t)
    
    r = f(r_linear)
    g = f(g_linear)
    b = f(b_linear)
    
    rgb = np.stack([r, g, b], axis=2)
    rgb = np.clip(rgb, 0.0, 1.0)
    
    return rgb.astype(np.float32)


def apply_lab_color_correction(image: np.ndarray,
                                reference_lab: Optional[np.ndarray] = None) -> np.ndarray:
    """
    LAB 색공간에서 색온도와 색조 보정을 적용합니다.
    
    LAB 색공간에서:
    - L 채널: 밝기 (Z-score 정규화로 보정)
    - a, b 채널: 색조 (히스토그램 매칭 또는 평균 맞추기로 보정)
    
    Args:
        image: 이미지 (H, W, 3) - uint8 또는 float32
        reference_lab: 참조 LAB 이미지 (H, W, 3) - float32, None이면 전역 평균 사용
    
    Returns:
        corrected: 보정된 이미지 (H, W, 3) - float32, 0~1 범위
    """
    # float32 0~1 범위로 변환
    if image.dtype == np.uint8:
        img_float = image.astype(np.float32) / 255.0
    else:
        img_float = np.clip(image.astype(np.float32), 0.0, 1.0)
    
    if len(img_float.shape) != 3:
        # 그레이스케일 이미지는 보정 불가
        return img_float
    
    # RGB to LAB
    lab = rgb_to_lab(img_float)
    
    # L 채널: 밝기 보정 (Z-score 정규화)
    L = lab[:, :, 0]
    L_mean = np.mean(L)
    L_std = np.std(L)
    
    # 목표: L 평균을 50 (중간 밝기)로 맞춤
    target_L_mean = 50.0
    target_L_std = 20.0  # 적절한 표준편차
    
    epsilon = 1e-10
    if L_std > epsilon:
        L_corrected = (L - L_mean) / L_std * target_L_std + target_L_mean
    else:
        L_corrected = L * 0.0 + target_L_mean
    
    # a, b 채널: 색조 보정
    a = lab[:, :, 1]
    b = lab[:, :, 2]
    
    if reference_lab is not None:
        # 참조 이미지의 a, b 평균으로 맞춤
        ref_a_mean = np.mean(reference_lab[:, :, 1])
        ref_b_mean = np.mean(reference_lab[:, :, 2])
        
        a_mean = np.mean(a)
        b_mean = np.mean(b)
        
        a_corrected = a - a_mean + ref_a_mean
        b_corrected = b - b_mean + ref_b_mean
    else:
        # 전역 평균으로 맞춤 (중립 색조)
        a_mean = np.mean(a)
        b_mean = np.mean(b)
        
        # 중립 색조로 맞춤 (a=0, b=0 근처)
        a_corrected = a - a_mean
        b_corrected = b - b_mean
    
    # LAB 값 클램핑
    L_corrected = np.clip(L_corrected, 0.0, 100.0)
    a_corrected = np.clip(a_corrected, -128.0, 127.0)
    b_corrected = np.clip(b_corrected, -128.0, 127.0)
    
    # LAB to RGB
    lab_corrected = np.stack([L_corrected, a_corrected, b_corrected], axis=2)
    rgb_corrected = lab_to_rgb(lab_corrected)
    
    return rgb_corrected.astype(np.float32)


def compute_exposure_compensation(image: np.ndarray) -> float:
    """
    이미지의 노출 보정 계수를 계산합니다.
    
    중간 밝기(median brightness)를 기준으로 노출 보정 계수를 계산합니다.
    여러 이미지의 중간 밝기를 맞추기 위해 사용됩니다.
    
    Args:
        image: 이미지 (H, W, 3) 또는 (H, W) - uint8 또는 float32
    
    Returns:
        exposure_factor: 노출 보정 계수 (float)
                       1.0보다 크면 밝게, 작으면 어둡게 조정
    """
    # float32 0~1 범위로 변환
    if image.dtype == np.uint8:
        img_float = image.astype(np.float32) / 255.0
    else:
        img_float = np.clip(image.astype(np.float32), 0.0, 1.0)
    
    # 그레이스케일로 변환 (밝기 계산용)
    if len(img_float.shape) == 3:
        gray = 0.299 * img_float[:, :, 0] + \
               0.587 * img_float[:, :, 1] + \
               0.114 * img_float[:, :, 2]
    else:
        gray = img_float
    
    # 중간 밝기 계산 (median)
    median_brightness = np.median(gray)
    
    # 목표 밝기 (0.5 = 중간 회색)
    target_brightness = 0.5
    
    # 노출 보정 계수 계산
    # median_brightness * exposure_factor = target_brightness
    if median_brightness > 1e-10:
        exposure_factor = target_brightness / median_brightness
    else:
        exposure_factor = 1.0
    
    # 과도한 조정 방지 (0.5 ~ 2.0 범위로 제한)
    exposure_factor = np.clip(exposure_factor, 0.5, 2.0)
    
    return float(exposure_factor)


def apply_exposure_compensation(image: np.ndarray, exposure_factor: float) -> np.ndarray:
    """
    노출 보정을 적용합니다.
    
    수식: adjusted = image * exposure_factor
    
    Args:
        image: 이미지 (H, W, 3) 또는 (H, W) - uint8 또는 float32
        exposure_factor: 노출 보정 계수 (float)
    
    Returns:
        adjusted: 보정된 이미지 (H, W, 3) 또는 (H, W) - float32, 0~1 범위
    """
    # float32 0~1 범위로 변환
    if image.dtype == np.uint8:
        img_float = image.astype(np.float32) / 255.0
    else:
        img_float = np.clip(image.astype(np.float32), 0.0, 1.0)
    
    # 노출 보정 적용
    adjusted = img_float * exposure_factor
    
    # 0~1 범위로 클램핑
    adjusted = np.clip(adjusted, 0.0, 1.0)
    
    return adjusted.astype(np.float32)


def apply_gamma_correction(image: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    """
    Gamma Correction을 적용합니다.
    
    수식: corrected = image^(1/gamma)
    
    Args:
        image: 이미지 (H, W, 3) 또는 (H, W) - uint8 또는 float32
        gamma: 감마 값 (float, 기본값: 2.2, 표준 sRGB 감마)
    
    Returns:
        corrected: 보정된 이미지 (H, W, 3) 또는 (H, W) - float32, 0~1 범위
    """
    # float32 0~1 범위로 변환
    if image.dtype == np.uint8:
        img_float = image.astype(np.float32) / 255.0
    else:
        img_float = np.clip(image.astype(np.float32), 0.0, 1.0)
    
    # Gamma Correction 적용
    # 수식: corrected = image^(1/gamma)
    # 0 값 처리: 0^power = 0이므로 안전
    corrected = np.power(img_float, 1.0 / gamma)
    
    return corrected.astype(np.float32)


def compute_histogram(image: np.ndarray, bins: int = 256) -> np.ndarray:
    """
    이미지의 히스토그램을 계산합니다.
    
    Args:
        image: 이미지 (H, W, 3) 또는 (H, W) - uint8 또는 float32
        bins: 히스토그램 bin 개수 (int, 기본값: 256)
    
    Returns:
        histogram: 히스토그램 (bins,) 또는 (bins, 3) - float32
                   RGB 이미지인 경우 각 채널별 히스토그램
    """
    # float32 0~1 범위로 변환
    if image.dtype == np.uint8:
        img_float = image.astype(np.float32) / 255.0
    else:
        img_float = np.clip(image.astype(np.float32), 0.0, 1.0)
    
    # 0~255 범위로 변환 (히스토그램 계산용)
    img_int = (img_float * (bins - 1)).astype(np.int32)
    img_int = np.clip(img_int, 0, bins - 1)
    
    if len(img_float.shape) == 3:
        # RGB 이미지: 각 채널별 히스토그램
        H, W, C = img_float.shape
        histogram = np.zeros((bins, C), dtype=np.float32)
        
        for c in range(C):
            channel_flat = img_int[:, :, c].flatten()
            hist, _ = np.histogram(channel_flat, bins=bins, range=(0, bins))
            histogram[:, c] = hist.astype(np.float32)
    else:
        # 그레이스케일 이미지
        img_flat = img_int.flatten()
        hist, _ = np.histogram(img_flat, bins=bins, range=(0, bins))
        histogram = hist.astype(np.float32)
    
    return histogram


def compute_cumulative_histogram(histogram: np.ndarray) -> np.ndarray:
    """
    누적 히스토그램(Cumulative Histogram)을 계산합니다.
    
    Args:
        histogram: 히스토그램 (bins,) 또는 (bins, C) - float32
    
    Returns:
        cumulative: 누적 히스토그램 (bins,) 또는 (bins, C) - float32
    """
    if len(histogram.shape) == 1:
        # 그레이스케일
        cumulative = np.cumsum(histogram)
    else:
        # RGB
        cumulative = np.cumsum(histogram, axis=0)
    
    # 정규화 (0~1 범위)
    total = cumulative[-1] if len(cumulative.shape) == 1 else cumulative[-1, :]
    if len(cumulative.shape) == 1:
        if total > 1e-10:
            cumulative = cumulative / total
    else:
        for c in range(cumulative.shape[1]):
            if total[c] > 1e-10:
                cumulative[:, c] = cumulative[:, c] / total[c]
    
    return cumulative.astype(np.float32)


def match_histogram(source_image: np.ndarray, target_histogram: np.ndarray, bins: int = 256) -> np.ndarray:
    """
    Histogram Matching을 수행합니다.
    
    source_image의 히스토그램을 target_histogram과 일치시킵니다.
    
    알고리즘:
    1. source와 target의 누적 히스토그램 계산
    2. 각 픽셀 값을 source 누적 히스토그램에서 찾고
    3. 같은 누적 값을 가진 target의 픽셀 값으로 매핑
    
    Args:
        source_image: 소스 이미지 (H, W, 3) 또는 (H, W) - uint8 또는 float32
        target_histogram: 타겟 히스토그램 (bins,) 또는 (bins, 3) - float32
        bins: 히스토그램 bin 개수 (int, 기본값: 256)
    
    Returns:
        matched: 히스토그램이 매칭된 이미지 (H, W, 3) 또는 (H, W) - float32, 0~1 범위
    """
    # float32 0~1 범위로 변환
    if source_image.dtype == np.uint8:
        img_float = source_image.astype(np.float32) / 255.0
    else:
        img_float = np.clip(source_image.astype(np.float32), 0.0, 1.0)
    
    # 소스 히스토그램 계산
    source_hist = compute_histogram(img_float, bins)
    
    # 누적 히스토그램 계산
    source_cumulative = compute_cumulative_histogram(source_hist)
    target_cumulative = compute_cumulative_histogram(target_histogram)
    
    # 히스토그램 매칭 LUT (Look-Up Table) 생성
    if len(img_float.shape) == 3:
        # RGB 이미지
        H, W, C = img_float.shape
        matched = np.zeros_like(img_float, dtype=np.float32)
        
        for c in range(C):
            # 각 채널별로 LUT 생성
            lut = np.zeros(bins, dtype=np.float32)
            
            for i in range(bins):
                # source_cumulative[i]와 가장 가까운 target_cumulative 값 찾기
                source_val = source_cumulative[i, c] if len(source_cumulative.shape) > 1 else source_cumulative[i]
                
                # target_cumulative에서 가장 가까운 값 찾기
                if len(target_cumulative.shape) > 1:
                    target_vals = target_cumulative[:, c]
                else:
                    target_vals = target_cumulative
                
                # 가장 가까운 인덱스 찾기
                diff = np.abs(target_vals - source_val)
                closest_idx = np.argmin(diff)
                lut[i] = closest_idx / float(bins - 1)
            
            # 이미지에 LUT 적용
            channel_int = (img_float[:, :, c] * (bins - 1)).astype(np.int32)
            channel_int = np.clip(channel_int, 0, bins - 1)
            matched[:, :, c] = lut[channel_int]
    else:
        # 그레이스케일 이미지
        lut = np.zeros(bins, dtype=np.float32)
        
        for i in range(bins):
            source_val = source_cumulative[i]
            
            if len(target_cumulative.shape) > 1:
                # target이 RGB인 경우 첫 번째 채널 사용
                target_vals = target_cumulative[:, 0]
            else:
                target_vals = target_cumulative
            
            diff = np.abs(target_vals - source_val)
            closest_idx = np.argmin(diff)
            lut[i] = closest_idx / float(bins - 1)
        
        # 이미지에 LUT 적용
        img_int = (img_float * (bins - 1)).astype(np.int32)
        img_int = np.clip(img_int, 0, bins - 1)
        matched = lut[img_int]
    
    return matched.astype(np.float32)


def tone_map_images(images: List[np.ndarray], 
                    method: str = "zscore_ratio",
                    reference_idx: Optional[int] = None) -> List[np.ndarray]:
    """
    여러 이미지에 Tone Mapping을 적용합니다.
    
    Args:
        images: 이미지 리스트 [img1, img2, ...], 각 이미지는 (H, W, 3) - uint8
        method: Tone Mapping 방법 (str)
                - "zscore": Z-score 정규화 기반 선형 매핑 (요구사항 구현)
                - "zscore_ratio": Z-score + 채널 비율 보정 (색온도 보정 포함, 권장)
                - "lab": LAB 색공간 기반 보정 (색온도/색조 보정)
                - "zscore_lab": Z-score + LAB 보정 결합 (최고 품질)
                - "exposure": Exposure Compensation (구버전)
                - "gamma": Gamma Correction
                - "histogram": Histogram Matching
                - "exposure_gamma": Exposure + Gamma
        reference_idx: 참조 이미지 인덱스 (int, None이면 중앙 이미지 사용)
                       Histogram Matching 및 LAB 보정에서 사용
    
    Returns:
        tone_mapped: Tone Mapping이 적용된 이미지 리스트, 각 이미지는 (H, W, 3) - float32, 0~1 범위
    """
    if len(images) == 0:
        return []
    
    if len(images) == 1:
        # 단일 이미지는 원본 반환 (float32로 변환만)
        img_float = images[0].astype(np.float32) / 255.0 if images[0].dtype == np.uint8 else images[0]
        return [np.clip(img_float, 0.0, 1.0)]
    
    tone_mapped = []
    
    if method == "zscore":
        # Z-score 정규화 기반 선형 매핑 (요구사항 구현)
        # 1. 전체 이미지 시퀀스의 채널별 전역 통계량 계산
        global_means, global_stds = compute_global_statistics(images)
        
        # 2. 각 이미지에 대해 Z-score 정규화 적용
        for img in images:
            # 각 이미지의 로컬 통계량 계산
            local_means, local_stds = compute_channel_statistics(img)
            
            # Z-score 정규화 적용
            normalized = apply_zscore_normalization(
                img, local_means, local_stds, global_means, global_stds
            )
            
            # uint8로 변환 후 다시 float32로 변환 (Clamping 적용)
            # 요구사항: np.clip을 통해 유효 범위 내로 고정시킨 후 uint8 형식으로 복원
            normalized_uint8 = (normalized * 255.0).astype(np.uint8)
            normalized_float = normalized_uint8.astype(np.float32) / 255.0
            
            tone_mapped.append(normalized_float)
    
    elif method == "zscore_ratio":
        # Z-score 정규화 + 채널 비율 보정 (색온도 보정 포함, 권장)
        # 1. 전체 이미지 시퀀스의 채널별 전역 통계량 및 비율 계산
        global_means, global_stds = compute_global_statistics(images)
        global_ratios = compute_global_channel_ratios(images)
        
        # 2. 각 이미지에 대해 Z-score 정규화 + 채널 비율 보정 적용
        for img in images:
            # 각 이미지의 로컬 통계량 및 비율 계산
            local_means, local_stds = compute_channel_statistics(img)
            local_ratios = compute_channel_ratios(img)
            
            # Z-score 정규화 적용
            normalized = apply_zscore_normalization(
                img, local_means, local_stds, global_means, global_stds
            )
            
            # 채널 비율 보정 적용 (색온도 보정)
            corrected = apply_channel_ratio_correction(
                normalized, local_ratios, global_ratios
            )
            
            # uint8로 변환 후 다시 float32로 변환
            corrected_uint8 = (corrected * 255.0).astype(np.uint8)
            corrected_float = corrected_uint8.astype(np.float32) / 255.0
            
            tone_mapped.append(corrected_float)
    
    elif method == "lab":
        # LAB 색공간 기반 보정 (색온도/색조 보정)
        # 참조 이미지 선택
        if reference_idx is None:
            reference_idx = len(images) // 2  # 중앙 이미지
        
        reference_img = images[reference_idx]
        reference_float = reference_img.astype(np.float32) / 255.0 if reference_img.dtype == np.uint8 else reference_img
        reference_lab = rgb_to_lab(reference_float)
        
        # 각 이미지에 대해 LAB 보정 적용
        for img in images:
            corrected = apply_lab_color_correction(img, reference_lab)
            tone_mapped.append(corrected)
    
    elif method == "zscore_lab":
        # Z-score + LAB 보정 결합 (최고 품질)
        # 1. Z-score 정규화로 밝기 통일
        global_means, global_stds = compute_global_statistics(images)
        
        # 참조 이미지 선택 (LAB 보정용)
        if reference_idx is None:
            reference_idx = len(images) // 2
        
        reference_img = images[reference_idx]
        reference_float = reference_img.astype(np.float32) / 255.0 if reference_img.dtype == np.uint8 else reference_img
        reference_lab = rgb_to_lab(reference_float)
        
        # 2. 각 이미지에 대해 Z-score 정규화 후 LAB 보정
        for img in images:
            # Z-score 정규화
            local_means, local_stds = compute_channel_statistics(img)
            normalized = apply_zscore_normalization(
                img, local_means, local_stds, global_means, global_stds
            )
            
            # LAB 보정 (색온도/색조 보정)
            corrected = apply_lab_color_correction(normalized, reference_lab)
            tone_mapped.append(corrected)
    
    elif method == "exposure":
        # Exposure Compensation: 각 이미지의 노출을 중간 밝기로 맞춤
        for img in images:
            exposure_factor = compute_exposure_compensation(img)
            adjusted = apply_exposure_compensation(img, exposure_factor)
            tone_mapped.append(adjusted)
    
    elif method == "gamma":
        # Gamma Correction만 적용
        for img in images:
            img_float = img.astype(np.float32) / 255.0 if img.dtype == np.uint8 else img
            corrected = apply_gamma_correction(img_float)
            tone_mapped.append(corrected)
    
    elif method == "histogram":
        # Histogram Matching: 참조 이미지의 히스토그램에 맞춤
        if reference_idx is None:
            reference_idx = len(images) // 2  # 중앙 이미지
        
        reference_img = images[reference_idx]
        target_hist = compute_histogram(reference_img)
        
        for img in images:
            matched = match_histogram(img, target_hist)
            tone_mapped.append(matched)
    
    elif method == "exposure_gamma":
        # Exposure Compensation + Gamma Correction
        for img in images:
            # 1. Exposure Compensation
            exposure_factor = compute_exposure_compensation(img)
            adjusted = apply_exposure_compensation(img, exposure_factor)
            
            # 2. Gamma Correction
            corrected = apply_gamma_correction(adjusted)
            tone_mapped.append(corrected)
    
    else:
        # 알 수 없는 방법: 원본 반환 (float32로 변환만)
        for img in images:
            img_float = img.astype(np.float32) / 255.0 if img.dtype == np.uint8 else img
            tone_mapped.append(np.clip(img_float, 0.0, 1.0))
    
    return tone_mapped

