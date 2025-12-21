"""
Tone Mapping: 여러 이미지의 노출 차이를 조정하여 자연스러운 블렌딩
RGB 채널별 독립 통계 분석 + Z-score 정규화 기반 선형 매핑
"""

import numpy as np
from typing import List, Tuple, Optional


def compute_channel_statistics(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if image.dtype == np.uint8:
        img_float = image.astype(np.float32) / 255.0
    else:
        img_float = np.clip(image.astype(np.float32), 0.0, 1.0)
    
    if len(img_float.shape) == 3:
        H, W, C = img_float.shape
        means = np.zeros(C, dtype=np.float32)
        stds = np.zeros(C, dtype=np.float32)
        
        for c in range(C):
            channel = img_float[:, :, c]
            means[c] = np.mean(channel)
            stds[c] = np.std(channel)
    else:
        means = np.array([np.mean(img_float)], dtype=np.float32)
        stds = np.array([np.std(img_float)], dtype=np.float32)
    
    return means, stds


def compute_global_statistics(images: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    if len(images) == 0:
        return np.array([0.5, 0.5, 0.5], dtype=np.float32), np.array([0.2, 0.2, 0.2], dtype=np.float32)
    
    all_means = []
    all_stds = []
    
    for img in images:
        means, stds = compute_channel_statistics(img)
        all_means.append(means)
        all_stds.append(stds)
    
    all_means = np.array(all_means)
    global_means = np.mean(all_means, axis=0)
    
    all_stds = np.array(all_stds)
    global_stds = np.mean(all_stds, axis=0)
    
    return global_means.astype(np.float32), global_stds.astype(np.float32)


def apply_zscore_normalization(image: np.ndarray, 
                                local_means: np.ndarray, 
                                local_stds: np.ndarray,
                                global_means: np.ndarray, 
                                global_stds: np.ndarray,
                                epsilon: float = 1e-10) -> np.ndarray:
    """
    normalized = (x - μ_local) / σ_local * σ_global + μ_global
    """
    if image.dtype == np.uint8:
        img_float = image.astype(np.float32) / 255.0
    else:
        img_float = np.clip(image.astype(np.float32), 0.0, 1.0)
    
    local_stds_safe = np.maximum(local_stds, epsilon)
    
    if len(img_float.shape) == 3:
        H, W, C = img_float.shape
        normalized = np.zeros_like(img_float, dtype=np.float32)
        
        for c in range(C):
            channel = img_float[:, :, c]
            local_mean = local_means[c] if len(local_means.shape) > 0 else local_means[0]
            local_std = local_stds_safe[c] if len(local_stds_safe.shape) > 0 else local_stds_safe[0]
            global_mean = global_means[c] if len(global_means.shape) > 0 else global_means[0]
            global_std = global_stds[c] if len(global_stds.shape) > 0 else global_stds[0]
            
            normalized[:, :, c] = (channel - local_mean) / local_std * global_std + global_mean
    else:
        local_mean = local_means[0] if len(local_means.shape) > 0 else local_means
        local_std = local_stds_safe[0] if len(local_stds_safe.shape) > 0 else local_stds_safe
        global_mean = global_means[0] if len(global_means.shape) > 0 else global_means
        global_std = global_stds[0] if len(global_stds.shape) > 0 else global_stds
        
        normalized = (img_float - local_mean) / local_std * global_std + global_mean
    
    normalized = np.clip(normalized, 0.0, 1.0)
    
    return normalized.astype(np.float32)


def compute_channel_ratios(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        img_float = image.astype(np.float32) / 255.0
    else:
        img_float = np.clip(image.astype(np.float32), 0.0, 1.0)
    
    if len(img_float.shape) != 3:
        return np.array([1.0, 1.0], dtype=np.float32)
    
    mean_r = np.mean(img_float[:, :, 0])
    mean_g = np.mean(img_float[:, :, 1])
    mean_b = np.mean(img_float[:, :, 2])
    
    epsilon = 1e-10
    ratio_rg = mean_r / (mean_g + epsilon)
    ratio_bg = mean_b / (mean_g + epsilon)
    
    return np.array([ratio_rg, ratio_bg], dtype=np.float32)


def compute_global_channel_ratios(images: List[np.ndarray]) -> np.ndarray:
    if len(images) == 0:
        return np.array([1.0, 1.0], dtype=np.float32)
    
    all_ratios = []
    
    for img in images:
        ratios = compute_channel_ratios(img)
        all_ratios.append(ratios)
    
    all_ratios = np.array(all_ratios)
    global_ratios = np.mean(all_ratios, axis=0)
    
    return global_ratios.astype(np.float32)


def apply_channel_ratio_correction(image: np.ndarray,
                                    local_ratios: np.ndarray,
                                    global_ratios: np.ndarray) -> np.ndarray:
    """
    R_new = R * (global_RG / local_RG)
    B_new = B * (global_BG / local_BG)
    """
    if image.dtype == np.uint8:
        img_float = image.astype(np.float32) / 255.0
    else:
        img_float = np.clip(image.astype(np.float32), 0.0, 1.0)
    
    if len(img_float.shape) != 3:
        return img_float
    
    corrected = img_float.copy()
    
    epsilon = 1e-10
    local_rg = local_ratios[0] if len(local_ratios.shape) > 0 else local_ratios
    local_bg = local_ratios[1] if len(local_ratios.shape) > 0 else local_ratios
    global_rg = global_ratios[0] if len(global_ratios.shape) > 0 else global_ratios
    global_bg = global_ratios[1] if len(global_ratios.shape) > 0 else global_ratios
    
    if local_rg > epsilon:
        r_scale = global_rg / local_rg
        corrected[:, :, 0] = corrected[:, :, 0] * r_scale
    
    if local_bg > epsilon:
        b_scale = global_bg / local_bg
        corrected[:, :, 2] = corrected[:, :, 2] * b_scale
    
    corrected = np.clip(corrected, 0.0, 1.0)
    
    return corrected.astype(np.float32)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """
    LAB 색공간: L(밝기), a(녹색-빨강), b(파랑-노랑)
    """
    rgb = np.clip(rgb, 0.0, 1.0)
    
    def f(t):
        return np.where(t > 0.04045,
                        np.power((t + 0.055) / 1.055, 2.4),
                        t / 12.92)
    
    r_linear = f(rgb[:, :, 0])
    g_linear = f(rgb[:, :, 1])
    b_linear = f(rgb[:, :, 2])
    
    x = 0.4124564 * r_linear + 0.3575761 * g_linear + 0.1804375 * b_linear
    y = 0.2126729 * r_linear + 0.7151522 * g_linear + 0.0721750 * b_linear
    z = 0.0193339 * r_linear + 0.1191920 * g_linear + 0.9503041 * b_linear
    
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
    
    lab = np.stack([L, a, b], axis=2)
    
    return lab.astype(np.float32)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    L = lab[:, :, 0]
    a = lab[:, :, 1]
    b = lab[:, :, 2]
    
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
    
    r_linear = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
    g_linear = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
    b_linear = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z
    
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
    LAB 색공간: L 채널(밝기) Z-score 정규화, a/b 채널(색조) 평균 맞추기
    """
    if image.dtype == np.uint8:
        img_float = image.astype(np.float32) / 255.0
    else:
        img_float = np.clip(image.astype(np.float32), 0.0, 1.0)
    
    if len(img_float.shape) != 3:
        return img_float
    
    lab = rgb_to_lab(img_float)
    
    L = lab[:, :, 0]
    L_mean = np.mean(L)
    L_std = np.std(L)
    
    target_L_mean = 50.0
    target_L_std = 20.0
    
    epsilon = 1e-10
    if L_std > epsilon:
        L_corrected = (L - L_mean) / L_std * target_L_std + target_L_mean
    else:
        L_corrected = L * 0.0 + target_L_mean
    
    a = lab[:, :, 1]
    b = lab[:, :, 2]
    
    if reference_lab is not None:
        ref_a_mean = np.mean(reference_lab[:, :, 1])
        ref_b_mean = np.mean(reference_lab[:, :, 2])
        
        a_mean = np.mean(a)
        b_mean = np.mean(b)
        
        a_corrected = a - a_mean + ref_a_mean
        b_corrected = b - b_mean + ref_b_mean
    else:
        a_mean = np.mean(a)
        b_mean = np.mean(b)
        
        a_corrected = a - a_mean
        b_corrected = b - b_mean
    
    L_corrected = np.clip(L_corrected, 0.0, 100.0)
    a_corrected = np.clip(a_corrected, -128.0, 127.0)
    b_corrected = np.clip(b_corrected, -128.0, 127.0)
    
    lab_corrected = np.stack([L_corrected, a_corrected, b_corrected], axis=2)
    rgb_corrected = lab_to_rgb(lab_corrected)
    
    return rgb_corrected.astype(np.float32)


def compute_exposure_compensation(image: np.ndarray) -> float:
    if image.dtype == np.uint8:
        img_float = image.astype(np.float32) / 255.0
    else:
        img_float = np.clip(image.astype(np.float32), 0.0, 1.0)
    
    if len(img_float.shape) == 3:
        gray = 0.299 * img_float[:, :, 0] + \
               0.587 * img_float[:, :, 1] + \
               0.114 * img_float[:, :, 2]
    else:
        gray = img_float
    
    median_brightness = np.median(gray)
    
    target_brightness = 0.5
    
    if median_brightness > 1e-10:
        exposure_factor = target_brightness / median_brightness
    else:
        exposure_factor = 1.0
    
    exposure_factor = np.clip(exposure_factor, 0.5, 2.0)
    
    return float(exposure_factor)


def apply_exposure_compensation(image: np.ndarray, exposure_factor: float) -> np.ndarray:
    """
    adjusted = image * exposure_factor
    """
    if image.dtype == np.uint8:
        img_float = image.astype(np.float32) / 255.0
    else:
        img_float = np.clip(image.astype(np.float32), 0.0, 1.0)
    
    adjusted = img_float * exposure_factor
    
    adjusted = np.clip(adjusted, 0.0, 1.0)
    
    return adjusted.astype(np.float32)


def apply_gamma_correction(image: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    """
    corrected = image^(1/gamma)
    """
    if image.dtype == np.uint8:
        img_float = image.astype(np.float32) / 255.0
    else:
        img_float = np.clip(image.astype(np.float32), 0.0, 1.0)
    
    corrected = np.power(img_float, 1.0 / gamma)
    
    return corrected.astype(np.float32)


def compute_histogram(image: np.ndarray, bins: int = 256) -> np.ndarray:
    if image.dtype == np.uint8:
        img_float = image.astype(np.float32) / 255.0
    else:
        img_float = np.clip(image.astype(np.float32), 0.0, 1.0)
    
    img_int = (img_float * (bins - 1)).astype(np.int32)
    img_int = np.clip(img_int, 0, bins - 1)
    
    if len(img_float.shape) == 3:
        H, W, C = img_float.shape
        histogram = np.zeros((bins, C), dtype=np.float32)
        
        for c in range(C):
            channel_flat = img_int[:, :, c].flatten()
            hist, _ = np.histogram(channel_flat, bins=bins, range=(0, bins))
            histogram[:, c] = hist.astype(np.float32)
    else:
        img_flat = img_int.flatten()
        hist, _ = np.histogram(img_flat, bins=bins, range=(0, bins))
        histogram = hist.astype(np.float32)
    
    return histogram


def compute_cumulative_histogram(histogram: np.ndarray) -> np.ndarray:
    if len(histogram.shape) == 1:
        cumulative = np.cumsum(histogram)
    else:
        cumulative = np.cumsum(histogram, axis=0)
    
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
    Histogram Matching:
    1. source와 target의 누적 히스토그램 계산
    2. 각 픽셀 값을 source 누적 히스토그램에서 찾고
    3. 같은 누적 값을 가진 target의 픽셀 값으로 매핑
    """
    if source_image.dtype == np.uint8:
        img_float = source_image.astype(np.float32) / 255.0
    else:
        img_float = np.clip(source_image.astype(np.float32), 0.0, 1.0)
    
    source_hist = compute_histogram(img_float, bins)
    
    source_cumulative = compute_cumulative_histogram(source_hist)
    target_cumulative = compute_cumulative_histogram(target_histogram)
    
    if len(img_float.shape) == 3:
        H, W, C = img_float.shape
        matched = np.zeros_like(img_float, dtype=np.float32)
        
        for c in range(C):
            lut = np.zeros(bins, dtype=np.float32)
            
            for i in range(bins):
                source_val = source_cumulative[i, c] if len(source_cumulative.shape) > 1 else source_cumulative[i]
                
                if len(target_cumulative.shape) > 1:
                    target_vals = target_cumulative[:, c]
                else:
                    target_vals = target_cumulative
                
                diff = np.abs(target_vals - source_val)
                closest_idx = np.argmin(diff)
                lut[i] = closest_idx / float(bins - 1)
            
            channel_int = (img_float[:, :, c] * (bins - 1)).astype(np.int32)
            channel_int = np.clip(channel_int, 0, bins - 1)
            matched[:, :, c] = lut[channel_int]
    else:
        lut = np.zeros(bins, dtype=np.float32)
        
        for i in range(bins):
            source_val = source_cumulative[i]
            
            if len(target_cumulative.shape) > 1:
                target_vals = target_cumulative[:, 0]
            else:
                target_vals = target_cumulative
            
            diff = np.abs(target_vals - source_val)
            closest_idx = np.argmin(diff)
            lut[i] = closest_idx / float(bins - 1)
        
        img_int = (img_float * (bins - 1)).astype(np.int32)
        img_int = np.clip(img_int, 0, bins - 1)
        matched = lut[img_int]
    
    return matched.astype(np.float32)


def tone_map_images(images: List[np.ndarray], 
                    method: str = "zscore_ratio",
                    reference_idx: Optional[int] = None) -> List[np.ndarray]:
    if len(images) == 0:
        return []
    
    if len(images) == 1:
        img_float = images[0].astype(np.float32) / 255.0 if images[0].dtype == np.uint8 else images[0]
        return [np.clip(img_float, 0.0, 1.0)]
    
    tone_mapped = []
    
    if method == "zscore":
        global_means, global_stds = compute_global_statistics(images)
        
        for img in images:
            local_means, local_stds = compute_channel_statistics(img)
            
            normalized = apply_zscore_normalization(
                img, local_means, local_stds, global_means, global_stds
            )
            
            normalized_uint8 = (normalized * 255.0).astype(np.uint8)
            normalized_float = normalized_uint8.astype(np.float32) / 255.0
            
            tone_mapped.append(normalized_float)
    
    elif method == "zscore_ratio":
        global_means, global_stds = compute_global_statistics(images)
        global_ratios = compute_global_channel_ratios(images)
        
        for img in images:
            local_means, local_stds = compute_channel_statistics(img)
            local_ratios = compute_channel_ratios(img)
            
            normalized = apply_zscore_normalization(
                img, local_means, local_stds, global_means, global_stds
            )
            
            corrected = apply_channel_ratio_correction(
                normalized, local_ratios, global_ratios
            )
            
            corrected_uint8 = (corrected * 255.0).astype(np.uint8)
            corrected_float = corrected_uint8.astype(np.float32) / 255.0
            
            tone_mapped.append(corrected_float)
    
    elif method == "lab":
        if reference_idx is None:
            reference_idx = len(images) // 2
        
        reference_img = images[reference_idx]
        reference_float = reference_img.astype(np.float32) / 255.0 if reference_img.dtype == np.uint8 else reference_img
        reference_lab = rgb_to_lab(reference_float)
        
        for img in images:
            corrected = apply_lab_color_correction(img, reference_lab)
            tone_mapped.append(corrected)
    
    elif method == "zscore_lab":
        global_means, global_stds = compute_global_statistics(images)
        
        if reference_idx is None:
            reference_idx = len(images) // 2
        
        reference_img = images[reference_idx]
        reference_float = reference_img.astype(np.float32) / 255.0 if reference_img.dtype == np.uint8 else reference_img
        reference_lab = rgb_to_lab(reference_float)
        
        for img in images:
            local_means, local_stds = compute_channel_statistics(img)
            normalized = apply_zscore_normalization(
                img, local_means, local_stds, global_means, global_stds
            )
            
            corrected = apply_lab_color_correction(normalized, reference_lab)
            tone_mapped.append(corrected)
    
    elif method == "exposure":
        for img in images:
            exposure_factor = compute_exposure_compensation(img)
            adjusted = apply_exposure_compensation(img, exposure_factor)
            tone_mapped.append(adjusted)
    
    elif method == "gamma":
        for img in images:
            img_float = img.astype(np.float32) / 255.0 if img.dtype == np.uint8 else img
            corrected = apply_gamma_correction(img_float)
            tone_mapped.append(corrected)
    
    elif method == "histogram":
        if reference_idx is None:
            reference_idx = len(images) // 2
        
        reference_img = images[reference_idx]
        target_hist = compute_histogram(reference_img)
        
        for img in images:
            matched = match_histogram(img, target_hist)
            tone_mapped.append(matched)
    
    elif method == "exposure_gamma":
        for img in images:
            exposure_factor = compute_exposure_compensation(img)
            adjusted = apply_exposure_compensation(img, exposure_factor)
            
            corrected = apply_gamma_correction(adjusted)
            tone_mapped.append(corrected)
    
    else:
        for img in images:
            img_float = img.astype(np.float32) / 255.0 if img.dtype == np.uint8 else img
            tone_mapped.append(np.clip(img_float, 0.0, 1.0))
    
    return tone_mapped

