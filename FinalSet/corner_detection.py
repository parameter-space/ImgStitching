"""
Harris Corner Detection 구현
Corner.pdf 강의자료 참고
"""

import numpy as np


def compute_image_gradients(image: np.ndarray) -> tuple:
    H, W = image.shape
    
    padded = np.pad(image, ((1, 1), (1, 1)), mode='edge')
    
    Ix = -1 * padded[0:H, 0:W] + 1 * padded[0:H, 2:W+2] + \
         -2 * padded[1:H+1, 0:W] + 2 * padded[1:H+1, 2:W+2] + \
         -1 * padded[2:H+2, 0:W] + 1 * padded[2:H+2, 2:W+2]
    Ix = Ix / 4.0
    
    Iy = -1 * padded[0:H, 0:W] - 2 * padded[0:H, 1:W+1] - 1 * padded[0:H, 2:W+2] + \
          1 * padded[2:H+2, 0:W] + 2 * padded[2:H+2, 1:W+1] + 1 * padded[2:H+2, 2:W+2]
    Iy = Iy / 4.0
    
    return Ix, Iy


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    center = size // 2
    x, y = np.meshgrid(np.arange(size) - center, np.arange(size) - center)
    
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = kernel / np.sum(kernel)
    
    return kernel.astype(np.float32)


def harris_corner_detection(image: np.ndarray, 
                            threshold: float = 0.01,
                            k: float = 0.04,
                            window_size: int = 3,
                            sigma: float = 1.0) -> np.ndarray:
    """
    Harris Corner Detection: R = det(M) - k * trace(M)^2
    """
    Ix, Iy = compute_image_gradients(image)
    
    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    Ixy = Ix * Iy
    
    gaussian = gaussian_kernel(window_size, sigma)
    pad_size = window_size // 2
    
    Ix2_padded = np.pad(Ix2, pad_size, mode='constant')
    Iy2_padded = np.pad(Iy2, pad_size, mode='constant')
    Ixy_padded = np.pad(Ixy, pad_size, mode='constant')
    
    H, W = image.shape
    M11 = np.zeros((H, W), dtype=np.float32)
    M12 = np.zeros((H, W), dtype=np.float32)
    M22 = np.zeros((H, W), dtype=np.float32)
    
    for i in range(H):
        for j in range(W):
            patch_x2 = Ix2_padded[i:i+window_size, j:j+window_size]
            patch_y2 = Iy2_padded[i:i+window_size, j:j+window_size]
            patch_xy = Ixy_padded[i:i+window_size, j:j+window_size]
            
            M11[i, j] = np.sum(patch_x2 * gaussian)
            M12[i, j] = np.sum(patch_xy * gaussian)
            M22[i, j] = np.sum(patch_y2 * gaussian)
    
    det_M = M11 * M22 - M12 * M12
    trace_M = M11 + M22
    R = det_M - k * trace_M * trace_M
    
    corner_mask = R > threshold
    
    nms_size = 3
    nms_pad = nms_size // 2
    R_padded = np.pad(R, nms_pad, mode='constant', constant_values=-np.inf)
    
    corners_list = []
    corner_responses = []
    
    for i in range(H):
        for j in range(W):
            if corner_mask[i, j]:
                neighborhood = R_padded[i:i+nms_size, j:j+nms_size]
                max_val = np.max(neighborhood)
                
                if R[i, j] >= max_val - 1e-6:
                    corners_list.append([j, i])
                    corner_responses.append(R[i, j])
    
    if len(corners_list) == 0:
        return np.array([], dtype=np.float32).reshape(0, 2)
    
    MAX_CORNERS = 1000
    
    if len(corners_list) > MAX_CORNERS:
        corner_responses = np.array(corner_responses, dtype=np.float32)
        sorted_indices = np.argsort(corner_responses)[::-1]
        selected_indices = sorted_indices[:MAX_CORNERS]
        
        corners_list = [corners_list[i] for i in selected_indices]
    
    corners = np.array(corners_list, dtype=np.float32)
    
    return corners

