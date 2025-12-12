"""
이미지 스티칭 모듈
Center-Reference Stitching을 구현합니다.
"""

import numpy as np
import cv2
from typing import Tuple, List, Optional
from geometry import apply_homography


def compute_canvas_size_center_ref(images: List[np.ndarray], global_homographies: List[np.ndarray]) -> Tuple[int, int, np.ndarray]:
    """Calculate canvas size based on global homographies relative to the center."""
    all_corners = []
    
    for i, (image, H_global) in enumerate(zip(images, global_homographies)):
        H, W = image.shape[:2]
        corners = np.array([[0, 0], [W, 0], [W, H], [0, H]], dtype=np.float32)
        transformed_corners = apply_homography(corners, H_global)
        all_corners.append(transformed_corners)
    
    all_corners = np.vstack(all_corners)
    
    min_x = np.min(all_corners[:, 0])
    max_x = np.max(all_corners[:, 0])
    min_y = np.min(all_corners[:, 1])
    max_y = np.max(all_corners[:, 1])
    
    width = int(np.ceil(max_x - min_x))
    height = int(np.ceil(max_y - min_y))
    
    # Offset to shift negative coordinates to positive
    offset_x = int(np.floor(-min_x))
    offset_y = int(np.floor(-min_y))
    
    return width, height, np.array([offset_x, offset_y], dtype=np.int32)


def inverse_warp(image: np.ndarray, H: np.ndarray, output_shape: Tuple[int, int], offset: np.ndarray) -> np.ndarray:
    """
    Warp image using H (Image -> Canvas).
    """
    H_inv = np.linalg.inv(H)  # Canvas -> Image
    H_out, W_out = output_shape
    offset_x, offset_y = offset
    
    # 1. Generate grid in Canvas coordinates
    y_out, x_out = np.indices((H_out, W_out), dtype=np.float32)
    
    # 2. Shift by offset (Canvas -> Global)
    x_out -= offset_x
    y_out -= offset_y
    
    # 3. Flatten and apply H_inv (Global -> Source Image)
    ones = np.ones_like(x_out)
    coords = np.stack([x_out.flatten(), y_out.flatten(), ones.flatten()])
    src_coords = H_inv @ coords
    
    # 4. Normalize
    w = src_coords[2, :]
    w = np.where(np.abs(w) < 1e-10, 1e-10, w)  # Avoid div by zero
    x_src = (src_coords[0, :] / w).reshape(H_out, W_out)
    y_src = (src_coords[1, :] / w).reshape(H_out, W_out)
    
    # 5. Remap (Interpolation)
    # Mask for valid boundaries
    H_src, W_src = image.shape[:2]
    mask = (x_src >= 0) & (x_src < W_src - 1) & (y_src >= 0) & (y_src < H_src - 1)
    
    # Bilinear Interpolation (Manual)
    x0 = np.floor(x_src).astype(np.int32)
    x1 = x0 + 1
    y0 = np.floor(y_src).astype(np.int32)
    y1 = y0 + 1
    
    x0 = np.clip(x0, 0, W_src-1)
    x1 = np.clip(x1, 0, W_src-1)
    y0 = np.clip(y0, 0, H_src-1)
    y1 = np.clip(y1, 0, H_src-1)
    
    alpha = x_src - x0
    beta = y_src - y0
    
    if len(image.shape) == 3:
        alpha = alpha[..., np.newaxis]
        beta = beta[..., np.newaxis]
        mask = mask[..., np.newaxis]
        
    Ia = image[y0, x0]
    Ib = image[y1, x0]
    Ic = image[y0, x1]
    Id = image[y1, x1]
    
    warped = (1-alpha)*(1-beta)*Ia + (1-alpha)*beta*Ib + alpha*(1-beta)*Ic + alpha*beta*Id
    warped = np.clip(warped, 0, 255).astype(np.uint8)
    
    return warped * mask.astype(np.uint8)


def stitch_multiple_images(images: List[np.ndarray], homographies: List[np.ndarray]) -> np.ndarray:
    """
    Main stitching function using Center-Reference.
    """
    if not images:
        return None
    
    mid = len(images) // 2
    global_homos = [None] * len(images)
    global_homos[mid] = np.eye(3, dtype=np.float32)
    
    # Propagate Right (mid -> end)
    # homographies[i] transforms img[i+1] -> img[i]
    for i in range(mid, len(images) - 1):
        # We want H_{i+1 -> mid}
        # We have H_{i -> mid} (global_homos[i])
        # And H_{i+1 -> i} (homographies[i])
        # H_{i+1 -> mid} = H_{i -> mid} @ H_{i+1 -> i}
        global_homos[i+1] = global_homos[i] @ homographies[i]
        
    # Propagate Left (mid -> 0)
    for i in range(mid - 1, -1, -1):
        # We want H_{i -> mid}
        # We have H_{i+1 -> mid} (global_homos[i+1])
        # And H_{i+1 -> i} (homographies[i])
        # We need H_{i -> i+1} = inv(homographies[i])
        # H_{i -> mid} = H_{i+1 -> mid} @ H_{i -> i+1}
        H_inv = np.linalg.inv(homographies[i])
        global_homos[i] = global_homos[i+1] @ H_inv

    # Compute Canvas
    W, H, offset = compute_canvas_size_center_ref(images, global_homos)
    print(f"Canvas Size: {W}x{H}, Offset: {offset}")
    
    # Warp and Blend (Simple Max Blending to avoid ghosting for now)
    panorama = np.zeros((H, W, 3), dtype=np.uint8)
    
    # Process center first, then outwards to layer correctly
    indices = [mid]
    for i in range(1, len(images)):
        if mid - i >= 0:
            indices.append(mid - i)
        if mid + i < len(images):
            indices.append(mid + i)
            
    for i in indices:
        print(f"Warping image {i+1}...")
        warped = inverse_warp(images[i], global_homos[i], (H, W), offset)
        
        # Simple masking and overlay
        mask = np.any(warped > 0, axis=2)
        panorama[mask] = warped[mask]  # Naive overwrite
    
    return panorama


# For backward compatibility with main.py
def compute_canvas_size(images: list, homographies: list) -> Tuple[int, int, np.ndarray]:
    """
    Legacy function for backward compatibility.
    This function is not used in Center-Reference Stitching but kept for compatibility.
    """
    # This is a placeholder - actual implementation would use the old linear chaining
    # For now, return dummy values (should not be called)
    return 1000, 1000, np.array([0, 0], dtype=np.int32)
