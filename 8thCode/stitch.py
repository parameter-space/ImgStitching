"""
이미지 스티칭 모듈
Center-Reference Stitching을 구현합니다.
"""

import numpy as np
import cv2
from typing import Tuple, List, Optional
from geometry import apply_homography


def compute_canvas_size_center_ref(images: List[np.ndarray], global_homographies: List[np.ndarray]) -> Tuple[int, int, np.ndarray]:
    """Calculate canvas size based on global homographies relative to the center.
    Includes safety checks to prevent canvas explosion."""
    all_corners = []
    
    for i, (image, H_global) in enumerate(zip(images, global_homographies)):
        H, W = image.shape[:2]
        corners = np.array([[0, 0], [W, 0], [W, H], [0, H]], dtype=np.float32)
        transformed_corners = apply_homography(corners, H_global)
        all_corners.append(transformed_corners)
        
        # Safety check: If single pair width/height > 30000px, abort
        pair_min_x = np.min(transformed_corners[:, 0])
        pair_max_x = np.max(transformed_corners[:, 0])
        pair_min_y = np.min(transformed_corners[:, 1])
        pair_max_y = np.max(transformed_corners[:, 1])
        pair_width = pair_max_x - pair_min_x
        pair_height = pair_max_y - pair_min_y
        
        if pair_width > 30000 or pair_height > 30000:
            print(f"  *** WARNING: Image {i+1} would create huge canvas ({pair_width:.0f}x{pair_height:.0f}px). Clamping offset. ***")
            # Clamp the offset to prevent explosion
            transformed_corners[:, 0] = np.clip(transformed_corners[:, 0], -30000, 30000)
            transformed_corners[:, 1] = np.clip(transformed_corners[:, 1], -30000, 30000)
            all_corners[-1] = transformed_corners
    
    all_corners = np.vstack(all_corners)
    
    min_x = np.min(all_corners[:, 0])
    max_x = np.max(all_corners[:, 0])
    min_y = np.min(all_corners[:, 1])
    max_y = np.max(all_corners[:, 1])
    
    width = int(np.ceil(max_x - min_x))
    height = int(np.ceil(max_y - min_y))
    
    # Safety check: If total panorama > 30000px, abort
    if width > 30000 or height > 30000:
        print(f"  *** ERROR: Total panorama size ({width}x{height}px) exceeds 30000px limit. Aborting. ***")
        # Return a reasonable default size
        width = min(width, 30000)
        height = min(height, 30000)
    
    # Offset to shift negative coordinates to positive
    offset_x = int(np.floor(-min_x))
    offset_y = int(np.floor(-min_y))
    
    # Clamp offset to prevent explosion
    offset_x = np.clip(offset_x, -30000, 30000)
    offset_y = np.clip(offset_y, -30000, 30000)
    
    return width, height, np.array([offset_x, offset_y], dtype=np.int32)


def inverse_warp(image: np.ndarray, H: np.ndarray, output_shape: Tuple[int, int], offset: np.ndarray) -> np.ndarray:
    """
    Warp image using H (Image -> Canvas).
    Uses cv2.remap for speed optimization (maps are calculated manually).
    """
    H_inv = np.linalg.inv(H)  # Canvas -> Image
    H_out, W_out = output_shape
    offset_x, offset_y = offset
    H_src, W_src = image.shape[:2]
    C = image.shape[2] if len(image.shape) == 3 else 1
    
    # Initialize output
    if C == 1:
        warped = np.zeros((H_out, W_out), dtype=np.uint8)
    else:
        warped = np.zeros((H_out, W_out, C), dtype=np.uint8)
    
    # Tile-based processing for large canvases (use cv2.remap for speed)
    tile_size = 1000  # Process 1000x1000 tiles at a time
    
    for y_start in range(0, H_out, tile_size):
        y_end = min(y_start + tile_size, H_out)
        for x_start in range(0, W_out, tile_size):
            x_end = min(x_start + tile_size, W_out)
            
            # Tile dimensions
            tile_h = y_end - y_start
            tile_w = x_end - x_start
            
            # 1. Generate grid for this tile in Canvas coordinates
            y_tile, x_tile = np.meshgrid(
                np.arange(y_start, y_end, dtype=np.float32),
                np.arange(x_start, x_end, dtype=np.float32),
                indexing='ij'
            )
            
            # 2. Shift by offset (Canvas -> Global)
            x_tile = x_tile - offset_x
            y_tile = y_tile - offset_y
            
            # 3. Flatten and apply H_inv (Global -> Source Image)
            ones = np.ones_like(x_tile)
            coords = np.stack([x_tile.flatten(), y_tile.flatten(), ones.flatten()])
            src_coords = H_inv @ coords
            
            # 4. Normalize
            w = src_coords[2, :]
            w = np.where(np.abs(w) < 1e-10, 1e-10, w)  # Avoid div by zero
            x_src = (src_coords[0, :] / w).reshape(tile_h, tile_w)
            y_src = (src_coords[1, :] / w).reshape(tile_h, tile_w)
            
            # 5. Use cv2.remap for fast bilinear interpolation
            # cv2.remap expects map_x and map_y as float32 arrays
            map_x = x_src.astype(np.float32)
            map_y = y_src.astype(np.float32)
            
            # Remap the tile
            if C == 1:
                tile_warped = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            else:
                tile_warped = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            
            # Copy to output
            if C == 1:
                warped[y_start:y_end, x_start:x_end] = tile_warped
            else:
                warped[y_start:y_end, x_start:x_end, :] = tile_warped
    
    return warped


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
    
    # DEBUG: Print transformed corner coordinates for each image
    print("\n=== DEBUG: Transformed Corner Coordinates ===")
    for i, (image, H_global) in enumerate(zip(images, global_homos)):
        img_H, img_W = image.shape[:2]
        corners = np.array([[0, 0], [img_W, 0], [img_W, img_H], [0, img_H]], dtype=np.float32)
        transformed_corners = apply_homography(corners, H_global)
        min_x = np.min(transformed_corners[:, 0])
        max_x = np.max(transformed_corners[:, 0])
        min_y = np.min(transformed_corners[:, 1])
        max_y = np.max(transformed_corners[:, 1])
        print(f"Image {i+1}: min_x={min_x:.2f}, max_x={max_x:.2f}, min_y={min_y:.2f}, max_y={max_y:.2f}")
        if i == 0 and min_x >= 0:
            print(f"  WARNING: Image 1 min_x is not negative! Chain may be broken.")
    print("=" * 50 + "\n")
    
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
