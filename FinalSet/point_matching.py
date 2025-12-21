"""
Point Matching 구현
Point_Matching.pdf 강의자료 참고
"""

import numpy as np
from typing import List, Tuple


def compute_descriptors(image: np.ndarray, corners: np.ndarray, patch_size: int = 21) -> np.ndarray:
    H, W = image.shape
    half_size = patch_size // 2
    
    padded_image = np.pad(image, half_size, mode='constant', constant_values=0)
    
    descriptors = []
    
    for corner in corners:
        x, y = int(corner[0]), int(corner[1])
        
        if x < half_size or x >= W - half_size or y < half_size or y >= H - half_size:
            descriptor = np.zeros(patch_size * patch_size, dtype=np.float32)
        else:
            patch = padded_image[y:y+patch_size, x:x+patch_size]
            descriptor = patch.flatten().astype(np.float32)
        
        descriptors.append(descriptor)
    
    return np.array(descriptors, dtype=np.float32)


def compute_ncc(descriptor1: np.ndarray, descriptor2: np.ndarray) -> float:
    """
    NCC = sum((d1 - mean1) * (d2 - mean2)) / (std1 * std2 * N)
    """
    mean1 = np.mean(descriptor1)
    mean2 = np.mean(descriptor2)
    
    d1_centered = descriptor1 - mean1
    d2_centered = descriptor2 - mean2
    
    std1 = np.std(descriptor1)
    std2 = np.std(descriptor2)
    
    if std1 < 1e-10 or std2 < 1e-10:
        return 0.0
    
    ncc = np.sum(d1_centered * d2_centered) / (std1 * std2 * len(descriptor1))
    
    return float(ncc)


def match_features(descriptors1: np.ndarray, 
                   descriptors2: np.ndarray,
                   threshold: float = 0.7) -> List[Tuple[int, int]]:
    """
    Ratio test: ratio = best2_ncc / best_ncc < threshold인 경우만 매칭
    """
    N1 = len(descriptors1)
    N2 = len(descriptors2)
    
    matches = []
    
    for i in range(N1):
        ncc_scores = []
        
        for j in range(N2):
            ncc = compute_ncc(descriptors1[i], descriptors2[j])
            ncc_scores.append(ncc)
        
        ncc_scores = np.array(ncc_scores)
        
        sorted_indices = np.argsort(ncc_scores)[::-1]
        
        if len(sorted_indices) < 2:
            continue
        
        best_idx = sorted_indices[0]
        second_best_idx = sorted_indices[1]
        
        best_ncc = ncc_scores[best_idx]
        second_best_ncc = ncc_scores[second_best_idx]
        
        if best_ncc > 0.5 and second_best_ncc > 0:
            ratio = second_best_ncc / best_ncc
            
            if ratio < threshold:
                matches.append((i, best_idx))
    
    return matches


def get_matched_points(corners1: np.ndarray, 
                       corners2: np.ndarray, 
                       matches: List[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
    points1 = []
    points2 = []
    
    for idx1, idx2 in matches:
        points1.append(corners1[idx1])
        points2.append(corners2[idx2])
    
    if len(points1) == 0:
        return np.array([], dtype=np.float32).reshape(0, 2), np.array([], dtype=np.float32).reshape(0, 2)
    
    return np.array(points1, dtype=np.float32), np.array(points2, dtype=np.float32)

