"""
Tone Mapping 모듈
이미지 톤 매핑 (선택적 구현, 추가 점수)
"""

import numpy as np
from typing import Optional


def apply_tone_mapping(image: np.ndarray, method: str = 'gamma') -> np.ndarray:
    """
    Tone Mapping을 적용합니다.
    
    Args:
        image: 이미지 배열 (H, W, 3) - uint8
        method: 매핑 방법 ('gamma', 기본값: 'gamma')
    
    Returns:
        mapped: 톤 매핑된 이미지 (H, W, 3) - uint8
    """
    if method == 'gamma':
        # Gamma correction: 이미지 밝기 보정
        # 감마 값 2.2는 표준 디스플레이 감마 (sRGB 표준)
        gamma = 2.2
        # 0-255 범위를 0.0-1.0으로 정규화
        normalized = image.astype(np.float32) / 255.0
        # 감마 보정: I_corrected = I_normalized^(1/gamma)
        # 감마 > 1이면 어두운 영역을 밝게, 밝은 영역을 어둡게 조정
        corrected = np.power(normalized, 1.0 / gamma)
        # 0.0-1.0 범위를 0-255로 되돌리고 클리핑
        # np.clip: 값의 범위를 [0, 255]로 제한
        mapped = np.clip(corrected * 255.0, 0, 255).astype(np.uint8)
    else:
        # 다른 방법은 구현되지 않음 (원본 반환)
        mapped = image.copy()
    
    return mapped

