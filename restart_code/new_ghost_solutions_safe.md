# Ghost 현상 해결 - 새로운 안전한 방법

## 현재 상황
- 경계 영역 적응형 가중치 지수 적용 중 (임계값 0.6, 지수 3.5)
- 가중 평균 블렌딩 사용 중
- 기존 스티칭 방식 유지 필요

## 지금까지 시도한 방법 (실패)
1. 가중치 차이 기반 선택적 블렌딩 - 이미지 망가짐
2. RANSAC threshold 강화 - 더 심해짐
3. Affine Fallback - 메모리 할당 실패

## 새로운 안전한 방법 제안

### 방안 1: 경계 영역에서 거리 기반 가중치 보정 (이미지 가장자리)

**원리:**
- 경계 영역에서 이미지 가장자리까지의 거리를 고려하여 가중치 추가 보정
- 가장자리에 가까울수록 가중치를 더 빠르게 감소
- 기존 블렌딩 방식 유지하면서 경계부 가중치만 강화

**방법:**
```python
# 이미지 가장자리까지의 거리 계산
edge_dist_x = np.minimum(x_img_filtered, W_img - x_img_filtered)
edge_dist_y = np.minimum(y_img_filtered, H_img - y_img_filtered)
edge_dist = np.minimum(edge_dist_x, edge_dist_y)
normalized_edge_dist = edge_dist / (min(W_img, H_img) / 2.0)

# 경계 영역에서만 가장자리 거리 기반 보정 적용
boundary_correction = np.power(normalized_edge_dist, 0.3)  # 가장자리에 가까울수록 가중치 감소
current_weights = base_weights * np.where(is_boundary, boundary_correction, 1.0)
```

**장점:**
- 구현 간단
- 현재 블렌딩 방식 완전히 유지
- 경계부 가중치 강화
- 안전함 (기존 방식 유지)

**예상 효과:** Ghost 현상 30-50% 감소

---

### 방안 2: 경계 영역에서 가중치 지수 점진적 증가

**원리:**
- 경계 영역에서 normalized_distances에 따라 가중치 지수를 점진적으로 증가
- 경계에 가까울수록 더 강한 가중치 감소
- 기존 블렌딩 방식 유지

**방법:**
```python
# 경계 영역에서 점진적 지수 증가
boundary_exp = 2.5 + (normalized_distances - 0.6) * 5.0  # 0.6~1.0 범위에서 2.5~4.5
boundary_exp = np.clip(boundary_exp, 2.5, 4.5)
current_weights = np.where(
    is_boundary,
    np.power(1.0 - normalized_distances, boundary_exp),
    np.power(1.0 - normalized_distances, center_exp)
)
```

**장점:**
- 구현 간단
- 현재 블렌딩 방식 완전히 유지
- 경계부 가중치 강화
- 안전함

**예상 효과:** Ghost 현상 30-50% 감소

---

### 방안 3: 경계 영역에서 픽셀 차이 기반 조건부 블렌딩

**원리:**
- 경계 영역에서 기존 픽셀과 새 픽셀의 차이가 작을 때만 블렌딩
- 차이가 크면 기존 픽셀 유지 (Ghost 현상 방지)
- 기존 블렌딩 방식 유지하되, 경계 영역에서만 조건부 적용

**방법:**
```python
# 경계 영역에서 픽셀 차이 계산
if is_boundary.any():
    pixel_diff = np.abs(existing_pixels - value)
    if len(images[i].shape) == 3:
        pixel_diff = np.mean(pixel_diff, axis=1)  # RGB 평균
    diff_threshold = 0.15  # 픽셀 차이 임계값 (0~1 범위)
    should_blend = pixel_diff < diff_threshold
    
    # 차이가 작을 때만 블렌딩, 차이가 크면 기존 픽셀 유지
    blended = np.where(
        should_blend[:, np.newaxis] if len(images[i].shape) == 3 else should_blend,
        blended,  # 블렌딩
        existing_pixels  # 기존 픽셀 유지
    )
```

**장점:**
- Ghost 현상 강력 감소 (60-80%)
- 기존 블렌딩 방식 유지
- 경계 영역에서만 조건부 적용

**단점:**
- 구현 복잡도 중간
- 픽셀 차이 임계값 조정 필요

**예상 효과:** Ghost 현상 60-80% 감소

---

### 방안 4: 경계 영역에서 가중치 비율 기반 조건부 블렌딩 (보수적)

**원리:**
- 경계 영역에서 새 가중치가 기존 가중치보다 충분히 높을 때만 블렌딩
- 가중치 비율이 작으면 기존 픽셀 유지
- 더 보수적인 임계값 사용 (0.15 → 0.3)

**방법:**
```python
# 경계 영역에서만 가중치 비율 기반 조건부 블렌딩
if is_boundary.any():
    weight_ratio = current_weights / (existing_weights + 1e-10)
    weight_ratio_threshold = 0.3  # 보수적 임계값
    should_blend = weight_ratio > weight_ratio_threshold
    
    # 새 가중치가 충분히 높을 때만 블렌딩
    blended = np.where(
        should_blend[:, np.newaxis] if len(images[i].shape) == 3 else should_blend,
        blended,  # 블렌딩
        existing_pixels  # 기존 픽셀 유지
    )
```

**장점:**
- Ghost 현상 강력 감소 (50-70%)
- 기존 블렌딩 방식 유지
- 보수적 접근으로 안전함

**예상 효과:** Ghost 현상 50-70% 감소

---

### 방안 5: 경계 영역에서 가중치 지수 적응형 증가 (거리 기반)

**원리:**
- 경계 영역에서 normalized_distances와 이미지 가장자리 거리를 모두 고려
- 두 거리를 결합하여 가중치 지수 적응형 증가
- 기존 블렌딩 방식 유지

**방법:**
```python
# 이미지 가장자리 거리 계산
edge_dist = np.minimum(
    np.minimum(x_img_filtered, W_img - x_img_filtered),
    np.minimum(y_img_filtered, H_img - y_img_filtered)
)
normalized_edge_dist = edge_dist / (min(W_img, H_img) / 2.0)

# 경계 영역에서 두 거리 결합
boundary_factor = (normalized_distances - 0.6) * 0.4 + (1.0 - normalized_edge_dist) * 0.6
boundary_exp = 2.5 + boundary_factor * 2.0  # 2.5 ~ 4.5
boundary_exp = np.clip(boundary_exp, 2.5, 4.5)
```

**장점:**
- 구현 간단
- 현재 블렌딩 방식 완전히 유지
- 경계부 가중치 강화
- 안전함

**예상 효과:** Ghost 현상 40-60% 감소

---

## 권장 방안

### 1순위: 방안 1 (경계 영역에서 거리 기반 가중치 보정)

**이유:**
- 가장 안전하고 간단
- 현재 블렌딩 방식 완전히 유지
- 경계부 가중치 강화
- 구현 시간 최소

**구현 난이도:** ⭐ (매우 쉬움)

---

### 2순위: 방안 2 (경계 영역에서 가중치 지수 점진적 증가)

**이유:**
- 구현 간단
- 현재 블렌딩 방식 완전히 유지
- 경계부 가중치 강화

**구현 난이도:** ⭐ (매우 쉬움)

---

### 3순위: 방안 3 (경계 영역에서 픽셀 차이 기반 조건부 블렌딩)

**이유:**
- Ghost 현상 강력 감소 (60-80%)
- 기존 블렌딩 방식 유지
- 경계 영역에서만 조건부 적용

**구현 난이도:** ⭐⭐ (쉬움)

---

## 하이브리드 접근

### 방안 1 + 방안 2
- 거리 기반 가중치 보정 + 가중치 지수 점진적 증가
- 강력한 Ghost 현상 방지

---

## 결론

**가장 추천하는 방법:**
1. **방안 1 (거리 기반 가중치 보정)**: 가장 안전하고 간단
2. **방안 2 (가중치 지수 점진적 증가)**: 구현 간단, 효과적
3. **방안 3 (픽셀 차이 기반 조건부 블렌딩)**: 가장 효과적, 구현 중간

