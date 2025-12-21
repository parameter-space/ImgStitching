# 보고서에 추가해야 할 구현 항목

## 현재 코드에 구현되어 있지만 보고서에 반영되지 않은 항목들

### 1. 이미지 전처리 (Preprocessing)

**구현 위치:** `preprocessing.py`

-   **Gaussian Blur 적용**: 노이즈 제거를 위한 가우시안 필터링
-   **수동 구현**: OpenCV의 `cv2.GaussianBlur` 대신 NumPy로 직접 구현
-   **커널 생성**: 2D 가우시안 커널 수동 생성 (`gaussian_kernel`)
-   **2D Convolution**: 수동 구현 (`_convolve_2d`)
-   **이미지 정규화**: 0~1 범위로 정규화

**보고서 작성 예시:**

```
### 2.1 이미지 전처리
노이즈 제거를 위해 가우시안 필터를 적용하였다.
가우시안 커널을 수동으로 생성하고, 2D Convolution을 NumPy로 직접 구현하여
OpenCV의 가우시안 블러 함수를 사용하지 않고 전처리를 수행하였다.
```

---

### 2. Tone Mapping (색감 및 밝기 보정)

**구현 위치:** `tone_mapping.py`

-   **Z-score 정규화**: 채널별 평균과 표준편차를 이용한 정규화
-   **채널 비율 보정**: R/G, B/G 비율을 이용한 색온도 보정
-   **LAB 색공간 보정**: RGB → LAB 변환 후 색온도/색조 보정
-   **히스토그램 매칭**: 히스토그램 매칭을 통한 색감 통일
-   **다중 방법 지원**: zscore, zscore_ratio, lab, zscore_lab, histogram 등

**보고서 작성 예시:**

```
### 2.2 Tone Mapping
여러 이미지 간의 노출 차이와 색감 차이를 보정하기 위해 Tone Mapping을 적용하였다.
- Z-score 정규화: 각 이미지의 RGB 채널별 평균과 표준편차를 계산하여 전역 통계로 정규화
- 채널 비율 보정: R/G, B/G 비율을 계산하여 색온도 차이 보정
- LAB 색공간 보정: RGB를 LAB 색공간으로 변환하여 색온도와 색조 차이 보정
- 히스토그램 매칭: 참조 이미지의 히스토그램에 맞춰 색감 통일
```

---

### 3. 적응형 하이브리드 전략 (Adaptive Hybrid Strategy)

**구현 위치:** `main.py` (line 202-295)

-   **Scale 기반 적응**: Homography의 scale factor에 따라 Projective/Affine 비율 조정
-   **Perspective 기반 적응**: Perspective 성분에 따라 적응적 조정
-   **선형 보간**: Projective와 Affine을 alpha 비율로 보간
-   **안전장치**: Inlier 개수가 적을 때 Affine 비중 증가

**보고서 작성 예시:**

```
### 3.3 적응형 하이브리드 전략
Projective Homography의 과도한 왜곡(Scale Explosion)을 방지하기 위해
Scale과 Perspective 성분에 따라 Projective와 Affine을 적응적으로 혼합하였다.

- Scale 기반 적응: Scale factor가 1.1 이하면 Projective, 2.0 이상이면 Affine 사용
- Perspective 기반 적응: Perspective 성분이 0.0005 이하면 Projective, 0.001 이상이면 Affine 사용
- 선형 보간: 두 지표 중 더 위험한 신호를 따라 alpha 값을 결정하고,
  H_interp = (1 - alpha) * H_proj + alpha * H_aff로 보간
```

---

### 4. Center-Reference Global Homography

**구현 위치:** `stitching.py` (line 62-198)

-   **중앙 이미지 기준**: 중앙 이미지를 기준으로 양방향으로 누적
-   **오차 분산**: 양방향 누적으로 오차 분산
-   **정규화 강화**: 누적 과정에서 주기적 정규화
-   **Determinant 검증**: 수치적 안정성을 위한 determinant 검증

**보고서 작성 예시:**

```
### 4.1 Center-Reference Global Homography
모든 이미지를 중앙 이미지 좌표계로 변환하기 위해 Center-Reference 방식을 사용하였다.
중앙 이미지를 기준으로 왼쪽과 오른쪽으로 각각 누적하여 오차를 분산시켰다.

- 왼쪽 누적: H_to_center = H[center_idx]^-1 * H[center_idx-1]^-1 * ... * H[img_idx]^-1
- 오른쪽 누적: H_to_center = H[center_idx] * H[center_idx+1] * ... * H[img_idx-1]
- 정규화: 각 누적 단계에서 H[2,2]로 정규화하여 수치적 안정성 확보
```

---

### 5. Auto-Scaling Canvas

**구현 위치:** `stitching.py` (line 480-520)

-   **동적 크기 조정**: 실제 필요한 크기에 따라 Canvas 크기 자동 조정
-   **메모리 보호**: 최대 크기 제한으로 메모리 폭발 방지
-   **Padding 적응**: 이미지 범위에 따라 적응적 padding 적용

**보고서 작성 예시:**

```
### 4.2 Auto-Scaling Canvas
메모리 효율성을 위해 실제 이미지 범위에 맞춰 Canvas 크기를 자동으로 조정하였다.
실제 필요한 크기를 계산하고, 최대 크기 제한(25000픽셀)을 두어 메모리 폭발을 방지하였다.
```

---

### 6. 경계 영역 적응형 가중치 블렌딩

**구현 위치:** `stitching.py` (line 920-940)

-   **경계 영역 감지**: normalized_distances > 0.6으로 경계 영역 판단
-   **적응형 가중치 지수**: 경계 영역은 지수 3.5, 중심부는 지수 2.5
-   **Ghost 현상 방지**: 경계 영역에서 가중치가 더 빠르게 감소하여 Ghost 현상 감소

**보고서 작성 예시:**

```
### 4.3 경계 영역 적응형 가중치 블렌딩
이미지 경계에서 발생하는 Ghost 현상을 줄이기 위해 경계 영역에서 가중치 지수를 높게 적용하였다.

- 경계 영역: normalized_distances > 0.6인 영역에서 가중치 지수 3.5 적용
- 중심부: normalized_distances <= 0.6인 영역에서 가중치 지수 2.5 적용
- 수식: weights = (1.0 - normalized_distances)^exp
  - 경계 영역: exp = 3.5
  - 중심부: exp = 2.5
```

---

### 7. Normalized DLT

**구현 위치:** `homography.py` (line 10-50, 53-118)

-   **점 정규화**: 평균 거리를 sqrt(2)로 만들도록 정규화
-   **Denormalization**: 정규화된 Homography를 원본 좌표계로 변환

**보고서 작성 예시:**

```
### 3.1 Normalized DLT
수치적 안정성을 위해 Normalized DLT 알고리즘을 사용하였다.
점들을 정규화하여 평균 거리를 sqrt(2)로 만들고,
정규화된 좌표계에서 Homography를 계산한 후 원본 좌표계로 역변환하였다.
```

---

### 8. Affine Homography 계산

**구현 위치:** `homography.py` (line 149-230)

-   **Affine 변환**: Perspective 성분이 없는 변환 (6 DOF)
-   **Normalized DLT**: Affine Homography도 Normalized DLT로 계산

**보고서 작성 예시:**

```
### 3.4 Affine Homography 계산
Perspective 성분이 없는 Affine Homography를 계산하여
Projective Homography와 보간에 사용하였다.
Affine 변환은 6개의 자유도를 가지며, Normalized DLT 알고리즘으로 계산하였다.
```

---

### 9. 벡터화된 이미지 스티칭

**구현 위치:** `stitching.py` (line 700-960)

-   **NumPy 벡터화**: Python for 루프 대신 NumPy 벡터화 사용
-   **메모리 효율성**: 벡터화를 통한 계산 속도 향상

**보고서 작성 예시:**

```
### 4.4 벡터화된 이미지 스티칭
성능 향상을 위해 NumPy 벡터화를 사용하여 이미지 스티칭을 구현하였다.
픽셀 단위 for 루프 대신 벡터 연산을 사용하여 계산 속도를 크게 향상시켰다.
```

---

### 10. 수치적 안정성 개선

**구현 위치:** 여러 파일

-   **정규화 강화**: Homography 누적 과정에서 주기적 정규화
-   **Determinant 검증**: 비정상적인 Homography 검증
-   **Scale 검증**: x와 y scale을 독립적으로 검증하여 부분적으로 비정상적인 경우도 허용

**보고서 작성 예시:**

```
### 4.5 수치적 안정성 개선
Homography 누적 과정에서 발생하는 수치 오차를 줄이기 위해
각 누적 단계에서 정규화를 수행하고, determinant와 scale을 검증하여
비정상적인 Homography를 제외하였다.
```

---

### 11. Ratio Test

**구현 위치:** `point_matching.py` (line 83-133)

-   **첫 번째와 두 번째 NCC 값 비교**: 모호성 낮은 매칭만 선택
-   **임계값**: 0.75 (기본값)

**보고서 작성 예시:**

```
### 2.3 Ratio Test
단순히 가장 높은 NCC 값을 선택하는 것이 아니라,
첫 번째로 높은 값과 두 번째로 높은 값의 비율을 확인하여
모호성이 낮은 매칭만 선택하였다. 비율이 0.75보다 작은 경우에만
최종 매칭 쌍으로 확정하였다.
```

---

## 보고서 구조 제안

### 1. 서론

-   파노라마 이미지 스티칭의 목적과 배경

### 2. 관련 연구 및 이론

-   Harris Corner Detection
-   Feature Matching (NCC, Ratio Test)
-   Homography (DLT, Normalized DLT)
-   RANSAC
-   Tone Mapping

### 3. 구현 방법

-   3.1 이미지 전처리 (Gaussian Blur)
-   3.2 Harris Corner Detection
-   3.3 Feature Matching (NCC, Ratio Test)
-   3.4 Homography 계산 (Normalized DLT, Affine, 적응형 하이브리드 전략)
-   3.5 RANSAC
-   3.6 Tone Mapping (Z-score, 채널 비율, LAB, 히스토그램 매칭)
-   3.7 Global Homography (Center-Reference)
-   3.8 이미지 스티칭 (Auto-Scaling Canvas, 경계 영역 적응형 가중치 블렌딩, 벡터화)

### 4. 실험 및 결과

-   4.1 실험 환경
-   4.2 실험 결과
-   4.3 성능 분석

### 5. 결론

-   구현 내용 요약
-   개선 사항 및 한계
