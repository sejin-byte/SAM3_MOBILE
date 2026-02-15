# SAM3_M4 품질 개선 실행안 (Phase 2 연장)

> **업데이트**: `2026-02-15`  
> **최근 리포트**: `docs/RETRAINING_PERFORMANCE_ANALYSIS_20260215.md`  
> **현재 단계**: **Phase 2 Extension (학습량 증대)**

---

## 1) 현재 상태 요약 (2026-02-15 기준)

| 항목 | Baseline (2/12) | **Current (2/15)** | 상태 |
|---|---:|---:|:---:|
| **mIoU** (Mask Quality) | 0.0605 | **0.0898** | ✅ **회복세 (Teacher의 59%)** |
| **Prompt Sensitivity** | 0.0015 | **0.1343** | ✅ **해결됨 (텍스트 반응성 확보)** |
| **Presence F1** | 0.0000 | 0.0000 | ⚠️ **지표 불능 (데이터셋 한계)** |
| **학습 단계** | Phase 2 (Old) | Phase 2 (New) 1차 완료 | **추가 학습 필요** |

**진단**:
- "From-Scratch" Phase 1+2 재학습이 1차 완료되었습니다.
- **구조적 문제(Tokenizer 충돌)는 해결**되었으나, mIoU(0.09)는 아직 목표치(0.12+)에 도달하지 못했습니다.
- 이는 "새로운 구조"에 대한 학습량이 절대적으로 부족하기 때문입니다.

---

## 2) 완료된 조치

| 조치 항목 | 결과 |
|---|---|
| **Phase 0** | 지표 신뢰성 확보 (Teacher Baseline, Prompt Sensitivity) |
| **Phase 1** | Tokenizer 분리, From-Scratch 재학습 (Feature Alignment) |
| **Phase 2 (1차)** | 기본 Epoch 수행 완료 → **mIoU 0.09 달성** |

---

## 3) 남은 과제 및 Next Steps (Phase 2 Extension ~ Phase 3)

### 우선순위 1: Phase 2 연장 학습 (Epoch 증대)

현재 완료된 Phase 2 체크포인트(`student_phase2_video_merged...`)를 시작점으로 **추가 학습**을 수행합니다.

- **전략**: `resume` 옵션을 사용하여 현재 상태에서 학습을 이어갑니다.
- **Action**: Phase 2 추가 5~10 Epoch 수행.
- **목표**: mIoU 0.12 이상 안착.

### 우선순위 2: Presence 지표 활성화 (Data Refinement)

- **Action**: 학습 데이터 로더에 **5%의 Empty Image** (또는 Noise Image) 추가.
- **기대 효과**: Presence Head가 실제로 동작하기 시작하여 F1 Score가 0.0에서 유의미한 수치로 변화.

### 우선순위 3: On-Device, 실시간 비디오 테스트

- **Action**: iOS/Android 앱 빌드 및 포팅.
- **검증**: 카메라 프리뷰에서의 FPS 및 발열, 마스크 실시간 반응성(지터링 여부) 확인.

---

## 4) 실행 로드맵 (수정됨)

| 단계 | 목표 | 주요 작업 | 예상 일정 |
|---|---|---|---|
| **Phase 1+2 (1차 완료)** | 구조적 결함 해결 | From-scratch 재학습 | **Done (2/15)** |
| **Phase 2 Extension** | 마스크 품질 극대화 | **Resume 후 추가 학습 (Epoch 5~10)** | **~2/17** |
| **Phase 3** | 실전 배포 최적화 | Negative Sample, On-Device 포팅 | **~2/19** |

---

## 5) 결론

> **"구조 개선은 통과했습니다. 이제 근육을 키울 시간입니다."**

Phase 2가 완료되었지만, 성능이 정점에 도달하지 않았으므로 **"연장전(Extension)"**에 돌입합니다.
별도의 코드 수정 없이, **학습 시간만 늘려도** 상당한 성능 향상이 기대됩니다.
