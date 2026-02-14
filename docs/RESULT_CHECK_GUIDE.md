# Result Check Guide (Step 2~5 완료 후)

이 문서는 `scripts/post_video_stage2_to_5.sh` 실행이 끝난 뒤,
"무엇을 어디서 확인해야 하는지"를 순서대로 안내합니다.

## 0) 작업 폴더 이동

```bash
cd /Users/sejinkim/developers/SAM3_M4
```

---

## 1) 생성 파일 존재 확인 (가장 먼저)

### 1-1. 양자화 체크포인트

```bash
ls -lh checkpoints/quantized/quantized_int8_int4.pt
```

정상 기준:
- 파일이 존재해야 함
- 대략 수백 MB 크기 (현재 환경 예: 약 300MB대)

### 1-2. 병합 체크포인트(학습+비디오 모듈 결합본)

```bash
ls -lt checkpoints/final/student_phase2_video_merged_*.pt | head -n 1
```

정상 기준:
- 최신 파일 1개 이상 존재
- 파일 크기가 약 1GB대여도 정상

### 1-3. 양자화 로그

```bash
ls -lt logs/quantization/compare_*.log | head -n 1
```

정상 기준:
- 최신 `compare_*.log` 파일이 존재

---

## 2) 양자화 로그 내용 확인

최신 로그 파일을 변수로 잡고 핵심 줄만 봅니다.

```bash
LATEST_QLOG=$(ls -t logs/quantization/compare_*.log | head -n 1)
echo "$LATEST_QLOG"
rg -n "Comparison Summary|Saved:|ERROR: quantization failed|Done\\." "$LATEST_QLOG"
```

해석 가이드:
- `Saved: checkpoints/quantized/quantized_int8_int4.pt`가 보이면 산출물 저장 성공
- `Comparison Summary`가 보이면 비교 단계 완료
- `int4` 관련 에러(`mslk`)가 있어도 `int8_int4`가 저장되었으면 현재 파이프라인 목적상 진행 가능

---

## 3) 시각 검증 결과 확인 (가장 중요)

### 3-1. 생성된 시각 검증 폴더 목록 확인

```bash
ls -d outputs/visual_eval/* | tail -n 20
```

정상 기준:
- `outputs/visual_eval/<run>_<prompt>/` 형태 폴더들이 다수 생성됨

### 3-2. 특정 프롬프트 결과 열기

예: `building` 프롬프트 결과를 브라우저로 열기

```bash
open outputs/visual_eval/20260211_235743_building/index.html
```

또는 `open` 대신 Finder로 직접 들어가서 `index.html` 더블클릭.

### 3-3. 무엇을 눈으로 보면 되는가

- 누락: 객체가 빠지지 않았는지
- 과분할: 배경까지 과도하게 마스크되지 않는지
- 프롬프트 반응성: `person`, `car`, `building` 등에서 결과 차이가 나는지
- 얇은/작은 객체: 경계가 너무 뭉개지지 않는지

---

## 4) Stage5 상태 확인

Stage5는 이제 실제 export 실행이 가능한 상태입니다.

### 4-1. 배포 커맨드 시트 확인

```bash
bash scripts/stage5_deploy_commands.sh
```

### 4-2. iOS(CoreML) .pte 생성

```bash
LATEST_MERGED=$(ls -t checkpoints/final/student_phase2_video_merged_*.pt | head -n 1)
python scripts/export_executorch.py \
  --checkpoint "$LATEST_MERGED" \
  --backend coreml \
  --output artifacts/executorch/mobile_sam3_ios_coreml.pte
```

정상 기준:
- `Saved: artifacts/executorch/mobile_sam3_ios_coreml.pte` 출력
- `artifacts/executorch/mobile_sam3_ios_coreml.pte.meta.json` 생성

### 4-3. Android(QNN) .pte 생성

```bash
python scripts/export_executorch.py \
  --checkpoint "$LATEST_MERGED" \
  --backend qnn \
  --output artifacts/executorch/mobile_sam3_android_qnn.pte
```

현재 Mac 환경에서는 QNN 파티셔너 모듈이 없어서 다음 문구와 함께 자동 fallback 될 수 있습니다.
- `WARN: QNN partition failed ... Falling back to backend=none`

fallback이어도 `.pte` 생성되면 Stage5 파이프라인 자체는 정상입니다.

### 4-4. 산출물 확인

```bash
ls -lh artifacts/executorch/mobile_sam3_ios_coreml.pte artifacts/executorch/mobile_sam3_android_qnn.pte
```

참고:
- `quantized_int8_int4.pt`를 직접 넣으면 현재 툴체인 제약으로 FP merged ckpt로 자동 fallback될 수 있음
- 해당 여부는 `.meta.json`의 `quantized_export_fallback` 필드로 확인 가능

---

## 5) 지금 당장 해야 할 다음 행동

1. `outputs/visual_eval/.../index.html` 여러 개를 열어 품질 확인
2. `scripts/export_executorch.py`로 iOS/Android용 `.pte` 생성
3. 생성된 `.pte`를 앱 프로젝트에 넣어 on-device latency 측정 시작
