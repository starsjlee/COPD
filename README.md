# COPD vs HD 분류 파이프라인

## 데이터 구조
```
data/
  HD/
    20240101_XXXXXX_1X1.csv
    20240101_XXXXXX_4X4.csv
    20240101_XXXXXXGAS.csv
    ...
  COPD/
    20240102_XXXXXX_1X1.csv
    ...
```
세션 1개 = 같은 날짜/시간 prefix를 가진 3개 파일 (_1X1, _4X4, GAS)

---

## 센서 구성
| 파일 | 센서 | Feature 수 |
|------|------|-----------|
| _1X1.csv | RGB 20채널 (1×1 픽셀) | 60 |
| _4X4.csv | RGB 16채널 (4×4 픽셀) | 48 |
| GAS.csv  | TGS/전기화학/MEMS 가스 센서 | 33 |

`Measuring` 단계 데이터만 사용 (Stabilizing 제외)

---

## 설치

```bash
pip install numpy pandas scikit-learn xgboost matplotlib seaborn

# PyTorch GPU (CUDA 12.1)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CUDA 버전 확인
nvidia-smi
```

---

## 실행

```bash
# ML + DL 둘 다 (기본)
python COPD_pipeline/train_v3.py --data_dir ./data --output_dir ./results --mode both

# ML만 (빠름)
python COPD_pipeline/train_v3.py --data_dir ./data --mode ml

# DL만
python COPD_pipeline/train_v3.py --data_dir ./data --mode dl --epochs 80 --batch_size 16

# 주요 옵션
--seq_len    900      # Measuring 시퀀스 고정 길이
--epochs     60       # DL 학습 epoch
--batch_size 8        # 배치 크기 (GPU 메모리에 맞게 조절)
--n_splits   5        # K-Fold 수
--lr         0.001    # 학습률
```

---

## 출력 파일 (results/)
| 파일 | 내용 |
|------|------|
| feature_matrix.csv | ML feature 행렬 전체 |
| ml_evaluation.png | XGBoost confusion matrix + ROC curve |
| ml_feature_importance.png | 상위 30개 중요 feature |
| dl_fold{N}_curve.png | 각 fold 학습 곡선 |
| dl_evaluation.png | CNN confusion matrix + ROC curve |
| dl_fold{N}_best.pt | 각 fold 최고 모델 가중치 |
| results_summary.csv | 최종 성능 요약 |

---

## 모델 설명

### XGBoost (ML)
- Measuring 구간에서 통계 feature 추출 (mean, std, min, max, IQR, skewness, kurtosis, slope, AUC)
- 3종 센서 × 각 채널 → 총 ~1,400개 feature
- GPU 가속: `tree_method='gpu_hist'`
- 5-Fold Stratified CV + Test AUC 보고

### Multi-Sensor 1D-CNN (DL)
- 3개 독립 Branch (각 센서별 1D-CNN)
- 각 Branch: Conv1d → BN → GELU → MaxPool → AdaptiveAvgPool
- Fusion: 3개 branch embedding concat → FC head
- GPU: CUDA AMP (mixed precision) 자동 적용
- Class imbalance: WeightedRandomSampler
- 5-Fold Stratified CV
