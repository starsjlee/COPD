"""
COPD vs HD Classification Pipeline v3
======================================
v2 대비 개선 사항 (데이터 분석 기반):

[DL 모델]
1. GAS branch: AdaptiveAvgPool → Temporal Attention
   - GAS SNR=182 (1X1/4X4의 150배), 초반 집중형 신호 구조
   - 시간 구간별 가중치를 학습해 초반 반응 정보 보존

2. 1X1 / 4X4 branch 경량화
   - SNR=1.2~1.4 (사실상 노이즈), 채널 간 다양성 0.001
   - 파라미터 낭비 방지: 64→32→64 채널 축소

3. 센서 타입별 독립 정규화 (SensorDataset)
   - GAS 채널 스케일 편차가 수백 배 → 채널별 z-score 유지하되
     센서 타입(TGS/NE4/SS/Flow 등) 그룹별로 클리핑 적용

4. Branch별 가중치 학습 (FusionGate)
   - 단순 concat 대신 branch 기여도를 학습 가능한 gate로 조절
   - GAS가 실제로 더 중요한 경우 자동으로 높은 가중치

[ML Feature Engineering]
5. GAS 초반 구간 feature 추가
   - 앞 20% 구간 mean/std/slope를 별도 feature로 추출
   - GAS 초반에 신호 집중 (std 앞=4.47 → 뒤=0.27)

6. GAS 변화율 feature 추가
   - 앞 20% vs 뒤 20% 평균 비율 (relative_drop)
   - 초반→후반 변화 방향이 COPD/HD 분류에 유효할 것으로 예상

Usage:
    python train_v3.py --data_dir ./data --mode both --output_dir ./results
"""

import os, re, argparse, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from datetime import datetime
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, accuracy_score)
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler

# ──────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────
PROCESS_PHASE = "Measuring"
ENCODING      = "cp949"
FOLDER_RE     = re.compile(r'^(CO|HD)(\d+)-\d+_(M\d+)$')

# GAS 채널 그룹 (센서 타입별 클리핑 임계값)
GAS_CLIP_GROUPS = {
    "flow":  (["Flow 1", "Flow 2", "Flow 3", "Flow 4"], 5.0),   # Flow는 안정적 → 5σ
    "tgs":   (["TGS2444","TGS2600","TGS2603","TGS2602",
                "TGS2610","TGS2611","TGS2612","TGS2620"], 4.0),
    "ne4":   (["NE4HCHO","NE4H2S","NE4NH3","NE4NO2","NE4CO","NE4NO"], 3.0),  # 변동 큼
    "other": (None, 4.0),   # 나머지 채널
}



# ──────────────────────────────────────────────
# 1. 데이터 로딩 & 구조 파싱 (v2와 동일)
# ──────────────────────────────────────────────

def parse_folder(name: str):
    m = FOLDER_RE.match(name)
    if not m:
        return None
    prefix  = m.group(1)
    pat_num = m.group(2)
    session = m.group(3)
    label   = "COPD" if prefix == "CO" else "HD"
    pat_id  = prefix + pat_num
    return label, pat_id, session


def find_sensor_files(folder: Path):
    files = list(folder.glob("*.csv"))
    result = {}
    for f in files:
        stem = f.stem.upper()
        if re.search(r"(?:_|\s)?1X1$", stem):
            result["1X1"] = f
        elif re.search(r"(?:_|\s)?4X4$", stem):
            result["4X4"] = f
        elif "GAS" in stem:
            result["GAS"] = f
    return result if result else None


def load_csv(path: Path) -> pd.DataFrame:
    for enc in [ENCODING, "utf-8", "utf-8-sig"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, Exception):
            continue
    return pd.DataFrame()


def get_measuring(df: pd.DataFrame) -> pd.DataFrame:
    if "Process" in df.columns:
        df = df[df["Process"] == PROCESS_PHASE].copy()
    df = df.drop(columns=["Time", "Process"], errors="ignore")
    df = df.select_dtypes(include=[np.number])
    df = df.dropna(axis=1, how="all")
    return df.reset_index(drop=True)


def scan_dataset(data_dir: Path):
    patients     = defaultdict(lambda: {"label": None, "sessions": []})
    gas_col_sets = []

    for label_dir in ["COPD", "HD"]:
        folder = data_dir / label_dir
        if not folder.exists():
            print(f"  ⚠️  {folder} 없음, 건너뜀")
            continue
        for sub in sorted(folder.iterdir()):
            if not sub.is_dir():
                continue
            parsed = parse_folder(sub.name)
            if parsed is None:
                continue
            label, pat_id, session = parsed
            sensor_files = find_sensor_files(sub)
            if sensor_files is None:
                print(f"  ⚠️  불완전한 세션: {sub} — 건너뜀")
                continue
            patients[pat_id]["label"] = label
            patients[pat_id]["sessions"].append(sensor_files)
            if "GAS" in sensor_files:
                df_gas = load_csv(sensor_files["GAS"])
                df_m   = get_measuring(df_gas)
                if len(df_m) > 0:
                    gas_col_sets.append(set(df_m.columns))

    gas_common = sorted(set.intersection(*gas_col_sets)) if gas_col_sets else []

    print(f"\n📋 스캔 결과:")
    n_copd = sum(1 for v in patients.values() if v["label"] == "COPD")
    n_hd   = sum(1 for v in patients.values() if v["label"] == "HD")
    print(f"  COPD 환자: {n_copd}명")
    print(f"  HD 환자  : {n_hd}명")
    print(f"  GAS 공통 컬럼: {len(gas_common)}개")
    total_sessions = sum(len(v["sessions"]) for v in patients.values())
    print(f"  총 세션 수: {total_sessions}개 (환자당 평균 {total_sessions/len(patients):.1f}개)")

    return dict(patients), gas_common


def make_output_dir(base_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


# ──────────────────────────────────────────────
# 2. Feature Engineering (ML) — v3 개선
# ──────────────────────────────────────────────

def extract_features(df: pd.DataFrame, prefix: str, col_filter=None,
                     is_gas: bool = False) -> dict:
    """
    시계열 → 통계 feature.
    is_gas=True 일 때:
      - 초반 20% 구간 별도 feature 추출  [개선 ⑤]
      - 초반/후반 변화율 feature 추출     [개선 ⑥]
    """
    feats = {}
    cols = col_filter if col_filter else list(df.columns)
    for col in cols:
        if col not in df.columns:
            continue
        s = df[col].dropna().values
        if len(s) == 0:
            continue
        tag = f"{prefix}__{col}"

        # 전구간 통계 (v2와 동일)
        feats[f"{tag}__mean"]  = np.mean(s)
        feats[f"{tag}__std"]   = np.std(s)
        feats[f"{tag}__min"]   = np.min(s)
        feats[f"{tag}__max"]   = np.max(s)
        feats[f"{tag}__range"] = np.max(s) - np.min(s)
        feats[f"{tag}__q25"]   = np.percentile(s, 25)
        feats[f"{tag}__q75"]   = np.percentile(s, 75)
        feats[f"{tag}__iqr"]   = np.percentile(s, 75) - np.percentile(s, 25)
        feats[f"{tag}__skew"]  = float(pd.Series(s).skew())
        feats[f"{tag}__kurt"]  = float(pd.Series(s).kurt())
        if len(s) > 1:
            feats[f"{tag}__slope"] = float(np.polyfit(np.arange(len(s)), s, 1)[0])
        feats[f"{tag}__auc"]   = float(np.trapezoid(s) if hasattr(np, 'trapezoid') else np.trapz(s))

        # ── GAS 전용 추가 feature ─────────────────
        if is_gas and len(s) >= 10:
            n = len(s)
            seg = max(1, n // 5)       # 20% 구간

            s_early = s[:seg]
            s_late  = s[n - seg:]

            # 초반 구간 통계 [개선 ⑤]
            feats[f"{tag}__early_mean"]  = np.mean(s_early)
            feats[f"{tag}__early_std"]   = np.std(s_early)
            if len(s_early) > 1:
                feats[f"{tag}__early_slope"] = float(
                    np.polyfit(np.arange(len(s_early)), s_early, 1)[0])

            # 초반→후반 변화율 [개선 ⑥]
            denom = abs(np.mean(s_early)) + 1e-8
            feats[f"{tag}__relative_drop"] = (np.mean(s_late) - np.mean(s_early)) / denom

            # 피크 위치 (신호가 최대인 시점 / 전체 길이)
            feats[f"{tag}__peak_pos"] = float(np.argmax(np.abs(s))) / n

    return feats


def build_feature_matrix(patients: dict, gas_common_cols: list):
    rows, labels, ids = [], [], []

    for pat_id, info in sorted(patients.items()):
        label_val = 1 if info["label"] == "COPD" else 0
        sessions  = info["sessions"]

        dfs = {"1X1": [], "4X4": [], "GAS": []}
        ok = True
        for sfiles in sessions:
            for key in ["1X1", "4X4", "GAS"]:
                if key not in sfiles:
                    continue
                df_raw = load_csv(sfiles[key])
                df_m   = get_measuring(df_raw)
                if len(df_m) < 5:
                    ok = False
                    break
                dfs[key].append(df_m)
            if not ok:
                break

        if not ok:
            print(f"  ⚠️  {pat_id}: Measuring 데이터 부족 → 건너뜀")
            continue

        row = {}
        if dfs["1X1"]:
            row.update(extract_features(pd.concat(dfs["1X1"], ignore_index=True), "s1x1"))
        if dfs["4X4"]:
            row.update(extract_features(pd.concat(dfs["4X4"], ignore_index=True), "s4x4"))
        if dfs["GAS"]:
            row.update(extract_features(pd.concat(dfs["GAS"],  ignore_index=True), "gas",
                                        col_filter=gas_common_cols, is_gas=True))
        if not row:
            print(f"  ⚠️  {pat_id}: 유효한 센서 feature 없음 → 건너뜀")
            continue
        rows.append(row)
        labels.append(label_val)
        ids.append(pat_id)

    df_feat = pd.DataFrame(rows).fillna(0)
    return df_feat, np.array(labels), ids


# ──────────────────────────────────────────────
# 3. 시계열 텐서 변환 (DL) — 정규화 개선
# ──────────────────────────────────────────────

def _get_gas_clip_sigma(col: str) -> float:
    """채널 이름으로 클리핑 sigma 결정."""
    for group_name, (cols, sigma) in GAS_CLIP_GROUPS.items():
        if cols is None:
            continue
        if col in cols:
            return sigma
    return GAS_CLIP_GROUPS["other"][1]


def normalize_sensor(arr: np.ndarray, is_gas: bool = False,
                     col_names: list = None) -> np.ndarray:
    """
    (C, T) 배열 정규화.
    - 기본: 채널별 z-score (v2와 동일)
    - GAS: 채널별 z-score 후 센서 타입별 sigma 클리핑  [개선 ③]
    """
    mu = arr.mean(axis=1, keepdims=True)
    sd = arr.std(axis=1, keepdims=True) + 1e-8
    norm = (arr - mu) / sd

    if is_gas and col_names is not None:
        for i, col in enumerate(col_names):
            sigma = _get_gas_clip_sigma(col)
            norm[i] = np.clip(norm[i], -sigma, sigma)

    return norm


def patient_to_tensors(info: dict, gas_common_cols: list, seq_len: int):
    dfs = {"1X1": [], "4X4": [], "GAS": []}
    for sfiles in info["sessions"]:
        for key in ["1X1", "4X4", "GAS"]:
            if key not in sfiles:
                continue
            df_raw = load_csv(sfiles[key])
            df_m   = get_measuring(df_raw)
            if len(df_m) > 0:
                dfs[key].append(df_m)

    tensors     = []
    col_names_list = []

    for key, col_filter in [("1X1", None), ("4X4", None), ("GAS", gas_common_cols)]:
        if not dfs[key]:
            return None
        df_cat = pd.concat(dfs[key], ignore_index=True)
        if col_filter:
            df_cat = df_cat[[c for c in col_filter if c in df_cat.columns]]
        df_cat = df_cat.fillna(0)
        col_names = list(df_cat.columns)
        arr = df_cat.values.T.astype(np.float32)    # (C, T)
        T = arr.shape[1]
        if T >= seq_len:
            arr = arr[:, :seq_len]
        else:
            arr = np.pad(arr, ((0, 0), (0, seq_len - T)), mode="edge")
        tensors.append(arr)
        col_names_list.append(col_names)

    return tensors, col_names_list


class SensorDataset(Dataset):
    def __init__(self, samples):
        # samples: list of ((tensors, col_names_list), label)
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        (tensors, col_names_list), label = self.samples[idx]
        norm = []
        for i, arr in enumerate(tensors):
            is_gas    = (i == 2)
            col_names = col_names_list[i] if is_gas else None
            normed    = normalize_sensor(arr, is_gas=is_gas, col_names=col_names)
            norm.append(torch.tensor(normed))
        return norm, torch.tensor(label, dtype=torch.long)


def collate_fn(batch):
    tensors_list, labels = zip(*batch)
    t1 = torch.stack([t[0] for t in tensors_list])
    t2 = torch.stack([t[1] for t in tensors_list])
    t3 = torch.stack([t[2] for t in tensors_list])
    return (t1, t2, t3), torch.stack(labels)


# ──────────────────────────────────────────────
# 4. DL 모델 — v3 개선
# ──────────────────────────────────────────────

class TemporalAttention(nn.Module):
    """
    [개선 ①] GAS branch용 Temporal Attention.
    Conv feature map (B, C, T) → attention 가중치로 weighted sum.
    초반 반응 구간을 선택적으로 집중할 수 있게 함.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(channels, channels // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(channels // 4, 1, kernel_size=1),  # (B, 1, T)
        )

    def forward(self, x):
        # x: (B, C, T)
        w = torch.softmax(self.attn(x), dim=-1)   # (B, 1, T)
        return (x * w).sum(dim=-1)                 # (B, C)


class LightBranch(nn.Module):
    """
    [개선 ②] 1X1 / 4X4용 경량 branch.
    SNR이 낮아 깊은 네트워크가 오히려 노이즈를 과적합할 수 있음.
    파라미터를 줄여 정규화 효과.
    """
    def __init__(self, in_channels: int, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32,  kernel_size=7, padding=3),
            nn.BatchNorm1d(32),  nn.GELU(),
            nn.Conv1d(32,  64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),  nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),  nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(64, out_dim)

    def forward(self, x):
        return self.proj(self.net(x).squeeze(-1))


class GASBranch(nn.Module):
    """
    [개선 ①] GAS 전용 branch with Temporal Attention.
    SNR이 높고 초반 집중형 신호 → attention이 유효.
    """
    def __init__(self, in_channels: int, out_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64,  kernel_size=7, padding=3),
            nn.BatchNorm1d(64),  nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.GELU(),
        )
        self.attn = TemporalAttention(256)      # AdaptiveAvgPool 대체
        self.proj = nn.Linear(256, out_dim)

    def forward(self, x):
        feat = self.conv(x)             # (B, 256, T//2)
        pooled = self.attn(feat)        # (B, 256)
        return self.proj(pooled)        # (B, out_dim)


class FusionGate(nn.Module):
    """
    [개선 ④] Branch 기여도 학습.
    단순 concat 대신 각 branch 임베딩에 학습 가능한 스칼라 게이트 적용.
    GAS가 실제로 더 중요할 경우 자동으로 높은 가중치를 부여함.
    """
    def __init__(self, n_branches: int, dim: int):
        super().__init__()
        # 각 branch마다 dim-dim linear → gating
        self.gates = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
            for _ in range(n_branches)
        ])

    def forward(self, embeddings: list):
        gated = [gate(e) * e for gate, e in zip(self.gates, embeddings)]
        return torch.cat(gated, dim=1)


class MultiSensorCNN(nn.Module):
    def __init__(self, ch_1x1, ch_4x4, ch_gas,
                 light_dim=64, gas_dim=128, num_classes=2, dropout=0.4):
        super().__init__()
        self.b1 = LightBranch(ch_1x1, light_dim)   # 경량 [개선 ②]
        self.b2 = LightBranch(ch_4x4, light_dim)   # 경량 [개선 ②]
        self.b3 = GASBranch(ch_gas, gas_dim)        # Attention [개선 ①]

        self.fusion = FusionGate(                   # 게이트 [개선 ④]
            n_branches=3,
            dim=light_dim  # b1, b2는 light_dim; b3는 gas_dim — 별도 처리
        )
        # FusionGate는 모두 같은 dim을 기대하므로 b3 출력을 light_dim으로 맞춤
        self.gas_proj = nn.Linear(gas_dim, light_dim)

        total_dim = light_dim * 3
        self.head = nn.Sequential(
            nn.Linear(total_dim, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 64),        nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x1, x2, x3):
        e1 = self.b1(x1)
        e2 = self.b2(x2)
        e3 = self.gas_proj(self.b3(x3))
        fused = self.fusion([e1, e2, e3])
        return self.head(fused)


# ──────────────────────────────────────────────
# 5. 학습 루프 (v2와 동일 구조, 모델 교체)
# ──────────────────────────────────────────────

def train_dl(samples, device, output_dir: Path,
             epochs=60, batch_size=8, lr=1e-3, n_splits=5):

    dataset    = SensorDataset(samples)
    labels_all = [s[1] for s in samples]
    skf        = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    indices    = np.arange(len(dataset))

    ch_1x1 = samples[0][0][0][0].shape[0]
    ch_4x4 = samples[0][0][0][1].shape[0]
    ch_gas  = samples[0][0][0][2].shape[0]

    fold_metrics = []
    all_probs, all_trues = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(indices, labels_all)):
        print(f"\n  [DL] Fold {fold+1}/{n_splits}  (train={len(train_idx)}, val={len(val_idx)})")

        train_sub = torch.utils.data.Subset(dataset, train_idx)
        val_sub   = torch.utils.data.Subset(dataset, val_idx)

        train_labels = [labels_all[i] for i in train_idx]
        class_counts = np.bincount(train_labels)
        weights  = [1.0 / class_counts[l] for l in train_labels]
        sampler  = WeightedRandomSampler(weights, len(weights), replacement=True)

        train_loader = DataLoader(train_sub, batch_size=batch_size,
                                  sampler=sampler, collate_fn=collate_fn,
                                  num_workers=2, pin_memory=(device.type == "cuda"))
        val_loader   = DataLoader(val_sub, batch_size=batch_size,
                                  shuffle=False, collate_fn=collate_fn,
                                  num_workers=2, pin_memory=(device.type == "cuda"))

        model     = MultiSensorCNN(ch_1x1, ch_4x4, ch_gas).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()
        scaler    = GradScaler(enabled=(device.type == "cuda"))

        best_val_acc = 0
        best_probs_fold, best_trues_fold = [], []
        train_losses, val_accs = [], []

        for epoch in range(epochs):
            model.train()
            ep_loss = 0
            for (x1, x2, x3), y in train_loader:
                x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)
                optimizer.zero_grad()
                with autocast(enabled=(device.type == "cuda")):
                    loss = criterion(model(x1, x2, x3), y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                ep_loss += loss.item()
            scheduler.step()
            train_losses.append(ep_loss / len(train_loader))

            model.eval()
            preds, trues, probs = [], [], []
            with torch.no_grad():
                for (x1, x2, x3), y in val_loader:
                    x1, x2, x3 = x1.to(device), x2.to(device), x3.to(device)
                    out = model(x1, x2, x3)
                    probs.extend(torch.softmax(out, 1)[:, 1].cpu().numpy())
                    preds.extend(out.argmax(1).cpu().numpy())
                    trues.extend(y.numpy())

            val_acc = accuracy_score(trues, preds)
            val_accs.append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_probs_fold, best_trues_fold = probs[:], trues[:]
                torch.save(model.state_dict(), output_dir / f"dl_fold{fold+1}_best.pt")

            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1:3d} | loss={train_losses[-1]:.4f} | val_acc={val_acc:.3f}")

        fold_metrics.append(best_val_acc)
        all_probs.extend(best_probs_fold)
        all_trues.extend(best_trues_fold)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(train_losses); ax1.set_title(f"Fold {fold+1} Loss"); ax1.set_xlabel("Epoch")
        ax2.plot(val_accs);     ax2.set_title(f"Fold {fold+1} Val Acc"); ax2.set_xlabel("Epoch")
        plt.tight_layout()
        plt.savefig(output_dir / f"dl_fold{fold+1}_curve.png", dpi=100)
        plt.close()

    print(f"\n  [DL] {n_splits}-Fold CV Acc: {np.mean(fold_metrics):.3f} ± {np.std(fold_metrics):.3f}")
    return fold_metrics, all_probs, all_trues


# ──────────────────────────────────────────────
# 6. 결과 시각화
# ──────────────────────────────────────────────

def plot_evaluation(y_true, y_pred, y_prob, label_names, title, output_dir, prefix):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names, ax=axes[0])
    axes[0].set_title("Confusion Matrix")
    axes[0].set_ylabel("True"); axes[0].set_xlabel("Predicted")

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    axes[1].plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR")
    axes[1].set_title("ROC Curve"); axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_evaluation.png", dpi=120)
    plt.close()
    return auc


def plot_feature_importance(model, output_dir, top_n=30):
    imp = model.get_booster().get_fscore()
    if not imp:
        return
    df_imp = pd.DataFrame({"feature": list(imp.keys()),
                           "importance": list(imp.values())})
    df_imp = df_imp.sort_values("importance", ascending=False).head(top_n)
    df_imp["short"] = df_imp["feature"].apply(
        lambda x: "__".join(x.split("__")[1:3]) if x.count("__") >= 2 else x)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(data=df_imp, x="importance", y="short", ax=ax, palette="viridis")
    ax.set_title(f"XGBoost Top-{top_n} Feature Importance")
    ax.set_xlabel("F-Score"); ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(output_dir / "ml_feature_importance.png", dpi=120)
    plt.close()


# ──────────────────────────────────────────────
# 7. Main
# ──────────────────────────────────────────────

def main(args):
    data_dir   = Path(args.data_dir)
    output_base = Path(args.output_dir)
    output_dir = make_output_dir(output_base)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("⚠️  GPU 없음, CPU 실행")

    print("\n📂 데이터 스캔 중...")
    patients, gas_common_cols = scan_dataset(data_dir)

    if len(patients) == 0:
        print("❌ 환자 데이터 없음. --data_dir 경로를 확인하세요.")
        return

    label_names     = ["HD", "COPD"]
    results_summary = {}

    # ── ML ──────────────────────────────────────
    if args.mode in ("ml", "both"):
        print("\n" + "="*55)
        print("🌲 ML 분석 (XGBoost + Statistical Features v3)")
        print("="*55)

        print("  Feature 추출 중 (GAS 시간 구간 feature 포함)...")
        df_feat, y, patient_ids = build_feature_matrix(patients, gas_common_cols)
        X = df_feat.values
        print(f"  Feature matrix: {X.shape}  (HD={sum(y==0)}, COPD={sum(y==1)})")

        df_save = df_feat.copy()
        df_save.insert(0, "patient_id", patient_ids)
        df_save.insert(1, "label", ["COPD" if v == 1 else "HD" for v in y])
        df_save.to_csv(output_dir / "feature_matrix.csv", index=False)

        scale_pos = max(sum(y==0), sum(y==1)) / min(sum(y==0), sum(y==1))
        xgb_params = {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": scale_pos,
            "eval_metric": "logloss",
            "tree_method": "gpu_hist" if device.type == "cuda" else "hist",
            "device": "cuda" if device.type == "cuda" else "cpu",
            "random_state": 42,
        }
        xgb_clf = xgb.XGBClassifier(**xgb_params)

        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
        try:
            cv_scores = cross_val_score(xgb_clf, X, y, cv=skf, scoring="roc_auc")
        except ValueError as exc:
            if "gpu_hist" in str(exc) or "gpu_hist" in repr(exc):
                print("  ⚠️  현재 XGBoost 빌드에서 GPU를 지원하지 않습니다. CPU hist로 재시도합니다.")
                xgb_params["tree_method"] = "hist"
                xgb_params["device"] = "cpu"
                xgb_clf = xgb.XGBClassifier(**xgb_params)
                cv_scores = cross_val_score(xgb_clf, X, y, cv=skf, scoring="roc_auc")
            else:
                raise
        print(f"  {args.n_splits}-Fold CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

        print("  전체 CV 예측 평가 중...")
        cv_pred = cross_val_predict(xgb_clf, X, y, cv=skf, method="predict")
        cv_prob = cross_val_predict(xgb_clf, X, y, cv=skf, method="predict_proba")[:, 1]
        auc_cv = roc_auc_score(y, cv_prob)
        print(f"  전체 {args.n_splits}-Fold CV AUC (aggregate): {auc_cv:.3f}")
        plot_evaluation(y, cv_pred, cv_prob, label_names,
                        f"XGBoost {args.n_splits}-Fold CV: HD vs COPD",
                        output_dir, "ml_cv")

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42)
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s  = sc.transform(X_te)

        try:
            xgb_clf.fit(X_tr_s, y_tr,
                        eval_set=[(X_te_s, y_te)], verbose=50)
        except xgb.core.XGBoostError as exc:
            if "gpu_hist" in str(exc) or "gpu_hist" in repr(exc):
                print("  ⚠️  XGBoost GPU 미지원으로 CPU hist로 재학습합니다.")
                xgb_params["tree_method"] = "hist"
                xgb_params["device"] = "cpu"
                xgb_clf = xgb.XGBClassifier(**xgb_params)
                xgb_clf.fit(X_tr_s, y_tr,
                            eval_set=[(X_te_s, y_te)], verbose=50)
            else:
                raise

        y_pred = xgb_clf.predict(X_te_s)
        y_prob = xgb_clf.predict_proba(X_te_s)[:, 1]

        print("\n  Classification Report:")
        print(classification_report(y_te, y_pred, target_names=label_names))

        auc_ml = plot_evaluation(y_te, y_pred, y_prob, label_names,
                                 "XGBoost: HD vs COPD (v3)", output_dir, "ml")
        plot_feature_importance(xgb_clf, output_dir)

        results_summary["XGBoost CV AUC"]    = f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}"
        results_summary["XGBoost CV AUC (agg)"] = f"{auc_cv:.3f}"
        results_summary["XGBoost Test AUC"]  = f"{auc_ml:.3f}"
        results_summary["XGBoost Accuracy"]  = f"{accuracy_score(y_te, y_pred):.3f}"

    # ── DL ──────────────────────────────────────
    if args.mode in ("dl", "both"):
        print("\n" + "="*55)
        print("🧠 DL 분석 (Multi-Sensor CNN v3: Attention + Gate)")
        print("="*55)

        print("  시계열 텐서 변환 중...")
        samples = []
        for pat_id, info in sorted(patients.items()):
            result = patient_to_tensors(info, gas_common_cols, args.seq_len)
            if result is None:
                print(f"  ⚠️  {pat_id}: 텐서 변환 실패 → 건너뜀")
                continue
            tensors, col_names_list = result
            label_val = 1 if info["label"] == "COPD" else 0
            samples.append(((tensors, col_names_list), label_val))

        n_copd = sum(1 for _, l in samples if l == 1)
        n_hd   = sum(1 for _, l in samples if l == 0)
        print(f"  총 {len(samples)}명 (HD={n_hd}, COPD={n_copd}) 변환 완료")

        fold_accs, all_probs, all_trues = train_dl(
            samples, device, output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            n_splits=args.n_splits,
        )

        y_pred_dl = [1 if p >= 0.5 else 0 for p in all_probs]
        auc_dl = roc_auc_score(all_trues, all_probs)
        print("\n  [DL] 최종 Classification Report:")
        print(classification_report(all_trues, y_pred_dl, target_names=label_names))

        plot_evaluation(all_trues, y_pred_dl, all_probs, label_names,
                        "1D-CNN Multi-Sensor v3: HD vs COPD", output_dir, "dl")

        results_summary["CNN CV Acc"]      = f"{np.mean(fold_accs):.3f} ± {np.std(fold_accs):.3f}"
        results_summary["CNN Overall AUC"] = f"{auc_dl:.3f}"

    # ── 최종 요약 ───────────────────────────────
    print("\n" + "="*55)
    print("📊 최종 결과 요약")
    print("="*55)
    for k, v in results_summary.items():
        print(f"  {k:<30} {v}")

    pd.DataFrame(list(results_summary.items()),
                 columns=["Metric", "Value"]).to_csv(
        output_dir / "results_summary.csv", index=False)
    print(f"\n✅ 모든 결과 저장: {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   type=str,   default="./data")
    parser.add_argument("--output_dir", type=str,   default="./results")
    parser.add_argument("--mode",       type=str,   default="both",
                        choices=["ml", "dl", "both"])
    parser.add_argument("--seq_len",    type=int,   default=900,
                        help="DL 시퀀스 길이 (M01+M02 concat 기준 ~900)")
    parser.add_argument("--epochs",     type=int,   default=60)
    parser.add_argument("--batch_size", type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--n_splits",   type=int,   default=5)
    args = parser.parse_args()
    main(args)
