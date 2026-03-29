"""EDA and data_profile.json generation (no LLM — deterministic)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from agent.schemas import DatasetType, TaskType
from tools import file_tools
from tools.state_store import project_root, state_dir, write_json


def profile_dataset(dataset_path: Path, readme_hint: str = "") -> dict[str, Any]:
    """Produce data_profile.json structure."""
    dataset_path = Path(dataset_path).resolve()
    hint = (readme_hint or "").lower()

    if not dataset_path.exists():
        return _empty_profile("path_missing")

    files = file_tools.list_dir(dataset_path, "*.csv")
    if files:
        return _profile_csv(dataset_path, files[0], hint)

    # image folder: subdirs = classes
    subdirs = [p for p in dataset_path.iterdir() if p.is_dir()]
    exts = (".jpg", ".jpeg", ".png", ".webp")
    imgs = [p for p in dataset_path.rglob("*") if p.suffix.lower() in exts]
    if subdirs and any(len(list(sd.glob("*"))) for sd in subdirs):
        return _profile_image_folder(dataset_path, subdirs)

    if imgs:
        return {
            "dataset_type": DatasetType.image_folder.value,
            "n_samples": len(imgs),
            "n_classes": 0,
            "notes": "Flat image folder without class subdirs — needs labeling column or README.",
            "issues_found": ["ambiguous layout"],
            "recommended_split": {"train": 0.7, "val": 0.15, "test": 0.15},
        }

    return _empty_profile("unknown_layout")


def _empty_profile(reason: str) -> dict[str, Any]:
    return {
        "dataset_type": DatasetType.unknown.value,
        "n_samples": 0,
        "n_classes": 0,
        "issues_found": [reason],
        "recommended_split": {"train": 0.7, "val": 0.15, "test": 0.15},
    }


def _profile_csv(root: Path, rel_csv: str, hint: str) -> dict[str, Any]:
    path = root / rel_csv
    df = pd.read_csv(path, nrows=50_000)
    n = len(df)
    cols = list(df.columns)
    target_guess = None
    for name in ("target", "label", "y", "class", "Class"):
        if name in df.columns:
            target_guess = name
            break
    if target_guess is None:
        target_guess = cols[-1]

    y = df[target_guess]
    numeric_cols = [c for c in cols if c != target_guess and pd.api.types.is_numeric_dtype(df[c])]
    task = TaskType.regression.value
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 20:
        task = TaskType.regression.value
    else:
        task = TaskType.multiclass_classification.value
        if y.nunique() == 2:
            task = TaskType.binary_classification.value

    if "regress" in hint or "mae" in hint or "rmse" in hint:
        task = TaskType.regression.value
    if "classif" in hint or "accuracy" in hint:
        if task == TaskType.regression.value and y.nunique() <= 20:
            task = TaskType.multiclass_classification.value

    class_distribution: dict[str, int] = {}
    imbalance_ratio = 1.0
    if task != TaskType.regression.value:
        vc = y.astype(str).value_counts()
        class_distribution = {str(k): int(v) for k, v in vc.items()}
        if len(vc) > 0:
            imbalance_ratio = float(vc.max() / max(vc.min(), 1))

    issues: list[str] = []
    miss = df.isna().mean()
    hi = miss[miss > 0.05]
    if len(hi):
        issues.append(f"columns with >5% missing: {list(hi.index)[:8]}")
    if imbalance_ratio > 2.0:
        issues.append("class imbalance — consider weighted loss or resampling")

    return {
        "dataset_type": DatasetType.tabular_csv.value,
        "csv_path": rel_csv,
        "n_samples": n,
        "n_features": len(cols) - 1,
        "n_classes": int(y.nunique()) if task != TaskType.regression.value else 0,
        "target_column": target_guess,
        "feature_columns": [c for c in cols if c != target_guess],
        "task_hint": task,
        "class_distribution": class_distribution,
        "imbalance_ratio": round(imbalance_ratio, 4),
        "recommended_split": {"train": 0.7, "val": 0.15, "test": 0.15},
        "recommended_normalization": {"type": "standard_scaler_numeric"},
        "issues_found": issues,
        "preprocessing_pipeline": ["StandardScaler on numeric features"],
    }


def _profile_image_folder(root: Path, subdirs: list[Path]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for sd in subdirs:
        n = len([p for p in sd.iterdir() if p.is_file()])
        counts[sd.name] = n
    total = sum(counts.values())
    return {
        "dataset_type": DatasetType.image_folder.value,
        "n_samples": total,
        "n_classes": len(counts),
        "class_distribution": counts,
        "imbalance_ratio": round(max(counts.values()) / max(min(counts.values()), 1), 4),
        "recommended_split": {"train": 0.7, "val": 0.15, "test": 0.15},
        "image_size_mode": [224, 224],
        "issues_found": [],
        "preprocessing_pipeline": ["Resize(224)", "ToTensor()", "ImageNet normalize"],
    }


def run_data_agent(dataset_path: Path, readme_text: str = "") -> dict[str, Any]:
    profile = profile_dataset(dataset_path, readme_text)
    write_json(state_dir() / "data_profile.json", profile)
    return profile


def infer_task_type_from_profile(dp: dict[str, Any]) -> TaskType:
    th = dp.get("task_hint")
    if th:
        try:
            return TaskType(th)
        except ValueError:
            pass
    dt = dp.get("dataset_type")
    if dt == DatasetType.image_folder.value:
        return TaskType.image_classification
    return TaskType.tabular
