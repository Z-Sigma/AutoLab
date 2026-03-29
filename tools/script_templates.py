"""Generate training scripts from ExperimentConfig + profiles (validated structure)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent.schemas import CustomMetricDef, ExperimentConfig, MetricSpec


def _indent_body(body: str, spaces: int = 4) -> str:
    pad = " " * spaces
    lines = body.strip().splitlines()
    return "\n".join(pad + ln if ln.strip() else ln for ln in lines)


def _safe_metric_fn(name: str) -> str:
    return "".join(c if c.isalnum() or c == "_" else "_" for c in name)


def _custom_metric_functions(defs: list[CustomMetricDef]) -> str:
    parts: list[str] = []
    for d in defs:
        sn = _safe_metric_fn(d.name)
        parts.append(
            f"def metric_fn_{sn}(y_true, y_pred):\n"
            f"    import numpy as np\n"
            f"{_indent_body(d.python_body, 4)}\n"
        )
    return "\n\n".join(parts)


def render_tabular_sklearn_script(
    dataset_path: str,
    target_column: str | None,
    config: ExperimentConfig,
    metric_spec: MetricSpec,
) -> str:
    """Full train.py for CSV tabular sklearn baseline + optional custom metrics/model."""
    cm = list(metric_spec.custom_metrics)
    if config.custom_metric_defs:
        cm.extend(config.custom_metric_defs)
    # Dedupe by name
    seen: set[str] = set()
    uniq: list[CustomMetricDef] = []
    for m in cm:
        if m.name not in seen:
            seen.add(m.name)
            uniq.append(m)
    custom_metric_block = _custom_metric_functions(uniq)

    model_factory = _sklearn_model_factory(config)

    custom_model_block = ""
    if config.custom_model_code:
        custom_model_block = (
            "# --- custom model code (agent-provided) ---\n"
            + config.custom_model_code.strip()
            + "\n# --- end custom model ---\n"
        )

    primary = metric_spec.primary_metric
    task = metric_spec.task_type_inferred.value

    script = f'''#!/usr/bin/env python3
"""Auto-generated training script — do not trust metrics except from METRICS line."""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn import metrics as skm

warnings.filterwarnings("ignore", category=UserWarning)

DATASET_PATH = Path(r"{dataset_path.replace(chr(92), '/')}")
TARGET_COL = {json.dumps(target_column)}
PRIMARY = {json.dumps(primary)}
TASK = {json.dumps(task)}

{custom_model_block}

{custom_metric_block}


def _build_model():
{_indent_body(model_factory, 4)}


def main():
    csvs = sorted(DATASET_PATH.glob("*.csv"))
    if not csvs:
        print("METRICS:" + json.dumps({{"error": 1.0}}))
        sys.exit(1)
    df = pd.read_csv(csvs[0])
    if TARGET_COL is None or TARGET_COL not in df.columns:
        # last column as target if not specified
        y = df.iloc[:, -1].values
        X = df.iloc[:, :-1].values
    else:
        y = df[TARGET_COL].values
        X = df.drop(columns=[TARGET_COL]).values

    # numeric only for sklearn baseline
    if X.dtype == object:
        X = pd.DataFrame(X).apply(pd.to_numeric, errors="coerce").fillna(0).values

    stratify = None
    if TASK in ("binary_classification", "multiclass_classification", "tabular"):
        if len(np.unique(y)) < 20 and len(y) > 10:
            stratify = y

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    le = None
    if TASK != "regression" and (
        y_train.dtype == object or (hasattr(y_train, "dtype") and str(y_train.dtype).startswith("str"))
    ):
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_val = le.transform(y_val)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    model = _build_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    out = {{}}
    if TASK == "regression":
        out["mae"] = float(skm.mean_absolute_error(y_val, y_pred))
        out["rmse"] = float(np.sqrt(skm.mean_squared_error(y_val, y_pred)))
        out["r2"] = float(skm.r2_score(y_val, y_pred))
    else:
        out["accuracy"] = float(skm.accuracy_score(y_val, y_pred))
        avg = "binary" if len(np.unique(y_val)) == 2 else "weighted"
        out["f1"] = float(skm.f1_score(y_val, y_pred, average=avg, zero_division=0))
        try:
            if len(np.unique(y_val)) == 2:
                proba = getattr(model, "predict_proba", None)
                if proba is not None:
                    pr = proba(X_val)[:, 1]
                    out["roc_auc"] = float(skm.roc_auc_score(y_val, pr))
        except Exception:
            pass

    _names = {json.dumps([m.name for m in uniq])}
    for cm in _names:
        fn = globals().get("metric_fn_" + "".join(c if c.isalnum() or c == "_" else "_" for c in cm))
        if callable(fn):
            try:
                out[cm] = float(fn(np.asarray(y_val), np.asarray(y_pred)))
            except Exception:
                pass

    print("METRICS:" + json.dumps(out))


if __name__ == "__main__":
    main()
'''
    return script


def _sklearn_model_factory(config: ExperimentConfig) -> str:
    mc = config.model_class.lower()
    extra = config.extra_kwargs or {}
    if mc in ("sklearn_rf", "random_forest"):
        n = int(extra.get("n_estimators", 200))
        return f"""from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
if TASK == "regression":
    return RandomForestRegressor(n_estimators={n}, random_state=42, n_jobs=-1)
return RandomForestClassifier(n_estimators={n}, random_state=42, n_jobs=-1)"""
    if mc in ("sklearn_gb", "gradient_boosting"):
        return """from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
if TASK == "regression":
    return GradientBoostingRegressor(random_state=42)
return GradientBoostingClassifier(random_state=42)"""
    if mc in ("sklearn_logreg", "logistic_regression"):
        return """from sklearn.linear_model import LogisticRegression, Ridge
if TASK == "regression":
    return Ridge(alpha=1.0)
return LogisticRegression(max_iter=2000, random_state=42)"""
    if mc == "sklearn_custom":
        return """return build_sklearn_model()"""
    # default
    return """from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
if TASK == "regression":
    return RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
return RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)"""


def write_experiment_bundle(
    exp_id: str,
    workspace: Path,
    dataset_path: Path,
    target_column: str | None,
    config: ExperimentConfig,
    metric_spec: MetricSpec,
) -> Path:
    """Write train.py + config.json; return path to train.py."""
    exp_dir = workspace / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    script = render_tabular_sklearn_script(
        str(dataset_path.resolve()),
        target_column,
        config,
        metric_spec,
    )
    (exp_dir / "train.py").write_text(script, encoding="utf-8")
    cfg_dump: dict[str, Any] = json.loads(config.model_dump_json())
    (exp_dir / "config.json").write_text(json.dumps(cfg_dump, indent=2), encoding="utf-8")
    return exp_dir / "train.py"
