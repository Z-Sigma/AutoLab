#!/usr/bin/env python3
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

DATASET_PATH = Path(r"C:/Users/Ajit/Desktop/autoresearcher/examples/iris_dataset")
TARGET_COL = "species"
PRIMARY = "accuracy"
TASK = "multiclass_classification"






def _build_model():
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    if TASK == "regression":
        return RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    return RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)


def main():
    csvs = sorted(DATASET_PATH.glob("*.csv"))
    if not csvs:
        print("METRICS:" + json.dumps({"error": 1.0}))
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

    out = {}
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

    _names = []
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
