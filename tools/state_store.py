"""JSON state persistence under ./state."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def state_dir() -> Path:
    d = project_root() / "state"
    d.mkdir(parents=True, exist_ok=True)
    return d


def experiments_dir() -> Path:
    d = project_root() / "experiments"
    d.mkdir(parents=True, exist_ok=True)
    return d


def report_dir() -> Path:
    d = project_root() / "report"
    d.mkdir(parents=True, exist_ok=True)
    return d


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def append_experiment_log(record: dict[str, Any]) -> None:
    p = state_dir() / "experiment_log.json"
    log = read_json(p)
    if not isinstance(log, list):
        log = []
    log.append(record)
    write_json(p, log)


def load_list(path: Path) -> list:
    data = read_json(path)
    return data if isinstance(data, list) else []


def load_tried_strategies() -> set[str]:
    p = state_dir() / "tried_strategies.json"
    data = read_json(p)
    if isinstance(data, dict) and "tried" in data:
        return set(data["tried"])
    return set()


def save_tried_strategies(tried: set[str]) -> None:
    write_json(state_dir() / "tried_strategies.json", {"tried": sorted(tried)})


def fingerprint_config(cfg: dict[str, Any]) -> str:
    """Stable dedupe key for tried strategies."""
    keys = (
        "model_class",
        "pretrained",
        "optimizer",
        "lr",
        "batch_size",
        "epochs",
        "scheduler",
        "loss_fn",
        "augmentations",
    )
    parts = [str(cfg.get(k, "")) for k in keys]
    ek = cfg.get("extra_kwargs") or {}
    if isinstance(ek, dict):
        parts.append(json.dumps(ek, sort_keys=True))
    return "_".join(parts).replace(" ", "")[:500]
