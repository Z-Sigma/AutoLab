"""Rank experiments by target metric from real logs only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent.schemas import Leaderboard, LeaderboardEntry, MetricDirection
from tools.state_store import read_json, state_dir, write_json


def rebuild_leaderboard(
    target_metric: str,
    direction: MetricDirection,
    top_k: int,
) -> Leaderboard:
    log_path = state_dir() / "experiment_log.json"
    log = read_json(log_path)
    if not isinstance(log, list):
        log = []

    rows: list[tuple[float, dict[str, Any]]] = []
    for e in log:
        if e.get("status") != "success":
            continue
        m = e.get("metrics") or {}
        if target_metric not in m:
            continue
        v = float(m[target_metric])
        # maximize -> higher better
        key = v if direction == MetricDirection.maximize else -v
        rows.append((key, e))

    rows.sort(key=lambda x: x[0], reverse=True)

    entries: list[LeaderboardEntry] = []
    baseline_val: float | None = None
    for i, (_, e) in enumerate(rows[:top_k]):
        m = e.get("metrics") or {}
        val = float(m.get(target_metric, 0.0))
        if i == 0:
            baseline_val = val
        imp = ""
        if baseline_val is not None and i > 0:
            first = float((rows[0][1].get("metrics") or {}).get(target_metric, val))
            if first != 0:
                pct = (val - first) / abs(first) * 100
                imp = f"{pct:+.2f}% vs rank1"
        cfg = e.get("config") or {}
        name = f"{cfg.get('model_class', 'model')}_{e.get('experiment_id', '')}"
        entries.append(
            LeaderboardEntry(
                rank=i + 1,
                experiment_id=str(e.get("experiment_id", "")),
                strategy_name=name,
                metrics=dict(m),
                improvement_over_baseline=imp,
            )
        )

    lb = Leaderboard(
        target_metric=target_metric,
        direction=direction,
        top_k=top_k,
        entries=entries,
    )
    write_json(state_dir() / "leaderboard.json", json.loads(lb.model_dump_json()))
    return lb


def best_metric_value(target_metric: str, direction: MetricDirection) -> float | None:
    lb = rebuild_leaderboard(target_metric, direction, top_k=1)
    if not lb.entries:
        return None
    v = lb.entries[0].metrics.get(target_metric)
    return float(v) if v is not None else None
