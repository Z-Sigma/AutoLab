"""Generate train script, run sandbox, append experiment_log — metrics from stdout only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent.schemas import ExperimentConfig, MetricSpec
from tools import metric_parser
from tools import script_templates
from tools.sandbox import run_python
from tools.state_store import (
    append_experiment_log,
    experiments_dir,
    read_json,
    state_dir,
    utc_now_iso,
    write_json,
)


def run_experiment(
    exp_id: str,
    dataset_path: Path,
    data_profile: dict[str, Any],
    config: ExperimentConfig,
    metric_spec: MetricSpec,
    timeout: float | None,
    use_docker: bool,
) -> dict[str, Any]:
    """Write bundle, execute train.py, return record dict."""
    target_col = data_profile.get("target_column")
    if data_profile.get("dataset_type") != "tabular_csv":
        record = {
            "experiment_id": exp_id,
            "strategy_id": config.strategy_id,
            "config": json.loads(config.model_dump_json()),
            "status": "failed",
            "metrics": {},
            "runtime_seconds": 0.0,
            "timestamp": utc_now_iso(),
            "stdout_tail": "",
            "stderr_tail": "tabular template only — use CSV dataset for Phase-1 runner",
            "script_path": "",
        }
        append_experiment_log(record)
        _append_error_log(record)
        return record

    script_path = script_templates.write_experiment_bundle(
        exp_id,
        experiments_dir(),
        Path(dataset_path),
        target_col,
        config,
        metric_spec,
    )
    res = run_python(script_path, cwd=script_path.parent, timeout=timeout, use_docker=use_docker)
    metrics = metric_parser.parse_metrics(res.stdout) or {}
    ok = res.returncode == 0 and bool(metrics) and "error" not in metrics

    record = {
        "experiment_id": exp_id,
        "strategy_id": config.strategy_id,
        "config": json.loads(config.model_dump_json()),
        "status": "success" if ok else "failed",
        "metrics": metrics if ok else {},
        "runtime_seconds": round(res.duration_seconds, 3),
        "timestamp": utc_now_iso(),
        "stdout_tail": res.stdout[-4000:],
        "stderr_tail": res.stderr[-4000:],
        "script_path": str(script_path),
    }
    append_experiment_log(record)
    if not ok:
        _append_error_log(record)
    return record


def _append_error_log(record: dict[str, Any]) -> None:
    p = state_dir() / "error_log.json"
    cur = read_json(p)
    if not isinstance(cur, list):
        cur = []
    cur.append(
        {
            "experiment_id": record.get("experiment_id"),
            "stderr_tail": record.get("stderr_tail"),
            "stdout_tail": record.get("stdout_tail"),
        }
    )
    write_json(p, cur[-200:])
