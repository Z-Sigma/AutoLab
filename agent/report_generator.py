"""Final Markdown report from logs + leaderboard (grounded numbers)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tools.state_store import read_json, report_dir, state_dir, utc_now_iso


def generate_report(
    readme_path: Path,
    dataset_path: Path,
    k: int,
    total_experiments: int,
    duration_seconds: float,
) -> Path:
    task = read_json(state_dir() / "task_profile.json") or {}
    lb = read_json(state_dir() / "leaderboard.json") or {}
    log = read_json(state_dir() / "experiment_log.json")
    if not isinstance(log, list):
        log = []
    ms = read_json(state_dir() / "metric_spec.json") or {}
    sess = read_json(state_dir() / "session_summary.json") or {}

    lines: list[str] = [
        "# Autonomous ML Experiment Report",
        "",
        f"**Task:** {task.get('goal_summary', '')}",
        f"**Dataset:** `{dataset_path}`",
        f"**Target metric (reasoned):** `{lb.get('target_metric', ms.get('primary_metric', ''))}`",
        f"**Metric rationale:** {ms.get('rationale', '')}",
        f"**Total experiments run:** {total_experiments}",
        f"**Wall time (approx):** {duration_seconds:.1f}s",
        f"**Generated:** {utc_now_iso()}",
        "",
    ]

    if sess.get("summary"):
        lines.extend(
            [
                "---",
                "",
                "## Agent session summary (autonomous mode)",
                "",
                str(sess.get("summary", "")),
                "",
                f"**Best experiment (id):** `{sess.get('best_experiment_id', '')}`",
                "",
                f"**Follow-up ideas:** {sess.get('follow_up_ideas', '')}",
                "",
            ]
        )

    lines.extend(
        [
            "---",
            "",
            "## Top-K Strategies",
            "",
        ]
    )

    entries = (lb.get("entries") or [])[:k]
    for e in entries:
        lines.append(f"### Rank {e.get('rank')} — {e.get('strategy_name')}")
        metrics_dict = e.get("metrics") or {}
        lines.append(f"- **Metrics:** " + ", ".join(f"{k}={v:.4g}" for k, v in metrics_dict.items()))
        lines.append(f"- **Experiment id:** `{e.get('experiment_id')}`")
        if e.get("improvement_over_baseline"):
            lines.append(f"- **Improvement:** {e.get('improvement_over_baseline')}")
        lines.append(f"- **Script:** `experiments/{e.get('experiment_id')}/train.py`")
        lines.append("")

    lines.extend(
        [
            "---",
            "",
            "## Improvement trajectory (chronological vs previous best)",
            "",
            f"| Experiment | Strategy | Key metric | Δ (vs Best) |",
            "|------------|----------|------------|-----------|",
        ]
    )
    tm = str(lb.get("target_metric") or ms.get("primary_metric") or "accuracy")
    direction_str = ms.get("direction", "maximize")
    from agent.schemas import MetricDirection
    direction = MetricDirection.maximize if direction_str == "maximize" else MetricDirection.minimize

    best_so_far = None
    for rec in log:
        if rec.get("status") != "success":
            continue
        mid = rec.get("experiment_id", "")
        strat = rec.get("strategy_id", "")[:40]
        m_val_str = (rec.get("metrics") or {}).get(tm, "")
        if m_val_str == "":
            continue
            
        m_val = float(m_val_str)
        delta_str = "—"
        
        if best_so_far is None:
            best_so_far = m_val
        else:
            diff = m_val - best_so_far
            if diff != 0:
                pct = (diff / abs(best_so_far)) * 100 if best_so_far != 0 else float('inf')
                sign = "+" if pct > 0 else ""
                delta_str = f"{sign}{pct:.2f}%"
            else:
                delta_str = "0.00%"
                
            improved = (direction == MetricDirection.maximize and m_val > best_so_far) or \
                       (direction == MetricDirection.minimize and m_val < best_so_far)
            if improved:
                best_so_far = m_val
                
        lines.append(f"| {mid} | {strat} |_**{m_val:.4g}**_| {delta_str} |")

    lines.extend(["", "## Failed runs", ""])
    fails = [x for x in log if x.get("status") != "success"]
    if not fails:
        lines.append("_None._")
    else:
        for rec in fails:
            lines.append(f"- `{rec.get('experiment_id')}`: {str(rec.get('stderr_tail', ''))[:200]}")

    lines.extend([
        "",
        "---",
        "",
        "## How to Reproduce Rank 1",
        ""
    ])
    if entries:
        r1_id = entries[0].get('experiment_id')
        lines.extend([
            "```bash",
            f"python experiments/{r1_id}/train.py \\",
            f"  --data {dataset_path.resolve()}",
            "```"
        ])
    else:
        lines.append("_No successful experiments to reproduce._")
        
    out = report_dir() / "final_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
