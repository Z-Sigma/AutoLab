#!/usr/bin/env python3
"""Autoresearcher — autonomous tabular ML experiments with metric reasoning."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from agent.autonomous_orchestrator import run_autonomous_session
from agent.orchestrator import run_session
from agent.report_generator import generate_report
from tools.state_store import project_root, state_dir, write_json


def _reset_state(clear: bool) -> None:
    sd = state_dir()
    if not clear:
        return
    for name in (
        "experiment_log.json",
        "leaderboard.json",
        "tried_strategies.json",
        "data_profile.json",
        "research_notes.json",
        "task_profile.json",
        "metric_spec.json",
        "error_log.json",
        "session_summary.json",
    ):
        p = sd / name
        if p.exists():
            p.unlink()
    j = sd / "agent_journal.jsonl"
    if j.exists():
        j.unlink()
    write_json(sd / "experiment_log.json", [])
    write_json(sd / "tried_strategies.json", {"tried": []})


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Autoresearcher: autonomous tool-calling agent (default) or legacy LangGraph pipeline"
    )
    ap.add_argument("--readme", type=Path, required=True, help="Path to README.md (task description)")
    ap.add_argument("--dataset", type=Path, required=True, help="Path to dataset folder")
    ap.add_argument(
        "--mode",
        choices=("autonomous", "legacy"),
        default="autonomous",
        help="autonomous=LLM decides analyses/metrics/models via tools (needs ANTHROPIC_API_KEY); "
        "legacy=fixed sklearn pipeline + LangGraph",
    )
    ap.add_argument("--k", type=int, default=5, help="Top-K strategies in report")
    ap.add_argument("--budget", type=int, default=None, help="Max training experiments")
    ap.add_argument("--max-turns", type=int, default=None, help="Max agent turns (autonomous mode only)")
    ap.add_argument("--clear-state", action="store_true", help="Clear ./state before run")
    ap.add_argument("--metric-threshold", type=float, default=None, help="(legacy) Stop if metric crosses threshold")
    args = ap.parse_args()

    root = project_root()
    if not args.readme.is_file():
        print(f"README not found: {args.readme}", file=sys.stderr)
        return 2
    if not args.dataset.is_dir():
        print(f"Dataset path not a directory: {args.dataset}", file=sys.stderr)
        return 2

    _reset_state(args.clear_state)

    import time

    cfg_path = project_root() / "config.yaml"
    yaml_cfg: dict = {}
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            yaml_cfg = yaml.safe_load(f) or {}

    budget = args.budget if args.budget is not None else int(yaml_cfg.get("max_experiments", 30))
    max_turns = args.max_turns if args.max_turns is not None else int(yaml_cfg.get("max_agent_turns", 48))

    extra: dict = {}
    if args.budget is not None:
        extra["max_experiments"] = args.budget
    if args.metric_threshold is not None:
        extra["target_metric_threshold"] = args.metric_threshold

    t0 = time.perf_counter()
    if args.mode == "autonomous":
        try:
            out = run_autonomous_session(
                args.readme.resolve(),
                args.dataset.resolve(),
                experiment_budget=budget,
                max_agent_turns=max_turns,
                extra_config=extra or None,
            )
        except RuntimeError as e:
            print(str(e), file=sys.stderr)
            return 1
    else:
        out = run_session(args.readme.resolve(), args.dataset.resolve(), args.k, extra_config=extra or None)
    dt = time.perf_counter() - t0

    log_path = state_dir() / "experiment_log.json"
    n = 0
    if log_path.exists():
        with open(log_path, encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                n = len(data)

    rep = generate_report(
        args.readme.resolve(),
        args.dataset.resolve(),
        args.k,
        total_experiments=n,
        duration_seconds=dt,
    )
    print(f"Done. Report: {rep}")
    print(f"State: {state_dir()}")
    print(f"Session output: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
