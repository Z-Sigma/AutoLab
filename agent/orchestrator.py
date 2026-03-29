"""LangGraph orchestrator: data → metrics → research → experiment loop."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Literal, TypedDict

import yaml
from langgraph.graph import END, StateGraph

from agent.data_agent import infer_task_type_from_profile, run_data_agent
from agent.debug_agent import attempt_debug
from agent.experiment_agent import run_experiment
from agent.leaderboard import best_metric_value, rebuild_leaderboard
from agent.metric_reasoner import derive_metric_spec
from agent.research_agent import run_research_agent
from agent.schemas import ExperimentConfig, MetricDirection, MetricSpec, TaskType
from agent.task_parser import parse_readme
from tools.state_store import (
    experiments_dir,
    fingerprint_config,
    load_tried_strategies,
    read_json,
    save_tried_strategies,
    state_dir,
    write_json,
)


class AgentState(TypedDict, total=False):
    readme_path: str
    dataset_path: str
    k: int
    budget: int
    config: dict[str, Any]
    readme_text: str
    task_profile: dict[str, Any]
    data_profile: dict[str, Any]
    metric_spec: dict[str, Any]
    research_notes: dict[str, Any]
    queue: list[dict[str, Any]]
    tried: list[str]
    exp_index: int
    runs_done: int
    last_best: float | None
    no_improve_count: int
    stop_reason: str
    t0: float
    use_docker: bool


def _load_yaml_cfg(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _map_arch_to_model(arch: str) -> str:
    a = arch.lower().replace("-", "_")
    if "efficientnet" in a or "resnet" in a or "vit" in a:
        return "sklearn_rf"  # tabular runner fallback
    if "gb" in a or "gradient" in a or "xgboost" in a:
        return "sklearn_gb"
    if "log" in a or "linear" in a or "ridge" in a:
        return "sklearn_logreg"
    if "rf" in a or "forest" in a:
        return "sklearn_rf"
    return "sklearn_rf"


def build_default_queue(
    data_profile: dict[str, Any],
    research: dict[str, Any],
    metric_spec: MetricSpec,
) -> list[ExperimentConfig]:
    """Seed experiments from research strategies + tabular defaults."""
    strategies = (research.get("strategies") or [])[:12]
    out: list[ExperimentConfig] = []
    if data_profile.get("dataset_type") != "tabular_csv":
        return out

    seen_arch: set[str] = set()
    for s in strategies:
        arch = str(s.get("architecture") or "sklearn_rf")
        mc = _map_arch_to_model(arch)
        if mc in seen_arch:
            continue
        seen_arch.add(mc)
        sid = str(s.get("id") or f"strat_{mc}")
        extra: dict[str, Any] = {}
        if mc == "sklearn_rf":
            extra["n_estimators"] = 300
        out.append(
            ExperimentConfig(
                experiment_id="",
                strategy_id=sid,
                model_class=mc,
                lr=float(s.get("lr") or 1e-3),
                epochs=1,
                batch_size=32,
                optimizer="n/a",
                loss_fn="default",
                extra_kwargs=extra,
                custom_metric_defs=list(metric_spec.custom_metrics),
            )
        )
    # Ensure at least 3 models
    for mc in ("sklearn_rf", "sklearn_gb", "sklearn_logreg"):
        if not any(x.model_class == mc for x in out):
            out.append(
                ExperimentConfig(
                    strategy_id=f"baseline_{mc}",
                    model_class=mc,
                    epochs=1,
                    extra_kwargs={"n_estimators": 200} if mc == "sklearn_rf" else {},
                    custom_metric_defs=list(metric_spec.custom_metrics),
                )
            )
    return out[:10]


def _anthropic_propose_experiment(
    state: AgentState,
    model: str,
) -> ExperimentConfig | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        import anthropic
    except ImportError:
        return None

    log = read_json(state_dir() / "experiment_log.json")
    if not isinstance(log, list):
        log = []
    sys = """You design the NEXT single ML experiment for a tabular sklearn training runner.

Output ONLY valid JSON for ExperimentConfig:
{
  "strategy_id": "string",
  "model_class": "sklearn_rf" | "sklearn_gb" | "sklearn_logreg" | "sklearn_custom",
  "pretrained": false,
  "optimizer": "AdamW",
  "lr": 0.001,
  "batch_size": 32,
  "epochs": 1,
  "scheduler": null,
  "loss_fn": "default",
  "augmentations": [],
  "extra_kwargs": {"n_estimators": 400},
  "custom_model_code": null,
  "custom_metric_defs": []
}

If sklearn_custom, you MUST include custom_model_code defining:
def build_sklearn_model():
    from sklearn.ensemble import ...
    return ...

Rules: change ONE main idea vs best prior run (e.g. different family or n_estimators). No prose."""
    user = json.dumps(
        {
            "task_profile": state.get("task_profile"),
            "data_profile": state.get("data_profile"),
            "metric_spec": state.get("metric_spec"),
            "experiment_log_tail": log[-8:],
            "tried": state.get("tried"),
        },
        indent=2,
    )[:24000]
    client = anthropic.Anthropic(api_key=key)
    msg = client.messages.create(
        model=model,
        max_tokens=2048,
        system=sys,
        messages=[{"role": "user", "content": user}],
    )
    text = ""
    for b in msg.content:
        if b.type == "text":
            text += b.text
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    d = json.loads(m.group())
    cms = d.get("custom_metric_defs") or []
    from agent.schemas import CustomMetricDef

    cmd = []
    for c in cms:
        if isinstance(c, dict):
            cmd.append(
                CustomMetricDef(
                    name=str(c.get("name", "custom")),
                    description=str(c.get("description", "")),
                    python_body=str(c.get("python_body", "return 0.0")),
                )
            )
    return ExperimentConfig(
        strategy_id=str(d.get("strategy_id") or "llm"),
        model_class=str(d.get("model_class") or "sklearn_rf"),
        pretrained=bool(d.get("pretrained", False)),
        optimizer=str(d.get("optimizer") or "AdamW"),
        lr=float(d.get("lr") or 1e-3),
        batch_size=int(d.get("batch_size") or 32),
        epochs=int(d.get("epochs") or 1),
        scheduler=d.get("scheduler"),
        loss_fn=str(d.get("loss_fn") or "default"),
        augmentations=list(d.get("augmentations") or []),
        extra_kwargs=dict(d.get("extra_kwargs") or {}),
        custom_model_code=d.get("custom_model_code"),
        custom_metric_defs=cmd,
    )


def node_init(state: AgentState) -> dict[str, Any]:
    root = Path(__file__).resolve().parent.parent
    cfg_path = root / "config.yaml"
    cfg = _load_yaml_cfg(cfg_path)
    cfg.update(state.get("config") or {})
    if state.get("k") is not None:
        cfg["top_k_report"] = int(state["k"])
    readme = Path(state["readme_path"]).read_text(encoding="utf-8", errors="replace")
    return {
        "config": cfg,
        "readme_text": readme,
        "t0": time.perf_counter(),
        "exp_index": 0,
        "runs_done": 0,
        "last_best": None,
        "no_improve_count": 0,
        "queue": [],
        "tried": list(load_tried_strategies()),
        "stop_reason": "",
    }


def node_parse_task(state: AgentState) -> dict[str, Any]:
    model = str(state["config"].get("model", "claude-sonnet-4-20250514"))
    tp = parse_readme(Path(state["readme_path"]), model)
    return {"task_profile": json.loads(tp.model_dump_json())}


def node_data(state: AgentState) -> dict[str, Any]:
    dp = run_data_agent(Path(state["dataset_path"]), state.get("readme_text") or "")
    return {"data_profile": dp}


def node_align_task(state: AgentState) -> dict[str, Any]:
    tp = dict(state.get("task_profile") or {})
    dp = state.get("data_profile") or {}
    inferred = infer_task_type_from_profile(dp)
    if tp.get("task_type") in (None, "unknown", TaskType.unknown.value):
        tp["task_type"] = inferred.value
    return {"task_profile": tp}


def node_metrics(state: AgentState) -> dict[str, Any]:
    cfg = state["config"]
    model = str(cfg.get("model", "claude-sonnet-4-20250514"))
    tp = state["task_profile"] or {}
    try:
        tt = TaskType(str(tp.get("task_type") or "tabular"))
    except ValueError:
        tt = TaskType.tabular
    ms = derive_metric_spec(
        state.get("readme_text") or "",
        state.get("data_profile") or {},
        tt,
        model,
    )
    return {"metric_spec": json.loads(ms.model_dump_json())}


def node_research(state: AgentState) -> dict[str, Any]:
    cfg = state["config"]
    model = str(cfg.get("model", "claude-sonnet-4-20250514"))
    summary = (state.get("task_profile") or {}).get("goal_summary") or ""
    tt = str((state.get("task_profile") or {}).get("task_type") or "tabular")
    notes = run_research_agent(summary, tt, model)
    return {"research_notes": json.loads(notes.model_dump_json())}


def node_build_queue(state: AgentState) -> dict[str, Any]:
    dp = state.get("data_profile") or {}
    if dp.get("dataset_type") != "tabular_csv":
        return {
            "queue": [],
            "stop_reason": "unsupported_dataset_type_tabular_only_in_v1",
        }
    ms = MetricSpec.model_validate(state["metric_spec"])
    q = build_default_queue(dp, state.get("research_notes") or {}, ms)
    return {"queue": [json.loads(x.model_dump_json()) for x in q]}


def _metric_spec_from_state(state: AgentState) -> MetricSpec:
    return MetricSpec.model_validate(state["metric_spec"])


def _primary_value(metrics: dict[str, float], spec: MetricSpec) -> float | None:
    k = spec.primary_metric
    if k not in metrics:
        return None
    return float(metrics[k])


def _is_better(new_val: float, old: float | None, direction: MetricDirection) -> bool:
    if old is None:
        return True
    if direction == MetricDirection.maximize:
        return new_val > old
    return new_val < old


def node_run_one(state: AgentState) -> dict[str, Any]:
    cfg = state["config"]
    max_exp = int(cfg.get("max_experiments", 30))
    timeout = float(cfg.get("max_time_per_experiment_seconds", 7200))
    model = str(cfg.get("model", "claude-sonnet-4-20250514"))
    use_docker = bool(cfg.get("use_docker_sandbox", False))
    ms = _metric_spec_from_state(state)
    direction = ms.direction

    if state.get("runs_done", 0) >= max_exp:
        return {"stop_reason": "budget_exhausted"}

    queue = list(state.get("queue") or [])
    exp_cfg_dict: dict[str, Any] | None = None

    while queue:
        cand = ExperimentConfig.model_validate(queue.pop(0))
        fp = fingerprint_config(json.loads(cand.model_dump_json()))
        tried = set(state.get("tried") or [])
        if fp in tried:
            continue
        exp_cfg_dict = json.loads(cand.model_dump_json())
        break

    if exp_cfg_dict is None:
        prop = _anthropic_propose_experiment(state, model)
        if prop is None:
            return {"stop_reason": "queue_empty", "queue": queue}
        fp = fingerprint_config(json.loads(prop.model_dump_json()))
        if fp in set(state.get("tried") or []):
            return {"stop_reason": "no_new_configs", "queue": queue}
        exp_cfg_dict = json.loads(prop.model_dump_json())

    exp_id = f"exp_{state.get('exp_index', 0) + 1:04d}"
    exp_cfg_dict["experiment_id"] = exp_id
    exp_cfg = ExperimentConfig.model_validate(exp_cfg_dict)

    record = run_experiment(
        exp_id,
        Path(state["dataset_path"]),
        state.get("data_profile") or {},
        exp_cfg,
        ms,
        timeout=timeout,
        use_docker=use_docker,
    )

    tried = set(state.get("tried") or [])
    tried.add(fingerprint_config(json.loads(exp_cfg.model_dump_json())))
    save_tried_strategies(tried)

    script_path = Path(record.get("script_path") or "")
    retries = int(cfg.get("max_debug_retries", 3))
    if record.get("status") != "success" and script_path.is_file():
        stderr = record.get("stderr_tail") or ""
        for _ in range(retries):
            attempt_debug(script_path, stderr, record.get("config") or {}, model)
            record = run_experiment(
                exp_id,
                Path(state["dataset_path"]),
                state.get("data_profile") or {},
                exp_cfg,
                ms,
                timeout=timeout,
                use_docker=use_docker,
            )
            if record.get("status") == "success":
                break
            stderr = record.get("stderr_tail") or ""

    rebuild_leaderboard(ms.primary_metric, direction, top_k=max(5, state.get("k", 3)))

    pv = None
    if record.get("status") == "success":
        pv = _primary_value(record.get("metrics") or {}, ms)

    last_best = state.get("last_best")
    no_improve = int(state.get("no_improve_count") or 0)
    if pv is not None:
        if _is_better(pv, last_best, direction):
            last_best = pv
            no_improve = 0
        else:
            no_improve += 1
    else:
        no_improve += 1

    cw = int(cfg.get("convergence_window", 8))
    thr = cfg.get("target_metric_threshold")

    stop_reason = ""
    if thr is not None and last_best is not None:
        if direction == MetricDirection.maximize and last_best >= float(thr):
            stop_reason = "threshold_met"
        elif direction == MetricDirection.minimize and last_best <= float(thr):
            stop_reason = "threshold_met"
    if not stop_reason and no_improve >= cw:
        stop_reason = "converged"

    return {
        "queue": queue,
        "exp_index": state.get("exp_index", 0) + 1,
        "runs_done": state.get("runs_done", 0) + 1,
        "last_best": last_best,
        "no_improve_count": no_improve,
        "tried": sorted(tried),
        "stop_reason": stop_reason,
    }


def route_loop(state: AgentState) -> Literal["continue", "stop"]:
    sr = state.get("stop_reason") or ""
    if sr:
        return "stop"
    cfg = state["config"]
    max_wall = float(cfg.get("max_total_wall_time_hours", 24)) * 3600.0
    if time.perf_counter() - float(state.get("t0") or time.perf_counter()) > max_wall:
        return "stop"
    return "continue"


def build_graph() -> Any:
    g = StateGraph(AgentState)
    g.add_node("init", node_init)
    g.add_node("parse_task", node_parse_task)
    g.add_node("data", node_data)
    g.add_node("align_task", node_align_task)
    g.add_node("metrics", node_metrics)
    g.add_node("research", node_research)
    g.add_node("build_queue", node_build_queue)
    g.add_node("run_one", node_run_one)

    g.set_entry_point("init")
    g.add_edge("init", "parse_task")
    g.add_edge("parse_task", "data")
    g.add_edge("data", "align_task")
    g.add_edge("align_task", "metrics")
    g.add_edge("metrics", "research")
    g.add_edge("research", "build_queue")

    def route_after_queue(state: AgentState) -> Literal["run", "end"]:
        if state.get("stop_reason"):
            return "end"
        return "run"

    g.add_conditional_edges(
        "build_queue",
        route_after_queue,
        {"run": "run_one", "end": END},
    )
    g.add_conditional_edges(
        "run_one",
        route_loop,
        {"continue": "run_one", "stop": END},
    )
    return g.compile()


def run_session(
    readme_path: Path,
    dataset_path: Path,
    k: int,
    extra_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Reset state files optional — caller can clear ./state first."""
    graph = build_graph()
    init: AgentState = {
        "readme_path": str(readme_path),
        "dataset_path": str(dataset_path),
        "k": k,
        "config": extra_config or {},
    }
    return graph.invoke(init)
