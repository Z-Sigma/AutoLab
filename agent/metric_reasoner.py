"""Infer primary/secondary/custom metrics from task + data (LLM + heuristics)."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from agent.schemas import CustomMetricDef, MetricDirection, MetricSpec, TaskType
from tools.state_store import state_dir, write_json


def _heuristic_metric_spec(task: TaskType, dp: dict[str, Any]) -> MetricSpec:
    imb = float(dp.get("imbalance_ratio") or 1.0)
    secondary: list[str] = []
    custom: list[CustomMetricDef] = []

    if task == TaskType.regression:
        primary = "mae"
        direction = MetricDirection.minimize
        secondary = ["rmse", "r2"]
    elif task in (TaskType.binary_classification, TaskType.multiclass_classification, TaskType.tabular):
        if imb > 1.5:
            primary = "f1"
            direction = MetricDirection.maximize
            secondary = ["accuracy", "roc_auc"]
            custom.append(
                CustomMetricDef(
                    name="balanced_accuracy",
                    description="Useful under imbalance",
                    python_body=(
                        "from sklearn import metrics as skm\n"
                        "return float(skm.balanced_accuracy_score(y_true, y_pred))"
                    ),
                )
            )
        else:
            primary = "accuracy"
            direction = MetricDirection.maximize
            secondary = ["f1", "roc_auc"]
    else:
        primary = "accuracy"
        direction = MetricDirection.maximize
        secondary = ["f1"]

    rationale = (
        f"Heuristic: task={task.value}, imbalance_ratio={imb:.2f}. "
        f"Primary={primary} ({direction.value})."
    )
    return MetricSpec(
        primary_metric=primary,
        direction=direction,
        secondary_metrics=secondary,
        rationale=rationale,
        custom_metrics=custom,
        task_type_inferred=task,
        uses_probability=task != TaskType.regression and "roc_auc" in secondary,
    )


SYSTEM = """You are a senior ML engineer. Given task context and data profile JSON, choose evaluation metrics.

Rules:
1. Pick ONE primary metric aligned with business/scientific goals and data properties (imbalance, cost asymmetry, rare events).
2. List 2-4 secondary metrics.
3. If generic accuracy/F1 is insufficient, propose up to 2 CUSTOM metrics as Python function BODIES only (variables y_true, y_pred are numpy arrays; for classification y_pred are class labels). Use sklearn/numpy only. No imports inside body except you may use: from sklearn import metrics — but prefer single-line sklearn calls.

Output ONLY valid JSON:
{
  "primary_metric": "string",
  "direction": "maximize" | "minimize",
  "secondary_metrics": ["..."],
  "rationale": "short",
  "custom_metrics": [ {"name": "snake_case", "description": "...", "python_body": "return float(...)" } ],
  "task_type_inferred": "binary_classification" | "multiclass_classification" | "regression" | ...
}
If no custom metrics needed, "custom_metrics": [].
"""


def _anthropic_reason(
    readme: str,
    dp: dict[str, Any],
    task_override: TaskType,
    model: str,
) -> MetricSpec | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        import anthropic
    except ImportError:
        return None

    client = anthropic.Anthropic(api_key=key)
    user = f"README excerpt:\n{readme[:8000]}\n\ndata_profile.json:\n{json.dumps(dp, indent=2)[:12000]}"
    msg = client.messages.create(
        model=model,
        max_tokens=4096,
        system=SYSTEM,
        messages=[{"role": "user", "content": user}],
    )
    text = ""
    for b in msg.content:
        if b.type == "text":
            text += b.text
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    raw = json.loads(m.group())
    customs_raw = raw.get("custom_metrics") or []
    customs: list[CustomMetricDef] = []
    for c in customs_raw:
        if not isinstance(c, dict):
            continue
        name = str(c.get("name", "custom"))
        body = str(c.get("python_body", "return 0.0"))
        customs.append(CustomMetricDef(name=name, description=str(c.get("description", "")), python_body=body))
    tt = task_override
    if raw.get("task_type_inferred"):
        try:
            tt = TaskType(str(raw["task_type_inferred"]))
        except ValueError:
            pass
    return MetricSpec(
        primary_metric=str(raw.get("primary_metric", "accuracy")),
        direction=MetricDirection(str(raw.get("direction", "maximize"))),
        secondary_metrics=list(raw.get("secondary_metrics") or []),
        rationale=str(raw.get("rationale", "")),
        custom_metrics=customs,
        task_type_inferred=tt,
        uses_probability="roc_auc" in str(raw.get("secondary_metrics")),
    )


def derive_metric_spec(
    readme: str,
    data_profile: dict[str, Any],
    task_type: TaskType,
    llm_model: str,
) -> MetricSpec:
    """LLM reasoning when API key present; else strong heuristics."""
    base = _heuristic_metric_spec(task_type, data_profile)
    spec = _anthropic_reason(readme, data_profile, task_type, llm_model)
    if spec is None:
        write_json(state_dir() / "metric_spec.json", json.loads(base.model_dump_json()))
        return base
    # Merge: keep LLM primary/secondary; add heuristic custom if LLM gave none and imbalance high
    if not spec.custom_metrics and base.custom_metrics:
        imb = float(data_profile.get("imbalance_ratio") or 1.0)
        if imb > 1.5 and spec.primary_metric == "accuracy":
            spec.custom_metrics = base.custom_metrics
            spec.rationale += " [added balanced_accuracy heuristic for imbalance]"
    write_json(state_dir() / "metric_spec.json", json.loads(spec.model_dump_json()))
    return spec
