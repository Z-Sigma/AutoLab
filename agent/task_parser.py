"""Parse README.md into task_profile.json."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from agent.schemas import TaskProfile, TaskType
from tools.state_store import state_dir, write_json


def _heuristic_parse(text: str) -> TaskProfile:
    low = text.lower()
    metric = None
    for pat in (r"metric[:\s]+([a-zA-Z0-9_\-]+)", r"evaluate[d]?\s+(?:with|on)\s+([a-zA-Z0-9_\-]+)"):
        m = re.search(pat, low)
        if m:
            metric = m.group(1)
            break

    tt = TaskType.unknown
    if "regress" in low or "mae" in low or "rmse" in low:
        tt = TaskType.regression
    elif "binary" in low or "two class" in low:
        tt = TaskType.binary_classification
    elif "classif" in low or "label" in low:
        tt = TaskType.multiclass_classification
    elif "image" in low:
        tt = TaskType.image_classification

    return TaskProfile(
        raw_readme_excerpt=text[:8000],
        task_type=tt,
        goal_summary=text[:500].replace("\n", " "),
        readme_stated_metric=metric,
        constraints=[],
    )


def _llm_parse(text: str, model: str) -> TaskProfile | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        import anthropic
    except ImportError:
        return None
    client = anthropic.Anthropic(api_key=key)
    sys = """Extract JSON only:
{"task_type":"binary_classification"|"multiclass_classification"|"regression"|"image_classification"|"tabular"|"unknown",
 "goal_summary":"one sentence",
 "readme_stated_metric": string or null,
 "constraints": [string],
 "domain_notes": string}"""
    msg = client.messages.create(
        model=model,
        max_tokens=1024,
        system=sys,
        messages=[{"role": "user", "content": text[:12000]}],
    )
    out = ""
    for b in msg.content:
        if b.type == "text":
            out += b.text
    m = re.search(r"\{[\s\S]*\}", out)
    if not m:
        return None
    d = json.loads(m.group())
    try:
        tt = TaskType(str(d.get("task_type") or "unknown"))
    except ValueError:
        tt = TaskType.unknown
    return TaskProfile(
        raw_readme_excerpt=text[:8000],
        task_type=tt,
        goal_summary=str(d.get("goal_summary") or ""),
        readme_stated_metric=d.get("readme_stated_metric"),
        constraints=list(d.get("constraints") or []),
        domain_notes=str(d.get("domain_notes") or ""),
    )


def parse_readme(readme_path: Path, model: str) -> TaskProfile:
    text = Path(readme_path).read_text(encoding="utf-8", errors="replace")
    prof = _llm_parse(text, model) or _heuristic_parse(text)
    write_json(state_dir() / "task_profile.json", json.loads(prof.model_dump_json()))
    return prof
