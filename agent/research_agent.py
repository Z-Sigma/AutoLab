"""Research agent: web search + PWC-style heuristics → research_notes.json."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from agent.schemas import ResearchNotes, ResearchStrategy
from tools import web_search
from tools.state_store import state_dir, write_json


def search_pwc_style(task_type: str) -> list[ResearchStrategy]:
    """Lightweight stand-in when PWC API is not wired — domain heuristics."""
    t = task_type.lower()
    out: list[ResearchStrategy] = []
    if "image" in t or "classif" in t:
        out.extend(
            [
                ResearchStrategy(
                    id="strat_img_1",
                    name="ResNet-50 ImageNet pretrain + AdamW",
                    source="https://paperswithcode.com/task/image-classification",
                    architecture="resnet50",
                    optimizer="AdamW",
                    lr=1e-4,
                    scheduler="CosineAnnealingLR",
                    augmentation=["RandomResizedCrop", "RandomHorizontalFlip"],
                    applicability_score=0.8,
                    notes="Strong general-purpose vision baseline",
                ),
                ResearchStrategy(
                    id="strat_img_2",
                    name="EfficientNet-B0 + RandAugment",
                    source="https://paperswithcode.com/",
                    architecture="efficientnet_b0",
                    optimizer="AdamW",
                    lr=3e-4,
                    applicability_score=0.85,
                    notes="Parameter-efficient vision SOTA family",
                ),
            ]
        )
    out.extend(
        [
            ResearchStrategy(
                id="strat_tab_1",
                name="RandomForest + standard scaling",
                source="internal_heuristic",
                architecture="sklearn_rf",
                optimizer="n/a",
                applicability_score=0.75,
                notes="Strong tabular default",
            ),
            ResearchStrategy(
                id="strat_tab_2",
                name="Gradient Boosting (sklearn)",
                source="internal_heuristic",
                architecture="sklearn_gb",
                optimizer="n/a",
                applicability_score=0.72,
                notes="Good for heterogeneous tabular features",
            ),
            ResearchStrategy(
                id="strat_tab_3",
                name="LogisticRegression / Ridge baseline",
                source="internal_heuristic",
                architecture="sklearn_logreg",
                optimizer="lbfgs",
                applicability_score=0.65,
                notes="Fast linear baseline",
            ),
        ]
    )
    return out


def _anthropic_synthesize(query: str, snippets: list[str], model: str) -> list[ResearchStrategy] | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key or not snippets:
        return None
    try:
        import anthropic
    except ImportError:
        return None
    client = anthropic.Anthropic(api_key=key)
    sys = """From search snippets, emit JSON only: {"strategies":[{"id":"strat_001","name":"...","source":"url","architecture":"...","optimizer":"...","lr":0.001 or null,"scheduler":null,"augmentation":[],"claimed_metric":{},"applicability_score":0.8,"notes":"..."}]}
Use at most 6 strategies. Prefer diverse, cited ideas."""
    user = f"Query: {query}\n\nSnippets:\n" + "\n---\n".join(snippets[:12])
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
    data = json.loads(m.group())
    strategies = []
    for i, s in enumerate(data.get("strategies") or []):
        strategies.append(
            ResearchStrategy(
                id=str(s.get("id") or f"strat_web_{i}"),
                name=str(s.get("name") or "strategy"),
                source=str(s.get("source") or ""),
                architecture=str(s.get("architecture") or "sklearn_rf"),
                optimizer=str(s.get("optimizer") or "AdamW"),
                lr=s.get("lr"),
                scheduler=s.get("scheduler"),
                augmentation=list(s.get("augmentation") or []),
                claimed_metric=dict(s.get("claimed_metric") or {}),
                applicability_score=float(s.get("applicability_score") or 0.7),
                notes=str(s.get("notes") or ""),
            )
        )
    return strategies


def run_research_agent(task_summary: str, task_type: str, model: str) -> ResearchNotes:
    query = f"state of the art {task_type} machine learning {task_summary}"[:240]
    results = web_search.web_search(query, max_results=8)
    if not results:
        results = web_search.web_search_fallback(query, max_results=8)
    snippets: list[str] = []
    for r in results:
        title = r.get("title") or ""
        content = r.get("content") or ""
        url = r.get("url") or ""
        snippets.append(f"{title}\n{url}\n{content}")

    synth = _anthropic_synthesize(query, snippets, model)
    if synth:
        notes = ResearchNotes(strategies=synth, query_used=query)
    else:
        notes = ResearchNotes(strategies=search_pwc_style(task_type), query_used=query)

    write_json(state_dir() / "research_notes.json", json.loads(notes.model_dump_json()))
    return notes
