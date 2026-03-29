"""Parse METRICS:{json} from process stdout (anti-hallucination)."""

from __future__ import annotations

import json
import re
from typing import Any


METRICS_PREFIX = "METRICS:"


def parse_metrics(stdout: str) -> dict[str, float] | None:
    """Extract last METRICS line; values must be numeric for leaderboard."""
    lines = stdout.splitlines()
    for line in reversed(lines):
        line = line.strip()
        if line.startswith(METRICS_PREFIX):
            payload = line[len(METRICS_PREFIX) :].strip()
            try:
                obj: Any = json.loads(payload)
            except json.JSONDecodeError:
                return None
            if not isinstance(obj, dict):
                return None
            out: dict[str, float] = {}
            for k, v in obj.items():
                if isinstance(v, bool):
                    continue
                if isinstance(v, (int, float)):
                    out[str(k)] = float(v)
                elif isinstance(v, str):
                    m = re.match(r"^[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?$", v.strip())
                    if m:
                        out[str(k)] = float(v)
            return out if out else None
    return None
