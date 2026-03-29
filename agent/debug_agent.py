"""Classify errors and apply minimal patches (rule-based + optional LLM)."""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any

from tools.file_tools import read_text, write_text

# Global lock: ensures only ONE debug LLM patch call fires at a time.
# Critical when run_parallel_experiments has 4 concurrent subprocesses
# that all fail simultaneously — without this, 4 API calls fire at once → RPM spike.
_LLM_PATCH_LOCK = threading.Lock()


def classify_error(stderr: str) -> str:
    s = stderr.lower()
    if "out of memory" in s or "oom" in s or "cuda out of memory" in s:
        return "oom"
    if "modulenotfound" in s or "importerror" in s or "no module named" in s:
        return "import_error"
    if "shape" in s and "mismatch" in s:
        return "shape_mismatch"
    if "dtype" in s or "could not convert" in s:
        return "dtype_error"
    if "nan" in s and "loss" in s:
        return "nan_loss"
    if "timeout" in s:
        return "timeout"
    return "unknown"


def rule_patch(script_path: Path, err_type: str, config: dict[str, Any]) -> bool:
    """Return True if patched."""
    text = read_text(script_path)
    if err_type == "oom":
        # Halve batch in generated scripts we don't have — tabular sklearn: reduce n_estimators if present
        m = re.search(r"n_estimators=(\d+)", text)
        if m:
            n = max(10, int(m.group(1)) // 2)
            text = re.sub(r"n_estimators=\d+", f"n_estimators={n}", text, count=1)
            write_text(script_path, text)
            return True
    if err_type == "import_error":
        # prepend common sklearn
        if "from sklearn" not in text:
            return False
    return False


def llm_patch(script_path: Path, traceback: str, config: dict[str, Any], model: str, truncation_bounds: dict[str, int] | None = None) -> bool:
    key = os.environ.get("ANTHROPIC_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    is_openai = bool(openai_key)

    if not key and not openai_key:
        return False

    src_limit = truncation_bounds.get("experiment_script", 8000) if truncation_bounds else 8000
    src = read_text(script_path)[:src_limit]
    sys_prompt = """Return JSON only: {"error_type":"...","line_to_replace":"exact single line from script","replacement":"new line"}
If unsure: {"error_type":"unknown","line_to_replace":"","replacement":""}"""
    tb_limit = truncation_bounds.get("python_stderr", 4000) if truncation_bounds else 4000
    user_prompt = f"Traceback:\n{traceback[-tb_limit:]}\n\nConfig:\n{json.dumps(config)[:4000]}\n\nScript:\n{src}"

    # Read per-call timeout from config (default 30s). Prevents lock starvation.
    llm_timeout = float(config.get("debug_llm_timeout_seconds", 30))

    text = ""
    # Serialize ALL LLM debug calls — prevents RPM spikes during parallel experiments.
    # Per-call timeout ensures a hung call releases the lock instead of blocking threads 2/3/4 forever.
    with _LLM_PATCH_LOCK:
        if is_openai:
            try:
                import openai
                base_url = os.environ.get("OPENAI_BASE_URL")
                client = openai.OpenAI(api_key=openai_key, base_url=base_url, timeout=llm_timeout)
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.2,
                )
                text = resp.choices[0].message.content or ""
            except ImportError:
                return False
            except Exception:
                return False  # timeout or API error — release lock cleanly
        else:
            try:
                import anthropic
                client = anthropic.Anthropic(
                    api_key=key,
                    timeout=anthropic.Timeout(llm_timeout, connect=5.0),
                )
                msg = client.messages.create(
                    model=model,
                    max_tokens=1024,
                    system=sys_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                for b in msg.content:
                    if b.type == "text":
                        text += b.text
            except ImportError:
                return False
            except Exception:
                return False  # timeout or API error — release lock cleanly


    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return False
    p = json.loads(m.group())
    line_old = p.get("line_to_replace") or ""
    line_new = p.get("replacement") or ""
    if not line_old or line_old not in src:
        return False
    new_src = src.replace(line_old, line_new, 1)
    write_text(script_path, new_src)
    return True


def attempt_debug(script_path: Path, stderr: str, config: dict[str, Any], model: str, truncation_bounds: dict[str, int] | None = None) -> str:
    et = classify_error(stderr)
    if rule_patch(script_path, et, config):
        return et
    if llm_patch(script_path, stderr, config, model, truncation_bounds):
        return "llm_patch"
    return et
