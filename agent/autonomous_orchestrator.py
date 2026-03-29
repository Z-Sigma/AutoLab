"""Tool-calling autonomous loop: the model decides analyses, research, metrics, and experiments."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import yaml

from agent.autonomous_tools import AutonomousToolContext, anthropic_tool_definitions, openai_tool_definitions
from tools.state_store import project_root, state_dir, write_json

# ── Bottleneck Fix 1: Foundational tool results that must never be evicted ──
# NOTE: read_task_readme is NOT pinned here — its content is embedded
# directly in messages[0] (user_intro) so it's always in context for free.
_PINNED_TOOLS = {
    "list_dataset",
    "inspect_dataset_schema",
    "set_evaluation_policy",
}

# ── Bottleneck Fix 2: Per-tool history character budget (tighter = fewer tokens) ──
_HISTORY_CHAR_LIMIT: dict[str, int] = {
    "run_python_analysis": 800,
    "web_search": 1000,
    "fetch_url": 600,
    "read_file": 500,
    "compute_dataset_stats": 2000,
    "get_experiment_history": 1500,
    "get_leaderboard": 1200,
    "append_journal": 200,
    "get_metric_spec": 400,
    "read_experiment_script": 2000,
}


def _compress_tool_result(name: str, content: str) -> str:
    """Apply per-tool character budget before storing in history."""
    limit = _HISTORY_CHAR_LIMIT.get(name)
    if limit is None or len(content) <= limit:
        return content
    half = limit // 2
    return content[:half] + f"\n...[history-truncated {len(content)}→{limit}]...\n" + content[-half:]


def _load_config() -> dict[str, Any]:
    p = project_root() / "config.yaml"
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_system_prompt() -> str:
    p = project_root() / "prompts" / "autonomous_system.txt"
    if p.exists():
        return p.read_text(encoding="utf-8")
    return "You are an autonomous ML researcher. Use tools. Metrics only from METRICS: JSON line."


def run_autonomous_session(
    readme_path: Path,
    dataset_path: Path,
    experiment_budget: int,
    max_agent_turns: int,
    model: str | None = None,
    extra_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    key = os.environ.get("ANTHROPIC_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not key and not openai_key:
        raise RuntimeError(
            "Autonomous mode requires ANTHROPIC_API_KEY or OPENAI_API_KEY in the environment."
        )

    is_openai = bool(openai_key)

    cfg = _load_config()
    cfg.update(extra_config or {})
    if is_openai:
        model = model or str(cfg.get("model", "qwen2.5-72b-instruct"))
    else:
        import anthropic
        model = model or str(cfg.get("model", "claude-sonnet-4-20250514"))
    analysis_timeout = float(cfg.get("analysis_timeout_seconds", 300))
    train_timeout = float(cfg.get("max_time_per_experiment_seconds", 7200))
    convergence_window = int(cfg.get("convergence_window", 8))
    max_history_turns = cfg.get("max_history_turns")
    truncation_bounds = cfg.get("truncation_bounds", {})

    # Dynamically resolve rate limits based on api_tier
    api_tier = str(cfg.get("api_tier", "none")).lower()
    _tier_defaults = {
        "free":  {"tpd": 90000,  "tpm": 0},       # Groq on_demand: 100K TPD hard cap
        "paid":  {"tpd": 0,      "tpm": 250000},   # Groq developer: 300K TPM hard cap
        "none":  {"tpd": 0,      "tpm": 0},        # No limiting
    }
    _defaults = _tier_defaults.get(api_tier, _tier_defaults["none"])
    tpd_limit = int(cfg["tokens_per_day_limit"] if cfg.get("tokens_per_day_limit") is not None else _defaults["tpd"])
    tpm_limit = int(cfg["tokens_per_minute_limit"] if cfg.get("tokens_per_minute_limit") is not None else _defaults["tpm"])
    if api_tier != "none":
        print(f"[autoresearcher] API tier: '{api_tier}' | TPD limit: {tpd_limit or 'disabled'} | TPM limit: {tpm_limit or 'disabled'}")

    ctx = AutonomousToolContext(
        dataset_root=dataset_path,
        readme_path=readme_path,
        experiment_budget=experiment_budget,
        analysis_timeout_s=analysis_timeout,
        train_timeout_s=train_timeout,
        model=model,
        truncation_bounds=truncation_bounds,
    )

    if is_openai:
        import openai
        base_url = os.environ.get("OPENAI_BASE_URL")
        client = openai.OpenAI(api_key=openai_key, base_url=base_url)
        tools = openai_tool_definitions()
    else:
        client = anthropic.Anthropic(api_key=key)
        tools = anthropic_tool_definitions()
    system = _load_system_prompt()

    # Embed README directly into user_intro so it lives in messages[0] permanently.
    # This means the agent NEVER needs to re-call read_task_readme to remember its task.
    try:
        readme_content = readme_path.read_text(encoding="utf-8", errors="replace")[:6000]
    except OSError:
        readme_content = "(README not found)"

    user_intro = f"""## Your assignment

**Dataset root:** `{dataset_path.resolve()}`
**Experiment budget:** {experiment_budget} training runs

### Task README (your assignment — always remember this):
{readme_content}

---
1. Use `list_dataset` then `inspect_dataset_schema` / `compute_dataset_stats` to understand the data.
2. Do NOT blindly `run_python_analysis` on full data — query metadata strictly.
3. Formulate hypotheses and log via `append_journal`.
4. Test via `run_training_experiment` or `run_parallel_experiments` (scripts MUST print: `print('METRICS:' + json.dumps({{...}}))`).
5. Reflect via `get_experiment_history` and `get_leaderboard`.
6. Repeat until budget runs out, then `finish_session`.
"""

    if is_openai:
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_intro}]
    else:
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_intro}]

    summary: str | None = None
    best_id: str | None = None
    finished = False
    turn = -1
    best_value: float | None = None
    runs_since_improvement = 0

    # ── Persistent daily token budget ──
    import datetime
    import collections
    # Store OUTSIDE state/ so --clear-state never resets today's counter
    _budget_path = project_root() / "daily_token_usage.json"

    def _load_budget() -> dict:
        try:
            data = json.loads(_budget_path.read_text(encoding="utf-8"))
            if data.get("date") == datetime.date.today().isoformat():
                return data
        except Exception:
            pass
        return {"date": datetime.date.today().isoformat(), "used": 0}

    def _save_budget(b: dict) -> None:
        try:
            _budget_path.write_text(json.dumps(b), encoding="utf-8")
        except Exception:
            pass

    def _check_and_update_budget(tokens: int) -> None:
        if not tpd_limit or not tokens:
            return
        b = _load_budget()
        b["used"] += tokens
        _save_budget(b)
        remaining = tpd_limit - b["used"]
        pct = int(b["used"] / tpd_limit * 100)
        print(f"[autoresearcher] Daily tokens: {b['used']:,} / {tpd_limit:,} used ({pct}%) — {max(remaining,0):,} remaining.")
        if b["used"] >= tpd_limit:
            raise RuntimeError(
                f"[autoresearcher] Daily token budget exhausted ({b['used']:,} / {tpd_limit:,}). "
                "Wait until tomorrow (UTC midnight) or increase tokens_per_day_limit in config.yaml."
            )

    _budget = _load_budget()
    if tpd_limit and _budget["used"] >= tpd_limit:
        raise RuntimeError(
            f"[autoresearcher] Daily token budget already exhausted ({_budget['used']:,} / {tpd_limit:,}). "
            "Wait until tomorrow (UTC midnight) or increase tokens_per_day_limit in config.yaml."
        )
    if tpd_limit:
        print(f"[autoresearcher] Daily budget: {_budget['used']:,} / {tpd_limit:,} tokens used so far today.")

    def _clean_for_api(msgs: list[dict]) -> list[dict]:
        """Strip internal fields and normalize messages for API compatibility.

        Gemini's OpenAI-compat layer is stricter than standard OpenAI:
        - Tool results MUST have non-empty 'name' matching the function call
        - Assistant messages must only contain role/content/tool_calls
        - tool_calls entries must only contain id/type/function
        """
        cleaned = []
        for m in msgs:
            role = m.get("role", "")

            if "_history_content" in m:
                m = {k: v for k, v in m.items() if k != "_history_content"}

            # Normalize assistant messages: strip any extra fields Gemini doesn't expect back
            if role == "assistant" and "tool_calls" in m:
                norm_calls = []
                for tc in (m.get("tool_calls") or []):
                    fn = tc.get("function", {})
                    norm_calls.append({
                        "id": tc.get("id", ""),
                        "type": tc.get("type", "function"),
                        "function": {
                            "name": fn.get("name", ""),
                            "arguments": fn.get("arguments", "{}"),
                        },
                    })
                m = {"role": "assistant", "content": m.get("content") or None, "tool_calls": norm_calls}
                if m["content"] is None:
                    del m["content"]  # exclude_none style

            # Ensure tool result name is never empty (Gemini rejects empty function_response.name)
            if role == "tool":
                if not m.get("name"):
                    # Fallback: try to infer name from tool_call_id pattern, else use placeholder
                    m = dict(m)
                    m["name"] = "tool_result"

            cleaned.append(m)
        return cleaned

    for turn in range(max_agent_turns):
        # ── LLM API call with exponential-backoff retry on rate limits ──
        max_api_retries = 6
        backoff = 15  # initial wait in seconds (15→30→60→120→240 — reaches 60s by retry 3 for RPM limits)
        for api_attempt in range(max_api_retries):
            try:
                if is_openai:
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "system", "content": system}] + _clean_for_api(messages),
                        tools=tools,
                        temperature=0.7,
                    )
                    msg = resp.choices[0].message
                    messages.append(msg.model_dump(exclude_none=True))
                    tool_blocks = msg.tool_calls or []
                    if not tool_blocks:
                        if msg.content:
                            print(f"[autoresearcher] No tool calls returned. LLM says: {msg.content}")
                        finished = True
                    break  # success
                else:
                    resp = client.messages.create(
                        model=model,
                        max_tokens=16384,
                        system=system,
                        messages=messages,
                        tools=tools,
                    )
                    messages.append({"role": "assistant", "content": resp.content})
                    tool_blocks = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
                    if not tool_blocks:
                        if getattr(resp, "stop_reason", None) == "end_turn":
                            finished = True
                    break  # success
            except Exception as e:
                err_str = str(e).lower()
                err_code = getattr(e, 'status_code', None) or getattr(getattr(e, 'response', None), 'status_code', None)
                is_daily_limit = any(x in err_str for x in ["tokens per day", "tpd", "daily"])
                is_tool_schema_error = err_code == 400 and "tool" in err_str and ("schema" in err_str or "validation" in err_str or "parameter" in err_str)
                is_rate_limit = any(x in err_str for x in ["429", "rate limit", "rate_limit", "too many requests", "quota"])

                if is_daily_limit:
                    raise RuntimeError(
                        f"[autoresearcher] Groq daily token quota exhausted (TPD). "
                        f"Wait until tomorrow UTC midnight or upgrade your plan.\nOriginal error: {e}"
                    ) from None

                if is_tool_schema_error:
                    # LLM passed wrong type for a tool parameter (e.g. "1" instead of 1).
                    # Inject a correction message and continue the loop instead of crashing.
                    print(f"[autoresearcher] Tool schema validation error (400). Injecting correction and retrying turn...")
                    correction = (
                        "Your last tool call was rejected by the server because a parameter had the wrong type "
                        "(e.g. a string where an integer was expected). "
                        "Please retry with correct parameter types. "
                        f"Original error: {e}"
                    )
                    messages.append({"role": "user", "content": correction})
                    tool_blocks = []
                    break  # break inner retry loop, outer turn loop will continue

                if is_rate_limit and api_attempt < max_api_retries - 1:
                    wait = backoff * (2 ** api_attempt)
                    print(f"[autoresearcher] Transient rate limit hit. Waiting {wait:.0f}s before retry {api_attempt + 1}/{max_api_retries - 1}...")
                    time.sleep(wait)
                    continue
                raise  # re-raise if not handled above

        # Track actual tokens from the API response against daily budget
        try:
            if is_openai and hasattr(resp, "usage") and resp.usage:
                _check_and_update_budget(resp.usage.total_tokens)
            elif not is_openai and hasattr(resp, "usage") and resp.usage:
                _check_and_update_budget(getattr(resp.usage, "input_tokens", 0) + getattr(resp.usage, "output_tokens", 0))
        except RuntimeError:
            raise  # re-raise budget exhaustion cleanly
        except Exception:
            pass  # never let budget tracking crash the agent

        if finished and not tool_blocks:
            break

        tool_results: list[dict[str, Any]] = []
        for block in tool_blocks:
            if is_openai:
                tid = block.id
                name = block.function.name
                inp = json.loads(block.function.arguments) if block.function.arguments else {}
            else:
                tid = block.id
                name = block.name
                inp = getattr(block, "input", {}) or {}

            if name == "finish_session":
                summary = str(inp.get("summary") or "")
                best_id = str(inp.get("best_experiment_id") or "")
                fu = str(inp.get("follow_up_ideas") or "")
                write_json(
                    state_dir() / "session_summary.json",
                    {
                        "summary": summary,
                        "best_experiment_id": best_id,
                        "follow_up_ideas": fu,
                        "turn": turn,
                    },
                )
                finished = True
                if is_openai:
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": tid,
                        "name": name,
                        "content": "Session finished."
                    })
                else:
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tid,
                            "content": "Session finished. No further tool calls needed.",
                        }
                    )
            else:
                out = ctx.execute(name, inp if isinstance(inp, dict) else {})
                out_str = out[:190000] if isinstance(out, str) else str(out)

                # Fix 2: apply per-tool history compression before storing
                history_str = _compress_tool_result(name, out_str)

                if name == "run_training_experiment":
                    from agent.schemas import MetricDirection
                    from agent.leaderboard import best_metric_value
                    from tools.state_store import read_json
                    ms = read_json(state_dir() / "metric_spec.json") or {}
                    pm = ms.get("primary_metric")
                    d_str = ms.get("direction", "maximize")
                    if pm:
                        direction = MetricDirection.maximize if d_str == "maximize" else MetricDirection.minimize
                        val = best_metric_value(pm, direction)
                        if val is not None:
                            if best_value is None:
                                best_value = val
                                runs_since_improvement = 0
                            else:
                                improved = (direction == MetricDirection.maximize and val > best_value) or \
                                           (direction == MetricDirection.minimize and val < best_value)
                                if improved:
                                    best_value = val
                                    runs_since_improvement = 0
                                else:
                                    runs_since_improvement += 1
                            if runs_since_improvement >= convergence_window:
                                out_str += f"\n\n[SYSTEM ALERT]: Convergence detected. '{pm}' hasn't improved in {convergence_window} runs. You MUST call finish_session next."

                if is_openai:
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": tid,
                        "name": name,
                        # LLM sees full output NOW; history stores compressed version
                        "content": out_str,
                        "_history_content": history_str,  # custom field, stripped before API call
                    })
                else:
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tid,
                            "content": out_str,
                            "_history_content": history_str,
                        }
                    )

        # Commit tool results into history (swap full output for compressed version for non-current turns)
        def _to_history(msg: dict) -> dict:
            """Replace content with compressed _history_content if present."""
            if "_history_content" in msg:
                m = {k: v for k, v in msg.items() if k != "_history_content"}
                m["content"] = msg["_history_content"]
                return m
            return msg

        if is_openai:
            messages.extend(tool_results)  # current turn: full content
        else:
            messages.append({"role": "user", "content": tool_results})

        # Sliding window — keeps last N assistant turns.
        # NOTE: We do NOT reinsert pinned tool results into the window.
        # Gemini (and strict OpenAI compat layers) require that every function_response
        # immediately follows the model's function_call. Orphan tool results inserted
        # before the retained window cause "function_response.name cannot be empty" errors.
        # The agent uses get_experiment_history / get_leaderboard to recall evicted data.
        if max_history_turns is not None and isinstance(max_history_turns, int) and max_history_turns > 0:
            boundaries = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
            if len(boundaries) > max_history_turns:
                cut_idx = boundaries[-max_history_turns]
                # Compress older messages inside the retained window
                retained = [_to_history(m) if idx >= cut_idx else m for idx, m in enumerate(messages)]
                messages = [messages[0]] + retained[cut_idx:]

        if finished:
            break

    return {
        "mode": "autonomous",
        "turns_used": turn + 1 if turn >= 0 else 0,
        "experiments_run": ctx.experiments_run,
        "summary": summary,
        "best_experiment_id": best_id,
        "finished": finished,
    }
