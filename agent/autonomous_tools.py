"""Tool implementations for the autonomous agent (web, EDA code, training runs)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import uuid
from pathlib import Path
from typing import Any

from tools import metric_parser, web_search
from tools.state_store import append_experiment_log, experiments_dir, read_json, state_dir, utc_now_iso, write_json


def _truncate(s: str, n: int = 12000) -> str:
    s = s or ""
    return s if len(s) <= n else s[: n // 2] + "\n...[truncated]...\n" + s[-(n // 2) :]


class AutonomousToolContext:
    """Shared paths, budgets, and counters for one session."""

    def __init__(
        self,
        dataset_root: Path,
        readme_path: Path,
        experiment_budget: int,
        analysis_timeout_s: float = 300.0,
        train_timeout_s: float = 7200.0,
        model: str = "claude-sonnet-4-20250514",
        truncation_bounds: dict[str, int] | None = None,
    ) -> None:
        self.dataset_root = dataset_root.resolve()
        self.readme_path = readme_path.resolve()
        self.experiment_budget = experiment_budget
        self.analysis_timeout_s = analysis_timeout_s
        self.train_timeout_s = train_timeout_s
        self.experiments_run = 0
        self.journal_path = state_dir() / "agent_journal.jsonl"
        self.model = model
        self.truncation_bounds = truncation_bounds or {}


    def _safe_under_dataset(self, rel: str) -> Path | None:
        rel = rel.replace("\\", "/").lstrip("/")
        if ".." in rel:
            return None
        p = (self.dataset_root / rel).resolve()
        try:
            p.relative_to(self.dataset_root)
        except ValueError:
            return None
        return p

    def web_search(self, query: str, max_results: int = 8) -> str:
        r = web_search.web_search(query, max_results=max_results)
        if not r:
            r = web_search.web_search_fallback(query, max_results=max_results)
        if not r:
            return "No results (set TAVILY_API_KEY or: pip install duckduckgo-search)."
        lines = []
        for i, x in enumerate(r, 1):
            title = x.get("title") or ""
            url = x.get("url") or x.get("href") or ""
            body = (x.get("content") or x.get("body") or "")[:600]
            lines.append(f"{i}. {title}\n   {url}\n   {body}")
        return "\n\n".join(lines)

    def fetch_url(self, url: str) -> str:
        return _truncate(web_search.fetch_url(url))

    def list_dataset(self, subpath: str = "") -> str:
        base = self._safe_under_dataset(subpath or ".")
        if base is None or not base.exists():
            return "Invalid path or does not exist."
        if base.is_file():
            return str(base.relative_to(self.dataset_root))
        files = sorted(base.rglob("*"))
        out = []
        for p in files[:400]:
            if p.is_file():
                try:
                    rel = p.relative_to(self.dataset_root)
                    out.append(str(rel).replace("\\", "/"))
                except ValueError:
                    pass
        return "\n".join(out) if out else "(empty)"

    def read_file(self, relative_path: str, max_chars: int = 8000) -> str:
        p = self._safe_under_dataset(relative_path)
        if p is None or not p.is_file():
            return "File not found or path not allowed."
        actual_max = min(max_chars, self.truncation_bounds.get("read_file", 8000)) if max_chars else self.truncation_bounds.get("read_file", 8000)
        try:
            return _truncate(p.read_text(encoding="utf-8", errors="replace"), actual_max)
        except OSError as e:
            return f"Read error: {e}"

    def read_readme(self) -> str:
        try:
            limit = self.truncation_bounds.get("read_readme", 16000)
            return _truncate(self.readme_path.read_text(encoding="utf-8", errors="replace"), limit)
        except OSError as e:
            return f"Could not read README: {e}"

    def inspect_dataset_schema(self, relative_path: str) -> str:
        p = self._safe_under_dataset(relative_path)
        if p is None or not p.is_file() or p.suffix.lower() != ".csv":
            return "Must target a valid .csv file in the dataset directory."
        try:
            import pandas as pd
            import json
            df = pd.read_csv(p)
            schema = {"shape": list(df.shape), "columns": {}}
            missing = df.isnull().sum()
            dtypes = df.dtypes
            for col in df.columns:
                schema["columns"][col] = {"dtype": str(dtypes[col]), "missing": int(missing[col])}
            return json.dumps(schema)
        except Exception as e:
            return f"Error reading schema: {e}"

    def compute_dataset_stats(self, relative_path: str, columns: list[str]) -> str:
        p = self._safe_under_dataset(relative_path)
        if p is None or not p.is_file() or p.suffix.lower() != ".csv":
            return "Must target a valid .csv file."
        try:
            import pandas as pd
            import json
            df = pd.read_csv(p, usecols=lambda c: c in columns)
            stats = {}
            for col in df.columns:
                s = df[col]
                if pd.api.types.is_numeric_dtype(s):
                    stats[col] = {
                        "min": float(s.min()) if not pd.isna(s.min()) else None,
                        "max": float(s.max()) if not pd.isna(s.max()) else None,
                        "mean": float(s.mean()) if not pd.isna(s.mean()) else None,
                        "std": float(s.std()) if not pd.isna(s.std()) else None,
                        "nunique": int(s.nunique())
                    }
                else:
                    stats[col] = {"nunique": int(s.nunique()), "mode": str(s.mode().iloc[0]) if len(s.mode()) > 0 else None}
            return json.dumps(stats)
        except Exception as e:
            return f"Error computing stats: {e}"

    def run_python_analysis(self, code: str, purpose: str = "") -> str:
        """Run arbitrary analysis code in sandbox; DATASET_ROOT and README_PATH set."""
        sd = state_dir() / "analysis_runs"
        sd.mkdir(parents=True, exist_ok=True)
        uid = uuid.uuid4().hex[:12]
        path = sd / f"analysis_{uid}.py"
        wrapper = f'''# purpose: {purpose[:200]}
import os, sys, json
DATASET_ROOT = os.environ["DATASET_ROOT"]
README_PATH = os.environ.get("README_PATH", "")
{code}
'''
        path.write_text(textwrap.dedent(wrapper), encoding="utf-8")
        env = {
            **os.environ,
            "DATASET_ROOT": str(self.dataset_root),
            "README_PATH": str(self.readme_path),
            "PYTHONUTF8": "1",
        }
        try:
            proc = subprocess.run(
                [sys.executable, str(path)],
                cwd=str(self.dataset_root),
                env=env,
                capture_output=True,
                text=True,
                timeout=self.analysis_timeout_s,
            )
            out_stdout = _truncate(proc.stdout, self.truncation_bounds.get("python_stdout", 10000))
            out_stderr = _truncate(proc.stderr, self.truncation_bounds.get("python_stderr", 6000))
            out = f"exit={proc.returncode}\n--- stdout ---\n{out_stdout}\n--- stderr ---\n{out_stderr}"
            return out
        except subprocess.TimeoutExpired:
            return "TIMEOUT: shorten analysis or increase timeout in config."
        except Exception as e:
            return f"Error: {e}"

    def run_training_experiment(
        self, experiment_name: str, python_script: str, experiment_config_json: dict[str, Any] | None = None
    ) -> str:
        if self.experiments_run >= self.experiment_budget:
            return (
                f"EXPERIMENT BUDGET EXHAUSTED ({self.experiment_budget}). "
                "Do not run more training; call finish_session with your conclusions."
            )
        if "METRICS:" not in python_script:
            return (
                "Script must print exactly one line: print('METRICS:' + json.dumps({...})) "
                "with numeric metric values (anti-hallucination). Revise and retry."
            )
        
        config_data = {"mode": "autonomous", "name": experiment_name}
        if experiment_config_json:
            try:
                from agent.schemas import ExperimentConfig
                validated = ExperimentConfig(**experiment_config_json)
                config_data.update(validated.model_dump())
            except Exception as e:
                return f"Validation error in experiment_config_json: {e}\nPlease fix JSON and retry."

        # Increment based on existing folders in the experiments directory
        existing_ids = []
        if experiments_dir().exists():
            for p in experiments_dir().iterdir():
                if p.is_dir() and p.name.startswith("exp_auto_"):
                    try:
                        existing_ids.append(int(p.name.split("_")[-1]))
                    except ValueError:
                        continue
        next_num = max(existing_ids, default=0) + 1
        exp_id = f"exp_auto_{next_num:04d}"
        ed = experiments_dir() / exp_id
        ed.mkdir(parents=True, exist_ok=True)
        script_path = ed / "train.py"
        script_path.write_text(python_script, encoding="utf-8")

        env = {**os.environ, "DATASET_ROOT": str(self.dataset_root), "README_PATH": str(self.readme_path)}
        metrics = {}
        ok = False
        import subprocess
        from agent.debug_agent import attempt_debug

        for attempt in range(4):  # 1 initial + up to 3 retries
            try:
                proc = subprocess.run(
                    [sys.executable, str(script_path)],
                    cwd=str(self.dataset_root),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=self.train_timeout_s,
                )
                metrics = metric_parser.parse_metrics(proc.stdout or "") or {}
                ok = proc.returncode == 0 and bool(metrics) and all(k != "error" for k in metrics.keys())
                
                if ok:
                    break
                
                # Failed, attempt debug
                if attempt < 3:
                    action = attempt_debug(script_path, proc.stderr or proc.stdout or "", config_data, self.model, self.truncation_bounds)
                    if action in ("unknown", "timeout"):
                        break
                else:
                    break  # Out of retries
            except subprocess.TimeoutExpired:
                record = {
                    "experiment_id": exp_id,
                    "strategy_id": experiment_name[:120],
                    "config": config_data,
                    "status": "failed",
                    "metrics": {"error": "timeout"},
                    "runtime_seconds": 0.0,
                    "timestamp": utc_now_iso(),
                    "stdout_tail": "",
                    "stderr_tail": "Training timeout out",
                    "script_path": str(script_path),
                }
                append_experiment_log(record)
                self.experiments_run += 1
                return json.dumps({"error": "training_timeout", "experiment_id": exp_id})

        record = {
            "experiment_id": exp_id,
            "strategy_id": experiment_name[:120],
            "config": config_data,
            "status": "success" if ok else "failed",
            "metrics": metrics if ok else {},
            "runtime_seconds": 0.0,
            "timestamp": utc_now_iso(),
            "stdout_tail": _truncate(proc.stdout or "", self.truncation_bounds.get("python_stdout", 4000)),
            "stderr_tail": _truncate(proc.stderr or "", self.truncation_bounds.get("python_stderr", 4000)),
            "script_path": str(script_path),
        }
        append_experiment_log(record)
        self.experiments_run += 1
        # Compress the return payload heavily! Eliminate raw stdout if successful.
        if record["status"] == "success":
            return json.dumps(
                {
                    "experiment_id": exp_id,
                    "status": "success",
                    "metrics": record["metrics"],
                    "retries_used": attempt,
                }
            )
        else:
            return json.dumps(
                {
                    "experiment_id": exp_id,
                    "status": "failed",
                    "stderr_excerpt": (proc.stderr or "")[-self.truncation_bounds.get("python_stderr", 1500):],
                    "retries_used": attempt,
                },
                indent=2,
            )

    def set_evaluation_policy(
        self,
        primary_metric: str,
        direction: str,
        rationale: str,
        secondary_metrics: list[str] | None = None,
    ) -> str:
        spec = {
            "primary_metric": primary_metric,
            "direction": direction,
            "secondary_metrics": secondary_metrics or [],
            "rationale": rationale,
            "source": "agent_decision",
        }
        write_json(state_dir() / "metric_spec.json", spec)
        return f"Saved evaluation policy: primary={primary_metric} ({direction})."

    def get_experiment_history(self, last_n: int = 15) -> str:
        log = read_json(state_dir() / "experiment_log.json")
        if not isinstance(log, list):
            return "[]"
        tail = log[-last_n:]
        # Compress the history dynamically to avoid tracebacks blowing out the agent's context 
        compressed = []
        for e in tail:
            compressed.append({
                "experiment_id": e.get("experiment_id"),
                "status": e.get("status"),
                "metrics": e.get("metrics"),
                "strategy": e.get("strategy_id", e.get("config", {}).get("name", "unknown"))
            })
        return json.dumps(compressed, indent=2)

    def get_metric_spec(self) -> str:
        p = state_dir() / "metric_spec.json"
        if not p.exists():
            return "{}"
        return p.read_text(encoding="utf-8", errors="replace")[:8000]

    def append_journal(self, note: str) -> str:
        line = json.dumps({"ts": utc_now_iso(), "note": note[:8000]}, ensure_ascii=False)
        with open(self.journal_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        return "Recorded."

    def get_leaderboard(self, top_k: int = 5) -> str:
        from agent.leaderboard import rebuild_leaderboard
        from agent.schemas import MetricDirection
        
        ms_p = state_dir() / "metric_spec.json"
        if not ms_p.exists():
            return "No evaluation policy set. Call set_evaluation_policy first."
            
        ms = read_json(ms_p) or {}
        primary = ms.get("primary_metric")
        if not primary:
            return "Primary metric not found in policy."
            
        direction_str = ms.get("direction", "maximize")
        direction = MetricDirection.maximize if direction_str == "maximize" else MetricDirection.minimize
        
        lb = rebuild_leaderboard(primary, direction, top_k)
        if not lb.entries:
            return "Leaderboard is empty. Run successful experiments first."
            
        return lb.model_dump_json(indent=2)
        
    def read_experiment_script(self, experiment_id: str) -> str:
        p = experiments_dir() / experiment_id / "train.py"
        if not p.exists():
            return f"No script found for {experiment_id}"
        limit = self.truncation_bounds.get("experiment_script", 16000)
        return _truncate(p.read_text(encoding="utf-8", errors="replace"), limit)

    def run_parallel_experiments(self, experiments: list[dict[str, Any]]) -> str:
        """Run multiple training experiments. Behavior depends on the
        'parallel_experiments' setting in config.yaml:
          true  → concurrent (up to 4 at once, debug LLM calls serialized)
          false → sequential one-at-a-time (safe for free API tiers)
        """
        import yaml
        from tools.state_store import project_root
        try:
            _cfg = yaml.safe_load((project_root() / "config.yaml").read_text()) or {}
            _parallel = bool(_cfg.get("parallel_experiments", False))
        except Exception:
            _parallel = False

        if not experiments:
            return json.dumps({"error": "No experiments provided."})

        def _run_one(exp: dict) -> tuple[str, str]:
            name = str(exp.get("experiment_name", "unnamed"))
            script = str(exp.get("python_script", ""))
            cfg = exp.get("experiment_config_json")
            return name, self.run_training_experiment(name, script, cfg)

        results: dict[str, Any] = {}

        if _parallel:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            max_workers = min(len(experiments), 4)
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(_run_one, exp): exp for exp in experiments}
                for future in as_completed(futures):
                    try:
                        name, result = future.result()
                        results[name] = json.loads(result) if isinstance(result, str) else result
                    except Exception as e:
                        results[futures[future].get("experiment_name", "unknown")] = {"error": str(e)}
        else:
            # Sequential fallback — safe for free API tiers
            for exp in experiments:
                try:
                    name, result = _run_one(exp)
                    results[name] = json.loads(result) if isinstance(result, str) else result
                except Exception as e:
                    results[exp.get("experiment_name", "unknown")] = {"error": str(e)}

        return json.dumps(results, indent=2)

    def execute(self, name: str, inp: dict[str, Any]) -> str:
        try:
            if name == "inspect_dataset_schema":
                return self.inspect_dataset_schema(str(inp.get("relative_path")))
            if name == "compute_dataset_stats":
                return self.compute_dataset_stats(str(inp.get("relative_path")), inp.get("columns", []))
            if name == "web_search":
                return self.web_search(str(inp["query"]), int(str(inp.get("max_results") or 8)))
            if name == "fetch_url":
                return self.fetch_url(str(inp["url"]))
            if name == "list_dataset":
                return self.list_dataset(str(inp.get("subpath") or ""))
            if name == "read_file":
                return self.read_file(str(inp["relative_path"]), int(str(inp.get("max_chars") or 8000)))
            if name == "read_task_readme":
                return self.read_readme()
            if name == "run_python_analysis":
                return self.run_python_analysis(str(inp["code"]), str(inp.get("purpose") or ""))
            if name == "run_training_experiment":
                return self.run_training_experiment(
                    str(inp["experiment_name"]),
                    str(inp["python_script"]),
                    inp.get("experiment_config_json"),
                )
            if name == "set_evaluation_policy":
                return self.set_evaluation_policy(
                    str(inp["primary_metric"]),
                    str(inp["direction"]),
                    str(inp.get("rationale") or ""),
                    list(inp.get("secondary_metrics") or []),
                )
            if name == "get_experiment_history":
                return self.get_experiment_history(int(inp.get("last_n") or 15))
            if name == "get_metric_spec":
                return self.get_metric_spec()
            if name == "get_leaderboard":
                return self.get_leaderboard(int(inp.get("top_k") or 5))
            if name == "read_experiment_script":
                return self.read_experiment_script(str(inp["experiment_id"]))
            if name == "run_parallel_experiments":
                return self.run_parallel_experiments(list(inp.get("experiments") or []))
            if name == "append_journal":
                return self.append_journal(str(inp["note"]))
            if name == "finish_session":
                return json.dumps({"finished": True, "summary_received": True})
            return f"Unknown tool: {name}"
        except Exception as e:
            return f"Tool error ({name}): {e}"


def anthropic_tool_definitions() -> list[dict[str, Any]]:
    """Anthropic Messages API tool definitions."""
    return [
        {
            "name": "inspect_dataset_schema",
            "description": "Loads a CSV and returns tightly structured JSON metadata of its shape, columns, missing values, and dtypes. ALWAYS prefer this over messy arbitrary python prints.",
            "input_schema": {
                "type": "object",
                "properties": {"relative_path": {"type": "string"}},
                "required": ["relative_path"]
            }
        },
        {
            "name": "compute_dataset_stats",
            "description": "Loads a CSV and returns exact mathematical aggregations for specific columns without returning full rows.",
            "input_schema": {
                "type": "object",
                "properties": {"relative_path": {"type": "string"}, "columns": {"type": "array", "items": {"type": "string"}}},
                "required": ["relative_path", "columns"]
            }
        },
        {
            "name": "web_search",
            "description": "Search the web for methods, papers, libraries, metrics, or best practices. Use when you lack domain knowledge or need up-to-date approaches.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "description": "Max results (default 8)"},
                },
                "required": ["query"],
            },
        },
        {
            "name": "fetch_url",
            "description": "Fetch and read text from a public URL (paper, blog, docs).",
            "input_schema": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
        {
            "name": "list_dataset",
            "description": "List files under the dataset root or a subfolder (paths relative to dataset).",
            "input_schema": {
                "type": "object",
                "properties": {"subpath": {"type": "string", "description": "Optional subpath, default ''"}},
            },
        },
        {
            "name": "read_file",
            "description": "Read a text file from the dataset directory (CSV preview, etc.). Path must be relative to dataset root.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "relative_path": {"type": "string"},
                    "max_chars": {"type": "integer"},
                },
                "required": ["relative_path"],
            },
        },
        {
            "name": "read_task_readme",
            "description": "Read the user-provided task README (full problem statement).",
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "run_python_analysis",
            "description": (
                "Run Python code for EDA, statistics, feature checks, leakage checks, visualizations (print only), "
                "or any exploratory analysis. Environment: DATASET_ROOT, README_PATH. "
                "Use pandas/numpy/sklearn freely. Print findings to stdout."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "purpose": {"type": "string", "description": "What you are testing"},
                    "code": {"type": "string", "description": "Full Python script body (imports allowed)"},
                },
                "required": ["code"],
            },
        },
        {
            "name": "run_training_experiment",
            "description": (
                "Write and execute a training/evaluation script. Must read data from DATASET_ROOT (env var). "
                "Must end by printing exactly: print('METRICS:' + json.dumps({...})) with float metrics. "
                "You choose model, preprocessing, CV strategy, and metric keys—fully autonomous."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "experiment_name": {"type": "string"},
                    "python_script": {"type": "string", "description": "Complete train.py content"},
                    "experiment_config_json": {
                        "type": "object",
                        "description": "Optional metadata extraction. Must match ExperimentConfig schema if provided."
                    }
                },
                "required": ["experiment_name", "python_script"],
            },
        },
        {
            "name": "set_evaluation_policy",
            "description": "Record which metric is primary (maximize/minimize) and why—after you decide from task + literature.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "primary_metric": {"type": "string"},
                    "direction": {"type": "string", "enum": ["maximize", "minimize"]},
                    "rationale": {"type": "string"},
                    "secondary_metrics": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["primary_metric", "direction"],
            },
        },
        {
            "name": "get_experiment_history",
            "description": "See past experiment results (real metrics from logs).",
            "input_schema": {
                "type": "object",
                "properties": {"last_n": {"type": "integer"}},
            },
        },
        {
            "name": "get_leaderboard",
            "description": "View the top-K best experiments ranked by the dynamic primary metric (from metric_spec.json). Use this to identify the best candidates for mutation/reflection.",
            "input_schema": {
                "type": "object",
                "properties": {"top_k": {"type": "integer"}},
            },
        },
        {
            "name": "read_experiment_script",
            "description": "Read the python code of a past experiment (e.g. out of the leaderboard). Use this during reflection to mutate past successful strategies.",
            "input_schema": {
                "type": "object",
                "properties": {"experiment_id": {"type": "string"}},
                "required": ["experiment_id"],
            },
        },
        {
            "name": "get_metric_spec",
            "description": "Current saved evaluation policy JSON if any.",
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "run_parallel_experiments",
            "description": (
                "Run multiple training experiments CONCURRENTLY (up to 4 at once) to save time. "
                "Pass a list of experiments, each with 'experiment_name' and 'python_script'. "
                "Use this when you have 2+ independent hypotheses ready to test simultaneously."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "experiments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "experiment_name": {"type": "string"},
                                "python_script": {"type": "string"},
                                "experiment_config_json": {"type": "object"},
                            },
                            "required": ["experiment_name", "python_script"],
                        },
                        "description": "List of experiments to run in parallel.",
                    }
                },
                "required": ["experiments"],
            },
        },
        {
            "name": "append_journal",
            "description": "Persist a short research note (hypothesis, decision, citation).",
            "input_schema": {
                "type": "object",
                "properties": {"note": {"type": "string"}},
                "required": ["note"],
            },
        },
        {
            "name": "finish_session",
            "description": "Call when done: budget reached, task solved, or no further useful work. Provide summary.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "best_experiment_id": {"type": "string"},
                    "follow_up_ideas": {"type": "string"},
                },
                "required": ["summary"],
            },
        },
    ]


def openai_tool_definitions() -> list[dict[str, Any]]:
    """Generic tool format for API endpoints compatible with OpenAI Message API."""
    out = []
    for tool in anthropic_tool_definitions():
        out.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool.get("input_schema", {"type": "object", "properties": {}})
            }
        })
    return out
