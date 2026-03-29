# Autoresearcher

Autoresearcher is a fully autonomous, Query Planner-driven Machine Learning experimentation system. Given a task description (`data/README.md`) and a raw dataset, it operates like a data scientist: querying dataset metadata, autonomously forming hypotheses, writing full feature-engineering and model-training scripts, debugging failures, and iterating to find the best-performing pipeline — all without human intervention.

---

## 🚀 Key Features

- **Query Planner Architecture**: The agent never ingests raw data into its prompt. Instead it queries compressed metadata (`inspect_dataset_schema`, `compute_dataset_stats`) and treats the dataset as a statistical black-box, enabling it to scale to arbitrarily large datasets.
- **Observe → Hypothesize → Test → Reflect Loop**: Instead of a fixed pipeline, the agent follows a rigorous scientific loop — observing data properties, forming a hypothesis, running an isolated experiment, and reflecting on results before the next iteration.
- **Parallel Experiment Execution**: Multiple training scripts can be run concurrently (up to 4 at once via `run_parallel_experiments`), reducing wall-clock time by 3–4×.
- **Dedicated Debug Agent**: If a training script crashes (OOM, NaNs, shape mismatches, or missing imports), a dedicated Debug Agent parses the traceback, generates a Python patch, and automatically retries — up to 3 times.
- **Externalized Memory**: All experiment results, journal notes, leaderboards, and metric policies are stored on disk (`state/`). The agent actively queries these files instead of relying on its in-context memory, making it immune to context-window amnesia.
- **Within-Session Improvement**: **Note:** The agent improves its strategies iteratively *within* a single active session by reflecting on its own logs. It does **not** currently feature cross-session learning (it doesn't "remember" lessons from Run A when starting a fresh Run B).
- **Sliding Window Context + Pinning**: The conversation history is capped at `max_history_turns` turns. Foundational tool results (dataset schema, evaluation policy) are pinned and never evicted. Older tool outputs are automatically compressed, cutting history token costs by 40–60%.
- **Per-Tool History Compression**: Each tool type has a configurable character budget for what stays in history. The agent receives the full result immediately, but only a compressed summary persists for future turns.
- **Dynamic Leaderboard & Metric Reasoning**: The agent autonomously selects the most statistically appropriate metric for the task (e.g. PR-AUC for imbalanced classification, RMSE for regression) and ranks all successful experiments in a live leaderboard.
- **Convergence Detection**: If `convergence_window` experiments pass without a leaderboard improvement, the system cleanly halts — saving API quota and compute.
- **Anti-Hallucination Safety**: Metrics are parsed strictly from subprocess stdout using a hardcoded `METRICS:` JSON prefix. The agent can never fake or extrapolate scores.
- **Tiered API Rate Limiting**: Built-in support for free and paid API tiers. Set `api_tier: "free"` to enforce a persistent daily token budget (with progress tracking across runs), or `api_tier: "paid"` for per-minute throttling. Exponential backoff handles transient 429 burst errors automatically.
- **Structured Top-K Reports**: At session end, a Markdown report is generated with chronological improvement trajectories, failed runs, and exact commands to reproduce the Rank 1 experiment.

---

## 💻 How to Run

### Prerequisites
```bash
pip install -r requirements.txt
```

### 1. Autonomous Mode (Default — Recommended)

The primary mode. The agent explores freely, reads literature from the web, runs experiments, and loops until convergence or budget exhaustion.

Requires either `ANTHROPIC_API_KEY` (Claude) **or** `OPENAI_API_KEY` (Groq / Qwen / any OpenAI-compatible endpoint).

**With Groq (Free Tier — Recommended for testing):**
```bash
export OPENAI_API_KEY="gsk_your_groq_key"
export OPENAI_BASE_URL="https://api.groq.com/openai/v1"
# config.yaml → model: "llama-3.3-70b-versatile"
# config.yaml → api_tier: "free"

python main.py --readme data/README.md --dataset data --budget 10 --clear-state
```

**With Anthropic (Claude):**
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python main.py --readme data/README.md --dataset data --budget 15 --clear-state
```

**With Hugging Face / Qwen:**
```bash
export OPENAI_API_KEY="hf_your_token"
export OPENAI_BASE_URL="https://router.huggingface.co/v1"
# config.yaml → model: "Qwen/Qwen2.5-72B-Instruct:novita"
# config.yaml → api_tier: "none"   (HF uses monthly credits, not TPD)

python main.py --readme data/README.md --dataset data --budget 10 --clear-state
```

**Artifacts generated:**
| Path | Contents |
|---|---|
| `state/agent_journal.jsonl` | Agent's research notes and decisions |
| `state/experiment_log.json` | Full metrics log for every run |
| `state/leaderboard.json` | Live ranked leaderboard |
| `state/daily_token_usage.json` | Running daily token counter (free tier) |
| `experiments/exp_auto_*/train.py` | Fully reproducible Python script for every experiment |
| `report/final_report.md` | Human-readable summary: trajectory, top models, reproduce commands |

---

### 2. Legacy Mode (Fixed Pipeline)

A deterministic, linear LangGraph pipeline (data profile → heuristic metrics → sklearn baselines). Useful for quick smoke tests without an agent loop.

```bash
python main.py --mode legacy --readme data/README.md --dataset data --budget 5 --clear-state
```

---

## ⚙️ Configuration (`config.yaml`)

### ⏱ Execution Limits

| Parameter | Default | Description |
|---|---|---|
| `max_experiments` | `30` | Hard cap on total training runs per session. Agent refuses to run beyond this. |
| `max_agent_turns` | `48` | Max LLM reasoning rounds. Each turn may invoke multiple tools. Prevents infinite loops. |
| `analysis_timeout_seconds` | `300` | Max time (5 min) an EDA python script can run before being killed. |
| `max_time_per_experiment_seconds` | `7200` | Max time (2 hours) a training script can run before being killed. |
| `max_debug_retries` | `3` | How many times the debug agent retries a failed experiment before giving up. |
| `debug_llm_timeout_seconds` | `30` | Max seconds the debug agent's LLM call can hang. Releases the lock so parallel threads don't deadlock. |
| `max_total_wall_time_hours` | `4` | Hard session wall-clock limit. Stops everything after 4 hours regardless of budget. |

### 🧠 Agent Intelligence

| Parameter | Default | Description |
|---|---|---|
| `convergence_window` | `8` | Stop if the leaderboard metric doesn't improve for this many consecutive experiments. |
| `max_research_sources` | `20` | Max web search results the agent may fetch during a session. |
| `reflection_every_n` | `5` | Every N experiments, the agent re-reads past scripts to plan mutations. |
| `target_metric_threshold` | `null` | If set (e.g. `0.95`), agent stops as soon as the primary metric hits this value. |
| `max_history_turns` | `5` | Sliding window size for conversation history. Foundational tool results (schema, evaluation policy) are pinned and never evicted. |

### ⚡ Parallel Execution

| Parameter | Default | Description |
|---|---|---|
| `parallel_experiments` | `false` | `true` → runs up to 4 training scripts simultaneously (3-4× faster). `false` → one at a time. Keep `false` on free API tiers — parallel debug retries can spike RPM. |

### 🔑 API Rate Limiting

| Parameter | Default | Description |
|---|---|---|
| `api_tier` | `"free"` | `"free"` → daily token budget (Groq free: 90K/day). `"paid"` → per-minute throttle (Groq dev: 250K TPM). `"none"` → no limiting (Anthropic / OpenAI / self-hosted). |
| `tokens_per_day_limit` | `null` | Override the free-tier daily budget. `null` uses tier default (90,000). |
| `tokens_per_minute_limit` | `null` | Override the paid-tier per-minute limit. `null` uses tier default (250,000). |

### ✂️ Context Compression (`truncation_bounds`)

| Key | Default | Description |
|---|---|---|
| `python_stdout` | `4000` | Max chars of a script's stdout stored in history. |
| `python_stderr` | `2000` | Max chars of a script's stderr stored in history. |
| `read_file` | `3000` | Max chars when reading any file via the `read_file` tool. |
| `read_readme` | `6000` | Max chars of the task README embedded into the agent's context. |
| `experiment_script` | `6000` | Max chars when re-reading a past training script via `read_experiment_script`. |

### 🤖 Model & Infrastructure

| Parameter | Default | Description |
|---|---|---|
| `model` | `llama-3.3-70b-versatile` | LLM used for orchestration, debugging, and metric reasoning. |
| `tavily_api_key_env` | `"TAVILY_API_KEY"` | Env variable name for the Tavily search API key. Falls back to DuckDuckGo if unset. |
| `use_docker_sandbox` | `false` | Run training scripts inside Docker for isolation. Keep `false` unless running untrusted code. |
| `docker_memory_limit` | `"16g"` | RAM limit for the Docker sandbox container. |
| `docker_cpu_limit` | `8` | CPU core limit for the Docker sandbox container. |

---

## 📝 Writing Your Task README (`data/README.md`)

The `data/README.md` file is your **direct instruction layer** to the agent. It is embedded into the agent's context at session start and governs all its decisions.

You can be as loose or as precise as you want:

- **Loose**: `"Predict customer churn."` — The agent will autonomously determine metrics, features, and models.
- **Strict**: List exact preprocessing steps, required metrics, forbidden model types, or mandatory evaluation strategies.

The agent will follow your README as its primary directive throughout the entire session.

---

## 🆕 Starting a Fresh Task

To start a completely new research project:

1.  **Clear Old Results**: Use the `--clear-state` flag in your next run, or manually delete the `state/` and `experiments/` directories.
2.  **Replace Dataset**: Place your new dataset (e.g., `my_data.csv`) into the `data/` directory.
3.  **Update README**: Edit `data/README.md` to describe your new goal and any specific constraints.
4.  **Run**:
    ```bash
    python main.py --readme data/README.md --dataset data --budget 10 --clear-state
    ```

---

## 🏗️ Beyond Tabular: DL & NLP

While the built-in EDA tools (`inspect_dataset_schema`, `compute_dataset_stats`) are currently optimized for **Tabular/CSV** data, the Autoresearcher is a general-purpose ML agent:

- **Computer Vision & NLP**: The agent can write and execute **any Python code**. It can import `torch`, `tensorflow`, or `transformers` to build Deep Learning models or handle unstructured text/image data.
- **Manual EDA**: For non-CSV formats, the agent will autonomously pivot to using `run_python_analysis` to explore your data manually rather than relying on the "shortcut" tabular tools.
- **Compute Requirements**: Ensure your environment has the necessary libraries and hardware (e.g., CUDA-enabled GPUs) if you assign it heavy DL tasks.

---

## 🧠 Architecture: Sensors vs. Reasoning

It is important to understand the Autoresearcher is **fully autonomous**, not a fixed-rule system:

1.  **The Sensors**: Tools like `inspect_dataset_schema` are simply "sensors" that provide high-speed, structured metadata to the agent. They don't dictate what to do; they just provide the map.
2.  **The Brain**: The LLM (e.g., Gemini or Claude) acts as the central reasoning engine. It takes the "sensor" data and **reasons from first principles** about features, model architectures, and debugging strategies.
3.  **The Execution**: The agent generates raw Python code from scratch for every experiment. It isn't picking from a list of templates; it is writing the code it believes will best solve the task.

---

## 🔒 Security Note

Autonomous mode locally **executes model-produced arbitrary Python code** in subprocesses scoped to your dataset path. Strict timeouts are enforced. Use in sandboxed environments (Dev Containers or Docker) and only with data you trust.
