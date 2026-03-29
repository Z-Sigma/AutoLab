# Autonomous ML/DL Research Agent — Blueprint

> A fully autonomous multi-agent system that reads a task description and dataset,
> autonomously experiments with models and strategies, self-corrects errors, and
> delivers a ranked report of the top-K approaches with full reproducibility.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Agent Definitions](#3-agent-definitions)
4. [Recursive Experiment Loop](#4-recursive-experiment-loop)
5. [Anti-Hallucination Design](#5-anti-hallucination-design)
6. [State & Memory Schema](#6-state--memory-schema)
7. [Tool Registry](#7-tool-registry)
8. [File & Folder Structure](#8-file--folder-structure)
9. [Tech Stack](#9-tech-stack)
10. [Implementation Phases](#10-implementation-phases)
11. [Prompt Templates](#11-prompt-templates)
12. [Report Output Format](#12-report-output-format)
13. [Hard Limits & Safety Guards](#13-hard-limits--safety-guards)

---

## 1. System Overview

The user provides:
- `README.md` — task description, goals, evaluation metric, constraints
- `/dataset/` — folder path to raw data
- `K` — number of top strategies to surface in the final report

The system autonomously:
1. Understands the ML task type (classification, regression, segmentation, NLP, etc.)
2. Researches SOTA approaches via web search (arxiv, Papers With Code, HuggingFace)
3. Profiles the dataset (EDA, class balance, feature statistics)
4. Proposes and executes experiments in a sandboxed environment
5. Captures real metrics from actual runs — never hallucinated
6. Debugs and self-corrects failures up to a retry budget
7. Ranks all successful experiments and iteratively mutates the best ones
8. Outputs a final report: top-K strategies, metrics, code, and improvement history

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                          │
│         README.md  ·  /dataset/  ·  K  ·  budget           │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              ORCHESTRATOR AGENT  (Claude Opus)              │
│  · Parses task · Builds experiment plan · Tracks state      │
│  · Routes sub-agents · Decides convergence / stopping       │
└────┬────────────┬────────────┬────────────┬─────────────────┘
     │            │            │            │
     ▼            ▼            ▼            ▼
┌─────────┐ ┌─────────┐ ┌──────────┐ ┌──────────┐
│Research │ │  Data   │ │Experiment│ │  Debug   │
│ Agent   │ │ Agent   │ │  Agent   │ │  Agent   │
│         │ │         │ │          │ │          │
│Web search│ │EDA/clean│ │Train/eval│ │Error fix │
│Papers   │ │Feature  │ │Tune/iter │ │Self-patch│
│SOTA     │ │engineer │ │Log metrics│ │Retry     │
└────┬────┘ └────┬────┘ └────┬─────┘ └────┬─────┘
     └───────────┴───────────┴────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    EXECUTION LAYER                          │
│   Python sandbox (Docker) · GPU/CPU compute · File I/O      │
│   Subprocess runner · stdout/stderr capture · Timeout       │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                 MEMORY + STATE STORE                        │
│  experiment_log.json · data_profile.json · leaderboard.db   │
│  tried_strategies.json · error_log.json · research_cache    │
└─────────────────────────┬───────────────────────────────────┘
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
    [budget / target met?]    [more budget left]
              │                       │
              ▼                       │
┌────────────────────────┐            │
│    REPORT GENERATOR    │◄───────────┘
│  Top-K strategies      │   (recursive improvement loop)
│  Metric tables         │
│  Per-strategy explain  │
│  Improvement history   │
│  Code + config         │
└────────────────────────┘
```

---

## 3. Agent Definitions

### 3.1 Orchestrator Agent

**Role:** The central brain. Reads all inputs, maintains global state, decides what to run next.

**Responsibilities:**
- Parse `README.md` to extract: task type, target metric, constraints, domain
- Classify the ML problem (binary classification / multi-class / regression / object detection / NLP / time series / etc.)
- Build an initial experiment plan (ordered list of strategy candidates)
- After each experiment, read the leaderboard and decide: mutate best? try new arch? stop?
- Enforce all hard limits (budget, retries, time)
- Never generate a metric value — only read from `experiment_log.json`

**Prompt backbone:** Claude claude-opus-4-6 with extended thinking enabled

**Input context every turn:**
```
- task_profile.json       (parsed from README)
- data_profile.json       (from Data Agent)
- experiment_log.json     (all past runs + metrics)
- leaderboard.json        (top-K so far)
- research_notes.json     (from Research Agent)
- tried_strategies.json   (to avoid repeats)
- current_budget.json     (experiments remaining)
```

---

### 3.2 Research Agent

**Role:** Grounds the system in real-world knowledge, eliminating hallucinated strategy suggestions.

**Responsibilities:**
- Given the task type, search for SOTA approaches:
  - arxiv (via arxiv API or Semantic Scholar)
  - Papers With Code (pwc API)
  - HuggingFace Hub (model cards)
  - Towards Data Science / Medium articles
- Extract: model architectures, loss functions, hyperparameter ranges, augmentation strategies, training tricks
- Return structured `research_notes.json` — each entry has: source URL, strategy name, claimed metric, applicability score for our task

**Tools available:**
- `web_search(query)` → Tavily / SerpAPI
- `fetch_url(url)` → scrape paper abstract or blog
- `search_pwc(task)` → Papers With Code API
- `search_hf(task)` → HuggingFace model search

**Output schema:**
```json
{
  "strategies": [
    {
      "id": "strat_001",
      "name": "EfficientNet-B3 with CosineAnnealingLR",
      "source": "https://paperswithcode.com/...",
      "architecture": "EfficientNet-B3",
      "optimizer": "AdamW",
      "lr": 3e-4,
      "scheduler": "CosineAnnealingLR",
      "augmentation": ["RandomHorizontalFlip", "MixUp"],
      "claimed_metric": {"accuracy": 0.943},
      "applicability_score": 0.87,
      "notes": "SOTA on ImageNet subset as of 2024"
    }
  ]
}
```

---

### 3.3 Data Agent

**Role:** Understands the dataset so every experiment is correctly configured from the start.

**Responsibilities:**
- Run automated EDA on the dataset folder
- Detect: data type (tabular/image/text/audio/time-series), shape, size
- Compute: class distribution, missing value rates, feature correlations, target statistics
- Flag issues: class imbalance, data leakage risk, low-variance features, duplicates
- Recommend: preprocessing pipeline, train/val/test split strategy, normalization approach
- Output `data_profile.json`

**Tools available:**
- `run_python(script)` → sandboxed execution
- `read_file(path)` → read CSV, Parquet, images, etc.
- `list_dir(path)` → enumerate dataset files

**Output schema:**
```json
{
  "dataset_type": "image_classification",
  "n_samples": 12400,
  "n_classes": 5,
  "class_distribution": {"cat": 2800, "dog": 2600, ...},
  "imbalance_ratio": 1.31,
  "recommended_split": {"train": 0.7, "val": 0.15, "test": 0.15},
  "image_size_mode": [224, 224],
  "recommended_normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
  "issues_found": ["mild class imbalance — consider weighted loss"],
  "preprocessing_pipeline": ["Resize(224)", "ToTensor()", "Normalize(...)"]
}
```

---

### 3.4 Experiment Agent

**Role:** Executes a single fully-specified experiment and returns verified metrics.

**Responsibilities:**
- Receive a complete experiment config (from Orchestrator)
- Generate the training script from a Pydantic-validated template (no free-form code gen)
- Run the script in Docker sandbox with timeout
- Capture stdout/stderr fully
- Parse the final JSON metrics line printed by the script
- Write result to `experiment_log.json`
- Never infer or guess a metric — if parsing fails, mark experiment as `failed`

**Experiment config schema (Pydantic):**
```python
class ExperimentConfig(BaseModel):
    experiment_id: str
    strategy_id: str
    model_class: str           # e.g. "resnet50", "bert-base-uncased"
    pretrained: bool
    optimizer: str             # "AdamW" | "SGD" | "Adam"
    lr: float
    batch_size: int
    epochs: int
    scheduler: Optional[str]
    loss_fn: str
    augmentations: List[str]
    extra_kwargs: Dict[str, Any]
```

**Required last line of every generated training script:**
```python
import json
print("METRICS:" + json.dumps({
    "accuracy": float(acc),
    "val_loss": float(val_loss),
    "f1": float(f1),
    "epoch_times": epoch_times
}))
```

---

### 3.5 Debug Agent

**Role:** Activated on any experiment failure. Reads the error, patches the script, retries.

**Responsibilities:**
- Receive: failed script + full stderr traceback + experiment config
- Classify error type:
  - `shape_mismatch` → fix tensor dims
  - `oom` → halve batch size, add gradient checkpointing
  - `import_error` → add missing install or fix import path
  - `dtype_error` → add explicit cast
  - `nan_loss` → reduce LR, clip gradients, check data normalization
  - `timeout` → reduce epochs for this run
  - `unknown` → log and skip (mark strategy `broken`)
- Apply targeted patch (not a full rewrite)
- Re-run experiment
- Max 3 retries per experiment before abandoning

**Error classification prompt:**
```
Given this Python traceback: {traceback}
And this experiment config: {config}
Classify the error as one of: [shape_mismatch, oom, import_error, dtype_error, nan_loss, timeout, unknown]
Then output ONLY a JSON patch: {"line_to_replace": "...", "replacement": "..."}
Do not rewrite the full script.
```

---

## 4. Recursive Experiment Loop

```
┌──────────────────────────────────────────────────────┐
│              PROPOSE EXPERIMENT                       │
│  Orchestrator picks next strategy from pool           │
│  (informed by leaderboard + research notes)           │
└─────────────────────┬────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────┐
│              EXECUTE IN SANDBOX                       │
│  Docker container · captured stdout/stderr            │
│  Hard timeout enforced                                │
└─────────────────────┬────────────────────────────────┘
                      │
              ┌───────┴───────┐
              │               │
           success?         failure?
              │               │
              ▼               ▼
┌─────────────────┐  ┌─────────────────────────────────┐
│ Parse METRICS:  │  │         DEBUG AGENT              │
│ JSON from stdout│  │  Classify error → patch → retry  │
│ Write to log    │  │  Max 3 retries → mark broken     │
└────────┬────────┘  └──────────────┬──────────────────┘
         │                          │ (after max retries)
         ▼                          ▼
┌──────────────────────────────────────────────────────┐
│           UPDATE LEADERBOARD                          │
│  Rank all experiments by target metric                │
│  Prune strategies with no improvement path            │
└─────────────────────┬────────────────────────────────┘
                      │
              ┌───────┴───────┐
              │               │
        budget left?    budget/target met?
              │               │
              ▼               ▼
┌─────────────────────┐  ┌──────────────────────────────┐
│  REFLECTION PASS    │  │      REPORT GENERATOR         │
│  Compare top-3 runs │  │  Pull top-K from leaderboard  │
│  Identify what      │  │  Generate Markdown report     │
│  helped (depth, LR, │  │  Include code + configs       │
│  augment, etc.)     │  │  Improvement trajectory chart │
│  Generate next      │  └──────────────────────────────┘
│  batch as mutations │
└─────────┬───────────┘
          │
          └──────────► [back to PROPOSE EXPERIMENT]
```

### Reflection Pass Logic

After every N experiments (configurable, default N=5), the orchestrator runs a reflection:

```
Given these experiment results: {leaderboard_top_5}
Identify which changes correlated with metric improvements.
Propose 3 new experiments as mutations of the best performer.
Rules:
  - Change only ONE hyperparameter per mutation
  - Stay within the ranges validated by research_notes.json
  - Do not propose any config already in tried_strategies.json
Output ONLY a JSON array of ExperimentConfig objects.
```

---

## 5. Anti-Hallucination Design

| Risk | Mitigation |
|------|------------|
| Model invents metric values | Metrics only parsed from `METRICS:` JSON stdout line |
| Model suggests configs it can't verify | Configs must match Pydantic schema — invalid fields rejected |
| Model re-runs failed strategies | `tried_strategies.json` is injected into every orchestrator prompt |
| Model references non-existent papers | Research agent always provides source URLs; orchestrator only uses cited strategies |
| Model inflates improvement claims | Report generator diffs metrics from actual log entries, not model memory |
| Reflection proposes impossible mutations | Mutations validated against data_profile.json constraints before dispatch |

**Key prompt constraint injected into Orchestrator every turn:**
```
HARD RULE: You may ONLY reference metric values that appear verbatim
in experiment_log.json. You must NEVER estimate, extrapolate, or assume
a metric. If a metric is missing, the experiment is marked FAILED.
```

---

## 6. State & Memory Schema

### `experiment_log.json`
```json
[
  {
    "experiment_id": "exp_007",
    "strategy_id": "strat_003",
    "config": { ... },
    "status": "success",
    "metrics": {
      "accuracy": 0.8731,
      "val_loss": 0.3412,
      "f1": 0.8654
    },
    "runtime_seconds": 847,
    "timestamp": "2024-11-15T14:32:00Z",
    "stdout_tail": "...",
    "script_path": "experiments/exp_007/train.py"
  }
]
```

### `leaderboard.json`
```json
{
  "target_metric": "accuracy",
  "top_k": 5,
  "entries": [
    {
      "rank": 1,
      "experiment_id": "exp_007",
      "strategy_name": "EfficientNet-B3 + MixUp + CosineAnnealingLR",
      "accuracy": 0.8731,
      "f1": 0.8654,
      "improvement_over_baseline": "+12.4%"
    }
  ]
}
```

### `tried_strategies.json`
```json
{
  "tried": [
    "resnet50_adam_lr0.001_bs32",
    "efficientnet_b3_adamw_lr0.0003_bs64_mixup"
  ]
}
```

---

## 7. Tool Registry

| Tool Name | Agent | Description |
|-----------|-------|-------------|
| `web_search(query)` | Research | Tavily / SerpAPI web search |
| `fetch_url(url)` | Research | Scrape page content |
| `search_pwc(task)` | Research | Papers With Code API |
| `search_hf(task)` | Research | HuggingFace Hub search |
| `run_python(script, timeout)` | Experiment, Data | Docker sandbox execution |
| `read_file(path)` | Data, Debug | Read any file from disk |
| `write_file(path, content)` | All | Write output files |
| `list_dir(path)` | Data | Enumerate directory |
| `parse_metrics(stdout)` | Experiment | Extract METRICS: JSON line |
| `patch_script(script, patch)` | Debug | Apply line-level patch |
| `read_log()` | Orchestrator | Read experiment_log.json |
| `read_leaderboard()` | Orchestrator | Read current top-K |
| `update_leaderboard(entry)` | Experiment | Add result to leaderboard |

---

## 8. File & Folder Structure

```
autonomous-ml-agent/
│
├── agent/
│   ├── orchestrator.py         # Main orchestrator loop
│   ├── research_agent.py       # Web search + paper parsing
│   ├── data_agent.py           # EDA + data profiling
│   ├── experiment_agent.py     # Training script gen + execution
│   ├── debug_agent.py          # Error classification + patching
│   └── report_generator.py     # Final report builder
│
├── tools/
│   ├── sandbox.py              # Docker subprocess runner
│   ├── web_search.py           # Tavily / SerpAPI wrapper
│   ├── file_tools.py           # read/write/list utilities
│   ├── metric_parser.py        # METRICS: stdout parser
│   └── script_templates.py     # Pydantic config → train script
│
├── state/
│   ├── experiment_log.json     # All experiment results
│   ├── leaderboard.json        # Top-K ranked results
│   ├── tried_strategies.json   # Deduplication store
│   ├── data_profile.json       # Dataset analysis output
│   ├── research_notes.json     # Grounded strategy candidates
│   ├── task_profile.json       # Parsed from README.md
│   └── error_log.json          # All failures + tracebacks
│
├── experiments/
│   ├── exp_001/
│   │   ├── train.py
│   │   ├── config.json
│   │   └── stdout.log
│   ├── exp_002/
│   └── ...
│
├── report/
│   └── final_report.md         # Output report
│
├── docker/
│   └── Dockerfile              # Sandbox environment
│
├── prompts/
│   ├── orchestrator_system.txt
│   ├── research_agent_system.txt
│   ├── debug_agent_system.txt
│   └── reflection_prompt.txt
│
├── config.yaml                 # Top-K, budget, timeouts
├── main.py                     # Entry point
└── README.md                   # How to run
```

---

## 9. Tech Stack

| Layer | Choice | Reason |
|-------|--------|--------|
| LLM backbone | Claude claude-opus-4-6 (extended thinking) | Best reasoning for complex planning |
| Agent orchestration | LangGraph | Explicit graph control, conditional edges, state passing |
| Web search | Tavily API | Best for research/academic queries |
| Code sandbox | Docker + subprocess | Isolation, prevents orchestrator crashes |
| Experiment tracking | JSON files + SQLite | Simple, inspectable, no server needed |
| ML framework | PyTorch + HuggingFace Transformers | Broadest model coverage |
| Hyperparameter tuning | Optuna (optional, Phase 4) | Bayesian search within promising configs |
| Config validation | Pydantic v2 | Prevents malformed experiment configs |
| Report output | Markdown + optional Weights & Biases | Human-readable + visual dashboards |

---

## 10. Implementation Phases

### Phase 1 — Scaffold (Week 1)
- [ ] `main.py` entry point: read README + dataset path + K
- [ ] Orchestrator: parse task, output experiment plan as JSON
- [ ] Experiment Agent: receive config, run hardcoded template, return metrics
- [ ] State store: write/read `experiment_log.json`
- [ ] Run 3 baseline experiments sequentially (no loops yet)

### Phase 2 — Research Integration (Week 2)
- [ ] Research Agent: connect Tavily API
- [ ] Parse Papers With Code API for top models per task type
- [ ] Output `research_notes.json`
- [ ] Orchestrator reads research notes before proposing experiments

### Phase 3 — Debug Loop (Week 2–3)
- [ ] Debug Agent: error classification prompt
- [ ] Patch application + retry logic
- [ ] Max retry budget enforcement
- [ ] `error_log.json` writing

### Phase 4 — Recursive Improvement (Week 3–4)
- [ ] Leaderboard ranking after each experiment
- [ ] Reflection pass prompt every N experiments
- [ ] Mutation generation (single-param changes)
- [ ] `tried_strategies.json` deduplication
- [ ] Convergence detection (if top metric hasn't improved in M experiments, stop)

### Phase 5 — Report + Polish (Week 4)
- [ ] Report generator: pull top-K, format Markdown
- [ ] Improvement trajectory table
- [ ] Per-strategy explanation (grounded in research notes)
- [ ] Save winning model checkpoints + configs
- [ ] CLI flags: `--readme`, `--dataset`, `--k`, `--budget`, `--metric`

---

## 11. Prompt Templates

### Orchestrator System Prompt
```
You are the Orchestrator of an autonomous ML experimentation system.

Your job:
1. Read the task profile, data profile, and research notes provided.
2. Review the experiment log and leaderboard to understand what has been tried.
3. Propose the NEXT experiment to run, as a valid ExperimentConfig JSON object.

HARD RULES:
- You may ONLY reference metric values that appear verbatim in experiment_log.json.
- NEVER estimate, guess, or extrapolate metrics.
- NEVER propose a config that already appears in tried_strategies.json.
- Output ONLY valid JSON matching the ExperimentConfig schema. No prose.
- If the target metric has not improved in the last {convergence_window} experiments, output {"action": "stop"}.
```

### Reflection Prompt
```
Given these top experiment results:
{leaderboard_top_5}

Identify which hyperparameter changes correlated with improvements.
Then propose exactly 3 new experiments as single-parameter mutations of the best result.

Constraints:
- Change only ONE parameter per mutation
- Keep all values within ranges in research_notes.json
- Do not repeat any config in tried_strategies.json

Output: JSON array of 3 ExperimentConfig objects only. No prose.
```

### Debug Agent Prompt
```
A training script failed with this error:
{traceback}

Experiment config: {config}

Step 1: Classify the error as one of:
  shape_mismatch | oom | import_error | dtype_error | nan_loss | timeout | unknown

Step 2: Output a minimal patch as JSON:
  {"error_type": "...", "line_to_replace": "...", "replacement": "..."}

Rules:
- Do NOT rewrite the full script
- Do NOT change the model architecture
- For OOM: halve batch_size only
- For nan_loss: multiply lr by 0.1 only
- Output only the JSON patch, no prose
```

---

## 12. Report Output Format

```markdown
# Autonomous ML Experiment Report

**Task:** {task_description}
**Dataset:** {dataset_path}
**Target metric:** {metric}
**Total experiments run:** {n}
**Total time:** {duration}

---

## Top-K Strategies

### Rank 1 — {strategy_name}
- **Metric:** accuracy = 0.8731, F1 = 0.8654
- **Architecture:** EfficientNet-B3
- **Key config:** lr=3e-4, batch=64, MixUp α=0.4, CosineAnnealingLR
- **Source:** [Papers With Code — CIFAR-5 SOTA](https://...)
- **Improvements over baseline:** +12.4% accuracy
- **Config file:** experiments/exp_007/config.json
- **Script:** experiments/exp_007/train.py

### Rank 2 — ...

---

## Improvement Trajectory

| Experiment | Strategy           | Accuracy | ΔAccuracy |
|------------|--------------------|----------|-----------|
| exp_001    | ResNet50 baseline  | 0.749    | —         |
| exp_003    | + MixUp            | 0.781    | +3.2%     |
| exp_007    | + EfficientNet-B3  | 0.873    | +9.2%     |

---

## What Made the Difference

1. **Architecture upgrade** (ResNet → EfficientNet): +6.1% accuracy
2. **MixUp augmentation**: +3.2% accuracy, better calibration
3. **CosineAnnealingLR vs StepLR**: +1.8% accuracy
4. **Weighted loss for class imbalance**: +0.9% F1

---

## Failed Strategies & Why

| Strategy | Error | Action Taken |
|----------|-------|-------------|
| ViT-B/16 | OOM (batch=32) | Reduced to 16, still OOM on available GPU |
| LSTM baseline | nan_loss | LR reduced ×10, still diverged — architecture dropped |

---

## How to Reproduce Rank 1

```bash
python experiments/exp_007/train.py \
  --data /path/to/dataset \
  --config experiments/exp_007/config.json
```
```

---

## 13. Hard Limits & Safety Guards

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `max_experiments` | 30 | Total experiment budget |
| `max_time_per_experiment` | 2 hours | Prevent runaway training jobs |
| `max_debug_retries` | 3 | Per-experiment debug attempts |
| `max_total_wall_time` | 24 hours | Total system runtime cap |
| `convergence_window` | 8 | Stop if no improvement in last N exps |
| `max_research_sources` | 20 | Limit web search to prevent distraction |
| `docker_memory_limit` | 16GB | Sandbox memory cap |
| `docker_cpu_limit` | 8 cores | Sandbox CPU cap |

### Convergence Criteria

The system stops early (before `max_experiments`) if any of these are true:
1. Target metric reaches user-specified threshold (e.g. accuracy ≥ 0.95)
2. No improvement in the last `convergence_window` experiments
3. All strategies from `research_notes.json` have been tried
4. `max_total_wall_time` exceeded

---

*Generated by Claude — Autonomous ML Agent Blueprint v1.0*
