"""Pydantic models for task profile, metrics, experiments, and research notes."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    binary_classification = "binary_classification"
    multiclass_classification = "multiclass_classification"
    regression = "regression"
    multilabel_classification = "multilabel_classification"
    image_classification = "image_classification"
    segmentation = "segmentation"
    nlp_classification = "nlp_classification"
    time_series = "time_series"
    tabular = "tabular"
    unknown = "unknown"


class DatasetType(str, Enum):
    tabular_csv = "tabular_csv"
    image_folder = "image_folder"
    text_folder = "text_folder"
    numpy_npz = "numpy_npz"
    unknown = "unknown"


class MetricDirection(str, Enum):
    maximize = "maximize"
    minimize = "minimize"


class CustomMetricDef(BaseModel):
    """Agent-proposed custom metric: name + Python body (function of y_true, y_pred)."""

    name: str = Field(..., description="Snake_case metric name for METRICS JSON")
    description: str = ""
    python_body: str = Field(
        ...,
        description="Function body only, with y_true, y_pred in scope (numpy arrays).",
    )


class MetricSpec(BaseModel):
    """Reasoned primary/secondary metrics and optional custom scorers."""

    primary_metric: str
    direction: MetricDirection = MetricDirection.maximize
    secondary_metrics: list[str] = Field(default_factory=list)
    rationale: str = ""
    custom_metrics: list[CustomMetricDef] = Field(default_factory=list)
    task_type_inferred: TaskType = TaskType.unknown
    uses_probability: bool = False  # e.g. ROC-AUC needs predict_proba


class TaskProfile(BaseModel):
    raw_readme_excerpt: str = ""
    task_type: TaskType = TaskType.unknown
    domain_notes: str = ""
    constraints: list[str] = Field(default_factory=list)
    goal_summary: str = ""
    readme_stated_metric: Optional[str] = None  # if README names one explicitly


class ExperimentConfig(BaseModel):
    experiment_id: str = ""
    strategy_id: str = ""
    model_class: str = "sklearn_rf"  # e.g. resnet50, bert-base-uncased, sklearn_rf
    pretrained: bool = False
    optimizer: str = "AdamW"
    lr: float = 1e-3
    batch_size: int = 32
    epochs: int = 10
    scheduler: Optional[str] = None
    loss_fn: str = "cross_entropy"
    augmentations: list[str] = Field(default_factory=list)
    extra_kwargs: dict[str, Any] = Field(default_factory=dict)
    # Custom model: optional full Python module defining build_model(**kwargs) -> torch.nn.Module or sklearn estimator
    custom_model_code: Optional[str] = None
    # Custom metrics merged into training script (names must match MetricSpec.custom_metrics)
    custom_metric_defs: list[CustomMetricDef] = Field(default_factory=list)


class ResearchStrategy(BaseModel):
    id: str
    name: str
    source: str = ""
    architecture: str = ""
    optimizer: str = ""
    lr: Optional[float] = None
    scheduler: Optional[str] = None
    augmentation: list[str] = Field(default_factory=list)
    claimed_metric: dict[str, float] = Field(default_factory=dict)
    applicability_score: float = 0.5
    notes: str = ""


class ResearchNotes(BaseModel):
    strategies: list[ResearchStrategy] = Field(default_factory=list)
    query_used: str = ""


class ExperimentRecord(BaseModel):
    experiment_id: str
    strategy_id: str
    config: dict[str, Any]
    status: Literal["success", "failed", "broken"]
    metrics: dict[str, float] = Field(default_factory=dict)
    runtime_seconds: float = 0.0
    timestamp: str = ""
    stdout_tail: str = ""
    stderr_tail: str = ""
    script_path: str = ""


class LeaderboardEntry(BaseModel):
    rank: int
    experiment_id: str
    strategy_name: str
    metrics: dict[str, float] = Field(default_factory=dict)
    improvement_over_baseline: str = ""


class Leaderboard(BaseModel):
    target_metric: str
    direction: MetricDirection = MetricDirection.maximize
    top_k: int = 5
    entries: list[LeaderboardEntry] = Field(default_factory=list)


class OrchestratorAction(BaseModel):
    """Either next experiment JSON or stop."""

    action: Literal["experiment", "stop", "reflect"] = "experiment"
    experiment: Optional[ExperimentConfig] = None
    reason: str = ""


class BudgetState(BaseModel):
    experiments_remaining: int
    wall_time_deadline_iso: str = ""
