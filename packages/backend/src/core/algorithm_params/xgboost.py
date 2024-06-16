from typing import Literal

from pydantic import Field

from .base import BaseAlgorithmParams

Booster = Literal["gbtree", "gblinear", "dart"]
SamplingMethod = Literal["uniform", "gradient_based"]
ImportanceType = Literal["weight", "gain", "cover", "total_gain", "total_cover"]


class XGBoostParams(BaseAlgorithmParams):
    class Config:
        title = "XGBoost"

    n_estimators: int = Field(default=100, title="Number of boosting rounds")
    max_depth: int | None = Field(default=None, title="Maximum tree depth")
    max_leaves: int | None = Field(default=None, title="Maximum number of leaves")
    max_bin: int | None = Field(default=None, title="Maximum number of bins per feature")
    learning_rate: float | None = Field(default=None, title="Learning rate")
    booster: Booster | None = Field(default=None, title="Booster type")
    gamma: float | None = Field(default=None, title="Minimum loss reduction")
    min_child_weight: float | None = Field(
        default=None,
        title="Minimum sum of instance weight needed in a child"
    )
    max_delta_step: float | None = Field(
        default=None,
        title="Maximum delta step for weight estimation"
    )
    sampling_method: SamplingMethod | None = Field(default=None, title="Sampling method")
    colsample_bytree: float | None = Field(
        default=None,
        title="Subsample ratio of columns for each tree"
    )
    colsample_bylevel: float | None = Field(
        default=None,
        title="Subsample ratio of columns for each level"
    )
    colsample_bynode: float | None = Field(
        default=None,
        title="Subsample ratio of columns for each split"
    )
    reg_alpha: float | None = Field(default=None, title="L1 regularization term on weights")
    reg_lambda: float | None = Field(default=None, title="L2 regularization term on weights")
    scale_pos_weight: float | None = Field(
        default=None,
        title="Balancing of positive and negative weights"
    )
    base_score: float | None = Field(default=None, title="Global bias")
    importance_type: ImportanceType | None = Field(default=None, title="Importance type")
