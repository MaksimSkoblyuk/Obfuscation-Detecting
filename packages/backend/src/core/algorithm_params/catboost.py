from typing import Literal

from pydantic import (
    Field,
    field_validator
)

from .base import BaseAlgorithmParams

LossFunction = Literal["Logloss", "CrossEntropy"]
OverfittingDetectorType = Literal["IncToDec", "Iter"]
NanProcessingPossibleModes = Literal["Min", "Max", "Forbidden"]
LeafEstimationMethod = Literal["Gradient", "Newton"]
FeatureBorderType = Literal[
    "Median",
    "Uniform",
    "MaxLogSum",
    "MinEntropy",
    "GreedyLogSum",
    "UniformAndQuantiles"
]


class CatBoostParams(BaseAlgorithmParams):
    class Config:
        title = "CatBoost"
        protected_namespaces = ()

    iterations: int = Field(default=500, title="Max count of trees", ge=1)
    learning_rate: float = Field(default=0.03, title="Learning rate", gt=0.0, le=1.0)
    depth: int = Field(default=6, title="Max depth of trees", ge=1, le=16)
    l2_leaf_reg: float = Field(default=3.0, title="L2 regularization coefficient", gt=0.0)
    model_size_reg: float | None = Field(
        default=None,
        title="Model size regularization coefficient"
    )
    loss_function: LossFunction = Field(default="Logloss", title="Loss function")
    nan_mode: NanProcessingPossibleModes | None = Field(
        default=None,
        title="Way to process missing values for numeric features"
    )
    leaf_estimation_method: LeafEstimationMethod | None = Field(
        default=None,
        title="Leaf estimation method"
    )
    best_model_min_trees: int | None = Field(default=None, title="Min number of trees")
    max_leaves: int = Field(default=31, title="Max number of leaves")

    @field_validator("model_size_reg", mode="after")
    @classmethod
    def validate_non_negative_number(cls, value: float | None) -> float | None:
        if isinstance(value, float) and value < 0.0:
            raise ValueError("Model size regularization coefficient must be non-negative")
        return value
