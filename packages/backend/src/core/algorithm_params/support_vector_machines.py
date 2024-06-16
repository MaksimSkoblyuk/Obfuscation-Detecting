from typing import Literal

from pydantic import (
    Field,
    field_validator
)

from .base import BaseAlgorithmParams

Gamma = Literal["scale", "auto"]
DecisionFunctionShape = Literal["ovo", "ovr"]
Kernel = Literal["linear", "poly", "rbf", "sigmoid", "precomputed"]


class SVCParams(BaseAlgorithmParams):
    class Config:
        title = "Support Vector Machines"

    C: float = Field(default=1.0, title="Regularization parameter")
    kernel: Kernel = Field(default="rbf", title="Kernel")
    degree: int = Field(default=3, title="Degree")
    gamma: Gamma | float = Field(default="scale", title="Gamma")
    coef0: float = Field(default=0.0, title="Independent term in kernel function")
    shrinking: bool = Field(default=False, title="Use shrinking heuristic")
    probability: bool = Field(default=False, title="Enable probability estimates")
    tol: float = Field(default=1e-3, title="Tolerance for stopping criterion")
    cache_size: float = Field(default=200.0, title="Size of the kernel cache in MB")
    max_iter: int = Field(default=-1, title="Limit on iterations within solver")
    decision_function_shape: DecisionFunctionShape = Field(
        default="ovr",
        title="Decision function of shape"
    )

    @field_validator("gamma", mode="after")
    @classmethod
    def validate_float_threshold(cls, value: Gamma | float) -> Gamma | float:
        if isinstance(value, float):
            if value <= 0:
                raise ValueError("Gamma must be greater than 0")
        return value
