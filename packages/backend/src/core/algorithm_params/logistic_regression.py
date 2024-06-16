from typing import Literal

from pydantic import Field

from .base import BaseAlgorithmParams

Penalty = Literal["l1", "l2", "elasticnet"]
MultiClass = Literal["auto", "ovr", "multinomial"]
Solver = Literal["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"]


class LogisticRegressionClassifierParams(BaseAlgorithmParams):
    class Config:
        title = "Logistic Regression"

    penalty: Penalty | None = Field(default="l2", title="Penalty")
    dual: bool = Field(default=False, title="Dual formulation")
    tol: float = Field(default=1e-4, title="Tolerance for stopping criteria")
    C: float = Field(default=1.0, title="Inverse of regularization strength")
    fit_intercept: bool = Field(default=False, title="Fit intercept")
    intercept_scaling: float = Field(default=1.0, title="Intercept scaling")
    solver: Solver = Field(default="lbfgs", title="Algorithm")
    max_iter: int = Field(default=100, title="Maximum number of iterations")
    warm_start: bool = Field(default=False, title="Reuse the solution of previous fits")
    l1_ratio: float | None = Field(default=None, title="L1 penalty ratio")
