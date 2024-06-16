from typing import Literal

from pydantic import Field

from .base import BaseAlgorithmParams

MaxFeatures = Literal["sqrt", "log2"]
Criterion = Literal["gini", "entropy", "log_loss"]


class RandomForestClassifierParams(BaseAlgorithmParams):
    class Config:
        title = "Random Forest"

    n_estimators: int = Field(default=100, title="Number of trees", ge=1)
    criterion: Criterion = Field(default="gini", title="Criterion")
    max_depth: int | None = Field(default=None, title="Maximum depth of tree")
    min_samples_split: int | float = Field(default=2, title="Minimum samples in an internal node")
    min_samples_leaf: int | float = Field(default=1, title="Minimum samples in a leaf node")
    min_weight_fraction_leaf: float = Field(
        default=0.0,
        title="Minimum weighted fraction in a leaf node"
    )
    max_features: MaxFeatures | int | float | None = Field(
        default="sqrt",
        title="Maximum number of features for the best split"
    )
    max_leaf_nodes: int | None = Field(default=None, title="Maximum number of leaf nodes")
    min_impurity_decrease: float = Field(default=0.0, title="Minimum impurity decrease")
    bootstrap: bool = Field(default=False, title="Build trees with bootstrapped samples")
    oob_score: bool = Field(default=False, title="Use out-of-bag samples to estimate")
    warm_start: bool = Field(default=False, title="Reuse the solution of previous fits")
    ccp_alpha: float = Field(default=0.0, title="Complexity parameter for pruning", ge=0.0)
    max_samples: int | float | None = Field(
        default=None,
        title="Maximum samples for training each base estimator"
    )
