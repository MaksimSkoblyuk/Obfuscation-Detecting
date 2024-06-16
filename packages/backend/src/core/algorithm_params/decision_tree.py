from typing import Literal

from pydantic import Field

from .base import BaseAlgorithmParams

Splitter = Literal["best", "random"]
MaxFeatures = Literal["sqrt", "log2"]
Criterion = Literal["gini", "entropy", "log_loss"]


class DecisionTreeClassifierParams(BaseAlgorithmParams):
    class Config:
        title = "Decision Tree"

    criterion: Criterion = Field(default="gini", title="Criterion")
    splitter: Splitter = Field(default="best", title="Splitter")
    max_depth: int | None = Field(default=None, title="Maximum depth")
    min_samples_split: int | float = Field(default=2, title="Minimum samples in an internal node")
    min_samples_leaf: int | float = Field(default=1, title="Minimum samples in a leaf node")
    min_weight_fraction_leaf: float = Field(
        default=0.0,
        title="Minimum weighted fraction in a leaf node"
    )
    max_features: MaxFeatures | int | float | None = Field(
        default=None,
        title="Maximum number of features for the best split"
    )
    max_leaf_nodes: int | None = Field(default=None, title="Maximum number of leaf nodes")
    min_impurity_decrease: float = Field(default=0.0, title="Minimum impurity decrease")
    ccp_alpha: float = Field(default=0.0, title="Complexity parameter for pruning", ge=0.0)
