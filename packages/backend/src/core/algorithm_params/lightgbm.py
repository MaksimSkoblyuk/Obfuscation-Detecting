from typing import Literal

from pydantic import Field

from .base import BaseAlgorithmParams

ImportanceType = Literal['split', 'gain']
BoostingType = Literal['gbdt', 'dart', 'rf']


class LightGBMParams(BaseAlgorithmParams):
    class Config:
        title = "LightGBM"

    boosting_type: BoostingType = Field(default='gbdt', title='Boosting type')
    num_leaves: int = Field(default=31, title='Maximum tree leaves')
    max_depth: int = Field(default=-1, title='Maximum tree depth')
    learning_rate: float = Field(default=0.1, title='Learning rate')
    n_estimators: int = Field(default=100, title='Number of boosted trees to fit')
    subsample_for_bin: int = Field(default=200000, title='Number of samples for constructing bins')
    min_split_gain: float = Field(
        default=0.0,
        title='Minimum loss reduction to make a further partition'
    )
    min_child_weight: float = Field(
        default=1e-3,
        title='Minimum sum of instance weight needed in a leaf'
    )
    min_child_samples: int = Field(default=20, title='Minimum number of data needed in a leaf')
    subsample: float = Field(default=1.0, title='Subsample ratio of the training instance')
    subsample_freq: int = Field(default=0, title='Frequency of subsample')
    colsample_bytree: float = Field(
        default=1.0,
        title='Subsample ratio of columns when constructing each tree'
    )
    reg_alpha: float = Field(default=0.0, title='L1 regularization term on weights')
    reg_lambda: float = Field(default=0.0, title='L2 regularization term on weights')
    importance_type: ImportanceType = Field(default='split', title='Feature importance type')
