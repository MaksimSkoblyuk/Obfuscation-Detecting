from typing import TypeVar

from .xgboost import XGBoostParams
from .lightgbm import LightGBMParams
from .catboost import CatBoostParams
from .base import BaseAlgorithmParams
from .enums import AvailableAlgorithm
from .support_vector_machines import SVCParams
from .gaussian_naive_bayes import GaussianNBParams
from .random_forest import RandomForestClassifierParams
from .decision_tree import DecisionTreeClassifierParams
from .k_nearest_neighbors import KNearestNeighborsClassifierParams
from .logistic_regression import LogisticRegressionClassifierParams
from .multinomial_naive_bayes import MultinomialNBClassifierParams


available_algorithms_params = {
    AvailableAlgorithm.GAUSSIAN_NAIVE_BAYES: GaussianNBParams,
    AvailableAlgorithm.MULTINOMIAL_NAIVE_BAYES: MultinomialNBClassifierParams,
    AvailableAlgorithm.SUPPORT_VECTOR_MACHINES: SVCParams,
    AvailableAlgorithm.K_NEAREST_NEIGHBORS: KNearestNeighborsClassifierParams,
    AvailableAlgorithm.LOGISTIC_REGRESSION: LogisticRegressionClassifierParams,
    AvailableAlgorithm.DECISION_TREE: DecisionTreeClassifierParams,
    AvailableAlgorithm.RANDOM_FOREST: RandomForestClassifierParams,
    AvailableAlgorithm.XGBOOST_CLASSIFIER: XGBoostParams,
    AvailableAlgorithm.CATBOOST_CLASSIFIER: CatBoostParams,
    AvailableAlgorithm.LIGHTGBM_CLASSIFIER: LightGBMParams
}

TAlgorithmParams = TypeVar("TAlgorithmParams", bound=BaseAlgorithmParams)
