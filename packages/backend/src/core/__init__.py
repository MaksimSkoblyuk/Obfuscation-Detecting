from typing import TypeVar

from minio import Minio
from redis import Redis
from sklearn.svm import SVC
from pymongo import MongoClient
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm.sklearn import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import (
    GaussianNB,
    MultinomialNB
)

from . import storage
from .config import Config
from .exceptions import LockException
from .algorithm_params import (
    TAlgorithmParams,
    AvailableAlgorithm,
    available_algorithms_params
)

global_config = Config()

minio = Minio(
    endpoint=global_config.minio.url,
    access_key=global_config.minio.access_key,
    secret_key=global_config.minio.secret_key,
    secure=False
)

mongodb_client = MongoClient(global_config.mongo.url)
mongo_database = mongodb_client[global_config.mongo.database]
mongo_collection_models = mongo_database[global_config.mongo.models_collection]
mongo_collection_datasets = mongo_database[global_config.mongo.datasets_collection]
mongo_collection_classifications = mongo_database[global_config.mongo.classifications_collection]
mongo_collection_common_classifications = mongo_database[
    global_config.mongo.common_classifications_collection
]

ALGORITHM_CLASS_BY_NAME_MAPPING = {
    AvailableAlgorithm.MULTINOMIAL_NAIVE_BAYES: MultinomialNB,
    AvailableAlgorithm.GAUSSIAN_NAIVE_BAYES: GaussianNB,
    AvailableAlgorithm.SUPPORT_VECTOR_MACHINES: SVC,
    AvailableAlgorithm.K_NEAREST_NEIGHBORS: KNeighborsClassifier,
    AvailableAlgorithm.LOGISTIC_REGRESSION: LogisticRegression,
    AvailableAlgorithm.DECISION_TREE: DecisionTreeClassifier,
    AvailableAlgorithm.RANDOM_FOREST: RandomForestClassifier,
    AvailableAlgorithm.XGBOOST_CLASSIFIER: XGBClassifier,
    AvailableAlgorithm.CATBOOST_CLASSIFIER: CatBoostClassifier,
    AvailableAlgorithm.LIGHTGBM_CLASSIFIER: LGBMClassifier
}

redis = Redis(host=global_config.redis.host, port=global_config.redis.port)

TModel = TypeVar(
    "TModel",
    MultinomialNB,
    GaussianNB,
    SVC,
    KNeighborsClassifier,
    LogisticRegression,
    DecisionTreeClassifier,
    RandomForestClassifier,
    XGBClassifier,
    CatBoostClassifier,
    LGBMClassifier
)
