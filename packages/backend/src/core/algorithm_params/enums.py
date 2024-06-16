from enum import Enum


class AvailableAlgorithm(str, Enum):
    GAUSSIAN_NAIVE_BAYES = "GaussianNaiveBayes"
    MULTINOMIAL_NAIVE_BAYES = "MultinomialNaiveBayes"
    SUPPORT_VECTOR_MACHINES = "SupportVectorMachines"
    K_NEAREST_NEIGHBORS = "KNearestNeighbors"
    LOGISTIC_REGRESSION = "LogisticRegression"
    DECISION_TREE = "DecisionTree"
    RANDOM_FOREST = "RandomForest"
    XGBOOST_CLASSIFIER = "XGBoostClassifier"
    CATBOOST_CLASSIFIER = "CatBoostClassifier"
    LIGHTGBM_CLASSIFIER = "LightGBMClassifier"
