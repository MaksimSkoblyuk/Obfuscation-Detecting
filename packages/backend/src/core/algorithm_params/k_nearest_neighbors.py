from typing import Literal

from pydantic import Field

from .base import BaseAlgorithmParams

WeightFunction = Literal["uniform", "distance"]
Algorithm = Literal["auto", "ball_tree", "kd_tree", "brute"]
Metric = Literal[
    "p",
    "l1",
    "l2",
    "dice",
    "yule",
    "cosine",
    "hamming",
    "jaccard",
    "canberra",
    "infinity",
    "chebyshev",
    "minkowski",
    "cityblock",
    "euclidean",
    "haversine",
    "manhattan",
    "russellrao",
    "seuclidean",
    "braycurtis",
    "sokalsneath",
    "sqeuclidean",
    "mahalanobis",
    "precomputed",
    "correlation",
    "nan_euclidean",
    "sokalmichener",
    "rogerstanimoto"
]


class KNearestNeighborsClassifierParams(BaseAlgorithmParams):
    class Config:
        title = "K-Nearest Neighbors"

    n_neighbors: int = Field(default=5, title="Number of neighbors")
    weights: WeightFunction = Field(default="uniform", title="Weight function")
    algorithm: Algorithm = Field(default="auto", title="Algorithm")
    leaf_size: int = Field(default=30, title="Leaf size")
    p: float = Field(default=2.0, title="Power parameter", gt=0.0)
    metric: Metric = Field(default="minkowski", title="Metric")
