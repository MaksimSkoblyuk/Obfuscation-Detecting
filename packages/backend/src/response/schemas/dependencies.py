from typing import Type

from fastapi import Path

from ...core import (
    TAlgorithmParams,
    AvailableAlgorithm,
    available_algorithms_params
)


def get_algorithm_params(algorithm_name: AvailableAlgorithm = Path()) -> Type[TAlgorithmParams]:
    return available_algorithms_params[algorithm_name]
