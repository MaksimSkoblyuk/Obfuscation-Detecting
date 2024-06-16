from typing import Type

from fastapi import APIRouter, Depends

from .service import clean_schema
from ...core import TAlgorithmParams
from .dependencies import get_algorithm_params

router = APIRouter(tags=["Settings schemas"], prefix="/algorithms")


@router.get(
    path="/{algorithm_name}/schemas",
    name="Получить json-схему параметров обучения алгоритма"
)
def get_json_schema(algorithm: Type[TAlgorithmParams] = Depends(get_algorithm_params)) -> dict:
    return clean_schema(schema=algorithm.model_json_schema())
