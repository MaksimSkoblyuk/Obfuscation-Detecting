from pydantic import BaseModel, Field

from ...core import AvailableAlgorithm


class TrainingParams(BaseModel):
    filename: str
    dataset_id: str
    training_params: dict
    algorithm_name: AvailableAlgorithm
    training_data_proportion: float = Field(gt=0, lt=1)


class TrainingStatistics(BaseModel):
    training_time: float


class Metrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
