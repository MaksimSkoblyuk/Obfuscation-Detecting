from io import BytesIO
from bson import ObjectId
from typing import TypeVar
from datetime import datetime

from pydantic import (
    Field,
    BaseModel,
    field_validator
)

from .algorithm_params import AvailableAlgorithm


class FileContent(BaseModel):
    file: BytesIO
    filename: str

    class Config:
        arbitrary_types_allowed = True


class ObjectIdModel(BaseModel):
    id: str = Field(alias="_id")

    @field_validator("id", mode="before")
    def validate_id(cls, value: ObjectId) -> str:
        return str(value)


class DatetimeModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    created_at: str | None

    @field_validator("created_at", mode="before")
    def validate_created_at(cls, value: datetime | str | None) -> str | None:
        if value is None or isinstance(value, str):
            return value
        return value.strftime("%Y-%m-%d %H:%M:%S")


class DatasetDocument(DatetimeModel, ObjectIdModel):
    id: str = Field(description="Id of dataset", alias="_id")
    name: str = Field(description="Name of file with .csv extension")
    md5: str = Field(description="MD5 hash of file in MinIO")
    created_at: str | None = Field(description="Datetime of starting loading dataset")
    size: int = Field(description="Size of file in bytes")
    samples: int = Field(description="Total samples in dataset")
    features: int = Field(description="Total features in dataset")


class Dataset(DatetimeModel):
    name: str = Field(description="Name of file with .csv extension")
    md5: str = Field(description="MD5 hash of file in MinIO")
    created_at: str | None = Field(description="Datetime of starting loading file")
    size: int = Field(description="Size of file in bytes")
    samples: int = Field(description="Total samples in dataset")
    features: int = Field(description="Total features in dataset")


class ModelDocument(DatetimeModel, ObjectIdModel):
    id: str = Field(description="Id of trained model", alias="_id")
    dataset_id: str | None = Field(
        description="Id of preprocessed dataset used for training and testing"
    )
    name: str = Field(description="Name of file with .pkl extension")
    md5: str = Field(description="MD5 hash of file in MinIO")
    algorithm: AvailableAlgorithm = Field(description="Algorithm name")
    created_at: str | None = Field(description="Datetime of starting training")
    training_time: float | None = Field(description="Training time in seconds")
    training_data_proportion: float = Field(
        gt=0.0,
        lt=1.0,
        description="Proportion of training data"
    )
    parameters: dict = Field(description="Training parameters")
    accuracy: float = Field(ge=0.0, le=1.0, description="Accuracy metric on test data")
    precision: float = Field(ge=0.0, le=1.0, description="Precision metric on test data")
    recall: float = Field(ge=0.0, le=1.0, description="Recall metric on test data")


class Model(DatetimeModel):
    """Model for adding documents to MongoDB collection."""
    name: str = Field(description="Name of file with .pkl extension")
    dataset_id: str = Field(
        description="Id of preprocessed dataset used for training and testing"
    )
    md5: str = Field(description="MD5 hash of file in MinIO")
    algorithm: AvailableAlgorithm = Field(description="Algorithm name")
    created_at: str | None = Field(description="Datetime of starting training")
    training_time: float | None = Field(description="Training time in seconds")
    training_data_proportion: float = Field(description="Proportion of training data")
    parameters: dict = Field(description="Training parameters")
    accuracy: float = Field(description="Accuracy metric on test data")
    precision: float = Field(description="Precision metric on test data")
    recall: float = Field(description="Recall metric on test data")


class ModelDTO(BaseModel):
    """Model for transfer data between server and client."""
    id: str = Field(description="Id of trained model")
    name: str = Field(description="Name of file with .pkl extension")
    dataset_name: str | None = Field(
        description="Name of preprocessed dataset used for training and testing"
    )
    md5: str = Field(description="MD5 hash of file in MinIO")
    algorithm: AvailableAlgorithm = Field(description="Algorithm name")
    created_at: str | None = Field(description="Datetime of starting training")
    training_time: float | None = Field(description="Training time in seconds")
    training_data_proportion: float = Field(description="Proportion of training data")
    parameters: dict = Field(description="Training parameters")
    accuracy: float = Field(description="Accuracy metric on test data")
    precision: float = Field(description="Precision metric on test data")
    recall: float = Field(description="Recall metric on test data")


class ClassificationDocument(ObjectIdModel):
    id: str = Field(description="Id of classification", alias="_id")
    model_id: str = Field(description="Id of trained model used for classification")
    command: str = Field(description="PowerShell command to be classified")
    is_obfuscated: bool = Field(description="Binary classification status")


class Classification(BaseModel):
    """Model for adding documents to MongoDB collection."""
    model_id: str = Field(description="Id of trained model used for classification")
    command: str = Field(description="PowerShell command to be classified")
    is_obfuscated: bool = Field(description="Binary classification status")


class ClassificationDTO(BaseModel):
    """Model for transfer data between server and client."""
    model_name: str = Field(description="Name of file of trained model used for classification")
    model_id: str = Field(description="Id of trained model used for classification")
    command: str = Field(description="PowerShell command to be classified")
    is_obfuscated: bool = Field(description="Binary classification status")


class CommonClassificationDocument(ObjectIdModel):
    id: str = Field(description="Id of classification", alias="_id")
    command: str = Field(description="PowerShell command to be classified")
    is_obfuscated: bool = Field(description="Binary classification status")


class CommonClassification(BaseModel):
    """
    Model for adding documents to MongoDB collection and transfer data between server and client.
    """
    command: str = Field(description="PowerShell command to be classified")
    is_obfuscated: bool = Field(description="Binary classification status")


TDocument = TypeVar(
    "TDocument",
    DatasetDocument,
    ModelDocument,
    ClassificationDocument,
    CommonClassificationDocument
)
