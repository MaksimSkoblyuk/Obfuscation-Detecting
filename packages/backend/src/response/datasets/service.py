import logging
from io import BytesIO
from urllib.parse import quote

import pandas as pd
from minio import Minio
from pymongo.collection import Collection

from .models import DatasetCharacteristics
from ...core import (
    storage,
    service as core_service
)
from ...core.models import (
    FileContent,
    ModelDocument,
    DatasetDocument
)

logger = logging.getLogger(__name__)


def get_dataset_data(md5: str, minio: Minio, bucket_name: str) -> bytes:
    return storage.download_file(
        minio_client=minio,
        bucket_name=bucket_name,
        file_name=md5
    )


def download_dataset(
        id_: str,
        minio: Minio,
        bucket_name: str,
        collection: Collection
) -> FileContent:
    dataset = core_service.get_document_by_id(
        id_=id_,
        collection=collection,
        document_class=DatasetDocument
    )
    return FileContent(
        file=BytesIO(get_dataset_data(md5=dataset.md5, minio=minio, bucket_name=bucket_name)),
        filename=quote(dataset.name)
    )


def get_dataset_characteristics(file_data: bytes) -> DatasetCharacteristics:
    dataframe = pd.read_csv(filepath_or_buffer=BytesIO(file_data))
    return DatasetCharacteristics(
        samples=len(dataframe),
        features=len(dataframe.columns) - 1,  # -1 for target
    )


def update_related_trained_models(dataset: DatasetDocument, models_collection: Collection) -> None:
    logger.info(f"Start deleting related trained models with dataset {dataset.name}")
    related_trained_models = core_service.get_documents_by_query(
        document_class=ModelDocument,
        collection=models_collection,
        field_name="dataset_id",
        value=dataset.id
    )

    logger.info(f"Found {len(related_trained_models)} related trained models")
    if not related_trained_models:
        return

    models_info = [f"md5: {model.md5}, name: {model.name}" for model in related_trained_models]
    logger.info(f"Related models: {'; '.join(models_info)}")

    models_collection.update_many(
        filter={"_id": {"$in": [model.id for model in related_trained_models]}},
        update={"$set": {"dataset_id": None}}
    )
    logger.info("Related trained models were updated successfully")
