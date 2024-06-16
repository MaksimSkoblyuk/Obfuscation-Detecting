import logging

from celery import shared_task

from . import service
from ..core import (
    minio,
    mongo_collection_models,
    mongo_collection_datasets
)

logger = logging.getLogger(__name__)


@shared_task()
def load_preprocessed_datasets(
        dataset_dir_path: str,
        statistics_dir_path: str,
        minio_bucket_name: str
) -> None:
    service.load_datasets(
        dataset_dir_path=dataset_dir_path,
        statistics_dir_path=statistics_dir_path,
        minio_client=minio,
        bucket_name=minio_bucket_name,
        dataset_collection=mongo_collection_datasets
    )


@shared_task()
def load_trained_models(
        model_dir_path: str,
        statistics_dir_path: str,
        minio_bucket_name: str
) -> None:
    service.load_models(
        model_dir_path=model_dir_path,
        statistics_dir_path=statistics_dir_path,
        minio_client=minio,
        bucket_name=minio_bucket_name,
        model_collection=mongo_collection_models,
        dataset_collection=mongo_collection_datasets
    )
