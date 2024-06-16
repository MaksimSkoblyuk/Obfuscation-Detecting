import logging
import datetime

from celery import Task
from redis import Redis

from . import service
from ... import worker
from ...core.tasks import DeletingTask
from ...core import (
    redis,
    minio,
    storage,
    global_config,
    LockException,
    mongo_collection_models,
    service as core_service,
    mongo_collection_datasets
)
from ...core.models import (
    Dataset,
    DatasetDocument
)

logger = logging.getLogger(__name__)


class UploadingDatasetTask(Task):
    redis: Redis
    locked_task_expiration: int

    def before_start(self, task_id, args, kwargs) -> None:
        logger.info(f"Start calculating md5 hash for dataset {kwargs['filename']!r}")
        md5 = core_service.calculate_md5(
            file_data=kwargs["file_data"],
            chunk_size=global_config.minio.part_size
        )
        logger.info(f"Md5 hash for dataset {kwargs['filename']!r}: {md5!r}")
        status = self.redis.set(md5, 'lock', ex=self.locked_task_expiration, nx=True)
        if not status:
            logger.error(f"Dataset {kwargs['filename']!r} has already locked by another task")
            raise LockException()
        setattr(self, 'md5', md5)

    def on_success(self, retval, task_id, args, kwargs) -> None:
        self.redis.delete(self.md5)

    def on_failure(self, exc, task_id, args, kwargs, einfo) -> None:
        if isinstance(exc, LockException):
            return
        self.redis.delete(self.md5)


@worker.celery.task(
    base=UploadingDatasetTask,
    bind=True,
    redis=redis,
    locked_task_expiration=global_config.locked_task_expiration
)
def upload_dataset(self, file_data: bytes, filename: str) -> None:
    logger.info(f"Start uploading dataset {filename!r}")
    if core_service.check_existing_file(md5=self.md5, collection=mongo_collection_datasets):
        raise FileExistsError(f"Dataset with md5 {self.md5} has already existed in database")
    logger.info(f"Start calculating characteristics for dataset {filename!r}")
    dataset_characteristics = service.get_dataset_characteristics(file_data=file_data)
    logger.info(f"Characteristics for dataset {filename!r}: {dataset_characteristics}")
    dataset = Dataset(
        name=filename,
        md5=self.md5,
        created_at=datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=3))),
        size=len(file_data),
        samples=dataset_characteristics.samples,
        features=dataset_characteristics.features
    )
    storage.upload_file(
        minio_client=minio,
        bucket_name=global_config.minio.preprocessed_datasets_bucket_name,
        file_name=self.md5,
        file_data=file_data,
        part_size=global_config.minio.part_size
    )
    logger.info(f"Add dataset {filename!r} to database")
    mongo_collection_datasets.insert_one(dataset.model_dump())
    logger.info(f"Dataset {filename!r} was uploaded to storage and added to database successfully")


@worker.celery.task(
    base=DeletingTask,
    redis=redis,
    locked_task_expiration=global_config.locked_task_expiration,
    countdown=global_config.locked_task_countdown,
    max_retries=global_config.locked_task_max_retries
)
def delete_dataset(id_: str) -> None:
    logger.info(
        f"Start deleting dataset with id {id_!r} and related entities from database and storage"
    )
    dataset = core_service.get_document_by_id(
        id_=id_,
        document_class=DatasetDocument,
        collection=mongo_collection_datasets
    )
    logger.info(
        f"Found dataset with id {id_!r} in database: "
        f"filename {dataset.name!r}, md5 hash {dataset.md5!r}"
    )
    md5 = dataset.md5
    core_service.delete_file(id_=id_, collection=mongo_collection_datasets)
    logger.info(f"Dataset with id {id_!r} was deleted from database successfully")
    storage.delete_file(
        minio_client=minio,
        bucket_name=global_config.minio.preprocessed_datasets_bucket_name,
        file_name=md5
    )
    logger.info(f"Dataset with id {id_!r} was deleted from storage and database successfully")
    service.update_related_trained_models(
        dataset=dataset,
        models_collection=mongo_collection_models
    )
