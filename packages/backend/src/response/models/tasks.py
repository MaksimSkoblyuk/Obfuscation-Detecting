import datetime
import logging
from io import BytesIO

from celery import Task
from redis import Redis

from . import service
from ... import worker
from ...core.enums import Extension
from ...core.tasks import DeletingTask
from ...core import (
    redis,
    minio,
    storage,
    LockException,
    global_config,
    AvailableAlgorithm,
    service as core_service,
    mongo_collection_models,
    mongo_collection_datasets,
    available_algorithms_params,
    ALGORITHM_CLASS_BY_NAME_MAPPING,
    mongo_collection_classifications,
    mongo_collection_common_classifications
)
from ...core.models import (
    Model,
    ModelDocument,
    DatasetDocument
)

logger = logging.getLogger(__name__)


class TrainingModelTask(Task):
    redis: Redis
    locked_task_expiration: int

    def before_start(self, task_id, args, kwargs) -> None:
        filename = core_service.render_filename(kwargs["filename"], Extension.PKL)
        if core_service.get_documents_by_query(
            document_class=ModelDocument,
            collection=mongo_collection_models,
            field_name="name",
            value=filename
        ):
            logger.error(f"Model with name {kwargs['filename']!r} has already existed in database")
            raise ValueError(f"Model with name {filename!r} has already existed in database")

        status = self.redis.set(kwargs["filename"], 'lock', ex=self.locked_task_expiration, nx=True)
        if not status:
            logger.error(
                f"Model with name {kwargs['filename']!r} has already locked by another task"
            )
            raise LockException()

    def on_success(self, retval, task_id, args, kwargs) -> None:
        redis.delete(kwargs["filename"])

    def on_failure(self, exc, task_id, args, kwargs, einfo) -> None:
        if isinstance(exc, LockException):
            return
        redis.delete(kwargs["filename"])


@worker.celery.task(
    base=TrainingModelTask,
    redis=redis,
    locked_task_expiration=global_config.locked_task_expiration,
)
def train_model(
        filename: str,
        dataset_id: str,
        training_params: dict,
        algorithm_name: AvailableAlgorithm,
        training_data_proportion: float
) -> None:
    logger.info(f"Start training model of algorithm {algorithm_name!r} for dataset {dataset_id!r}")
    logger.info(f"Start searching for dataset with id {dataset_id!r} in database")
    dataset = core_service.get_document_by_id(
        id_=dataset_id,
        collection=mongo_collection_datasets,
        document_class=DatasetDocument
    )
    filename = core_service.render_filename(raw_filename=filename, expected_extension=Extension.PKL)
    logger.info(
        f"Found dataset with id {dataset_id!r} in database: "
        f"md5 {dataset.md5!r}, filename {dataset.name!r}"
    )
    dataset_data = storage.download_file(
        minio_client=minio,
        bucket_name=global_config.minio.preprocessed_datasets_bucket_name,
        file_name=dataset.md5
    )
    logger.info(
        f"Split dataset for training and testing with proportion {training_data_proportion}"
    )
    x_train, x_test, y_train, y_test = service.split_dataset(
        dataset_data=BytesIO(dataset_data),
        training_data_proportion=training_data_proportion
    )
    algorithm_model = ALGORITHM_CLASS_BY_NAME_MAPPING[algorithm_name](
        **available_algorithms_params[algorithm_name](**training_params).model_dump()
    )
    logger.info(f"Train model of algorithm {algorithm_name!r}")
    trained_model, training_statistics = service.train_model(
        model=algorithm_model,
        x_train=x_train,
        y_train=y_train
    )
    logger.info(f"Serialize model of algorithm {algorithm_name!r}")
    serialized_model = service.serialize_model(model=trained_model)
    md5 = core_service.calculate_md5(
        file_data=serialized_model,
        chunk_size=global_config.minio.part_size
    )
    logger.info(f"Got md5 {md5!r} for serialized model")
    if core_service.check_existing_file(md5=md5, collection=mongo_collection_models):
        logger.error(f"Model with md5 {md5!r} has already existed in database")
        raise FileExistsError(f"Model with md5 {md5!r} has already existed in database")
    logger.info(f"Calculate metrics for model with md5 {md5!r}")
    metrics = service.calculate_metrics(model=trained_model, x_test=x_test, y_test=y_test)
    model = Model(
        name=filename,
        dataset_id=dataset.id,
        md5=md5,
        algorithm=algorithm_name,
        created_at=datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=3))),
        training_time=training_statistics.training_time,
        training_data_proportion=training_data_proportion,
        parameters=training_params,
        accuracy=metrics.accuracy,
        precision=metrics.precision,
        recall=metrics.recall
    )
    storage.upload_file(
        minio_client=minio,
        bucket_name=global_config.minio.trained_models_bucket_name,
        file_name=md5,
        file_data=serialized_model,
        part_size=global_config.minio.part_size
    )
    logger.info(f"Add model with md5 {md5!r} to database")
    mongo_collection_models.insert_one(model.model_dump())
    logger.info(
        f"Model with md5 {md5!r} and filename {filename!r} "
        f"was uploaded to storage and saved to database successfully"
    )


@worker.celery.task(
    base=DeletingTask,
    redis=redis,
    locked_task_expiration=global_config.locked_task_expiration,
    countdown=global_config.locked_task_countdown,
    max_retries=global_config.locked_task_max_retries
)
def delete_model(id_: str) -> None:
    logger.info(f"Start deleting trained model with id {id_!r}")
    model = core_service.get_document_by_id(
        id_=id_,
        collection=mongo_collection_models,
        document_class=ModelDocument
    )
    logger.info(
        f"Found model with id {id_!r} in database: filename {model.name!r}, md5 hash {model.md5!r}"
    )
    md5 = model.md5
    core_service.delete_file(id_=id_, collection=mongo_collection_models)
    logger.info(f"Model with id {id_!r} was deleted from database successfully")
    storage.delete_file(
        minio_client=minio,
        bucket_name=global_config.minio.trained_models_bucket_name,
        file_name=md5
    )
    logger.info(f"Model with id {id_!r} was deleted from storage and database successfully")
    service.delete_related_classifications(
        model=model,
        classifications_collection=mongo_collection_classifications,
        common_classifications_collection=mongo_collection_common_classifications
    )
