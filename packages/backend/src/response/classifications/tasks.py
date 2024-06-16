import pickle
import logging

from celery import Task
from redis import Redis

from . import service
from ... import worker
from ...core import (
    redis,
    minio,
    storage,
    global_config,
    LockException,
    mongo_collection_models,
    service as core_service,
    mongo_collection_classifications,
    mongo_collection_common_classifications
)
from ...core.models import (
    ModelDocument,
    ClassificationDocument
)

logger = logging.getLogger(__name__)


class ClassificationTask(Task):
    redis: Redis
    locked_task_expiration: int

    def before_start(self, task_id, args, kwargs) -> None:
        status = self.redis.set("classification", "lock", ex=self.locked_task_expiration, nx=True)
        if not status:
            raise LockException()
        logger.info("Cleaning up old classifications in database")
        mongo_collection_classifications.delete_many({})
        mongo_collection_common_classifications.delete_many({})

    def on_success(self, retval, task_id, args, kwargs) -> None:
        self.redis.delete("classification")

    def on_failure(self, exc, task_id, args, kwargs, einfo) -> None:
        if isinstance(exc, LockException):
            return
        self.redis.delete("classification")


@worker.celery.task(
    base=ClassificationTask,
    redis=redis,
    locked_task_expiration=global_config.locked_task_expiration
)
def classify_commands(commands_data: bytes, models_ids: list[str]) -> None:
    logger.info(f"Start classifying commands")

    logger.info(f"Loading commands into Dataframe")
    commands_dataframe, commands = service.load_commands(
        data=commands_data,
        commands_column_name=global_config.commands_column_name
    )
    logger.info(f"Got {len(commands)} commands")

    logger.info(f"Find models with ids: {models_ids}")
    model_documents = [
        core_service.get_document_by_id(
            id_=id_,
            collection=mongo_collection_models,
            document_class=ModelDocument
        ) for id_ in models_ids
    ]
    logger.info(f"Got trained models: md5 hashes={[model.md5 for model in model_documents]}")

    for model_document in model_documents:
        logger.info(
            f"Classify commands with model: name={model_document.name}, md5={model_document.md5}"
        )
        model = pickle.loads(storage.download_file(
            minio_client=minio,
            bucket_name=global_config.minio.trained_models_bucket_name,
            file_name=model_document.md5
        ))
        classifications = service.classify_commands(
            model=model,
            model_id=model_document.id,
            dataframe=commands_dataframe,
            commands=commands
        )
        mongo_collection_classifications.insert_many(
            [classification.model_dump() for classification in classifications]
        )
        logger.info(
            f"Classifications with model name={model_document.name}, md5={model_document.md5} "
            f"were saved in database"
        )

    common_classifications = []
    for command in set(commands):
        command_classifications = core_service.get_documents_by_query(
            document_class=ClassificationDocument,
            collection=mongo_collection_classifications,
            field_name="command",
            value=command
        )
        common_classification = service.get_common_classification(
            classifications=command_classifications
        )
        common_classifications.append(common_classification)
    mongo_collection_common_classifications.insert_many(
        [classification.model_dump() for classification in common_classifications]
    )
    logger.info(f"Common classifications were saved in database")
