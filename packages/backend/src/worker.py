import logging

from celery import Celery, chain
from celery.signals import worker_ready

from . import initializer
from .core import global_config

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)-8s %(name)s:%(funcName)s:%(lineno)d - %(message)s",
    handlers=[
        logging.FileHandler("tip_sandbox_logs.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

celery = Celery(
    "app",
    broker=f"redis://{global_config.redis.host}:{global_config.redis.port}",
    backend=f"redis://{global_config.redis.host}:{global_config.redis.port}",
    include=[
        "src.initializer.tasks",
        "src.response.datasets.tasks",
        "src.response.models.tasks",
        "src.response.classifications.tasks"
    ],
    worker_hijack_root_logger=False
)


@worker_ready.connect
def on_startup(**kwargs) -> None:
    celery.control.purge()
    logger.info(f"Start tasks for initial loading dataset and trained models")
    chain(
        initializer.tasks.load_preprocessed_datasets.s(
            dataset_dir_path=global_config.initial_files.preprocessed_datasets_dir_path,
            statistics_dir_path=global_config.initial_files.statistics_dir_path,
            minio_bucket_name=global_config.minio.preprocessed_datasets_bucket_name
        ),
        initializer.tasks.load_trained_models.si(
            model_dir_path=global_config.initial_files.trained_models_dir_path,
            statistics_dir_path=global_config.initial_files.statistics_dir_path,
            minio_bucket_name=global_config.minio.trained_models_bucket_name
        )
    )()

