import json
import pathlib
import logging

from minio import Minio
from pymongo.collection import Collection

from ..core import (
    storage,
    global_config,
    service as core_service
)
from ..core.models import (
    Model,
    Dataset,
    ModelDTO,
    DatasetDocument
)

logger = logging.getLogger(__name__)


def load_statistics(dir_path: str, statistics_dir_path: str) -> dict[str, dict]:
    files_directory = pathlib.Path(dir_path)
    statistics_directory = pathlib.Path(statistics_dir_path)
    statistics = {}
    for file in files_directory.iterdir():
        logger.info(f"Processing file {file.name!r}")
        if not file.is_file():
            logger.info(f"Skipping non-file {file.name!r}")
            continue
        statistics_file_path = statistics_directory / f"{file.stem}.json"
        logger.info(f"Loadings statistics for {file.name!r}")
        statistics[file.name] = _get_statistics(file_path=statistics_file_path)
    return statistics


def load_datasets(
        dataset_dir_path: str,
        statistics_dir_path: str,
        minio_client: Minio,
        bucket_name: str,
        dataset_collection: Collection
) -> None:
    dataset_files_directory = pathlib.Path(dataset_dir_path)
    statistics = load_statistics(dir_path=dataset_dir_path, statistics_dir_path=statistics_dir_path)
    datasets = []
    for file in dataset_files_directory.iterdir():
        if not statistics.get(file.name):
            continue
        logger.info(f"Loading dataset {file.name!r}")
        dataset = Dataset(**statistics[file.name])
        datasets.append(dataset)
        storage.upload_file(
            minio_client=minio_client,
            bucket_name=bucket_name,
            file_name=dataset.md5,
            file_data=file.read_bytes(),
            part_size=global_config.minio.part_size
        )
    logger.info(f"Adding datasets {', '.join(dataset.name for dataset in datasets)!r} to database")
    dataset_collection.insert_many([dataset.model_dump() for dataset in datasets])


def load_models(
        model_dir_path: str,
        statistics_dir_path: str,
        minio_client: Minio,
        bucket_name: str,
        model_collection: Collection,
        dataset_collection: Collection
) -> None:
    model_files_directory = pathlib.Path(model_dir_path)
    statistics = load_statistics(dir_path=model_dir_path, statistics_dir_path=statistics_dir_path)
    models = []
    for file in model_files_directory.iterdir():
        if not statistics.get(file.name):
            continue
        logger.info(f"Loading trained model {file.name!r}")
        model_dto = ModelDTO(**statistics[file.name], id="")
        dataset = core_service.get_documents_by_query(
            document_class=DatasetDocument,
            collection=dataset_collection,
            field_name="name",
            value=model_dto.dataset_name
        )[0]  # TODO: на каждый файл модели делается запрос в бд, лучше так не делать
        model = Model(**model_dto.model_dump(), dataset_id=dataset.id)
        models.append(model)
        storage.upload_file(
            minio_client=minio_client,
            bucket_name=bucket_name,
            file_name=model.md5,
            file_data=file.read_bytes(),
            part_size=global_config.minio.part_size
        )
    logger.info(f"Adding trained models {', '.join(model.name for model in models)} to database")
    model_collection.insert_many([model.model_dump() for model in models])


def _get_statistics(file_path: pathlib.Path) -> dict:
    if not file_path.exists():
        raise ValueError(f"File {file_path!r} is not found")
    with open(file_path) as file:
        return json.load(file)
