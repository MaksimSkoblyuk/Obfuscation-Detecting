import time
import pickle
import logging
from io import BytesIO
from urllib.parse import quote

import numpy as np
import pandas as pd
from minio import Minio
from pymongo.collection import Collection
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    recall_score,
    accuracy_score,
    precision_score
)

from ...core import service as core_service
from ..classifications import service as classification_service
from ...core import (
    TModel,
    storage
)
from .models import (
    Metrics,
    TrainingStatistics
)
from ...core.models import (
    ModelDTO,
    FileContent,
    ModelDocument,
    DatasetDocument,
    ClassificationDocument
)

logger = logging.getLogger(__name__)


def get_models(model_collection: Collection, dataset_collection: Collection) -> list[ModelDTO]:
    model_documents = core_service.get_documents(
        document_class=ModelDocument,
        collection=model_collection
    )
    dataset_ids = {
        model_document.dataset_id for model_document in model_documents if model_document.dataset_id
    }
    dataset_documents = {}
    for dataset_id in dataset_ids:
        try:
            dataset_documents[dataset_id] = core_service.get_document_by_id(
                id_=dataset_id,
                collection=dataset_collection,
                document_class=DatasetDocument
            )
        except Exception:
            continue

    models = []
    for model_document in model_documents:
        if not (model_document.dataset_id and dataset_documents.get(model_document.dataset_id)):
            dataset_name = None
        else:
            dataset_name = dataset_documents[model_document.dataset_id].name
        models.append(ModelDTO(**model_document.model_dump(), dataset_name=dataset_name))
    return models


def download_model(
        id_: str,
        minio: Minio,
        bucket_name: str,
        collection: Collection
) -> FileContent:
    model = core_service.get_document_by_id(
        id_=id_,
        collection=collection,
        document_class=ModelDocument
    )
    return FileContent(
        file=BytesIO(storage.download_file(
            minio_client=minio,
            bucket_name=bucket_name,
            file_name=model.md5
        )),
        filename=quote(model.name)
    )


def split_dataset(
        dataset_data: BytesIO,
        training_data_proportion: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    dataframe = pd.read_csv(filepath_or_buffer=dataset_data)
    x, y = dataframe.iloc[:, :-1], dataframe.iloc[:, -1]
    return train_test_split(x, y, train_size=training_data_proportion, random_state=42)


def train_model(
        model: TModel,
        x_train: pd.DataFrame,
        y_train: pd.Series
) -> tuple[TModel, TrainingStatistics]:
    start_time = time.time()
    model.fit(x_train.to_numpy(), y_train.to_numpy())
    end_time = time.time()
    return model, TrainingStatistics(training_time=np.round(end_time - start_time, 6))


def calculate_metrics(model: TModel, x_test: pd.DataFrame, y_test: pd.Series) -> Metrics:
    y_pred = model.predict(x_test.to_numpy())
    return Metrics(
        accuracy=np.round(accuracy_score(y_test, y_pred), 6),
        precision=np.round(precision_score(y_test, y_pred), 6),
        recall=np.round(recall_score(y_test, y_pred), 6)
    )


def serialize_model(model: TModel) -> bytes:
    return pickle.dumps(model)


def delete_related_classifications(
        model: ModelDocument,
        classifications_collection: Collection,
        common_classifications_collection: Collection
) -> None:
    logger.info(f"Start deleting related classifications with trained model {model.name}")
    related_classifications = core_service.get_documents_by_query(
        document_class=ClassificationDocument,
        collection=classifications_collection,
        field_name="model_id",
        value=model.id
    )
    logger.info(f"Found {len(related_classifications)} related classifications")
    if not related_classifications:
        return
    core_service.delete_documents_by_query(
        collection=classifications_collection,
        field_name="model_id",
        value=model.id
    )
    logger.info(
        f"Related classifications with trained model {model.name} were deleted successfully"
    )

    logger.info("Update common classifications for commands in deleted related classifications")
    classified_commands = {classification.command for classification in related_classifications}
    core_service.delete_documents_by_multiple_query(
        collection=common_classifications_collection,
        query={"command": {"$in": list(classified_commands)}}
    )

    updated_common_classifications = []
    for command in classified_commands:
        remained_command_classifications = core_service.get_documents_by_query(
            document_class=ClassificationDocument,
            collection=classifications_collection,
            field_name="command",
            value=command
        )
        if not remained_command_classifications:
            continue
        new_common_classification = classification_service.get_common_classification(
            classifications=remained_command_classifications
        )
        updated_common_classifications.append(new_common_classification)
    if updated_common_classifications:
        common_classifications_collection.insert_many(
            [classification.model_dump() for classification in updated_common_classifications]
        )
    logger.info(
        f"Common classifications were updated successfully for commands "
        f"in deleted related classifications with trained model {model.name}"
    )
