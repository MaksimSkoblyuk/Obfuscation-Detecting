import logging
from io import BytesIO
from urllib.parse import quote

import pandas as pd
from pymongo.collection import Collection

from ...core import TModel
from ...core.enums import Extension
from .models import ClassificationResponse
from ...core import service as core_service
from ...core.models import (
    FileContent,
    ModelDocument,
    Classification,
    ClassificationDTO,
    ClassificationDocument,
    CommonClassification,
    CommonClassificationDocument
)

logger = logging.getLogger(__name__)


def get_classifications(
        common_classification_collection: Collection,
        limit: int
) -> ClassificationResponse:
    classification_documents = core_service.get_limited_documents(
        document_class=CommonClassificationDocument,
        collection=common_classification_collection,
        limit=limit
    )
    return ClassificationResponse(
        total=common_classification_collection.count_documents({}),
        classifications=[
            CommonClassification(**document.model_dump()) for document in classification_documents
        ]
    )


def get_command_classifications(
        command: str,
        classification_collection: Collection,
        model_collection: Collection
) -> list[ClassificationDTO]:
    classification_documents = core_service.get_documents_by_query(
        document_class=ClassificationDocument,
        collection=classification_collection,
        field_name="command",
        value=command
    )
    model_ids = {classification.model_id for classification in classification_documents}
    model_documents = {}
    for model_id in model_ids:
        model_documents[model_id] = core_service.get_document_by_id(
            id_=model_id,
            collection=model_collection,
            document_class=ModelDocument
        )
    classifications = []
    for classification_document in classification_documents:
        classification = ClassificationDTO(
            **classification_document.model_dump(),
            model_name=model_documents[classification_document.model_id].name
        )
        classifications.append(classification)
    return classifications


def load_commands(data: bytes, commands_column_name: str) -> tuple[pd.DataFrame, list[str]]:
    commands_dataframe = pd.read_csv(filepath_or_buffer=BytesIO(data))
    commands_list = commands_dataframe[commands_column_name]
    commands_dataframe = commands_dataframe.drop(commands_column_name, axis=1)
    return commands_dataframe, commands_list


def classify_commands(
        model: TModel,
        model_id: str,
        dataframe: pd.DataFrame,
        commands: list[str]
) -> list[Classification]:
    predictions = model.predict(dataframe.to_numpy())
    return [
        Classification(
            model_id=model_id,
            command=command,
            is_obfuscated=bool(predictions[i])
        ) for i, command in enumerate(commands)
    ]


def get_common_classification(
        classifications: list[ClassificationDocument]
) -> CommonClassification:
    if not classifications:
        raise ValueError("Got empty classifications for calculated common classification")
    command_predictions = [int(classification.is_obfuscated) for classification in classifications]
    return CommonClassification(
        command=classifications[0].command,
        is_obfuscated=(sum(command_predictions) / len(command_predictions)) >= 0.5
    )


def download_classifications(
        filename: str,
        model_collection: Collection,
        classification_collection: Collection,
        common_classification_collection: Collection,
        app_url: str
) -> FileContent:
    filename = core_service.render_filename(raw_filename=filename, expected_extension=Extension.CSV)
    logger.info(f"Start downloading classifications to {filename}")

    logger.info(f"Find classified commands")
    common_classifications = core_service.get_documents(
        document_class=CommonClassificationDocument,
        collection=common_classification_collection
    )
    logger.info(f"Got {len(common_classifications)} classified commands")

    df_data = []
    for common_classification in common_classifications:
        command_data = {}
        command_classifications = get_command_classifications(
            command=common_classification.command,
            classification_collection=classification_collection,
            model_collection=model_collection
        )
        logger.info(
            f"Got {len(command_classifications)} classifications "
            f"for command: {common_classification.command}"
        )
        command_data['command'] = common_classification.command
        command_data["is_obfuscated"] = int(common_classification.is_obfuscated)

        models = {}
        for command_classification in command_classifications:
            if not models.get(command_classification.model_name):
                models[command_classification.model_name] = core_service.get_documents_by_query(
                    document_class=ModelDocument,
                    collection=model_collection,
                    field_name="name",
                    value=command_classification.model_name
                )[0]
            command_data[command_classification.model_name] = int(
                command_classification.is_obfuscated
            )
            command_data[f"{command_classification.model_name}_download_link"] = (
                f"{app_url}/api/v1/models/{models[command_classification.model_name].id}/download/"
            )
        df_data.append(command_data)
    return FileContent(
        file=BytesIO(pd.DataFrame(data=df_data).to_csv(index=False).encode('utf-8')),
        filename=quote(filename)
    )
