from typing import Annotated

from fastapi import status
from fastapi_pagination.utils import disable_installed_extensions_check
from fastapi import (
    Form,
    Query,
    APIRouter,
    UploadFile
)
from fastapi.responses import (
    JSONResponse,
    StreamingResponse
)

from . import tasks
from . import service
from ...core.enums import Extension
from .models import ClassificationResponse
from ...core import service as core_service
from ...core.models import ClassificationDTO
from ...core import (
    global_config,
    mongo_collection_models,
    mongo_collection_classifications,
    mongo_collection_common_classifications
)

router = APIRouter(tags=["Classifications"], prefix="/classifications")
disable_installed_extensions_check()


@router.post(path="/", name="Классифицировать предобработанные команды")
def classify_commands(
        models_ids: Annotated[list[str], Form()],
        commands: UploadFile
) -> JSONResponse:
    if not core_service.is_right_file_extension(commands.filename, Extension.CSV):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "status": f"Wrong file {commands.filename} extension. "
                          f"Expected {Extension.CSV.value}"
            }
        )
    tasks.classify_commands.apply_async(
        kwargs={
            "commands_data": commands.file.read(),
            "models_ids": models_ids[0].split(",")
        }
    )
    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={"status": "Commands classification is started"}
    )


@router.get(
    path="/commands",
    name="Получить результаты классификации команды на разных моделях",
    response_model=list[ClassificationDTO]
)
def get_command_classifications(command: str = Query()) -> list[ClassificationDTO]:
    return service.get_command_classifications(
        command=command,
        classification_collection=mongo_collection_classifications,
        model_collection=mongo_collection_models
    )


@router.get(
    path="/",
    name="Получить общие результаты классификации команд",
    response_model=ClassificationResponse
)
def get_classifications(limit: int) -> ClassificationResponse:
    return service.get_classifications(
        common_classification_collection=mongo_collection_common_classifications,
        limit=limit
    )


@router.get(path="/download", name="Скачать подробные результаты классификации")
def download_classifications(filename: str) -> StreamingResponse:
    file = service.download_classifications(
        filename=filename,
        model_collection=mongo_collection_models,
        classification_collection=mongo_collection_classifications,
        common_classification_collection=mongo_collection_common_classifications,
        app_url=global_config.app.url
    )
    return StreamingResponse(
        content=file.file,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={file.filename}"}
    )
