from fastapi import (
    Path,
    Query,
    status,
    APIRouter
)
from fastapi.responses import (
    JSONResponse,
    StreamingResponse
)
from fastapi_pagination.utils import disable_installed_extensions_check
from fastapi_pagination import (
    Page,
    paginate
)

from . import tasks
from . import service
from ...core.models import ModelDTO
from .models import TrainingParams
from ...core import (
    minio,
    global_config,
    mongo_collection_models,
    mongo_collection_datasets
)

router = APIRouter(prefix="/models", tags=["Models"])
disable_installed_extensions_check()


@router.post(path="/", name="Обучить модель")
def train_model(params: TrainingParams) -> JSONResponse:
    tasks.train_model.apply_async(
        kwargs={
            "filename": params.filename,
            "dataset_id": params.dataset_id,
            "training_params": params.training_params,
            "algorithm_name": params.algorithm_name,
            "training_data_proportion": params.training_data_proportion
        }
    )
    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={
            "status": f"Model training with algorithm {params.algorithm_name} "
                      f"and params {params.training_params} is started"
        }
    )


@router.get(path="/", name="Получить модели", response_model=Page[ModelDTO])
def get_models() -> Page[ModelDTO]:
    return paginate(
        service.get_models(
            model_collection=mongo_collection_models,
            dataset_collection=mongo_collection_datasets
        )
    )


@router.get(path="/{id}/download", name="Скачать модель")
def download_model(id_: str = Path(alias="id")) -> StreamingResponse:
    file = service.download_model(
        id_=id_,
        minio=minio,
        bucket_name=global_config.minio.trained_models_bucket_name,
        collection=mongo_collection_models
    )
    return StreamingResponse(
        content=file.file,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={file.filename}"}
    )


@router.delete(path="/", name="Удалить модели")
def delete_models(ids: list[str] = Query()) -> JSONResponse:
    for id_ in ids:
        tasks.delete_model.apply_async(kwargs={"id_": id_})
    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={"status": "Models deleting is started"}
    )
