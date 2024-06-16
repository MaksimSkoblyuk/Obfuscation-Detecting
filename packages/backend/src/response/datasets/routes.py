from fastapi import (
    Path,
    Query,
    status,
    APIRouter,
    UploadFile
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
from ...core.enums import Extension
from ...core.models import DatasetDocument
from ...core import service as core_service
from ...core import (
    minio,
    global_config,
    mongo_collection_datasets
)

router = APIRouter(prefix="/datasets", tags=["Datasets"])
disable_installed_extensions_check()


@router.post(path="/", name="Загрузить наборы данных")
def upload_datasets(datasets: list[UploadFile]) -> JSONResponse:
    for dataset in datasets:
        if not core_service.is_right_file_extension(dataset.filename, Extension.CSV):
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "status": f"Wrong file {dataset.filename} extension. "
                              f"Expected {Extension.CSV.value}"
                }
            )
    for dataset in datasets:
        tasks.upload_dataset.apply_async(
            kwargs={
                "file_data": dataset.file.read(),
                "filename": dataset.filename
            }
        )
    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={"status": "Datasets uploading is started"}
    )


@router.get(path="/", name="Получить наборы данных", response_model=Page[DatasetDocument])
def get_datasets() -> Page[DatasetDocument]:
    return paginate(
        core_service.get_documents(
            document_class=DatasetDocument,
            collection=mongo_collection_datasets
        )
    )


@router.get(path="/{id}/download", name="Скачать набор данных")
def download_dataset(id_: str = Path(alias="id")) -> StreamingResponse:
    file = service.download_dataset(
        id_=id_,
        minio=minio,
        bucket_name=global_config.minio.preprocessed_datasets_bucket_name,
        collection=mongo_collection_datasets
    )
    return StreamingResponse(
        content=file.file,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={file.filename}"}
    )


@router.delete(path="/", name="Удалить наборы данных")
def delete_datasets(ids: list[str] = Query()) -> JSONResponse:
    for id_ in ids:
        tasks.delete_dataset.apply_async(kwargs={"id_": id_})
    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={"status": "Datasets deleting is started"}
    )
