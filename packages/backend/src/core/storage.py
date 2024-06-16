import logging
from io import BytesIO

from minio import Minio

logger = logging.getLogger(__name__)


def upload_file(
        minio_client: Minio,
        bucket_name: str,
        file_name: str,
        file_data: bytes,
        part_size: int = 10 * 8 * 1024 * 1024
) -> str:
    logger.info(f"Start uploading {file_name!r} to Minio bucket {bucket_name!r}")
    response = minio_client.put_object(
        bucket_name=bucket_name,
        object_name=file_name,
        data=BytesIO(file_data),
        length=-1,
        part_size=part_size
    )
    logger.info(f"File {file_name!r} was uploaded to Minio bucket {bucket_name!r}")
    return response.etag


def download_file(minio_client: Minio, bucket_name: str, file_name: str) -> bytes:
    logger.info(f"Start downloading {file_name!r} from Minio bucket {bucket_name!r}")
    file = minio_client.get_object(bucket_name=bucket_name, object_name=file_name)
    logger.info(f"File {file_name!r} was downloaded from Minio bucket {bucket_name!r}")
    return file.read()


def delete_file(minio_client: Minio, bucket_name: str, file_name: str) -> None:
    logger.info(f"Start deleting {file_name!r} from Minio bucket {bucket_name!r}")
    minio_client.remove_object(bucket_name=bucket_name, object_name=file_name)
    logger.info(f"File {file_name!r} was deleted from Minio bucket {bucket_name!r}")
