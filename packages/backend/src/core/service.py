import hashlib
import pathlib
from typing import Type
from bson import ObjectId

from pymongo.collection import Collection

from .enums import Extension
from .models import TDocument


def calculate_md5(file_data: bytes, chunk_size: int = 4096) -> str:
    md5 = hashlib.md5()
    for i in range(0, len(file_data), chunk_size):
        md5.update(file_data[i:i + chunk_size])
    return md5.hexdigest()


def delete_file(id_: str, collection: Collection) -> None:
    collection.delete_one({"_id": ObjectId(id_)})


def check_existing_file(md5: str, collection: Collection) -> bool:
    file = collection.find_one({"md5": md5})
    return bool(file)


def get_documents(document_class: Type[TDocument], collection: Collection) -> list[TDocument]:
    return [document_class(**document) for document in collection.find()]


def get_limited_documents(
        document_class: Type[TDocument],
        collection: Collection,
        limit: int
) -> list[TDocument]:
    return [document_class(**document) for document in collection.find().limit(limit)]


def get_document_by_id(
        document_class: Type[TDocument],
        collection: Collection,
        id_: str
) -> TDocument:
    document = collection.find_one({"_id": ObjectId(id_)})
    if not document:
        raise ValueError(f"Document with id {id_} was not found in collection {collection.name}")
    return document_class(**document)


def get_documents_by_query(
        document_class: Type[TDocument],
        collection: Collection,
        field_name: str,
        value: str
) -> list[TDocument]:
    return [document_class(**document) for document in collection.find({field_name: value})]


def delete_documents_by_query(collection: Collection, field_name: str, value: str) -> None:
    collection.delete_many({field_name: value})


def delete_documents_by_multiple_query(collection: Collection, query: dict) -> None:
    collection.delete_many(query)


def render_filename(raw_filename: str, expected_extension: Extension) -> str:
    if not is_right_file_extension(filename=raw_filename, expected_extension=expected_extension):
        return f"{raw_filename}.{expected_extension.value}"
    return raw_filename


def is_right_file_extension(filename: str, expected_extension: Extension) -> bool:
    return pathlib.Path(filename).suffix == f'.{expected_extension.value}'
