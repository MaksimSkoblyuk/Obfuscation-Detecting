from pydantic_settings import BaseSettings


class BaseConfig(BaseSettings):
    class Config:
        frozen = True
        extra = "ignore"
        env_file_encoding = "utf-8"
        env_nested_delimiter = '__'
        env_file = "../../../.docker/.env"

    def __hash__(self) -> int:
        return hash(self.model_dump_json())


class AppConfig(BaseConfig):
    url: str


class MongoConfig(BaseConfig):
    url: str
    database: str
    models_collection: str
    datasets_collection: str
    classifications_collection: str
    common_classifications_collection: str


class RedisConfig(BaseConfig):
    host: str
    port: int


class MinioConfig(BaseConfig):
    url: str
    access_key: str
    secret_key: str
    trained_models_bucket_name: str
    preprocessed_datasets_bucket_name: str
    part_size: int = 8 * 1024 * 1024  # 1 megabyte


class InitialFilesConfig(BaseConfig):
    statistics_dir_path: str
    trained_models_dir_path: str
    preprocessed_datasets_dir_path: str


class Config(BaseConfig):
    app: AppConfig
    mongo: MongoConfig
    redis: RedisConfig
    minio: MinioConfig
    initial_files: InitialFilesConfig
    locked_task_expiration: int = 1800  # 30 minutes
    locked_task_countdown: int = 15  # 15 seconds
    locked_task_max_retries: int = 100
    commands_column_name: str = "command"
