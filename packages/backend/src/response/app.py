import typing
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi_pagination import add_pagination
from fastapi.middleware.cors import CORSMiddleware

from ..core import mongodb_client
from .models import router as models_router
from .schemas import router as schemas_router
from .datasets import router as datasets_router
from .classifications import router as classifications_router

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)-8s %(name)s:%(funcName)s:%(lineno)d - %(message)s",
    handlers=[
        logging.FileHandler("tip_sandbox_logs.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> typing.AsyncGenerator[None, None]:
    yield
    mongodb_client.close()
    logger.info("Connection with MongoDB was closed successfully")


app = FastAPI(lifespan=lifespan, title="Obfuscation Detecting", root_path="/api/v1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"],
    allow_credentials=True,
)
app.include_router(router=schemas_router)
app.include_router(router=datasets_router)
app.include_router(router=models_router)
app.include_router(router=classifications_router)
add_pagination(parent=app)
