from pydantic import BaseModel

from ...core.models import CommonClassification


class ClassificationResponse(BaseModel):
    total: int = 0
    classifications: list[CommonClassification]
