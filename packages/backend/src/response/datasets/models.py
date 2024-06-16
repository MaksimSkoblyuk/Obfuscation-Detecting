from pydantic import BaseModel


class DatasetCharacteristics(BaseModel):
    features: int
    samples: int
