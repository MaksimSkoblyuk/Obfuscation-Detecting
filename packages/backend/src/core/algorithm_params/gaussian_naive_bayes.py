from pydantic import (
    Field,
    field_validator
)

from .base import BaseAlgorithmParams


class GaussianNBParams(BaseAlgorithmParams):
    class Config:
        title = "Gaussian Naive Bayes"

    var_smoothing: float = Field(default=1e-09, title="Portion of the largest variance")
    priors: str | list[float] | None = Field(
        default=None,
        title="Prior probabilities of the classes",
        description="Укажите значения через запятую. Например: 0.6,0.4",
    )

    @field_validator("priors", mode="after")
    @classmethod
    def cast_str_to_list(cls, value: str | None | list[float]) -> list[float] | None:
        if not value:
            return None
        if isinstance(value, list):
            return value
        return [float(prior.strip()) for prior in value.strip().split(",") if prior]
