from pydantic import (
    Field,
    field_validator
)

from .base import BaseAlgorithmParams


class MultinomialNBClassifierParams(BaseAlgorithmParams):
    class Config:
        title = "Multinomial Naive Bayes"

    alpha: float | str | list[float] = Field(
        default=1.0,
        title="Additive smoothing parameter",
        description="Если несколько значений, то укажите их через запятую. Например: 0.6,0.4"
    )
    fit_prior: bool = Field(default=False, title="Learn class prior probabilities")
    class_prior: str | list[float] | None = Field(
        default=None,
        title="Prior probabilities of the classes",
        description="Укажите значения через запятую. Например: 0.6,0.4",
    )

    @field_validator("class_prior", mode="after")
    @classmethod
    def cast_str_with_none(cls, value: str | None | list[float]) -> list[float] | None:
        if not value:
            return None
        if isinstance(value, list):
            return value
        return [float(prior.strip()) for prior in value.strip().split(",") if prior]

    @field_validator("alpha", mode="after")
    @classmethod
    def cast_str_without_none(cls, value: str | float | list[float]) -> list[float] | float:
        if isinstance(value, (str, list)) and not value:
            return 1.0
        if isinstance(value, (float, list)):
            return value
        if len(value.strip().split(",")) == 1:
            return float(value.strip())
        return [float(alpha.strip()) for alpha in value.strip().split(",") if alpha]
