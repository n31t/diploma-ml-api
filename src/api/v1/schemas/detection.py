from typing import Literal

from pydantic import BaseModel, Field


class AiSpan(BaseModel):
    start: int
    end: int
    score: float


class DetectionRequest(BaseModel):
    text: str = Field(min_length=1)
    language: Literal["ru", "kk"] = "ru"


class DetectionResponse(BaseModel):
    label: str
    ai_probability: float
    certainty: float
    ai_spans: list[AiSpan]
    model_used: str
