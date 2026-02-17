from dataclasses import dataclass, field


@dataclass
class AiSpanDTO:
    start: int
    end: int
    score: float


@dataclass
class DetectionInputDTO:
    text: str


@dataclass
class DetectionResultDTO:
    label: str
    ai_probability: float
    certainty: float
    ai_spans: list[AiSpanDTO] = field(default_factory=list)
    model_used: str = "rubert"
