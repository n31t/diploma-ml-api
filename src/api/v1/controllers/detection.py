from dishka.integrations.fastapi import DishkaRoute, FromDishka
from fastapi import APIRouter

from src.api.v1.schemas.detection import AiSpan, DetectionRequest, DetectionResponse
from src.dtos.detection_dto import DetectionInputDTO
from src.services.detection_service import DetectionService

router = APIRouter(
    route_class=DishkaRoute,
    prefix="/api/v1/detection",
    tags=["Detection"],
)


@router.post("/", response_model=DetectionResponse)
async def detect_text(
    request: DetectionRequest,
    service: FromDishka[DetectionService],
) -> DetectionResponse:
    """Detect whether the provided text is human-written or AI-generated (RuBERT)."""
    input_dto = DetectionInputDTO(text=request.text)
    result = await service.detect(input_dto)
    return DetectionResponse(
        label=result.label,
        ai_probability=result.ai_probability,
        certainty=result.certainty,
        ai_spans=[AiSpan(start=s.start, end=s.end, score=s.score) for s in result.ai_spans],
        model_used=result.model_used,
    )


@router.post("/gigacheck", response_model=DetectionResponse)
async def detect_text_gigacheck(
    request: DetectionRequest,
    service: FromDishka[DetectionService],
) -> DetectionResponse:
    """Detect whether the provided text is human-written or AI-generated (GigaCheck)."""
    input_dto = DetectionInputDTO(text=request.text)
    result = await service.detect_gigacheck(input_dto)
    return DetectionResponse(
        label=result.label,
        ai_probability=result.ai_probability,
        certainty=result.certainty,
        ai_spans=[AiSpan(start=s.start, end=s.end, score=s.score) for s in result.ai_spans],
        model_used=result.model_used,
    )
