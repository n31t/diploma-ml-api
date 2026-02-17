"""AI text detection service using RuBERT."""
import asyncio

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.core.config import BASE_DIR
from src.core.logging import get_logger
from src.dtos.detection_dto import DetectionInputDTO, DetectionResultDTO

logger = get_logger(__name__)

_RUBERT_BASE_MODEL = "DeepPavlov/rubert-base-cased"
_RUBERT_CHECKPOINT = str(BASE_DIR / "src" / "models" / "rubert-base-ainl-peft" / "checkpoint-3072")
_RUBERT_ID2LABEL = {0: "HUMAN", 1: "AI"}


def _get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class RuBertService:
    """Fine-tuned RuBERT with LoRA for binary AI text classification."""

    def __init__(self) -> None:
        self._device = _get_device()
        self._tokenizer = None
        self._model = None

    def _load_model_sync(self) -> None:
        tokenizer = AutoTokenizer.from_pretrained(_RUBERT_BASE_MODEL)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            _RUBERT_BASE_MODEL,
            num_labels=2,
        )
        model = PeftModel.from_pretrained(base_model, _RUBERT_CHECKPOINT)
        model.config.id2label = _RUBERT_ID2LABEL
        model.config.label2id = {v: k for k, v in _RUBERT_ID2LABEL.items()}
        model = model.to(self._device)
        model.eval()
        self._tokenizer = tokenizer
        self._model = model

    async def load(self) -> None:
        """Load model in a thread pool so the event loop is not blocked."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_model_sync)

    async def detect(self, dto: DetectionInputDTO) -> DetectionResultDTO:
        """Run inference and return detection result."""
        inputs = self._tokenizer(
            dto.text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self._model(**inputs).logits

        probs = torch.softmax(logits, dim=-1)[0]
        predicted_id = logits.argmax().item()
        label = _RUBERT_ID2LABEL[predicted_id]
        certainty = float(100.0 * probs[predicted_id])
        ai_probability = float(100.0 * probs[1])

        return DetectionResultDTO(
            label=label,
            ai_probability=ai_probability,
            certainty=certainty,
            ai_spans=[],
            model_used="rubert",
        )


class DetectionService:
    """AI text detection service using RuBERT."""

    def __init__(self, rubert: RuBertService) -> None:
        self._rubert = rubert

    async def load(self) -> None:
        """Load the RuBERT model."""
        await self._rubert.load()
        logger.info("rubert_loaded_successfully")

    async def detect(self, dto: DetectionInputDTO) -> DetectionResultDTO:
        """Run inference and return detection result."""
        return await self._rubert.detect(dto)
