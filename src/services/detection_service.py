"""AI text detection services: GigaCheck (primary) and RuBERT (fallback)."""
import asyncio

import torch
from peft import PeftModel
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from src.core.config import BASE_DIR
from src.core.logging import get_logger
from src.dtos.detection_dto import AiSpanDTO, DetectionInputDTO, DetectionResultDTO

logger = get_logger(__name__)

_RUBERT_BASE_MODEL = "DeepPavlov/rubert-base-cased"
_RUBERT_CHECKPOINT = str(BASE_DIR / "src" / "models" / "rubert-base-ainl-peft" / "checkpoint-3072")
_RUBERT_ID2LABEL = {0: "HUMAN", 1: "AI"}

_GIGACHECK_MODEL = "iitolstykh/GigaCheck-Detector-Multi"


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


class GigaCheckService:
    """GigaCheck Mistral-7B model for character-level AI span detection."""

    def __init__(self) -> None:
        # Force CPU for GigaCheck to avoid MPS memory limits
        # MPS has ~18GB limit which is insufficient for 7B model
        self._device = "cpu"
        self._model = None

    def _load_model_sync(self) -> None:
        model = AutoModel.from_pretrained(
            _GIGACHECK_MODEL,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
        model = model.to(self._device)
        model.eval()
        self._model = model

    async def load(self) -> None:
        """Load model in a thread pool so the event loop is not blocked."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_model_sync)

    async def detect(self, dto: DetectionInputDTO) -> DetectionResultDTO:
        """Run inference and return character-level AI span detection result."""
        loop = asyncio.get_event_loop()
        output = await loop.run_in_executor(
            None,
            lambda: self._model([dto.text], conf_interval_thresh=0.5),
        )

        # ai_intervals: List[Tensor[Num_Intervals, 3]], one tensor per batch item
        # Each row: (start_char, end_char, score)
        intervals_tensor = output.ai_intervals[0] if output.ai_intervals else None
        if intervals_tensor is not None and intervals_tensor.shape[0] > 0:
            ai_spans = [
                AiSpanDTO(start=int(row[0]), end=int(row[1]), score=float(row[2]))
                for row in intervals_tensor
            ]
        else:
            ai_spans = []

        # classification_head_probs: Tensor[Batch, Num_Classes] — use for label and probability
        # pred_label_ids: Tensor[Batch] — predicted class index
        if output.classification_head_probs is not None:
            probs = output.classification_head_probs[0]  # [Num_Classes]
            pred_id = int(output.pred_label_ids[0])
            id2label = self._model.config.id2label  # e.g. {0: "Human", 1: "AI", 2: "Mixed"}
            pred_label_str = id2label.get(pred_id, "HUMAN").upper()
            label = "AI" if pred_label_str in ("AI", "MIXED") else "HUMAN"
            # AI probability: sum of AI + Mixed class probabilities if present
            ai_class_ids = [k for k, v in id2label.items() if v.upper() in ("AI", "MIXED")]
            ai_probability = float(sum(probs[i] for i in ai_class_ids) * 100)
            certainty = float(probs[pred_id] * 100)
        else:
            # Fallback: derive from spans
            label = "AI" if ai_spans else "HUMAN"
            ai_probability = float(max((s.score for s in ai_spans), default=0.0) * 100)
            certainty = ai_probability

        return DetectionResultDTO(
            label=label,
            ai_probability=ai_probability,
            certainty=certainty,
            ai_spans=ai_spans,
            model_used="gigacheck",
        )


class DetectionService:
    """Orchestrates GigaCheck (primary) with RuBERT as automatic fallback."""

    def __init__(self, gigacheck: GigaCheckService, rubert: RuBertService) -> None:
        self._gigacheck = gigacheck
        self._rubert = rubert

    async def load(self) -> None:
        """Load both models, gracefully handling GigaCheck failures."""
        # Always load RuBERT first (fallback)
        await self._rubert.load()
        logger.info("rubert_loaded_successfully")
        
        # Try to load GigaCheck, but don't fail if unavailable
        try:
            await self._gigacheck.load()
            logger.info("gigacheck_loaded_successfully")
        except Exception as e:
            logger.warning(
                "gigacheck_load_failed_using_rubert_only",
                error=str(e),
                error_type=type(e).__name__
            )
            self._gigacheck._model = None  # Mark as unavailable

    async def detect(self, dto: DetectionInputDTO) -> DetectionResultDTO:
        """Try GigaCheck first; fall back to RuBERT on any error."""
        if self._gigacheck._model is not None:
            try:
                return await self._gigacheck.detect(dto)
            except Exception as e:
                logger.warning(
                    "gigacheck_detection_failed_falling_back_to_rubert",
                    error=str(e)
                )
        
        return await self._rubert.detect(dto)
