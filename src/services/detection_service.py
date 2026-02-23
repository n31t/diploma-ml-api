"""AI text detection service using RuBERT and GigaCheck."""
import asyncio

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.core.config import BASE_DIR, config
from src.core.logging import get_logger
from src.dtos.detection_dto import DetectionInputDTO, DetectionResultDTO
from src.utils.chunk_aggregator import aggregate_chunk_results
from src.utils.text_chunker import split_text_into_chunks

logger = get_logger(__name__)

_RUBERT_BASE_MODEL = "DeepPavlov/rubert-base-cased"
_RUBERT_CHECKPOINT = str(BASE_DIR / "src" / "models" / "rubert-base-ainl-peft" / "checkpoint-7000")
_RUBERT_ID2LABEL = {0: "HUMAN", 1: "AI"}

_GIGACHECK_MODEL = "iitolstykh/GigaCheck-Classifier-Multi"


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
        tokenizer = AutoTokenizer.from_pretrained(_RUBERT_BASE_MODEL, cache_dir=config.hf_cache_dir)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            _RUBERT_BASE_MODEL,
            num_labels=2,
            cache_dir=config.hf_cache_dir
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

    async def detect_chunk(self, dto: DetectionInputDTO) -> DetectionResultDTO:
        """Run inference on a single chunk and return its result."""
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

    async def detect(self, dto: DetectionInputDTO) -> DetectionResultDTO:
        """Detect AI-generated text with chunking for long inputs."""
        chunks = split_text_into_chunks(dto.text)
        logger.info("rubert_chunking", total_chunks=len(chunks))

        chunk_results = [
            await self.detect_chunk(DetectionInputDTO(text=chunk))
            for chunk in chunks
        ]
        return aggregate_chunk_results(chunk_results, chunks)


class GigaCheckService:
    """GigaCheck-Classifier-Multi for binary AI text classification."""

    def __init__(self) -> None:
        self._device = _get_device()
        self._model = None

    def _load_model_sync(self) -> None:
        from transformers import AutoModel  # noqa: PLC0415

        dtype = torch.bfloat16 if self._device != "cpu" else torch.float32
        model = AutoModel.from_pretrained(
            _GIGACHECK_MODEL,
            trust_remote_code=True,
            device_map="cpu",
            dtype=dtype,
            cache_dir=config.hf_cache_dir,
        )
        model.eval()
        self._model = model

    async def load(self) -> None:
        """Load model in a thread pool so the event loop is not blocked."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_model_sync)

    async def detect_chunk(self, dto: DetectionInputDTO) -> DetectionResultDTO:
        """Run inference on a single chunk and return its result."""
        text = dto.text.replace("\n", " ")
        output = await asyncio.get_event_loop().run_in_executor(
            None, lambda: self._model([text])
        )
        label_id = int(output.pred_label_ids[0])
        label = self._model.config.id2label[label_id]

        if hasattr(output, 'classification_head_probs') and output.classification_head_probs is not None:
            probs = output.classification_head_probs[0].detach().cpu().numpy()
            ai_probability = float(probs[0] * 100.0)  # Class 0 is AI probability
            certainty = float(probs[label_id] * 100.0)  # Confidence in predicted class
        else:
            is_ai = label.lower() == "ai"
            ai_probability = 100.0 if is_ai else 0.0
            certainty = 100.0

        return DetectionResultDTO(
            label=label,
            ai_probability=ai_probability,
            certainty=certainty,
            ai_spans=[],
            model_used="gigacheck",
        )

    async def detect(self, dto: DetectionInputDTO) -> DetectionResultDTO:
        """Detect AI-generated text with chunking for long inputs."""
        chunks = split_text_into_chunks(dto.text)
        logger.debug("gigacheck_chunking", total_chunks=len(chunks))

        chunk_results = [
            await self.detect_chunk(DetectionInputDTO(text=chunk))
            for chunk in chunks
        ]
        return aggregate_chunk_results(chunk_results, chunks)


class DetectionService:
    """AI text detection: GigaCheck primary, RuBERT fallback."""

    def __init__(self, rubert: RuBertService, gigacheck: GigaCheckService) -> None:
        self._rubert = rubert
        self._gigacheck = gigacheck
        self._gigacheck_available = False

    async def load(self) -> None:
        """Load both models, gracefully handling GigaCheck failures."""
        await self._rubert.load()
        logger.info("rubert_loaded_successfully")

        try:
            await self._gigacheck.load()
            self._gigacheck_available = True
            logger.info("gigacheck_loaded_successfully")
        except Exception as e:
            logger.warning("gigacheck_load_failed_using_rubert_only", error=str(e))

    async def detect(self, dto: DetectionInputDTO) -> DetectionResultDTO:
        """Try GigaCheck first; fall back to RuBERT on failure."""
        if self._gigacheck_available:
            try:
                return await self._gigacheck.detect(dto)
            except Exception as e:
                logger.warning("gigacheck_inference_failed_falling_back_to_rubert", error=str(e))

        return await self._rubert.detect(dto)