"""Splits text into semantically complete chunks for ML inference."""
import re


_SENTENCE_ENDINGS = re.compile(r'(?<=[.!?])\s+')
_MAX_WORDS = 500
_MIN_WORDS = 200


def _count_words(text: str) -> int:
    return len(text.split())


def _split_paragraph_into_chunks(paragraph: str) -> list[str]:
    """Split a long paragraph into sentence-boundary-respecting chunks."""
    sentences = _SENTENCE_ENDINGS.split(paragraph.strip())
    chunks: list[str] = []
    current_parts: list[str] = []
    current_word_count = 0

    for sentence in sentences:
        sentence_words = _count_words(sentence)
        if current_word_count + sentence_words > _MAX_WORDS and current_parts:
            chunks.append(" ".join(current_parts))
            current_parts = [sentence]
            current_word_count = sentence_words
        else:
            current_parts.append(sentence)
            current_word_count += sentence_words

    if current_parts:
        chunks.append(" ".join(current_parts))

    return [c for c in chunks if c.strip()]


def split_text_into_chunks(text: str) -> list[str]:
    """Split text into logically complete chunks of 200–500 words.

    Splits on paragraphs first; further splits long paragraphs at sentence
    boundaries. Never cuts a sentence in the middle.
    """
    paragraphs = [p.strip() for p in re.split(r'\n{2,}|\r\n{2,}', text) if p.strip()]

    chunks: list[str] = []
    pending: list[str] = []
    pending_words = 0

    for paragraph in paragraphs:
        para_words = _count_words(paragraph)

        if para_words > _MAX_WORDS:
            # Flush pending before handling oversized paragraph
            if pending:
                chunks.append("\n\n".join(pending))
                pending, pending_words = [], 0
            chunks.extend(_split_paragraph_into_chunks(paragraph))
            continue

        if pending_words + para_words > _MAX_WORDS and pending:
            chunks.append("\n\n".join(pending))
            pending, pending_words = [], 0

        pending.append(paragraph)
        pending_words += para_words

        if pending_words >= _MIN_WORDS:
            chunks.append("\n\n".join(pending))
            pending, pending_words = [], 0

    if pending:
        chunks.append("\n\n".join(pending))

    return chunks or [text]