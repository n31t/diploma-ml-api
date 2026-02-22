"""Aggregates per-chunk detection results into a single document-level result."""
from src.dtos.detection_dto import AiSpanDTO, DetectionResultDTO


def aggregate_chunk_results(
    results: list[DetectionResultDTO],
    chunks: list[str],
) -> DetectionResultDTO:
    """Combine per-chunk DetectionResultDTO list into one document-level result.

    Uses word-count-weighted averaging for probabilities and certainty.
    """
    if not results:
        raise ValueError("No chunk results to aggregate")

    if len(results) == 1:
        return results[0]

    word_counts = [len(c.split()) for c in chunks]
    total_words = sum(word_counts)
    weights = [wc / total_words for wc in word_counts]

    weighted_ai_prob = sum(r.ai_probability * w for r, w in zip(results, weights))
    weighted_certainty = sum(r.certainty * w for r, w in zip(results, weights))

    # Majority-vote label
    ai_votes = sum(1 for r in results if r.label.upper() == "AI")
    label = "AI" if ai_votes > len(results) / 2 else "HUMAN"

    # Remap ai_spans to absolute character offsets
    ai_spans: list[AiSpanDTO] = []
    offset = 0
    for chunk, result in zip(chunks, results):
        for span in result.ai_spans:
            ai_spans.append(AiSpanDTO(
                start=span.start + offset,
                end=span.end + offset,
                score=span.score,
            ))
        offset += len(chunk) + 2  # +2 for "\n\n" separator

    model_used = results[0].model_used

    return DetectionResultDTO(
        label=label,
        ai_probability=round(weighted_ai_prob, 4),
        certainty=round(weighted_certainty, 4),
        ai_spans=ai_spans,
        model_used=model_used,
    )