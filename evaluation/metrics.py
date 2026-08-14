"""
Evaluation metrics for the LLM Evaluation CI/CD Pipeline.

Five metrics:
  1. Accuracy (relevancy to ground truth)
  2. Hallucination rate (faithfulness to context)
  3. Latency (seconds per query)
  4. Cost (USD per query)
  5. Failed question detection (threshold breach check)
"""

import re
from difflib import SequenceMatcher


def _normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, strip, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    return text


def _extract_keywords(text: str) -> set:
    """Extract meaningful keywords from text (words with 3+ characters)."""
    normalized = _normalize_text(text)
    words = normalized.split()
    # Filter out very short/common words
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above",
        "below", "between", "out", "off", "over", "under", "again",
        "further", "then", "once", "and", "but", "or", "nor", "not", "so",
        "yet", "both", "either", "neither", "each", "every", "all", "any",
        "few", "more", "most", "other", "some", "such", "no", "only",
        "own", "same", "than", "too", "very", "just", "because", "about",
        "that", "this", "these", "those", "it", "its", "they", "them",
        "their", "we", "our", "you", "your", "he", "him", "his", "she",
        "her", "who", "whom", "which", "what", "where", "when", "how",
    }
    return {w for w in words if len(w) >= 3 and w not in stopwords}


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    sentences = re.split(r"[.!?]+", text)
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]


def accuracy_score(predicted: str, ground_truth: str) -> float:
    """
    Calculate accuracy as a combination of sequence similarity and keyword overlap.

    Uses SequenceMatcher for fuzzy string matching (60% weight) and
    keyword overlap (40% weight) for a balanced accuracy score.

    Args:
        predicted: The LLM's answer.
        ground_truth: The expected correct answer.

    Returns:
        Float between 0 and 1 (higher is better).
    """
    if not predicted or not ground_truth:
        return 0.0

    # Component 1: Sequence similarity (fuzzy string match)
    norm_predicted = _normalize_text(predicted)
    norm_truth = _normalize_text(ground_truth)
    sequence_sim = SequenceMatcher(None, norm_predicted, norm_truth).ratio()

    # Component 2: Keyword overlap (precision-recall F1)
    pred_keywords = _extract_keywords(predicted)
    truth_keywords = _extract_keywords(ground_truth)

    if not truth_keywords:
        return sequence_sim

    overlap = pred_keywords & truth_keywords
    if not overlap:
        keyword_f1 = 0.0
    else:
        precision = len(overlap) / len(pred_keywords) if pred_keywords else 0.0
        recall = len(overlap) / len(truth_keywords)
        keyword_f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Weighted combination
    accuracy = 0.6 * sequence_sim + 0.4 * keyword_f1
    return round(min(accuracy, 1.0), 4)


def hallucination_score(answer: str, context: str) -> float:
    """
    Measure how much of the answer is NOT grounded in the provided context.

    Splits the answer into atomic sentences/claims and verifies semantic
    entailment against the retrieved context using an LLM Judge (Gemini).
    A higher score means more hallucination (worse).

    Args:
        answer: The LLM's generated answer.
        context: The retrieved context that was provided to the LLM.

    Returns:
        Float between 0.0 and 1.0 (0.0 = fully grounded, 1.0 = fully hallucinated).
    """
    if not answer or not context:
        return 1.0 if answer else 0.0

    try:
        from google import genai
        from src.config import GEMINI_API_KEY, MODEL_NAME

        client = genai.Client(api_key=GEMINI_API_KEY)
        model_name = MODEL_NAME.replace("models/", "")

        # Step 1: Extract atomic claims from the answer
        extraction_prompt = (
            "Extract all atomic factual claims from the following answer. "
            "Each claim should be a single, self-contained statement that can be "
            "independently verified. Return ONLY a numbered list of claims, one per line. "
            "If the answer contains no verifiable claims, return 'NO_CLAIMS'.\n\n"
            f"Answer: {answer}"
        )

        extraction_response = client.models.generate_content(
            model=model_name,
            contents=extraction_prompt,
        )
        claims_text = extraction_response.text.strip()

        if "NO_CLAIMS" in claims_text:
            return 0.0

        # Parse claims from numbered list
        claims = []
        for line in claims_text.split("\n"):
            line = line.strip()
            # Remove numbering (e.g., "1.", "1)", "- ")
            cleaned = re.sub(r"^\d+[\.\)]\s*", "", line)
            cleaned = re.sub(r"^[-•]\s*", "", cleaned)
            cleaned = cleaned.strip()
            if cleaned and len(cleaned) > 10:
                claims.append(cleaned)

        if not claims:
            return 0.0

        # Step 2: Verify each claim against the context
        verification_prompt = (
            "You are a hallucination detection judge. For each claim below, determine "
            "whether it is SUPPORTED or NOT_SUPPORTED by the provided context.\n\n"
            "Rules:\n"
            "- SUPPORTED: The claim can be directly inferred from the context.\n"
            "- NOT_SUPPORTED: The claim contains information not present in or "
            "contradicted by the context.\n\n"
            "Return ONLY one verdict per line in the format: 'CLAIM_N: SUPPORTED' or "
            "'CLAIM_N: NOT_SUPPORTED' where N is the claim number.\n\n"
            f"Context:\n{context}\n\n"
            "Claims:\n"
        )
        for i, claim in enumerate(claims, 1):
            verification_prompt += f"{i}. {claim}\n"

        verification_response = client.models.generate_content(
            model=model_name,
            contents=verification_prompt,
        )
        verdicts_text = verification_response.text.strip()

        # Parse verdicts
        not_supported_count = 0
        verdict_count = 0
        for line in verdicts_text.split("\n"):
            line = line.strip().upper()
            if "NOT_SUPPORTED" in line:
                not_supported_count += 1
                verdict_count += 1
            elif "SUPPORTED" in line:
                verdict_count += 1

        # Use the number of claims we sent if parsing returned fewer verdicts
        total = max(verdict_count, len(claims))
        hallucination_rate = not_supported_count / total if total > 0 else 0.0

        return round(hallucination_rate, 4)

    except Exception as e:
        print(f"[Metrics] LLM Judge hallucination check failed, using fallback: {e}")
        return _fallback_hallucination_score(answer, context)


def _fallback_hallucination_score(answer: str, context: str) -> float:
    """
    Fallback hallucination scoring using keyword overlap and sequence matching.
    Used when the LLM Judge is unavailable.
    """
    answer_sentences = _split_into_sentences(answer)
    if not answer_sentences:
        return 0.0

    context_normalized = _normalize_text(context)
    context_keywords = _extract_keywords(context)

    ungrounded_count = 0

    for sentence in answer_sentences:
        sentence_keywords = _extract_keywords(sentence)
        if not sentence_keywords:
            continue

        overlap = sentence_keywords & context_keywords
        overlap_ratio = len(overlap) / len(sentence_keywords) if sentence_keywords else 0

        norm_sentence = _normalize_text(sentence)
        seq_sim = SequenceMatcher(None, norm_sentence, context_normalized).ratio()

        if overlap_ratio < 0.5 and seq_sim < 0.3:
            ungrounded_count += 1

    hallucination_rate = ungrounded_count / len(answer_sentences)
    return round(hallucination_rate, 4)


def latency_metric(latency_seconds: float) -> float:
    """
    Pass-through metric for latency.

    Args:
        latency_seconds: Time taken for the LLM call.

    Returns:
        The latency value in seconds.
    """
    return round(latency_seconds, 4)


def cost_metric(token_usage: dict, input_rate: float = None, output_rate: float = None) -> float:
    """
    Calculate the cost of a query based on token usage.

    Args:
        token_usage: Dict with prompt_tokens and completion_tokens.
        input_rate: Cost per input token (USD). Uses config default if None.
        output_rate: Cost per output token (USD). Uses config default if None.

    Returns:
        Cost in USD.
    """
    from src.config import INPUT_COST_PER_TOKEN, OUTPUT_COST_PER_TOKEN

    input_rate = input_rate or INPUT_COST_PER_TOKEN
    output_rate = output_rate or OUTPUT_COST_PER_TOKEN

    prompt_cost = token_usage.get("prompt_tokens", 0) * input_rate
    completion_cost = token_usage.get("completion_tokens", 0) * output_rate

    return round(prompt_cost + completion_cost, 8)


def is_failed(
    accuracy: float,
    hallucination: float,
    latency: float,
    cost: float,
    thresholds: dict,
) -> bool:
    """
    Determine if a single question's evaluation has failed any threshold.

    Args:
        accuracy: Accuracy score (higher is better).
        hallucination: Hallucination rate (lower is better).
        latency: Latency in seconds (lower is better).
        cost: Cost in USD (lower is better).
        thresholds: Dict with threshold values.

    Returns:
        True if any threshold is breached.
    """
    if accuracy < thresholds.get("accuracy", 0.7):
        return True
    if hallucination > thresholds.get("hallucination", 0.2):
        return True
    if latency > thresholds.get("latency", 10.0):
        return True
    if cost > thresholds.get("cost", 0.05):
        return True
    return False
