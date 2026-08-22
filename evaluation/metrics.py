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
    Args:
        predicted: The LLM's answer.
        ground_truth: The expected correct answer.

    Returns:
        Float between 0.0 and 1.0 (higher is better).
    """
    if not predicted or not ground_truth:
        return 0.0

    if "ERROR:" in predicted or "don't have enough information" in predicted.lower():
        if "don't have enough information" in ground_truth.lower():
            return 1.0
        return 0.0

    try:
        import json
        from groq import Groq
        from src.config import GROQ_API_KEY, MODEL_NAME

        client = Groq(api_key=GROQ_API_KEY)

        prompt = (
            "You are an expert LLM evaluation judge.\n"
            "Evaluate the accuracy and completeness of the Predicted Answer compared to the Ground Truth answer.\n\n"
            f"Ground Truth:\n{ground_truth}\n\n"
            f"Predicted Answer:\n{predicted}\n\n"
            "Scoring Guidelines:\n"
            "- Focus on FACTUAL EQUIVALENCE and COMPLETENESS, NOT exact wording.\n"
            "- 1.0: Contains all key facts, numbers, conditions, and deadlines in Ground Truth.\n"
            "- 0.7-0.9: Correct core facts, but missing minor details or extra phrasing.\n"
            "- 0.4-0.6: Partially correct, but missing major key facts or conditions.\n"
            "- 0.0-0.3: Incorrect, misleading, or completely missing the point.\n\n"
            "Return ONLY a single valid JSON object in this format:\n"
            '{"accuracy_score": 0.95}'
        )

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        text = response.choices[0].message.content.strip()
        data = json.loads(text)
        score = float(data.get("accuracy_score", 0.0))
        return round(max(0.0, min(1.0, score)), 4)

    except Exception as e:
        print(f"[Metrics] Accuracy LLM Judge failed ({e}). Using fallback string scorer.")
        return _fallback_accuracy_score(predicted, ground_truth)


def _fallback_accuracy_score(predicted: str, ground_truth: str) -> float:
    """Fallback fuzzy string matching accuracy score."""
    norm_predicted = _normalize_text(predicted)
    norm_truth = _normalize_text(ground_truth)
    sequence_sim = SequenceMatcher(None, norm_predicted, norm_truth).ratio()

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

    accuracy = 0.6 * sequence_sim + 0.4 * keyword_f1
    return round(min(accuracy, 1.0), 4)


def hallucination_score(answer: str, context: str) -> float:
    """
    Args:
        answer: The LLM's generated answer.
        context: The retrieved context that was provided to the LLM.

    Returns:
        Float between 0.0 and 1.0 (0.0 = fully grounded, 1.0 = fully hallucinated).
    """
    if not answer or not context:
        return 1.0 if answer else 0.0

    try:
        import json
        from groq import Groq
        from src.config import GROQ_API_KEY, MODEL_NAME

        client = Groq(api_key=GROQ_API_KEY)

        prompt = (
            "You are an expert hallucination evaluation judge for a RAG system.\n"
            "Compare the Answer against the Context and determine the proportion of factual claims "
            "in the Answer that are NOT supported by or grounded in the Context.\n\n"
            f"Context:\n{context}\n\n"
            f"Answer:\n{answer}\n\n"
            "Evaluate faithfulness:\n"
            "- Score 0.0 means 100% grounded in context (0% hallucination).\n"
            "- Score 1.0 means completely ungrounded or contradictory.\n\n"
            "Return ONLY a single valid JSON object in this format:\n"
            '{"hallucination_score": 0.0}'
        )

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        text = response.choices[0].message.content.strip()
        data = json.loads(text)
        score = float(data.get("hallucination_score", 0.0))
        return round(max(0.0, min(1.0, score)), 4)

    except Exception as e:
        print(f"[Metrics] LLM Judge hallucination check skipped ({e}). Using fallback scorer.")
        return _fallback_hallucination_score(answer, context)


def _fallback_hallucination_score(answer: str, context: str) -> float:
    
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
   
    return round(latency_seconds, 4)


def cost_metric(token_usage: dict, input_rate: float = None, output_rate: float = None) -> float:
    
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
    
    if accuracy < thresholds.get("accuracy", 0.7):
        return True
    if hallucination > thresholds.get("hallucination", 0.2):
        return True
    if latency > thresholds.get("latency", 10.0):
        return True
    if cost > thresholds.get("cost", 0.05):
        return True
    return False
