import time
import re
from groq import Groq

from src.config import (
    GROQ_API_KEY,
    MODEL_NAME,
    TEMPERATURE,
    MAX_TOKENS,
    INPUT_COST_PER_TOKEN,
    OUTPUT_COST_PER_TOKEN,
)


SYSTEM_INSTRUCTION = (
    "You are a helpful university assistant. When answering the student's question, "
    "provide a clear, complete, and comprehensive response based strictly on the provided context. "
    "Be sure to include all relevant numbers, deadlines, time limits, conditions, exceptions, and specific details. "
    "Do not omit important details. "
    "If the context does not contain enough information to answer, say "
    "'I don't have enough information to answer this question based on the available rules.'"
)


class LLMClient:
    """
    Wrapper around the Groq API (groq SDK).

    Tracks latency, token usage, and cost per query.
    """

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or GROQ_API_KEY
        self.model_name = model or MODEL_NAME

        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Please set it in .env or as an environment variable."
            )

        self.client = Groq(api_key=self.api_key)

    def generate(self, prompt: str, context: str, max_retries: int = 5) -> dict:
        """
        Generate a response from the LLM via Groq.

        Args:
            prompt: The user's question.
            context: Retrieved context to ground the response.
            max_retries: Number of retry attempts on failure.

        Returns:
            Dict with keys: answer, usage, latency, cost
        """
        user_message = (
            f"Context:\n{context}\n\n"
            f"Question: {prompt}\n\n"
            f"Answer based only on the context above:"
        )

        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": user_message},
        ]

        last_error = None
        for attempt in range(max_retries):
            try:
                start_time = time.time()

                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )

                latency = time.time() - start_time

                # Extract response data
                answer = response.choices[0].message.content.strip()

                # Extract token usage
                usage_obj = response.usage
                usage = {
                    "prompt_tokens": getattr(usage_obj, "prompt_tokens", 0) or 0,
                    "completion_tokens": getattr(usage_obj, "completion_tokens", 0) or 0,
                    "total_tokens": getattr(usage_obj, "total_tokens", 0) or 0,
                }

                # Calculate cost
                cost = (
                    usage["prompt_tokens"] * INPUT_COST_PER_TOKEN
                    + usage["completion_tokens"] * OUTPUT_COST_PER_TOKEN
                )

                return {
                    "answer": answer,
                    "usage": usage,
                    "latency": round(latency, 4),
                    "cost": round(cost, 8),
                }

            except Exception as e:
                last_error = e
                err_str = str(e)

                if attempt < max_retries - 1:
                    if "429" in err_str or "rate_limit" in err_str.lower() or "quota" in err_str.lower():
                        match = re.search(r"try again in (\d+(?:\.\d+)?)s", err_str, re.IGNORECASE)
                        wait_sec = float(match.group(1)) + 1.0 if match else 5.0 * (attempt + 1)
                        print(f"[LLMClient] ⏳ Groq Rate Limit (429) hit. Waiting {wait_sec:.1f}s before retry...")
                        time.sleep(wait_sec)
                    else:
                        print(f"[LLMClient] Attempt {attempt + 1}/{max_retries} failed: {e}")
                        time.sleep(2 ** attempt)

        raise RuntimeError(
            f"[LLMClient] All {max_retries} attempts failed. Last error: {last_error}"
        )
