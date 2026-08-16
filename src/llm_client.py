import time
from google import genai
from google.genai import types

from src.config import (
    GEMINI_API_KEY,
    MODEL_NAME,
    TEMPERATURE,
    MAX_TOKENS,
    INPUT_COST_PER_TOKEN,
    OUTPUT_COST_PER_TOKEN,
)


SYSTEM_INSTRUCTION = (
    "You are a helpful university assistant. Answer the student's question "
    "strictly based on the provided context. If the context does not contain "
    "enough information to answer, say 'I don't have enough information to "
    "answer this question based on the available rules.' Do not make up "
    "information or add details not present in the context."
)


class LLMClient:
    """
    Wrapper around the Google Gemini API (google-genai SDK).

    Tracks latency, token usage, and cost per query.
    """

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or GEMINI_API_KEY
        self.model_name = (model or MODEL_NAME).replace("models/", "")

        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Please set it in .env or as an environment variable."
            )

        self.client = genai.Client(api_key=self.api_key)

    def generate(self, prompt: str, context: str, max_retries: int = 5) -> dict:
        """
        Generate a response from the LLM.

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

        last_error = None
        for attempt in range(max_retries):
            try:
                start_time = time.time()

                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=user_message,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        temperature=TEMPERATURE,
                        max_output_tokens=MAX_TOKENS,
                    ),
                )

                latency = time.time() - start_time

                # Extract response data
                answer = response.text.strip()

                # Extract token usage from Gemini's usage metadata
                usage_metadata = response.usage_metadata
                usage = {
                    "prompt_tokens": getattr(usage_metadata, "prompt_token_count", 0) or 0,
                    "completion_tokens": getattr(usage_metadata, "candidates_token_count", 0) or 0,
                    "total_tokens": getattr(usage_metadata, "total_token_count", 0) or 0,
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
                    # Check for rate limit / 429 quota exhaustion
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                        import re
                        match = re.search(r"retry in (\d+(?:\.\d+)?)s", err_str, re.IGNORECASE)
                        match_delay = re.search(r"retryDelay': '(\d+)s'", err_str)
                        if match:
                            wait_sec = float(match.group(1)) + 2.0
                        elif match_delay:
                            wait_sec = float(match_delay.group(1)) + 2.0
                        else:
                            wait_sec = 15.0 * (attempt + 1)
                        
                        print(f"[LLMClient] ⏳ Rate limit (429) hit. Waiting {wait_sec:.1f}s before retry (Attempt {attempt + 1}/{max_retries})...")
                        time.sleep(wait_sec)
                    else:
                        print(f"[LLMClient] Attempt {attempt + 1}/{max_retries} failed: {e}")
                        time.sleep(2 ** attempt)

        raise RuntimeError(
            f"[LLMClient] All {max_retries} attempts failed. Last error: {last_error}"
        )
