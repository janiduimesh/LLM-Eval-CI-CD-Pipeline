import time
import google.generativeai as genai

from src.config import (
    GEMINI_API_KEY,
    MODEL_NAME,
    TEMPERATURE,
    MAX_TOKENS,
    INPUT_COST_PER_TOKEN,
    OUTPUT_COST_PER_TOKEN,
)


class LLMClient:
    """
    Wrapper around the Google Gemini Generative AI API.

    Tracks latency, token usage, and cost per query.
    """

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or GEMINI_API_KEY
        self.model_name = model or MODEL_NAME

        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Please set it in .env or as an environment variable."
            )

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=genai.GenerationConfig(
                temperature=TEMPERATURE,
                max_output_tokens=MAX_TOKENS,
            ),
            system_instruction=(
                "You are a helpful university assistant. Answer the student's question "
                "strictly based on the provided context. If the context does not contain "
                "enough information to answer, say 'I don't have enough information to "
                "answer this question based on the available rules.' Do not make up "
                "information or add details not present in the context."
            ),
        )

    def generate(self, prompt: str, context: str, max_retries: int = 3) -> dict:
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

                response = self.model.generate_content(user_message)

                latency = time.time() - start_time

                # Extract response data
                answer = response.text.strip()

                # Extract token usage from Gemini's usage metadata
                usage_metadata = response.usage_metadata
                usage = {
                    "prompt_tokens": usage_metadata.prompt_token_count,
                    "completion_tokens": usage_metadata.candidates_token_count,
                    "total_tokens": usage_metadata.total_token_count,
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
                print(f"[LLMClient] Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff

        raise RuntimeError(
            f"[LLMClient] All {max_retries} attempts failed. Last error: {last_error}"
        )
