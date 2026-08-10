import time
from openai import OpenAI

from src.config import (
    OPENAI_API_KEY,
    MODEL_NAME,
    TEMPERATURE,
    MAX_TOKENS,
    INPUT_COST_PER_TOKEN,
    OUTPUT_COST_PER_TOKEN,
)


class LLMClient:
    """
    Wrapper around the OpenAI Chat Completions API.

    Tracks latency, token usage, and cost per query.
    """

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or OPENAI_API_KEY
        self.model = model or MODEL_NAME
        self.client = OpenAI(api_key=self.api_key)

        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Please set it in .env or as an environment variable."
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
        system_message = (
            "You are a helpful university assistant. Answer the student's question "
            "strictly based on the provided context. If the context does not contain "
            "enough information to answer, say 'I don't have enough information to "
            "answer this question based on the available rules.' Do not make up "
            "information or add details not present in the context."
        )

        user_message = (
            f"Context:\n{context}\n\n"
            f"Question: {prompt}\n\n"
            f"Answer based only on the context above:"
        )

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        last_error = None
        for attempt in range(max_retries):
            try:
                start_time = time.time()

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )

                latency = time.time() - start_time

                # Extract response data
                answer = response.choices[0].message.content.strip()
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
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
