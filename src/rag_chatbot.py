"""
RAG Chatbot orchestrator.
Combines the retriever and LLM client into a full RAG pipeline.
"""

from src.retriever import Retriever
from src.llm_client import LLMClient
from src.config import TOP_K_RETRIEVAL


class RAGChatbot:
    """
    Retrieval-Augmented Generation chatbot for university rules Q&A.

    Pipeline: Question → Retrieve relevant chunks → Build prompt → LLM call → Answer
    """

    def __init__(self, retriever: Retriever = None, llm_client: LLMClient = None):
        """
        Initialize the RAG chatbot.

        Args:
            retriever: Pre-initialized Retriever instance. Creates one if not provided.
            llm_client: Pre-initialized LLMClient instance. Creates one if not provided.
        """
        print("[RAGChatbot] Initializing...")
        self.retriever = retriever or Retriever()
        self.llm_client = llm_client or LLMClient()
        print("[RAGChatbot] Ready.")

    def answer(self, question: str) -> dict:
        """
        Answer a question using the RAG pipeline.

        Args:
            question: The student's question.

        Returns:
            Dict with keys:
                - answer: The LLM's response text
                - context_used: The concatenated context chunks sent to the LLM
                - sources: List of source filenames
                - retrieval_scores: Similarity scores from retriever
                - latency: Time taken for LLM call (seconds)
                - cost: Estimated cost in USD
                - tokens: Token usage breakdown
        """
        retrieved_chunks = self.retriever.retrieve(question, top_k=TOP_K_RETRIEVAL)

        if not retrieved_chunks:
            return {
                "answer": "I don't have enough information to answer this question based on the available rules.",
                "context_used": "",
                "sources": [],
                "retrieval_scores": [],
                "latency": 0.0,
                "cost": 0.0,
                "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }

        context_parts = []
        sources = []
        scores = []
        for chunk in retrieved_chunks:
            context_parts.append(chunk["text"])
            sources.append(chunk["source"])
            scores.append(chunk["score"])

        context_string = "\n\n---\n\n".join(context_parts)

        llm_result = self.llm_client.generate(
            prompt=question,
            context=context_string,
        )

        return {
            "answer": llm_result["answer"],
            "context_used": context_string,
            "sources": list(set(sources)),  # Deduplicate
            "retrieval_scores": scores,
            "latency": llm_result["latency"],
            "cost": llm_result["cost"],
            "tokens": llm_result["usage"],
        }
