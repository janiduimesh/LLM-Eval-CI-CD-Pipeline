from src.retriever import Retriever
from src.llm_client import LLMClient
from src.config import TOP_K_RETRIEVAL


class RAGChatbot:

    def __init__(self, retriever: Retriever = None, llm_client: LLMClient = None):
        
        print("[RAGChatbot] Initializing...")
        self.retriever = retriever or Retriever()
        self.llm_client = llm_client or LLMClient()
        print("[RAGChatbot] Ready.")

    def answer(self, question: str) -> dict:
        
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
