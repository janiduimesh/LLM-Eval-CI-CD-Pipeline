"""
TF-IDF + Cosine Similarity retriever for the knowledge base.
Loads .txt files, chunks them, and retrieves top-k relevant chunks.
"""

import os
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import KNOWLEDGE_BASE_DIR, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RETRIEVAL


class Retriever:
    """
    A simple TF-IDF based document retriever.

    Loads all .txt files from the knowledge base directory, splits them into
    overlapping chunks, and uses TF-IDF vectorization with cosine similarity
    for retrieval.
    """

    def __init__(self, knowledge_base_dir: str = None, chunk_size: int = None, chunk_overlap: int = None):
        self.knowledge_base_dir = knowledge_base_dir or KNOWLEDGE_BASE_DIR
        self.chunk_size = chunk_size or CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or CHUNK_OVERLAP
        self.chunks = []
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = None

        self._load_and_chunk()
        self._build_index()

    def _load_and_chunk(self):
        """Load all .txt files and split into overlapping chunks."""
        txt_files = glob.glob(os.path.join(self.knowledge_base_dir, "*.txt"))

        if not txt_files:
            raise FileNotFoundError(
                f"No .txt files found in {self.knowledge_base_dir}"
            )

        for filepath in sorted(txt_files):
            filename = os.path.basename(filepath)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            # Split into chunks with overlap
            start = 0
            while start < len(content):
                end = start + self.chunk_size
                chunk_text = content[start:end].strip()
                if chunk_text:
                    self.chunks.append({
                        "text": chunk_text,
                        "source": filename,
                        "start_char": start,
                    })
                start += self.chunk_size - self.chunk_overlap

        print(f"[Retriever] Loaded {len(self.chunks)} chunks from {len(txt_files)} files.")

    def _build_index(self):
        """Build the TF-IDF index from chunks."""
        if not self.chunks:
            raise ValueError("No chunks to index. Check knowledge base directory.")

        texts = [chunk["text"] for chunk in self.chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        print(f"[Retriever] TF-IDF index built with vocabulary size {len(self.vectorizer.vocabulary_)}.")

    def retrieve(self, query: str, top_k: int = None) -> list[dict]:
        """
        Retrieve top-k most relevant chunks for a given query.

        Args:
            query: The search query.
            top_k: Number of top results to return. Defaults to config TOP_K_RETRIEVAL.

        Returns:
            List of dicts with keys: text, source, score.
        """
        top_k = top_k or TOP_K_RETRIEVAL

        # Transform query using the fitted vectorizer
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Get top-k indices sorted by similarity (descending)
        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include non-zero similarity
                results.append({
                    "text": self.chunks[idx]["text"],
                    "source": self.chunks[idx]["source"],
                    "score": float(similarities[idx]),
                })

        return results
