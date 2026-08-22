import os
import glob
import hashlib
import chromadb
from google import genai

from src.config import (
    KNOWLEDGE_BASE_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_RETRIEVAL,
    CHROMA_DB_DIR,
    GEMINI_API_KEY,
    EMBEDDING_MODEL,
)


class GeminiEmbeddingFunction(chromadb.EmbeddingFunction):
    """
    Custom ChromaDB embedding function using Google Gemini's text-embedding-004.
    Uses the new google-genai SDK.
    """

    def __init__(self, api_key: str, model_name: str):
        self._client = genai.Client(api_key=api_key)
        self._model_name = model_name.replace("models/", "") if model_name else "text-embedding-004"

    def __call__(self, input: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        embeddings = []
        for text in input:
            result = self._client.models.embed_content(
                model=self._model_name,
                contents=text,
            )
            embeddings.append(result.embeddings[0].values)
        return embeddings


class Retriever:
    """
    ChromaDB-powered document retriever with Google Gemini embeddings.

    """

    COLLECTION_NAME = "university_rules"

    def __init__(self, knowledge_base_dir: str = None, chunk_size: int = None, chunk_overlap: int = None):
        self.knowledge_base_dir = knowledge_base_dir or KNOWLEDGE_BASE_DIR
        self.chunk_size = chunk_size or CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or CHUNK_OVERLAP

        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

        self.embedding_fn = GeminiEmbeddingFunction(
            api_key=GEMINI_API_KEY,
            model_name=EMBEDDING_MODEL,
        )

        self.collection = self.chroma_client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

        self._sync_knowledge_base()

    def _compute_kb_hash(self) -> str:
        """Compute a hash of all knowledge base files and chunk settings to detect changes."""
        hasher = hashlib.md5()
        hasher.update(f"cs_{self.chunk_size}_co_{self.chunk_overlap}".encode("utf-8"))
        txt_files = sorted(glob.glob(os.path.join(self.knowledge_base_dir, "*.txt")))
        for filepath in txt_files:
            with open(filepath, "rb") as f:
                hasher.update(f.read())
        return hasher.hexdigest()

    def _sync_knowledge_base(self):
        """Load and index documents if the collection is empty or knowledge base has changed."""
        kb_hash = self._compute_kb_hash()
        existing_count = self.collection.count()

        metadata = self.collection.metadata or {}
        stored_hash = metadata.get("kb_hash", "")

        if existing_count > 0 and stored_hash == kb_hash:
            print(f"[Retriever] ChromaDB collection '{self.COLLECTION_NAME}' is up to date ({existing_count} chunks).")
            return

        if existing_count > 0:
            print(f"[Retriever] Knowledge base changed. Re-indexing...")
            all_ids = self.collection.get()["ids"]
            if all_ids:
                self.collection.delete(ids=all_ids)

        chunks = self._load_and_chunk()
        if not chunks:
            raise ValueError("No chunks to index. Check knowledge base directory.")

        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            self.collection.add(
                ids=[c["id"] for c in batch],
                documents=[c["text"] for c in batch],
                metadatas=[{"source": c["source"], "start_char": c["start_char"]} for c in batch],
            )

        self.collection.modify(metadata={"kb_hash": kb_hash})

        print(f"[Retriever] Indexed {len(chunks)} chunks into ChromaDB collection '{self.COLLECTION_NAME}'.")

    def _load_and_chunk(self) -> list[dict]:
        """Load all .txt files and split into overlapping chunks."""
        txt_files = glob.glob(os.path.join(self.knowledge_base_dir, "*.txt"))

        if not txt_files:
            raise FileNotFoundError(
                f"No .txt files found in {self.knowledge_base_dir}"
            )

        chunks = []
        chunk_idx = 0

        for filepath in sorted(txt_files):
            filename = os.path.basename(filepath)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            start = 0
            while start < len(content):
                end = start + self.chunk_size
                chunk_text = content[start:end].strip()
                if chunk_text:
                    chunks.append({
                        "id": f"chunk_{chunk_idx:04d}",
                        "text": chunk_text,
                        "source": filename,
                        "start_char": start,
                    })
                    chunk_idx += 1
                start += self.chunk_size - self.chunk_overlap

        print(f"[Retriever] Loaded {len(chunks)} chunks from {len(txt_files)} files.")
        return chunks

    def retrieve(self, query: str, top_k: int = None) -> list[dict]:
        top_k = top_k or TOP_K_RETRIEVAL

        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        retrieved = []
        if results and results["documents"] and results["documents"][0]:
            for doc, metadata, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                similarity = 1 - distance
                if similarity > 0:
                    retrieved.append({
                        "text": doc,
                        "source": metadata.get("source", "unknown"),
                        "score": round(float(similarity), 4),
                    })

        return retrieved
