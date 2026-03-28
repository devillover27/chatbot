import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List

class VectorStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.chunks = []
        self.embeddings = None

    def build(self, chunks: List[str]):
        if not chunks:
            return
        self.chunks = chunks
        # Encode all chunks and normalize for cosine similarity via dot product
        raw_embeddings = self.model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
        # L2 normalize
        self.embeddings = raw_embeddings / np.linalg.norm(raw_embeddings, axis=1, keepdims=True)

    def search(self, query: str, top_k: int = 8) -> List[str]:
        if self.embeddings is None:
            return []
        
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # Dot product for cosine similarity on normalized vectors
        similarities = np.dot(self.embeddings, query_embedding.T).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [self.chunks[i] for i in top_indices]

# Singleton instance
vector_store = VectorStore()
