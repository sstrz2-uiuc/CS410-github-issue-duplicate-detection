# embed.py
from sentence_transformers import SentenceTransformer
import numpy as np

class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = None):
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts, batch_size=64) -> np.ndarray:
        return np.asarray(self.model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True))
