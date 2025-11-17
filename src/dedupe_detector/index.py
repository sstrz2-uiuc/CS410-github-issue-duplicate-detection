# index.py — pure NumPy exact cosine search (no FAISS/Annoy)
from typing import Tuple, List
import numpy as np

class NumpyCosineIndex:
    """Exact top-k via cosine similarity. Embeddings must be L2-normalized."""
    def __init__(self, dim: int):
        self.dim = dim
        self._X = None
        self.ids: List[int] = []

    def add(self, vecs: np.ndarray, ids: list[int]):
        assert vecs.ndim == 2 and vecs.shape[1] == self.dim
        assert vecs.shape[0] == len(ids)
        self._X = vecs.astype("float32")
        self.ids = list(ids)

    def search(self, q: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        sims = (self._X @ q[0].astype("float32"))  # cosine = dot (normalized)
        if k >= sims.shape[0]:
            top_idx = np.argsort(-sims)
        else:
            top_idx = np.argpartition(-sims, k)[:k]
            top_idx = top_idx[np.argsort(-sims[top_idx])]
        D = sims[top_idx][None, :]
        I = top_idx[None, :].astype(np.int64)
        return D, I

    def id_lookup(self, local_i: int) -> int:
        return self.ids[local_i]
