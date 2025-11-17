# detection.py
import numpy as np
#from .index import FaissIndex
from .index import NumpyCosineIndex as FaissIndex  
from .embed import Embedder
from .preprocess import clean_text

class Detector:
    def __init__(self, embedder: Embedder, index: FaissIndex, issue_id_to_meta: dict[int, dict]):
        self.embedder = embedder
        self.index = index
        self.meta = issue_id_to_meta  # {issue_number: {...}}

    def query(self, title: str, body: str, k: int = 10, exclude_issue: int | None = None):
        text = clean_text(title, body)
        qv = self.embedder.encode([text])
        D, I = self.index.search(qv, k + 5)  # over-fetch, then filter
        results = []
        for d, i in zip(D[0], I[0]):
            if i < 0: continue
            cand_id = self.index.id_lookup(i)
            if exclude_issue is not None and cand_id == exclude_issue:
                continue
            results.append({"issue_number": cand_id, "score": float(d), "meta": self.meta.get(cand_id, {})})
            if len(results) == k: break
        return results
