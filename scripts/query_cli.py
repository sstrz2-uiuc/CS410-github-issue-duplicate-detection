#!/usr/bin/env python
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dedupe_detector.embed import Embedder
from src.dedupe_detector.index import NumpyCosineIndex as FaissIndex
from src.dedupe_detector.detection import Detector

DATA = Path("data")

def main():
    X = np.load(DATA / "embeddings.npy")
    with open(DATA / "ids.json") as f: ids = json.load(f)
    with open(DATA / "meta.json") as f: meta = json.load(f)

    fx = FaissIndex(X.shape[1])
    fx.add(X, ids)
    det = Detector(Embedder(), fx, meta)

    title = input("Enter title: ").strip()
    body = input("Enter body (optional): ").strip()
    res = det.query(title, body, k=5)
    for r in res:
        issue_num = str(r['issue_number'])
        m = meta.get(issue_num, {})
        print(f"#{r['issue_number']} | {r['score']:.3f} | {m.get('title','')}\n  {m.get('html_url','')}\n")

if __name__ == "__main__":
    main()
