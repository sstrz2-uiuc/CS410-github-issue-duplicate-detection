#!/usr/bin/env python
import json, sys
from pathlib import Path
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dedupe_detector.embed import Embedder
from src.dedupe_detector.index import NumpyCosineIndex as FaissIndex
from src.dedupe_detector.detection import Detector
from src.dedupe_detector.eval import evaluate_pairs
from src.dedupe_detector.github_client import GitHubClient

DATA = Path("data")

def load_index():
    with open(DATA / "ids.json") as f:
        ids = json.load(f)
    with open(DATA / "meta.json") as f:
        meta = json.load(f)
    X = np.load(DATA / "embeddings.npy")
    dim = X.shape[1]
    index = FaissIndex(dim)
    index.add(X, ids)
    return index, meta




def build_gold(repo: str):
    gh = GitHubClient(repo)
    issues = gh.fetch_issues(state="all", max_pages=60)
    num_to_dupes = {}
    for it in tqdm(issues, desc="parse dupes"):
        n = it["number"]
        targets = gh.get_duplicate_targets(n)
        if targets:
            num_to_dupes[n] = set(targets)
    # Also ensure symmetry: if A dup B, treat B’s gold set to include A
    sym = {}
    for a, bs in num_to_dupes.items():
        for b in bs:
            sym.setdefault(b, set()).add(a)
        sym.setdefault(a, set()).update(bs)
    return issues, sym

def main(repo: str, kmax=10):
    index, meta = load_index()
    emb = Embedder()
    det = Detector(emb, index, meta)

    issues, gold = build_gold(repo)
    # Keep only issues that exist in meta (indexed)
    test = [ {"number": it["number"], "title": it["title"], "body": it.get("body","")}
             for it in issues if str(it["number"]) in meta ]

    res = evaluate_pairs(det, test, gold, ks=(1,5,10))
    print("\n=== Evaluation Summary ===")
    for k, v in res.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    args = ap.parse_args()
    main(args.repo)
