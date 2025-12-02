#!/usr/bin/env python
import json, os, sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dedupe_detector.github_client import GitHubClient
from src.dedupe_detector.preprocess import clean_text
from src.dedupe_detector.embed import Embedder
#from src.dedupe_detector.index import FaissIndex
from src.dedupe_detector.index import NumpyCosineIndex as FaissIndex

DATA = Path("data")

def main(repo: str, state="all"):
    DATA.mkdir(exist_ok=True)
    gh = GitHubClient(repo)
    issues = gh.fetch_issues(state=state, max_pages=60)
    # Cache raw
    (DATA / "issues_raw.json").write_text(json.dumps(issues, indent=2))

    # Minimal fields
    rows, ids = [], []
    for it in issues:
        num = it["number"]
        ids.append(num)
        rows.append({
            "number": num,
            "title": it.get("title",""),
            "body": it.get("body","") or "",
            "state": it.get("state",""),
            "labels": [l["name"] for l in it.get("labels",[])],
            "html_url": it.get("html_url","")
        })

    # Texts
    texts = [clean_text(r["title"], r["body"]) for r in rows]

    # Embeddings
    emb = Embedder()
    X = emb.encode(texts)
    dim = X.shape[1]

    # FAISS
    fx = FaissIndex(dim)
    fx.add(X, [r["number"] for r in rows])

    # Persist
    npy = X.astype("float32")
    import numpy as np, pickle
    np.save(DATA / "embeddings.npy", npy)
    with open(DATA / "meta.json","w") as f:
        json.dump({r["number"]: r for r in rows}, f)
    #faiss.write_index(fx.index, str(DATA / "faiss.index"))
    with open(DATA / "ids.json","w") as f:
        json.dump(fx.ids, f)
    print(f"Indexed {len(rows)} issues.")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="e.g. microsoft/vscode")
    ap.add_argument("--state", default="all")
    args = ap.parse_args()
    main(args.repo, args.state)
#!/usr/bin/env python
