# eval.py
from typing import List, Dict
import numpy as np

def recall_at_k(ranked_ids: List[int], gold_set: set[int], k: int) -> float:
    return 1.0 if any(x in gold_set for x in ranked_ids[:k]) else 0.0

def mrr_at_k(ranked_ids: List[int], gold_set: set[int], k: int) -> float:
    for rank, iid in enumerate(ranked_ids[:k], start=1):
        if iid in gold_set:
            return 1.0 / rank
    return 0.0

def evaluate_pairs(detector, test_issues: List[Dict], gold_duplicates: Dict[int, set[int]], ks=(1,5,10)):
    """
    test_issues: list of dicts with keys {number, title, body}
    gold_duplicates: issue_number -> set(canonical_dupe_targets)
    """
    r_at = {k: [] for k in ks}
    mrr_at = {k: [] for k in ks}

    for it in test_issues:
        num = it["number"]
        gold = gold_duplicates.get(num, set())
        if not gold: continue

        ranked = detector.query(it["title"], it.get("body",""), k=max(ks), exclude_issue=num)
        ranked_ids = [r["issue_number"] for r in ranked]

        for k in ks:
            r_at[k].append(recall_at_k(ranked_ids, gold, k))
            mrr_at[k].append(mrr_at_k(ranked_ids, gold, k))

    summary = {
        "n_queries": len(test_issues),
        **{f"Recall@{k}": float(np.mean(r_at[k])) if r_at[k] else 0.0 for k in ks},
        **{f"MRR@{k}": float(np.mean(mrr_at[k])) if mrr_at[k] else 0.0 for k in ks},
    }
    return summary
