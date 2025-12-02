#!/usr/bin/env python
import json
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dedupe_detector.embed import Embedder
from src.dedupe_detector.index import NumpyCosineIndex as FaissIndex
from src.dedupe_detector.cluster_detector import ClusterDetector
from src.dedupe_detector.github_client import GitHubClient

DATA = Path("data")

def load_index():
    # issue IDs
    with open(DATA / "ids.json") as file:
        ids = json.load(file)
    
    # issue metadata
    with open(DATA / "meta.json") as file:
        meta = {int(k): v for k, v in json.load(file).items()}
    
    # embeddings
    X = np.load(DATA / "embeddings.npy")
    
    # Create index
    dim = X.shape[1]
    index = FaissIndex(dim)
    index.add(X, ids)
    
    return index, meta, X, ids

# list of issues and find duplicate pairs (live)
def build_gold(repo: str):
    gh = GitHubClient(repo)
    issues = gh.fetch_issues(state="all", max_pages=50)
    
    num_to_dupes = {}
    for issue in tqdm(issues, desc="parse dupes"):
        issue_num = issue["number"]
        # Get duplicate targets for this issue from comments
        duplicate_targets = gh.get_duplicate_targets(issue_num)

        # Only add to map if there are duplicates found
        if duplicate_targets:
            num_to_dupes[issue_num] = set(duplicate_targets)
    
    # Make the mapping symmetric: if A is a dup of B, then B is also related to A
    symmetric_map = {}
    
    # First pass: add B -> A for all A -> B relationships
    for issue_a, issues_b in num_to_dupes.items():
        for issue_b in issues_b:
            symmetric_map.setdefault(issue_b, set()).add(issue_a)
    
    # Second pass: add the original A -> B relationships
    for issue_a, issues_b in num_to_dupes.items():
        symmetric_map.setdefault(issue_a, set()).update(issues_b)
    
    return issues, symmetric_map

# evaluate the cluster ranking
def evaluate_cluster_ranking(cluster_detector, test_issues, gold_dup_map, ks=(1, 5, 10)):

    # Initialize lists to track recalls and mean reciprocal ranks
    r_at = {k: [] for k in ks}  # recall at k
    mrr_at = {k: [] for k in ks}  # mean reciprocal rank at k
    
    for it in test_issues:
        num = it["number"]
        gold = gold_dup_map.get(num, set())
        
        # Skip if no gold duplicates for this issue
        if not gold:
            continue
        
        try:
            results = cluster_detector.query_cluster(
                it["title"], 
                it.get("body", ""), 
                exclude_issue=num
            )
        except Exception:
            continue
        
        ranked_ids = [r["issue_number"] for r in results]
        
        # Calculate recall and MRR for each k value
        for k in ks:
            found_any = any(x in gold for x in ranked_ids[:k])
            r_at[k].append(1.0 if found_any else 0.0)
            rr = 0.0

            for rank, iid in enumerate(ranked_ids[:k], start=1):
                if iid in gold:
                    rr = 1.0 / rank
                    break

            mrr_at[k].append(rr)
    
    # Calculate the averages
    n_valid = len(r_at[ks[0]]) if ks else 0
    
    summary = {
        "n_queries": n_valid,
    }
    
    # recall
    for k in ks:
        avg_recall = float(np.mean(r_at[k])) if r_at[k] else 0.0
        summary[f"ClusterRecall@{k}"] = avg_recall
    
    # mean reciprocal rank
    for k in ks:
        avg_mrr = float(np.mean(mrr_at[k])) if mrr_at[k] else 0.0
        summary[f"ClusterMRR@{k}"] = avg_mrr
    
    return summary


def main(repo: str, method: str = "ward", max_clusters: int = 100):
    
    index, meta, X, ids = load_index()
    
    issues, gold = build_gold(repo)
    
    # Keep only issues that were indexed
    test = []
    for issue in issues:
        issue_num = issue["number"]
        if issue_num in meta:
            test_issue = {
                "number": issue_num, 
                "title": issue["title"], 
                "body": issue.get("body", "")
            }
            test.append(test_issue)
    
    print(f"Test set size: {len(test)} issues")
    
    # Perform clustering
    print(f"\nClustering (method={method}, max_clusters={max_clusters})...")
    embedder = Embedder()
    cluster_detector = ClusterDetector(embedder, index, meta, X, ids)
    cluster_stats = cluster_detector.cluster(method=method, max_clusters=max_clusters)
    
    print("\n=== Clustering Statistics ===")
    for key, value in cluster_stats.items():
        if key == "cluster_sizes":
            # Only show first 10 cluster sizes
            print(f"{key}: {value[:10]}{'...' if len(value) > 10 else ''}")
        else:
            print(f"{key}: {value}")
    
    # Evaluate cluster ranking
    print("\n=== Cluster-Ranking Evaluation ===")
    cluster_ranking_eval = evaluate_cluster_ranking(cluster_detector, test, gold, ks=(1, 5, 10))
    for key, value in cluster_ranking_eval.items():
        print(f"{key}: {value}")
    
    # Convert stats to JSON
    cluster_stats_serializable = {
        "n_clusters": int(cluster_stats.get("n_clusters")),
        "n_issues": int(cluster_stats.get("n_issues")),
        "cluster_sizes": [],
        "avg_cluster_size": float(cluster_stats.get("avg_cluster_size")),
        "max_cluster_size": int(cluster_stats.get("max_cluster_size")),
        "min_cluster_size": int(cluster_stats.get("min_cluster_size")),
    }
    
    # Convert cluster sizes array
    for size in cluster_stats.get("cluster_sizes", []):
        cluster_stats_serializable["cluster_sizes"].append(int(size))
    
    # Prepare final results dictionary
    results = {
        "clustering_stats": cluster_stats_serializable,
        "cluster_ranking": cluster_ranking_eval,
    }
    
    # Save results to JSON file
    output_file = DATA / "cluster_eval_results.json"
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Evaluate cluster ranking on GitHub duplicate issue detection")
    ap.add_argument("--repo", required=True, help="Repo to evaluate (e.g. microsoft/TypeScript)")
    ap.add_argument("--method", choices=["ward", "complete", "average", "single"], default="ward", help="Clustering linkage method")
    ap.add_argument("--max-clusters", type=int, default=100, help="Max number of clusters")
    
    args = ap.parse_args()
    main(args.repo, method=args.method, max_clusters=args.max_clusters)
