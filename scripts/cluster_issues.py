#!/usr/bin/env python
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dedupe_detector.clustering import IssueClustering

DATA = Path("data")

# generate a md file to show all the cluster information
def generate_cluster_summary(clusterer, output_file: str = "data/cluster_summary.md"):
    
    # Get all clusters and statistics
    all_clusters = clusterer.get_all_clusters()
    stats = clusterer._compute_stats()
    
    # Sort clusters by size (large --> small)
    sorted_clusters = sorted(all_clusters.items(), key=lambda x: len(x[1]), reverse=True)
    top_5_clusters = sorted_clusters[:5]
    
    # Create the .md content
    lines = []
    lines.append("# Cluster Summary Report\n")
    
    # Add all the statistics
    lines.append("## Overall Statistics\n")
    lines.append(f"- **Total Clusters:** {stats['n_clusters']}\n")
    lines.append(f"- **Total Issues:** {stats['n_issues']}\n")
    lines.append(f"- **Average Cluster Size:** {stats['avg_cluster_size']:.2f}\n")
    lines.append(f"- **Max Cluster Size:** {stats['max_cluster_size']}\n")
    lines.append(f"- **Min Cluster Size:** {stats['min_cluster_size']}\n\n")
    lines.append("------------------------------------------------------------------------------------------------------------------\n\n")
    
    lines.append(f"## All {stats['n_clusters']} Clusters\n\n")
    
    # add clusters
    for cluster_id, members in sorted_clusters:
        lines.append(f"### Cluster #{cluster_id} - {len(members)} Issues\n\n")
        
        # table heading
        lines.append("| # | Issue ID | Title | State | Labels |\n")
        lines.append("|---|----------|-------|-------|--------|\n")
        
        # Add all issues in this cluster to the table
        for idx, issue in enumerate(members, 1):
            issue_num = issue["issue_number"]
            meta = issue["meta"]
            full_title = meta.get("title", "N/A")
            state = meta.get("state", "unknown")
            labels = ", ".join(meta.get("labels", [])[:3]) or "none"
            
            lines.append(f"| {idx} | #{issue_num} | {full_title} | {state} | {labels} |\n")
        
        lines.append("\n------------------------------------------------------------------------------------------------------------------\n\n")
    
    with open(output_file, "w", encoding="utf-8") as file:
        file.writelines(lines)
    
    # print top 5 clusters to console for quick view for the user
    print("\n=== Top 5 Clusters ===")
    for cluster_id, members in top_5_clusters:
        print(f"\nCluster #{cluster_id} ({len(members)} issues):")

        for issue in members[:5]:
            meta = issue["meta"]
            title = meta.get("title", "N/A")[:70]
            print(f"  #{issue['issue_number']}: {title}")

        if len(members) > 5:
            print(f"  ... and {len(members) - 5} more issues")


def main(method: str = "ward", max_clusters: int = 100, output: str = "data/clusters.json"):
    
    # embeddings
    X = np.load(DATA / "embeddings.npy")
    
    # issue IDs
    with open(DATA / "ids.json") as file:
        ids = json.load(file)
    
    # metadata
    with open(DATA / "meta.json") as file:
        meta = json.load(file)
    
    # Convert metadata keys to integers (they were stored as strings in JSON)
    meta_int_keys = {}
    for key_str, value in meta.items():
        key_int = int(key_str)
        meta_int_keys[key_int] = value
    meta = meta_int_keys
    
    print(f"Loaded {len(ids)} issues with embeddings of shape {X.shape}")
    
    # run clustering
    print(f"\nPerforming hierarchical clustering (method={method}, max_clusters={max_clusters})...")
    clusterer = IssueClustering(X, ids, meta)
    stats = clusterer.cluster(method=method, max_clusters=max_clusters)
    
    # Print clustering statistics
    print("\n=== Clustering Statistics ===")
    for key, value in stats.items():
        if key == "cluster_sizes":
            print(f"{key}: {value[:10]}{'...' if len(value) > 10 else ''}")
        else:
            print(f"{key}: {value}")
    
    # Save to a JSON file
    clusterer.save_clusters(output)
    summary_file = output.replace(".json", "_summary.md")
    generate_cluster_summary(clusterer, summary_file)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Cluster GitHub issues")
    ap.add_argument("--method", choices=["ward", "complete", "average", "single"], default="ward", help="Linkage method for hierarchical clustering")
    ap.add_argument("--max-clusters", type=int, default=100, help="Max number of clusters")
    ap.add_argument("--output", type=str, default="data/clusters.json", help="Output file for clusters")
    
    args = ap.parse_args()
    main(method=args.method, max_clusters=args.max_clusters, output=args.output)
