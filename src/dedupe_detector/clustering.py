#!/usr/bin/env python
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from typing import List, Dict, Optional
import json

class IssueClustering:

    def __init__(self, embeddings: np.ndarray, issue_ids: List[int], issue_meta: Dict[int, dict]):
        
        self.embeddings = embeddings
        self.issue_ids = np.array(issue_ids)
        self.issue_meta = issue_meta
        self.linkage_matrix = None
        self.clusters = None
        self.cluster_map = None  # maps issue id to cluster id

    # perform hierarchical clustering [method: ward (default), complete, average, single] [distance_metric: euclidean (default) or cosine]
    def cluster(self, method: str = "ward", distance_metric: str = "euclidean", max_clusters: Optional[int] = None) -> Dict:
        DEFAULT_MAX = 100
        
        distances = pdist(self.embeddings, metric=distance_metric)
        
        # Perform hierarchical clustering
        print(f"Performing hierarchical clustering with " + method)
        self.linkage_matrix = linkage(distances, method=method)
        
        #check if max_clusters is provided
        if max_clusters is not None:
            cluster_labels = fcluster(self.linkage_matrix, max_clusters, criterion="maxclust")
        else:
            cluster_labels = fcluster(self.linkage_matrix, DEFAULT_MAX, criterion="maxclust")
        
        # map issue ids to cluster ids
        self.cluster_map = {issue_id: cluster_id for issue_id, cluster_id in zip(self.issue_ids, cluster_labels)}
        self.clusters = cluster_labels
        
        stats = self._compute_stats()
        return stats

    # get some fun stats about clustering results
    def _compute_stats(self) -> Dict:
        unique_clusters = np.unique(self.clusters)
        cluster_sizes = [np.sum(self.clusters == c) for c in unique_clusters]
        
        return {
            "n_clusters": len(unique_clusters),
            "n_issues": len(self.issue_ids),
            "cluster_sizes": sorted(cluster_sizes, reverse=True),
            "avg_cluster_size": float(np.mean(cluster_sizes)),
            "max_cluster_size": int(np.max(cluster_sizes)),
            "min_cluster_size": int(np.min(cluster_sizes)),
        }

    def get_cluster_members(self, cluster_id: int) -> List[Dict]:
        
        member_ids = []
        for issue_id, cluster_cid in self.cluster_map.items():
            if cluster_cid == cluster_id:
                member_ids.append(issue_id)
        
        members = []
        for issue_id in member_ids:
            issue_info = {
                "issue_number": issue_id,
                "meta": self.issue_meta.get(issue_id, {})
            }
            members.append(issue_info)
        
        return members

    # get the clustering in a dict
    def get_all_clusters(self) -> Dict[int, List[Dict]]:
        
        # cluster_id ---> list of issues
        clusters_dict = {}
        
        # Iterate through all issue-cluster assignments
        for issue_id, cluster_id in self.cluster_map.items():
            # Create cluster entry if it doesn't exist
            if cluster_id not in clusters_dict:
                clusters_dict[cluster_id] = []
            
            # Build issue info and add to cluster
            issue_info = {
                "issue_number": issue_id,
                "meta": self.issue_meta.get(issue_id, {})
            }
            clusters_dict[cluster_id].append(issue_info)
        
        return clusters_dict

    def get_cluster_for_issue(self, issue_id: int) -> Optional[int]:
        
        cluster_id = self.cluster_map.get(issue_id)
        return cluster_id

    def save_clusters(self, output_file: str):
        
        # Convert cluster assignments to JSON
        cluster_assignments = {}
        for issue_id, cluster_id in self.cluster_map.items():
            key = str(int(issue_id))
            value = int(cluster_id)
            cluster_assignments[key] = value

        # Convert clusters to JSON
        raw_clusters = self.get_all_clusters()
        clusters_json = {}
        
        for cluster_id, members in raw_clusters.items():
            # Convert cluster ID to string for JSON keys
            cluster_key = str(int(cluster_id))
            clusters_json[cluster_key] = []
            
            # Convert each member to JSON
            for member in members:
                issue_number = int(member.get("issue_number"))
                meta_info = member.get("meta", {})
                
                member_copy = {
                    "issue_number": issue_number,
                    "meta": meta_info
                }
                clusters_json[cluster_key].append(member_copy)

        stats_raw = self._compute_stats()
        
        stats = {
            "n_clusters": int(stats_raw.get("n_clusters")),
            "n_issues": int(stats_raw.get("n_issues")),
            "cluster_sizes": [],
            "avg_cluster_size": float(stats_raw.get("avg_cluster_size")),
            "max_cluster_size": int(stats_raw.get("max_cluster_size")),
            "min_cluster_size": int(stats_raw.get("min_cluster_size")),
        }
        
        for size in stats_raw.get("cluster_sizes", []):
            stats["cluster_sizes"].append(int(size))

        clusters_data = {
            "cluster_assignments": cluster_assignments,
            "clusters": clusters_json,
            "stats": stats,
        }

        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(clusters_data, file, indent=2)

        print(f"Clusters saved to {output_file}")

    def load_clusters(self, input_file: str):

        with open(input_file, "r") as file:
            data = json.load(file)
        
        cluster_assignments = data["cluster_assignments"]
        self.cluster_map = {}
        for issue_id_str, cluster_id in cluster_assignments.items():
            issue_id = int(issue_id_str)
            self.cluster_map[issue_id] = cluster_id
        
        clusters_list = []
        for issue_id in self.issue_ids:
            cluster_id = self.cluster_map[issue_id]
            clusters_list.append(cluster_id)
        
        self.clusters = np.array(clusters_list)
        
        print(f"Clusters loaded from {input_file}")
