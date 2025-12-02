#!/usr/bin/env python
import numpy as np
from typing import List, Dict, Optional
from .detection import Detector
from .clustering import IssueClustering
from .preprocess import clean_text

class ClusterDetector:
    
    def __init__(self, embedder, index, issue_meta: dict, embeddings: np.ndarray, issue_ids: List[int]):
        
        self.detector = Detector(embedder, index, issue_meta)
        self.embeddings = embeddings
        self.issue_ids = issue_ids
        self.issue_meta = issue_meta
        self.clusterer = None
        
    # performs clustering on all issues [method: ward (default), complete, average, single]
    def cluster(self, method: str = "ward", max_clusters: Optional[int] = None) -> Dict:
        
        self.clusterer = IssueClustering(self.embeddings, self.issue_ids, self.issue_meta)
        stats = self.clusterer.cluster(method=method, max_clusters=max_clusters)
        return stats
    
    # ranked query within cluster
    def query_ranked(self, title: str, body: str, k: int = 10, exclude_issue: int = None) -> List[Dict]:
        return self.detector.query(title, body, k=k, exclude_issue=exclude_issue)
    

    def query_cluster(self, title: str, body: str, exclude_issue: int = None) -> List[Dict]:
        
        # Convert text to embedding
        text = clean_text(title, body)
        qv = self.detector.embedder.encode([text])
        
        # Find nearest issue in index
        D, I = self.detector.index.search(qv, k=1)
        nearest_idx = I[0][0]
        nearest_issue_id = self.detector.index.id_lookup(nearest_idx)
        
        # Get the cluster of the nearest issue
        cluster_id = self.clusterer.cluster_map.get(nearest_issue_id)
        
        # Get all members of the cluster
        cluster_members = self.clusterer.get_cluster_members(cluster_id)
        qv_normalized = qv[0].astype("float32")
        
        # Score each member by similarity to the query
        results = []
        for member in cluster_members:
            issue_id = member["issue_number"]
            
            if exclude_issue is not None and issue_id == exclude_issue:
                continue
            
            # Get embedding for this issue
            issue_idx = self.issue_ids.index(issue_id)
            issue_embedding = self.embeddings[issue_idx]
            
            similarity = float(np.dot(issue_embedding, qv_normalized))
            
            results.append({
                "issue_number": issue_id,
                "score": similarity,
                "meta": self.issue_meta.get(issue_id, {})
            })
        
        # Sort by similarity score (highest first)
        results.sort(key=lambda x: x["score"], reverse=True)
        return results
    
    def get_cluster_members(self, issue_id: int) -> List[Dict]:
        cluster_id = self.clusterer.cluster_map.get(issue_id)
        return self.clusterer.get_cluster_members(cluster_id)
    
    def save_clusters(self, output_file: str):
        self.clusterer.save_clusters(output_file)
