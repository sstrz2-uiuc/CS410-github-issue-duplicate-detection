"""
Main duplicate detection logic.
Uses data.py and embedding.py to find duplicates.
"""

from typing import List, Dict
from . import data
from . import embedding


def find_duplicates(repo_name: str, issue_text: str, similarity_threshold: float = 0.5, top_k: int = 10) -> List[Dict]:
    """
    Find duplicate issues for given text.
    
    Args:
        repo_name: Repository name (e.g. "microsoft/vscode")
        issue_text: Raw issue text (title + body)
        similarity_threshold: Minimum similarity (0-1)
        top_k: Number of results
    
    Returns:
        List of potential duplicates with similarity scores
    """
    _, collection = embedding.load_vector_store(repo_name)
    cleaned_text = data.clean_text(issue_text)
    query_embedding = embedding.embed_text(cleaned_text)
    results = embedding.search_similar(collection, query_embedding, top_k)
    
    duplicates = []
    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        similarity = 1 - distance
        
        if similarity < similarity_threshold:
            continue
        
        duplicates.append({
            "issue_number": results["metadatas"][0][i]["number"],
            "title": results["metadatas"][0][i]["title"],
            "url": results["metadatas"][0][i]["url"],
            "similarity": similarity
        })
    
    return duplicates


def find_duplicates_by_number(repo_name: str, issue_number: int, similarity_threshold: float = 0.5, top_k: int = 10) -> List[Dict]:
    """Find duplicates for an issue already in the database."""
    _, collection = embedding.load_vector_store(repo_name)
    query_embedding = embedding.get_issue_embedding(collection, issue_number)
    results = embedding.search_similar(collection, query_embedding, top_k + 1)
    
    duplicates = []
    for i in range(len(results["ids"][0])):
        result_number = results["metadatas"][0][i]["number"]
        
        if result_number == issue_number:
            continue
        
        distance = results["distances"][0][i]
        similarity = 1 - distance
        
        if similarity < similarity_threshold:
            continue
        
        duplicates.append({
            "issue_number": result_number,
            "title": results["metadatas"][0][i]["title"],
            "url": results["metadatas"][0][i]["url"],
            "similarity": similarity
        })
        
        if len(duplicates) >= top_k:
            break
    
    return duplicates

