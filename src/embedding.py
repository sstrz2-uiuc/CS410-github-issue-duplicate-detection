"""
Embedding generation and vector storage.
Convert text to embeddings, store in ChromaDB.
"""

import numpy as np
import chromadb
from typing import List, Dict
from sentence_transformers import SentenceTransformer


# Global model (load once)
_model = None

def get_model(model_name: str = "all-MiniLM-L6-v2"):
    """Get or create sentence transformer model."""
    global _model
    if _model is None:
        _model = SentenceTransformer(model_name)
    return _model


def embed_text(text: str, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Generate embedding for single text."""
    model = get_model(model_name)
    if not text or not text.strip():
        return np.zeros(model.get_sentence_embedding_dimension())
    return model.encode(text, convert_to_numpy=True)


def embed_batch(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Generate embeddings for batch of texts."""
    model = get_model(model_name)
    processed_texts = [text if text and text.strip() else " " for text in texts]
    return model.encode(processed_texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True)


def create_vector_store(repo_name: str):
    """Initialize ChromaDB vector store (collection name = repo name)."""
    collection_name = repo_name.replace("/", "_")
    client = chromadb.PersistentClient(path="./chroma_db")
    
    try:
        client.delete_collection(name=collection_name)
    except:
        pass
    
    collection = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
    return client, collection


def load_vector_store(repo_name: str):
    """Load existing ChromaDB vector store (collection name = repo name)."""
    collection_name = repo_name.replace("/", "_")
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(name=collection_name)
    return client, collection


def add_to_vector_store(collection, embeddings: np.ndarray, issues: List[Dict]):
    """Add issues to vector store (skips duplicates)."""
    seen_ids = set()
    unique_issues = []
    unique_embeddings = []
    
    for i, issue in enumerate(issues):
        issue_id = f"issue_{issue['number']}"
        if issue_id not in seen_ids:
            seen_ids.add(issue_id)
            unique_issues.append(issue)
            unique_embeddings.append(embeddings[i])
    
    if not unique_issues:
        return
    
    unique_embeddings = np.array(unique_embeddings)
    batch_size = 100
    n_issues = len(unique_issues)
    
    for i in range(0, n_issues, batch_size):
        batch_end = min(i + batch_size, n_issues)
        batch_issues = unique_issues[i:batch_end]
        batch_embeddings = unique_embeddings[i:batch_end]
        
        ids = [f"issue_{issue['number']}" for issue in batch_issues]
        embeddings_list = batch_embeddings.tolist()
        metadatas = [{"number": issue["number"], "title": issue["title"][:500], "url": issue["url"], "state": issue["state"]} for issue in batch_issues]
        documents = [issue["title"] for issue in batch_issues]
        
        collection.add(ids=ids, embeddings=embeddings_list, metadatas=metadatas, documents=documents)


def search_similar(collection, query_embedding: np.ndarray, top_k: int = 10) -> Dict:
    """Search for similar issues."""
    results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=top_k)
    return results


def get_issue_embedding(collection, issue_number: int) -> np.ndarray:
    """Get embedding for specific issue."""
    result = collection.get(ids=[f"issue_{issue_number}"], include=["embeddings"])
    if not result["ids"]:
        raise ValueError(f"Issue #{issue_number} not found")
    return np.array(result["embeddings"][0])


def get_all_embeddings(collection) -> tuple:
    """Get all embeddings from vector store (for clustering)."""
    all_data = collection.get(include=["embeddings", "metadatas"])
    embeddings = np.array(all_data["embeddings"])
    issue_numbers = [meta["number"] for meta in all_data["metadatas"]]
    return embeddings, issue_numbers

