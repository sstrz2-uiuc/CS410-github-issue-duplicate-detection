"""
One-time setup: build the vector database.
Run this once to index all issues.
"""

from src import data, embedding

REPO_NAME = "microsoft/vscode"
FROM_FILE = "data/5k_issues.json"

issues = data.fetch_issues(from_file=FROM_FILE)
texts = data.preprocess_batch(issues)
embeddings = embedding.embed_batch(texts)
client, collection = embedding.create_vector_store(REPO_NAME)
embedding.add_to_vector_store(collection, embeddings, issues)
