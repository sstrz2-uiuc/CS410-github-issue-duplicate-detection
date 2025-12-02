
Python-based semantic duplicate detector for GitHub issues.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# NOTE: GITHUB only allows 5000 tokens per hour on the free verison
$env:GITHUB_TOKEN = "ADD_GITHUB_TOKEN_HERE"

python scripts/build_index.py --repo microsoft/TypeScript
python scripts/cluster_issues.py --method ward --max-clusters 100
python scripts/evaluate.py --repo microsoft/TypeScript
python scripts/evaluate_clusters.py --repo microsoft/TypeScript
python scripts/query_cli.py
```
