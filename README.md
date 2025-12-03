
Python-based semantic duplicate detector for GitHub issues.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# NOTE: GitHub only allows 5000 requests per hour on the free version
export GITHUB_TOKEN="your_token_here"

python scripts/build_index.py --repo Homebrew/brew
python scripts/cluster_issues.py --method ward --max-clusters 100
python scripts/evaluate.py --repo Homebrew/brew
python scripts/evaluate_clusters.py --repo Homebrew/brew
python scripts/query_cli.py
```
