
Python-based semantic duplicate detector for GitHub issues.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python scripts/build_index.py --repo microsoft/TypeScript
python scripts/evaluate.py --repo microsoft/TypeScript
python scripts/query_cli.py

