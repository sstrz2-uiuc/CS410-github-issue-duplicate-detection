# github_client.py
from __future__ import annotations
import os, re, time
from typing import Dict, List, Optional
import requests

GITHUB_API = "https://api.github.com"
DUP_REF_RE = re.compile(r"(?:duplicate of|dup of|dupe of)\s*#(\d+)", re.I)

class GitHubClient:
    def __init__(self, repo: str, token: Optional[str] = None):
        self.repo = repo
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token or os.getenv('GITHUB_TOKEN','')}"
        })

    def _get(self, path, params=None):
        url = f"{GITHUB_API}/repos/{self.repo}{path}"
        r = self.session.get(url, params=params)
        if r.status_code == 403 and "rate limit" in r.text.lower():
            reset = int(r.headers.get("X-RateLimit-Reset", "0"))
            sleep_s = max(0, reset - int(time.time()) + 1)
            time.sleep(sleep_s)
            r = self.session.get(url, params=params)
        r.raise_for_status()
        return r.json()

    def fetch_issues(self, state="all", max_pages=50) -> List[Dict]:
        out, page = [], 1
        while page <= max_pages:
            items = self._get("/issues", {
                "state": state, "per_page": 100, "page": page
            })
            # Exclude PRs
            items = [it for it in items if "pull_request" not in it]
            if not items: break
            out.extend(items); page += 1
        return out

    def fetch_comments(self, issue_number: int) -> List[Dict]:
        out, page = [], 1
        while True:
            items = self._get(f"/issues/{issue_number}/comments", {"per_page":100,"page":page})
            if not items: break
            out.extend(items); page += 1
        return out

    def get_duplicate_targets(self, issue_number: int) -> List[int]:
        """
        Heuristic: parse comments for 'Duplicate of #NNNN'.
        You can extend with timeline events when you enable the preview header:
        Accept: application/vnd.github.mockingbird-preview+json
        """
        ds = []
        for c in self.fetch_comments(issue_number):
            m = DUP_REF_RE.search(c.get("body", "") or "")
            if m:
                ds.append(int(m.group(1)))
        return list(set(ds))
