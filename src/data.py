"""
Data fetching and preprocessing.
Fetch issues from GitHub API or file, clean text.
"""

import json
import re
import requests
from typing import List, Dict, Optional


def fetch_issues(token: Optional[str] = None, repo_name: Optional[str] = None, limit: Optional[int] = None, from_file: Optional[str] = None) -> List[Dict]:
    """Fetch issues from API or file."""
    if from_file:
        with open(from_file, 'r') as f:
            issues = json.load(f)
        if limit:
            issues = issues[:limit]
        return issues
    
    if not token:
        raise ValueError("token required for API fetching")
    if not repo_name:
        raise ValueError("repo_name required")
    
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    issues = []
    page = 1
    per_page = 100
    
    while not limit or len(issues) < limit:
        url = f"https://api.github.com/repos/{repo_name}/issues"
        params = {"state": "all", "per_page": per_page, "page": page}
        
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code != 200:
            break
        
        batch = response.json()
        
        if not batch:
            break
        
        for item in batch:
            if "pull_request" in item:
                continue
            
            issue_data = {
                "number": item["number"],
                "title": item["title"],
                "body": item["body"] or "",
                "state": item["state"],
                "labels": [label["name"] for label in item["labels"]],
                "url": item["html_url"],
            }
            
            issues.append(issue_data)
            
            if limit and len(issues) >= limit:
                break
        
        page += 1
    
    return issues


def clean_text(text: str) -> str:
    """Clean issue text (remove code, markdown, etc.)."""
    if not text:
        return ""
    
    text = re.sub(r'```[\s\S]*?```', ' ', text)
    text = re.sub(r'`[^`]+`', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[#*_~]', ' ', text)
    
    return text.strip()


def preprocess_issue(issue: Dict) -> str:
    """Combine title + body and clean."""
    title = issue.get("title", "")
    body = issue.get("body", "")
    text = f"{title}. {body}"
    return clean_text(text)


def preprocess_batch(issues: List[Dict]) -> List[str]:
    """Preprocess multiple issues."""
    return [preprocess_issue(issue) for issue in issues]

