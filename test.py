"""
Test duplicate detection.
Run setup_vectors.py first!
"""

from src import detector
import json

REPO_NAME = "microsoft/vscode"
INPUT_FILE = "data/5k_issues.json"
MAX_ISSUES = 200
OUTPUT_FILE = "test_results.txt"
SIMILARITY_THRESHOLD = 0.5

def main():
    with open(INPUT_FILE, 'r') as f:
        issues = json.load(f)[:MAX_ISSUES]
    
    with open(OUTPUT_FILE, 'w') as f:
        for issue in issues:
            try:
                duplicates = detector.find_duplicates_by_number(
                    repo_name=REPO_NAME,
                    issue_number=issue['number'], 
                    top_k=1, 
                    similarity_threshold=SIMILARITY_THRESHOLD
                )
                
                if duplicates:
                    similar = duplicates[0]
                    f.write(f"{issue['number']} : {similar['issue_number']} ({similar['similarity']:.3f})\n")
                else:
                    f.write(f"{issue['number']} : no_duplicates\n")
            except Exception as e:
                f.write(f"{issue['number']} : error\n")
    
    print(f"Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
