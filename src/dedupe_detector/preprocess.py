# preprocess.py
import re

CODE_BLOCK = re.compile(r"```.*?```", re.S)
INLINE_CODE = re.compile(r"`([^`]+)`")
URL = re.compile(r"https?://\S+")

def clean_text(title: str, body: str) -> str:
    body = body or ""
    body = CODE_BLOCK.sub(" ", body)
    body = INLINE_CODE.sub(r"\1", body)
    body = URL.sub(" ", body)
    text = f"{title.strip()} \n {body.strip()}"
    text = re.sub(r"\s+", " ", text)
    return text[:2000]  # keep reasonable length
