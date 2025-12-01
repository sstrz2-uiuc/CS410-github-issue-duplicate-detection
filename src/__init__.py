# New modular structure uses src/dedupe_detector/
# Old modules (embedding.py, detector.py, etc.) are deprecated
# Import from src.dedupe_detector instead

from . import dedupe_detector

__all__ = ["dedupe_detector"]

