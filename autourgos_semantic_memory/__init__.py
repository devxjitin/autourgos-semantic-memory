"""
autourgos-semantic-memory — TF-IDF keyword retrieval memory for Autourgos agents.

    from autourgos_semantic_memory import KeywordRetriever, KeywordMemory
"""
from .memory import (
    tokenize,
    KeywordRetriever,
    KeywordMemory,
    SimpleSemanticRetriever,
    HierarchicalSemanticMemory,
)

try:
    from importlib.metadata import version as _v
    __version__ = _v("autourgos-semantic-memory")
except Exception:
    __version__ = "1.0.1"

__all__ = [
    "tokenize",
    "KeywordRetriever",
    "KeywordMemory",
    "SimpleSemanticRetriever",
    "HierarchicalSemanticMemory",
]
