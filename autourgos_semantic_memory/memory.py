"""
memory.py — TF-IDF keyword retrieval memory. Self-contained, zero dependencies.
"""
from __future__ import annotations

import math
import re
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from .base import BaseMemory, BaseRetriever, Document, MemoryMessage


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


# ── Inlined short-term buffer (used as default in KeywordMemory) ───────────────

class _RuntimeShortTermMemory(BaseMemory):
    def __init__(self, max_messages: int = 10) -> None:
        self.max_messages = max_messages
        self._messages: List[MemoryMessage] = []

    def add_message(self, role: str, content: str, timestamp: Optional[datetime] = None) -> MemoryMessage:
        msg = MemoryMessage(role=role, content=content, timestamp=timestamp or datetime.now(timezone.utc))
        self._messages.append(msg)
        if len(self._messages) > self.max_messages:
            self._messages = self._messages[-self.max_messages:]
        return msg

    def add_user_message(self, content: str) -> MemoryMessage:
        return self.add_message("user", content)

    def add_agent_message(self, content: str) -> MemoryMessage:
        return self.add_message("agent", content)

    def add_tool_message(self, tool_name: str, result: str) -> MemoryMessage:
        return self.add_message("tool", f"[{tool_name} returned]: {result}")

    def get_messages(self) -> List[MemoryMessage]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages = []

    def format_for_llm(self, query: Optional[str] = None) -> str:
        if not self._messages:
            return ""
        lines = "\n".join(f"{m.role}: {m.content}" for m in self._messages)
        return f"\n--- Previous Conversation Context ---\n{lines}\n--------------------------------------\n"


# ── KeywordRetriever ───────────────────────────────────────────────────────────

class KeywordRetriever(BaseRetriever):
    """Zero-dependency TF-IDF cosine-similarity retriever."""

    def __init__(self) -> None:
        self.documents: List[Document] = []
        self._doc_tfs: List[Dict[str, float]] = []
        self._df: Dict[str, int] = {}

    def _idf(self, term: str, N: int) -> float:
        return math.log((1 + N) / (1 + self._df.get(term, 0))) + 1.0

    def _tfidf_vector(self, tf: Dict[str, float], N: int) -> Dict[str, float]:
        return {t: w * self._idf(t, N) for t, w in tf.items()}

    @staticmethod
    def _norm(vec: Dict[str, float]) -> float:
        s = sum(v * v for v in vec.values())
        return math.sqrt(s) if s else 0.0

    @staticmethod
    def _cosine(q_vec: Dict[str, float], q_norm: float, d_vec: Dict[str, float], d_norm: float) -> float:
        if not q_norm or not d_norm:
            return 0.0
        dot = sum(q_vec[t] * d_vec[t] for t in q_vec if t in d_vec)
        return dot / (q_norm * d_norm)

    def add_document(self, doc: Document) -> None:
        tokens = tokenize(doc.content)
        raw: Dict[str, int] = {}
        for t in tokens:
            raw[t] = raw.get(t, 0) + 1
        n = len(tokens) or 1
        tf = {t: c / n for t, c in raw.items()}
        for t in tf:
            self._df[t] = self._df.get(t, 0) + 1
        self.documents.append(doc)
        self._doc_tfs.append(tf)

    def add_documents(self, docs: List[Document]) -> None:
        for doc in docs:
            self.add_document(doc)

    def clear(self) -> None:
        self.documents.clear()
        self._doc_tfs.clear()
        self._df.clear()

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        if not self.documents or not query.strip():
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        N = len(self.documents)
        q_raw: Dict[str, int] = {}
        for t in query_tokens:
            q_raw[t] = q_raw.get(t, 0) + 1
        n_q = len(query_tokens)
        q_tf = {t: c / n_q for t, c in q_raw.items()}
        q_vec = self._tfidf_vector(q_tf, N)
        q_norm = self._norm(q_vec)
        scored: List[Tuple[Document, float]] = []
        for doc, d_tf in zip(self.documents, self._doc_tfs):
            d_vec = self._tfidf_vector(d_tf, N)
            d_norm = self._norm(d_vec)
            score = self._cosine(q_vec, q_norm, d_vec, d_norm)
            if score > 0.0:
                scored.append((doc, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [Document(content=d.content, metadata=d.metadata, score=s, source=d.source) for d, s in scored[:top_k]]


# ── KeywordMemory ──────────────────────────────────────────────────────────────

class KeywordMemory(BaseMemory):
    """Dual-store: sliding short-term buffer + TF-IDF keyword retrieval."""

    def __init__(
        self,
        short_term: Optional[BaseMemory] = None,
        retriever: Optional[BaseRetriever] = None,
        top_k: int = 3,
    ) -> None:
        self.short_term = short_term or _RuntimeShortTermMemory(max_messages=10)
        self.retriever  = retriever or KeywordRetriever()
        self.top_k      = top_k

    def _index(self, content: str, role: str, ts: datetime) -> None:
        if hasattr(self.retriever, "add_document"):
            self.retriever.add_document(Document(
                content=content,
                metadata={"role": role, "timestamp": ts.astimezone(timezone.utc).isoformat()},
            ))

    def add_user_message(self, content: str) -> MemoryMessage:
        msg = self.short_term.add_user_message(content)
        self._index(content, "user", msg.timestamp)
        return msg

    def add_agent_message(self, content: str) -> MemoryMessage:
        msg = self.short_term.add_agent_message(content)
        self._index(content, "agent", msg.timestamp)
        return msg

    def add_tool_message(self, tool_name: str, result: str) -> MemoryMessage:
        msg = self.short_term.add_tool_message(tool_name, result)
        self._index(msg.content, "tool", msg.timestamp)
        return msg

    def format_for_llm(self, query: Optional[str] = None) -> str:
        st_context = self.short_term.format_for_llm()
        if not query or not self.retriever:
            return st_context
        recent: set = set()
        get_msgs = getattr(self.short_term, "get_messages", None)
        if callable(get_msgs):
            recent = {
                m.content if hasattr(m, "content") else m.get("content", "")
                for m in get_msgs()
            }
        relevant = [d for d in self.retriever.retrieve(query, top_k=self.top_k) if d.content not in recent]
        if not relevant:
            return st_context
        past = "\n--- Relevant Past Context ---\n"
        for doc in relevant:
            prefix = f"[{doc.metadata['role']}]: " if "role" in doc.metadata else ""
            past += f"{prefix}{doc.content}\n"
        past += "-----------------------------\n\n"
        return past + st_context

    def clear(self) -> None:
        self.short_term.clear()
        if hasattr(self.retriever, "clear"):
            self.retriever.clear()


# back-compat aliases
SimpleSemanticRetriever    = KeywordRetriever
HierarchicalSemanticMemory = KeywordMemory
