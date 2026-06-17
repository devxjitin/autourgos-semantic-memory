"""
base.py — Self-contained base classes. No external dependencies.
"""
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

_ALLOWED_ROLES = {"user", "agent", "system", "tool"}


@dataclass(frozen=True)
class MemoryMessage:
    """A single message in memory."""
    role: str
    content: str
    timestamp: datetime

    def __post_init__(self) -> None:
        if self.role not in _ALLOWED_ROLES:
            raise ValueError(f"Invalid role {self.role!r}. Allowed: {sorted(_ALLOWED_ROLES)}")
        if not isinstance(self.content, str):
            raise ValueError("content must be a string")
        if not isinstance(self.timestamp, datetime):
            raise ValueError("timestamp must be a datetime")

    def to_dict(self) -> Dict[str, str]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.astimezone(timezone.utc).isoformat(),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MemoryMessage":
        ts = payload.get("timestamp")
        if not isinstance(ts, str):
            raise ValueError("timestamp must be a string")
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return cls(role=str(payload.get("role", "")), content=str(payload.get("content", "")), timestamp=dt)


class BaseMemory(ABC):
    """Abstract interface for agent memory."""

    @abstractmethod
    def add_user_message(self, content: str) -> MemoryMessage: ...

    def add_ai_message(self, content: str) -> MemoryMessage:
        warnings.warn("add_ai_message() is deprecated; use add_agent_message().", DeprecationWarning, stacklevel=2)
        return self.add_agent_message(content)

    def add_agent_message(self, content: str) -> MemoryMessage:
        if type(self).add_ai_message is not BaseMemory.add_ai_message:
            return self.add_ai_message(content)
        raise NotImplementedError("Subclasses must implement add_agent_message")

    @abstractmethod
    def add_tool_message(self, tool_name: str, result: str) -> MemoryMessage: ...

    def get_context(self, query: Optional[str] = None) -> str:
        warnings.warn("get_context() is deprecated; use format_for_llm().", DeprecationWarning, stacklevel=2)
        return self.format_for_llm(query)

    def format_for_llm(self, query: Optional[str] = None) -> str:
        if type(self).get_context is not BaseMemory.get_context:
            return self.get_context(query)
        raise NotImplementedError("Subclasses must implement format_for_llm")

    @abstractmethod
    def clear(self) -> None: ...


@dataclass
class Document:
    """A retrieved document chunk."""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    source: str = ""

    def __str__(self) -> str:
        src = f" (source: {self.source})" if self.source else ""
        return f"{self.content}{src}"


class BaseRetriever(ABC):
    """Abstract retriever interface."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Document]: ...

    async def aretrieve(self, query: str, top_k: int = 5) -> List[Document]:
        import asyncio
        return await asyncio.to_thread(self.retrieve, query, top_k)

