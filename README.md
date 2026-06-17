# autourgos-semantic-memory

TF-IDF keyword retrieval memory for [Autourgos](https://github.com/devxjitin) agents.

Combines a short-term message buffer with a TF-IDF keyword index. When the agent asks for context, relevant past messages are retrieved by keyword similarity and prepended — even if they fell outside the short-term window.

Zero external dependencies. No embeddings, no vector database required.

---

## Install

```bash
pip install autourgos-semantic-memory
```

---

## Classes

### KeywordMemory

Dual-store memory: recent messages in a ring buffer + all messages indexed for TF-IDF retrieval.

```python
from autourgos_semantic_memory import KeywordMemory
from autourgos_react_agent import ReactAgent

memory = KeywordMemory(top_k=3)  # surface top 3 relevant past messages
agent  = ReactAgent(llm=my_llm, memory=memory)

agent.invoke("The server is running on port 8080")
agent.invoke("The database password is hunter2")
# ... many more messages ...
agent.invoke("What port is the server on?")
# → retrieves the port message from the keyword index even if it left the buffer
```

### KeywordRetriever

Standalone TF-IDF retriever. Plug it into `KeywordMemory` or use directly with your own memory:

```python
from autourgos_semantic_memory import KeywordRetriever
from autourgos_memory import Document

retriever = KeywordRetriever()
retriever.add_document(Document(content="Paris is the capital of France.", source="wiki"))
retriever.add_document(Document(content="Berlin is the capital of Germany.", source="wiki"))

results = retriever.retrieve("What is the capital of France?", top_k=1)
print(results[0].content)
# → "Paris is the capital of France."
```

### Custom short-term store

```python
from autourgos_semantic_memory import KeywordMemory
from autourgos_local_memory import SQLiteMemory

# Use SQLite as the short-term buffer (survives restarts)
memory = KeywordMemory(
    short_term=SQLiteMemory(db_path="./data/agent.db"),
    top_k=5,
)
```

---

## Parameters

### KeywordMemory

| Parameter | Type | Default | Description |
|---|---|---|---|
| `short_term` | BaseMemory | `RuntimeShortTermMemory(10)` | Short-term buffer shown in full. |
| `retriever` | BaseRetriever | `KeywordRetriever()` | Retriever for past context. |
| `top_k` | int | `3` | Max relevant past messages to surface. |

---

## How TF-IDF works here

- Every message is tokenized (lowercase alphanumeric) and indexed.
- IDF weights are computed at query time — so scores stay accurate as more messages are added.
- Cosine similarity ranks results. Only messages with score > 0 are returned.
- Back-compat aliases: `SimpleSemanticRetriever = KeywordRetriever`, `HierarchicalSemanticMemory = KeywordMemory`.

---

## Links

- PyPI: https://pypi.org/project/autourgos-semantic-memory/
- GitHub: https://github.com/devxjitin/autourgos-semantic-memory
- Issues: https://github.com/devxjitin/autourgos-semantic-memory/issues

---

## License

MIT — see [LICENSE](LICENSE)
