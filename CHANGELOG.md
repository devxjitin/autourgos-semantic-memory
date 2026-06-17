# Changelog

All notable changes to `autourgos-semantic-memory` are documented here.

---

## [1.0.0] - 2026-06-17

### Added
- Initial release.
- TF-IDF keyword retrieval memory with dual short-term/long-term store.
- Self-contained package — no dependency on `autourgos-core` or sibling packages.
- All base interfaces (`BaseMemory`, `BaseRetriever`, `MemoryMessage`, `Document`) inlined.
- Thread-safe implementation using `threading.RLock`.
- Full type annotations and `py.typed` marker.

