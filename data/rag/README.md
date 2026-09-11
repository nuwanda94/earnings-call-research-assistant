# RAG corpus

Versioned retrieval inventories live under `data/rag/corpus_v*/`.

Large `chunks.jsonl` files are gitignored. Rebuild locally:

```bash
python scripts/build_rag_corpus.py
```

`manifest.json` records measured **N** (`n_chunks`), `n_documents`, source
breakdown, and license notes. Do not invent N in reports — read the manifest.
