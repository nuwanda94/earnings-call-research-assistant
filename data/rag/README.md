# RAG corpus and indices

Versioned retrieval inventories live under `data/rag/corpus_v*/`.
BM25 JSON indices live under `data/rag/indices/bm25/`.

Large `chunks.jsonl` and `index.json` files are gitignored. Rebuild locally:

```bash
python scripts/build_rag_corpus.py
python scripts/build_rag_index.py
python scripts/build_rag_index.py --query "operating margin guidance" --k 3
```

`manifest.json` records measured **N** (`n_chunks`). Do not invent N in reports.
