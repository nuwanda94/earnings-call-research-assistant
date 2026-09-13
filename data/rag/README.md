# RAG corpus and indices

Versioned retrieval inventories live under `data/rag/corpus_v*/`.
BM25 JSON indices live under `data/rag/indices/bm25/`.
Dense indices (`ids.json` + `embeddings.npy`) live under `data/rag/indices/dense/`.

Large `chunks.jsonl`, `index.json`, and embedding files are gitignored. Rebuild locally:

```bash
python scripts/build_rag_corpus.py
python scripts/build_rag_index.py
python scripts/build_rag_index.py --query "operating margin guidance" --k 3
python scripts/build_rag_index.py --run   # sentence-transformers on Kaggle
```

`manifest.json` records measured **N** (`n_chunks`). Do not invent N in reports.
