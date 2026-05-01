# Local RAG — setup and usage (local-rag.ipynb)

This small guide describes how to set up a local environment to run `local-rag.ipynb`, including ChromaDB vector store, a BM25 lexical retriever, and a cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`).

## 1. Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
# source venv/bin/activate
```

## 2. Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements_local.txt
```

Notes:
- If you have a CUDA-capable GPU and want to run models on GPU, install a compatible `torch` build (see https://pytorch.org). Installing `sentence-transformers` will pull `transformers`.
- `bitsandbytes` is required for 4-bit quantized LLM loading (used in the notebook for Gemma examples). If you plan to load large local LLMs, follow `bitsandbytes` install instructions for your platform.
- `flash-attn` is optional and may speed up attention for supported GPUs.

## 3. Download spaCy model

```bash
python -m spacy download en_core_web_sm
```

## 4. Jupyter / Notebook

Start Jupyter and open the notebook:

```bash
pip install jupyterlab notebook
jupyter lab  # or jupyter notebook
```

Open `local-rag.ipynb` and run cells in order. Key cells added/modified by the project:

- Embedding creation and saving to `text_chunks_and_embeddings_df.csv` (first section).
- New: ChromaDB collection creation + population (added near the end).
- New: BM25 index construction and `hybrid_retrieval(query, top_k, alpha)` — hybrid lexical + vector retrieval.
- New: Cross-encoder reranker using `cross-encoder/ms-marco-MiniLM-L-6-v2` via `sentence_transformers.CrossEncoder` and `rerank_with_cross_encoder(query, candidates, top_k)`.

## 5. Typical workflow

1. Run PDF download & text extraction cells to produce `pages_and_chunks`.
2. Run the embedding cell (uses `sentence-transformers`) to create embeddings for chunks.
3. Run the ChromaDB + BM25 + reranker cell (this will create a Chroma collection and a BM25 index).
4. Use `hybrid_retrieval()` to get candidate passages and `rerank_with_cross_encoder()` to get the top precise passages.

Example usage (from within the notebook):

```python
query = "good foods for protein"
candidates = hybrid_retrieval(query=query, top_k=50, alpha=0.4)
top5 = rerank_with_cross_encoder(query=query, candidates=candidates, top_k=5)
for i, item in enumerate(top5, 1):
    print(i, item['rerank_score'], item['page_num'])
    print(item['text'])
    print('---')
```

## 6. Persistence and Chroma

By default the notebook uses the Chroma default client (in-memory). To persist Chroma to disk, configure the Chroma `Client` to use a local persist directory. Example (not in the notebook):

```python
from chromadb.config import Settings
client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="./chroma_db"))
collection = client.create_collection(name="text_chunks")
```

After adding documents call `client.persist()` to write to disk.

## 7. Tuning

- `alpha` in `hybrid_retrieval()` balances BM25 (lexical) vs. vector scores. Try values like `0.3-0.6`.
- `top_k` for candidates before reranking can be tuned for speed/quality tradeoffs.
- The cross-encoder is CPU/GPU dependent — use GPU where possible for speed.

## 8. Troubleshooting

- If you get CUDA or `bitsandbytes` errors, make sure your `torch` version is compatible and that you installed `bitsandbytes` built for your CUDA.
- If `chromadb` fails to create a collection, try restarting the kernel and ensure no previous in-memory client is blocking resources.
