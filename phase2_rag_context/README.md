# Phase 2 — RAG context construction

Builds the retrieval index from the Radiopaedia corpus and writes the
question file consumed by Phase 3, with retrieved context packed into
multiple conditions.

## Pipeline

```
datasets/radiopedia.jsonl
  │
  ▼ 01_chunk_radiopaedia.py    300-token chunks, 50-token overlap
  ▼ 02_build_embeddings.py     BGE-large-en-v1.5, L2-normalised
  ▼ 03_build_faiss_index.py    Flat inner-product index
  ▼ 04_retrieve_context.py     top-10 chunks per question
  ▼ 05_add_context_to_datasets.py
                               assemble final per-question prompts:
                                  zero_shot, top_1, top_5, top_10,
                                  padded_top_1, extended_150k
                               (Phase 3 slices extended_150k into
                               context_32k / context_100k / context_max
                               using each model's own tokenizer)
  │
  ▼
phase2_rag_context/final_datasets/risk_radiorag_final.jsonl
```

## Inputs / outputs

| Path                                                          | Role         |
| ------------------------------------------------------------- | ------------ |
| `datasets/radiopedia.jsonl`                                   | corpus       |
| `datasets/standardized/risk_radiorag_unified.jsonl`           | questions    |
| `phase2_rag_context/embeddings/`                              | chunks + vectors |
| `phase2_rag_context/indices/`                                 | FAISS index  |
| `phase2_rag_context/retrievals/`                              | per-question top-k |
| `phase2_rag_context/final_datasets/risk_radiorag_final.jsonl` | → Phase 3    |

## Setup

```bash
export WORKSPACE=/abs/path/to/this/checkout
apptainer build $WORKSPACE/environment/rag.sif $WORKSPACE/environment/rag.def
```

## Run

```bash
SIF=$WORKSPACE/environment/rag.sif
EXEC="apptainer exec --bind $WORKSPACE:$WORKSPACE"
GPU="apptainer exec --nv --bind $WORKSPACE:$WORKSPACE"

$EXEC $SIF python3 phase2_rag_context/scripts/01_chunk_radiopaedia.py
$GPU  $SIF python3 phase2_rag_context/scripts/02_build_embeddings.py
$EXEC $SIF python3 phase2_rag_context/scripts/03_build_faiss_index.py
$GPU  $SIF python3 phase2_rag_context/scripts/04_retrieve_context.py
$EXEC $SIF python3 phase2_rag_context/scripts/05_add_context_to_datasets.py
```

## Parameters (`config.py`)

| Parameter           | Value                       |
| ------------------- | --------------------------- |
| Chunk size          | 300 tokens (50 overlap)     |
| Embedding model     | `BAAI/bge-large-en-v1.5` (1024-d) |
| Retrieval top-k     | 1, 5, 10                    |
| FAISS index         | `Flat` (inner product)      |
| Extended-context cap| 150 k tokens (sliced later in Phase 3) |
| Tokenizer for counts| `gpt-4` (tiktoken)          |

## Per-question output schema

```json
{
  "question_id": "...",
  "question_text": "...",
  "options": {...},
  "correct_answer": "...",
  "contexts": {
    "zero_shot":       { "prompt": "...", "tokens": 0,  "truncated": false },
    "top_1":           { "prompt": "...", "chunks": [...], "tokens":  400, "truncated": false },
    "top_5":           { "prompt": "...", "chunks": [...], "tokens": 2000, "truncated": false },
    "top_10":          { "prompt": "...", "chunks": [...], "tokens": 4000, "truncated": false },
    "padded_top_1":    { "prompt": "...", "chunks": [...], "tokens": 4000, "truncated": false },
    "extended_150k":   { "prompt": "...", "chunks": [...], "tokens": 150000 }
  }
}
```

## Properties enforced

- **Superset.** `top_5 ⊃ top_1`, `top_10 ⊃ top_5` (chunk-rank-stable).
- **Padded-top-1.** Same chunk content as top-1, repeated until matching the
  top-10 token count — used as a causal control for context length.
- **Truncation logged.** Any prompt that hits the token cap is recorded.
