#!/usr/bin/env python3
"""
Configuration for SaFE-Scale Phase 2 — RAG context construction.

Builds a Radiopaedia-derived FAISS index and writes per-question retrieved
context (top-1, top-5, top-10, plus a padded top-1 variant) into
`final_datasets/risk_radiorag_final.jsonl`, which is consumed by Phase 3.
"""

import os
from pathlib import Path

# Base paths — set RIDR_BASE_DIR to override at runtime.
BASE_DIR        = Path(os.environ.get("RIDR_BASE_DIR", "/path/to/workspace"))
PHASE2_DIR      = BASE_DIR / "phase2_rag_context"
DATASETS_DIR    = BASE_DIR / "datasets"
STANDARDIZED_DIR = DATASETS_DIR / "standardized"

# Inputs
RADIOPAEDIA_PATH    = DATASETS_DIR / "radiopedia.jsonl"             # corpus
RISK_RADIORAG_PATH  = STANDARDIZED_DIR / "risk_radiorag_unified.jsonl"  # questions

# Outputs
INDICES_DIR     = PHASE2_DIR / "indices"
EMBEDDINGS_DIR  = PHASE2_DIR / "embeddings"
PROMPTS_DIR     = PHASE2_DIR / "prompts"
LOGS_DIR        = PHASE2_DIR / "logs"

# Chunking
CHUNK_SIZE      = 300    # tokens
CHUNK_OVERLAP   = 50     # tokens
MAX_CHUNK_SIZE  = 512    # hard limit

# Embedding
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
EMBEDDING_DIM   = 1024

# Retrieval
TOP_K_RETRIEVAL = [1, 5, 10]
FAISS_INDEX_TYPE = "Flat"

# Token limits for the long-context conditions (sliced later in Phase 3)
EXTENDED_TOKEN_LIMIT = 1_500_000
TOKENIZER_MODEL      = "gpt-4"

# OpenAI key — read from environment only. NEVER hard-code.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Prompt templates (mirrored in Phase 3 config.py)
ZERO_SHOT_TEMPLATE = """Question: {question}

Options:
{options}

Answer:"""

CONTEXT_TEMPLATE = """Context:
{context}

Question: {question}

Options:
{options}

Answer:"""

# Only the risk_radiorag dataset is used in the published paper.
DATASET_CONFIGS = {
    "risk_radiorag": {
        "question_field":   "question_text",
        "options_field":    "options",
        "context_sources":  ["radiopaedia"],
        "conditions":       ["zero_shot", "top_1", "top_5", "top_10", "padded_top_1"],
    },
}

# Logging
LOG_LEVEL  = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
