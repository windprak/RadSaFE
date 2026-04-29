#!/usr/bin/env python3
"""
Phase 3 — Model Inference Configuration
"""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(os.environ.get("WORKSPACE", "/path/to/workspace"))
PHASE2_DIR      = BASE_DIR / "phase2_rag_context"
PHASE3_DIR      = BASE_DIR / "phase3_inference"
FINAL_DATASETS  = PHASE2_DIR / "final_datasets"
RESULTS_DIR     = PHASE3_DIR / "results"
LOGS_DIR        = PHASE3_DIR / "logs"

MODEL_HUB       = Path("${MODELS_DIR}")

# Apptainer images
VLLM_SIF        = MODEL_HUB / "vllm-openai_gptoss.sif"
RAG_SIF         = PHASE2_DIR / "rag_environment.sif"   # has transformers + openai

# ── Model Definitions ──────────────────────────────────────────────────────────
# All models run with tensor_parallel_size=4 on H100 partition.
# context_limits: max context tokens to pass (reserves room for q+options in window)
#   "32k"  → 30_000 ctx tokens  (for 32k window models)
#   "100k" → 96_000 ctx tokens  (for 128k window models)
MODELS = [
    # ── Qwen2.5 family ─────────────────────────────────────────────────────────
    {
        "index":        0,
        "name":         "Qwen2.5-0.5B-Instruct",
        "path":         str(MODEL_HUB / "Qwen2.5-0.5B-Instruct"),
        "max_model_len": 32768,
        "context_limits": {"32k": 30_000, "max": 25_624},
        "vision":       False,
        "tp":           1,  # 14 attention heads not divisible by 4
        "debug":        True,   # start here for smoke-test
    },
    {
        "index":        1,
        "name":         "Qwen2.5-1.5B-Instruct",
        "path":         str(MODEL_HUB / "Qwen2.5-1.5B-Instruct"),
        "max_model_len": 32768,
        "context_limits": {"32k": 30_000, "max": 25_624},
        "vision":       False,
        "tp":           1,  # small model, tp=1 sufficient
    },
    {
        "index":        2,
        "name":         "Qwen2.5-3B-Instruct",
        "path":         str(MODEL_HUB / "Qwen2.5-3B-Instruct"),
        "max_model_len": 32768,
        "context_limits": {"32k": 30_000, "max": 25_624},
        "vision":       False,
        "tp":           1,  # small model, tp=1 sufficient
    },
    {
        "index":        3,
        "name":         "Qwen2.5-7B-Instruct",
        "path":         str(MODEL_HUB / "Qwen2.5-7B-Instruct"),
        "max_model_len": 32768,
        "context_limits": {"32k": 30_000, "max": 25_624},
        "vision":       False,
        "tp":           4,
    },
    {
        "index":        4,
        "name":         "Qwen2.5-14B-Instruct",
        "path":         str(MODEL_HUB / "Qwen2.5-14B-Instruct"),
        "max_model_len": 32768,
        "context_limits": {"32k": 30_000, "max": 25_624},
        "vision":       False,
        "tp":           4,
    },
    {
        "index":        5,
        "name":         "Qwen2.5-32B-Instruct",
        "path":         str(MODEL_HUB / "Qwen2.5-32B-Instruct"),
        "max_model_len": 32768,
        "context_limits": {"32k": 30_000, "max": 25_624},
        "vision":       False,
        "tp":           4,
    },
    # ── Gemma-3 family (vision-capable) ────────────────────────────────────────
    {
        "index":        6,
        "name":         "gemma-3-4b-it",
        "path":         str(MODEL_HUB / "gemma-3-4b-it"),
        "max_model_len": 100_000,
        "context_limits": {"32k": 30_000, "max": 92_856},
        "vision":       True,
        "tp":           4,
    },
    {
        "index":        7,
        "name":         "gemma-3-12b-it",
        "path":         str(MODEL_HUB / "gemma-3-12b-it"),
        "max_model_len": 100_000,
        "context_limits": {"32k": 30_000, "max": 92_856},
        "vision":       True,
        "tp":           4,
    },
    {
        "index":        8,
        "name":         "gemma-3-27b-it",
        "path":         str(MODEL_HUB / "gemma-3-27b-it"),
        "max_model_len": 100_000,
        "context_limits": {"32k": 30_000, "max": 92_856},
        "vision":       True,
        "tp":           4,
    },
    # ── MedGemma family ─────────────────────────────────────────────────────────
    {
        "index":        9,
        "name":         "medgemma-4b-it",
        "path":         str(MODEL_HUB / "medgemma-4b-it"),
        "max_model_len": 100_000,
        "context_limits": {"32k": 30_000, "max": 92_856},
        "vision":       True,   # multimodal
        "tp":           4,
    },
    {
        "index":        10,
        "name":         "medgemma-27b-text-it",
        "path":         str(MODEL_HUB / "medgemma-27b-text-it"),
        "max_model_len": 100_000,
        "context_limits": {"32k": 30_000, "max": 92_856},
        "vision":       False,  # text-only variant
        "tp":           4,
    },
    # ── Qwen3 family ────────────────────────────────────────────────────────────
    {
        "index":        11,
        "name":         "Qwen3-4B",
        "path":         str(MODEL_HUB / "Qwen3-4B"),
        "max_model_len": 40_000,
        "context_limits": {"32k": 30_000, "max": 32_856},  # 40k - 2048 - 4096 - 1000
        "vision":       False,
        "tp":           4,
    },
    {
        "index":        12,
        "name":         "Qwen3-8B",
        "path":         str(MODEL_HUB / "Qwen3-8B"),
        "max_model_len": 40_000,
        "context_limits": {"32k": 30_000, "max": 32_856},  # 40k - 2048 - 4096 - 1000
        "vision":       False,
        "tp":           4,
    },
    {
        "index":        13,
        "name":         "Qwen3-14B",
        "path":         str(MODEL_HUB / "Qwen3-14B"),
        "max_model_len": 40_000,
        "context_limits": {"32k": 30_000, "max": 32_856},  # 40k - 2048 - 4096 - 1000
        "vision":       False,
        "tp":           4,
    },
    {
        "index":        14,
        "name":         "Qwen3-32B",
        "path":         str(MODEL_HUB / "Qwen3-32B"),
        "max_model_len": 40_000,
        "context_limits": {"32k": 30_000, "max": 32_856},  # 40k - 2048 - 4096 - 1000
        "vision":       False,
        "tp":           4,
    },
    # ── GPT-OSS family ──────────────────────────────────────────────────────────
    {
        "index":        16,
        "name":         "gpt-oss-20b",
        "path":         str(MODEL_HUB / "gpt-oss-20b"),
        "max_model_len": 100_000,
        "context_limits": {"32k": 30_000, "max": 92_856},
        "vision":       False,
        "tp":           4,
    },
    {
        "index":        17,
        "name":         "gpt-oss-120b",
        "path":         str(MODEL_HUB / "gpt-oss-120b"),
        "max_model_len": 100_000,
        "context_limits": {"32k": 30_000, "max": 92_856},
        "vision":       False,
        "tp":           4,
    },
    # ── Gemma-4 family (multimodal) ────────────────────────────────────────────
    {
        "index":        19,
        "name":         "gemma-4-E4B-it",
        "path":         str(MODEL_HUB / "gemma-4-E4B-it"),
        "tokenizer_path": str(MODEL_HUB / "gemma-3-4b-it"),
        "max_model_len": 131_072,
        "context_limits": {"32k": 30_000, "100k": 96_000, "max": 123_928},
        "vision":       True,
        "tp":           1,  # 8 attn heads — not divisible by 4
    },
    {
        "index":        20,
        "name":         "gemma-4-31B-it",
        "path":         str(MODEL_HUB / "gemma-4-31B-it"),
        "tokenizer_path": str(MODEL_HUB / "gemma-3-4b-it"),
        "max_model_len": 131_072,  # conservative; native 262k
        "context_limits": {"32k": 30_000, "100k": 96_000, "max": 123_928},
        "vision":       True,
        "tp":           4,
    },
    # ── Mistral family ──────────────────────────────────────────────────────────
    {
        "index":        21,
        "name":         "Ministral-3-3B-Instruct-2512",
        "path":         str(MODEL_HUB / "Ministral-3-3B-Instruct-2512"),
        "tokenizer_path": str(MODEL_HUB / "Mistral-Large-Instruct-2407"),
        "max_model_len": 131_072,
        "context_limits": {"32k": 30_000, "100k": 96_000, "max": 123_928},
        "vision":       True,
        "tp":           1,
    },
    {
        "index":        22,
        "name":         "Ministral-3-8B-Instruct-2512",
        "path":         str(MODEL_HUB / "Ministral-3-8B-Instruct-2512"),
        "tokenizer_path": str(MODEL_HUB / "Mistral-Large-Instruct-2407"),
        "max_model_len": 131_072,
        "context_limits": {"32k": 30_000, "100k": 96_000, "max": 123_928},
        "vision":       True,
        "tp":           4,
    },
    {
        "index":        23,
        "name":         "Ministral-3-14B-Instruct-2512",
        "path":         str(MODEL_HUB / "Ministral-3-14B-Instruct-2512"),
        "tokenizer_path": str(MODEL_HUB / "Mistral-Large-Instruct-2407"),
        "max_model_len": 131_072,
        "context_limits": {"32k": 30_000, "100k": 96_000, "max": 123_928},
        "vision":       True,
        "tp":           4,
    },
    {
        "index":        24,
        "name":         "Mistral-Small-3.2-24B-Instruct-2506",
        "path":         str(MODEL_HUB / "Mistral-Small-3.2-24B-Instruct-2506"),
        "tokenizer_path": str(MODEL_HUB / "Mistral-Large-Instruct-2407"),
        "max_model_len": 131_072,
        "context_limits": {"32k": 30_000, "100k": 96_000, "max": 123_928},
        "vision":       False,
        "tp":           4,
    },
    {
        "index":        25,
        "name":         "Mistral-Small-4-119B-2603",
        "path":         str(MODEL_HUB / "Mistral-Small-4-119B-2603"),
        "tokenizer_path": str(MODEL_HUB / "Mistral-Large-Instruct-2407"),
        "max_model_len": 131_072,
        "context_limits": {"32k": 30_000, "100k": 96_000, "max": 123_928},
        "vision":       False,
        "tp":           4,
    },
    # ── Qwen3.5 family ──────────────────────────────────────────────────────────
    {
        "index":        26,
        "name":         "Qwen3.5-0.8B",
        "path":         str(MODEL_HUB / "Qwen3.5-0.8B"),
        "max_model_len": 131_072,
        "context_limits": {"32k": 30_000, "100k": 96_000, "max": 123_928},
        "vision":       False,
        "tp":           1,
        "max_num_seqs": 64,
    },
    {
        "index":        27,
        "name":         "Qwen3.5-2B",
        "path":         str(MODEL_HUB / "Qwen3.5-2B"),
        "max_model_len": 131_072,
        "context_limits": {"32k": 30_000, "100k": 96_000, "max": 123_928},
        "vision":       False,
        "tp":           1,
        "max_num_seqs": 64,
    },
    {
        "index":        28,
        "name":         "Qwen3.5-4B",
        "path":         str(MODEL_HUB / "Qwen3.5-4B"),
        "max_model_len": 131_072,
        "context_limits": {"32k": 30_000, "100k": 96_000, "max": 123_928},
        "vision":       False,
        "tp":           1,
        "max_num_seqs": 64,
    },
    {
        "index":        29,
        "name":         "Qwen3.5-9B",
        "path":         str(MODEL_HUB / "Qwen3.5-9B"),
        "max_model_len": 131_072,
        "context_limits": {"32k": 30_000, "100k": 96_000, "max": 123_928},
        "vision":       False,
        "tp":           4,
        "concurrency":  16,
        "max_num_seqs": 32,
    },
    {
        "index":        30,
        "name":         "Qwen3.5-27B",
        "path":         str(MODEL_HUB / "Qwen3.5-27B"),
        "max_model_len": 131_072,
        "context_limits": {"32k": 30_000, "100k": 96_000, "max": 123_928},
        "vision":       False,
        "tp":           4,
        "concurrency":  8,
        "max_num_seqs": 16,
    },
    {
        "index":        31,
        "name":         "Qwen3.5-35B-A3B",
        "path":         str(MODEL_HUB / "Qwen3.5-35B-A3B"),
        "max_model_len": 131_072,
        "context_limits": {"32k": 30_000, "100k": 96_000, "max": 123_928},
        "vision":       False,
        "tp":           4,
        "concurrency":  16,
        "max_num_seqs": 32,
    },
    {
        "index":        32,
        "name":         "Qwen3.5-122B-A10B",
        "path":         str(MODEL_HUB / "Qwen3.5-122B-A10B"),
        "max_model_len": 131_072,
        "context_limits": {"32k": 30_000, "100k": 96_000, "max": 123_928},
        "vision":       False,
        "tp":           4,
        "concurrency":  8,
        "max_num_seqs": 16,
    },
    # ── Llama family ────────────────────────────────────────────────────────────
    {
        "index":        33,
        "name":         "Llama-3.2-3B-Instruct",
        "path":         str(MODEL_HUB / "Llama-3.2-3B-Instruct"),
        "max_model_len": 131_072,
        "context_limits": {"32k": 30_000, "100k": 96_000, "max": 123_928},
        "vision":       False,
        "tp":           1,
    },
    {
        "index":        34,
        "name":         "Meta-Llama-3-8B-Instruct",
        "path":         str(MODEL_HUB / "Meta-Llama-3-8B-Instruct"),
        "max_model_len": 8_192,  # original Llama 3 — 8k context only
        "context_limits": {"max": 1_048},  # 8192 - 2048 - 4096 - 1000
        "vision":       False,
        "tp":           4,
    },
    {
        "index":        35,
        "name":         "llama3_370b",  # Llama 3.1 70B (confirmed via config.json)
        "path":         str(MODEL_HUB / "llama3_370b"),
        "max_model_len": 131_072,
        "context_limits": {"32k": 30_000, "100k": 96_000, "max": 123_928},
        "vision":       False,
        "tp":           4,
    },
    {
        "index":        36,
        "name":         "Llama-3.2-1B-Instruct",
        "path":         str(MODEL_HUB / "Llama-3.2-1B-Instruct"),
        "max_model_len": 131_072,
        "context_limits": {"32k": 30_000, "100k": 96_000, "max": 123_928},
        "vision":       False,
        "tp":           1,
    },
    {
        "index":        37,
        "name":         "Llama-3.3-70B-Instruct",
        "path":         str(MODEL_HUB / "Llama-3.3-70B-Instruct"),
        "max_model_len": 131_072,
        "context_limits": {"32k": 30_000, "100k": 96_000, "max": 123_928},
        "vision":       False,
        "tp":           4,
    },
    {
        "index":        38,
        "name":         "Meta-Llama-3.1-405B-Instruct-FP8",
        "path":         str(MODEL_HUB / "Meta-Llama-3.1-405B-Instruct-FP8"),
        "max_model_len": 131_072,
        "context_limits": {"32k": 30_000, "100k": 96_000, "max": 123_928},
        "vision":       False,
        "tp":           4,  # FP8 weights ~405GB; requires H200 (4x141GB=564GB)
    },
]

NUM_MODELS = len(MODELS)  # locally hostable open-weight models

# ── Dataset ────────────────────────────────────────────────────────────────────
# Only the risk_radiorag (RadSaFE-200) dataset is used in the published paper.
MCQ_DATASETS = [
    {
        "name":          "risk_radiorag",
        "path":          FINAL_DATASETS / "risk_radiorag_final.jsonl",
        "question_field": "question_text",
        "options_field":  "options",
        "answer_field":   "correct_answer",
        # The dataset carries pre-built context fields used as additional
        # context-injection conditions in the paper.
        "extra_conditions": ["deep_research", "evidence_clean", "evidence_conflict"],
    },
]

# ── Context conditions ─────────────────────────────────────────────────────────
# Fixed conditions come from pre-built prompt_conditions in final_datasets.
# Extended conditions are built on-the-fly by truncating extended_150k context
# with the model's own tokenizer.
FIXED_CONDITIONS = ["zero_shot", "top_1", "top_5", "top_10"]
EXTENDED_CONDITIONS = {
    "context_32k":  "32k",   # maps to context_limits key
    "context_100k": "100k",
    "context_max":  "max",   # max_model_len - 2048(question) - 4096(max_new_tokens) - 1000(margin)
}

# ── Inference parameters ───────────────────────────────────────────────────────
VLLM_PORT          = 6000
VLLM_STARTUP_WAIT  = 180   # seconds to wait for vLLM health endpoint

GREEDY_PARAMS = {
    "temperature": 0.0,
    "max_tokens":  10,
    "n":           1,
}

STOCHASTIC_PARAMS = {
    "temperature": 0.7,
    "max_tokens":  10,
    "n":           20,   # 20 samples for majority vote
}

# Reasoning models need more tokens for chain-of-thought before they emit
# the final letter answer.
REASONING_PARAMS_GREEDY = {
    "temperature": 0.0,
    "max_tokens":  4096,
    "n":           1,
}

REASONING_PARAMS_STOCHASTIC = {
    "temperature": 0.7,
    "max_tokens":  4096,
    "n":           20,
}

# Letters we parse/score
ANSWER_LETTERS = ["A", "B", "C", "D", "E"]

# ── Prompt template ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an expert medical assistant. "
    "Answer the following multiple-choice question by selecting only the letter "
    "of the correct answer. Do not explain. Output only the letter."
)

CONTEXT_USER_TEMPLATE = """{context}

Question: {question}

Options:
{options}

Answer:"""

NO_CONTEXT_USER_TEMPLATE = """Question: {question}

Options:
{options}

Answer:"""
