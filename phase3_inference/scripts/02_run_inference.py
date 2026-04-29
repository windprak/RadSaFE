#!/usr/bin/env python3
"""
Phase 3 — Step 2: Run inference for one model across all datasets and context conditions.

Sends requests CONCURRENTLY to vLLM (controlled by --concurrency semaphore).
vLLM handles internal batching; flooding it with 32-64 concurrent requests
saturates GPU throughput significantly vs sequential.

Usage:
    python3 02_run_inference.py --model-index 0                # debug (0.5B)
    python3 02_run_inference.py --model-index 3 --concurrency 64
    python3 02_run_inference.py --model-index $SLURM_ARRAY_TASK_ID
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from openai import AsyncOpenAI
from transformers import AutoTokenizer
from tqdm.asyncio import tqdm_asyncio

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    MODELS, MCQ_DATASETS, RESULTS_DIR, LOGS_DIR,
    VLLM_PORT, VLLM_STARTUP_WAIT, ANSWER_LETTERS,
    GREEDY_PARAMS, STOCHASTIC_PARAMS,
    REASONING_PARAMS_GREEDY, REASONING_PARAMS_STOCHASTIC,
    SYSTEM_PROMPT, CONTEXT_USER_TEMPLATE, NO_CONTEXT_USER_TEMPLATE,
    FIXED_CONDITIONS, EXTENDED_CONDITIONS,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def wait_for_vllm(port: int, timeout: int = VLLM_STARTUP_WAIT) -> bool:
    url = f"http://localhost:{port}/health"
    print(f"Waiting for vLLM on port {port} (timeout={timeout}s)...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                print("vLLM is ready.")
                return True
        except Exception:
            pass
        time.sleep(5)
    print("ERROR: vLLM did not become ready in time.")
    return False


def format_options(options: Dict) -> str:
    return "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))


def build_prompt(question: str, options: Dict, context: str = "") -> str:
    opts = format_options(options)
    if context:
        return CONTEXT_USER_TEMPLATE.format(context=context, question=question, options=opts)
    return NO_CONTEXT_USER_TEMPLATE.format(question=question, options=opts)


def parse_answer(text: str) -> Optional[str]:
    """Extract the first A/B/C/D letter from model output."""
    text = text.strip()
    # Try first character
    if text and text[0] in ANSWER_LETTERS:
        return text[0]
    # Try pattern "Answer: X" or "(X)"
    m = re.search(r"\b([ABCDE])\b", text)
    if m:
        return m.group(1)
    return None


def majority_vote(answers: List[Optional[str]]) -> Tuple[Optional[str], Dict[str, int]]:
    counts = {l: 0 for l in ANSWER_LETTERS}
    for a in answers:
        if a in counts:
            counts[a] += 1
    best = max(counts, key=lambda k: counts[k])
    return best if counts[best] > 0 else None, counts


# ── Context builder ────────────────────────────────────────────────────────────

class ContextBuilder:
    def __init__(self, model_path: str, model_cfg: Dict):
        self.model_cfg = model_cfg
        tok_path = model_cfg.get("tokenizer_path", model_path)
        print(f"Loading tokenizer from: {tok_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)

    def get_context_text(self, record: Dict, condition: str) -> str:
        """
        For fixed conditions (zero_shot, top_1, top_5, top_10):
            return the pre-built context text from retrieved_contexts.
        For extended conditions (context_32k, context_100k):
            tokenize extended_150k text with model tokenizer and truncate.
        For deep_research:
            return deep_research_report from metadata.
        """
        if condition == "zero_shot":
            return ""

        if condition == "deep_research":
            return record.get("metadata", {}).get("deep_research_report", "")

        if condition in ("evidence_clean", "evidence_conflict"):
            return record.get("metadata", {}).get(condition, "")

        if condition in FIXED_CONDITIONS:
            # Use pre-built prompt context from prompt_conditions
            pc = record.get("prompt_conditions", {}).get(condition, {})
            # Extract context text (between "Context:" and "Question:")
            prompt = pc.get("prompt", "")
            context_match = re.search(r"^Context:\n(.*?)\n\nQuestion:", prompt, re.DOTALL)
            if context_match:
                return context_match.group(1).strip()
            return ""

        if condition in EXTENDED_CONDITIONS:
            limit_key = EXTENDED_CONDITIONS[condition]
            max_ctx_tokens = self.model_cfg["context_limits"].get(limit_key)
            if max_ctx_tokens is None:
                return None  # signal: condition not available for this model

            # Get raw extended context text
            raw_ctx = (
                record.get("retrieved_contexts", {})
                      .get("extended_150k", {})
                      .get("text", "")
            )
            if not raw_ctx:
                return ""

            # Tokenize + truncate with model's own tokenizer
            tokens = self.tokenizer.encode(raw_ctx, add_special_tokens=False)
            if len(tokens) > max_ctx_tokens:
                tokens = tokens[:max_ctx_tokens]
                raw_ctx = self.tokenizer.decode(tokens, skip_special_tokens=True)
            return raw_ctx

        return ""


# ── Async Inference runner ─────────────────────────────────────────────────────

class InferenceRunner:
    def __init__(self, model_name: str, model_path: str, port: int = VLLM_PORT, concurrency: int = 32):
        self.model_name = model_name
        # Convert host path to container path for vLLM:
        #   ${MODELS_DIR}/X  →  /models/X  (vLLM is started with
        #   --bind ${MODELS_DIR}:/models, see slurm/run_inference_array.sh).
        models_dir_host = os.environ.get("MODELS_DIR")
        if models_dir_host and model_path.startswith(models_dir_host.rstrip("/") + "/"):
            self.model_path = "/models/" + model_path[len(models_dir_host.rstrip("/")) + 1:]
        else:
            self.model_path = model_path
        self.concurrency = concurrency
        base_url = f"http://localhost:{port}/v1"
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key="token-not-needed",
        )
        self.semaphore = asyncio.Semaphore(concurrency)

    async def run_greedy(self, messages: List[Dict]) -> Dict:
        async with self.semaphore:
            t0 = time.time()
            try:
                # Use reasoning params for gpt-oss, Qwen3, Gemma-4, DeepSeek-R1, and medgemma-27b models
                is_reasoning = 'gpt-oss' in self.model_name or 'Qwen3' in self.model_name or 'gemma-4' in self.model_name or 'DeepSeek-R1' in self.model_name or 'medgemma-27b' in self.model_name
                params = REASONING_PARAMS_GREEDY if is_reasoning else GREEDY_PARAMS
                
                for _attempt in range(4):
                    try:
                        resp = await self.client.chat.completions.create(
                            model=self.model_path,
                            messages=messages,
                            **params,
                        )
                        break
                    except Exception as _e:
                        if 'connect' in str(_e).lower() and _attempt < 3:
                            await asyncio.sleep(15 * (2 ** _attempt))
                        else:
                            raise
                elapsed = time.time() - t0
                choice = resp.choices[0]
                # Handle models that put answer in reasoning_content instead of content
                raw = choice.message.content or getattr(choice.message, 'reasoning_content', None) or ""
                
                # For reasoning models, strip reasoning tokens to get final answer only
                answer_text = raw
                if is_reasoning:
                    # Qwen3: <think>...</think>answer
                    if '</think>' in raw:
                        answer_text = raw.split('</think>')[-1].strip()
                    # gpt-oss: ...<|return|>answer
                    elif '<|return|>' in raw:
                        answer_text = raw.split('<|return|>')[-1].strip()
                    # Gemma 4: <|channel>thought\n...reasoning...<channel|>answer
                    # Bug: tokens get stripped, so output is "thought\n...\nanswer"
                    elif '<channel|>' in raw:
                        answer_text = raw.split('<channel|>')[-1].strip()
                    elif raw.startswith('thought\n') or raw.startswith('thought\r'):
                        # Fallback: strip from last newline paragraph
                        lines = raw.strip().split('\n')
                        answer_text = lines[-1].strip()
                    elif '<unused94>' in raw:
                        # medgemma-27b: <unused94>thought\n...reasoning...\nanswer
                        lines = raw.strip().split('\n')
                        answer_text = lines[-1].strip()
                
                parsed = parse_answer(answer_text)
                return {
                    "raw_output":    raw,
                    "parsed_answer": parsed,
                    "input_tokens":  resp.usage.prompt_tokens,
                    "output_tokens": resp.usage.completion_tokens,
                    "elapsed_s":     round(elapsed, 3),
                    "error":         None,
                }
            except Exception as e:
                return {"error": str(e), "parsed_answer": None, "raw_output": None,
                        "input_tokens": 0, "output_tokens": 0, "elapsed_s": 0.0}


    async def run_stochastic(self, messages: List[Dict]) -> Dict:
        async with self.semaphore:
            t0 = time.time()
            try:
                # Use reasoning params for gpt-oss, Qwen3, Gemma-4, DeepSeek-R1, and medgemma-27b models
                is_reasoning = 'gpt-oss' in self.model_name or 'Qwen3' in self.model_name or 'gemma-4' in self.model_name or 'DeepSeek-R1' in self.model_name or 'medgemma-27b' in self.model_name
                params = REASONING_PARAMS_STOCHASTIC if is_reasoning else STOCHASTIC_PARAMS
                
                for _attempt in range(4):
                    try:
                        resp = await self.client.chat.completions.create(
                            model=self.model_path,
                            messages=messages,
                            **params,
                        )
                        break
                    except Exception as _e:
                        if 'connect' in str(_e).lower() and _attempt < 3:
                            await asyncio.sleep(15 * (2 ** _attempt))
                        else:
                            raise
                elapsed = time.time() - t0
                # Handle models that put answer in reasoning_content instead of content
                raws = [c.message.content or getattr(c.message, 'reasoning_content', None) or "" for c in resp.choices]
                
                # For reasoning models, strip reasoning tokens to get final answer only
                answer_texts = []
                for raw in raws:
                    answer_text = raw
                    if is_reasoning:
                        # Qwen3: <think>...</think>answer
                        if '</think>' in raw:
                            answer_text = raw.split('</think>')[-1].strip()
                        # gpt-oss: ...<|return|>answer
                        elif '<|return|>' in raw:
                            answer_text = raw.split('<|return|>')[-1].strip()
                        # Gemma 4: <|channel>thought\n...reasoning...<channel|>answer
                        elif '<channel|>' in raw:
                            answer_text = raw.split('<channel|>')[-1].strip()
                        elif raw.startswith('thought\n') or raw.startswith('thought\r'):
                            lines = raw.strip().split('\n')
                            answer_text = lines[-1].strip()
                        elif '<unused94>' in raw:
                            # medgemma-27b: <unused94>thought\n...reasoning...\nanswer
                            lines = raw.strip().split('\n')
                            answer_text = lines[-1].strip()
                    answer_texts.append(answer_text)
                
                parsed = [parse_answer(a) for a in answer_texts]
                vote, counts = majority_vote(parsed)
                return {
                    "raw_outputs":    raws,
                    "parsed_answers": parsed,
                    "majority_vote":  vote,
                    "vote_counts":    counts,
                    "input_tokens":   resp.usage.prompt_tokens,
                    "elapsed_s":      round(elapsed, 3),
                    "error":          None,
                }
            except Exception as e:
                return {"error": str(e), "majority_vote": None, "parsed_answers": [],
                        "raw_outputs": [], "vote_counts": {}, "input_tokens": 0, "elapsed_s": 0.0}

    async def run_question_condition(
        self, record: Dict, condition: str, dataset_cfg: Dict,
        ctx_builder: "ContextBuilder"
    ) -> Optional[Tuple[str, Dict, Dict]]:
        """Run greedy + stochastic for one (question, condition) pair. Returns (condition, greedy, stochastic)."""
        ctx_text = ctx_builder.get_context_text(record, condition)
        if ctx_text is None:
            return None  # condition not available for this model

        question = record.get(dataset_cfg["question_field"], "")
        options  = record.get(dataset_cfg["options_field"], {})
        user_msg = build_prompt(question, options, ctx_text)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ]

        greedy, stochastic = await asyncio.gather(
            self.run_greedy(messages),
            self.run_stochastic(messages),
        )
        return condition, greedy, stochastic


# ── Async dataset processing ───────────────────────────────────────────────────

def _record_has_errors(record: Dict) -> bool:
    for cond_data in record.get("conditions", {}).values():
        if cond_data.get("greedy", {}).get("error"):
            return True
        if cond_data.get("stochastic", {}).get("error"):
            return True
    return False


async def process_dataset_async(
    dataset_cfg: Dict,
    model_cfg: Dict,
    ctx_builder: ContextBuilder,
    runner: InferenceRunner,
    out_path: Path,
    retry_errors: bool = True,
):
    ds_name = dataset_cfg["name"]
    ds_path = dataset_cfg["path"]

    if not Path(ds_path).exists():
        print(f"  Dataset not found: {ds_path} — skipping.")
        return

    print(f"\n{'='*60}\n  {ds_name.upper()} → {out_path}\n{'='*60}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine which conditions to run for this model
    conditions = list(FIXED_CONDITIONS)
    for cond_name, limit_key in EXTENDED_CONDITIONS.items():
        if limit_key in model_cfg["context_limits"]:
            conditions.append(cond_name)
    # Add dataset-specific extra conditions (e.g. deep_research for risk_radiorag)
    for extra in dataset_cfg.get("extra_conditions", []):
        conditions.append(extra)

    # Load all records
    records = []
    with open(ds_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # Resume support: find already-completed question_ids
    already_done: set = set()
    errored_ids: set = set()
    kept_lines: list = []
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    qid = d["question_id"]
                    if retry_errors and _record_has_errors(d):
                        errored_ids.add(qid)
                    else:
                        already_done.add(qid)
                        kept_lines.append(line)
                except Exception:
                    pass
        # Rewrite output file without errored records so they get re-appended
        if errored_ids:
            with open(out_path, "w") as f:
                for line in kept_lines:
                    f.write(line + "\n")

    records_to_run = [r for r in records if r.get("question_id", "") not in already_done]

    print(f"  Loaded {len(records)} records | {len(conditions)} conditions | "
          f"concurrency={runner.concurrency}")
    if already_done or errored_ids:
        print(f"  Resuming: {len(already_done)} done, {len(errored_ids)} errored→retrying, {len(records_to_run)} remaining.")

    if not records_to_run:
        print("  All records already completed — nothing to do.")
        return

    parse_errors = 0

    # Build tasks: one coroutine per (record, condition)
    async def process_record(record: Dict) -> Dict:
        q_id    = record.get("question_id", "")
        correct = record.get(dataset_cfg["answer_field"], "")

        result = {
            "question_id":    q_id,
            "dataset":        ds_name,
            "correct_answer": correct,
            "model":          model_cfg["name"],
            "conditions":     {},
        }

        coro_results = await asyncio.gather(
            *[runner.run_question_condition(record, cond, dataset_cfg, ctx_builder)
              for cond in conditions]
        )

        for r in coro_results:
            if r is None:
                continue
            condition, greedy, stochastic = r

            greedy["correct"] = greedy.get("parsed_answer") == correct if greedy.get("parsed_answer") else False
            stochastic["majority_correct"] = stochastic.get("majority_vote") == correct if stochastic.get("majority_vote") else False

            result["conditions"][condition] = {
                "greedy": greedy,
                "stochastic": stochastic,
                "total_elapsed_s": round(greedy.get("elapsed_s", 0) + stochastic.get("elapsed_s", 0), 3),
            }

        return result

    # Write each record immediately as it completes (enables resume on restart)
    file_mode = "a" if (already_done or errored_ids) else "w"
    with open(out_path, file_mode) as fout:
        async def process_and_write(record: Dict) -> Dict:
            nonlocal parse_errors
            result = await process_record(record)
            fout.write(json.dumps(result) + "\n")
            fout.flush()
            for cond_data in result["conditions"].values():
                if not cond_data["greedy"].get("parsed_answer"):
                    parse_errors += 1
            return result

        desc = f"{ds_name} ({len(records_to_run)}/{len(records)})" if already_done else ds_name
        await tqdm_asyncio.gather(
            *[process_and_write(r) for r in records_to_run],
            desc=desc,
        )

    n_cond_calls = len(records_to_run) * len(conditions)
    parse_rate = 1.0 - (parse_errors / max(n_cond_calls, 1))
    print(f"\n  Processed {len(records_to_run)}/{len(records)} records | Parse rate: {parse_rate:.1%} | Errors: {parse_errors}")
    if parse_rate < 0.95:
        print("  WARNING: parse rate below 95% — review parsing logic!")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-index",  type=int, required=True)
    parser.add_argument("--port",         type=int, default=VLLM_PORT)
    parser.add_argument("--concurrency",  type=int, default=32,
                        help="Max concurrent requests to vLLM (default: 32)")
    parser.add_argument("--skip-wait",    action="store_true")
    parser.add_argument("--datasets",      type=str, default=None,
                        help="Comma-separated dataset names to run (default: all)")
    parser.add_argument("--resume-from",   type=str, default=None,
                        help="question_id to resume from (truncates output at this ID and reprocesses from here)")
    parser.add_argument("--no-retry-errors", action="store_true",
                        help="Disable automatic retry of records with errors (default: retry errors)")
    args = parser.parse_args()

    model_cfg = MODELS[args.model_index]
    print(f"\n{'='*60}")
    print(f"  Model: {model_cfg['name']}")
    print(f"  Context limits: {model_cfg['context_limits']}")
    print(f"  Vision: {model_cfg['vision']}")
    print(f"  Concurrency: {model_cfg.get('concurrency', args.concurrency)}")
    print(f"{'='*60}\n")

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_wait:
        if not wait_for_vllm(args.port):
            sys.exit(1)

    ctx_builder = ContextBuilder(model_cfg["path"], model_cfg)
    concurrency = model_cfg.get("concurrency", args.concurrency)
    runner      = InferenceRunner(
        model_cfg["name"],
        model_cfg["path"],
        args.port,
        concurrency,
    )

    model_result_dir = RESULTS_DIR / model_cfg["name"]
    model_result_dir.mkdir(parents=True, exist_ok=True)

    dataset_filter = set(args.datasets.split(",")) if args.datasets else None

    async def run_all():
        for dataset_cfg in MCQ_DATASETS:
            if dataset_filter and dataset_cfg["name"] not in dataset_filter:
                print(f"  Skipping dataset: {dataset_cfg['name']} (not in --datasets filter)")
                continue
            out_path = model_result_dir / f"{dataset_cfg['name']}.jsonl"
            if args.resume_from and out_path.exists():
                # Truncate output file: keep only records BEFORE resume_from id
                kept = []
                with open(out_path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            d = json.loads(line)
                            if d["question_id"] == args.resume_from:
                                break  # stop here; reprocess from this id onward
                            kept.append(line)
                        except Exception:
                            pass
                with open(out_path, "w") as f:
                    for line in kept:
                        f.write(line + "\n")
                print(f"  --resume-from: truncated to {len(kept)} records, reprocessing from {args.resume_from}")
            await process_dataset_async(
                dataset_cfg, model_cfg, ctx_builder, runner,
                out_path=out_path,
                retry_errors=not args.no_retry_errors,
            )

    asyncio.run(run_all())

    print(f"\n{'='*60}")
    print(f"  DONE: {model_cfg['name']}")
    print(f"  Results: {model_result_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
