#!/usr/bin/env python3
"""
Phase 4 — Count tokens of every `raw_output` (greedy) and `raw_outputs[]`
(stochastic) across every `risk_radiorag.jsonl` in phase3 results.

Goal: size the verifier's max context window. Reports per-model and global:
    count, total_tokens, min, max, mean, p50, p95, p99

Uses the Mistral-Small-4 tokenizer (Mistral-Large-Instruct-2407 vocab, matching
`config.py`), because that's the model that will be doing the checking.

Usage:
    python3 count_raw_output_tokens.py                    # all models
    python3 count_raw_output_tokens.py --model gpt-oss-120b
    python3 count_raw_output_tokens.py --results-dir /path/to/results
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, Iterable, List

from transformers import AutoTokenizer

DEFAULT_RESULTS_DIR = Path("/path/to/workspace/phase3_inference/results")
DEFAULT_TOKENIZER   = "${MODELS_DIR}/Mistral-Large-Instruct-2407"
DATASET_FILENAME    = "risk_radiorag.jsonl"


def iter_raw_outputs(record: Dict) -> Iterable[str]:
    """Yield every raw_output string from a phase3 result record."""
    for cond_data in record.get("conditions", {}).values():
        g_raw = cond_data.get("greedy", {}).get("raw_output")
        if isinstance(g_raw, str):
            yield g_raw
        for s_raw in cond_data.get("stochastic", {}).get("raw_outputs", []) or []:
            if isinstance(s_raw, str):
                yield s_raw


def percentile(values: List[int], p: float) -> int:
    if not values:
        return 0
    values = sorted(values)
    k = (len(values) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return int(values[f] + (values[c] - values[f]) * (k - f))


def summarize(lengths: List[int]) -> Dict:
    if not lengths:
        return {"count": 0, "total": 0, "min": 0, "max": 0, "mean": 0.0,
                "p50": 0, "p95": 0, "p99": 0}
    return {
        "count":  len(lengths),
        "total":  sum(lengths),
        "min":    min(lengths),
        "max":    max(lengths),
        "mean":   round(statistics.mean(lengths), 1),
        "p50":    percentile(lengths, 50),
        "p95":    percentile(lengths, 95),
        "p99":    percentile(lengths, 99),
    }


def fmt_row(name: str, s: Dict) -> str:
    return (f"{name:<45s}  n={s['count']:>7d}  "
            f"total={s['total']:>12,d}  "
            f"min={s['min']:>5d}  p50={s['p50']:>5d}  "
            f"p95={s['p95']:>6d}  p99={s['p99']:>6d}  max={s['max']:>6d}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--tokenizer",   type=str, default=DEFAULT_TOKENIZER,
                    help="HF tokenizer path or name (default: Mistral-Large-Instruct-2407)")
    ap.add_argument("--model",       type=str, default=None,
                    help="Only analyze a specific model subdirectory")
    ap.add_argument("--dataset",     type=str, default=DATASET_FILENAME,
                    help="Filename to look for (default: risk_radiorag.jsonl)")
    ap.add_argument("--json-out",    type=Path, default=None,
                    help="Optional JSON path to dump full stats")
    args = ap.parse_args()

    if not args.results_dir.exists():
        sys.exit(f"Results dir not found: {args.results_dir}")

    print(f"Loading tokenizer: {args.tokenizer}")
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    # Discover model dirs
    model_dirs = sorted(d for d in args.results_dir.iterdir()
                        if d.is_dir() and d.name != "backup")
    if args.model:
        model_dirs = [d for d in model_dirs if d.name == args.model]
        if not model_dirs:
            sys.exit(f"Model '{args.model}' not found under {args.results_dir}")

    per_model: Dict[str, Dict] = {}
    global_lengths: List[int] = []
    global_max_example: Dict = {"tokens": -1, "model": None, "qid": None, "preview": ""}

    print(f"\nScanning {len(model_dirs)} model directory(ies) for '{args.dataset}'...\n")

    for mdir in model_dirs:
        f = mdir / args.dataset
        if not f.exists():
            continue

        lengths: List[int] = []
        n_records = 0
        try:
            with open(f, "r") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    n_records += 1
                    qid = rec.get("question_id", "?")
                    for raw in iter_raw_outputs(rec):
                        if not raw:
                            lengths.append(0)
                            continue
                        n = len(tok.encode(raw, add_special_tokens=False))
                        lengths.append(n)
                        if n > global_max_example["tokens"]:
                            global_max_example = {
                                "tokens":  n,
                                "model":   mdir.name,
                                "qid":     qid,
                                "preview": raw[:240].replace("\n", " ") + ("…" if len(raw) > 240 else ""),
                            }
        except Exception as e:
            print(f"  [warn] failed to process {f}: {e}", file=sys.stderr)
            continue

        stats = summarize(lengths)
        stats["records"] = n_records
        per_model[mdir.name] = stats
        global_lengths.extend(lengths)
        print(fmt_row(mdir.name, stats))

    print("\n" + "=" * 130)
    global_stats = summarize(global_lengths)
    print(fmt_row("ALL MODELS (risk_radiorag raw_outputs)", global_stats))
    print("=" * 130)

    print("\nLargest single raw_output:")
    print(f"  {global_max_example['tokens']} tokens  "
          f"(model={global_max_example['model']}, qid={global_max_example['qid']})")
    print(f"  preview: {global_max_example['preview']}")

    # Recommendation
    suggested_input  = max(512, int(global_stats["p99"] * 1.2))
    suggested_ceiling = max(suggested_input, int(global_stats["max"] * 1.1))
    print("\nSuggested verifier sizing (prompt side, in Mistral tokens):")
    print(f"  p99*1.2  ~ {suggested_input:>6d}  (covers 99% of samples comfortably)")
    print(f"  max*1.1  ~ {suggested_ceiling:>6d}  (hard upper bound incl. prompt overhead)")
    print(f"  Add ~300 tokens for instructions + JSON scaffold + question letters.")

    if args.json_out:
        out = {
            "tokenizer": args.tokenizer,
            "dataset":   args.dataset,
            "per_model": per_model,
            "global":    global_stats,
            "largest_example": global_max_example,
            "suggested_input_window": suggested_input,
            "suggested_input_ceiling": suggested_ceiling,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nStats written to {args.json_out}")


if __name__ == "__main__":
    main()
