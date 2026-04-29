#!/usr/bin/env python3
"""
Phase 5 paired McNemar p-values between conditions of the same model,
using the phase-4 confirmed_answer correctness vectors.

For every model, we enumerate a set of condition pairs and report the
exact two-sided McNemar p-value. By default we compare every condition
against zero_shot (closed-book) and also the "clean vs conflict",
"clean vs top_10", "RaR vs top_10" contrasts, mirroring the delta
columns of Table 1.

Outputs
-------
<OUT>/pvalues_all.csv
<OUT>/pvalues_all.json

Run it directly:
    python run_pvalues.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np

PHASE4_DIR = Path("/path/to/workspace/phase4_checking_results/results")
DEFAULT_OUT = Path("/path/to/workspace/phase5_bootstrapping/bootstrap_results")
DATASET_BASENAME = "risk_radiorag_checked.jsonl"

# Pairs to test (left, right). Delta = accuracy(left) - accuracy(right).
DEFAULT_PAIRS: list[tuple[str, str]] = [
    ("evidence_clean",    "zero_shot"),         # clean vs closed
    ("evidence_conflict", "evidence_clean"),    # conflict vs clean
    ("top_1",             "zero_shot"),         # top-1 RAG vs closed
    ("top_5",             "zero_shot"),
    ("top_10",            "zero_shot"),
    ("deep_research",     "top_10"),            # RaR vs standard RAG
    ("deep_research",     "zero_shot"),
    ("context_100k",      "zero_shot"),
    ("context_max",       "zero_shot"),
    ("context_100k",      "top_10"),
    ("context_max",       "top_10"),
    ("context_max",       "context_100k"),
]


# ---------------------------------------------------------------------------
# Correctness (same logic as run_bootstrap.py)
# ---------------------------------------------------------------------------
def _confirmed(checked: Optional[dict]) -> Optional[str]:
    if not isinstance(checked, dict):
        return None
    ans = checked.get("confirmed_answer")
    return ans if isinstance(ans, str) and ans else None


def greedy_correct(cond_block: dict, truth: str) -> int:
    greedy = (cond_block or {}).get("greedy", {}) or {}
    pred = _confirmed(greedy.get("checked_answer"))
    return int(pred is not None and pred == truth)


def majority_correct(cond_block: dict, truth: str) -> int:
    stoch = (cond_block or {}).get("stochastic", {}) or {}
    checks = stoch.get("checked_answers")
    if isinstance(checks, list) and checks:
        preds = [_confirmed(c) for c in checks]
    else:
        preds = [p if isinstance(p, str) and p else None
                 for p in stoch.get("parsed_answers", [])]
    preds = [p for p in preds if p is not None]
    if not preds:
        return 0
    top_answer, _ = Counter(preds).most_common(1)[0]
    return int(top_answer == truth)


# ---------------------------------------------------------------------------
# McNemar
# ---------------------------------------------------------------------------
def mcnemar_exact_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) * (0.5 ** n) for i in range(k + 1))
    p_two = 2.0 * tail
    return min(p_two, 1.0)


def mcnemar_stats(left: np.ndarray, right: np.ndarray):
    b = int(np.sum((left == 1) & (right == 0)))  # left correct, right wrong
    c = int(np.sum((left == 0) & (right == 1)))
    n = b + c
    return n, b, c, mcnemar_exact_p(b, c)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def load_records(path: Path) -> list[dict]:
    recs = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    recs.sort(key=lambda r: r.get("question_id", ""))
    return recs


def build_correctness(records: list[dict], mode: str) -> dict[str, np.ndarray]:
    """Return {condition: int8 array of length n}."""
    truths = [r.get("correct_answer", "") for r in records]
    n = len(records)
    conds = set()
    for r in records:
        conds.update((r.get("conditions") or {}).keys())

    fn = greedy_correct if mode == "greedy" else majority_correct
    out: dict[str, np.ndarray] = {}
    for cond in conds:
        arr = np.zeros(n, dtype=np.int8)
        for i, rec in enumerate(records):
            block = (rec.get("conditions") or {}).get(cond)
            if block:
                arr[i] = fn(block, truths[i])
        out[cond] = arr
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--modes", default="greedy,majority",
                    help="Correctness modes to test (comma-separated).")
    ap.add_argument("--models", default="",
                    help="Optional comma-separated model dir names to restrict.")
    args = ap.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    filt = {m.strip() for m in args.models.split(",") if m.strip()} or None

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    model_dirs = sorted(d for d in PHASE4_DIR.iterdir() if d.is_dir())
    if filt:
        model_dirs = [d for d in model_dirs if d.name in filt]

    for mdir in model_dirs:
        jf = mdir / DATASET_BASENAME
        if not jf.exists():
            continue
        records = load_records(jf)
        if not records:
            continue

        for mode in modes:
            correct = build_correctness(records, mode)
            for left, right in DEFAULT_PAIRS:
                if left not in correct or right not in correct:
                    continue
                l_arr, r_arr = correct[left], correct[right]
                n_disc, b, c, pv = mcnemar_stats(l_arr, r_arr)
                acc_l = float(l_arr.mean()) * 100
                acc_r = float(r_arr.mean()) * 100
                rows.append({
                    "model": mdir.name,
                    "mode": mode,
                    "left": left,
                    "right": right,
                    "acc_left": round(acc_l, 3),
                    "acc_right": round(acc_r, 3),
                    "delta": round(acc_l - acc_r, 3),
                    "n_discordant": n_disc,
                    "b_left_correct_right_wrong": b,
                    "c_left_wrong_right_correct": c,
                    "p_value": round(pv, 6),
                })
        print(f"[+] {mdir.name}: pairs tested")

    # ---- write -----------------------------------------------------------
    csv_path = out_dir / "pvalues_all.csv"
    with csv_path.open("w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    with (out_dir / "pvalues_all.json").open("w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n[+] Wrote {len(rows)} pairwise tests -> {csv_path}")


if __name__ == "__main__":
    main()
