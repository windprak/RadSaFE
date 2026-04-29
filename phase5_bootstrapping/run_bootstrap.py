#!/usr/bin/env python3
"""
Phase 5 bootstrap on phase 4 (checked) results.

For every model directory under
    /path/to/workspace/phase4_checking_results/results/<MODEL>/
        risk_radiorag_checked.jsonl
we compute, for every condition present in the JSONL:

  * greedy correctness: 1 if checked_answer.confirmed_answer == correct_answer,
    else 0 (null / missing -> 0, per user preference).
  * majority correctness: majority vote over stochastic checked_answers,
    dropping null/missing entries before voting; if nothing remains -> 0.

We then bootstrap B resamples of the per-question correctness vector (fixed
seed, same resampling indices reused across every (model, condition) pair so
results are directly comparable) and report mean / std / 95% percentile CI.

Outputs
-------
<OUT_DIR>/bootstrap_indices.json              (B lists of length n)
<OUT_DIR>/bootstrap_means_<model>__<cond>__<mode>.csv    (B values, percent)
<OUT_DIR>/summary/<model>.json                (per-model compact summary)
<OUT_DIR>/summary_all.csv                     (long-form summary table)
<OUT_DIR>/summary_all.json                    (same, JSON)

Run it directly:
    python run_bootstrap.py
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PHASE4_DIR = Path("/path/to/workspace/phase4_checking_results/results")
DEFAULT_OUT = Path("/path/to/workspace/phase5_bootstrapping/bootstrap_results")
DATASET_BASENAME = "risk_radiorag_checked.jsonl"


# ---------------------------------------------------------------------------
# Correctness extraction
# ---------------------------------------------------------------------------
def _confirmed(checked: Optional[dict]) -> Optional[str]:
    if not isinstance(checked, dict):
        return None
    ans = checked.get("confirmed_answer")
    return ans if isinstance(ans, str) and ans else None


def greedy_correct(cond_block: dict, truth: str) -> int:
    """Correctness of the greedy sample using the phase-4 checker.
    Null / missing confirmed_answer counts as incorrect."""
    greedy = cond_block.get("greedy", {}) or {}
    pred = _confirmed(greedy.get("checked_answer"))
    return int(pred is not None and pred == truth)


def majority_correct(cond_block: dict, truth: str) -> int:
    """Majority vote over stochastic checked_answers.

    Conservative null handling: a null/missing confirmed_answer is treated as
    its own ``"NULL"`` ballot. If ``"NULL"`` ties or wins the vote, the row
    counts as wrong. This keeps the denominator at 20 samples and prevents
    a model that abstains on most samples from "winning" majority via a
    handful of valid-but-uncertain answers."""
    stoch = cond_block.get("stochastic", {}) or {}
    checks = stoch.get("checked_answers")
    if isinstance(checks, list) and checks:
        preds = [_confirmed(c) or "NULL" for c in checks]
    else:
        preds = [p if isinstance(p, str) and p else "NULL"
                 for p in stoch.get("parsed_answers", [])]
    if not preds:
        return 0
    counts = Counter(preds)
    top_count = counts.most_common(1)[0][1]
    tied = [a for a, c in counts.items() if c == top_count]
    # If NULL is among the tied winners (or sole winner) -> wrong.
    if "NULL" in tied:
        return 0
    # Otherwise tie-break alphabetically (deterministic), then compare to truth.
    return int(sorted(tied)[0] == truth)


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------
def get_bootstrap_indices(out_dir: Path, n: int, B: int, seed: int) -> np.ndarray:
    idx_path = out_dir / "bootstrap_indices.json"
    if idx_path.exists():
        with idx_path.open() as f:
            data = json.load(f)
        if data.get("n") == n and data.get("B") == B and data.get("seed") == seed:
            return np.asarray(data["indices"], dtype=np.int32)
        print(f"[!] {idx_path.name} mismatch (n/B/seed); regenerating.")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n), dtype=np.int32)
    with idx_path.open("w") as f:
        json.dump({"n": n, "B": B, "seed": seed, "indices": idx.tolist()}, f)
    print(f"[+] Wrote bootstrap indices -> {idx_path}")
    return idx


def bootstrap_stats(correct: np.ndarray, idx: np.ndarray):
    """correct: (n,) int8 0/1. idx: (B, n). Returns (mean%, std%, lo%, hi%, dist_pct).
    `dist_pct` is a (B,) numpy array of per-resample accuracies in percent."""
    sampled = correct[idx]               # (B, n)
    per_boot = sampled.mean(axis=1)      # (B,)
    dist = per_boot * 100.0
    mean = float(dist.mean())
    std = float(dist.std(ddof=1))
    lo, hi = np.percentile(dist, [2.5, 97.5]).tolist()
    return mean, std, float(lo), float(hi), dist


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def load_model_records(path: Path) -> list[dict]:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def process_all(out_dir: Path, B: int, seed: int, models_filter: set[str] | None):
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = out_dir / "summary"
    dist_dir = out_dir / "dists"
    summary_dir.mkdir(exist_ok=True)
    dist_dir.mkdir(exist_ok=True)

    model_dirs = sorted(d for d in PHASE4_DIR.iterdir() if d.is_dir())
    if models_filter:
        model_dirs = [d for d in model_dirs if d.name in models_filter]

    # Use the first model to discover n (should be 200 for risk_radiorag).
    n_expected = None
    idx = None
    all_rows: list[dict] = []

    for mdir in model_dirs:
        jf = mdir / DATASET_BASENAME
        if not jf.exists():
            print(f"  [skip] {mdir.name}: no {DATASET_BASENAME}")
            continue
        records = load_model_records(jf)
        if not records:
            print(f"  [skip] {mdir.name}: empty file")
            continue

        # Sort by question_id for determinism
        records.sort(key=lambda r: r.get("question_id", ""))
        n = len(records)
        if n_expected is None:
            n_expected = n
            idx = get_bootstrap_indices(out_dir, n, B, seed)
        elif n != n_expected:
            print(f"  [warn] {mdir.name}: n={n} != expected {n_expected}; skipping")
            continue

        truths = [r.get("correct_answer", "") for r in records]

        # Collect the set of conditions present for this model
        cond_names: list[str] = []
        seen = set()
        for r in records:
            for c in r.get("conditions", {}).keys():
                if c not in seen:
                    seen.add(c)
                    cond_names.append(c)

        per_model: dict = {"model": mdir.name, "n": n, "conditions": {}}

        for cond in cond_names:
            g_arr = np.zeros(n, dtype=np.int8)
            m_arr = np.zeros(n, dtype=np.int8)
            g_null = 0
            for i, rec in enumerate(records):
                block = rec.get("conditions", {}).get(cond)
                if not block:
                    # condition absent for this question -> count as incorrect
                    continue
                t = truths[i]
                if _confirmed((block.get("greedy") or {}).get("checked_answer")) is None:
                    g_null += 1
                g_arr[i] = greedy_correct(block, t)
                m_arr[i] = majority_correct(block, t)

            for mode, arr in (("greedy", g_arr), ("majority", m_arr)):
                mean, std, lo, hi, dist = bootstrap_stats(arr, idx)
                correct_n = int(arr.sum())
                row = {
                    "model": mdir.name,
                    "condition": cond,
                    "mode": mode,
                    "n": n,
                    "correct": correct_n,
                    "accuracy_point": round(correct_n / n * 100, 4),
                    "mean": round(mean, 4),
                    "std": round(std, 4),
                    "ci_lower": round(lo, 4),
                    "ci_upper": round(hi, 4),
                    "greedy_null_checker": g_null if mode == "greedy" else None,
                    "pretty": f"{mean:.1f} \u00b1 {std:.1f} [{lo:.1f}, {hi:.1f}]",
                }
                all_rows.append(row)
                per_model["conditions"].setdefault(cond, {})[mode] = row

                dist_path = dist_dir / f"{mdir.name}__{cond}__{mode}.csv"
                np.savetxt(dist_path, dist, fmt="%.6f")

        with (summary_dir / f"{mdir.name}.json").open("w") as f:
            json.dump(per_model, f, indent=2)
        print(f"[+] {mdir.name}: {len(cond_names)} conditions processed")

    # ---- global long-form summary -----------------------------------------
    fields = ["model", "condition", "mode", "n", "correct",
              "accuracy_point", "mean", "std", "ci_lower", "ci_upper",
              "greedy_null_checker", "pretty"]
    csv_path = out_dir / "summary_all.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)
    with (out_dir / "summary_all.json").open("w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\n[+] Summary -> {csv_path}")
    print(f"    Rows: {len(all_rows)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="Output directory (default: ./bootstrap_results)")
    ap.add_argument("-B", "--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models", default="",
                    help="Optional comma-separated model-dir names to restrict.")
    args = ap.parse_args()

    filt = {m.strip() for m in args.models.split(",") if m.strip()} or None
    process_all(Path(args.out), args.bootstrap, args.seed, filt)


if __name__ == "__main__":
    main()
