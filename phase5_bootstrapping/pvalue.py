from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# McNemar helpers
# ---------------------------------------------------------------------------
def mcnemar_exact_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) * (0.5 ** n) for i in range(k + 1))
    p_two = 2.0 * tail
    return p_two if p_two <= 1.0 else 1.0


def mcnemar_stats(a: np.ndarray, b_: np.ndarray) -> Tuple[int, int, int, float]:
    if a.shape != b_.shape:
        raise ValueError("length mismatch")
    b = int(np.sum((a == 1) & (b_ == 0)))
    c = int(np.sum((a == 0) & (b_ == 1)))
    return b + c, b, c, mcnemar_exact_p(b, c)

# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
ALLOWED = {"deepresearch", "norag", "radiorag"}


def load_correct(fp: Path) -> np.ndarray:
    with fp.open(encoding="utf-8") as f:
        data = json.load(f)
    return np.fromiter((int(d["correct"]) for d in data), dtype=np.int8)


def discover(root: Path) -> Dict[Tuple[str, str, str], np.ndarray]:
    out: Dict[Tuple[str, str, str], np.ndarray] = {}
    for fp in root.rglob("results_*.json"):
        if any(part.lower() == "old" for part in fp.parts):
            print("Ignoring old results:", fp, file=sys.stderr)
            continue
        parts = [p.lower() for p in fp.parts]
        try:
            i = next(j for j, p in enumerate(parts) if p in ALLOWED)
        except StopIteration:
            continue
        if len(parts) < i + 3:
            continue
        key = tuple(parts[i : i + 3])  # (cat, strat, model)
        if key in out:
            print(f"[!] duplicate {key}, keeping first", file=sys.stderr)
            continue
        try:
            out[key] = load_correct(fp)
        except Exception as e:
            print(f"[!] failed {fp}: {e}", file=sys.stderr)
    return out

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="results root")
    ap.add_argument("-o", "--out", default="paired_stats", help="output folder")
    ap.add_argument("--strict-length", action="store_true",
                    help="abort if array lengths differ")
    return ap.parse_args()

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    p = args()
    root = Path(p.root).expanduser().resolve()
    out_dir = Path(p.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    res = discover(root)
    if not res:
        sys.exit("[!] no results found")

    deep_strats = ["cot_2steps"]
    ref_strats: List[Tuple[str, str]] = [
        ("norag", "cot_2steps"),
        ("norag", "zero_shot"),
        ("radiorag", "radio"),
        ("deepresearch", "zero_shot_basedon"),
    ]

    rows: List[Tuple[str, str, str, str, int, int, int, float]] = []
    by_model: Dict[str, List[Dict]] = defaultdict(list)

    for deep in deep_strats:
        left_key = ("deepresearch", deep)

        for ref_cat, ref_strat in ref_strats:
            right_key = (ref_cat, ref_strat)

            for (cat, strat, model), left_arr in res.items():
                if (cat, strat) != left_key:
                    continue
                right_arr = res.get((right_key[0], right_key[1], model))
                if right_arr is None:
                    continue

                if left_arr.shape != right_arr.shape:
                    msg = (f"[!] length mismatch {model}: "
                           f"{left_key}={left_arr.size}, {right_key}={right_arr.size}")
                    if p.strict_length:
                        sys.exit(msg)
                    print(msg, file=sys.stderr)
                    continue

                n, b, c, pv = mcnemar_stats(left_arr, right_arr)
                # if pv>0.5:
                #     pv = 1-pv
                # if pv == 0:
                #     pv = 0.001
                rec = (deep, ref_cat, ref_strat, model, n, b, c, round(pv, 6))
                rows.append(rec)

                by_model[model].append({
                    "left": f"deepresearch/{deep}",
                    "right": f"{ref_cat}/{ref_strat}",
                    "n": n, "b": b, "c": c, "p": round(pv, 6),
                })

    if not rows:
        sys.exit("No overlapping pairs.")

    # ---------- CSV / JSON (unchanged) ---------------------------------
    header = ["deepresearch_strategy", "reference_category", "reference_strategy",
              "model", "n_discordant", "b_DR_correct_ref_wrong",
              "c_DR_wrong_ref_correct", "p_value"]

    csv_path = out_dir / "pval_all_deepresearch_vs_reference.csv"
    with csv_path.open("w", newline="") as f:
        csv.writer(f).writerows([header, *rows])

    json_path = out_dir / "pval_all_deepresearch_vs_reference.json"
    with json_path.open("w") as f:
        json.dump([
            dict(zip(header, r)) for r in rows
        ], f, indent=2)

    # ---------- NEW: console summary grouped by model ------------------
    for m in sorted(by_model):
        print(f"\n=== Model: {m} " + "=" * (50 - len(m)))
        for rec in by_model[m]:
            print(f"  {rec['left']:<25} vs {rec['right']:<20}: "
                  f"p = {rec['p']:.4f}  "
                  f"(n={rec['n']}, b={rec['b']}, c={rec['c']})")

    print(f"\n[+] Files written:\n    {csv_path}\n    {json_path}\n",
          file=sys.stderr)


if __name__ == "__main__":
    main()
