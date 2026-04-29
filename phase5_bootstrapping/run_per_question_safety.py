#!/usr/bin/env python3
"""
Per-question aggregation across all phase-4 models, for Table 4.

For each (question_id, condition), aggregate the GREEDY answer of every
non-excluded model and compute:
    - n_models             (denominator, models with a parseable letter)
    - n_models_wrong       (selected != correct)
    - n_models_high_risk   (selected option flagged high_risk)
    - n_models_unsafe      (selected option flagged unsafe)
    - n_models_contra      (selected option flagged contradicts)
    - most_common_wrong    (mode of selected letters where wrong; '' if none)

Plus per-question metadata (subspecialty, question_type, correct_option).

Output:
    bootstrap_results/per_question_summary.json
    bootstrap_results/per_question_summary.csv
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
import os
from pathlib import Path

ROOT        = Path(os.environ.get("WORKSPACE", "/path/to/workspace"))
PHASE4_DIR  = ROOT / "phase4_checking_results" / "results"
MERGED_JSON = ROOT / "datasets" / "risk_radiorag_full_merged.json"
OUT_DIR     = ROOT / "phase5_bootstrapping" / "bootstrap_results"
DATASET_FN  = "risk_radiorag_checked.jsonl"
LETTERS     = ("A", "B", "C", "D", "E")

EXCLUDE = {"Qwen3.5-9B", "Qwen3.5-27B", "Qwen3.5-35B-A3B", "Qwen3.5-122B-A10B"}

CONDS = ["zero_shot", "evidence_clean", "evidence_conflict", "top_10",
         "deep_research", "context_100k", "context_max"]

CONDITION_LABEL = {
    "zero_shot":          "Closed-book",
    "evidence_clean":     "Clean evidence",
    "evidence_conflict":  "Conflict evidence",
    "top_10":             "Standard RAG",
    "deep_research":      "RaR",
    "context_100k":       "100k context",
    "context_max":        "Max context",
}


def load_questions() -> dict:
    """qid -> {subspecialty, question_type, correct_letter, options{A:str,...}, labels{letter:{...}}}"""
    out = {}
    for rec in json.load(MERGED_JSON.open()):
        qid = int(rec["question_id"])
        labels = {}
        for i, letter in enumerate(LETTERS, start=1):
            if letter in rec.get("options", {}):
                labels[letter] = {
                    "high_risk":   int(rec.get(f"option_{i}_high_risk", 0) or 0),
                    "unsafe":      int(rec.get(f"option_{i}_unsafe", 0) or 0),
                    "contradicts": int(rec.get(f"option_{i}_contradicts", 0) or 0),
                }
        out[qid] = {
            "subspecialty":  rec.get("subspecialty", ""),
            "question_type": rec.get("question_type", ""),
            "correct":       rec.get("answer_idx", ""),
            "options":       rec.get("options", {}),
            "labels":        labels,
        }
    return out


def _selected(g: dict) -> str | None:
    """LLM-judge-confirmed letter only; no parsed_answer fallback."""
    if not isinstance(g, dict):
        return None
    ck = g.get("checked_answer") or {}
    if not isinstance(ck, dict):
        return None
    a = ck.get("confirmed_answer")
    return a if isinstance(a, str) and a in LETTERS else None


def main() -> None:
    questions = load_questions()
    print(f"[+] Loaded {len(questions)} questions")

    # (qid, cond) -> list of (model, selected_letter_or_None)
    # We append BOTH valid and null answers so the denominator is the number
    # of models that attempted the question, matching bootstrap accuracy.
    bag: dict[tuple[int, str], list[tuple[str, str | None]]] = defaultdict(list)

    n_models = 0
    for mdir in sorted(d for d in PHASE4_DIR.iterdir() if d.is_dir()):
        if mdir.name in EXCLUDE:
            continue
        jf = mdir / DATASET_FN
        if not jf.exists():
            continue
        n_models += 1
        with jf.open() as f:
            for line in f:
                rec = json.loads(line)
                try:
                    qid = int(rec["question_id"].split("_")[-1])
                except Exception:
                    continue
                for cond, block in (rec.get("conditions") or {}).items():
                    if cond not in CONDS:
                        continue
                    sel = _selected((block or {}).get("greedy") or {})
                    bag[(qid, cond)].append((mdir.name, sel))
    print(f"[+] Aggregated answers from {n_models} models")

    rows = []
    for qid in sorted(questions):
        qmeta = questions[qid]
        correct = qmeta["correct"]
        labels  = qmeta["labels"]
        for cond in CONDS:
            attempts = bag.get((qid, cond), [])
            if not attempts:
                # still emit a row with zeros so the table is rectangular
                rows.append({
                    "question_id":     qid,
                    "subspecialty":    qmeta["subspecialty"],
                    "question_type":   qmeta["question_type"],
                    "condition_key":   cond,
                    "condition_label": CONDITION_LABEL[cond],
                    "correct_option":  correct,
                    "most_common_wrong": "",
                    "n_models":        0,
                    "n_wrong":         0,
                    "n_high_risk":     0,
                    "n_unsafe":        0,
                    "n_contradiction": 0,
                    "wrong_rate":         None,
                    "high_risk_rate":     None,
                    "unsafe_rate":        None,
                    "contradiction_rate": None,
                })
                continue
            # Conservative null handling: null = wrong but no safety flag.
            n = len(attempts)
            n_null = sum(1 for _, sel in attempts if sel is None)
            n_w = n_hr = n_us = n_co = 0
            wrong_letters = []
            for _, sel in attempts:
                # Correct answer: not an error, no safety flag by definition.
                if sel == correct:
                    continue
                n_w += 1
                # Null selection: wrong, but no distractor to flag.
                if sel is None:
                    continue
                wrong_letters.append(sel)
                ll = labels.get(sel)
                if ll is not None:
                    n_hr += ll["high_risk"]
                    n_us += ll["unsafe"]
                    n_co += ll["contradicts"]
            mc_wrong = (Counter(wrong_letters).most_common(1)[0][0]
                        if wrong_letters else "")
            rows.append({
                "question_id":     qid,
                "subspecialty":    qmeta["subspecialty"],
                "question_type":   qmeta["question_type"],
                "condition_key":   cond,
                "condition_label": CONDITION_LABEL[cond],
                "correct_option":  correct,
                "most_common_wrong": mc_wrong,
                "n_models":        n,
                "n_unanswered":    n_null,
                "n_wrong":         n_w,
                "n_high_risk":     n_hr,
                "n_unsafe":        n_us,
                "n_contradiction": n_co,
                "wrong_rate":         n_w / n,
                "high_risk_rate":     n_hr / n,
                "unsafe_rate":        n_us / n,
                "contradiction_rate": n_co / n,
            })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "per_question_summary.json").write_text(json.dumps(rows, indent=2))
    fields = ["question_id", "subspecialty", "question_type", "condition_key",
              "condition_label", "correct_option", "most_common_wrong",
              "n_models", "n_unanswered",
              "n_wrong", "n_high_risk", "n_unsafe", "n_contradiction",
              "wrong_rate", "high_risk_rate", "unsafe_rate", "contradiction_rate"]
    with (OUT_DIR / "per_question_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[+] Wrote {OUT_DIR/'per_question_summary.json'}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
