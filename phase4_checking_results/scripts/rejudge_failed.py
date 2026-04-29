#!/usr/bin/env python3
"""
Surgically re-run the LLM-judge for any phase-4 sample whose
``checked_answer.error`` field is set (judge call failed previously).

This does NOT touch any sample that already has a successful judge result.
Patches each JSONL file in place; writes a small JSON report summarising
recovered / still-failed counts.

Designed to share the prompt + parsing logic with
``run_answer_check.py`` so re-judged samples are byte-identical to a
fresh run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import time
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
sys.path.insert(0, str(THIS_FILE.parent))
import run_answer_check as rac  # noqa: E402   (reuse prompt + parser)

P4_RESULTS = Path("/path/to/workspace/phase4_checking_results/results")
DATASET_FN = "risk_radiorag_checked.jsonl"
DATASET_UNIFIED = Path("/path/to/workspace/datasets/standardized/risk_radiorag_unified.jsonl")
EXCLUDE = {"Qwen3.5-9B", "Qwen3.5-27B", "Qwen3.5-35B-A3B", "Qwen3.5-122B-A10B"}


# ---------------------------------------------------------------------------
def find_failed_samples():
    """Return list of dicts describing every sample to re-judge.

    Each entry has: model_dir, file, line_idx, condition, kind ("greedy" or
    "stoch[i]"), question_id, raw_output. The patch loop later uses these to
    update the corresponding cells.
    """
    work = []
    for mdir in sorted(d for d in P4_RESULTS.iterdir() if d.is_dir()):
        if mdir.name in EXCLUDE:
            continue
        jf = mdir / DATASET_FN
        if not jf.exists():
            continue
        with jf.open() as f:
            for line_idx, line in enumerate(f):
                rec = json.loads(line)
                qid = rec.get("question_id")
                for cond, block in (rec.get("conditions") or {}).items():
                    g = (block or {}).get("greedy") or {}
                    ck = g.get("checked_answer") or {}
                    if isinstance(ck, dict) and ("error" in ck) and not ck.get("confirmed_answer"):
                        work.append({
                            "model_dir": mdir.name,
                            "file":      str(jf),
                            "line_idx":  line_idx,
                            "qid":       qid,
                            "condition": cond,
                            "kind":      "greedy",
                            "stoch_idx": None,
                            "raw_output":   g.get("raw_output"),
                            "parsed_answer": g.get("parsed_answer"),
                        })
                    s = (block or {}).get("stochastic") or {}
                    checks = s.get("checked_answers") or []
                    raws   = s.get("raw_outputs") or []
                    parseds = s.get("parsed_answers") or []
                    for i, c in enumerate(checks):
                        if isinstance(c, dict) and ("error" in c) and not c.get("confirmed_answer"):
                            work.append({
                                "model_dir": mdir.name,
                                "file":      str(jf),
                                "line_idx":  line_idx,
                                "qid":       qid,
                                "condition": cond,
                                "kind":      f"stoch[{i}]",
                                "stoch_idx": i,
                                "raw_output":   raws[i]    if i < len(raws)    else None,
                                "parsed_answer": parseds[i] if i < len(parseds) else None,
                            })
    return work


# ---------------------------------------------------------------------------
async def rejudge(work: list[dict], checker: rac.Checker, qbank: dict) -> list[dict]:
    """Call the checker on each work item; attach `new_check` field with result."""
    sem = asyncio.Semaphore(64)

    async def one(item):
        async with sem:
            qinfo = qbank.get(item["qid"])
            if qinfo is None:
                item["new_check"] = {"reasoning": "qid not in bank",
                                     "confirmed_answer": None,
                                     "error": "qid_missing"}
                return
            try:
                item["new_check"] = await checker.check_one(
                    item["raw_output"], item.get("parsed_answer"), qinfo)
            except Exception as e:
                item["new_check"] = {"reasoning": "", "confirmed_answer": None,
                                     "error": f"rejudge_outer:{e}"[:300]}

    await asyncio.gather(*(one(it) for it in work))
    return work


def apply_patches(work: list[dict]) -> dict:
    """Write each new_check back into the relevant JSONL line; return stats."""
    by_file: dict[str, list[dict]] = {}
    for it in work:
        by_file.setdefault(it["file"], []).append(it)

    stats = {"recovered": 0, "still_failed": 0, "by_model": {}}
    for path, items in by_file.items():
        p = Path(path)
        bak = p.with_suffix(p.suffix + ".pre_rejudge.bak")
        if not bak.exists():
            shutil.copy2(p, bak)
        with p.open() as f:
            lines = f.readlines()

        # Index items by line for in-place patching
        idx_map: dict[int, list[dict]] = {}
        for it in items:
            idx_map.setdefault(it["line_idx"], []).append(it)

        for line_idx, patches in idx_map.items():
            rec = json.loads(lines[line_idx])
            for it in patches:
                cond = it["condition"]
                block = (rec["conditions"] or {}).get(cond) or {}
                if it["stoch_idx"] is None:
                    g = block.get("greedy") or {}
                    g["checked_answer"] = it["new_check"]
                    block["greedy"] = g
                else:
                    s = block.get("stochastic") or {}
                    arr = list(s.get("checked_answers") or [])
                    while len(arr) <= it["stoch_idx"]:
                        arr.append(None)
                    arr[it["stoch_idx"]] = it["new_check"]
                    s["checked_answers"] = arr
                    block["stochastic"] = s
                rec["conditions"][cond] = block

                ok = bool(it["new_check"].get("confirmed_answer"))
                stats["recovered" if ok else "still_failed"] += 1
                key = it["model_dir"]
                stats["by_model"].setdefault(key, {"recovered": 0, "still_failed": 0})
                stats["by_model"][key]["recovered" if ok else "still_failed"] += 1
            lines[line_idx] = json.dumps(rec) + "\n"

        with p.open("w") as f:
            f.writelines(lines)
    return stats


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=6000)
    ap.add_argument("--base-url", default=None,
                    help="Override base URL (default http://localhost:PORT/v1)")
    ap.add_argument("--checker-model-path", default=rac.CHECKER_MODEL_PATH)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--skip-wait", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="Just list the failed cases, do not call the checker")
    args = ap.parse_args()

    base_url = args.base_url or f"http://localhost:{args.port}/v1"

    print("[+] Scanning phase-4 results for judge errors ...")
    work = find_failed_samples()
    print(f"[+] Found {len(work)} samples to re-judge")
    if not work:
        return

    if args.dry_run:
        from collections import Counter
        c = Counter(it["model_dir"] for it in work)
        for m, n in c.most_common():
            print(f"   {m:40s}  {n}")
        return

    if not args.skip_wait:
        if not rac.wait_for_vllm(base_url):
            sys.exit("ERROR: vLLM not reachable")

    qbank = rac.load_question_bank(DATASET_UNIFIED)
    print(f"[+] Question bank: {len(qbank)} entries")

    checker = rac.Checker(base_url, args.concurrency, args.checker_model_path)

    t0 = time.time()
    asyncio.run(rejudge(work, checker, qbank))
    print(f"[+] Re-judging done in {time.time()-t0:.1f}s")

    stats = apply_patches(work)
    print(f"\n=== Patch stats ===")
    print(f"  recovered (now have a letter): {stats['recovered']}")
    print(f"  still_failed (still null):     {stats['still_failed']}")
    for m, d in sorted(stats["by_model"].items()):
        print(f"   {m:40s}  rec={d['recovered']:>3}  still={d['still_failed']:>3}")

    report = Path("/path/to/workspace/phase4_checking_results/rejudge_report.json")
    report.write_text(json.dumps({
        "timestamp": time.time(),
        "n_attempted": len(work),
        **stats,
    }, indent=2))
    print(f"\n[+] Report -> {report}")


if __name__ == "__main__":
    main()
