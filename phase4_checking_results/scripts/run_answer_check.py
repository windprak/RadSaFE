#!/usr/bin/env python3
"""
Phase 4 — Verify every `raw_output` against its `parsed_answer` using
Mistral-Small-4-119B served by vLLM (OpenAI-compatible endpoint).

For every record in phase3_inference/results/<MODEL>/risk_radiorag.jsonl,
for every condition, for greedy and each stochastic sample, ask the checker
model to produce:

    {"reasoning": "...", "confirmed_answer": "A"|"B"|"C"|"D"|"E"|null}

and attach it to the record:

    conditions[cond]["greedy"]["checked_answer"]          = {...}
    conditions[cond]["stochastic"]["checked_answers"]     = [{...}, ...]

Output is written to:
    /path/to/workspace/phase4_checking_results/results/<MODEL>/risk_radiorag_checked.jsonl

Per-question resume is supported (re-runs skip question_ids already finished).

Usage:
    python3 run_answer_check.py --port 6000 --concurrency 128
    python3 run_answer_check.py --model gpt-oss-120b --concurrency 64
    python3 run_answer_check.py --models-filter 'DeepSeek-R1-671B,Qwen3-32B'
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import re
import sys
import time
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(os.environ.get("WORKSPACE", "/path/to/workspace"))
PHASE3_RES    = BASE_DIR / "phase3_inference" / "results"
PHASE4_DIR    = BASE_DIR / "phase4_checking_results"
PHASE4_RES    = PHASE4_DIR / "results"
LOGS_DIR      = PHASE4_DIR / "logs"

CHECKER_MODEL_NAME = "Mistral-Small-4-119B-2603"
# Container path vLLM registers under (matches phase3 convention)
CHECKER_MODEL_PATH = "/models/Mistral-Small-4-119B-2603"

# Source dataset with question_text + options keyed by question_id
DATASET_UNIFIED = BASE_DIR / "datasets" / "standardized" / "risk_radiorag_unified.jsonl"

DATASET_FILENAME          = "risk_radiorag.jsonl"
OUTPUT_DATASET_FILENAME   = "risk_radiorag_checked.jsonl"
MISMATCH_FILENAME         = "mismatches.jsonl"
MISMATCH_PREVIEW_CHARS    = 600

ANSWER_LETTERS = ["A", "B", "C", "D", "E"]

# ── Prompt ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a strict extractor, not an interpreter. You receive (1) a "
    "multiple-choice medical question with its lettered options, and (2) the "
    "raw text output of another LLM that was asked to answer that question. "
    "Your ONLY job is to find whether the raw text contains an explicit final "
    "selection of one of the letters A, B, C, D, or E.\n\n"
    "RULES (follow them strictly):\n"
    "  1. First decide: is there an EXPLICIT letter selection in the raw text? "
    "Answer `clear_selection: \"yes\"` or `\"no\"` BEFORE writing reasoning.\n"
    "  2. An explicit selection looks like: a standalone letter 'A'..'E', "
    "'Option B', 'Answer: C', 'The answer is D', '$\\\\boxed{A}$', "
    "'<unused95>B', '**E**', etc. The letter must be clearly marked as the "
    "model's final choice, not just mentioned while discussing options.\n"
    "  3. You MUST NOT infer a letter from a diagnosis/entity name even if it "
    "matches exactly one of the options. If the model only writes 'the "
    "diagnosis is X' without ever writing a letter, `clear_selection` is "
    "\"no\" and `confirmed_answer` is null. No semantic mapping. No guessing.\n"
    "  4. If the text mentions multiple letters (chain-of-thought comparing "
    "options) but never commits to one, `clear_selection` is \"no\".\n"
    "  5. You are NOT told the correct answer and you are NOT asked to judge "
    "correctness.\n"
    "  6. If `clear_selection` is \"no\", `confirmed_answer` MUST be null.\n\n"
    "You MUST respond with a single valid JSON object and nothing else."
)

USER_TEMPLATE = """QUESTION (for context only; do NOT use it to infer the letter):
{question_text}

OPTIONS (for context only; do NOT map diagnosis/entity names to letters):
{options_block}

Answer that was extracted by the automatic parser: {parsed}

Below is the full raw text that the model produced. Scan it for an EXPLICIT
letter selection (A/B/C/D/E) as defined in the system rules. If none is
present, return `clear_selection: "no"` and `confirmed_answer: null`.

RAW MODEL OUTPUT (delimited by <<< and >>>):
<<<
{raw}
>>>

Respond with JSON ONLY, exactly in this schema and field order:
{{
  "clear_selection": "yes" | "no",
  "reasoning": "<one or two short sentences; if yes, quote the snippet where the letter appears; if no, say why>",
  "confirmed_answer": "A" | "B" | "C" | "D" | "E" | null
}}"""

# Verifier generation params
CHECK_PARAMS = {
    "temperature": 0.0,
    "max_tokens":  384,
    "n":           1,
    "response_format": {"type": "json_object"},
}

MAX_RAW_CHARS = 24_000  # safety cap on raw input we forward to the checker


# ── JSON helpers ──────────────────────────────────────────────────────────────

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _coerce_letter(v) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip().upper()
        if s in ANSWER_LETTERS:
            return s
        # tolerate "answer: B" etc.
        m = re.search(r"\b([ABCDE])\b", s)
        if m:
            return m.group(1)
    return None


def _coerce_yesno(v) -> Optional[str]:
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("yes", "y", "true"):
            return "yes"
        if s in ("no", "n", "false"):
            return "no"
    return None


def parse_checker_json(text: str, model_raw: Optional[str] = None) -> Dict:
    """Best-effort parse of the checker's JSON reply.

    Null-safety: ``confirmed_answer`` is guaranteed to be ``None`` unless
    (a) ``clear_selection == "yes"`` AND (b) the chosen letter physically
    appears as a standalone A/B/C/D/E token in the original ``model_raw``.
    """
    empty = {"clear_selection": None, "reasoning": "", "confirmed_answer": None}
    if not text:
        return {**empty, "parse_error": "empty"}
    raw = text.strip()
    # Try direct
    try:
        obj = json.loads(raw)
    except Exception:
        m = _JSON_OBJ_RE.search(raw)
        if not m:
            return {**empty, "parse_error": "no_json", "raw_reply": raw[:500]}
        try:
            obj = json.loads(m.group(0))
        except Exception as e:
            return {**empty, "parse_error": f"decode:{e}", "raw_reply": raw[:500]}

    clear     = _coerce_yesno(obj.get("clear_selection"))
    reasoning = str(obj.get("reasoning", ""))[:2000]
    letter    = _coerce_letter(obj.get("confirmed_answer"))

    # Guard 1: rule 6 — if not a clear selection, confirmed_answer must be null.
    if clear != "yes":
        letter = None

    # Guard 2: the letter must physically appear in the raw model output as a
    # standalone token (not inside a word). If the checker hallucinates a
    # letter not present in the text, force null and record the reason.
    letter_not_in_raw = False
    if letter is not None and model_raw is not None:
        if not re.search(rf"(?<![A-Za-z0-9]){letter}(?![A-Za-z0-9])", model_raw):
            letter_not_in_raw = True
            letter = None

    result = {
        "clear_selection":  clear,
        "reasoning":        reasoning,
        "confirmed_answer": letter,
    }
    if letter_not_in_raw:
        result["guard_override"] = "letter_not_in_raw"
    return result


# ── vLLM helpers ──────────────────────────────────────────────────────────────

def wait_for_vllm(base_url: str, timeout: int = 180) -> bool:
    # base_url ends with /v1
    url = base_url.rsplit("/v1", 1)[0] + "/health"
    print(f"Waiting for vLLM at {url} (timeout={timeout}s)...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(url, timeout=5).status_code == 200:
                print("vLLM is ready.")
                return True
        except Exception:
            pass
        time.sleep(5)
    return False


# ── Checker ────────────────────────────────────────────────────────────────────

class Checker:
    def __init__(self, base_url: str, concurrency: int, model_path: str):
        self.client = AsyncOpenAI(base_url=base_url, api_key="token-not-needed")
        self.sem = asyncio.Semaphore(concurrency)
        self.model_path = model_path

    async def check_one(self, raw: Optional[str], parsed: Optional[str],
                        question_info: Dict) -> Dict:
        # Fast paths — no LLM call needed
        if raw is None:
            return {"clear_selection": "no",
                    "reasoning": "raw output is null",
                    "confirmed_answer": None,
                    "skipped": "null_raw"}
        r = raw.strip()
        if len(r) == 1 and r in ANSWER_LETTERS:
            return {"clear_selection": "yes",
                    "reasoning": "raw output is a single letter",
                    "confirmed_answer": r,
                    "skipped": "single_letter"}

        text_for_check = raw if len(raw) <= MAX_RAW_CHARS else (
            raw[: MAX_RAW_CHARS // 2] + "\n...[TRUNCATED]...\n"
            + raw[-MAX_RAW_CHARS // 2 :]
        )
        user_msg = USER_TEMPLATE.format(
            question_text=question_info.get("question_text", "(question unavailable)"),
            options_block=question_info.get("options_block", "(options unavailable)"),
            parsed=parsed if parsed else "null",
            raw=text_for_check,
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ]
        async with self.sem:
            for attempt in range(4):
                try:
                    resp = await self.client.chat.completions.create(
                        model=self.model_path,
                        messages=messages,
                        **CHECK_PARAMS,
                    )
                    content = resp.choices[0].message.content or ""
                    out = parse_checker_json(content, model_raw=raw)
                    out["checker_input_tokens"]  = resp.usage.prompt_tokens
                    out["checker_output_tokens"] = resp.usage.completion_tokens
                    return out
                except Exception as e:
                    if attempt < 3:
                        await asyncio.sleep(5 * (2 ** attempt))
                        continue
                    return {"reasoning": "", "confirmed_answer": None,
                            "error": str(e)[:300]}

    async def check_record(self, record: Dict, question_info: Dict) -> Dict:
        """Walk a record, schedule checks for every raw output, attach results."""
        out = copy.deepcopy(record)

        jobs: List[Tuple[List, int, asyncio.Task]] = []
        # Keep references so we can assign results back in-place.
        # For greedy: a list of length 1. For stochastic: list of length N.

        for cond_name, cond_data in out.get("conditions", {}).items():
            # greedy
            greedy = cond_data.get("greedy") or {}
            g_raw     = greedy.get("raw_output")
            g_parsed  = greedy.get("parsed_answer")
            greedy_slot = [None]
            jobs.append((greedy_slot, 0,
                         asyncio.create_task(self.check_one(g_raw, g_parsed, question_info))))
            cond_data["greedy"]["_greedy_slot"] = greedy_slot  # temp pointer

            # stochastic
            stoch = cond_data.get("stochastic") or {}
            raws    = stoch.get("raw_outputs", []) or []
            parseds = stoch.get("parsed_answers", []) or []
            slots: List[Optional[Dict]] = [None] * len(raws)
            for i, raw_i in enumerate(raws):
                parsed_i = parseds[i] if i < len(parseds) else None
                jobs.append((slots, i,
                             asyncio.create_task(self.check_one(raw_i, parsed_i, question_info))))
            cond_data["stochastic"]["_stoch_slots"] = slots

        # Await all checks for this record in parallel
        await asyncio.gather(*(t for _, _, t in jobs))
        for container, idx, task in jobs:
            container[idx] = task.result()

        # Finalize: move slot contents into checked_answer / checked_answers
        for cond_name, cond_data in out.get("conditions", {}).items():
            g_slot = cond_data["greedy"].pop("_greedy_slot", [None])
            cond_data["greedy"]["checked_answer"] = g_slot[0]

            s_slots = cond_data["stochastic"].pop("_stoch_slots", [])
            cond_data["stochastic"]["checked_answers"] = s_slots

        return out


# ── Dataset processing ────────────────────────────────────────────────────────

def load_question_bank(path: Path) -> Dict[str, Dict]:
    """Load unified dataset and return {question_id: {question_text, options_block}}.

    `options_block` is a pre-formatted multi-line string like
        A. BI-RADS 4 - Suspicious abnormality requiring biopsy
        B. BI-RADS 2 - Rim Calcifications
        ...
    """
    bank: Dict[str, Dict] = {}
    if not path.exists():
        print(f"[warn] Unified dataset not found at {path} — checker will run without question/options context")
        return bank
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            qid = d.get("question_id")
            if not qid:
                continue
            opts = d.get("options") or {}
            # Preserve A,B,C,D,E order
            lines = [f"{L}. {opts[L]}" for L in ANSWER_LETTERS if L in opts]
            bank[qid] = {
                "question_text": str(d.get("question_text", "")).strip(),
                "options_block": "\n".join(lines),
            }
    return bank


def load_done_ids(path: Path) -> set:
    done = set()
    if not path.exists():
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                qid = d.get("question_id")
                if qid:
                    done.add(qid)
            except Exception:
                continue
    return done


def _preview(s: Optional[str], n: int = MISMATCH_PREVIEW_CHARS) -> str:
    if not s:
        return ""
    s = s.replace("\r", " ")
    if len(s) <= n:
        return s
    return s[: n // 2] + "\n...[TRUNCATED]...\n" + s[-n // 2 :]


def extract_mismatches(result: Dict, original: Dict) -> List[Dict]:
    """Return list of mismatch entries for one record.

    A 'mismatch' is any sample where the checker disagrees with the parser or
    could not identify an answer (confirmed_answer is None).
    """
    out: List[Dict] = []
    model_name = result.get("model") or original.get("model")
    qid        = result.get("question_id") or original.get("question_id")
    correct    = result.get("correct_answer")

    for cond_name, cond_data in result.get("conditions", {}).items():
        # greedy
        g      = cond_data.get("greedy", {}) or {}
        chk    = g.get("checked_answer") or {}
        parsed = g.get("parsed_answer")
        conf   = chk.get("confirmed_answer")
        if chk.get("skipped"):
            continue  # trivial case, no real check happened
        if conf is None or conf != parsed:
            out.append({
                "model":             model_name,
                "question_id":       qid,
                "condition":         cond_name,
                "sample":            "greedy",
                "correct_answer":    correct,
                "parsed_answer":     parsed,
                "confirmed_answer":  conf,
                "clear_selection":   chk.get("clear_selection"),
                "reason":            "no_answer" if conf is None else "disagreement",
                "checker_reasoning": chk.get("reasoning", ""),
                "checker_error":     chk.get("error"),
                "raw_output_preview": _preview(g.get("raw_output")),
            })

        # stochastic
        s         = cond_data.get("stochastic", {}) or {}
        raws      = s.get("raw_outputs", []) or []
        parseds   = s.get("parsed_answers", []) or []
        checks    = s.get("checked_answers", []) or []
        for i, chk_i in enumerate(checks):
            chk_i  = chk_i or {}
            if chk_i.get("skipped"):
                continue
            parsed_i = parseds[i] if i < len(parseds) else None
            conf_i   = chk_i.get("confirmed_answer")
            if conf_i is None or conf_i != parsed_i:
                out.append({
                    "model":             model_name,
                    "question_id":       qid,
                    "condition":         cond_name,
                    "sample":            f"stochastic[{i}]",
                    "correct_answer":    correct,
                    "parsed_answer":     parsed_i,
                    "confirmed_answer":  conf_i,
                    "clear_selection":   chk_i.get("clear_selection"),
                    "reason":            "no_answer" if conf_i is None else "disagreement",
                    "checker_reasoning": chk_i.get("reasoning", ""),
                    "checker_error":     chk_i.get("error"),
                    "raw_output_preview": _preview(raws[i] if i < len(raws) else None),
                })
    return out


async def process_model_dir(mdir: Path, checker: Checker,
                            out_root: Path, limit: Optional[int],
                            question_bank: Dict[str, Dict],
                            overwrite: bool = False) -> None:
    src = mdir / DATASET_FILENAME
    if not src.exists():
        print(f"[{mdir.name}] no {DATASET_FILENAME} found — skipping")
        return

    out_dir   = out_root / mdir.name
    out_path  = out_dir / OUTPUT_DATASET_FILENAME
    mm_path   = out_dir / MISMATCH_FILENAME
    out_dir.mkdir(parents=True, exist_ok=True)

    if overwrite:
        for p in (out_path, mm_path):
            if p.exists():
                p.unlink()
        done_ids: set = set()
    else:
        done_ids = load_done_ids(out_path)

    records: List[Dict] = []
    with open(src) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue

    remaining = [r for r in records if r.get("question_id") not in done_ids]
    if limit is not None:
        remaining = remaining[:limit]

    print(f"\n[{mdir.name}] total={len(records)}  done={len(done_ids)}  to_run={len(remaining)}")
    if not remaining:
        return

    out_mode = "a" if done_ids else "w"
    mm_mode  = "a" if (done_ids and mm_path.exists()) else "w"
    n_mismatch = 0
    with open(out_path, out_mode) as fout, open(mm_path, mm_mode) as fmm:
        lock = asyncio.Lock()

        missing_qids: List[str] = []

        async def _run(rec: Dict):
            nonlocal n_mismatch
            qid = rec.get("question_id")
            qinfo = question_bank.get(qid)
            if qinfo is None:
                missing_qids.append(qid)
                qinfo = {"question_text": "(question unavailable)",
                         "options_block": "(options unavailable)"}
            result = await checker.check_record(rec, qinfo)
            mismatches = extract_mismatches(result, rec)
            async with lock:
                fout.write(json.dumps(result) + "\n")
                fout.flush()
                for m in mismatches:
                    fmm.write(json.dumps(m) + "\n")
                if mismatches:
                    fmm.flush()
                    n_mismatch += len(mismatches)
            return result

        await tqdm_asyncio.gather(
            *[_run(r) for r in remaining],
            desc=f"{mdir.name}",
        )
    print(f"[{mdir.name}] new mismatches logged: {n_mismatch}  → {mm_path}")
    if missing_qids:
        uniq = sorted(set(missing_qids))
        print(f"[{mdir.name}] [warn] {len(uniq)} unique question_id(s) missing from question bank: "
              f"{uniq[:5]}{'...' if len(uniq) > 5 else ''}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=PHASE3_RES,
                    help="Source phase3 results directory")
    ap.add_argument("--out-dir",     type=Path, default=PHASE4_RES,
                    help="Destination phase4 results directory")
    ap.add_argument("--port",        type=int, default=6000)
    ap.add_argument("--base-url",    type=str, default=None,
                    help="Override base URL for the verifier vLLM endpoint; takes precedence over --port.")
    ap.add_argument("--checker-model-path", type=str, default=CHECKER_MODEL_PATH,
                    help="Model id as registered in vLLM (default container path)")
    ap.add_argument("--concurrency", type=int, default=128)
    ap.add_argument("--model",       type=str, default=None,
                    help="Process only this single model subdirectory")
    ap.add_argument("--models-filter", type=str, default=None,
                    help="Comma-separated list of model names to include")
    ap.add_argument("--models-file", type=Path, default=None,
                    help="Path to a newline-separated file of model names to include. "
                         "Lines starting with '#' and blank lines are ignored. "
                         "Order is preserved. Use different files to run disjoint subsets "
                         "in parallel on different jobs.")
    ap.add_argument("--exclude",     type=str, default="backup",
                    help="Comma-separated list of dirs to exclude")
    ap.add_argument("--limit",       type=int, default=None,
                    help="Limit number of records per model (debugging)")
    ap.add_argument("--skip-wait",   action="store_true")
    ap.add_argument("--dataset-file", type=Path, default=DATASET_UNIFIED,
                    help="Unified dataset JSONL with question_text+options keyed by question_id")
    ap.add_argument("--overwrite",   action="store_true",
                    help="Ignore existing per-model outputs and re-check from scratch "
                         "(use this when the prompt/logic changes).")
    args = ap.parse_args()

    base_url = args.base_url or f"http://localhost:{args.port}/v1"

    PHASE4_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_wait:
        if not wait_for_vllm(base_url):
            sys.exit("ERROR: vLLM not reachable")

    # Discover available model dirs
    exclude = {x.strip() for x in args.exclude.split(",") if x.strip()}
    available = {d.name: d for d in args.results_dir.iterdir()
                 if d.is_dir() and d.name not in exclude}

    ordered_names: List[str]
    if args.models_file:
        if not args.models_file.exists():
            sys.exit(f"Models file not found: {args.models_file}")
        ordered_names = []
        with open(args.models_file) as f:
            for line in f:
                # Strip inline comments (everything after '#') and whitespace
                name = line.split("#", 1)[0].strip()
                if not name:
                    continue
                ordered_names.append(name)
    elif args.model:
        ordered_names = [args.model]
    elif args.models_filter:
        ordered_names = [x.strip() for x in args.models_filter.split(",") if x.strip()]
    else:
        ordered_names = sorted(available.keys())

    # Validate + preserve order
    model_dirs: List[Path] = []
    missing: List[str] = []
    for name in ordered_names:
        if name in available:
            model_dirs.append(available[name])
        else:
            missing.append(name)
    if missing:
        print(f"[warn] {len(missing)} model(s) from selection not found in results dir: {missing}")
    if not model_dirs:
        sys.exit("No valid model directories selected.")

    question_bank = load_question_bank(args.dataset_file)

    print(f"Checker endpoint : {base_url}")
    print(f"Checker model id : {args.checker_model_path}")
    print(f"Concurrency      : {args.concurrency}")
    print(f"Models to process: {len(model_dirs)}")
    print(f"Question bank    : {len(question_bank)} entries from {args.dataset_file}")
    print(f"Overwrite mode   : {args.overwrite}")

    checker = Checker(base_url, args.concurrency, args.checker_model_path)

    async def run_all():
        for mdir in model_dirs:
            await process_model_dir(mdir, checker, args.out_dir, args.limit,
                                    question_bank, overwrite=args.overwrite)

    asyncio.run(run_all())

    print("\nDone.")
    print(f"Outputs under: {args.out_dir}")


if __name__ == "__main__":
    main()
