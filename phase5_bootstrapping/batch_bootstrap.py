#!/usr/bin/env python3
import json
import argparse
import csv
from pathlib import Path
from collections import defaultdict
import numpy as np
ALLOWED_STRATEGIES = {"zero_shot", "zero_shot_basedon", "CoT_2steps", "radio"}  # case‑insensitive
#EXCLUDE_DIRS       = {"old"}       # add more names here if needed
#EXCLUDE_DIRS = {"gpt-3.5-turbo"}
EXCLUDE_DIRS = {"old"}
def generate_bootstrap_indices(n, B=1000, seed=42):
    """
    Generate B bootstrap samples (with replacement) of size n,
    save to 'bootstrap_indices.json', and return the list of samples.
    """
    rng = np.random.default_rng(seed)
    samples = [rng.choice(n, size=n, replace=True).tolist() for _ in range(B)]
    with open("bootstrap_indices.json", "w") as f:
        json.dump(samples, f, indent=2)
    print(f"[+] Generated {B} bootstrap samples for n={n}, saved to bootstrap_indices.json")
    return samples

def load_bootstrap_indices(path="bootstrap_indices.json", expected_B=None):
    """
    Load existing bootstrap samples from path. If expected_B is provided
    and mismatches, return None to signal regeneration is needed.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            samples = json.load(f)
    except FileNotFoundError:
        # first run: no file yet
        return None
    if expected_B is not None and len(samples) != expected_B:
        print(f"[!] bootstrap_indices.json has {len(samples)} samples but expected {expected_B}; regenerating.")
        return None
    return samples
def compute_stats(correct_arr, samples):
    """
    Given a numpy array of 0/1 correctness flags and a list of index-samples,
    return (mean%, std%, ci_lower%, ci_upper%, bootstrap_means_array).
    """
    means = [correct_arr[s].mean() for s in samples]
    arr = np.array(means)
    mean = arr.mean() * 100
    std  = arr.std(ddof=1) * 100
    lo, hi = np.percentile(arr, [2.5, 97.5]) * 100
    return mean, std, lo, hi, arr

def paired_p(dist1, dist2):
    """
    Two-sided p-value for paired bootstrap distributions.
    """
    diffs = dist1 - dist2
    p_lower = np.mean(diffs <= 0)
    p_upper = np.mean(diffs >= 0)
    return 2 * min(p_lower, p_upper)

# def find_result_files(bases, pattern):
#     """
#     Recursively search each base directory for files matching pattern
#     under Strategy/Model subdirectories.
#     """
#     files = []
#     for base in bases:
#         if not base.exists():
#             continue
#         for strategy in base.iterdir():
#             if not strategy.is_dir():
#                 continue
#             for model in strategy.iterdir():
#                 if not model.is_dir():
#                     continue
#                 for f in model.glob(pattern):
#                     files.append(f)
#     return sorted(files)

def find_result_files(
    bases,
    pattern,
    allowed_strategies=ALLOWED_STRATEGIES,
    exclude_dirs=EXCLUDE_DIRS,
):
    """
    Walk <base>/<strategy>/<model>/files…
       • keep only <strategy> in `allowed_strategies`
       • ignore any <model> directory whose *own* name is in `exclude_dirs`
    """
    allowed = {s.lower() for s in (allowed_strategies or [])}
    #exclude = set(exclude_dirs or [])
    exclude = {d.lower() for d in (exclude_dirs or [])}

    files = []
    for base in bases:
        if not base.exists():
            continue

        # --- strategy level -------------------------------------------------
        for strategy_dir in base.iterdir():
            if not strategy_dir.is_dir():
                continue
            if allowed and strategy_dir.name.lower() not in allowed:
                # skip strategies we don't care about
                continue

            # --- model level -------------------------------------------------
            for model_dir in strategy_dir.iterdir():
                # print(f"Checking {model_dir}")
                if not model_dir.is_dir():
                    continue
                if  model_dir.name.lower() in exclude:
                    print(model_dir.name.lower())
                    # e.g. "old"
                    #files.extend(model_dir.glob(pattern))
                    continue
                # if exclude and model_dir.name.lower() not in exclude:
                #     # skip models we want to exclude
                #     continue
                # print(f"Searching in {model_dir}")
                files.extend(model_dir.glob(pattern))

    return sorted(files)


def main():
    parser = argparse.ArgumentParser(
        description="Batch bootstrap analysis for model accuracies."
    )
    parser.add_argument(
        "--pattern", "-p",
        default="results_*.json",
        help="glob pattern for per-question JSONs (default: 'results_*.json')"
    )
    parser.add_argument(
        "--bootstrap", "-B",
        type=int,
        default=1000,
        help="number of bootstrap redraws (default: 1000)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for reproducibility (default: 42)"
    )
    args = parser.parse_args()

    bases = [Path("noRAG"), Path("deepresearch"), Path("radioRAG")]
    result_files = find_result_files(bases, args.pattern)
    if not result_files:
        print("No result files found! Check your --pattern and ensure you're in the correct directory.")
        return

    # Determine number of questions from the first results file
    with open(result_files[0], "r", encoding="utf-8") as f:
        first_data = json.load(f)
    n_questions = len(first_data)

    # Load or generate bootstrap indices
    samples = load_bootstrap_indices("bootstrap_indices.json", expected_B=args.bootstrap)
    if samples is None:
        samples = generate_bootstrap_indices(n_questions, args.bootstrap, args.seed)
    elif samples == []:
        samples = generate_bootstrap_indices(n_questions, args.bootstrap, args.seed)

    stats = {}
    dists = {}

        # Process each results file
    for fn in result_files:
        with open(fn, "r", encoding="utf-8") as f:
            data = json.load(f)
        correct = np.array([int(r["correct"]) for r in data])

        # Compute bootstrap stats
        mean, std, lo, hi, dist = compute_stats(correct, samples)
        mean = int(round(mean))
        std = int(round(std))
        lo, hi = map(int, map(round, (lo, hi)))
        total_correct = correct.sum()

        # Parse dataset & model from filename: results_<dataset>_<model>.json
        stem = fn.stem   # e.g. "results_test_set_MedQA_deepseek70br1"
        parts = stem.split("_")
        if parts[0] == "results":
            parts = parts[1:]
        model   = parts[-1]
        dataset = "_".join(parts[:-1])

        # 1) Save the full bootstrap distribution
        out_dist = fn.with_name(f"bootstrap_means_{dataset}_{model}.json")
        with open(out_dist, "w") as outf:
            json.dump(dist.tolist(), outf, indent=2, ensure_ascii=False)
        print(f"[+] Saved bootstrap means → {out_dist}")
        # save 1000 values to CSV, in percent
        out_dist_csv = fn.with_name(f"bootstrap_means_{dataset}_{model}.csv")
        np.savetxt(out_dist_csv, dist*100, fmt="%.6f", delimiter=",")
        # 2) Save a JSON summary for this model
        summary_data = {
            "mean":    round(mean, 4),
            "std":     round(std,  4),
            "ci_lower":round(lo,   4),
            "ci_upper":round(hi,   4),
            "correct": int(total_correct),
            "n":       n_questions,
            "accuracy": f"{model}: {mean}% ± {std} [95% CI: {lo}, {hi}] ({total_correct}/{n_questions})"
        }
        out_sum_json = fn.with_name(f"summary_{dataset}_{model}.json")
        with open(out_sum_json, "w") as fsum:
            json.dump(summary_data, fsum, indent=2, ensure_ascii=False)
        print(f"[+] Saved per-model summary JSON → {out_sum_json}")

        # 3) (Optional) a one-line text summary
        out_sum_txt = fn.with_name(f"summary_{dataset}_{model}.txt")
        with open(out_sum_txt, "w") as ftxt:
            ftxt.write(
                    f"{mean}% ± {std} [95% CI: {lo}, {hi}] "
                    f"({total_correct}/{n_questions})"
                )
        print(f"[+] Saved per-model summary TXT → {out_sum_txt}")

        # Store for the global summary
        stats[fn] = {
            "dataset":  dataset,
            "model":    model,
            "mean":     mean,
            "std":      std,
            "ci_lower": lo,
            "ci_upper": hi,
            "correct":  int(total_correct),
            "n":        n_questions
        

        }
        dists[fn] = dist


    # Print summary
    # print("\nModel accuracies:")
    # for fn, st in stats.items():
    #     #name = str(fn.relative_to(Path.cwd())
    #     name = str(fn)
    #     print(
    #         f" {name}: "
    #         f"{st['mean']}% ± {st['std']} "
    #         f"[95% CI: {st['ci_lower']}, {st['ci_upper']}] "
    #         f"({st['correct']}/{st['n']})"
    #     )

    print("\nModel accuracies (grouped by model):")
    grouped_stats = defaultdict(list)
    for fn, st in stats.items():
        grouped_stats[st["model"]].append((Path(fn), st))

    # Helper to show "base/strategy" from the path, e.g. noRAG/zero_shot
    def path_label(p: Path) -> str:
        parts = p.parts
        return "/".join(parts[:2]) if len(parts) >= 2 else parts[0]
    
    # Iterate models alphabetically; change the sort key if you prefer
    print("Specific format:")
    for model in sorted(grouped_stats):
        print(f"\n{model}:")
        # Sort by folder label for stable ordering
        for fn, st in sorted(grouped_stats[model], key=lambda t: path_label(t[0])):
            label = path_label(fn)
            print(
                f"  {label}: "
                f"{st['mean']} ± {st['std']} "
                f"[{st['ci_lower']}, {st['ci_upper']}] "
                f"({st['correct']}/{st['n']})"
            )

    # Iterate models alphabetically; change the sort key if you prefer
    print("\nDetailed format:")
    for model in sorted(grouped_stats):
        print(f"\n{model}:")
        # Sort by folder label for stable ordering
        for fn, st in sorted(grouped_stats[model], key=lambda t: path_label(t[0])):
            label = path_label(fn)
            print(
                f"  {label}: "
                f"{st['mean']}% ± {st['std']} "
                f"[95% CI: {st['ci_lower']}, {st['ci_upper']}] "
                f"({st['correct']}/{st['n']})"
            )

    # Save summary to JSON and CSV
    summary_list = [
        {
            "file": str(fn),
            "dataset": st["dataset"],
            "model": st["model"],
            "mean": round(st["mean"], 4),
            "std": round(st["std"], 4),
            "ci_lower": round(st["ci_lower"], 4),
            "ci_upper": round(st["ci_upper"], 4),
            "correct": st["correct"],
            "n": st["n"],
            "accuracy": (
            f"{st['mean']}% ± {st['std']} "
            f"[95% CI: {st['ci_lower']}, {st['ci_upper']}] "
            f"({st['correct']}/{st['n']})"
        )
        }
        for fn, st in stats.items()
    ]
    with open("bootstrap_summary.json", "w") as f:
        json.dump(summary_list, f, indent=2, ensure_ascii=False)
    print("[+] Saved summary to bootstrap_summary.json")

    with open("bootstrap_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_list[0].keys())
        writer.writeheader()
        writer.writerows(summary_list)
    print("[+] Saved summary to bootstrap_summary.csv")

if __name__ == "__main__":
    main()
