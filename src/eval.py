"""
eval.py
Problem calibration and multi-session evaluation loop.

Usage (Kaggle / local):
    python src/eval.py

Usage (HPC via SLURM — see hpc/run_evaluation.slurm):
    python src/eval.py --model_id google/gemma-2-2b-it \
                       --n_sessions 5 \
                       --user_id hpc_run_1 \
                       --output_dir /path/to/results
"""

import argparse
import json
import os
import sys
import torch
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# PROBLEM LOADING
# ─────────────────────────────────────────────────────────────

def load_humaneval(path: str = None) -> list:
    """
    Load HumanEval problems. Tries local file first, then HuggingFace datasets.
    Returns list of dicts with keys: task_id, prompt, test, entry_point.
    """
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)

    print("Local file not found — downloading from HuggingFace datasets...")
    from datasets import load_dataset
    ds = load_dataset("openai_humaneval", split="test")
    problems = [dict(row) for row in ds]

    # Save for future runs
    save_path = path or "problems/humaneval_raw.json"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(problems, f, indent=2)
    print(f"Saved {len(problems)} problems → {save_path}")
    return problems


def load_calibrated(path: str = "problems/calibrated_20.json") -> list:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Calibrated problem set not found at {path}. "
            "Run calibrate_problem_difficulty() first."
        )
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────
# CALIBRATION
# ─────────────────────────────────────────────────────────────

def calibrate_problem_difficulty(model, tokenizer,
                                  problems: list,
                                  n_attempts: int = 3,
                                  target_range: tuple = (0.3, 0.7),
                                  save_path: str = "problems/calibrated_20.json"
                                  ) -> list:
    """
    Run base model (no memory) on each problem n_attempts times.
    Keep only problems where pass rate ∈ target_range.

    These are the 'sweet spot' problems where memory has something to learn:
    - Too easy (>0.6): base model already solves them → no room for improvement
    - Too hard (<0.3): reward almost always 0 → no learning signal
    """
    from sandbox import execute_humaneval

    calibrated = []
    print(f"Calibrating {len(problems)} problems "
          f"(target pass rate: {target_range[0]}–{target_range[1]})...\n")

    # Temporarily disable memory writing during calibration
    from model_surgery import MemoryAugmentedDecoderLayer
    for name, module in model.named_modules():
        if isinstance(module, MemoryAugmentedDecoderLayer):
            module.memory.write_enabled = False

    for p in problems:
        passes = 0
        for attempt in range(n_attempts):
            # Import here to avoid circular — session_manager imports sandbox
            from session_manager import SessionManager
            # Use a dummy manager just for code generation
            mgr = SessionManager(model, tokenizer, {}, user_id="calib")
            code = mgr.generate_code(p["prompt"])
            reward, _ = execute_humaneval(code, p)
            if reward >= 0.9:
                passes += 1

        pass_rate = passes / n_attempts

        if target_range[0] <= pass_rate <= target_range[1]:
            p["baseline_pass_rate"] = pass_rate
            calibrated.append(p)
            print(f"  KEEP {p['task_id']}: pass_rate={pass_rate:.2f}")
        elif pass_rate < target_range[0]:
            print(f"  DROP {p['task_id']}: too hard  (pass_rate={pass_rate:.2f})")
        else:
            print(f"  DROP {p['task_id']}: too easy  (pass_rate={pass_rate:.2f})")

    print(f"\nCalibrated set: {len(calibrated)} / {len(problems)} problems")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(calibrated[:20], f, indent=2)
        print(f"Saved calibrated set → {save_path}")

    # Re-enable memory writing for the actual evaluation phase
    for name, module in model.named_modules():
        if isinstance(module, MemoryAugmentedDecoderLayer):
            module.memory.write_enabled = True

    return calibrated[:20]


# ─────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────

def compute_improvement_rate(session_1_results: list,
                              session_n_results: list) -> tuple:
    """
    Of all problems FAILED in session 1, what fraction PASS in session N?

    This is the primary metric.
    Baseline (no memory): ~5%
    Target (memory model): >30% by session 5
    """
    failed_in_s1 = {r["problem_id"] for r in session_1_results
                    if not r["passed"]}
    now_passing  = {r["problem_id"] for r in session_n_results
                    if r["problem_id"] in failed_in_s1 and r["passed"]}

    if not failed_in_s1:
        return 0.0, 0

    rate = len(now_passing) / len(failed_in_s1)
    return rate, len(failed_in_s1)


def print_session_table(all_results: list):
    s1_rate = all_results[0]["pass_rate"]
    print(f"\n{'Session':<10} {'Pass%':<10} {'vs S1':<10}")
    print("─" * 35)
    for r in all_results:
        delta = r["pass_rate"] - s1_rate
        sign  = "+" if delta >= 0 else ""
        print(f"{r['session']:<10} "
              f"{r['pass_rate']*100:<10.1f}% "
              f"{sign}{delta*100:.1f}%")

    if len(all_results) >= 2:
        impr, n_failed = compute_improvement_rate(
            all_results[0]["results"],
            all_results[-1]["results"]
        )
        print(f"\nImprovement on previously-failed problems: {impr*100:.1f}%"
              f"  (n={n_failed})")
        print("Baseline/no-memory target: ~5%  |  Memory model target: >30%")


# ─────────────────────────────────────────────────────────────
# EVALUATION LOOP
# ─────────────────────────────────────────────────────────────

def run_evaluation(model, tokenizer, memory_modules: dict,
                   n_sessions: int = 5,
                   user_id: str = "eval",
                   output_dir: str = ".",
                   problems_path: str = "problems/calibrated_20.json",
                   temperature: float = 0.2
                   ) -> list:
    """
    Run N sessions over the calibrated problem set.
    Same problems every session — measures cross-session improvement.
    Saves a summary JSON to output_dir.
    """
    from session_manager import SessionManager

    problems = load_calibrated(problems_path)
    all_results = []

    mgr = SessionManager(model, tokenizer, memory_modules, user_id=user_id, temperature=temperature)

    for s in range(1, n_sessions + 1):
        print(f"\n{'='*50}")
        print(f"SESSION {s}/{n_sessions}")
        print(f"{'='*50}")

        summary = mgr.run_session(problems, session_num=s)
        summary["session"] = s
        all_results.append(summary)

        # Diagnostic snapshot after each session
        mgr.log_diagnostics()

    print_session_table(all_results)

    # Save full results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results_path = os.path.join(output_dir, f"eval_{user_id}_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved → {results_path}")

    return all_results


# ─────────────────────────────────────────────────────────────
# ENTRYPOINT (HPC / CLI)
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate Reward-Conditioned Neural Memory Layer"
    )
    p.add_argument("--model_id",      default="google/gemma-2-2b-it")
    p.add_argument("--n_sessions",    default=5,  type=int)
    p.add_argument("--user_id",       default="eval_run_1")
    p.add_argument("--output_dir",    default="./results")
    p.add_argument("--problems_path", default="problems/calibrated_20.json")
    p.add_argument("--memory_layers", default="4,8,16,24",
                   help="Comma-separated layer indices for memory injection")
    p.add_argument("--calibrate",     action="store_true",
                   help="Run calibration scan before evaluation")
    p.add_argument("--hf_token",      default=None,
                   help="HuggingFace token (or set HF_TOKEN env var)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # ── HuggingFace auth ──
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token, add_to_git_credential=False)
        print("HuggingFace: authenticated")

    # ── Load model ──
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print(f"Model loaded on: {next(model.parameters()).device}")

    # ── Inject memory ──
    sys.path.insert(0, os.path.dirname(__file__))
    from model_surgery import inject_memory_layers, measure_perplexity

    layer_indices = [int(x) for x in args.memory_layers.split(",")]
    model, memory_modules = inject_memory_layers(model, layer_indices)

    # ── Calibration (optional) ──
    if args.calibrate:
        raw_problems = load_humaneval("problems/humaneval_raw.json")
        # Scan first 80 problems for the calibrated set
        calibrate_problem_difficulty(
            model, tokenizer,
            problems=raw_problems[:80],
            save_path=args.problems_path
        )

    # ── Run evaluation ──
    run_evaluation(
        model, tokenizer, memory_modules,
        n_sessions=args.n_sessions,
        user_id=args.user_id,
        output_dir=args.output_dir,
        problems_path=args.problems_path
    )
