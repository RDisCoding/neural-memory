# %% [markdown]
# # Multi-Seed Validation — Reward-Conditioned Neural Memory Layer
#
# **Purpose:** Validate the 71.4% improvement-on-failed result across 3 seeds.
# **Runtime:** GPU (T4 16 GB), ~3 hours total (3 seeds × 5 sessions × 16 problems).
# **Success criteria:** All 3 seeds show >40% improvement on failed problems.
#
# | Seed | Expected outcome |
# |------|-----------------|
# | 42   | Baseline seed   |
# | 123  | Variance check  |
# | 456  | Variance check  |

# %% Cell 1: Setup
import subprocess, sys
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "accelerate", "datasets", "faiss-cpu"],
    check=True
)

import os, json, torch, random, numpy as np
from pathlib import Path

sys.path.insert(0, "/kaggle/working/src")

for d in ["src", "problems", "memory_states", "results", "logs"]:
    Path(f"/kaggle/working/{d}").mkdir(parents=True, exist_ok=True)

print("Env ready. CUDA:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# %% Cell 2: Copy source files
import shutil

SOURCE = "/kaggle/input/memory-agent/src"
DEST   = "/kaggle/working/src"

if os.path.exists(SOURCE):
    for f in os.listdir(SOURCE):
        shutil.copy(os.path.join(SOURCE, f), os.path.join(DEST, f))
    print("Source files copied:", os.listdir(DEST))
else:
    print(f"WARNING: {SOURCE} not found.")

# %% Cell 3: Isolation tests
from memory_module import run_all_isolation_tests
run_all_isolation_tests()

# %% Cell 4: Load model
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

import os
HF_TOKEN = os.environ.get("HF_TOKEN", "YOUR_HF_TOKEN")
MODEL_ID = "google/gemma-2-2b-it"

login(token=HF_TOKEN, add_to_git_credential=False)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto"
)
print("Model loaded. Device:", next(model.parameters()).device)

# %% Cell 5: Inject memory + perplexity check
from model_surgery import inject_memory_layers, measure_perplexity

VALIDATION_TEXTS = [
    "def add(a, b):\n    return a + b",
    "def reverse(s):\n    return s[::-1]",
    "for i in range(10):\n    print(i)",
    "x = [1, 2, 3]\nx.sort()\nprint(x)",
    "def is_prime(n):\n    return n > 1 and all(n % i for i in range(2, n))",
]

ppl_before = measure_perplexity(model, tokenizer, VALIDATION_TEXTS)
print(f"Perplexity before injection: {ppl_before:.4f}")

MEMORY_LAYERS = [4, 8, 16, 24]
model, memory_modules = inject_memory_layers(model, MEMORY_LAYERS)

ppl_after = measure_perplexity(model, tokenizer, VALIDATION_TEXTS)
delta_pct = abs(ppl_after - ppl_before) / ppl_before * 100
print(f"Perplexity after injection:  {ppl_after:.4f}")
print(f"Delta: {delta_pct:.2f}%  (must be < 1% to proceed)")

assert delta_pct < 1.0, f"FAIL: perplexity degraded by {delta_pct:.2f}%."
print("\nGate OK. Safe to proceed.")

# Capture pristine memory state IMMEDIATELY after injection,
# BEFORE calibration can write to it through forward passes.
initial_mem_states = {}
for idx, mem in memory_modules.items():
    initial_mem_states[idx] = {
        'layers': {k: v.clone() for k, v in mem.layers.state_dict().items()},
        'W_sa': {k: v.clone() for k, v in mem.W_sa.state_dict().items()},
        'momentum': {k: v.clone() for k, v in mem.momentum.items()},
    }

# %% Cell 6: Load calibrated problems (reuse from previous run or recalibrate)
from eval import load_humaneval, calibrate_problem_difficulty

PROBLEMS_PATH = "/kaggle/working/problems/calibrated_20.json"

if os.path.exists(PROBLEMS_PATH):
    with open(PROBLEMS_PATH) as f:
        calibrated = json.load(f)
    print(f"Loaded {len(calibrated)} calibrated problems from cache.")
else:
    raw_problems = load_humaneval("/kaggle/working/problems/humaneval_raw.json")
    print(f"Loaded {len(raw_problems)} raw HumanEval problems.")
    calibrated = calibrate_problem_difficulty(
        model, tokenizer,
        problems=raw_problems[:100],
        n_attempts=3,
        target_range=(0.3, 0.7),
        save_path=PROBLEMS_PATH
    )
    print(f"\nCalibrated set: {len(calibrated)} problems saved.")

print(f"\nUsing {len(calibrated)} problems for multi-seed evaluation.")

# %% Cell 7: Multi-seed evaluation loop
from session_manager import SessionManager
from eval import compute_improvement_rate
from memory_module import NeuralMemoryModule

SEEDS = [42, 123, 456]
N_SESSIONS = 5
all_seed_results = {}

# initial_mem_states already captured in Cell 5 (before calibration)

for seed in SEEDS:
    print(f"\n{'#'*60}")
    print(f"  SEED {seed}")
    print(f"{'#'*60}")

    # Seed ONLY problem-ordering randomness (random + numpy).
    # DO NOT seed torch RNG — model.generate(do_sample=True) needs
    # stochasticity to explore different code approaches across sessions.
    # Fixed torch seeds freeze generation output, so the same wrong code
    # repeats every session and reward_update never fires.
    random.seed(seed)
    np.random.seed(seed)

    # Reset memory modules to initial (untrained) state
    for idx, mem in memory_modules.items():
        mem.layers.load_state_dict(
            {k: v.clone() for k, v in initial_mem_states[idx]['layers'].items()}
        )
        mem.W_sa.load_state_dict(
            {k: v.clone() for k, v in initial_mem_states[idx]['W_sa'].items()}
        )
        mem.momentum = {
            k: v.clone() for k, v in initial_mem_states[idx]['momentum'].items()
        }

    # Clear any prior checkpoints for this seed
    user_id = f"seed_{seed}"
    mem_dir = Path(f"./memory_states/{user_id}")
    if mem_dir.exists():
        for f in mem_dir.glob("*.pt"):
            f.unlink()
        for f in mem_dir.glob("*.json"):
            f.unlink()

    seed_results = []
    for s in range(1, N_SESSIONS + 1):
        print(f"\n{'='*50}")
        print(f"SEED {seed} — SESSION {s}/{N_SESSIONS}")
        print(f"{'='*50}")

        mgr = SessionManager(model, tokenizer, memory_modules, user_id=user_id, temperature=0.2)
        summary = mgr.run_session(calibrated)
        summary["session"] = s
        summary["seed"] = seed
        seed_results.append(summary)
        mgr.log_diagnostics()

    all_seed_results[seed] = seed_results

    # Compute improvement for this seed
    impr, n_failed = compute_improvement_rate(
        seed_results[0]["results"],
        seed_results[-1]["results"]
    )
    print(f"\n  SEED {seed} SUMMARY: {impr*100:.1f}% improvement on {n_failed} failed problems")
    print(f"  Pass rate: {seed_results[0]['pass_rate']*100:.1f}% → {seed_results[-1]['pass_rate']*100:.1f}%")

# %% Cell 8: Aggregate results + statistical summary
print("\n" + "="*70)
print("  MULTI-SEED AGGREGATE RESULTS")
print("="*70)

improvements = []
pass_trajectories = {}

for seed, results in all_seed_results.items():
    impr, n_failed = compute_improvement_rate(
        results[0]["results"],
        results[-1]["results"]
    )
    improvements.append(impr * 100)

    trajectory = [r["pass_rate"] * 100 for r in results]
    pass_trajectories[seed] = trajectory

    print(f"\nSeed {seed}:")
    print(f"  Sessions: {' → '.join(f'{t:.1f}%' for t in trajectory)}")
    print(f"  Improvement on failed: {impr*100:.1f}% (n={n_failed})")

mean_impr = np.mean(improvements)
std_impr  = np.std(improvements)

print(f"\n{'─'*40}")
print(f"  Mean improvement: {mean_impr:.1f}% ± {std_impr:.1f}%")
print(f"  Individual:       {', '.join(f'{x:.1f}%' for x in improvements)}")
print(f"  Target:           >40% (minimum), >60% (strong)")
print(f"{'─'*40}")

if mean_impr > 40:
    print("\n  ✅ RESULT IS STATISTICALLY DEFENSIBLE")
    print("     Ready for baselines and paper results section.")
else:
    print("\n  ⚠️ RESULT IS BELOW TARGET")
    print("     Investigate variance before running baselines.")

# Per-session mean across seeds
print("\n  Per-session mean pass rate:")
for s in range(N_SESSIONS):
    rates = [pass_trajectories[seed][s] for seed in SEEDS]
    print(f"    S{s+1}: {np.mean(rates):.1f}% ± {np.std(rates):.1f}%")

# %% Cell 9: Save all results
results_dir = Path("/kaggle/working/results")
results_dir.mkdir(parents=True, exist_ok=True)

# Save per-seed results
for seed, results in all_seed_results.items():
    path = results_dir / f"eval_seed_{seed}_results.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved: {path.name}")

# Save aggregate summary
summary = {
    "seeds": SEEDS,
    "n_sessions": N_SESSIONS,
    "n_problems": len(calibrated),
    "improvements": {str(s): imp for s, imp in zip(SEEDS, improvements)},
    "mean_improvement": round(mean_impr, 2),
    "std_improvement": round(std_impr, 2),
    "pass_trajectories": {str(s): t for s, t in pass_trajectories.items()},
}
with open(results_dir / "multiseed_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("Saved: multiseed_summary.json")

# %% Cell 10: Zip + download
import zipfile

ZIP_PATH = "/kaggle/working/multiseed_checkpoint.zip"

with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
    for folder in ["memory_states", "results", "problems"]:
        root = f"/kaggle/working/{folder}"
        for dirpath, _, files in os.walk(root):
            for file in files:
                fpath = os.path.join(dirpath, file)
                zf.write(fpath, os.path.relpath(fpath, "/kaggle/working"))

size_mb = os.path.getsize(ZIP_PATH) / 1e6
print(f"Zipped → {ZIP_PATH}  ({size_mb:.1f} MB)")
print("Download via: Output panel → multiseed_checkpoint.zip → Download")
