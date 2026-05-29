# ============================================================
# Kaggle Quickstart — Reward-Conditioned Neural Memory Layer
# ============================================================
# Paste each "## CELL N" block into a separate Kaggle notebook cell.
# Runtime: GPU (T4 16GB is sufficient for Steps 1–5).
# Run cells IN ORDER. Do not skip.
# ============================================================

# ── CELL 1: Install and imports ──────────────────────────────
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "accelerate", "datasets", "faiss-cpu"], check=True)

import os, json, torch
from pathlib import Path

# Add src to path
sys.path.insert(0, "/kaggle/working/src")

# Create directory structure
for d in ["src", "problems", "memory_states", "results", "logs"]:
    Path(f"/kaggle/working/{d}").mkdir(parents=True, exist_ok=True)

print("Env ready. CUDA:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")


# ── CELL 2: Copy source files from your dataset ──────────────
# Option A: You uploaded memory-agent/ as a Kaggle dataset
# Then source files are at /kaggle/input/memory-agent/src/*.py
# Copy them to /kaggle/working/src/

import shutil

SOURCE = "/kaggle/input/memory-agent/src"   # adjust if dataset name differs
DEST   = "/kaggle/working/src"

if os.path.exists(SOURCE):
    for f in os.listdir(SOURCE):
        shutil.copy(os.path.join(SOURCE, f), os.path.join(DEST, f))
    print("Source files copied from dataset.")
else:
    # Option B: Files are in the notebook's working dir already
    # (e.g. you added them via the notebook editor)
    print("Source dir not found at", SOURCE)
    print("Ensure memory_module.py, model_surgery.py, sandbox.py,")
    print("session_manager.py, eval.py are in /kaggle/working/src/")


# ── CELL 3: ISOLATION TESTS (CPU, no GPU needed) ─────────────
# This is Step 2. Must pass before loading the LLM.

from memory_module import run_all_isolation_tests
run_all_isolation_tests()
# Expected: 5 PASS lines. If any FAIL, fix before proceeding.


# ── CELL 4: Load base model ───────────────────────────────────
# Step 1 & 3 combined.

from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

import os
HF_TOKEN  = os.environ.get("HF_TOKEN", "YOUR_HF_TOKEN")   # your token
MODEL_ID  = "google/gemma-2-2b-it"

login(token=HF_TOKEN, add_to_git_credential=False)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto"
)
print("Model loaded. Device:", next(model.parameters()).device)

# Quick sanity check
inputs = tokenizer("Write a Python function to reverse a string:",
                   return_tensors="pt").to("cuda")
out = model.generate(**inputs, max_new_tokens=80, temperature=0.1)
print(tokenizer.decode(out[0], skip_special_tokens=True))


# ── CELL 5: Inject memory layers + perplexity check ──────────
# Step 3.

from model_surgery import inject_memory_layers, measure_perplexity

VALIDATION_TEXTS = [
    "def add(a, b):\n    return a + b",
    "def reverse(s):\n    return s[::-1]",
    "for i in range(10):\n    print(i)",
    "x = [1, 2, 3]\nx.sort()\nprint(x)",
    "def is_prime(n):\n    return n > 1 and all(n % i for i in range(2, n))",
    "import os\nprint(os.getcwd())",
    "d = {'a': 1, 'b': 2}\nprint(d.get('a', 0))",
    "result = [x**2 for x in range(5)]",
    "def fib(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)",
    "with open('test.txt', 'w') as f:\n    f.write('hello')",
]

ppl_before = measure_perplexity(model, tokenizer, VALIDATION_TEXTS)
print(f"Perplexity before injection: {ppl_before:.4f}")

MEMORY_LAYERS = [4, 8, 16, 24]
model, memory_modules = inject_memory_layers(model, MEMORY_LAYERS)

ppl_after = measure_perplexity(model, tokenizer, VALIDATION_TEXTS)
delta_pct = abs(ppl_after - ppl_before) / ppl_before * 100
print(f"Perplexity after injection:  {ppl_after:.4f}")
print(f"Delta: {delta_pct:.2f}%  (must be < 1% to proceed)")

assert delta_pct < 1.0, (
    f"FAIL: perplexity degraded by {delta_pct:.2f}%. "
    "Check gate.bias init (should be -5.0)."
)
print("Gate OK. Proceeding.")


# ── CELL 6: Sandbox smoke test ────────────────────────────────
# Step 4. Run from /kaggle/working/ so subprocess finds python.

from sandbox import execute_code_safely, execute_humaneval

# Generic test
good = "def reverse_string(s):\n    return s[::-1]"
tests = [{"input": "reverse_string('hello')", "expected": "'olleh'"}]
r, fb = execute_code_safely(good, tests)
assert r == 1.0
print(f"Sandbox generic test: reward={r}  ✓")

# HumanEval full parser test
fake = {
    "entry_point": "add",
    "test": (
        "def check(candidate):\n"
        "    assert candidate(1, 2) == 3\n"
        "    assert candidate(0, 0) == 0\n"
    )
}
r2, fb2 = execute_humaneval("def add(a, b):\n    return a + b", fake)
assert r2 == 1.0
print(f"Sandbox HumanEval test:  reward={r2}  ✓")


# ── CELL 7: Download and calibrate HumanEval problems ─────────
# Step 5 — calibration.

from eval import load_humaneval, calibrate_problem_difficulty

raw_problems = load_humaneval("/kaggle/working/problems/humaneval_raw.json")
print(f"Loaded {len(raw_problems)} raw problems.")

# Scan first 80 for calibration (takes ~20 min on T4)
# If you want faster: reduce to problems[:40] or load pre-calibrated set
calibrated = calibrate_problem_difficulty(
    model, tokenizer,
    problems=raw_problems[:80],
    n_attempts=3,
    target_range=(0.3, 0.6),
    save_path="/kaggle/working/problems/calibrated_20.json"
)
print(f"Calibrated set: {len(calibrated)} problems")


# ── CELL 8: Full session smoke test (3 problems) ──────────────

from session_manager import SessionManager

mgr = SessionManager(model, tokenizer, memory_modules, user_id="smoke_test")
summary = mgr.run_session(calibrated[:3])   # 3 problems only
print(json.dumps(summary, indent=2, default=str))


# ── CELL 9: Full 5-session evaluation ────────────────────────
# Step 6. This is the main experiment. ~2–4 hours on Kaggle P100.

from eval import run_evaluation

all_results = run_evaluation(
    model, tokenizer, memory_modules,
    n_sessions=5,
    user_id="eval_memory",
    output_dir="/kaggle/working/results/",
    problems_path="/kaggle/working/problems/calibrated_20.json"
)


# ── CELL 10: Save checkpoint (download before session expires) ─
import shutil, zipfile

# Zip the memory states + results
with zipfile.ZipFile("/kaggle/working/memory_checkpoint.zip", "w") as zf:
    for root, dirs, files in os.walk("/kaggle/working/memory_states"):
        for file in files:
            fpath = os.path.join(root, file)
            zf.write(fpath, os.path.relpath(fpath, "/kaggle/working"))
    for root, dirs, files in os.walk("/kaggle/working/results"):
        for file in files:
            fpath = os.path.join(root, file)
            zf.write(fpath, os.path.relpath(fpath, "/kaggle/working"))

print("Checkpoint zipped → /kaggle/working/memory_checkpoint.zip")
print("Download this file before the Kaggle session expires!")
print("(File → Download from the Output panel)")
