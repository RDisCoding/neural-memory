import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell('''# Phase 3: MemRL Policy Network

**Purpose:** Train a tiny 3-layer MLP policy via GRPO to replace the fixed `if reward > 0.7` memory update rule, then evaluate it across 3 seeds.
**Runtime:** GPU (T4 16 GB), ~6 hours total.
'''))

cells.append(nbf.v4.new_code_cell('''# -- CELL 1: Install dependencies + create directory structure --
import subprocess, sys
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "accelerate", "datasets", "faiss-cpu"],
    check=True
)

import os, json, torch, random, shutil
import numpy as np
from pathlib import Path

sys.path.insert(0, "/kaggle/working/src")

for d in ["src", "problems", "memory_states", "results", "logs"]:
    Path(f"/kaggle/working/{d}").mkdir(parents=True, exist_ok=True)

print("Env ready. CUDA:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
'''))

# Embed both src/ and problems/ directly into the notebook
embedded_code = ""
for folder in ["src", "problems"]:
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(".py") or file.endswith(".json"):
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    content = f.read()
                    embedded_code += f"with open('/kaggle/working/{folder}/{file}', 'w', encoding='utf-8') as f:\n"
                    embedded_code += f"    f.write({repr(content)})\n"

cells.append(nbf.v4.new_code_cell(f'''# -- CELL 2: Inject source code and problems directly --
import os
os.makedirs("/kaggle/working/src", exist_ok=True)
os.makedirs("/kaggle/working/problems", exist_ok=True)
{embedded_code}
print("Source files and problems written directly to /kaggle/working!")
'''))

cells.append(nbf.v4.new_code_cell('''# -- CELL 3: Isolation tests (CPU - no GPU needed) --
from memory_module import run_all_isolation_tests
run_all_isolation_tests()

print("\\nTesting memory_policy...")
import memory_policy
print("memory_policy imported successfully.")
'''))

cells.append(nbf.v4.new_code_cell('''# -- CELL 4: Load Gemma 2B --
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
'''))

cells.append(nbf.v4.new_code_cell('''# -- CELL 5: Inject memory layers + perplexity gate check --
from model_surgery import inject_memory_layers, measure_perplexity

VALIDATION_TEXTS = [
    "def add(a, b):\\n    return a + b",
    "def reverse(s):\\n    return s[::-1]",
    "for i in range(10):\\n    print(i)",
    "x = [1, 2, 3]\\nx.sort()\\nprint(x)",
    "def is_prime(n):\\n    return n > 1 and all(n % i for i in range(2, n))",
]

ppl_before = measure_perplexity(model, tokenizer, VALIDATION_TEXTS)
print(f"Perplexity before injection: {ppl_before:.4f}")

MEMORY_LAYERS = [4, 8, 16, 24]
model, memory_modules = inject_memory_layers(model, MEMORY_LAYERS)

ppl_after = measure_perplexity(model, tokenizer, VALIDATION_TEXTS)
delta_pct = abs(ppl_after - ppl_before) / ppl_before * 100
print(f"Perplexity after injection:  {ppl_after:.4f}")
print(f"Delta: {delta_pct:.2f}%  (must be < 1% to proceed)")

assert delta_pct < 1.0, f"FAIL: perplexity degraded by {delta_pct:.2f}%"
print("\\nGate OK. Safe to proceed.")
'''))

cells.append(nbf.v4.new_code_cell('''# -- CELL 6: Load Calibrated HumanEval --
import json

PROBLEMS_PATH = "/kaggle/working/problems/caliberated_20.json"

with open(PROBLEMS_PATH) as f:
    calibrated = json.load(f)

print(f"\\nLoaded {len(calibrated)} problems from {PROBLEMS_PATH}")
'''))

cells.append(nbf.v4.new_code_cell('''# -- STAGE 1: Trajectory Collection --
# Run sessions with the threshold rule to collect training data
from session_manager import SessionManager

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

traj_dir = Path("./memory_states/trajectory")
if traj_dir.exists():
    shutil.rmtree(traj_dir)

# Disable Phase 1 for all memory modules
for mem in memory_modules.values():
    mem.phase1_enabled = False

mgr = SessionManager(model, tokenizer, memory_modules, user_id="trajectory", temperature=0.2)

TRAJECTORY_SESSIONS = 6
for session in range(1, TRAJECTORY_SESSIONS + 1):
    print(f"\\n{'='*50}\\nTRAJECTORY COLLECTION - SESSION {session}/{TRAJECTORY_SESSIONS}\\n{'='*50}")
    mgr.run_session(calibrated, session_num=session)
    mgr.log_diagnostics()

trajectory_data = mgr.trajectory
print(f"\\nCollected {len(trajectory_data)} trajectory points.")
'''))

cells.append(nbf.v4.new_code_cell('''# -- STAGE 2: GRPO Training --
from memory_policy import train_policy

print("Training MemoryPolicy via GRPO...")
policy = train_policy(trajectory_data, n_epochs=50, lr=3e-4, K_samples=8, entropy_coef=0.05, print_every=10)
print("Training complete.")
'''))

cells.append(nbf.v4.new_code_cell('''# -- STAGE 2c: Policy Diagnostics --
from memory_policy import analyze_policy

analyze_policy(policy, trajectory_data)
'''))

cells.append(nbf.v4.new_code_cell('''# -- STAGE 3a: 3-Seed Evaluation with Policy --
from eval import compute_improvement_rate

SEEDS = [42, 123, 456]
N_SESSIONS = 5
all_seed_results = {}

# Save initial state for resets
initial_mem_states = {}
for idx, mem in memory_modules.items():
    initial_mem_states[idx] = {
        'layers': {k: v.clone() for k, v in mem.layers.state_dict().items()},
        'W_sa': {k: v.clone() for k, v in mem.W_sa.state_dict().items()},
        'reward_head': {k: v.clone() for k, v in mem.reward_head.state_dict().items()},
        'momentum': {k: v.clone() for k, v in mem.momentum.items()},
    }

for seed in SEEDS:
    print(f"\\n{'#'*60}\\n  SEED {seed}\\n{'#'*60}")
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    for idx, mem in memory_modules.items():
        mem.layers.load_state_dict({k: v.clone() for k, v in initial_mem_states[idx]['layers'].items()})
        mem.W_sa.load_state_dict({k: v.clone() for k, v in initial_mem_states[idx]['W_sa'].items()})
        mem.reward_head.load_state_dict({k: v.clone() for k, v in initial_mem_states[idx]['reward_head'].items()})
        mem.momentum = {k: v.clone() for k, v in initial_mem_states[idx]['momentum'].items()}
        mem.phase1_enabled = False  # Ensure phase 1 is disabled during evaluation too

    user_id = f"memrl_seed_{seed}"
    mem_dir = Path(f"./memory_states/{user_id}")
    if mem_dir.exists():
        shutil.rmtree(mem_dir)

    # Pass the trained policy into SessionManager
    eval_mgr = SessionManager(model, tokenizer, memory_modules, user_id=user_id, temperature=0.2, policy=policy)
    
    seed_results = []
    for s in range(1, N_SESSIONS + 1):
        print(f"\\n{'='*50}\\nSEED {seed} - SESSION {s}/{N_SESSIONS}\\n{'='*50}")
        summary = eval_mgr.run_session(calibrated, session_num=s)
        summary["session"] = s
        summary["seed"] = seed
        seed_results.append(summary)
        eval_mgr.log_diagnostics()

    all_seed_results[seed] = seed_results
'''))

cells.append(nbf.v4.new_code_cell('''# -- STAGE 3b: Comparison Table --
print("\\n" + "="*70)
print("  MULTI-SEED AGGREGATE RESULTS (MEMRL POLICY)")
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

mean_impr = np.mean(improvements)
std_impr  = np.std(improvements)

print(f"\\n{'-'*40}")
print(f"  Mean improvement: {mean_impr:.1f}% - {std_impr:.1f}%")
print(f"  Individual:       {', '.join(f'{x:.1f}%' for x in improvements)}")
print(f"{'-'*40}")

print("\\n-----------------------------------------------")
print("- System           - Mean improvement (failed)-")
print("-----------------------------------------------")
print("- No memory        - ~5-10%                   -")
print("- Fixed rule (P2)  - 49.0% - 7.0%             -")
print(f"- MemRL policy (P3)- {mean_impr:.1f}% - {std_impr:.1f}%             -")
print("-----------------------------------------------")
'''))

cells.append(nbf.v4.new_code_cell('''# -- STAGE 3c: Zip Checkpoint --
import zipfile

ZIP_PATH = "/kaggle/working/phase3_checkpoint.zip"

with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
    for folder in ["memory_states", "results", "problems"]:
        root = f"/kaggle/working/{folder}"
        if not os.path.exists(root): continue
        for dirpath, _, files in os.walk(root):
            for file in files:
                fpath = os.path.join(dirpath, file)
                zf.write(fpath, os.path.relpath(fpath, "/kaggle/working"))

size_mb = os.path.getsize(ZIP_PATH) / 1e6
print(f"Zipped - {ZIP_PATH}  ({size_mb:.1f} MB)")
'''))

nb.cells = cells
with open('d:/X/notebooks/x-phase-3.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print('Notebook created at d:/X/notebooks/x-phase-3.ipynb')
