# Multi-Seed Run Post-Mortem

## Result Summary

| Metric | Value | Target |
|--------|-------|--------|
| Mean improvement on failed | **21.4% ± 5.1%** | >40% (minimum) |
| Seed 42 trajectory | 50→44→50→63→50% | Monotone up |
| Seed 123 trajectory | 50→63→50→50→56% | Monotone up |
| Seed 456 trajectory | 56→69→50→50→44% | Monotone up |

**Verdict:** All seeds oscillate rather than improve. Memory is not accumulating useful signal.

---

## Root Cause Analysis (from [todo.md](file:///d:/X/todo.md))

### Root Cause 1 — Calibration Writes Corrupting Memory ❌ ALREADY FIXED

The `todo.md` claims `write_enabled` flag is set but never checked in `write()`. 

**Actually:** [memory_module.py:62-63](file:///d:/X/src/memory_module.py#L62-L63) already has the guard:
```python
def write(self, x: torch.Tensor) -> float:
    if not getattr(self, "write_enabled", True):
        return 0.0
```

**However**, the Layer 24 norm evidence is still damning. Session 1 norms across all seeds:

| Seed | Layer 4 | Layer 8 | Layer 16 | Layer 24 |
|------|---------|---------|----------|----------|
| 42   | 0.34    | 1.97    | 2.00     | **10.32** |
| 123  | 0.38    | 2.00    | 2.00     | **10.45** |
| 456  | 2.06    | 1.91    | 2.00     | **9.61**  |

Layer 24 is 5× higher than others at session start. This means either:
1. The `write_enabled` guard was added *after* this run was executed (the Kaggle notebook copies files from `/kaggle/input/memory-agent/src` — could be stale), OR
2. Calibration IS writing through another path (e.g., the forward pass triggers writes through `MemoryAugmentedDecoderLayer.forward()` even when `write_enabled=False` on the module)

> [!IMPORTANT]
> **Verify**: Does `MemoryAugmentedDecoderLayer.forward()` call `self.memory.write()` directly, or does it have its own write path that bypasses the flag? Check [model_surgery.py](file:///d:/X/src/model_surgery.py).

### Root Cause 2 — Fixed Seeds Freeze Generation ✅ CONFIRMED, UNFIXED

[kaggle_multiseed.py:139-142](file:///d:/X/notebooks/kaggle_multiseed.py#L139-L142):
```python
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)        # ← THIS freezes model.generate() output
torch.cuda.manual_seed_all(seed)
```

This is the **primary cause**. With `torch.manual_seed(seed)` set once before Session 1 and never reset, `model.generate()` with `do_sample=True, temperature=0.05` produces near-deterministic output. The same wrong code for HE/5 every session → reward never fires → memory never updates → frozen.

**Evidence from logs:**

```
HE/5:   Seed42 F→F→F→F→F    Seed123 F→F→F→F→F    Seed456 F→F→P→F→F
HE/49:  0.14 every session   0.14 every session    0.14 every session
HE/73:  0.25 every session   0.25 every session    0.25 every session
HE/29:  0.00 every session   0.00 every session    0.00 every session
```

Problems that fail once fail *identically* forever — same partial-credit scores, same everything.

### Root Cause 3 — Memory Not Reset Between Seeds ❌ ALREADY FIXED

[kaggle_multiseed.py:145-154](file:///d:/X/notebooks/kaggle_multiseed.py#L145-L154) already resets memory properly:
```python
for idx, mem in memory_modules.items():
    mem.layers.load_state_dict(...)
    mem.W_sa.load_state_dict(...)
    mem.momentum = {...}
```

But the reset restores from `initial_mem_states` which was captured *after* model loading + memory injection. If calibration ran before this capture and corrupted the weights, the "initial" state is already corrupted.

### Root Cause 4 — Layer 24 Norm Unclamped ❌ ALREADY FIXED

[memory_module.py:112-121](file:///d:/X/src/memory_module.py#L112-L121) already has `_clamp_weight_norms()` called after both `write()` and `reward_update()`, with `max_weight_norm=5.0`. But Layer 24 sits at 10+ in the logs, meaning either:
- The clamp wasn't in the version that ran on Kaggle, OR
- `max_weight_norm` was different (or not applied to all params)

---

## What Actually Needs Fixing For the Next Run

### Fix A — Don't seed torch RNG globally (CRITICAL)

The seed should control problem ordering only, not model generation. This is the single biggest fix.

```diff
# kaggle_multiseed.py — Cell 7
  for seed in SEEDS:
      # Set all random seeds
      random.seed(seed)
      np.random.seed(seed)
-     torch.manual_seed(seed)
-     torch.cuda.manual_seed_all(seed)
+     # DO NOT set torch.manual_seed — model.generate() needs stochasticity
+     # Seed only controls problem ordering via random/numpy
```

### Fix B — Capture initial memory state BEFORE calibration

Move the initial state capture to right after `inject_memory_layers()`, before calibration can touch it:

```diff
# kaggle_multiseed.py — Cell 5 (after injection)
  model, memory_modules = inject_memory_layers(model, MEMORY_LAYERS)
+ 
+ # Capture pristine state BEFORE calibration can corrupt it
+ initial_mem_states = {}
+ for idx, mem in memory_modules.items():
+     initial_mem_states[idx] = {
+         'layers': {k: v.clone() for k, v in mem.layers.state_dict().items()},
+         'W_sa': {k: v.clone() for k, v in mem.W_sa.state_dict().items()},
+         'momentum': {k: v.clone() for k, v in mem.momentum.items()},
+     }
```

And remove the duplicate capture in Cell 7.

### Fix C — Verify source files are current on Kaggle

The Kaggle notebook copies from `/kaggle/input/memory-agent/src`. If you uploaded a dataset with old `memory_module.py` (without the `write_enabled` guard or `_clamp_weight_norms`), the fixes you have locally won't run on Kaggle.

> [!WARNING]
> **Before the next run:** Re-upload the dataset with the current `src/` directory, or verify the uploaded version matches what's in [d:\X\src](file:///d:/X/src).

### Fix D — Raise temperature to restore exploration

`temperature=0.05` with `do_sample=True` is effectively greedy. Even without fixed seeds, the model will rarely explore alternative approaches.

```diff
- mgr = SessionManager(model, tokenizer, memory_modules, user_id=user_id, temperature=0.05)
+ mgr = SessionManager(model, tokenizer, memory_modules, user_id=user_id, temperature=0.2)
```

---

## Priority Order

| Priority | Fix | Impact | Risk |
|----------|-----|--------|------|
| **P0** | A: Remove torch.manual_seed | Restores generation stochasticity | None |
| **P0** | C: Re-upload source files | Ensures all patches are active | None |
| **P1** | B: Capture state before calibration | Prevents calibration corruption | Low |
| **P2** | D: Raise temperature to 0.2 | More exploration diversity | Slight noise increase |
