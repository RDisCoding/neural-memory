# Post-Mortem Fixes — Session 5 Catastrophic Regression

All 6 fixes from the diagnosis have been implemented and verified locally (10/10 isolation test passes).

## Changes Made

### Fix 1 — Gate Reward Updates on Success Only (Critical)
**File:** [session_manager.py](file:///d:/X/src/session_manager.py#L155-L163)

**Problem:** Failed episodes (reward=0) were calling `reward_update()`, writing noise that destroyed established correct associations. This is the root cause of the Session 5 collapse (77.8% → 50%).

**Change:** Only call `reward_update` when `reward > 0.7`. Also deferred `action_vec` encoding until after we know the reward, saving a full forward pass on every failed episode.

```diff
-        for mem in self.memory_modules.values():
-            if reward > 0.7:
-                mem.reward_update(state_vec, action_vec, reward)
-            elif reward < 0.1:
-                mem.reward_update(state_vec, action_vec, reward * 0.3)
-            else:
-                mem.reward_update(state_vec, action_vec, reward)
+        if reward > 0.7:
+            action_vec = self.get_hidden_state(code)
+            for mem in self.memory_modules.values():
+                mem.reward_update(state_vec, action_vec, reward)
```

---

### Fix 2 — Reduce Forget Gate Decay Cap (High)
**File:** [memory_module.py](file:///d:/X/src/memory_module.py#L82-L83)

**Problem:** `alpha = min(alpha, 0.9)` allowed up to 90% weight decay per step. Over many tokens, this killed layers 4 and 16 (weight norms stuck at 0.05–0.06 — essentially dead).

**Change:** Capped at 0.5 and moved forget gate bias from -5.0 to -7.0 (sigmoid(-7) ≈ 0.001).

---

### Fix 3 — Fix Trainable Parameter Count (High)
**File:** [model_surgery.py](file:///d:/X/src/model_surgery.py#L86-L99)

**Problem:** The old check `"memory" in name or "gate" in name` matched Gemma's internal parameter names through the `MemoryAugmentedDecoderLayer` wrapper, unfreezing 743M params (26.5%) instead of ~50M.

**Change:** Freeze ALL parameters first, then explicitly unfreeze only the injected submodules by direct reference.

```diff
-    for name, param in model.named_parameters():
-        param.requires_grad = ("memory" in name or "gate" in name)
+    for param in model.parameters():
+        param.requires_grad = False
+    for i in layer_indices:
+        augmented = model.model.layers[i]
+        for param in augmented.memory.parameters():
+            param.requires_grad = True
+        for param in augmented.gate.parameters():
+            param.requires_grad = True
```

---

### Fix 4 & 5 — State/Action Encoding ✅ Already Correct
The `session_manager.py` already uses `get_hidden_state()` which calls `output_hidden_states=True` and indexes into the correct layer. Both state and action encoding were already properly implemented from a prior session.

---

### Fix 6 — Fix Isolation Test Thresholds (Medium)
**File:** [memory_module.py](file:///d:/X/src/memory_module.py#L348-L366)

**Problem:** Old threshold `assert final_norm > 0.01` let a 97.6% collapse (12.4 → 0.295) pass. The "module still functional: NO" was a print statement, not an assertion.

**Change:** 
- `assert final_norm > 0.3 * initial_norm` (must retain >30% of weight norm)
- Post-stress retrieval is now a real assertion: `assert post_stress_err < 0.5`

---

### Additional Stabilization
- **`surprise_threshold`**: Raised from 0.1 → 0.5 (only write genuinely novel content, prevents weight erosion from redundant writes)
- **`lr_assoc`**: Lowered from 0.05 → 0.005
- **`lr_reward`**: Lowered from 0.05 → 0.01
- **Deterministic tests**: Added `torch.manual_seed(42)` to surprise_decay test for 100% reliability

## Local Verification

```
10/10 passed

  PASS associative_storage: retrieval_error=0.1058
  PASS surprise_decay: 2.2200 -> 1.2913
  PASS reward_learning: all 3 pairs within 0.15 tolerance
  PASS generalisation: base=0.97, similar=0.95, diff=0.10
  PASS weight_norm_stability: 12.326 -> 8.355 (retained >68% of initial norm)
         post-stress retrieval error: 0.1317 (functional: YES)
```

## Files to Upload to Kaggle

| File | Changes |
|------|---------|
| `src/memory_module.py` | Fixes 2, 6 + stabilization |
| `src/model_surgery.py` | Fix 3 (param freezing) |
| `src/session_manager.py` | Fix 1 (gated reward updates) |
| `src/eval.py` | Already has calibration write-disable |

## Expected Impact on Next Run

| Metric | Run 1 (broken) | Expected Run 2 |
|--------|----------------|-----------------|
| Session 5 regression | 77.8% → 50% | Should hold ≥70% |
| Trainable params | 743M (26.5%) | ~50M (~1.8%) |
| Weight norms (layers 4,16) | 0.05–0.06 (dead) | Should be >1.0 |
| Improvement on failed | 22.2% | Target >30% |
