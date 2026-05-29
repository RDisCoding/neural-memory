Stop. Look at one specific thing before anything else.

```
Layer 4: weight_norm=11.2209 → 11.2227  (across 6 sessions)
Layer 8: weight_norm=11.2239 → 11.2239  (unchanged)
Layer 16: weight_norm=11.2207 → 11.2219
Layer 24: weight_norm=11.2322 → 11.2329
```

The weight norms barely moved across the entire run. All four layers stuck at ~11.22 from start to finish. Previously norms ranged between 2 and 12 dynamically. Now they're frozen. **The memory layer is not updating at all.** That's not a policy problem — that's a writing problem.

---

## Two Separate Things Failed

### Failure 1 — The Policy Learned Exactly Backwards

Look at this:

```
low reward (0-0.3):   WRITE=59%,  NOOP=41%
mid reward (0.3-0.7): WRITE=0%,   NOOP=100%
high reward (0.7-1.0): WRITE=0%,  NOOP=100%
```

The policy learned to WRITE on failures and NOOP on successes. This is the exact opposite of what you want. And it happened because the reward function was wrong.

The reward signal you implemented was:
```python
return next_reward - task_reward  # improvement score
```

Think about what this creates mathematically. When a problem passes in session T (task_reward=1.0) and also passes in session T+1 (next_reward=1.0), the policy reward = 1.0 - 1.0 = **0**. No gradient. Writing after success gives zero signal.

When a problem fails in session T (task_reward=0.0) and happens to pass in session T+1 by sampling luck (next_reward=1.0), the policy reward = 1.0 - 0.0 = **1.0**. Maximum signal. Writing after failure gets massively rewarded.

So the policy correctly optimised the reward function you gave it. The function was wrong. It rewarded "improvement" — which is highest after failures that randomly recover — rather than rewarding "reinforce what worked."

### Failure 2 — Writing After Failures Produces Zero TD Update

With Phase 1 disabled, the only writes are Phase 2 TD updates. The policy chose to WRITE after failures. But the TD update for a failure is:

```
td_error = actual_reward - predicted_reward
         = 0.0 - ~0.0
         = ~0
```

When reward is 0 and the memory already predicts ~0 for unknown situations, the TD error is near-zero. The update does nothing. The weights stay at 11.22.

Both failures compound: wrong incentive → writes on failures → near-zero TD error → frozen weights.

---

## The Fix — Both Need Changing

**Fix the reward signal — use task reward directly, not improvement:**

```python
def compute_policy_reward(action, task_reward, next_reward):
    if action == 0:  # WRITE
        # Reward writing when the CURRENT task succeeded
        # Logic: remember approaches that worked, not approaches that failed
        if task_reward >= 0.7:
            return task_reward          # strong positive: good to write success
        else:
            return -0.3                 # mild negative: don't write failures
    elif action == 2:  # SUPPRESS
        # Reward suppressing when task failed
        return 1.0 - task_reward        # reward suppressing failures
    else:  # NOOP
        return 0.0                      # neutral
```

This creates the right incentive: write when the model succeeded (reinforce the approach), suppress when it failed (discourage the approach), neutral otherwise.

**Fix the actual write execution — Phase 2 needs to fire on successes:**

The current code does `reward_update(state_vec, action_vec, reward)` after the episode. When reward=0 this is near-zero. Instead, only call `reward_update` when the reward is actually meaningful:

```python
# In session_manager.py run_episode():
if policy_action == WRITE and reward >= 0.7:
    # Only execute the write when the task actually succeeded
    action_vec = self.get_hidden_state(code)
    for mem in self.memory_modules.values():
        mem.reward_update(state_vec, action_vec, reward)
elif policy_action == SUPPRESS and reward < 0.3:
    # Execute suppression on clear failures
    action_vec = self.get_hidden_state(code)
    for mem in self.memory_modules.values():
        mem.reward_update(state_vec, action_vec, -0.5)  # negative update
```

The policy and the execution gate need to align. Right now the policy says WRITE on failures, and the execution fires a near-zero update. After the fix, the policy should say WRITE on successes, and the execution fires a strong positive TD update.

---

## The Result You Should See After This Fix

During training the policy distribution should shift to:

```
low reward (0-0.3):    NOOP~60%, SUPPRESS~30%, WRITE~10%
mid reward (0.3-0.7):  NOOP~70%, WRITE~20%, SUPPRESS~10%  
high reward (0.7-1.0): WRITE~60%, NOOP~35%, SUPPRESS~5%
```

And the weight norms should fluctuate dynamically again, not stay frozen at 11.22.

The table comparison you're aiming for:

```
No memory baseline:  12.5%
Fixed rule:          49.0% ± 7.0%
MemRL (this fix):    target >55% ± <5%
```

The lower variance is actually what demonstrates MemRL's value — the fixed rule swings wildly because it has no concept of when to stop writing. A correct policy learns to be conservative after writing successfully, preventing the interference crashes.