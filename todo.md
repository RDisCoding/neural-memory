The problem is not the MemRL concept — it's three specific implementation decisions that caused the collapse. Fix these in order.

---

## What the Logs Actually Tell You

The policy collapse to 100% NOOP is informative, not a failure. Read this line:

```
Training on 33/80 trajectory points with nonzero policy reward
```

Only 33 of 80 training points had any learning signal. The other 47 were zero. GRPO with 41% sparse reward on 80 total samples will always find NOOP — it's the minimum variance solution. The policy isn't wrong. It correctly concluded that writing causes more interference than it fixes, given the current setup. The problem is the setup, not the policy.

There are three specific things to fix, in order of importance.

---

## Fix 1 — Disable Phase 1 During MemRL (Critical)

This is the root cause. Phase 1 (the surprise-gated associative write during the forward pass) runs on **every token** regardless of what the policy decides. So when the policy chooses WRITE_STRONG for problem A, Phase 1 simultaneously writes noise from all the tokens in problems B through P. By the next session, the Phase 2 write from A has been partially overwritten by Phase 1 noise from 15 other problems.

The policy correctly detected this. Its NOOP decision was: "my Phase 2 write gets undone by Phase 1 anyway, so why bother."

The fix is to disable Phase 1 entirely during the MemRL evaluation phase. Make the memory layer purely reward-driven.

```python
# In MemoryAugmentedDecoderLayer.forward():
def forward(self, hidden_states, **kwargs):
    attn_out = self.original(hidden_states, **kwargs)
    if isinstance(attn_out, tuple):
        attn_hidden, *rest = attn_out
    else:
        attn_hidden, rest = attn_out, []

    B, T, D = attn_hidden.shape

    # Phase 1 write — disable this when using MemRL policy
    if getattr(self.memory, 'phase1_enabled', True):
        for t in range(T):
            self.memory.write(attn_hidden[0, t, :])

    # Read from memory
    mem_out = torch.stack(
        [self.memory.read(attn_hidden[0, t, :]) for t in range(T)],
        dim=0
    ).unsqueeze(0).to(attn_hidden.dtype)

    g = torch.sigmoid(self.gate(attn_hidden))
    combined = g * mem_out + (1 - g) * attn_hidden
    return (combined, *rest) if rest else combined
```

Then when running MemRL:
```python
# Disable Phase 1 for all memory modules
for mem in memory_modules.values():
    mem.phase1_enabled = False
```

With Phase 1 disabled, the policy has full control over what gets written. Nothing undermines it between sessions. This is the single biggest change.

---

## Fix 2 — Replace the Reward Signal

The delayed K=2 session reward is too sparse and too noisy. The policy needs a denser, more immediate signal.

**Current (broken):**
```
policy_reward = mean_reward_next_2_sessions - reward_this_session
```
This requires waiting 2 sessions and then attributing the change to one decision among 16 per session. The signal is too diluted.

**Replacement — per-problem write quality:**

```python
def compute_policy_reward(
    problem_id: str,
    action: int,
    task_reward: float,
    session_results: list,    # all session results so far
    current_session: int,
    K: int = 1               # only look 1 session ahead
) -> float:
    """
    Dense, immediate policy reward:
    
    If action was WRITE (0,1,2):
        policy_reward = next_session_reward - current_reward
        (did the write help this specific problem next time?)
    
    If action was SUPPRESS (3):
        policy_reward = -(next_session_reward)
        (penalise suppression if the problem would have passed)
    
    If action was NOOP (4):
        policy_reward = 0  
        (neutral — neither helped nor hurt)
    """
    if current_session + 1 >= len(session_results):
        return 0.0  # no next session to evaluate against
    
    next_reward = get_problem_reward(
        problem_id, current_session + 1, session_results
    )
    
    if action in [0, 1, 2]:  # WRITE actions
        return next_reward - task_reward
    elif action == 3:  # SUPPRESS
        return -next_reward
    else:  # NOOP
        return 0.0
```

This gives the policy a reward for every WRITE decision immediately in the next session — dense, attributable, and directly tied to the action. NOOP is neutral (reward=0), so the policy has no incentive to always NOOP. WRITE has positive reward when it helps and negative when it hurts.

---

## Fix 3 — Fix the GRPO Training

The current GRPO has three problems: too many epochs on too little data, no entropy regularization, and it only trains on non-zero reward points (discarding 47 perfectly valid zero-reward NOOP decisions).

```python
def train_grpo(policy, trajectories, 
               n_epochs=50,           # was 200 — too many
               K_samples=8,
               entropy_coef=0.01):    # prevents collapse
    
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    
    # Include ALL trajectory points, not just nonzero
    # Zero-reward NOOPs are valid training signal
    print(f"Training on {len(trajectories)} trajectory points")
    
    for epoch in range(n_epochs):
        total_loss = 0.0
        
        for features, logged_action, policy_reward in trajectories:
            features = features.unsqueeze(0)
            logits = policy(features).squeeze(0)
            
            # Sample K actions from current policy
            probs = F.softmax(logits, dim=-1)
            sampled_actions = torch.multinomial(probs, K_samples, replacement=True)
            
            # Estimate rewards for sampled actions using logged data
            sampled_rewards = []
            for a in sampled_actions:
                a = a.item()
                if a == logged_action:
                    sampled_rewards.append(policy_reward)
                elif a == 4:  # NOOP
                    sampled_rewards.append(0.0)
                elif a in [0, 1, 2]:  # any WRITE
                    # Scale write reward by action strength
                    scale = [2.0, 1.0, 0.5][a]
                    sampled_rewards.append(policy_reward * scale)
                else:  # SUPPRESS
                    sampled_rewards.append(-abs(policy_reward))
            
            sampled_rewards = torch.tensor(sampled_rewards)
            
            # GRPO: advantages = rewards - mean
            mean_r = sampled_rewards.mean()
            std_r = sampled_rewards.std() + 1e-8
            advantages = (sampled_rewards - mean_r) / std_r
            
            # Policy gradient loss
            log_probs = F.log_softmax(logits, dim=-1)
            pg_loss = -sum(
                advantages[i] * log_probs[sampled_actions[i]]
                for i in range(K_samples)
            ) / K_samples
            
            # Entropy regularization — CRITICAL to prevent collapse
            entropy = -(probs * log_probs).sum()
            
            loss = pg_loss - entropy_coef * entropy
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            with torch.no_grad():
                # Check distribution hasn't collapsed
                sample_features = trajectories[0][0].unsqueeze(0)
                sample_logits = policy(sample_features).squeeze(0)
                dist = F.softmax(sample_logits, dim=-1)
                action_names = ['WS', 'WN', 'WW', 'SUP', 'NOOP']
                dist_str = ', '.join(f'{n}={v:.2f}' for n,v in zip(action_names, dist))
                print(f"  Epoch {epoch:3d}: loss={total_loss/len(trajectories):.4f}  [{dist_str}]")
```

The entropy regularization is the key addition. Without it, GRPO always finds a corner solution where one action has probability 1.0.

---

## Fix 4 — Collect More Trajectory Data

80 points from 5 sessions is too small for GRPO. With Phase 1 disabled, you need 10 sessions of collection to get ~160 points. More importantly, with the new per-problem reward signal (Fix 2), you need at least one "next session" to evaluate against — so collect 6 sessions but only generate policy rewards for sessions 1-5.

```python
TRAJECTORY_SESSIONS = 6   # collect 6, train on 1-5
EVAL_SESSIONS = 5         # then evaluate trained policy for 5 sessions
```

---

## Fix 5 — Simplify the Action Space

Start with 3 actions, not 5. The problem with 5 actions on 80 training points is that SUPPRESS and WRITE_WEAK never get enough examples to learn when to use them. Collapse to NOOP is the result.

```python
# Phase 1: train with 3 actions
ACTIONS_SIMPLE = {
    0: 'WRITE',    # apply the TD update at full strength
    1: 'NOOP',     # do nothing  
    2: 'SUPPRESS'  # apply negative TD update
}

# Phase 2: add WRITE_STRONG and WRITE_WEAK after the 3-action policy works
```

---

## The Full Expected Behavior After These Fixes

With Phase 1 disabled and the new reward signal, you should see the policy learn a meaningful distribution by epoch 30. The target distribution:

```
High reward (task passes, reward > 0.7):
    → WRITE ~60-70%, NOOP ~30-40%
    
Low reward (task fails, reward < 0.3):
    → NOOP ~70-80%, SUPPRESS ~20-30%
    
Partial credit (0.3-0.7):
    → NOOP ~50%, WRITE ~30%, SUPPRESS ~20%
```

If you see this distribution, the policy has learned something meaningful. If NOOP is still >90% after adding entropy regularization, the policy reward signal itself is broken — check that `compute_policy_reward` is returning non-zero values for WRITE decisions.

---

## Expected Results After Fixes

The comparison you want to show in the paper:

```
No memory baseline:     12.5%
Fixed-rule memory:      49.0% ± 7.0%
MemRL policy:           target >65% ± <5%
```

The reduced variance is actually the more important claim. The fixed-rule system swings between 40-75% across sessions. A working policy should narrow that to 55-70% by learning when NOT to write, eliminating the interference-driven crashes.

Start with Fix 1 and Fix 2 together — those two alone should change the result significantly even before touching GRPO. Run a quick 3-session test with Phase 1 disabled and the new reward signal before doing the full training run.