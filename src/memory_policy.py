"""
memory_policy.py
MemRL Policy Network — learns WHEN and HOW STRONGLY to write to memory.

Replaces the fixed `if reward > 0.7` threshold with a learned policy
trained via GRPO (Group Relative Policy Optimization).

Actions:
    WRITE_STRONG = 0   # lr_reward * 2.0
    WRITE_NORMAL = 1   # lr_reward * 1.0
    WRITE_WEAK   = 2   # lr_reward * 0.3
    SUPPRESS     = 3   # negative update (penalise this approach)
    NOOP         = 4   # do nothing

Features (8-dim):
    0: surprise           — how novel is this input? (0–5)
    1: reward             — what reward did we get? (0–1)
    2: session_num / 10   — how far into training? (0–1)
    3: weight_norm_mean   — how full is memory? (normalised)
    4: weight_norm_std    — how unbalanced are layers? (normalised)
    5: attempt_count / 5  — how many times seen this problem?
    6: partial_credit_trend — is partial credit trending up/down? (-1 to 1)
    7: layer_idx_norm     — which layer? (0–1, or 0.5 if averaged)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional


# ─────────────────────────────────────────────────────────────
# ACTION SPACE
# ─────────────────────────────────────────────────────────────

WRITE    = 0
NOOP     = 1
SUPPRESS = 2

ACTION_NAMES = {
    WRITE:    "WRITE",
    NOOP:     "NOOP",
    SUPPRESS: "SUPPRESS",
}

# Strength multiplier for each action (applied to lr_reward)
ACTION_STRENGTHS = {
    WRITE:     1.0,
    NOOP:      0.0,
    SUPPRESS: -0.5,
}


# ─────────────────────────────────────────────────────────────
# POLICY NETWORK
# ─────────────────────────────────────────────────────────────

class MemoryPolicy(nn.Module):
    """
    Tiny 3-layer MLP: 8 input features → 5 action logits.
    This is NOT the LLM — it's the memory manager.
    Total parameters: ~5K (negligible).
    """
    def __init__(self, n_features: int = 8, n_actions: int = 3,
                 hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_actions),
        )
        self.n_actions = n_actions

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Returns raw logits over actions."""
        return self.net(features)

    def get_action(self, features: torch.Tensor,
                   deterministic: bool = False) -> int:
        """
        Select an action given a feature vector.
        deterministic=True → argmax (for evaluation)
        deterministic=False → sample from softmax (for exploration)
        """
        with torch.no_grad():
            logits = self.forward(features)
            if deterministic:
                return logits.argmax(dim=-1).item()
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, 1).item()

    def get_action_probs(self, features: torch.Tensor) -> torch.Tensor:
        """Return action probabilities (for logging/diagnostics)."""
        with torch.no_grad():
            return F.softmax(self.forward(features), dim=-1)


# ─────────────────────────────────────────────────────────────
# FEATURE BUILDER
# ─────────────────────────────────────────────────────────────

def get_policy_features(surprise: float, reward: float,
                        session_num: int,
                        weight_norm_mean: float,
                        weight_norm_std: float,
                        problem_attempt_count: int,
                        partial_credit_trend: float,
                        layer_idx_normalised: float = 0.5
                        ) -> torch.Tensor:
    """
    Build the 8-dimensional input feature vector for the policy.
    All values are normalised to roughly [0, 1] range.
    """
    return torch.tensor([
        min(surprise / 5.0, 1.0),            # 0: surprise (clamped)
        reward,                               # 1: reward (already 0–1)
        session_num / 10.0,                   # 2: session progress
        min(weight_norm_mean / 10.0, 1.0),    # 3: memory fullness
        min(weight_norm_std / 5.0, 1.0),      # 4: layer imbalance
        min(problem_attempt_count / 5.0, 1.0),# 5: attempt count
        max(min(partial_credit_trend, 1.0), -1.0),  # 6: trend
        layer_idx_normalised,                 # 7: layer position
    ], dtype=torch.float32)


def compute_partial_credit_trend(reward_history: list) -> float:
    """
    Compute the linear trend of the last few rewards for a problem.
    Positive → improving, negative → degrading, 0 → flat/no history.

    Uses least-squares slope on the last 3 entries.
    """
    if len(reward_history) < 2:
        return 0.0

    recent = reward_history[-3:]  # last 3 rewards
    n = len(recent)
    x = np.arange(n, dtype=np.float64)
    y = np.array(recent, dtype=np.float64)

    # Least-squares slope: (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
    sx = x.sum()
    sy = y.sum()
    sxy = (x * y).sum()
    sxx = (x * x).sum()
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-8:
        return 0.0

    slope = (n * sxy - sx * sy) / denom
    return float(np.clip(slope, -1.0, 1.0))


# ─────────────────────────────────────────────────────────────
# POLICY REWARD (CREDIT ASSIGNMENT)
# ─────────────────────────────────────────────────────────────

def compute_policy_rewards(trajectory_data: List[Dict], K: int = 1):
    """
    Dense, immediate policy reward with counterfactual for NOOP.
    
    WRITE:    policy_reward = next_session_reward - current_reward
              (positive = write helped, negative = write hurt)
    SUPPRESS: policy_reward = -(next_session_reward)
              (penalise if the problem would have passed anyway)
    NOOP:     policy_reward = -(next_session_reward - current_reward)
              (counterfactual: if improvement happened WITHOUT writing,
               NOOP was fine. If degradation happened, NOOP was fine too.
               But if next session improved and we didn't write, we missed
               an opportunity — that's negative for NOOP.)
    """
    reward_lookup = {}
    for traj in trajectory_data:
        pid = traj['problem_id']
        sess = traj['session']
        if pid not in reward_lookup:
            reward_lookup[pid] = {}
        reward_lookup[pid][sess] = traj['reward']

    for traj in trajectory_data:
        pid = traj['problem_id']
        sess = traj['session']
        current_reward = traj['reward']
        action = traj['action']

        if sess + 1 in reward_lookup.get(pid, {}):
            next_reward = reward_lookup[pid][sess + 1]
            delta = next_reward - current_reward
            
            if action == WRITE:
                # Write helped if next session improved
                traj['policy_reward'] = delta
            elif action == SUPPRESS:
                # Suppress is bad if the problem would have passed
                traj['policy_reward'] = -next_reward
            else:  # NOOP
                # Counterfactual: if things improved without writing,
                # NOOP was neutral. If things degraded, NOOP was correct
                # (we avoided writing bad patterns). If things could have
                # improved with a write, NOOP missed an opportunity.
                # Use negative delta: NOOP is penalised when improvement
                # could have happened (delta > 0 → we should have written)
                traj['policy_reward'] = -delta * 0.5
        else:
            traj['policy_reward'] = 0.0


# ─────────────────────────────────────────────────────────────
# GRPO TRAINER
# ─────────────────────────────────────────────────────────────

def train_policy(trajectory_data: List[Dict],
                 n_epochs: int = 50,
                 lr: float = 3e-4,
                 K_samples: int = 8,
                 entropy_coef: float = 0.05,
                 print_every: int = 10) -> MemoryPolicy:
    """
    Full GRPO training loop with entropy regularization.
    Returns the trained MemoryPolicy.
    """
    compute_policy_rewards(trajectory_data, K=1)

    # We train on all data points where we have a next session (so policy_reward can be computed)
    # The last session points have policy_reward = 0.0 but we can skip them to be safe, 
    # or just include everything. Let's include everything except the last session of each problem.
    train_data = [t for t in trajectory_data if t['session'] < max(x['session'] for x in trajectory_data)]
    print(f"Training on {len(train_data)} trajectory points")

    policy = MemoryPolicy(n_actions=3)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    for epoch in range(n_epochs):
        total_loss = 0.0
        
        for traj in train_data:
            features = torch.tensor(traj['features'], dtype=torch.float32).unsqueeze(0)
            logged_action = traj['action']
            policy_reward = traj.get('policy_reward', 0.0)

            logits = policy(features).squeeze(0)
            probs = F.softmax(logits, dim=-1)
            
            # Sample actions
            sampled_actions = torch.multinomial(probs, K_samples, replacement=True)
            
            sampled_rewards = []
            for a in sampled_actions:
                a = a.item()
                if a == logged_action:
                    sampled_rewards.append(policy_reward)
                elif a == NOOP:
                    sampled_rewards.append(0.0)
                elif a == WRITE:
                    sampled_rewards.append(policy_reward)
                else: # SUPPRESS
                    sampled_rewards.append(-abs(policy_reward))
                    
            sampled_rewards = torch.tensor(sampled_rewards)
            
            mean_r = sampled_rewards.mean()
            std_r = sampled_rewards.std() + 1e-8
            advantages = (sampled_rewards - mean_r) / std_r
            
            log_probs = F.log_softmax(logits, dim=-1)
            pg_loss = -sum(
                advantages[i] * log_probs[sampled_actions[i]]
                for i in range(K_samples)
            ) / K_samples
            
            entropy = -(probs * log_probs).sum()
            loss = pg_loss - entropy_coef * entropy
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
        if epoch % print_every == 0 or epoch == n_epochs - 1:
            with torch.no_grad():
                sample_features = torch.tensor(train_data[0]['features'], dtype=torch.float32)
                probs = policy.get_action_probs(sample_features)
                dist_str = ", ".join(f"{ACTION_NAMES[i]}={probs[i]:.2f}" for i in range(3))
                print(f"  Epoch {epoch:3d}: loss={total_loss/len(train_data):.4f}  [{dist_str}]")

    return policy


# ─────────────────────────────────────────────────────────────
# DIAGNOSTICS
# ─────────────────────────────────────────────────────────────

def print_action_distribution(action_counts: dict):
    """Pretty-print action distribution from a session."""
    total = sum(action_counts.values())
    if total == 0:
        print("  (no actions recorded)")
        return
    print("  Action distribution:")
    for action_id in sorted(action_counts.keys()):
        name = ACTION_NAMES.get(action_id, f"UNKNOWN_{action_id}")
        count = action_counts[action_id]
        pct = 100 * count / total
        bar = "█" * int(pct / 2)
        print(f"    {name:14s}: {count:3d} ({pct:5.1f}%) {bar}")


def analyze_policy(policy: MemoryPolicy, trajectory_data: List[Dict]):
    """
    Analyze the trained policy's decisions across the trajectory.
    Shows correlations between features and actions.
    """
    print("\n── Policy Analysis ──")

    # Compute policy's action for each trajectory point
    action_by_reward_bin = {
        'low (0-0.3)': [],
        'mid (0.3-0.7)': [],
        'high (0.7-1.0)': [],
    }

    action_by_session = {}

    for traj in trajectory_data:
        features = torch.tensor(traj['features'], dtype=torch.float32)
        action = policy.get_action(features, deterministic=True)
        reward = traj['reward']
        session = traj['session']

        # Bin by reward
        if reward < 0.3:
            action_by_reward_bin['low (0-0.3)'].append(action)
        elif reward < 0.7:
            action_by_reward_bin['mid (0.3-0.7)'].append(action)
        else:
            action_by_reward_bin['high (0.7-1.0)'].append(action)

        # Bin by session
        if session not in action_by_session:
            action_by_session[session] = []
        action_by_session[session].append(action)

    # Print reward-conditioned action distribution
    print("\n  Actions by reward range:")
    for bin_name, actions in action_by_reward_bin.items():
        if not actions:
            continue
        counts = {i: actions.count(i) for i in range(3)}
        total = len(actions)
        dist = ", ".join(
            f"{ACTION_NAMES[i]}={100*counts.get(i,0)/total:.0f}%"
            for i in range(3)
        )
        print(f"    {bin_name}: [{dist}]")

    # Print NOOP rate by session
    print("\n  NOOP rate by session (should increase):")
    for session in sorted(action_by_session.keys()):
        actions = action_by_session[session]
        noop_rate = actions.count(NOOP) / len(actions) if actions else 0
        bar = "█" * int(noop_rate * 20)
        print(f"    Session {session}: {noop_rate*100:5.1f}% {bar}")


# ─────────────────────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running memory_policy smoke tests...\n")

    # Test 1: Policy forward pass
    policy = MemoryPolicy()
    features = get_policy_features(
        surprise=2.0, reward=0.8, session_num=3,
        weight_norm_mean=5.0, weight_norm_std=2.0,
        problem_attempt_count=2, partial_credit_trend=0.3,
        layer_idx_normalised=0.5
    )
    action = policy.get_action(features)
    assert 0 <= action <= 4, f"Invalid action: {action}"
    print(f"  PASS forward pass: action={ACTION_NAMES[action]}")

    # Test 2: Feature builder
    f = get_policy_features(0, 0, 0, 0, 0, 0, 0, 0)
    assert f.shape == (8,), f"Wrong shape: {f.shape}"
    print(f"  PASS feature builder: shape={f.shape}")

    # Test 3: Partial credit trend
    trend = compute_partial_credit_trend([0.14, 0.14, 0.29, 0.57])
    assert trend > 0, f"Expected positive trend, got {trend}"
    print(f"  PASS partial_credit_trend: {trend:.3f} (positive, correct)")

    trend_flat = compute_partial_credit_trend([0.5, 0.5, 0.5])
    assert abs(trend_flat) < 0.01, f"Expected ~0 trend, got {trend_flat}"
    print(f"  PASS flat trend: {trend_flat:.3f}")

    # Test 4: Policy reward computation
    fake_trajectory = [
        {'problem_id': 'P1', 'session': 1, 'reward': 0.0, 'features': f.tolist(), 'action': NOOP},
        {'problem_id': 'P1', 'session': 2, 'reward': 0.5, 'features': f.tolist(), 'action': WRITE},
        {'problem_id': 'P1', 'session': 3, 'reward': 1.0, 'features': f.tolist(), 'action': WRITE},
    ]
    compute_policy_rewards(fake_trajectory, K=1)
    # P1 session 1: action=NOOP, delta=0.5-0.0=0.5, reward = -0.5*0.5 = -0.25
    assert abs(fake_trajectory[0]['policy_reward'] - (-0.25)) < 0.01
    # P1 session 2: action=WRITE, delta=1.0-0.5=0.5, reward = 0.5
    assert abs(fake_trajectory[1]['policy_reward'] - 0.5) < 0.01
    # P1 session 3: no future → 0.0
    assert fake_trajectory[2]['policy_reward'] == 0.0
    print(f"  PASS policy reward computation")

    # Test 5: GRPO training doesn't crash
    policy = train_policy(fake_trajectory, n_epochs=2, lr=1e-3, K_samples=4, entropy_coef=0.05)
    print(f"  PASS GRPO update")

    print("\nAll memory_policy smoke tests passed.")
