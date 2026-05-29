# Memory-Augmented LLM Agent — Research Roadmap

> This document describes the full research vision beyond the initial prototype.
> Each area is a self-contained research problem, roughly ordered by dependency.
> Nothing here needs to be built now — this is the map of where we are going.

---

## The Core Thesis

Current LLMs are brilliant but stateless. Every session starts from zero.
Their intelligence is frozen at training time. They cannot:

- Remember what they did with you last week
- Learn from their own mistakes without expensive retraining
- Know who they are across thousands of conversations
- Track how you feel, not just what you say
- Grow into a personalised agent that knows you better than any generic model

We are building a system that can do all of these things — by adding a
**neural memory layer** to existing open-source LLMs that updates in real time
from experience, persists across sessions, and accumulates a model of the user
that no generic frontier model can match.

The architecture builds in layers. Each layer is a research contribution.
Each one is testable independently.

```
Layer 0 (Prototype):     Reward-conditioned memory — learn from task outcomes
Layer 1 (Paper 1):       Consistency loss — protect established beliefs
Layer 2 (Paper 2):       Emotional memory — encode affective context
Layer 3 (Paper 3):       RL memory policy — learn optimal memory management
Layer 4 (Paper 0):       LifeMemBench — benchmark that makes all of this measurable
```

---

## Current State of the Field — Why This Matters

All existing "memory" in production AI systems (ChatGPT memory, Claude projects,
Mem0, Zep, MemGPT) shares the same fundamental limitation:

**Memory is external. The model itself doesn't change.**

They store conversation history in a vector database and inject relevant chunks
back into the context window. This is intelligent retrieval — not learning.
The model that finishes your 1,000th conversation is bit-for-bit identical to
the model that started your first.

The research frontier (Titans, TTT, Algorithm Distillation) has proven that
in-weights test-time learning is possible. But nobody has combined it with:
- A reward signal from task outcomes (RL loop)
- Protection of established beliefs (consistency)
- Emotional salience modulation (affective memory)
- A benchmark long enough to measure any of it (LifeMemBench)

That is the gap we are filling.

---

## Research Area 1: Consistency Loss and Identity Coherence

### The Problem

The Titans memory layer updates weights based on surprise — how unexpected
is this new information? But it has no concept of whether the new information
contradicts something the agent has already established.

If you have told the agent for 100 sessions that you prefer concise answers,
and one day you ask it a complex question that elicits a long response, the
memory layer will happily update to "this user likes long answers" — and
erase 100 sessions of accumulated knowledge. There is no protection.

More broadly: if an agent accumulates months of interaction, does it remain
the "same" agent? Is there any stable self? Currently, no. There is no
formalism for AI agent identity, no architecture that attempts it, and no
benchmark that measures it.

### The Proposed Solution

**Three-tier memory architecture:**

```
Tier 1 — Immutable Core (🔒 never auto-updates)
    - User's name, hard preferences, non-negotiables
    - Agent's fundamental character and values
    - Written once at onboarding. Requires explicit human confirmation to change.
    - Implemented as a frozen subspace of M_θ with an impenetrable gate.

Tier 2 — Protected Self-Model (🔑 updates slowly, with evidence)
    - Established beliefs about the user
    - Persistent behavioural patterns
    - Long-term goals and preferences
    - Requires 5+ consistent signals before updating. High consistency loss (large λ).

Tier 3 — Episodic Memory (✏️ updates freely)
    - Recent events, session context, task history
    - Full Titans update rule. Subject to decay.
    - Normal working memory of the agent.
```

**The consistency loss (new mathematical contribution):**

```
L_consistency(θ, Δθ) = Σ_{(k,v) ∈ S} ‖M_{θ + Δθ}(k) − v‖²

    where S = self-model: set of (key, value) pairs encoding established beliefs

Full update loss:
L_total = L_associative + λ_t · L_consistency

Adaptive λ (identity solidifies with experience):
λ_t = λ_base · (1 − e^{−τ · N_t})
    N_t = sessions seen so far
    τ   = rate of identity solidification
```

**Contradiction gate:**

```
c_t = max_{(k,v) ∈ Tier1 ∪ Tier2} cosine_distance(v_t, v)
gate_t = sigmoid(−β · (c_t − τ_threshold))
θ_t = θ_{t-1} + gate_t · Δθ_t

gate_t → 0 when c_t > τ   (contradicts established beliefs → block update)
gate_t → 1 when c_t ≈ 0   (compatible with self-model → allow update freely)
```

### Key Novelty

EWC (2017) protects weights that were important for past tasks, using Fisher
information matrices. MAML finds weight initialisations that adapt quickly.
Neither applies at inference time, and neither protects a *semantic* self-model.

The consistency loss is the first formalisation of identity-critical beliefs
as a protected subspace within a test-time learning module.

### What to Compare Against

- Titans (ablation: same model, no consistency loss)
- EWC + Titans
- MemGPT (editable memory blocks — the closest engineering analogue)
- Full-context GPT-4o (upper bound)

### Relevant Papers to Read

- Kirkpatrick et al., 2017 — Elastic Weight Consolidation (EWC) [arXiv:1612.00796]
- Finn et al., 2017 — MAML [arXiv:1703.03400]
- Packer et al., 2023 — MemGPT [arXiv:2310.08560]
- Hu et al., 2025 — Memory in the Age of AI Agents [arXiv:2512.13564]

### Venue Target

NeurIPS / ICLR. Timeline: ~4–6 months after prototype.

---

## Research Area 2: Emotional Memory

### The Problem

Every piece of information is currently treated with equal salience. A
breakthrough realisation and a throwaway remark are encoded identically.
But human memory doesn't work this way — emotionally significant events
are encoded more strongly and resist forgetting.

This is not decorative. It is adaptive. Emotionally significant moments are
more likely to matter again in the future. And in a personal AI assistant,
knowing that the user was frustrated when they asked about X is qualitatively
different from knowing they were excited — even if the factual content is the same.

### The Proposed Solution

**Emotion encoder:** A small classifier (or lightweight LLM call) that maps
each input to a (valence, arousal, dominance) vector. Valence = positive/negative.
Arousal = intensity. Dominance = control.

```
e_t = EmotionEncoder(x_t) → (valence, arousal, dominance) ∈ [−1,1]³
emo_t = |valence_t| · arousal_t  ∈ [0, 1]   (emotional salience scalar)
```

**Emotion-amplified surprise metric:**

```
s̃_t = s_t · (1 + γ_e · emo_t)

    High arousal + strong valence → more surprising than the raw gradient says
    Routine neutral text → emo_t ≈ 0 → surprise unchanged
```

**Emotion-resistant decay:**

```
α̃_t = α_t · (1 − δ_e · emo_t)

    High emotional salience → slower forgetting
    Low emotional salience → normal decay
```

**Emotional key-value storage (emotion as a memory dimension):**

```
k_t^emo = [W_K · x_t ‖ W_Ke · e_t]    (concatenate semantic + emotion)
v_t^emo = [W_V · x_t ‖ emo_t]

Retrieval:  y_t = M_θ([q_t ‖ e_t])     (query with emotional context)
```

This means the model can retrieve: "the time the user was frustrated about X"
separately from "the time the user was excited about X" — even if factually
similar.

### What This Enables (Practically)

- **Tone matching:** Agent recalls that you were stressed last Tuesday and
  adjusts its communication style accordingly without being told.
- **Emotional arc tracking:** Detects that your sentiment on a topic has been
  declining over 3 weeks and responds proactively.
- **High-stakes recall:** Memories encoded under high emotional arousal are
  retrieved more strongly — matching human episodic memory under stress.
- **Appropriate escalation:** Notices rising frustration in a session and
  changes approach before you have to explicitly say anything is wrong.

### What to Compare Against

- LUFY (emotion-driven forgetting) [arXiv:2409.12524]
- Standard Titans (no emotion) — ablation
- RAG with no emotional weighting

### Benchmark Needed

A-MBER (Apr 2026) [arXiv:2604.07017] — first benchmark for affective memory
across sessions. Also extend LifeMemBench (see Area 4) with an emotional
arc dimension.

### Relevant Papers

- LUFY [arXiv:2409.12524]
- A-MBER [arXiv:2604.07017]
- HippoRAG [arXiv:2405.14831] — hippocampal indexing (biological foundation)
- Frontiers 2024 — Information Bottleneck Hebbian Learning

### Venue Target

ACL / EMNLP / ACII (Affective Computing and Intelligent Interaction).
Timeline: ~4–5 months after consistency loss paper.

---

## Research Area 3: RL Memory Policy (MemRL)

### The Problem

The prototype uses a fixed update rule for memory: surprise-driven write,
TD-error-driven reward update. But these rules are hand-coded heuristics.
The agent doesn't *learn* what to remember — it follows a recipe written
by us. The thresholds (surprise > 0.1, reward > 0.5) are guesses.
Different users, different domains, and different task types all have
different optimal thresholds. The hand-coded rule is a one-size-fits-all
approximation that will never be optimal.

Memory-R1 (2025) proved that RL-trained memory management — where a policy
learns to decide ADD / UPDATE / DELETE / NOOP for each new piece of information
— dramatically outperforms hand-coded rules on external RAG: 28% F1 improvement
over Mem0 on LoCoMo, trained on only 152 QA pairs.

The open problem: nobody has done this for *internal* weight-space memory.
Applying RL to learn the optimal memory operation policy for a Titans-style
memory module is completely unexplored territory.

### Fixed Rule vs. Learned Policy — The Exact Difference

To make this concrete, here is the hand-coded rule the prototype uses:

```python
# Prototype (hand-coded rule):
if surprise > threshold AND reward > 0.5:
    write_strongly()        # high confidence the information is worth keeping
elif reward < 0.1:
    suppress_and_decay()    # this approach failed — weaken the association
else:
    write_weakly()          # uncertain — small update, wait for more signal
```

This works well enough to prove the concept. But "threshold", "0.5", "0.1"
are all numbers we picked. The learned policy replaces these arbitrary
choices with a trained neural network:

```
# Learned policy (Paper 3):
a_t = π_RL(surprise_t, reward_t, memory_age, task_embedding, user_context)
a_t ∈ {WRITE_STRONG, WRITE_WEAK, UPDATE, SUPPRESS, NOOP}

# π_RL is trained with GRPO on (memory_state, action, outcome) trajectories.
# The reward for π: did this memory operation improve future task performance?
# Memory-R1 proved this works for external memory.
# Applying it to internal weight-space memory = Paper 3.
```

The policy network learns when each action is appropriate from data,
not from our intuition. For a user who does primarily creative tasks,
it might learn to SUPPRESS more aggressively. For a user doing repetitive
code tasks, it might learn WRITE_STRONG on first success. This kind of
user-specific and task-specific adaptation is impossible with a fixed rule.

### The Proposed Solution

**Action space:**

```
a_t ∈ {WRITE_STRONG, WRITE_WEAK, UPDATE, SUPPRESS, NOOP}

WRITE_STRONG  — high learning rate update, strong decay suppression
WRITE_WEAK    — low learning rate update, minimal decay change
UPDATE        — modify an existing association (softer than fresh write)
SUPPRESS      — actively decay the weight region associated with this encoding
NOOP          — do nothing this timestep
```

**State representation fed to the policy:**

```
s_t = (θ_{t-1}, x_t, context, e_t)

θ_{t-1}  = current memory weight statistics (norm, sparsity, age distribution)
x_t      = current token / input encoding
context  = recent reward history, task type embedding
e_t      = emotional salience (from Area 2, if implemented)
```

Note: θ_{t-1} itself is too large to feed directly into the policy — we
use summary statistics of the current memory state (weight norms per layer,
mean age of recent writes, gradient variance). This is a tractable
representation of "what does memory currently know and how confident is it."

**Reward function for training π_RL:**

```
r_t = α · recall_accuracy
    + β · consistency_score
    − γ · memory_cost

recall_accuracy   = how well M_θ answers future questions about this episode
                    (measured by querying M_θ on held-out questions from the session)
consistency_score = how well established beliefs are preserved after the update
                    (same as Area 1's consistency loss — reused here)
memory_cost       = KL divergence from prior θ (prevents the policy from
                    writing everything WRITE_STRONG and wasting capacity)
```

**Training procedure:**

Use GRPO (Group Relative Policy Optimization — same algorithm as DeepSeek R1)
on a dataset of multi-session conversation trajectories.

```
For each decision point t:
  Sample K candidate actions from π_RL(s_t)
  Execute each action → get resulting M_θ
  Run next N interactions with resulting M_θ → measure downstream recall
  Reward each action by its downstream recall improvement
  Update π_RL using GRPO: actions that led to better recall get higher weight
```

This is the hardest of the four research areas — but the most impactful.
An agent that *learns* what to remember about *you specifically* compounds
in value over time in a way no fixed rule can match. It also creates a
competitive moat: the longer a user interacts, the better the policy
gets tuned to their patterns, and the more personalised the memory
management becomes.

### What to Compare Against

- Memory-R1 [arXiv:2508.19828] — same idea, external memory
- Fixed-rule Titans (ablation)
- ConsistencyMem prototype (Area 1)
- Full-context GPT-4o (upper bound)

### Relevant Papers

- Memory-R1 [arXiv:2508.19828]
- GRPO / DeepSeek R1 [arXiv:2501.12948]
- MemEvolve — meta-evolutionary memory architecture
- Algorithm Distillation [arXiv:2210.14215]

### Venue Target

NeurIPS / ICML. Timeline: 6–8 months after prototype.

---

## Research Area 4: LifeMemBench

### The Problem

Every benchmark currently used in memory research measures the same thing:
**factual recall accuracy within a bounded session.**

LoCoMo: 35-session conversations, ~9K tokens.
LongMemEval: 500 questions, up to 1.5M tokens.
MemBench: similar scope.

None of them measure:
- Whether an agent remains *consistent about who it is* after 1,000 sessions
- Whether it handles adversarial belief contradictions intelligently
- Whether it tracks a user's emotional arc across weeks
- Whether preferences that evolve gradually are updated smoothly
- Whether memory under a fixed budget prioritises correctly

These are the dimensions that matter for a real personal AI. And because
no benchmark measures them, no paper can claim to solve them.

### The Proposed Solution

**LifeMemBench** — a benchmark designed around a *simulated life*, not a
conversation. A months-long relationship between a simulated user and an agent.

**Design:**

```
1,000+ simulated sessions per agent
Scripted user persona with:
    - Stable core facts (name, profession, location)
    - Gradually evolving preferences (communication style changes over time)
    - Episodic emotional arcs (scripted periods of stress, excitement, frustration)
    - Deliberate contradictions injected at session 50, 200, 500 (adversarial)
    - Domain knowledge that evolves (user learns new skills over the simulation)
```

**Five evaluation dimensions:**

| Dimension | What It Measures | Metric |
|---|---|---|
| Factual recall | Standard retrieval accuracy | F1, Accuracy |
| Identity coherence | Does agent stay consistent? | ICS (below) |
| Belief update quality | Does agent handle evolving info? | BUQ score |
| Emotional arc tracking | Does agent recall affective history? | EARS score |
| Consolidation fidelity | Did agent learn the right pattern? | CF@N |

**Identity Coherence Score (ICS) — new metric definition:**

```
Given a set of established belief pairs B = {(q_i, a_i)},
query the agent at session N on all q_i.

ICS@N = (1/|B|) Σ_i sim(agent_answer_i, a_i)

where sim = semantic similarity (BERTScore or LLM-judge)

ICS should be high and stable across sessions.
ICS degrading over sessions = identity drift.
ICS dropping sharply after a contradiction injection = failed protection.
```

**Adversarial contradiction protocol:**

At sessions 50, 200, and 500, inject one statement that directly contradicts
an established Tier 2 belief. Example: if the user has established "I prefer
concise answers" for 49 sessions, inject "Actually I love long detailed
explanations." Measure:

- Does the agent blindly overwrite the old belief? (bad)
- Does the agent ignore the new statement? (also bad — should update eventually)
- Does the agent note the contradiction and update slowly over next 5 sessions? (correct)

### Why This Is a Paper on Its Own

LoCoMo (2024) and LongMemEval (2025) were both accepted at major venues
purely as benchmark contributions — not architecture papers. They are cited
by nearly every subsequent memory paper.

A benchmark paper that defines ICS, BUQ, EARS, and CF as first-class
evaluation dimensions will be cited before any architecture paper can
claim improvements on those dimensions. This should be published first.

### Venue Target

NeurIPS Datasets & Benchmarks track / ACL / EMNLP Findings.
Timeline: Can begin now, in parallel with prototype. 3–4 months.

---

## Multi-User Memory — A Separate Problem

The prototype is single-user by design. One memory state θ, one user.
But the design space for multiple users is worth understanding because
it affects architectural decisions even in the prototype.

**Option A — Per-user θ files (prototype choice):**

```
memory_states/
  alice/memory_20250601.pt   ← Alice's private weight state
  bob/memory_20250601.pt     ← Bob's private weight state
```

Each user gets their own M_θ. Simple. Clean. Scales horizontally. The
base model is shared; the memory layer is private. Downside: if Alice
and Bob both learn that "recursion works for tree problems", they each
learn it independently. No shared knowledge accumulation.

**Option B — Shared θ with user embeddings:**

```
# One shared memory layer, user-conditioned queries:
y = M_θ(W_Q · [q_t ‖ user_emb_alice])   # Alice's personalised read
y = M_θ(W_Q · [q_t ‖ user_emb_bob])     # Bob's personalised read
```

All users share one M_θ but query it with their personalised embedding
appended. Allows knowledge sharing (general task patterns are shared)
while maintaining personal context (user-specific patterns are different
directions in embedding space). Much harder to implement — user embeddings
need to be learned without interfering with each other.

**Option C — Hierarchical θ (research frontier):**

```
θ_shared   = world-level memory (updated by all users)
θ_alice    = personal model of Alice (updated by Alice's sessions only)
θ_bob      = personal model of Bob (updated by Bob's sessions only)

read: y = gate · M_{θ_shared}(q) + (1-gate) · M_{θ_personal}(q)
```

Two memory layers: a shared layer for general knowledge that improves
with more users, and a personal layer that models each user individually.
This is the right design for a production personal AI — but implementing
it cleanly is a research contribution on its own. Nobody has done this
for weight-space memory.

**Decision: use Option A for everything through Paper 2. Option B/C = Paper 3 territory.**

---

## Architecture Vision: The Full System

Once all four areas are implemented, the complete forward pass looks like this:

```
Input token x_t
    │
    ├─► EmotionEncoder(x_t) → e_t           [Area 2]
    │
    ├─► k_t, v_t, q_t projections
    │
    ├─► Surprise metric s_t = ‖∇_θ ℓ‖
    │
    ├─► Emotion-amplified surprise s̃_t = s_t · (1 + γ·emo_t)   [Area 2]
    │
    ├─► Contradiction gate c_t, gate_t       [Area 1]
    │
    ├─► RL policy π_RL(s̃_t, c_t, e_t, θ_{t-1}) → action a_t   [Area 3]
    │
    ├─► Combined loss:
    │       L = L_associative + λ_t · L_consistency + r_t · L_reward   [Areas 1+0]
    │
    ├─► Gated update:
    │       θ_t = (1−α̃_t) · θ_{t-1} + gate_t · [a_t≠NOOP] · Δθ_t
    │
    ├─► Read: y_mem = M_{θ_t}([q_t ‖ e_t])
    │
    └─► Gate with attention:
            y_t = σ(W_g·x_t) ⊙ y_mem + (1−σ(W_g·x_t)) ⊙ y_attn
```

This is not one paper. It is four. Each layer is independent and testable.
Build them in order. Don't mix them in one prototype.

---

## What This System Does vs Current Models

| Capability | ChatGPT / Claude | Cortogen + LLM | This System |
|---|---|---|---|
| Cross-session memory | External RAG only | Better RAG | In-weights |
| Learning from task outcomes | None | None | ✓ (RL loop) |
| Identity coherence | None | None | ✓ (Area 1) |
| Emotional context recall | None | None | ✓ (Area 2) |
| Optimal memory policy | Hard-coded | Hard-coded | ✓ (Area 3) |
| Improves per user over time | No | Partially | ✓ |
| Works offline / private | No | Partially | ✓ |
| Context window for memory | O(n tokens) | O(retrieved chunks) | O(1) weights |

---

## Benchmarks Summary

| Benchmark | Measures | Use For | Status |
|---|---|---|---|
| HumanEval (subset) | Code correctness | Prototype evaluation | Available now |
| LoCoMo | Multi-session factual recall | Paper 1 baseline | Available now |
| LongMemEval | 5-ability memory eval | Paper 1 baseline | Available now |
| A-MBER | Affective memory across sessions | Paper 2 (emotion) | Apr 2026 |
| MemAgentBench | Cognitive memory tasks | General comparison | Available |
| LifeMemBench | Identity coherence, emotional arc, adversarial | Papers 1–3 | **Build this** |

---

## Papers to Read (Full List)

### Foundations (read first)
- Park et al., 2023 — Generative Agents [arXiv:2304.03442]
- Packer et al., 2023 — MemGPT [arXiv:2310.08560]
- Shinn et al., 2023 — Reflexion [arXiv:2303.11366]

### The memory layer lineage (in order)
- Ba, Hinton et al., 2016 — Fast Weights [arXiv:1610.06258]
- Schlag & Schmidhuber, 2021 — DeltaNet [arXiv:2102.11174]
- Sun et al., 2024 — TTT [arXiv:2407.04620]
- Behrouz et al., 2025 — Titans [arXiv:2501.00663]
- ATLAS, 2025 — Omega rule [arXiv:2505.23735]

### The RL memory lineage
- Pritzel et al., 2017 — Neural Episodic Control [arXiv:1703.01988]
- Laskin et al., 2022 — Algorithm Distillation [arXiv:2210.14215]
- Memory-R1, 2025 — RL memory management [arXiv:2508.19828]

### Surveys (field maps)
- Hu et al., Dec 2025 — Memory in the Age of AI Agents [arXiv:2512.13564]
- Du et al., Mar 2026 — Memory for Autonomous LLM Agents [arXiv:2603.07670]

### Knowledge graphs and temporal memory
- Gutiérrez et al., 2024 — HippoRAG [arXiv:2405.14831]
- Gutiérrez et al., 2025 — HippoRAG 2 [arXiv:2502.14802]
- Rasmussen et al., 2025 — Zep / Graphiti [arXiv:2501.13956]

### Continual learning (catastrophic forgetting)
- Kirkpatrick et al., 2017 — EWC [arXiv:1612.00796]
- I-LoRA, 2024 — LoRA for continual learning [arXiv:2402.18865]

### Emotion
- LUFY [arXiv:2409.12524]
- A-MBER [arXiv:2604.07017]

---

## Dependency Graph

```
[Prototype: RL loop, code tasks]  ← START HERE
        │
        ├──► [LifeMemBench: evaluation infrastructure]  ← run in parallel
        │
        ▼
[Paper 1: Consistency loss + identity tiers]
        │
        ▼
[Paper 2: Emotional memory + A-MBER evaluation]
        │
        ▼
[Paper 3: RL memory policy (MemRL)]
```

Each node depends on the one above it.
Paper 1 needs LifeMemBench to have a meaningful evaluation.
Paper 2 needs Paper 1's consistency infrastructure.
Paper 3 needs Papers 1 and 2 to have a meaningful reward function.

---

## The One-Sentence Research Claim

> "We propose a reward-conditioned neural memory module that stores
> action-value and episodic associations in weight space — updating via
> temporal difference error and emotional salience at inference time,
> protecting established identity beliefs via a consistency gate, and
> optimising memory operations via a learned RL policy — enabling an
> LLM agent to improve task performance, maintain coherent identity,
> and build a persistent personalised model of the user across an
> unlimited number of sessions without any offline retraining."

That is four papers. Build them one at a time.
The prototype is the proof that the foundation holds.
