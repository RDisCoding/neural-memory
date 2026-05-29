# Reward-Conditioned Neural Memory Layer — Prototype

> **The goal in one sentence:** Build a working prototype of an LLM agent with an
> in-weights memory layer that updates from task outcomes at inference time —
> so the model gets measurably better at your specific tasks across sessions
> without any offline retraining.

---

## Table of Contents

1. [What We Are Building and Why](#1-what-we-are-building-and-why)
2. [How It Differs From ChatGPT, Cortogen, and Everything Else](#2-how-it-differs)
3. [The Core Mechanism — Every Component Explained](#3-the-core-mechanism)
4. [What Exactly Are State, Action, and Reward Here?](#4-state-action-reward)
5. [Why We Need Both Surprise AND Reward](#5-why-both-surprise-and-reward)
6. [Building the Memory Module in Isolation](#6-isolation-build-and-test)
7. [Problem Difficulty Calibration](#7-problem-difficulty-calibration)
8. [Build Steps — In Strict Order](#8-build-steps)
9. [Potential Failure Modes and How to Fix Them](#9-failure-modes)
10. [Evaluation Protocol](#10-evaluation-protocol)
11. [Repository Structure and Compute Requirements](#11-infrastructure)

---

## 1. What We Are Building and Why

### The Problem With Every Current LLM

ChatGPT, Claude, Gemini, and every other deployed language model share one
fundamental architectural limitation: **their weights are frozen at inference
time.** Every conversation starts from zero. The model that finishes your
thousandth conversation is bit-for-bit identical to the model that started
your first. It has learned nothing from you.

The "memory" features in these products (ChatGPT Memory, Claude Projects,
Mem0, Zep, MemGPT, Cortogen) are all variations of the same workaround:
store your conversation history in an external vector database and inject
relevant chunks back into the context window when needed. This is intelligent
retrieval — not learning. The model itself is unchanged. The retrieval system
is a librarian handing you notes — not a mind that has grown.

### What We Are Building Instead

A model where a **dedicated memory layer's weights actually change** during
your conversation — not via expensive backpropagation through the whole
network, but via a local update rule triggered by each new input token and,
crucially, by whether the model's output actually worked.

```
Standard LLM:
  Input → [Frozen Transformer] → Output
           (identical every session)

Our System:
  Input → [Frozen Base LLM] + [Memory Layer M_θ] → Output
                                     ↑
                         θ updates token-by-token during forward pass
                         θ updates again after reward arrives
                         θ is saved at session end, loaded at session start
                         → model gets better at YOUR tasks over time
```

The base model (Llama, Gemma, Mistral) provides the language ability and
world knowledge — this never changes. The memory layer M_θ is a small MLP
grafted into the transformer's decoder blocks. It learns: what approaches
worked in situations like this one, what failed, and what the user tends
to need. This knowledge lives in its weights and persists across sessions.

### Why This Is Different From Everything That Exists

| System | Where memory lives | Does the model change? | Persists across sessions? |
|---|---|---|---|
| ChatGPT Memory | External vector DB | No | Via retrieval only |
| Cortogen | External vector DB | No | Via retrieval only |
| MemGPT / Letta | External memory blocks | No | Via retrieval only |
| Titans (2025) | In-weights of memory MLP | Yes — per token, no reward | Yes — but no RL loop |
| **This prototype** | **In-weights of memory MLP** | **Yes — per token + per reward** | **Yes — full RL loop** |

The specific gap we fill: Titans (the closest prior work, arXiv:2501.00663)
updates memory weights based on surprise — how unexpected is this input?
But it has no concept of whether the action it took actually worked. Our
prototype adds the reward signal: after the model generates code and we
run it, the outcome (pass/fail) flows back into the memory weights. The
memory layer learns not just "this was surprising" but "this approach led
to success" or "this approach failed."

---

## 2. How It Differs

### Level 1 — ChatGPT / Claude / Gemini

Frozen weights. Every session is a blank slate. Context window is the only
"memory." When the window fills up, the oldest content falls out. All memory
features are RAG bolted on externally. The model itself learns nothing from you.

**The ceiling:** The model cannot stop making the same mistake twice. If you
correct it on Tuesday, it will make the same error on Wednesday.

### Level 2 — Cortogen

Also frozen weights. Also external memory. But smarter retrieval — stores your
actual conversation history across LLMs and injects relevant past context when
the current model is hallucinating or forgetting. The model is still unchanged.
Cortogen is the librarian, not the brain.

**The ceiling:** Cortogen compensates for the symptom (the model forgot) rather
than the cause (the model has no persistent memory). It retrieves facts; it
cannot retrieve learned approaches or adapt the model's behaviour.

### Level 3 — This Prototype

Memory lives in the weights of M_θ, not in a database. The model's behaviour
changes after each session. It learns which approaches work for your tasks,
adapts to your patterns, and does not repeat corrected mistakes — because the
correction updates the weights, not just a text file.

**The ceiling (honest):** The base model's raw intelligence is bounded by its
parameter count and training data. A 3B memory-augmented model will not beat
GPT-4 on single-turn general tasks. It will beat GPT-4 on your specific
repeated tasks, because it has accumulated a model of how to help you
specifically — and GPT-4 resets every conversation.

---

## 3. The Core Mechanism

### The Neural Memory Module M_θ

M_θ is a small deep MLP — a stack of linear layers with nonlinear activations.
What makes it radical is not its structure but its behaviour: its weights θ
change during every single forward pass. Normal transformers: frozen at inference.
Our memory layer: alive at inference.

```
Structure of M_θ:
  M_θ(x) = W_L · σ(W_{L-1} · ... · σ(W_1 · x))

  θ = {W_1, W_2, ..., W_L} — these weights change at test time
  L = 2 layers for prototype (scale up later)
  dim = same as base model hidden dimension (e.g. 2048 for Gemma 2B)
```

Knowledge is stored in the weight geometry — not as explicit key-value
entries, not as text. Just as the main transformer stores "the sky is blue"
as a distributed pattern across billions of weights, M_θ stores
"recursive approaches tend to work for tree problems" as a pattern across
its weights. This is what allows it to generalise.

### The Two Projection Systems

Every token x_t produces three vectors:

```
k_t = W_K · x_t    Key:   "what information does this token carry?"
v_t = W_V · x_t    Value: "what content should be stored?"
q_t = W_Q · x_t    Query: "what does this token want to retrieve?"

W_K, W_V, W_Q ∈ ℝ^{d×d} — learned during pretraining, FROZEN at inference
```

These are the same projections as in standard attention — but here they
serve two separate purposes:
- k_t and v_t feed the WRITE path: updating M_θ's weights
- q_t feeds the READ path: retrieving from the updated M_θ

### The Two-Phase Write — This Is The Critical Design

The memory operates in two completely separate phases. Understanding this
distinction is essential before writing any code.

**Phase 1 — Associative write (during forward pass, per token):**

```
For each token t in the input sequence:

  loss_assoc = ‖ M_θ(k_t) − v_t ‖²
  surprise_t = ‖ ∇_θ loss_assoc ‖     ← gradient norm through M_θ only

  if surprise_t > threshold:           ← only write surprising things
    g_t = sigmoid(W_forget · x_t)      ← adaptive decay gate
    θ ← (1 − g_t) · θ − η_assoc · surprise_t · ∇_θ loss_assoc
```

This runs during every forward pass, for every token above the surprise
threshold. It is the "learning what this input is about" phase.

**Phase 2 — Reward-conditioned write (after task completion, once per task):**

```
After task completes and reward r is observed:

  situation_action = encode(problem_hidden, solution_hidden)
  predicted_r = M_θ(situation_action).mean()          ← what memory expected
  td_error = r_actual − predicted_r                   ← how wrong was memory?

  loss_reward = (td_error)²
  θ ← θ − η_reward · ∇_θ loss_reward
```

This runs once after the sandbox returns a reward. It is the "did that
approach work?" phase.

### Memory Retrieval

After the write phases, the model reads from the updated M_θ:

```
y_mem = M_θ(q_t)    ← forward pass through updated MLP, no gradient

Gate blends memory output with standard attention output:
  g_blend = sigmoid(W_g · x_t)
  y_t = g_blend ⊙ y_mem + (1 − g_blend) ⊙ y_attn

g_blend is learnable — the model learns WHEN to trust memory vs attention.
```

**Critical property:** Retrieval cost is O(1) regardless of how long the
conversation history is. All history is compressed into θ. A model with
10 sessions of history and a model with 1000 sessions of history have
identical retrieval cost. This is impossible with RAG.

---

## 4. State, Action, and Reward

This section answers the question: *"we don't have a ready-made (state, action,
reward) tuple — so what exactly are these things and how do we construct them?
And how are the weights storing this information if we never feed them an
explicit tuple?"*

### How the Weights Store SAR — No Explicit Tuple Required

This is the most important conceptual point. The memory MLP does **not** store
(state, action, reward) as a database row. There is no lookup table. What it
stores is the *function that maps (state, action) → expected reward*, encoded
spread across the weight geometry of the MLP.

Think of it this way:

```
A regular lookup table stores:
  (situation_A, action_X) → reward 1.0
  (situation_B, action_Y) → reward 0.0

M_θ instead learns a continuous function:
  f(situation, action) = expected_reward

The function is approximated by the weight matrix pattern.
Any (situation, action) pair — even ones never seen before — gets a prediction
via the weight geometry. Nearby situations and actions get nearby predictions.
This is generalisation, and it's impossible with a lookup table.
```

Concretely, after 5 sessions of reward updates:
- The weight matrices W_1, W_2 in M_θ have been shaped by gradient descent
  on the TD loss: `(actual_reward - M_θ(situation_action))²`
- The weights are not pointing at any specific stored tuple
- They define a landscape: inputs that resemble successful past encodings
  produce high outputs; inputs resembling failed encodings produce low outputs
- A new problem that semantically resembles past successes will have its
  situation_action vector land in a "high-output" region of this landscape
  — and M_θ will predict a high reward, guiding the model toward
  similar approaches

This is exactly how Q-functions work in deep RL (DQN, TD3, SAC), except
the Q-function is M_θ itself, embedded in the transformer.

There are no pre-made tuples. You construct all three from the model's own
internal representations. Here is the exact construction.

### State — The Situation Encoding

The state is the model's compressed understanding of the current problem,
captured as a vector before any generation happens.

**How to construct it:**

```python
# Process the problem description tokens through the model
# Extract hidden states at the memory layer during this forward pass
inputs = tokenizer(problem_description, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)
    # hidden_states[layer_idx] shape: [batch=1, seq_len, hidden_dim]
    hidden = outputs.hidden_states[MEMORY_LAYER_IDX]

# State = mean of hidden states over the sequence
# (alternatives: last token, learned weighted sum)
state_vec = hidden.mean(dim=1).squeeze()   # shape: [hidden_dim]
```

**The key property you need:** Two problems that require similar solution
approaches should have similar state_vec values (cosine similarity > 0.7).
Two unrelated problems should have distant state_vecs. This happens naturally
because the base model's embeddings cluster semantically similar content.

**Test this explicitly:**

```python
# "Reverse a string" and "Reverse a list" should be close
# "Reverse a string" and "Find prime numbers" should be distant
sim_related = cosine_similarity(state("Reverse a string"), state("Reverse a list"))
sim_unrelated = cosine_similarity(state("Reverse a string"), state("Find primes"))
assert sim_related > sim_unrelated + 0.3, "State encoding not discriminative enough"
```

If this assertion fails, mean pooling is losing too much information. Switch
to using only the hidden state of the last token of the problem description,
or add a learned W_state projection trained to maximise inter-problem distance.

### Action — The Solution Encoding

The action is a vector representation of what the model generated. Since the
"action" is an entire sequence of tokens (the generated code), we need to
compress it into a single vector.

**How to construct it:**

```python
# After generation, encode the generated solution
solution_tokens = tokenizer(generated_code, return_tensors="pt").to("cuda")

with torch.no_grad():
    sol_outputs = model(**solution_tokens, output_hidden_states=True)
    sol_hidden = sol_outputs.hidden_states[MEMORY_LAYER_IDX]

action_vec = sol_hidden.mean(dim=1).squeeze()   # shape: [hidden_dim]
```

**Combine state and action for the memory:**

```python
# Option A: concatenation + learned projection (recommended)
W_sa = nn.Linear(2 * hidden_dim, hidden_dim).to("cuda")   # learned
situation_action = W_sa(torch.cat([state_vec, action_vec]))   # [hidden_dim]

# Option B: element-wise addition (simpler, slightly less expressive)
situation_action = state_vec + action_vec
```

**The key property:** Two different approaches to the same problem (e.g.
recursive vs iterative reversal) should produce different action_vecs even
if the state_vec is the same. This lets the memory learn "recursive worked,
iterative didn't" for the same problem class.

### Reward — The Outcome Scalar

This is the clean part. Run the code. Measure what happened. Normalize to [0, 1].

```python
reward = tests_passed / total_tests

# Examples:
# All 3 tests pass      → reward = 1.0   (perfect)
# 2 of 3 tests pass     → reward = 0.67  (partial credit — important!)
# Syntax error          → reward = 0.0   (complete failure)
# Runtime timeout       → reward = −0.1  (penalise infinite loops)
# Wrong output type     → reward = 0.1   (at least it ran)
```

**Why partial credit matters:** Binary 0/1 rewards are too coarse. If the
model writes a function that handles 2 of 3 edge cases, it deserves a 0.67
signal — not a 0.0. Partial credit gives the memory richer gradient and
reduces the variance of the TD error, making learning more stable.

### The Complete Tuple Construction

```python
def construct_sar_tuple(model, tokenizer, problem, generated_code, reward):
    """
    Constructs the (state, action, reward) triple from raw model internals.
    This runs AFTER generation and AFTER sandbox execution.
    """
    # State: hidden representation of the problem
    prob_inputs = tokenizer(problem["description"], return_tensors="pt").to("cuda")
    with torch.no_grad():
        prob_out = model(**prob_inputs, output_hidden_states=True)
        state_vec = prob_out.hidden_states[MEMORY_LAYER_IDX].mean(dim=1).squeeze()

    # Action: hidden representation of the generated solution
    sol_inputs = tokenizer(generated_code, return_tensors="pt").to("cuda")
    with torch.no_grad():
        sol_out = model(**sol_inputs, output_hidden_states=True)
        action_vec = sol_out.hidden_states[MEMORY_LAYER_IDX].mean(dim=1).squeeze()

    # Situation-action encoding
    situation_action = W_sa(torch.cat([state_vec, action_vec]))

    return situation_action, float(reward)
```

**Important timing:** Phase 1 (associative write) runs during the forward pass
that generates the code — the model doesn't know the reward yet. Phase 2 runs
after the sandbox returns the reward. The memory only learns *what worked* in
retrospect. This is correct — it mirrors how reinforcement learning works in
biological systems.

> **The crucial insight:** After enough episodes, M_θ has become a *value
> function* — a compressed, distributed representation of which (situation,
> approach) combinations tend to produce high rewards. During generation,
> the model reads from M_θ via the gate blend. High-reward regions of M_θ
> produce stronger memory signals that bias generation toward approaches
> that historically worked. The weights are not storing tuples; they are
> storing a learned policy in distributed form.

---

## 5. Why We Need Both Surprise AND Reward

This is a question that will come up when reading the code, and again when
reading the future-work roadmap: "if we eventually train an RL policy to
manage memory, why do we still need the surprise metric? Won't the policy
decide when to write?"

The answer is that they solve **completely different problems**, operate at
**different timescales**, and their interaction is not redundant.

### What Each One Does

**Surprise metric** — answers: *"Is this token worth writing into memory at all?"*

- Runs **during** the forward pass, for every token, in real time
- Computed as the gradient norm of the associative loss w.r.t. M_θ's weights:
  `surprise_t = ‖∇_θ ‖M_θ(k_t) − v_t‖²‖`
- High surprise = M_θ has never seen this kind of association before = write
- Low surprise = M_θ already knows this = skip the write
- Operates as a **novelty filter** on the raw token stream
- Cost: one backward pass per token through M_θ only (cheap)

**Reward-conditioned update** — answers: *"Did the approach this memory state led to actually work?"*

- Runs **once**, after the task completes and the sandbox returns a reward
- Computed as the TD error: `td_error = actual_reward − M_θ(situation_action)`
- Positive TD error = we underestimated how good this was → strengthen
- Negative TD error = we overestimated → weaken
- Operates as an **outcome signal** on the memory's value function
- Cost: one backward pass through M_θ on one vector (very cheap)

### What Breaks If You Remove One

**Without surprise (only reward):**
The memory writes every token with equal weight. For a 500-token problem,
that's 500 weight updates per forward pass — most of them on tokens like
"def", "return", ":", "(", which appear in every problem and carry zero
signal about what's different about this one. The memory saturates with
noise. The reward signal is correct but it has nothing coherent to latch
onto. Result: memory learns nothing useful despite correct reward signal.

**Without reward (only surprise):**
The memory writes what's surprising — unusual syntax, novel patterns,
unexpected constructs. But "surprising" ≠ "what led to success."
The model might strongly memorise an unusual bug it introduced, because
that's surprising, then keep repeating it — because there's no correction
signal saying "that led to failure." Result: memory learns interesting
things, not useful things.

### The Interaction — They're Sequential Filters

```
Token stream
    ↓
Phase 1: Surprise filter
    "Is this token surprising enough to write?"
    YES → write to M_θ (associative update)
    NO  → skip
    ↓
[time passes: model generates solution, sandbox runs]
    ↓
Phase 2: Reward filter
    "Did the associations written in Phase 1 lead to success?"
    HIGH REWARD → strengthen Phase 1 associations
    LOW REWARD  → weaken Phase 1 associations
```

Surprise decides **what gets through the door** (novelty gate).
Reward decides **which of those things deserve to stay** (outcome gate).
You need both.

### Why Surprise Is Still Needed in the Future RL Policy (Paper 3)

In Paper 3 (RL Memory Policy), we train a policy π_RL to decide
{WRITE_STRONG, WRITE_WEAK, UPDATE, SUPPRESS, NOOP} for each timestep.
Doesn't this make the surprise metric redundant?

**No. Here's why:**

The RL policy operates at the *episode level* — it decides memory operations
for each full input/output event (one problem attempt, one conversation turn).
The surprise metric operates at the *token level* — it filters individual
tokens within the forward pass.

The policy cannot operate token-by-token during the forward pass without
an unacceptable computational cost (it would require a full policy inference
per token). The surprise metric is a cheap, local signal that handles the
fine-grained filtering the policy cannot do. In Paper 3, the policy's action
SWRITE_STRONG means "when surprise fires, write with a high learning rate."
SUPPRESS means "when surprise fires, ignore the update anyway." The policy
controls the *strength and direction* of memory operations; surprise controls
*whether the token is eligible for an operation at all*. They are orthogonal.

---

## 6. Isolation Build and Test

**Rule:** Never attach the memory module to the LLM until it passes all
isolation tests. Debugging a broken memory module inside a 3B model is
10x harder than debugging it standalone. The isolation tests take one hour.
Finding the same bug inside the full model takes two days.

### What "Isolation" Means

No LLM. No tokenizer. No transformer. Just:
- The `NeuralMemoryModule` class
- Synthetic tensors with known properties
- Assertions that verify the mechanism works as designed

### The Four Properties You Must Verify

**Property 1: Associative storage works**
Write (k→v), then read(k). The retrieved vector should be close to v.
This verifies the MLP can store and retrieve associations in its weights.

**Property 2: Surprise decays with repetition**
Write the same (k,v) pair 20 times. The surprise on write 1 should be
significantly higher than on write 20. This verifies the gradient norm
correctly signals "I already know this."

**Property 3: Reward learning converges**
Given three fixed (situation_action→reward) pairs (1.0, 0.0, 0.5), run
100 TD update steps. The memory's predictions should converge to within
0.15 of the actual rewards. This verifies the reward signal propagates
into the weights.

**Property 4: Generalisation holds**
Train memory that encoding A → reward 1.0. Then query with encoding B,
which is a small perturbation of A (plus noise). B should receive a higher
predicted reward than an unrelated encoding C. This is the property that
makes the whole system work — similar situations get similar predicted
rewards without having been seen before.

### The Complete Isolation Test Suite

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralMemoryModule(nn.Module):
    def __init__(self, dim: int, hidden_mult: int = 2,
                 lr_assoc: float = 0.01, lr_reward: float = 0.005):
        super().__init__()
        self.dim = dim
        self.lr_assoc = lr_assoc
        self.lr_reward = lr_reward

        # M_θ: the memory MLP — this is what stores everything
        self.layers = nn.Sequential(
            nn.Linear(dim, dim * hidden_mult),
            nn.SiLU(),
            nn.Linear(dim * hidden_mult, dim)
        )

        # Projection matrices (learned during any fine-tuning phase)
        self.W_K = nn.Linear(dim, dim, bias=False)
        self.W_V = nn.Linear(dim, dim, bias=False)
        self.W_Q = nn.Linear(dim, dim, bias=False)

        # Adaptive forget gate: α_t = sigmoid(W_forget · x_t)
        self.forget_gate = nn.Linear(dim, 1)

        # Situation-action encoder for RL phase
        self.W_sa = nn.Linear(dim * 2, dim)

        # Momentum buffers (one per parameter in self.layers)
        self.momentum = {
            name: torch.zeros_like(p)
            for name, p in self.layers.named_parameters()
        }
        self.beta = 0.9  # momentum coefficient
        self.surprise_threshold = 0.1

        # Initialize forget gate bias to large negative
        # → gate starts near 0 → memory starts nearly invisible
        nn.init.constant_(self.forget_gate.bias, -5.0)

    # ─────────────────────────────────────────────
    # CORE API: these are the three methods you call
    # ─────────────────────────────────────────────

    def write(self, x: torch.Tensor) -> float:
        """
        Phase 1: Associative write from token x.
        Called during the forward pass for each token.
        Returns the surprise score.
        """
        k = self.W_K(x.detach())
        v = self.W_V(x.detach())

        pred = self.layers(k)
        loss = F.mse_loss(pred, v.detach())

        grads = torch.autograd.grad(
            loss, self.layers.parameters(),
            create_graph=False, retain_graph=False
        )
        surprise = sum(g.norm().item() for g in grads)

        if surprise < self.surprise_threshold:
            return surprise   # not surprising enough — skip write

        alpha = torch.sigmoid(self.forget_gate(x.detach())).item()
        alpha = min(alpha, 0.9)  # never allow full decay

        with torch.no_grad():
            for (name, param), grad in zip(
                    self.layers.named_parameters(), grads):
                # Clip gradient to prevent explosion
                grad_clipped = grad.clamp(-1.0, 1.0)
                self.momentum[name] = (
                    self.beta * self.momentum[name] + grad_clipped
                )
                param.mul_(1 - alpha)
                param.sub_(self.lr_assoc * self.momentum[name])

        return surprise

    def read(self, x: torch.Tensor) -> torch.Tensor:
        """
        Retrieve from memory using query derived from x.
        Called during the forward pass after writing.
        """
        q = self.W_Q(x.detach())
        return self.layers(q)

    def reward_update(self, state_vec: torch.Tensor,
                      action_vec: torch.Tensor,
                      actual_reward: float):
        """
        Phase 2: Reward-conditioned update.
        Called ONCE after task completion when reward is known.
        """
        if abs(actual_reward) < 0.01:
            return   # near-zero reward → no signal worth updating for

        # Encode (state, action) pair
        sa = self.W_sa(torch.cat([
            state_vec.detach(), action_vec.detach()
        ]))

        # Predict reward from memory
        predicted_r = self.layers(sa).mean()
        td_error = actual_reward - predicted_r.item()

        if abs(td_error) < 0.01:
            return   # memory was already accurate

        # Update toward actual reward
        loss = (predicted_r - actual_reward) ** 2
        grads = torch.autograd.grad(
            loss, self.layers.parameters(),
            create_graph=False, retain_graph=False
        )

        with torch.no_grad():
            for param, grad in zip(self.layers.parameters(), grads):
                grad_clipped = grad.clamp(-0.5, 0.5)
                param.sub_(self.lr_reward * grad_clipped)

    # ─────────────────────────────────────
    # ISOLATION TEST HELPERS
    # ─────────────────────────────────────

    def write_kv(self, k: torch.Tensor, v: torch.Tensor) -> float:
        """Direct k,v write bypassing W_K/W_V — used in isolation tests."""
        pred = self.layers(k)
        loss = F.mse_loss(pred, v.detach())
        grads = torch.autograd.grad(
            loss, self.layers.parameters(),
            create_graph=False, retain_graph=False
        )
        surprise = sum(g.norm().item() for g in grads)
        alpha = 0.01  # fixed small decay for isolation tests
        with torch.no_grad():
            for (name, param), grad in zip(
                    self.layers.named_parameters(), grads):
                self.momentum[name] = (
                    self.beta * self.momentum[name] + grad.clamp(-1, 1)
                )
                param.mul_(1 - alpha)
                param.sub_(self.lr_assoc * self.momentum[name])
        return surprise

    def get_state(self) -> dict:
        return {
            'layers': self.layers.state_dict(),
            'W_sa': self.W_sa.state_dict(),
            'momentum': {k: v.clone() for k, v in self.momentum.items()}
        }

    def load_state(self, state: dict):
        self.layers.load_state_dict(state['layers'])
        self.W_sa.load_state_dict(state['W_sa'])
        self.momentum = {k: v.clone() for k, v in state['momentum'].items()}


# ─────────────────────────────────────────────────────────────
# ISOLATION TEST SUITE — run this before touching the LLM
# ─────────────────────────────────────────────────────────────

def test_associative_storage(dim=64):
    """
    Verify: memory can store and retrieve a key-value association.
    Expected: retrieval error < 0.1 after 50 repeated writes.
    """
    mem = NeuralMemoryModule(dim=dim)
    k = torch.randn(dim)
    v = torch.randn(dim)

    for _ in range(50):
        mem.write_kv(k, v)

    retrieved = mem.read(k)
    error = F.mse_loss(retrieved, v).item()
    assert error < 0.1, f"FAIL associative storage: error={error:.4f} (need < 0.1)"
    print(f"  PASS associative_storage: retrieval_error={error:.4f}")


def test_surprise_decay(dim=64):
    """
    Verify: surprise is high on first write, low after repeated writes.
    Expected: surprise[0] > surprise[-1] * 3
    """
    mem = NeuralMemoryModule(dim=dim)
    k = torch.randn(dim)
    v = torch.randn(dim)

    surprises = [mem.write_kv(k, v) for _ in range(20)]

    assert surprises[0] > surprises[-1] * 3, (
        f"FAIL surprise_decay: "
        f"first={surprises[0]:.4f}, last={surprises[-1]:.4f} (need 3x ratio)"
    )
    print(f"  PASS surprise_decay: {surprises[0]:.4f} → {surprises[-1]:.4f}")


def test_reward_learning(dim=64):
    """
    Verify: memory converges to predict correct rewards.
    Expected: |predicted − actual| < 0.15 for all 3 pairs after 100 steps.
    """
    mem = NeuralMemoryModule(dim=dim)

    pairs = [
        (torch.randn(dim), torch.randn(dim), 1.0),   # good approach
        (torch.randn(dim), torch.randn(dim), 0.0),   # bad approach
        (torch.randn(dim), torch.randn(dim), 0.5),   # partial
    ]

    for _ in range(100):
        for s, a, r in pairs:
            pred = mem.layers(mem.W_sa(torch.cat([s, a]))).mean().item()
            mem.reward_update(s, a, r)

    for i, (s, a, expected_r) in enumerate(pairs):
        sa = mem.W_sa(torch.cat([s.detach(), a.detach()]))
        predicted = mem.layers(sa).mean().item()
        assert abs(predicted - expected_r) < 0.15, (
            f"FAIL reward_learning pair {i}: "
            f"predicted={predicted:.3f}, expected={expected_r}"
        )
    print(f"  PASS reward_learning: all 3 pairs within 0.15 tolerance")


def test_generalisation(dim=64):
    """
    Verify: similar situation-action encodings get similar reward predictions.
    Expected: similar vector predicts closer to trained reward than distant vector.
    """
    mem = NeuralMemoryModule(dim=dim)

    base_s = torch.randn(dim)
    base_a = torch.randn(dim)
    similar_s = base_s + 0.1 * torch.randn(dim)   # small perturbation
    similar_a = base_a + 0.1 * torch.randn(dim)
    different_s = torch.randn(dim)                  # unrelated
    different_a = torch.randn(dim)

    for _ in range(80):
        mem.reward_update(base_s, base_a, 1.0)

    def pred(s, a):
        sa = mem.W_sa(torch.cat([s.detach(), a.detach()]))
        return mem.layers(sa).mean().item()

    r_base    = pred(base_s, base_a)
    r_similar = pred(similar_s, similar_a)
    r_diff    = pred(different_s, different_a)

    assert abs(r_similar - 1.0) < abs(r_diff - 1.0), (
        f"FAIL generalisation: "
        f"base={r_base:.2f}, similar={r_similar:.2f}, diff={r_diff:.2f}"
    )
    print(f"  PASS generalisation: base={r_base:.2f}, "
          f"similar={r_similar:.2f}, diff={r_diff:.2f}")


def test_weight_norm_stability(dim=64):
    """
    Verify: weight norm doesn't collapse to 0 or explode over 500 writes.
    Expected: norm stays in [0.1 * initial, 10 * initial]
    """
    mem = NeuralMemoryModule(dim=dim)
    initial_norm = sum(p.norm().item() for p in mem.layers.parameters())

    for _ in range(500):
        x = torch.randn(dim)
        mem.write_kv(x, torch.randn(dim))

    final_norm = sum(p.norm().item() for p in mem.layers.parameters())
    assert final_norm > 0.1 * initial_norm, (
        f"FAIL weight collapse: norm went {initial_norm:.3f} → {final_norm:.3f}"
    )
    assert final_norm < 10 * initial_norm, (
        f"FAIL weight explosion: norm went {initial_norm:.3f} → {final_norm:.3f}"
    )
    print(f"  PASS weight_norm_stability: {initial_norm:.3f} → {final_norm:.3f}")


def run_all_isolation_tests():
    print("Running isolation tests on NeuralMemoryModule...")
    print("(These must all pass before attaching to any LLM)\n")
    test_associative_storage()
    test_surprise_decay()
    test_reward_learning()
    test_generalisation()
    test_weight_norm_stability()
    print("\nAll isolation tests passed. Safe to proceed to LLM integration.")


if __name__ == "__main__":
    run_all_isolation_tests()
```

---

## 7. Problem Difficulty Calibration

**The floor-ceiling problem** will silently invalidate your results if ignored.

### Why Easy and Hard Problems Are Both Useless

If you test on problems the base model always solves (>90% pass rate without
memory), both models will score ~90% across all sessions. The memory layer
shows no improvement — not because it doesn't work, but because there's
nothing to improve. The ceiling masks the signal.

If you test on problems the base model never solves (<10% pass rate), reward
is almost always 0. TD error is near-zero. The memory has no signal to learn
from. Nothing improves — not because the architecture is broken, but because
the problems are too hard to generate any learning signal.

**What you need:** Problems the base model passes 30–60% of the time in
Session 1. These are problems where the model sometimes gets the right
approach and sometimes doesn't — creating real reward variance for the
memory to learn from.

### The Calibration Protocol — Do This First

Before any memory experiment:

```python
def calibrate_problem_difficulty(model, tokenizer, problems,
                                 n_attempts=3, target_range=(0.3, 0.6)):
    """
    Run base model (no memory) on each problem n_attempts times.
    Keep only problems in the target pass-rate range.
    """
    calibrated = []

    for p in problems:
        passes = 0
        for attempt in range(n_attempts):
            code = generate_code(model, tokenizer, p["description"])
            reward, _ = execute_code_safely(code, p["test_cases"])
            if reward >= 0.5:
                passes += 1

        pass_rate = passes / n_attempts

        if target_range[0] <= pass_rate <= target_range[1]:
            p["baseline_pass_rate"] = pass_rate
            calibrated.append(p)
            print(f"  KEEP {p['id']}: pass_rate={pass_rate:.2f}")
        elif pass_rate < target_range[0]:
            print(f"  DROP {p['id']}: too hard (pass_rate={pass_rate:.2f})")
        else:
            print(f"  DROP {p['id']}: too easy (pass_rate={pass_rate:.2f})")

    print(f"\nCalibrated set: {len(calibrated)} / {len(problems)} problems")
    return calibrated
```

### Use Problem Clusters, Not Random Selection

Random selection gives you noisy measurement. Cluster selection lets you
measure whether the memory generalises within a class.

**5 clusters of 4 problems each (20 total):**

```
Cluster 1: String manipulation
  - Reverse a string
  - Check if palindrome
  - Count vowels
  - Capitalise every other character

Cluster 2: Array/list operations
  - Find the maximum subarray sum
  - Remove duplicates preserving order
  - Rotate array by k positions
  - Merge two sorted arrays

Cluster 3: Recursion
  - Fibonacci with memoisation
  - Flatten a nested list
  - Power function (x^n)
  - Binary search recursive

Cluster 4: Dictionary/hash operations
  - Two-sum problem
  - Group anagrams
  - Count word frequency
  - Find first non-repeating character

Cluster 5: Math/modular arithmetic
  - Check if prime
  - GCD of two numbers
  - Count set bits
  - Integer square root
```

**Why this matters:** If the memory learns "recursive approach → high reward"
from problem 3.1 (Fibonacci), it should generalise to 3.2 (flatten list).
Measuring within-cluster improvement across sessions is more informative than
overall pass rate.

### The Metric That Actually Matters

```python
# Not overall pass rate — improvement on previously-FAILED problems

def compute_improvement_rate(session_1_results, session_n_results):
    """
    Of all problems failed in session 1, what fraction pass in session N?
    This is the learning signal. Base model: ~0. Memory model: should increase.
    """
    failed_in_s1 = {r["problem_id"] for r in session_1_results if not r["passed"]}
    now_passing  = {r["problem_id"] for r in session_n_results
                    if r["problem_id"] in failed_in_s1 and r["passed"]}

    if not failed_in_s1:
        return 0.0, 0

    rate = len(now_passing) / len(failed_in_s1)
    return rate, len(failed_in_s1)


# Target: improvement_rate > 0.30 by session 5
# Baseline (no memory): improvement_rate ≈ 0.05 (random variance only)
```

---

## 8. Build Steps — In Strict Order

Do not skip steps. Each milestone must be verified before proceeding.

### Step 1 — Environment and Base Model (Day 1)

**Minimum hardware:**
- Free Colab T4 (15GB VRAM) — sufficient for Step 2 only
- Colab Pro A100 (40GB) or RunPod A10G ($0.80/hr) — required from Step 3

```bash
pip install torch transformers accelerate datasets
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_ID = "google/gemma-2-2b-it"   # 2B params, fits in 8GB VRAM at fp16

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Verify: model generates coherent Python
inputs = tokenizer(
    "Write a Python function to reverse a string:",
    return_tensors="pt"
).to("cuda")
out = model.generate(**inputs, max_new_tokens=150, temperature=0.1)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

**Milestone:** Model loads, generates coherent Python code. Move on.

---

### Step 2 — Build and Test NeuralMemoryModule in Isolation (Days 2–3)

Copy the full `NeuralMemoryModule` class and isolation test suite from
Section 6 into `src/memory_module.py`. Run:

```bash
python src/memory_module.py
```

Expected output:
```
Running isolation tests on NeuralMemoryModule...

  PASS associative_storage: retrieval_error=0.0432
  PASS surprise_decay: 2.4821 → 0.0312
  PASS reward_learning: all 3 pairs within 0.15 tolerance
  PASS generalisation: base=0.97, similar=0.81, diff=0.23
  PASS weight_norm_stability: 3.412 → 4.891

All isolation tests passed. Safe to proceed to LLM integration.
```

If any test fails, fix it before touching the LLM. Common first failures:
- `surprise_decay` fails → momentum coefficient too high, reduce beta to 0.7
- `reward_learning` fails → lr_reward too low, increase to 0.02
- `weight_norm_stability` → forget gate not clamped, add `min(alpha, 0.9)`

---

### Step 3 — Graft Memory into the LLM (Days 4–5)

```python
class MemoryAugmentedDecoderLayer(nn.Module):
    def __init__(self, original_layer, memory_module):
        super().__init__()
        self.original = original_layer       # frozen base model layer
        self.memory = memory_module          # our trainable memory
        hidden_size = memory_module.dim

        # Gate: learn when to use memory vs attention
        self.gate = nn.Linear(hidden_size, 1)
        # Critical: initialise to strongly favour attention at start
        nn.init.constant_(self.gate.bias, -5.0)

    def forward(self, hidden_states, **kwargs):
        # 1. Run frozen base model layer
        attn_out = self.original(hidden_states, **kwargs)
        if isinstance(attn_out, tuple):
            attn_hidden, *rest = attn_out
        else:
            attn_hidden, rest = attn_out, []

        B, T, D = attn_hidden.shape

        # 2. Write to memory for each token (surprise-gated)
        if self.training or self.memory.surprise_threshold > 0:
            for t in range(T):
                self.memory.write(attn_hidden[0, t, :])   # batch=1 for now

        # 3. Read from memory
        mem_out = torch.stack(
            [self.memory.read(attn_hidden[0, t, :]) for t in range(T)],
            dim=0
        ).unsqueeze(0).to(attn_hidden.dtype)

        # 4. Blend memory and attention
        g = torch.sigmoid(self.gate(attn_hidden))   # [B, T, 1]
        combined = g * mem_out + (1 - g) * attn_hidden

        return (combined, *rest) if rest else combined


def inject_memory_layers(model, layer_indices: list):
    """
    Graft NeuralMemoryModule at specified decoder layer indices.
    All base model weights remain frozen.
    """
    hidden_size = model.config.hidden_size
    memory_modules = {}

    for i, layer in enumerate(model.model.layers):
        if i in layer_indices:
            mem = NeuralMemoryModule(dim=hidden_size).to(model.device)
            model.model.layers[i] = MemoryAugmentedDecoderLayer(layer, mem)
            memory_modules[i] = mem
            print(f"  Memory injected at layer {i}")

    # Freeze everything except memory + gate weights
    for name, param in model.named_parameters():
        param.requires_grad = "memory" in name or "gate" in name

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {n_trainable:,} / {n_total:,} "
          f"({100*n_trainable/n_total:.3f}%)")
    return model, memory_modules


# Use every 8th layer — balance coverage vs overhead
MEMORY_LAYERS = [4, 8, 16, 24]
model, memory_modules = inject_memory_layers(model, MEMORY_LAYERS)
```

**Immediately verify model quality is preserved:**

```python
# Compute perplexity on 20 fixed sentences before and after injection
# Should be within 1% — if worse, gate bias not negative enough
def measure_perplexity(model, tokenizer, texts):
    total_loss = 0
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            loss = model(**inputs, labels=inputs.input_ids).loss
        total_loss += loss.item()
    return total_loss / len(texts)

ppl_after = measure_perplexity(model, tokenizer, VALIDATION_TEXTS)
print(f"Perplexity after injection: {ppl_after:.4f}")
# Should be within 1% of pre-injection baseline
```

**Milestone:** Model generates coherent output after injection, perplexity
within 1% of pre-injection. Gate is nearly-zero at initialisation.

---

### Step 4 — Code Execution Sandbox (Day 6)

```python
import subprocess, tempfile, os, ast

def execute_code_safely(code: str, test_cases: list,
                        timeout: int = 10) -> tuple:
    """
    Returns (reward: float, feedback: str)
    reward is in [-0.1, 1.0]
    """
    try:
        ast.parse(code)
    except SyntaxError as e:
        return 0.0, f"SyntaxError: {e}"

    passed = 0
    feedback = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, test in enumerate(test_cases):
            src = f"{code}\n\n_result = {test['input']}\nprint(repr(_result))"
            fpath = os.path.join(tmpdir, "sol.py")
            with open(fpath, "w") as f:
                f.write(src)

            try:
                r = subprocess.run(
                    ["python", fpath],
                    capture_output=True, text=True,
                    timeout=timeout, cwd=tmpdir
                )
                actual = r.stdout.strip()
                if actual == test["expected"].strip():
                    passed += 1
                    feedback.append(f"Test {i+1}: PASS")
                else:
                    feedback.append(
                        f"Test {i+1}: FAIL got={actual} want={test['expected']}"
                    )
            except subprocess.TimeoutExpired:
                feedback.append(f"Test {i+1}: TIMEOUT")
                return -0.1, "\n".join(feedback)   # penalise infinite loops
            except Exception as e:
                feedback.append(f"Test {i+1}: ERROR {e}")

    reward = passed / len(test_cases) if test_cases else 0.0
    return reward, "\n".join(feedback)
```

**Milestone:** Returns 1.0 for correct code, 0.0 for wrong, -0.1 for
timeout. Handles syntax errors without crashing.

---

### Step 5 — Session Manager and RL Loop (Days 7–9)

```python
import json, torch
from pathlib import Path
from datetime import datetime


class SessionManager:
    def __init__(self, model, tokenizer, memory_modules: dict,
                 user_id: str = "default"):
        self.model = model
        self.tokenizer = tokenizer
        self.memory_modules = memory_modules
        self.user_id = user_id
        self.mem_dir = Path(f"./memory_states/{user_id}")
        self.mem_dir.mkdir(parents=True, exist_ok=True)
        self.log = []

    def load_latest(self):
        checkpoints = sorted(self.mem_dir.glob("memory_*.pt"))
        if not checkpoints:
            print("No prior memory. Starting fresh.")
            return
        state = torch.load(checkpoints[-1], map_location="cuda")
        for idx, mem_state in state.items():
            self.memory_modules[idx].load_state(mem_state)
        print(f"Loaded memory: {checkpoints[-1].name}")

    def save(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.mem_dir / f"memory_{ts}.pt"
        torch.save(
            {i: m.get_state() for i, m in self.memory_modules.items()},
            path
        )
        print(f"Memory saved: {path.name}")

    def get_hidden_state(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
        # Use hidden state at first memory layer
        layer_idx = list(self.memory_modules.keys())[0]
        return out.hidden_states[layer_idx].mean(dim=1).squeeze()

    def generate_code(self, description: str) -> str:
        prompt = (
            "Solve the following Python problem. "
            "Return only the function code.\n\n"
            f"Problem: {description}\n\nSolution:\n```python\n"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=350,
                temperature=0.2,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        text = self.tokenizer.decode(
            out[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        return text.split("```")[0].strip()

    def run_episode(self, problem: dict) -> dict:
        # 1. State encoding (before generation)
        state_vec = self.get_hidden_state(problem["description"])

        # 2. Generate
        code = self.generate_code(problem["description"])

        # 3. Action encoding (after generation)
        action_vec = self.get_hidden_state(code)

        # 4. Execute → reward
        reward, feedback = execute_code_safely(code, problem["test_cases"])

        # 5. Reward-conditioned update on all memory layers
        for mem in self.memory_modules.values():
            mem.reward_update(state_vec, action_vec, reward)

        result = {
            "problem_id": problem["id"],
            "reward": reward,
            "passed": reward >= 0.9,
            "feedback": feedback,
        }
        self.log.append(result)
        status = "PASS" if result["passed"] else "FAIL"
        print(f"  [{status}] {problem['id']}: reward={reward:.2f}")
        return result

    def run_session(self, problems: list) -> dict:
        self.load_latest()
        results = [self.run_episode(p) for p in problems]
        self.save()

        pass_rate = sum(r["passed"] for r in results) / len(results)
        summary = {
            "pass_rate": pass_rate,
            "n_passed": sum(r["passed"] for r in results),
            "n_total": len(results),
            "results": results,
        }
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = self.mem_dir / f"session_{ts}.json"
        with open(log_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nSession: {pass_rate*100:.1f}% "
              f"({summary['n_passed']}/{summary['n_total']})")
        return summary
```

**Milestone:** Full loop runs. Code generates, executes, reward returns, memory
updates, state saves and loads correctly across sessions.

---

### Step 6 — Evaluation Loop (Days 10–12)

```python
def run_evaluation(n_sessions: int = 5, user_id: str = "eval"):
    """5-session evaluation. Same problems every session. Tracks improvement."""
    problems = load_calibrated_problems()   # from calibration step
    all_results = []

    for s in range(1, n_sessions + 1):
        print(f"\n{'='*50}\nSESSION {s}/{n_sessions}\n{'='*50}")
        manager = SessionManager(model, tokenizer, memory_modules,
                                 user_id=user_id)
        summary = manager.run_session(problems)
        all_results.append({"session": s, **summary})

    # Print summary table
    print(f"\n{'Session':<10} {'Pass%':<10} {'vs S1'}")
    print("-" * 35)
    s1_rate = all_results[0]["pass_rate"]
    for r in all_results:
        delta = r["pass_rate"] - s1_rate
        sign = "+" if delta >= 0 else ""
        print(f"{r['session']:<10} {r['pass_rate']*100:<10.1f}% "
              f"{sign}{delta*100:.1f}%")

    # Improvement rate on failed problems
    s1_fails = {r["problem_id"] for r in all_results[0]["results"]
                if not r["passed"]}
    s5_pass  = {r["problem_id"] for r in all_results[-1]["results"]
                if r["passed"] and r["problem_id"] in s1_fails}
    impr = len(s5_pass) / len(s1_fails) if s1_fails else 0
    print(f"\nImprovement on previously-failed problems: {impr*100:.1f}%")
    print("(Baseline/no-memory target: ~5%. Memory target: >30%)")
    return all_results
```

---

### Step 7 — Iteration and Debugging

Once the loop runs:

| Symptom | Diagnosis | Fix |
|---|---|---|
| Pass rate flat across sessions | Reward not propagating | Log whether θ changes after reward_update() |
| Pass rate goes DOWN with memory | Session 1 noise interfering | Only save updates where reward > 0.7 |
| Model outputs gibberish | Gate not initialised to -5 | Reset gate.bias to -5.0, re-run |
| NaN in outputs | Gradient explosion | Add grad.clamp(−1, 1) in write() |
| Memory module "dead" (no effect) | Gate collapsed to 0 | Add gate regularisation loss |
| Very slow inference | Too many memory writes | Raise surprise_threshold to 0.3 |

---

## 9. Failure Modes

Every failure mode listed here was identified before implementation so you
can detect and fix it in minutes rather than days.

### ① Gate Collapse — Memory Layer Has No Effect

**What happens:** The blend gate g = sigmoid(W_g · x) learns to output ≈ 0
for every input. The final output is always 100% attention, 0% memory.
The memory layer is architecturally present but functionally dead.

**How to detect:** Add a logging hook:
```python
gate_values = []
def gate_hook(module, input, output):
    g = torch.sigmoid(module.gate(input[0]))
    gate_values.append(g.mean().item())
```
If median gate value < 0.05 after 500 tokens, the gate has collapsed.

**How to fix:**
```python
# In training loop, add gate regularisation loss:
gate_loss = (mean_gate_value - 0.3) ** 2    # penalise too-low or too-high
total_loss = task_loss + 0.1 * gate_loss
```
Also: ensure gate bias is initialised to -5.0, not 0. A gate biased toward
zero at init can collapse before the reward signal has a chance to open it.

---

### ② Memory Saturation — Weights Collapse to Zero

**What happens:** The forget gate α_t ≈ 1.0 everywhere. The weight update
rule `param.mul_(1 - alpha)` nearly zeros the weights every step.
M_θ outputs near-zero for any query. Storage fails completely.

**How to detect:** Monitor weight norm after each session:
```python
norm = sum(p.norm().item() for p in mem.layers.parameters())
print(f"Memory weight norm: {norm:.4f}")   # should stay > 0.5 * initial
```

**How to fix:**
```python
# Clamp the forget gate output:
alpha = min(torch.sigmoid(self.forget_gate(x.detach())).item(), 0.9)
# Never allow alpha > 0.9 — always preserve at least 10% of old weights
```

---

### ③ Gradient Explosion in the Write Loop

**What happens:** Computing ∇_θ for every token, the gradients accumulate
across a 500-token sequence. By token 300, gradient norms are in the hundreds.
Weights blow up. Model outputs NaN within a few steps.

**How to detect:** NaN in generation output after memory injection, or
`torch.isnan(param).any()` returns True.

**How to fix (two-part):**

Part 1 — Clip gradients in the write function:
```python
grad_clipped = grad.clamp(-1.0, 1.0)
```

Part 2 — Only write on surprising tokens (already in the design):
```python
if surprise < self.surprise_threshold:   # default 0.1
    return surprise   # skip write entirely
```
This reduces the number of writes per sequence from O(seq_len) to O(novel_tokens),
cutting the gradient accumulation problem significantly.

---

### ④ State Encoding Collapse — All Problems Look the Same

**What happens:** The mean pooling of hidden states over a 200-token problem
description produces nearly identical vectors for all problems. Cosine
similarity between different problems is > 0.95. The memory generalises
so aggressively that it thinks "reverse a string" and "find all primes" are
the same situation.

**How to detect:**
```python
states = [get_hidden_state(p["description"]) for p in problems]
sims = [[F.cosine_similarity(states[i].unsqueeze(0),
                              states[j].unsqueeze(0)).item()
         for j in range(len(states))] for i in range(len(states))]
mean_sim = sum(sims[i][j] for i in range(len(states))
               for j in range(len(states)) if i != j) / (len(states)**2 - len(states))
print(f"Mean inter-problem similarity: {mean_sim:.4f}")
# Should be < 0.7. If > 0.9, encoding is collapsing.
```

**How to fix:**
```python
# Option A: use last-token hidden state instead of mean
state_vec = out.hidden_states[layer_idx][0, -1, :]   # last token only

# Option B: use only hidden states of "content tokens" (heuristic)
# skip first 10 tokens (usually "Write a Python function that...")
state_vec = out.hidden_states[layer_idx][0, 10:, :].mean(dim=0)
```

---

### ⑤ Reward Sparsity — Memory Never Gets a Real Signal

**What happens:** Problem calibration was off. The model almost always fails,
so reward is almost always 0.0 or -0.1. TD error is near-zero. Memory weights
barely update. No learning happens. You might think the architecture is broken,
but the problem is actually the data.

**How to detect:** Mean reward per session < 0.1.

**How to fix:** Go back to the calibration protocol. Your eval set is too hard
for this model. Switch to a larger model (8B instead of 2B), or select easier
problems from the calibration scan.

---

### ⑥ Session Interference — Memory From Session 1 Hurts Session 2

**What happens:** Memory weights from wrong answers in session 1 are pointing
in a direction that interferes with the correct answers for different problems
in session 2. Pass rate goes DOWN with memory compared to baseline.

**How to detect:** Session 2 pass rate < session 1 on the memory model. Baseline
model (no memory) has stable or higher pass rate than memory model.

**How to fix:**
```python
# Only apply reward update for successful episodes
if reward > 0.7:
    mem.reward_update(state_vec, action_vec, reward)
# For failures, optionally apply a mild suppression update
elif reward < 0.1:
    mem.reward_update(state_vec, action_vec, reward * 0.3)  # weakened signal
```

---

### ⑦ Base Model Degradation From Memory Grafting

**What happens:** Inserting the wrapper layer changes the residual stream.
Even with a near-zero gate, floating-point differences compound over 24+
layers and the model outputs incoherent text immediately after injection.

**How to detect:** Measure perplexity before and after injection. If increase
> 1%, the wrapper is not transparent enough at initialisation.

**How to fix:** Verify gate bias is −5.0 (not 0.0). Also verify the combined
output dtype matches the original:
```python
mem_out = mem_out.to(attn_hidden.dtype)   # must match fp16/bf16
```

---

## 10. Evaluation Protocol

### The Three Numbers

| Metric | Baseline (no memory) | Target (memory model) |
|---|---|---|
| Session 1 pass rate | ~40% (calibrated) | ~40% (same start) |
| Session 5 pass rate | ~41% (random variance) | >55% |
| Improvement on failed | ~5% (random) | >30% |

The improvement on previously-failed problems is the number that matters.
If that number is above 30% by session 5 and the baseline is near 5%, the
prototype works. Everything else is secondary.

### Baseline Models to Compare Against

You need at least two comparisons to make any claim:

1. **Same base model, no memory layer** — proves improvement comes from memory,
   not model variance or problem selection bias.

2. **Same base model + naive RAG** (Cortogen-style: store previous solutions
   in a vector DB, retrieve top-3 at each session) — proves in-weights memory
   is better than external retrieval for this task.

---

## 11. Infrastructure

### Repository Structure

```
memory-agent/
├── README_PROTOTYPE.md              ← this file
├── README_RESEARCH_ROADMAP.md       ← future work
├── src/
│   ├── memory_module.py             ← NeuralMemoryModule + isolation tests
│   ├── model_surgery.py             ← inject_memory_layers()
│   ├── sandbox.py                   ← execute_code_safely()
│   ├── session_manager.py           ← SessionManager class
│   └── eval.py                      ← calibration + evaluation loops
├── problems/
│   ├── humaneval_raw.json           ← full HumanEval dataset
│   └── calibrated_20.json           ← problems passing the difficulty scan
├── memory_states/
│   └── {user_id}/
│       ├── memory_20250101_120000.pt  ← versioned θ checkpoint
│       └── session_20250101_120000.json
└── notebooks/
    └── colab_quickstart.ipynb
```

### The Memory State File

The `.pt` file is not a pickle of Python objects — it is a PyTorch state dict:
a dictionary mapping parameter names to raw tensor data. For a 2-layer memory
MLP on top of Gemma 2B with 4 memory layers:

```
Approximate size: 4 layers × (2048×4096 + 4096×2048) weights ≈ 256MB
```

It is versioned by timestamp. You can roll back to any previous session's
memory state exactly like git — just `torch.load` an older checkpoint.

### Compute Requirements

| Stage | Hardware | Estimated Cost |
|---|---|---|
| Isolation tests (Step 2) | CPU only | Free |
| Steps 3–4 (2B, no training) | Colab free T4 | Free |
| Steps 5–6 (full eval, 2B) | Colab Pro A100 | ~$5–15 |
| Scale to 8B | RunPod A100 40GB | ~$30–60 |
| Serious RL training | Lambda Labs A100 80GB | ~$100–300 |

Start on free Colab. Only pay for compute once the loop passes all
isolation tests and produces coherent output after grafting.

### Success Criteria

The prototype is complete when all three are true:

- [ ] Memory module passes all 5 isolation tests
- [ ] Model generates coherent code after memory layer injection
      (perplexity within 1% of base model)
- [ ] Improvement-on-failed-problems metric > 30% by session 5,
      compared to ~5% for the no-memory baseline
