"""
session_manager.py
Manages the full episode loop:
  problem → generate → sandbox → reward → memory update → save state

Handles session persistence: save/load memory weights as versioned .pt files.
"""

import json
import sys
import torch
from datetime import datetime
from pathlib import Path

from sandbox import execute_humaneval
import numpy as np
try:
    from memory_policy import (
        get_policy_features, compute_partial_credit_trend,
        ACTION_STRENGTHS, WRITE, NOOP, SUPPRESS
    )
except ImportError:
    # Handle environment where memory_policy is missing temporarily
    ACTION_STRENGTHS = {0: 1.0, 1: 0.0, 2: -0.5}
    WRITE = 0
    NOOP = 1
    SUPPRESS = 2


class SessionManager:
    def __init__(self, model, tokenizer, memory_modules: dict,
                 user_id: str = "default",
                 memory_layer_idx: int = 4,
                 temperature: float = 0.2,
                 policy=None):
        self.model = model
        self.tokenizer = tokenizer
        self.memory_modules = memory_modules
        self.user_id = user_id
        # Index of first memory layer — used for state/action encoding
        self.memory_layer_idx = (
            list(memory_modules.keys())[0]
            if memory_modules else memory_layer_idx
        )
        self.temperature = temperature
        self.mem_dir = Path(f"./memory_states/{user_id}")
        self.mem_dir.mkdir(parents=True, exist_ok=True)
        self.log = []

        # MemRL fields
        self.policy = policy
        self.current_session = 1
        self.attempt_counts = {}
        self.reward_history = {}
        self.trajectory = []

    # ─────────────────────────────────────────────
    # PERSISTENCE
    # ─────────────────────────────────────────────

    def load_latest(self):
        checkpoints = sorted(self.mem_dir.glob("memory_*.pt"))
        if not checkpoints:
            print("No prior memory state. Starting fresh.")
            return
        device = next(self.model.parameters()).device
        state = torch.load(checkpoints[-1], map_location=device)
        for idx, mem_state in state.items():
            idx = int(idx)
            if idx in self.memory_modules:
                self.memory_modules[idx].load_state(mem_state)
        print(f"Loaded memory: {checkpoints[-1].name}")

    def save(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.mem_dir / f"memory_{ts}.pt"
        torch.save(
            {str(i): m.get_state() for i, m in self.memory_modules.items()},
            path
        )
        print(f"Memory saved: {path.name}")
        return path

    # ─────────────────────────────────────────────
    # ENCODING
    # ─────────────────────────────────────────────

    def get_hidden_state(self, text: str) -> torch.Tensor:
        """
        Encode text → mean-pooled hidden state at the first memory layer.
        Used to construct state_vec (problem) and action_vec (solution).
        """
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(device)

        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)

        # hidden_states[layer_idx] shape: [1, seq_len, hidden_dim]
        hidden = out.hidden_states[self.memory_layer_idx]
        return hidden.mean(dim=1).squeeze()

    # ─────────────────────────────────────────────
    # GENERATION
    # ─────────────────────────────────────────────

    def generate_code(self, prompt_text: str, temperature: float = None) -> str:
        """
        Generate a Python solution for the given problem prompt.
        Returns only the code body (strips markdown fences if present).
        """
        device = next(self.model.parameters()).device
        if temperature is None:
            temperature = self.temperature

        # HumanEval prompts already include the function signature.
        # We append a generation cue.
        full_prompt = (
            prompt_text.rstrip() +
            "\n    # Implementation:\n"
        )

        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=350,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode only newly generated tokens
        generated_ids = out[0][inputs.input_ids.shape[1]:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Strip markdown code fences if model wrapped the output
        if "```python" in text:
            text = text.split("```python")[1]
        if "```" in text:
            text = text.split("```")[0]

        # Strip trailing non-code content (explanations, markdown, etc.)
        # The model often appends "**Explanation:**" or similar after the code
        clean_lines = []
        for line in text.split("\n"):
            stripped = line.strip()
            # Stop at markdown-style headers, bold text, or horizontal rules
            if stripped.startswith("**") or stripped.startswith("##") or stripped == "---":
                break
            # Stop at lines that look like natural language paragraphs
            # (non-indented, starts with uppercase, no Python keywords)
            if (stripped and not line.startswith(" ") and not line.startswith("\t")
                and not stripped.startswith("def ") and not stripped.startswith("class ")
                and not stripped.startswith("return") and not stripped.startswith("if ")
                and not stripped.startswith("for ") and not stripped.startswith("while ")
                and not stripped.startswith("import ") and not stripped.startswith("from ")
                and not stripped.startswith("#") and not stripped.startswith("@")
                and not stripped.startswith("print") and not stripped.startswith("raise")
                and not stripped.startswith("try") and not stripped.startswith("except")
                and not stripped.startswith("with ") and not stripped.startswith("assert")
                and not stripped.startswith("yield") and not stripped.startswith("async")
                and not stripped.startswith("elif") and not stripped.startswith("else")
                and stripped[0].isupper()):
                break
            clean_lines.append(line)
        text = "\n".join(clean_lines)

        # Re-attach the original prompt so the function definition is complete
        return prompt_text + text.rstrip()

    # ─────────────────────────────────────────────
    # EPISODE
    # ─────────────────────────────────────────────

    def run_episode(self, problem: dict) -> dict:
        """
        One full episode:
        1. Encode problem → state_vec
        2. Generate code
        3. Encode solution → action_vec
        4. Execute → reward
        5. Phase 3 policy-driven memory update (or threshold rule fallback)
        """
        # 1. State (before generation — model doesn't know reward yet)
        state_vec = self.get_hidden_state(problem["prompt"])

        # Gather surprise & norms for policy features
        surprise = 0.0
        for name, module in self.model.named_modules():
            if hasattr(module, "last_mean_surprise"):
                surprise = module.last_mean_surprise
                break
        
        norms = [sum(p.norm().item() for p in mem.layers.parameters()) 
                 for mem in self.memory_modules.values()]
        
        weight_norm_mean = sum(norms) / max(len(norms), 1)
        weight_norm_std = 0.0
        if len(norms) > 1:
            mean_n = sum(norms) / len(norms)
            weight_norm_std = (sum((n - mean_n) ** 2 for n in norms) / len(norms)) ** 0.5

        # 2. Generate
        code = self.generate_code(problem["prompt"])

        # 3. Execute → reward (action encoding deferred until we know reward)
        reward, feedback = execute_humaneval(code, problem)

        # 4. Feature construction & Policy action
        pid = problem["task_id"]
        self.attempt_counts[pid] = self.attempt_counts.get(pid, 0) + 1
        history = self.reward_history.get(pid, [])
        trend = compute_partial_credit_trend(history[-3:]) if len(history) >= 2 else 0.0

        features = get_policy_features(
            surprise=surprise,
            reward=reward,
            session_num=self.current_session,
            weight_norm_mean=weight_norm_mean,
            weight_norm_std=weight_norm_std,
            problem_attempt_count=self.attempt_counts[pid],
            partial_credit_trend=trend,
            layer_idx_normalised=0.5
        )

        history.append(reward)
        self.reward_history[pid] = history

        if self.policy is not None:
            action = self.policy.get_action(features, deterministic=True)
        else:
            action = WRITE if reward > 0.7 else NOOP

        # 5. Execute Memory Action
        if self.policy is not None:
            if action == WRITE and reward >= 0.7:
                action_vec = self.get_hidden_state(code)
                for mem in self.memory_modules.values():
                    mem.reward_update(state_vec, action_vec, reward)
            elif action == SUPPRESS and reward < 0.3:
                action_vec = self.get_hidden_state(code)
                for mem in self.memory_modules.values():
                    mem.reward_update(state_vec, action_vec, -0.5)  # negative update to suppress representation
        else:
            if action == WRITE:
                action_vec = self.get_hidden_state(code)
                for mem in self.memory_modules.values():
                    mem.reward_update(state_vec, action_vec, reward)

        # Log trajectory
        self.trajectory.append({
            'features': features.tolist(),
            'action': action,
            'problem_id': pid,
            'session': self.current_session,
            'reward': reward,
        })

        result = {
            "problem_id": problem["task_id"],
            "reward": round(reward, 4),
            "passed": reward >= 0.9,
            "feedback": feedback,
            "action": action
        }
        self.log.append(result)
        status = "PASS" if result["passed"] else "FAIL"
        print(f"  [{status}] {problem['task_id']}: reward={reward:.2f} (action={action})")
        return result

    # ─────────────────────────────────────────────
    # SESSION
    # ─────────────────────────────────────────────

    def run_session(self, problems: list, session_num: int = 1) -> dict:
        """
        Run a full session over a list of problems.
        Loads prior memory state, runs all episodes, saves state.
        """
        self.current_session = session_num
        self.load_latest()
        results = [self.run_episode(p) for p in problems]
        mem_path = self.save()

        pass_rate = sum(r["passed"] for r in results) / len(results)
        summary = {
            "pass_rate": round(pass_rate, 4),
            "n_passed":  sum(r["passed"] for r in results),
            "n_total":   len(results),
            "results":   results,
            "memory_checkpoint": str(mem_path),
        }

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = self.mem_dir / f"session_{ts}.json"
        with open(log_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nSession complete: {pass_rate*100:.1f}% "
              f"({summary['n_passed']}/{summary['n_total']})")
        print(f"Log: {log_path.name}")
        return summary

    # ─────────────────────────────────────────────
    # DIAGNOSTICS
    # ─────────────────────────────────────────────

    def log_diagnostics(self):
        """Print weight norms and gate info for all memory layers."""
        print("\n── Memory Diagnostics ──")
        for layer_idx, mem in self.memory_modules.items():
            norm = sum(p.norm().item() for p in mem.layers.parameters())
            print(f"  Layer {layer_idx}: weight_norm={norm:.4f}")
