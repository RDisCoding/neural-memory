"""
model_surgery.py
Grafts NeuralMemoryModule into a HuggingFace decoder-only LLM.
All base model weights remain frozen.
Only memory + gate weights are trainable.
"""

import torch
import torch.nn as nn
from memory_module import NeuralMemoryModule


class MemoryAugmentedDecoderLayer(nn.Module):
    def __init__(self, original_layer, memory_module):
        super().__init__()
        self.original = original_layer   # frozen base model layer
        self.memory   = memory_module    # trainable memory
        hidden_size = memory_module.dim

        # Gate: learn when to use memory vs attention
        self.gate = nn.Linear(hidden_size, 1)
        # Init weights to zero and bias strictly negative 
        # so memory starts perfectly invisible regardless of input magnitude
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -5.0)

        # Track mean surprise across tokens for the MemRL policy
        self.last_mean_surprise = 0.0

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "original":
                raise
            return getattr(self.original, name)

    def forward(self, hidden_states, **kwargs):
        # 1. Run frozen base model layer
        attn_out = self.original(hidden_states, **kwargs)
        if isinstance(attn_out, tuple):
            attn_hidden, *rest = attn_out
        else:
            attn_hidden, rest = attn_out, []

        B, T, D = attn_hidden.shape

        # 2. Write to memory for each token (surprise-gated) if enabled
        surprises = []
        if getattr(self.memory, 'phase1_enabled', True):
            for t in range(T):
                s = self.memory.write(attn_hidden[0, t, :])   # batch=1
                surprises.append(s)
            self.last_mean_surprise = sum(surprises) / max(len(surprises), 1)
        else:
            self.last_mean_surprise = 0.0

        # 3. Read from memory for each token
        mem_out = torch.stack(
            [self.memory.read(attn_hidden[0, t, :]) for t in range(T)],
            dim=0
        ).unsqueeze(0).to(attn_hidden.dtype)   # must match fp16/bf16

        # 4. Blend memory and attention via learned gate
        g = torch.sigmoid(self.gate(attn_hidden))   # [B, T, 1]
        combined = g * mem_out + (1 - g) * attn_hidden

        return (combined, *rest) if rest else combined


def inject_memory_layers(model, layer_indices: list) -> tuple:
    """
    Graft NeuralMemoryModule at specified decoder layer indices.
    Returns (modified model, dict of {layer_idx: NeuralMemoryModule}).

    After injection:
    - All original model weights are frozen
    - Only memory MLP weights and gate weights are trainable
    """
    hidden_size = model.config.hidden_size
    memory_modules = {}

    for i, layer in enumerate(model.model.layers):
        if i in layer_indices:
            # Find exactly which GPU this specific layer is on (for device_map="auto")
            layer_device = next(layer.parameters()).device
            mem = NeuralMemoryModule(dim=hidden_size)
            augmented = MemoryAugmentedDecoderLayer(layer, mem).to(
                dtype=model.dtype, device=layer_device
            )
            model.model.layers[i] = augmented
            memory_modules[i] = mem
            print(f"  Memory injected at layer {i} (device={layer_device}, dtype={model.dtype})")

    # Freeze ALL parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze ONLY the injected memory + gate submodules
    for i in layer_indices:
        augmented = model.model.layers[i]
        for param in augmented.memory.parameters():
            param.requires_grad = True
        for param in augmented.gate.parameters():
            param.requires_grad = True

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {n_trainable:,} / {n_total:,} "
          f"({100 * n_trainable / n_total:.3f}%)")

    return model, memory_modules


def measure_perplexity(model, tokenizer, texts: list) -> float:
    """
    Compute mean cross-entropy loss over a list of texts.
    Use to verify injection does not degrade base model quality.
    Delta should be < 1% vs pre-injection baseline.
    """
    model.eval()
    total_loss = 0.0
    device = next(model.parameters()).device

    # Temporarily disable writing to memory during evaluation
    write_enabled = {}
    for name, module in model.named_modules():
        if isinstance(module, MemoryAugmentedDecoderLayer):
            write_enabled[name] = getattr(module.memory, "write_enabled", True)
            module.memory.write_enabled = False

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            loss = model(**inputs, labels=inputs.input_ids).loss
        total_loss += loss.item()

    # Re-enable writing
    for name, module in model.named_modules():
        if isinstance(module, MemoryAugmentedDecoderLayer):
            module.memory.write_enabled = write_enabled.get(name, True)

    return total_loss / len(texts)


def get_gate_stats(model) -> dict:
    """
    Inspect gate activation statistics across all injected layers.
    Median gate value < 0.05 after 500 tokens → gate has collapsed (bad).
    """
    stats = {}
    for i, layer in enumerate(model.model.layers):
        if isinstance(layer, MemoryAugmentedDecoderLayer):
            gate_weight = layer.gate.weight
            gate_bias   = layer.gate.bias
            stats[i] = {
                "gate_bias": gate_bias.item(),
                "gate_weight_norm": gate_weight.norm().item(),
            }
    return stats


def log_weight_norms(memory_modules: dict):
    """Print weight norms for all memory layers. Use as a diagnostic."""
    for layer_idx, mem in memory_modules.items():
        norm = sum(p.norm().item() for p in mem.layers.parameters())
        print(f"  Layer {layer_idx} memory weight norm: {norm:.4f}")
