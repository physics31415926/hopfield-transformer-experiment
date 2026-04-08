"""
Hopfield-Enhanced Transformer Model

Three variants for comparison:
1. Vanilla Transformer (baseline)
2. HopfieldTransformer — replace attention with multi-step Hopfield attention
3. HopfieldAugmented — standard attention + associative memory layer

All share the same parameter budget for fair comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from hopfield_layers import HopfieldAttention, AssociativeMemoryLayer


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# Variant 1: Vanilla Transformer (baseline)
# ============================================================
class VanillaTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, attn_mask=mask)
        x = x + h
        x = x + self.ff(self.ln2(x))
        return x, None  # None for energy


# ============================================================
# Variant 2: Hopfield Transformer (multi-step Hopfield attention)
# ============================================================
class HopfieldTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 num_steps: int = 3, dropout: float = 0.1):
        super().__init__()
        self.attn = HopfieldAttention(d_model, num_heads, num_steps=num_steps, dropout=dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None, return_energy=False):
        h = self.ln1(x)
        h, energy = self.attn(h, h, h, mask=mask, return_energy=return_energy)
        x = x + h
        x = x + self.ff(self.ln2(x))
        return x, energy


# ============================================================
# Variant 3: Augmented — standard attn + Hopfield memory layer
# ============================================================
class HopfieldAugmentedBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 num_memories: int = 64, num_steps: int = 3, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.memory = AssociativeMemoryLayer(d_model, num_memories, num_steps)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None, return_energy=False):
        # Standard self-attention
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, attn_mask=mask)
        x = x + h
        # Hopfield associative memory retrieval
        x, energy = self.memory(self.ln2(x) + x, return_energy=return_energy)
        # FFN
        x = x + self.ff(self.ln3(x))
        return x, energy


# ============================================================
# Unified Model Wrapper
# ============================================================
class HopfieldLM(nn.Module):
    """
    Small language model for experiments.
    Supports three modes: 'vanilla', 'hopfield', 'augmented'
    """

    def __init__(self, vocab_size: int, d_model: int = 256, num_heads: int = 4,
                 d_ff: int = 512, num_layers: int = 4, max_seq_len: int = 256,
                 dropout: float = 0.1, mode: str = 'vanilla',
                 hopfield_steps: int = 3, num_memories: int = 64,
                 energy_reg_weight: float = 0.01):
        super().__init__()
        self.d_model = d_model
        self.mode = mode
        self.energy_reg_weight = energy_reg_weight

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        # Build layers
        if mode == 'vanilla':
            self.layers = nn.ModuleList([
                VanillaTransformerBlock(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ])
        elif mode == 'hopfield':
            self.layers = nn.ModuleList([
                HopfieldTransformerBlock(d_model, num_heads, d_ff, hopfield_steps, dropout)
                for _ in range(num_layers)
            ])
        elif mode == 'augmented':
            self.layers = nn.ModuleList([
                HopfieldAugmentedBlock(d_model, num_heads, d_ff, num_memories, hopfield_steps, dropout)
                for _ in range(num_layers)
            ])
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor,
                targets: Optional[torch.Tensor] = None) -> dict:
        """
        input_ids: (B, L) token indices
        targets: (B, L) target token indices for LM loss
        Returns dict with 'logits', 'loss', 'energy_loss', 'total_loss'
        """
        B, L = input_ids.shape
        device = input_ids.device

        # Causal mask
        mask = torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()
        mask = ~mask  # True = attend, for nn.MultiheadAttention we need float mask
        causal_mask = torch.zeros(L, L, device=device)
        causal_mask.masked_fill_(torch.triu(torch.ones(L, L, device=device), diagonal=1).bool(), float('-inf'))

        # Embeddings
        pos = torch.arange(L, device=device).unsqueeze(0)
        x = self.drop(self.token_emb(input_ids) + self.pos_emb(pos))

        # Forward through layers
        total_energy = 0.0
        energy_count = 0
        for layer in self.layers:
            if self.mode == 'vanilla':
                x, _ = layer(x, mask=causal_mask)
            else:
                x, energy = layer(x, mask=mask if self.mode == 'hopfield' else None,
                                  return_energy=True)
                if energy is not None:
                    total_energy = total_energy + energy.mean()
                    energy_count += 1

        x = self.ln_f(x)
        logits = self.head(x)

        result = {'logits': logits}

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            result['loss'] = loss

            # Energy regularization: encourage low energy (well-retrieved patterns)
            if energy_count > 0:
                energy_loss = self.energy_reg_weight * (total_energy / energy_count)
                result['energy_loss'] = energy_loss
                result['total_loss'] = loss + energy_loss
            else:
                result['energy_loss'] = torch.tensor(0.0, device=device)
                result['total_loss'] = loss

        return result

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(mode: str, vocab_size: int, **kwargs) -> HopfieldLM:
    """Factory function for building models."""
    defaults = dict(
        d_model=256, num_heads=4, d_ff=512, num_layers=4,
        max_seq_len=256, dropout=0.1, hopfield_steps=3,
        num_memories=64, energy_reg_weight=0.01,
    )
    defaults.update(kwargs)
    return HopfieldLM(vocab_size=vocab_size, mode=mode, **defaults)
