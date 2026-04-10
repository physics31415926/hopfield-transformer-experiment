"""
Modern Hopfield Network Layers for Transformer Enhancement

Reference: Ramsauer et al. (2021) "Hopfield Networks is All You Need"
Core insight: softmax attention = one-step update of Modern Continuous Hopfield Network
    Energy: E = -lse(β, X^T ξ) + 0.5 * ξ^T ξ
    Update: ξ_new = X · softmax(β * X^T ξ)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class ModernHopfieldLayer(nn.Module):
    """
    Modern Continuous Hopfield Network.
    Stores patterns X and retrieves via iterative energy minimization.

    Standard attention is a special case with num_steps=1.
    Multiple steps allow convergence to sharper, more precise retrievals.
    """

    def __init__(self, dim: int, num_steps: int = 3, beta_init: float = 1.0,
                 learn_beta: bool = True):
        super().__init__()
        self.dim = dim
        self.num_steps = num_steps

        if learn_beta:
            self.log_beta = nn.Parameter(torch.tensor(math.log(beta_init)))
        else:
            self.register_buffer('log_beta', torch.tensor(math.log(beta_init)))

    @property
    def beta(self):
        return self.log_beta.exp()

    def energy(self, patterns: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Hopfield energy: E = -lse(β, X^T ξ) + 0.5 * ||ξ||^2
        patterns: (B, N, D) - stored patterns X
        state: (B, M, D) - query states ξ
        Returns: (B, M) energy per query
        """
        # (B, M, N)
        scores = torch.bmm(state, patterns.transpose(1, 2)) * self.beta
        lse = torch.logsumexp(scores, dim=-1)  # (B, M)
        norm = 0.5 * (state ** 2).sum(dim=-1)  # (B, M)
        return -lse + norm

    def update(self, patterns: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        One Hopfield update step: ξ_new = X · softmax(β * X^T ξ)
        """
        # (B, M, N)
        scores = torch.bmm(state, patterns.transpose(1, 2)) * self.beta
        attn = F.softmax(scores, dim=-1)
        # (B, M, D)
        new_state = torch.bmm(attn, patterns)
        return new_state

    def forward(self, patterns: torch.Tensor, state: torch.Tensor,
                return_energy: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Run T steps of Hopfield retrieval.
        patterns: (B, N, D) stored patterns (keys/values in attention terms)
        state: (B, M, D) initial query state
        """
        xi = state
        for _ in range(self.num_steps):
            xi = self.update(patterns, xi)

        energy = self.energy(patterns, xi) if return_energy else None
        return xi, energy


class HopfieldAttention(nn.Module):
    """
    Multi-head attention where each head uses multi-step Hopfield retrieval
    instead of single-step softmax attention.

    Drop-in replacement for nn.MultiheadAttention.
    """

    def __init__(self, d_model: int, num_heads: int, num_steps: int = 3,
                 beta_init: float = None, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # Standard QKV projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        if beta_init is None:
            beta_init = 1.0 / math.sqrt(self.d_head)

        # Per-head Hopfield layers
        self.hopfield = ModernHopfieldLayer(
            dim=self.d_head,
            num_steps=num_steps,
            beta_init=beta_init,
            learn_beta=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                return_energy: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        query, key, value: (B, L, D)
        mask: (B, L, S) or None — causal mask support
        """
        B, L, _ = query.shape
        S = key.shape[1]

        # Project and reshape to (B*H, L, d_head)
        q = self.W_q(query).view(B, L, self.num_heads, self.d_head).transpose(1, 2)
        k = self.W_k(key).view(B, S, self.num_heads, self.d_head).transpose(1, 2)
        v = self.W_v(value).view(B, S, self.num_heads, self.d_head).transpose(1, 2)

        q = q.reshape(B * self.num_heads, L, self.d_head)
        k = k.reshape(B * self.num_heads, S, self.d_head)
        v = v.reshape(B * self.num_heads, S, self.d_head)

        # Prepare mask: support 2D (L, S) and 3D (B, L, S)
        expanded_mask = None
        if mask is not None:
            if mask.dim() == 2:
                # (L, S) -> (B*H, L, S)
                expanded_mask = mask.unsqueeze(0).expand(B * self.num_heads, -1, -1)
            elif mask.dim() == 3:
                # (B, L, S) -> (B*H, L, S)
                expanded_mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
                expanded_mask = expanded_mask.reshape(B * self.num_heads, L, S)

        # Hopfield retrieval: use K as patterns for state update, then read from V
        xi = q
        beta = self.hopfield.beta
        for _ in range(self.hopfield.num_steps):
            scores = torch.bmm(xi, k.transpose(1, 2)) * beta
            if expanded_mask is not None:
                scores = scores.masked_fill(expanded_mask == 0, float('-inf'))
            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            xi = torch.bmm(attn, k)  # Update state using keys as patterns

        # Final readout from values using converged attention
        scores = torch.bmm(xi, k.transpose(1, 2)) * beta
        if expanded_mask is not None:
            scores = scores.masked_fill(expanded_mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.bmm(attn, v)

        # Reshape back
        out = out.view(B, self.num_heads, L, self.d_head).transpose(1, 2).reshape(B, L, self.d_model)
        out = self.W_o(out)

        # Compute energy if requested (for regularization)
        energy = None
        if return_energy:
            energy = self.hopfield.energy(k, xi)  # (B*H, L)
            energy = energy.view(B, self.num_heads, L).mean(dim=1)  # (B, L)

        return out, energy


class HopfieldMemoryBank(nn.Module):
    """
    External associative memory using Modern Hopfield Network.

    Maintains a learnable memory bank M of stored patterns.
    Input queries retrieve from this memory, providing the model
    with a persistent, content-addressable external memory.
    """

    def __init__(self, d_model: int, num_memories: int = 64,
                 num_steps: int = 3, beta_init: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.num_memories = num_memories

        # Learnable memory patterns
        self.memory = nn.Parameter(torch.randn(num_memories, d_model) * 0.02)

        # Projections
        self.query_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

        self.hopfield = ModernHopfieldLayer(
            dim=d_model, num_steps=num_steps,
            beta_init=beta_init, learn_beta=True
        )

    def forward(self, x: torch.Tensor,
                return_energy: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        x: (B, L, D)
        Returns: x + memory_retrieval, energy
        """
        B, L, D = x.shape
        query = self.query_proj(x)  # (B, L, D)

        # Expand memory for batch: (B, num_memories, D)
        patterns = self.memory.unsqueeze(0).expand(B, -1, -1)

        retrieved, energy = self.hopfield(patterns, query, return_energy=return_energy)
        out = self.out_proj(retrieved)

        return self.layer_norm(x + out), energy


# Alias for backward compatibility
AssociativeMemoryLayer = HopfieldMemoryBank
