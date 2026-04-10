"""
HuggingFace Model Integration for Hopfield Attention

Provides drop-in replacements for HF attention layers:
1. HopfieldAttentionWrapper — replaces softmax attention with T-step Hopfield retrieval
2. HopfieldMemoryWrapper — adds associative memory bank after original attention
3. patch_model_attention() — utility to patch any supported HF model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Callable
from .hopfield_layers import ModernHopfieldLayer, HopfieldMemoryBank


class HopfieldAttentionWrapper(nn.Module):
    """
    Drop-in replacement for HF attention modules.
    Keeps original Q/K/V/O projections and RoPE, replaces single-step
    softmax attention with T-step Hopfield energy minimization.

    Supports GQA (grouped query attention) used by Qwen3, Llama, etc.
    """

    def __init__(self, original_attn, num_steps=3, beta_init=1.0):
        super().__init__()
        self.original = original_attn
        self.num_steps = num_steps
        self.log_beta = nn.Parameter(torch.tensor(math.log(beta_init)))

        # Copy attributes from original
        self.config = original_attn.config
        self.layer_idx = original_attn.layer_idx
        self.head_dim = original_attn.head_dim
        self.num_key_value_groups = original_attn.num_key_value_groups
        self.scaling = original_attn.scaling
        self.is_causal = original_attn.is_causal

    @property
    def beta(self):
        return self.log_beta.exp()

    def _repeat_kv(self, hidden_states, n_rep):
        """Expand KV heads for GQA: (B, num_kv_heads, L, D) -> (B, num_heads, L, D)"""
        if n_rep == 1:
            return hidden_states
        B, num_kv_heads, L, D = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(B, num_kv_heads, n_rep, L, D)
        return hidden_states.reshape(B, num_kv_heads * n_rep, L, D)

    def _hopfield_attention(self, query, key, value, attention_mask=None):
        """
        Multi-step Hopfield attention.
        Iterates in key-space to refine attention weights, then retrieves from values.
        T=1 is exactly standard attention (when beta=1).
        """
        beta = self.beta
        scale = self.scaling

        # Iterate: refine query state in key-space
        state = query  # (B, H, L_q, D)

        for t in range(self.num_steps):
            scores = torch.matmul(state, key.transpose(-2, -1)) * scale * beta

            if attention_mask is not None:
                scores = scores + attention_mask

            attn_weights = F.softmax(scores, dim=-1)

            if t < self.num_steps - 1:
                # Intermediate steps: update state in key-space
                state = torch.matmul(attn_weights, key)
            else:
                # Final step: retrieve from values
                state = torch.matmul(attn_weights, value)

        return state, attn_weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        orig = self.original
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Use original projections + norms
        query_states = orig.q_norm(orig.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = orig.k_norm(orig.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = orig.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # Apply RoPE
        cos, sin = position_embeddings
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Handle KV cache
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # Expand KV for GQA
        key_states = self._repeat_kv(key_states, self.num_key_value_groups)
        value_states = self._repeat_kv(value_states, self.num_key_value_groups)

        # Hopfield multi-step attention
        attn_output, attn_weights = self._hopfield_attention(
            query_states, key_states, value_states, attention_mask
        )

        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = orig.o_proj(attn_output)
        return attn_output, attn_weights


class HopfieldMemoryWrapper(nn.Module):
    """
    Wraps original attention + adds Hopfield associative memory bank.
    Original attention runs normally, then memory retrieval is added as residual.
    """

    def __init__(self, original_attn, d_model, num_memories=64, num_steps=3):
        super().__init__()
        self.original = original_attn
        self.memory_bank = HopfieldMemoryBank(
            d_model=d_model, num_memories=num_memories,
            num_steps=num_steps, beta_init=1.0
        )

        # Copy attributes
        self.config = original_attn.config
        self.layer_idx = original_attn.layer_idx
        self.head_dim = original_attn.head_dim
        self.num_key_value_groups = original_attn.num_key_value_groups
        self.scaling = original_attn.scaling
        self.is_causal = original_attn.is_causal

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        # Run original attention
        attn_output, attn_weights = self.original(
            hidden_states, position_embeddings, attention_mask,
            past_key_values=past_key_values, cache_position=cache_position,
            **kwargs,
        )

        # Add memory retrieval (residual)
        attn_output, _ = self.memory_bank(attn_output, return_energy=False)

        return attn_output, attn_weights


def patch_model_attention(model, mode='hopfield', num_steps=3,
                          num_memories=64, layers=None):
    """
    Patch a HuggingFace causal LM model's attention layers.

    Args:
        model: HF model (e.g., Qwen3ForCausalLM)
        mode: 'hopfield' (replace attention) or 'augmented' (add memory bank)
        num_steps: Hopfield iteration steps
        num_memories: number of memory patterns (augmented mode)
        layers: list of layer indices to patch (None = all)

    Returns:
        model with patched attention, count of patched layers
    """
    # Find the transformer layers
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        transformer_layers = model.model.layers  # Qwen3, Llama
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        transformer_layers = model.transformer.h  # GPT2
    else:
        raise ValueError(f"Unsupported model architecture: {type(model)}")

    num_layers = len(transformer_layers)
    if layers is None:
        layers = list(range(num_layers))

    patched = 0
    for idx in layers:
        layer = transformer_layers[idx]

        # Find the attention module
        if hasattr(layer, 'self_attn'):
            attn_attr = 'self_attn'  # Qwen3, Llama
        elif hasattr(layer, 'attn'):
            attn_attr = 'attn'  # GPT2
        else:
            continue

        original_attn = getattr(layer, attn_attr)
        d_model = model.config.hidden_size

        if mode == 'hopfield':
            new_attn = HopfieldAttentionWrapper(
                original_attn, num_steps=num_steps
            )
        elif mode == 'augmented':
            new_attn = HopfieldMemoryWrapper(
                original_attn, d_model=d_model,
                num_memories=num_memories, num_steps=num_steps
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        setattr(layer, attn_attr, new_attn)
        patched += 1

    return model, patched


def count_new_parameters(model):
    """Count parameters added by Hopfield patching (non-original params)."""
    new_params = 0
    for name, param in model.named_parameters():
        if 'log_beta' in name or 'memory_bank' in name:
            new_params += param.numel()
    return new_params
