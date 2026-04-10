"""Hopfield-Enhanced Transformer - Core modules."""
from .hopfield_layers import ModernHopfieldLayer, HopfieldAttention, HopfieldMemoryBank
from .model import HopfieldLM, build_model
