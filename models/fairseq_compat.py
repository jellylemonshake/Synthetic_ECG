"""Small, local compatibility layer for the Fairseq APIs used by DSAT-ECG.

DSAT only uses these model-building helpers; it does not use Fairseq training,
data loading, or checkpointing. Keeping the equivalents here makes DSAT
portable to current Python/PyTorch installations without changing its network.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default)


def get_activation_fn(activation: str):
    activations = {
        "relu": F.relu,
        "gelu": F.gelu,
        "gelu_fast": F.gelu,
        "gelu_accurate": F.gelu,
        "tanh": torch.tanh,
        "linear": lambda x: x,
        "swish": F.silu,
        "silu": F.silu,
    }
    try:
        return activations[activation]
    except KeyError as exc:
        raise RuntimeError(f"Unsupported activation function: {activation}") from exc


class FairseqDropout(nn.Dropout):
    """Drop-in dropout module; ``module_name`` is informational in Fairseq."""

    def __init__(self, p: float, module_name: Optional[str] = None):
        super().__init__(p=p)
        self.module_name = module_name


def LayerNorm(normalized_shape, export: bool = False, **kwargs):
    """Fairseq's LayerNorm delegates to PyTorch for this usage."""
    return nn.LayerNorm(normalized_shape, **kwargs)


def quant_noise(module: nn.Module, p: float, block_size: int) -> nn.Module:
    """Return the unmodified layer when quantization noise is disabled.

    DSAT's configuration uses the Transformer default of ``pq=0``. Raising a
    clear error for non-zero values avoids silently changing training behavior.
    """
    if p not in (None, 0, 0.0):
        raise NotImplementedError("DSAT's local compatibility layer supports quant_noise=0 only")
    return module


def with_incremental_state(cls):
    """Add the Fairseq incremental-state helpers used by LocalAttention."""

    def get_incremental_state(self, incremental_state: Optional[Dict], key: str):
        if incremental_state is None:
            return None
        return incremental_state.get(key)

    def set_incremental_state(self, incremental_state: Optional[Dict], key: str, value):
        if incremental_state is not None:
            incremental_state[key] = value
        return incremental_state

    cls.get_incremental_state = get_incremental_state
    cls.set_incremental_state = set_incremental_state
    return cls


@dataclass
class DecoderConfig:
    embed_dim: int = 512
    ffn_embed_dim: int = 2048
    attention_heads: int = 8
    normalize_before: bool = False


@dataclass
class EncoderConfig:
    embed_dim: int = 512


@dataclass
class QuantNoiseConfig:
    pq: float = 0.0
    pq_block_size: int = 8


@dataclass
class TransformerConfig:
    """Subset of Fairseq TransformerConfig consumed by SPADEModel."""

    activation_fn: str = "relu"
    dropout: float = 0.1
    attention_dropout: float = 0.0
    activation_dropout: float = 0.0
    relu_dropout: float = 0.0
    cross_self_attention: bool = False
    scale_attn: bool = False
    scale_heads: bool = False
    scale_fc: bool = False
    scale_resids: bool = False
    export: bool = False
    max_target_positions: int = 1024
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    quant_noise: QuantNoiseConfig = field(default_factory=QuantNoiseConfig)
