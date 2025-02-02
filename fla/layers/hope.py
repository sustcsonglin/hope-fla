# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange
from transformers.utils import logging

from fla.modules import RotaryEmbedding
from fla.ops.hope.parallel import parallel_hope



if TYPE_CHECKING:
    from fla.models.utils import Cache

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import (index_first_axis, pad_input,
                                         unpad_input)
except ImportError:
    warnings.warn(
        "Flash Attention is not installed. Please install it via `pip install flash-attn --no-build-isolation`",
        category=ImportWarning
    )
    flash_attn_func = None

logger = logging.get_logger(__name__)


class HoPEAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        num_kv_heads: Optional[int] = None,
        window_size: Optional[int] = None,
        max_position_embeddings: Optional[int] = None,
        layer_idx: int = None,
        householder_block_size: int = 16,
    ):
        super().__init__()

        self.num_heads = num_heads
        if num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        else:
            self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.hidden_size = hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.window_size = window_size
        self.max_position_embeddings = max_position_embeddings
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=False)
        self.beta_proj = nn.Linear(self.hidden_size, self.num_heads, bias=True)
        # self.w_proj = nn.Sequential(
        #     nn.Linear(self.hidden_size, 16, bias=False),
        #     nn.Linear(16, self.hidden_size, bias=False)
        # )
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.rotary = RotaryEmbedding(self.head_dim)
        self.householder_block_size = householder_block_size
        self.apply(self._initialize_weights)

        if flash_attn_func is None:
            raise ImportError("Please install Flash Attention via `pip install flash-attn --no-build-isolation` first")


    def _initialize_weights(self, module: nn.Module):
        if getattr(module, "_is_hf_initialized", False):
            return
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        module._is_hf_initialized = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )
        # assert past_key_values is None, "HoPEAttention does not support past_key_values yet." 
        # assert attention_mask is None, "HoPEAttention requires attention_mask to be provided."
        batch_size, q_len, _ = hidden_states.size()
        q, k, v = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
        beta = self.beta_proj(hidden_states).sigmoid()

        if attention_mask is not None:
            v = (torch.mul if self.training else torch.mul_)(v, attention_mask[:, -v.shape[1]:, None])
        
        q, k, v = map(lambda x: rearrange(x, "b n (h d) -> b n h d", h=self.num_heads), (q, k, v))
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        beta = beta.transpose(1, 2)
        o = parallel_hope(q, k, v, beta)
        o = rearrange(o, "b h n d -> b n (h d)")
        o = self.o_proj(o)
        if not output_attentions:
            attentions = None
        return o, attentions, past_key_values


    def _upad_input(self, q, k, v, attention_mask, q_len):
        seqlens = attention_mask.sum(-1, dtype=torch.int32)
        indices_k = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
        max_seqlen_k = seqlens.max().item()
        cu_seqlens_k = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
        batch_size, seq_len, num_key_value_heads, head_dim = k.shape

        k = index_first_axis(k.reshape(batch_size * seq_len, num_key_value_heads, head_dim), indices_k)
        v = index_first_axis(v.reshape(batch_size * seq_len, num_key_value_heads, head_dim), indices_k)
        if q_len == seq_len:
            q = index_first_axis(q.reshape(batch_size * seq_len, self.num_heads, head_dim), indices_k)
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_q = max_seqlen_k
            indices_q = indices_k
        elif q_len == 1:
            max_seqlen_q = 1
            # There is a memcpy here, that is very bad.
            cu_seqlens_q = torch.arange(batch_size + 1, dtype=torch.int32, device=q.device)
            indices_q = cu_seqlens_q[:-1]
            q = q.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -q_len:]
            q, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(q, attention_mask)

        return q, k, v, indices_q, (cu_seqlens_q, cu_seqlens_k), (max_seqlen_q, max_seqlen_k)


if __name__ == "__main__":
    layer = HoPEAttention(hidden_size=2048, num_heads=32).cuda().to(torch.bfloat16)
    hidden_states = torch.randn(8, 1024, 2048).cuda().to(torch.bfloat16)

    o, attentions, past_key_values = layer(hidden_states)
    breakpoint()

