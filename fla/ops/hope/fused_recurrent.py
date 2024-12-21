# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang

import torch
import triton
import triton.language as tl
from fla.modules.l2norm import l2_norm
from fla.utils import contiguous


# on-the-fly computation without materializing hidden statets into HBMs
@triton.jit
def fused_recurrent_fwd_kernel(
    q,  # query [B, H, L, K]
    k,  # key [B, H, L, V]
    k_l2,  # l2 norm of key [B, H, L]
    q_reflected, 
    k_reflected,
    dk_l2_partial,
    T,  # seq_len
    D: tl.constexpr,
    BK: tl.constexpr,  # BLOCK SIZE along the K dimension
):

    # indices
    i_b, i_h = tl.program_id(0), tl.program_id(1)
    p_q = q + i_b * T * D + i_h * BK + tl.arange(0, BK) 
    p_k = k + i_b * T * D + i_h * BK + tl.arange(0, BK)
    p_k_l2 = k_l2 + i_b * T * D + i_h * BK + tl.arange(0, BK)
    p_q_reflected = q_reflected + i_b * T * D + i_h * BK + tl.arange(0, BK)
    p_k_reflected = k_reflected + i_b * T * D + i_h * BK + tl.arange(0, BK)
    p_dk_l2_partial = dk_l2_partial + i_b * T * D + i_h * BK + tl.arange(0, BK)
    h = tl.zeros([BK, BK], dtype=tl.float32) + (tl.arange(0, BK)[:, None] == tl.arange(0, BK)[None, :])

    for _ in range(0, T):
        b_k_l2 = tl.load(p_k_l2).to(tl.float32)
        b_q = tl.load(p_q).to(tl.float32)
        b_k = tl.load(p_k).to(tl.float32)

        tmp = tl.sum(h * b_k_l2[None, :], axis=1)
        h -=  2 * b_k_l2[None, :] * tmp[:, None]

        b_q = tl.sum(h * b_q[None, :], axis=1)
        b_k = tl.sum(h * b_k[None, :], axis=1)

        tl.store(p_q_reflected, b_q.to(p_q_reflected.dtype.element_ty))
        tl.store(p_k_reflected, b_k.to(p_k_reflected.dtype.element_ty))
        tl.store(p_dk_l2_partial, tmp.to(p_dk_l2_partial.dtype.element_ty))

        p_q += D 
        p_k += D 
        p_k_l2 += D
        p_q_reflected += D
        p_k_reflected += D
        p_dk_l2_partial += D 

    


# Similar to Algorithm1 of https://arxiv.org/abs/2006.16236
@triton.jit
def fused_recurrent_bwd_kernel(
    q,  # query [B, H, L, K]
    k,  # key [B, H, L, V]
    k_l2, 
    dq,  # gradient of query [NV, B, H, L, K]
    dk,  # gradient of key [NV, B, H, L, K]
    dk_l2,
    dk_l2_partial_fwd, # for computing the gradient for dk?
    dk_l2_partial_bwd, # for computing the gradient for dk?
    dq_reflected, 
    dk_reflected, 
    T,  # seq_len
    D: tl.constexpr,
    BK: tl.constexpr,  # BLOCK SIZE along the K dimension
):
    i_b, i_h = tl.program_id(0), tl.program_id(1)
    
    d_h = tl.zeros([BK, BK], dtype=tl.float32)
    offset = i_b * T * D + i_h * BK + tl.arange(0, BK) + (T - 1) * D
    p_q = q + offset
    p_k = k + offset
    p_k_l2 = k_l2 + offset

    p_dq_reflected = dq_reflected + offset
    p_dk_reflected = dk_reflected + offset
    p_dq = dq + offset
    p_dk = dk + offset
    p_dk_l2 = dk_l2 + offset
    p_dk_l2_partial_fwd = dk_l2_partial_fwd + offset
    p_dk_l2_partial_bwd = dk_l2_partial_bwd + offset

    for _ in range(T):
        b_q = tl.load(p_q).to(tl.float32) 
        b_k = tl.load(p_k).to(tl.float32)
        b_k_l2 = tl.load(p_k_l2).to(tl.float32)
        b_dq_reflected = tl.load(p_dq_reflected).to(tl.float32)
        b_dk_reflected = tl.load(p_dk_reflected).to(tl.float32)
        d_h += b_q[None, :] * b_dq_reflected[:, None] + b_k[None, :] * b_dk_reflected[:, None]

        b_dk_l2_partial_fwd = tl.load(p_dk_l2_partial_fwd).to(tl.float32)
        b_dk_l2 = -2 * tl.sum(b_dk_l2_partial_fwd[:, None] * d_h, axis=0)
        tl.store(p_dk_l2, b_dk_l2.to(p_dk_l2.dtype.element_ty))

        b_dk_l2_partial = tl.sum(d_h * b_k_l2[None,  :], axis=1)
        d_h -= 2 * b_k_l2[None, :] * b_dk_l2_partial[:, None]

        tl.store(p_dk_l2_partial_bwd, b_dk_l2_partial.to(p_dk_l2_partial_bwd.dtype.element_ty))
        p_dq_reflected -= D
        p_dk_reflected -= D
        p_q -= D
        p_k -= D
        p_k_l2 -= D
        p_dk_l2_partial_fwd -= D
        p_dk_l2_partial_bwd -= D
        p_dk_l2 -= D

    tl.debug_barrier()
    offset =  i_b * T * D + i_h * BK + tl.arange(0, BK)
    p_q = q + offset
    p_k = k + offset
    p_k_l2 = k_l2 + offset

    p_dq_reflected = dq_reflected + offset
    p_dk_reflected = dk_reflected + offset
    p_dq = dq + offset
    p_dk = dk + offset
    p_dk_l2 = dk_l2 + offset
    p_dk_l2_partial_bwd = dk_l2_partial_bwd + offset
    h = tl.zeros([BK, BK], dtype=tl.float32) + (tl.arange(0, BK)[:, None] == tl.arange(0, BK)[None, :])

    for _ in range(T):
        b_k_l2 = tl.load(p_k_l2).to(tl.float32)
        b_dk_l2_partial = tl.load(p_dk_l2_partial_bwd).to(tl.float32)
        b_dk_l2 = -2 *tl.sum(h * b_dk_l2_partial[:, None], axis=0)
        b_dk_l2 += tl.load(p_dk_l2)
        tl.store(p_dk_l2, b_dk_l2.to(p_dk_l2.dtype.element_ty))
        tmp = tl.sum(h * b_k_l2[None, :], axis=1)
        h -=  2 * b_k_l2[None, :] * tmp[:, None]

        b_dq_reflected = tl.load(p_dq_reflected).to(tl.float32)
        b_dk_reflected = tl.load(p_dk_reflected).to(tl.float32)

        b_dq = tl.sum(b_dq_reflected[:, None] * h, axis=0)
        b_dk = tl.sum(b_dk_reflected[:, None] * h, axis=0)
    
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty))
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty))

        p_q += D
        p_k += D
        p_k_l2 += D
        p_dq_reflected += D
        p_dk_reflected += D
        p_dk_l2 += D
        p_dk_l2_partial_bwd += D
        p_dq += D
        p_dk += D


class FusedRecurrentFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    def forward(ctx, q, k, k_l2, BD=16):
        B, T, D = q.shape
        H = D // BD

        grid = (B, H)
        q_reflected = torch.empty_like(q)
        k_reflected = torch.empty_like(k)
        dk_l2_partial = torch.empty_like(k_l2)

        fused_recurrent_fwd_kernel[grid](
            q, k, k_l2, q_reflected, k_reflected, dk_l2_partial,
            T=T,D=D, BK=BD,
            num_warps=1 if BD == 16 else 2
        )
        ctx.save_for_backward(q, k, k_l2, dk_l2_partial)
        ctx.grid = grid
        ctx.BD = BD 
        return q_reflected, k_reflected

    @staticmethod
    @contiguous
    def backward(ctx, dq_reflected, dk_reflected):
        q, k, k_l2, dk_l2_partial_fwd = ctx.saved_tensors
        dq, dk = torch.empty_like(q), torch.empty_like(k)
        dk_l2 = torch.empty_like(k_l2)
        dk_l2_partial_bwd = torch.empty_like(k_l2)

        grid= ctx.grid
        BD = ctx.BD 
        B, T, D = q.shape

        fused_recurrent_bwd_kernel[grid](   
            q, k, k_l2, 
            dq, dk, dk_l2,  dk_l2_partial_fwd, dk_l2_partial_bwd, dq_reflected, dk_reflected,
            T=T,D=D, BK=BD,
            num_warps=1 if BD == 16 else 2
        )
        return dq, dk, dk_l2, None


reflect_qk_recurrent = FusedRecurrentFunction.apply
from einops import rearrange
from flash_attn import flash_attn_func

### VERY IMPORTANT USE L2 norm inside!!!!!
def householder_attention(q, k, w, v, householder_block_size, head_size):
    """
    Args:
        q: [B, L, D] query
        k: [B, L, D] key
        w: [B, L, D] I-2ww^T. Unnormalized! Do not apply l2-norm before.
        v: [B, L, D] value
        householder_block_size: int
        head_size: int.
        Return: 
            o: [B, L, D] output
    """
    assert len(q.shape) == 3
    assert len(k.shape) == 3
    assert len(w.shape) == 3
    assert  head_size % householder_block_size == 0
    B, L, D = q.shape
    # Remember to apply 12 norm only in block size. 
    w = l2_norm(w.reshape(B, L, -1, householder_block_size), output_dtype=torch.float32)
    w = w.reshape(B, L, -1)
    q, k = reflect_qk_recurrent(q, k, w, householder_block_size)
    q = q.reshape(B, L, -1, head_size)
    k = k.reshape(B, L, -1, head_size)
    v = v.reshape(B, L, -1, head_size)
    o = flash_attn_func(q, k, v, causal=True)
    return o.reshape(B, L, -1)
