

# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang

import math
from typing import Tuple

import torch
import triton
import triton.language as tl
from einops import rearrange

from fla.ops.delta_rule.wy_fast import fwd_prepare_T
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, contiguous

import triton
import triton.language as tl
import torch

# todo: to make k2 already k * beta[:, None]. 
@triton.jit
def parallel_softmax_delta_rule_bwd_kernel(
    q_large,
    k_large,
    q,
    k,
    v,  
    h,
    hc,
    A,
    dA,
    do,
    dq,
    dk,
    dh,
    dv,
    D, # delta
    L, # logsumexp
    scale,
    B: tl.constexpr,
    H: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
    NT_large: tl.constexpr,
    BT_large: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    p_do = tl.make_block_ptr(do + i_bh * T * V, (T, V), (V, 1), (i_t * BT, 0), (BT, BV), (1, 0))
    # [BT, BV]
    b_do = tl.load(p_do, boundary_check=(0, 1))
    # [BT, BK]
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    sm_scale = scale * 1.44269504

    p_delta = tl.make_block_ptr(D + i_bh * T, (T, ), (1, ), (i_t * BT, ), (BT, ), (0, ))
    delta = tl.load(p_delta, boundary_check=(0, ))
    p_L = tl.make_block_ptr(L + i_bh * T, (T, ), (1, ), (i_t * BT, ), (BT, ), (0, ))
    l = tl.load(p_L, boundary_check=(0, ))
    curr_end = (tl.floor(i_t * BT / BT_large).to(tl.int32) * BT_large).to(tl.int32) 

    # p_q = tl.make_block_ptr(q_large + ((1*B*H) + i_bh) * T * K, (T, K), (K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    # b_q = tl.load(p_q, boundary_check=(0, 1))

    b_q = tl.zeros([BT, BK], dtype=tl.float32)

    for offset in range(0, curr_end, BS):
        if offset % BT_large == 0:
            idx = offset // BT_large
            p_q = tl.make_block_ptr(q_large + (((idx + 1)*B*H) + i_bh) * T * K, (T, K), (K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
            b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)

        p_k = tl.make_block_ptr(k + i_bh * T * K, (T, K), (K, 1), (offset, 0), (BS, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        p_hc = tl.make_block_ptr(hc + (i_bh * NT + tl.cdiv(offset, BS)) * K * K, (K, K), (K, 1), (0, 0), (BK, BK), (1, 0))
        b_hc = tl.load(p_hc, boundary_check=(0, 1))
        b_q2 = b_q - tl.dot(b_q.to(b_hc.dtype), b_hc)
        b_dh = -tl.dot(tl.trans(b_q2), b_dq)
        tl.atomic_add(dh + (i_bh * NT + tl.cdiv(offset, BS)) * K * K + tl.arange(0, K)[:, None] * K + tl.arange(0, K)[None, :], b_dh, sem='relaxed')

        b_A = tl.dot(b_q2.to(b_k.dtype), tl.trans(b_k))
        b_A_softmax = tl.math.exp2(b_A * sm_scale - l[:, None])
        b_dv = tl.dot(tl.trans(b_do), b_A_softmax.to(b_do.dtype))
        tl.atomic_add(dv + i_bh * T * V + (offset + tl.arange(0, BS))[None, :] * V + tl.arange(0, BK)[:, None], b_dv, sem='relaxed')

        p_v = tl.make_block_ptr(v + i_bh * T * K, (V, T), (1, V), (0, offset), (BK, BS), (0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_dp = tl.dot(b_do, b_v)
        b_dA = ((b_dp - delta[:, None]) * b_A_softmax * scale).to(b_v.dtype)
        
        p_h = tl.make_block_ptr(h + (i_bh * NT + tl.cdiv(offset, BS)) * K * K, (K, K), (1, K), (0, 0), (BK, BK), (0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))

        b_dq_new = tl.dot(b_dA.to(b_k.dtype), b_k)
        b_dk = tl.dot(tl.trans(b_dA), b_q2.to(b_v.dtype))
        tl.atomic_add(dk + i_bh * T * K + (offset + tl.arange(0, BS))[:, None] * K + tl.arange(0, BK)[None, :], b_dk, sem='relaxed')
        
        b_dq_new -= tl.dot(b_dq.to(b_h.dtype), b_h)
        b_dq += b_dq_new
    
    p_q = tl.make_block_ptr(q + i_bh * T * K, (T, K), (K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))

    for offset in range(curr_end, i_t * BT, BS):
        p_k = tl.make_block_ptr(k + i_bh * T * K, (T, K), (K, 1), (offset, 0), (BS, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_q2 = tl.zeros([BT, BK], dtype=tl.float32)
        b_q2 += b_q
        for i_t_small in range(i_t * BT - BS, offset, -BS):
            p_h = tl.make_block_ptr(h + (i_bh * NT + tl.cdiv(i_t_small, BS)) * K * K, (K, K), (K, 1), (0, 0), (BK, BK), (1, 0))
            b_h = tl.load(p_h, boundary_check=(0, 1))
            b_q2 -= tl.dot(b_q2.to(b_h.dtype), b_h)
        
        b_dh = -tl.dot(tl.trans(b_q2), b_dq)
        
        tl.atomic_add(dh + (i_bh * NT + tl.cdiv(offset, BS)) * K * K + tl.arange(0, K)[:, None] * K + tl.arange(0, K)[None, :], b_dh, sem='relaxed')
        b_A = tl.dot(b_q2.to(b_k.dtype), tl.trans(b_k))
        b_A_softmax = tl.math.exp2(b_A * sm_scale - l[:, None])
        b_dv = tl.dot(tl.trans(b_do), b_A_softmax.to(b_do.dtype))
        tl.atomic_add(dv + i_bh * T * V + (offset + tl.arange(0, BS))[None, :] * V + tl.arange(0, BK)[:, None], b_dv, sem='relaxed')
        p_v = tl.make_block_ptr(v + i_bh * T * K, (V, T), (1, V), (0, offset), (BK, BS), (0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_dp = tl.dot(b_do, b_v)
        b_dA = ((b_dp - delta[:, None]) * b_A_softmax * scale).to(b_k.dtype)
        p_h = tl.make_block_ptr(h + (i_bh * NT + tl.cdiv(offset, BS)) * K * K, (K, K), (1, K), (0, 0), (BK, BK), (0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_dq_new = tl.dot(b_dA.to(b_k.dtype), b_k)
        b_dk = tl.dot(tl.trans(b_dA), b_q2.to(b_v.dtype))
        tl.atomic_add(dk + i_bh * T * K + (offset + tl.arange(0, BS))[:, None] * K + tl.arange(0, BK)[None, :], b_dk, sem='relaxed')
        b_dq_new -= tl.dot(b_dq.to(b_h.dtype), b_h)
        b_dq += b_dq_new
    p_dq = tl.make_block_ptr(dq + i_bh * T * K, (T, K), (K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    tl.store(p_dq, b_dq.to(dq.dtype.element_ty), boundary_check=(0, 1))



@triton.jit
def _bwd_preprocess_kernel(
    o,
    do,
    D,
    T: tl.constexpr,
    BT: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    p_o = tl.make_block_ptr(o + i_bh * T * V, (T, V), (V, 1), (i_t * BT, 0), (BT, BV), (1, 0))
    p_do = tl.make_block_ptr(do + i_bh * T * V, (T, V), (V, 1), (i_t * BT, 0), (BT, BV), (1, 0))
    b_o = tl.load(p_o, boundary_check=(0, 1))
    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_delta = tl.sum(b_o * b_do, 1)
    p_delta = tl.make_block_ptr(D + i_bh * T, (T, ), (1, ), (i_t * BT, ), (BT, ), (0, ))
    tl.store(p_delta, b_delta.to(p_delta.dtype.element_ty), boundary_check=(0, ))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
    ],
    key=["BT", "K"],
)
@triton.jit
def chunk_cumprod_householder_fwd_kernel(
    h,
    hc,
    BT: tl.constexpr, # previous small chunk size
    BT_large: tl.constexpr, # larger chunk size
    NT: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    NT_small = triton.cdiv(BT_large, BT) # how many small chunks in a large chunk    
    
    p_h = tl.make_block_ptr(h + (i_bh * NT + i_t * NT_small + NT_small - 1) * K * K , (K, K), (K, 1), (0, 0), (BK, BK), (1, 0))
    b_h = tl.load(p_h, boundary_check=(0, 1)).to(tl.float32)

    for i_t_small in range(NT_small-2, -1, -1):
        p_h2 = tl.make_block_ptr(hc + (i_bh * NT + i_t * NT_small + i_t_small) * K * K , (K, K), (K, 1), (0, 0), (BK, BK), (1, 0))
        tl.store(p_h2, b_h.to(hc.dtype.element_ty), boundary_check=(0, 1))
        p_h2 = tl.make_block_ptr(h + (i_bh * NT + i_t * NT_small + i_t_small) * K * K , (K, K), (K, 1), (0, 0), (BK, BK), (1, 0))
        b_h2 = tl.load(p_h2, boundary_check=(0, 1)).to(tl.float32)
        b_h = b_h + b_h2 - tl.dot(b_h, b_h2)




@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
    ],
    key=["K"],
)
@triton.jit
def chunk_cumprod_householder_fwd2_kernel(
    h,
    h_pairwise, 
    NT: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    p_h = tl.make_block_ptr(h + (i_bh * NT + i_t) * K * K, (K, K), (K, 1), (0, 0), (BK, BK), (1, 0))
    b_h = tl.load(p_h, boundary_check=(0, 1))
    p_h_pairwise = tl.make_block_ptr(h_pairwise + (i_bh * NT * NT + i_t * NT + i_t) * K * K, (K, K), (K, 1), (0, 0), (BK, BK), (1, 0))
    tl.store(p_h_pairwise, b_h.to(h_pairwise.dtype.element_ty), boundary_check=(0, 1))
    for i_s in range(i_t-1, -1, -1):
        p_h2 = tl.make_block_ptr(h + (i_s) * K * K, (K, K), (K, 1), (0, 0), (BK, BK), (1, 0))
        b_h2 = tl.load(p_h2, boundary_check=(0, 1))
        b_h = tl.dot(b_h, b_h2)
        p_h_pairwise = tl.make_block_ptr(h_pairwise + (i_bh * NT * NT + i_t * NT + i_s) * K * K, (K, K), (K, 1), (0, 0), (BK, BK), (1, 0))
        tl.store(p_h_pairwise, b_h.to(h_pairwise.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
    ],
    key=["BT", "K", "V"],
)
@triton.heuristics({
    'NT': lambda args: triton.cdiv(args['T'], args['BT'])
})
@triton.jit
def chunk_transform_qk_fwd_kernel(
    q,
    k,
    v,
    w,
    beta,
    o,
    A,
    L,
    M,  
    h,
    q_new,
    k_new,
    A_local,
    scale,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BT: tl.constexpr,
    OUTPUT_ATTENTIONS: tl.constexpr,
    NT: tl.constexpr,
    # SAVE_ATTENTION: tl.constexpr
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    sm_scale = scale * 1.44269504
    p_q = tl.make_block_ptr(q + i_bh * T * K, (T, K), (K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * T * K, (K, T), (1, K), (0, i_t * BT), (BK, BT), (0, 1))
    p_w = tl.make_block_ptr(w + i_bh * T * K, (T, K), (K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * T * V, (T, V), (V, 1), (i_t * BT, 0), (BT, BV), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)
    b_kt = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
    b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)
    b_w = tl.load(p_w, boundary_check=(0, 1)).to(tl.float32)
    p_T = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_T = tl.load(p_T, boundary_check=(0, 1))

    o_i = tl.arange(0, BT)
    m_t = o_i[:, None] >= o_i[None, :]
    # b_T = tl.where(m_t, b_T, 0)
  
    p_beta = tl.make_block_ptr(beta + i_bh * T, (T, ), (1, ), (i_t * BT, ), (BT, ), (0, ))
    b_beta = tl.load(p_beta, boundary_check=(0, )).to(tl.float32)
    b_w_beta = (b_w * b_beta[:, None])
    
    b_Twb = tl.dot(b_T, b_w_beta)
    b_h = tl.dot(tl.trans(b_w), b_Twb)
    p_h = tl.make_block_ptr(h + (i_bh * NT + i_t) * K * K, (K, K), (K, 1), (0, 0), (BK, BK), (1, 0))
    tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))

    b_qw = tl.where(m_t, tl.dot(b_q, tl.trans(b_w)), 0)
    # crucial to disallow tf32 heree.
    b_qwT = tl.dot(b_qw, b_T, allow_tf32=False)
    b_wbk = tl.where(o_i[:, None] > o_i[None, :], tl.dot(b_w_beta, b_kt), 0)
    b_A = tl.where(m_t, tl.dot(b_q, b_kt) - tl.dot(b_qwT, b_wbk, allow_tf32=False), 0)
    
    b_q = b_q - tl.dot(b_qwT, b_w_beta)
    p_q_new = tl.make_block_ptr(q_new + i_bh * T * K, (T, K), (K, 1), (i_t * BT, 0), (BT, K), (1, 0))
    tl.store(p_q_new, b_q.to(p_q_new.dtype.element_ty), boundary_check=(0, 1))
    b_T_wbk = tl.dot(b_T, b_wbk)
    b_kt = b_kt - tl.dot(tl.trans(b_w), b_T_wbk)
    p_k_new = tl.make_block_ptr(k_new + i_bh * T * K, (K, T), (1, K), (0, i_t * BT), (BK, BT), (0, 1))
    tl.store(p_k_new, b_kt.to(p_k_new.dtype.element_ty), boundary_check=(0, 1))

    if OUTPUT_ATTENTIONS:
        p_a = tl.make_block_ptr(A_local + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
        tl.store(p_a, (b_A * scale).to(p_a.dtype.element_ty), boundary_check=(0, 1))

    b_qkT_softmax = tl.where(o_i[:, None] >= o_i[None, :], b_A * sm_scale, float("-inf"))
    m_i = tl.max(b_qkT_softmax, 1)
    b_qkT_softmax = tl.math.exp2(b_qkT_softmax - m_i[:, None])
    l_i = tl.sum(b_qkT_softmax, 1)
    b_o = tl.dot(b_qkT_softmax.to(b_v.dtype), b_v)
    p_o = tl.make_block_ptr(o + i_bh * T * V, (T, V), (V, 1), (i_t * BT, 0), (BT, BV), (1, 0))
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    p_l = L + i_bh * T + i_t * BT + tl.arange(0, BT)
    p_m = M + i_bh * T + i_t * BT + tl.arange(0, BT)
    mask = i_t * BT + tl.arange(0, BT) < T
    tl.store(p_m, m_i.to(p_m.dtype.element_ty), mask=mask)
    tl.store(p_l, l_i.to(p_l.dtype.element_ty), mask=mask)


@triton.autotune(
    configs=[
        # triton.Config({}, num_warps=1),
        # trito、
        triton.Config({}, num_warps=4),
    ],

    key=["BT", "K"],
)
@triton.jit
def chunk_transform_qk_bwd_kernel(
    q,
    k,
    v,
    beta,
    A,
    AT,
    L,
    D,
    do,
    dv,
    dq,
    dk,
    dk2,
    dbeta,
    scale,
    H: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    sm_scale = scale * 1.44269504

    m_t = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]

    p_q = tl.make_block_ptr(q + i_bh * T * K, (T, K), (K, 1), (i_t * BT, 0), (BT, K), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * T * K, (T, K), (K, 1), (i_t * BT, 0), (BT, K), (1, 0))
    p_dq = tl.make_block_ptr(dq + i_bh * T * K, (T, K), (K, 1), (i_t * BT, 0), (BT, K), (1, 0))
    p_dk = tl.make_block_ptr(dk + i_bh * T * K, (T, K), (K, 1), (i_t * BT, 0), (BT, K), (1, 0))
    p_beta = tl.make_block_ptr(beta + i_bh * T, (T, ), (1, ), (i_t * BT, ), (BT, ), (0, ))
    b_beta = tl.load(p_beta, boundary_check=(0, ))

    p_A_local = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_A_local = tl.load(p_A_local, boundary_check=(0, 1))
    p_L = tl.make_block_ptr(L + i_bh * T, (T, ), (1, ), (i_t * BT, ), (BT, ), (0, ))
    b_L = tl.load(p_L, boundary_check=(0, ))
    p_D = tl.make_block_ptr(D + i_bh * T, (T, ), (1, ), (i_t * BT, ), (BT, ), (0, ))
    b_D = tl.load(p_D, boundary_check=(0, ))

    p_do = tl.make_block_ptr(do + i_bh * T * V, (T, V), (V, 1), (i_t * BT, 0), (BT, BV), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * T * V, (V, T), (1, V), (0, i_t * BT), (BV, BT), (0, 1))
    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_dqkT = tl.dot(b_do, b_v)

    b_A_softmax = tl.math.exp2(tl.where(m_t, b_A_local * sm_scale - b_L[:, None], float("-inf")))
    b_dv = tl.dot(tl.trans(b_A_softmax.to(b_do.dtype)), b_do)
    p_dv = tl.make_block_ptr(dv + i_bh * T * V, (T, V), (V, 1), (i_t * BT, 0), (BT, BV), (1, 0))
    b_dv += tl.load(p_dv, boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_dq = tl.load(p_dq, boundary_check=(0, 1))
    b_dk = tl.load(p_dk, boundary_check=(0, 1))

    b_k_beta = (b_k * b_beta[:, None]).to(b_k.dtype)
    b_dqkT = (b_dqkT - b_D[:, None]) * b_A_softmax * scale
    b_dqkT -= tl.dot(b_dq.to(b_k.dtype), tl.trans(b_k_beta))

    b_dkkT = -tl.dot(b_k_beta, tl.trans(b_dk).to(b_k.dtype))
    b_qk = tl.where(m_t, tl.dot(b_q, tl.trans(b_k)), 0)

    p_T = tl.make_block_ptr(AT + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_T = tl.load(p_T, boundary_check=(0, 1))
    # [m, n] x [n, k] = [m, k]
    # b_dT = tl.dot(tl.trans(b_qk), b_dqkT)
    # backward. [m, n] = [m, k] x [k, n]
    b_dqk = tl.dot(b_dqkT, tl.trans(b_T))
    # qkT[m, k] x T[k, n] = dqkT[m, n]
    b_dkk = tl.dot(b_dkkT, tl.trans(b_T))

    b_kk = tl.where(tl.arange(0, BT)[:, None] > tl.arange(0, BT)[None, :], tl.dot(b_k, tl.trans(b_k)), 0)

    b_dT = tl.dot(tl.trans(b_qk), b_dqkT) + tl.dot(tl.trans(b_kk), b_dkkT)
    b_dT = tl.where(tl.arange(0, BT)[:, None] > tl.arange(0, BT)[None, :], b_dT, 0)
    b_dT = tl.dot(b_dT, tl.trans(b_T))
    b_dT = tl.dot(tl.trans(b_T), b_dT)
    b_dT = tl.where(tl.arange(0, BT)[:, None] > tl.arange(0, BT)[None, :], -b_dT, 0)

    # b_dq, b_dk = tl.dot(b_T.to(b_dq.dtype), b_dq), tl.dot(b_T.to(b_dk.dtype), b_dk)
    # qk[m, n] x kbeta[n, k] = q[m, k]
    # [m, k] x [n, k].transpose = [m, n]
    # [m, k] = [m, n] x [n,]

    b_dq_final = tl.zeros([BT, K], dtype=tl.float32)
    b_dk_final = tl.zeros([BT, K], dtype=tl.float32)
    b_dbeta_final = tl.zeros([BT], dtype=tl.float32)
    b_dk_final += tl.dot(tl.trans(b_dT).to(b_k.dtype), b_k_beta)
    b_qkT = tl.dot(b_qk, b_T)
    b_kkT = tl.dot(b_kk, b_T)
    b_dkbeta = tl.dot(b_dT.to(b_k.dtype), b_k) - tl.dot(tl.trans(b_qkT.to(b_dq.dtype)), b_dq) - tl.dot(b_kkT.to(b_dk.dtype), b_dk)
    b_dqk = tl.where(m_t, b_dqk, 0)
    b_dkk = tl.where(tl.arange(0, BT)[:, None] > tl.arange(0, BT)[None, :], b_dkk, 0)
    b_dq_final += tl.dot(b_dqk.to(b_k.dtype), b_k)
    b_dk_final += tl.dot(tl.trans(b_dqk).to(b_q.dtype), b_q)
    b_dk_final += tl.dot(b_dkk.to(b_k.dtype), b_k)
    b_dk_final += tl.dot(tl.trans(b_dkk).to(b_k.dtype), b_k)
    p_dk2 = tl.make_block_ptr(dk2 + i_bh * T * K, (T, K), (K, 1), (i_t * BT, 0), (BT, K), (1, 0))
    b_dkbeta += tl.load(p_dk2, boundary_check=(0, 1))
    b_dk_final += b_dkbeta * b_beta[:, None]
    b_dbeta_final += tl.sum(b_dkbeta * b_k, 1)
    p_dq = tl.make_block_ptr(dq + i_bh * T * K, (T, K), (K, 1), (i_t * BT, 0), (BT, K), (1, 0))
    b_dq_final += tl.load(p_dq, boundary_check=(0, 1))
    tl.store(p_dq, b_dq_final.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    p_dk = tl.make_block_ptr(dk + i_bh * T * K, (T, K), (K, 1), (i_t * BT, 0), (BT, K), (1, 0))
    b_dk_final += tl.load(p_dk, boundary_check=(0, 1))
    tl.store(p_dk, b_dk_final.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    p_dbeta = tl.make_block_ptr(dbeta + i_bh * T, (T, ), (1, ), (i_t * BT, ), (BT, ), (0, ))
    tl.store(p_dbeta, b_dbeta_final.to(p_dbeta.dtype.element_ty), boundary_check=(0, ))



def chunk_transform_qk_fwd_fn(q, k, v, w, beta, A, scale, BT, output_attentions):
    B, H, T, K = k.shape
    q_new = torch.empty_like(q)
    k_new = torch.empty_like(k)
    o = torch.empty_like(v)
    NT = triton.cdiv(T, BT)
    grid = (NT, B*H)
    V = v.shape[-1]
    A_local = torch.zeros_like(A).fill_(float("-inf")) if output_attentions else None
    L = torch.empty(B, H, T, dtype=torch.float32, device=q.device)
    M = torch.empty(B, H, T, dtype=torch.float32, device=q.device)
    h = torch.zeros(B, H, NT, K, K, dtype=torch.float32, device=q.device)
    chunk_transform_qk_fwd_kernel[grid](
    q=q,
    k=k,
    v=v,
    w=w,
    beta=beta,
    o=o,
    A=A,
    L=L,
    M=M,  
    h=h,
    q_new=q_new,
    k_new=k_new,
    A_local=A_local,
    scale=scale,
    T=T,
    K=K,
    V=V,
    BK=triton.next_power_of_2(K),
    BV=triton.next_power_of_2(V),
    BT=BT,
    OUTPUT_ATTENTIONS=output_attentions,
    )

    # q, k, v, w, beta = map(lambda x: x.to(torch.float32), [q, k, v, w, beta])
    # w_beta = w * beta[..., None]
    # q, k, v, w, w_beta = map(lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=BT), [q, k, v, w, w_beta])
    # mask = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=0)
    # T = -(w_beta @ w.transpose(-1, -2)).masked_fill(mask, 0)
    # for i in range(1, BT):
    #     T[..., i, :i] = T[..., i, :i].clone() + (T[..., i, :, None].clone() * T[..., :, :i].clone()).sum(-2)
    # T = T + torch.eye(BT, dtype=q.dtype, device=q.device)

    # # A_local2 = (w_beta @ k.transpose(-1, -2)).masked_fill(mask, 0)
    # #  - (q @ w.transpose(-1, -2)).tril() @ T @ (w_beta @ k.transpose(-1, -2)).masked_fill(mask, 0)
    # A_local2 = (q @ k.transpose(-1, -2)).tril() - (q @ w.transpose(-1, -2)).tril() @ T @ (w_beta @ k.transpose(-1, -2)).masked_fill(mask, 0)
    # print( get_err_ratio(A_local.reshape_as(A_local2)[..., 32:64,0:32], A_local2[...,32:64,:32]))
    # breakpoint()    
    return q_new, k_new, h, o, A_local, L, M, None, None


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
    ],
    key=["BT"],
)
@triton.jit
def save_intra_chunk_attn(
    A,
    A_local,
    T: tl.constexpr,
    BT: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    p_A = tl.make_block_ptr(A + i_bh * T * T, (T, T), (T, 1), (i_t * BT, i_t * BT), (BT, BT), (1, 0))
    p_A_local = tl.make_block_ptr(A_local + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_A_local = tl.load(p_A_local, boundary_check=(0, 1))
    tl.store(p_A, b_A_local.to(p_A.dtype.element_ty), boundary_check=(0, 1))



@triton.heuristics({
    'OUTPUT_ATTENTIONS': lambda args: args['attn'] is not None
})
@triton.jit
def parallel_delta_rule_fwd_kernel(
    q,
    k,
    v,
    o,
    o_new,
    h,
    attn,
    scale,
    L,
    L_new,
    M,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
    OUTPUT_ATTENTIONS: tl.constexpr
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    p_q = tl.make_block_ptr(q + i_bh * T * K, (T, K), (K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    sm_scale = scale * 1.44269504

    # the Q block is kept in the shared memory throughout the whole kernel
    # [BT, BK]
    b_q = tl.zeros([BT, BK], dtype=tl.float32)
    b_q += tl.load(p_q, boundary_check=(0, 1))

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    p_o = tl.make_block_ptr(o + i_bh * T * V, (T, V), (V, 1), (i_t * BT, 0), (BT, BV), (1, 0))
    b_o += tl.load(p_o, boundary_check=(0, 1))

    p_L = L + i_bh * T + i_t * BT + tl.arange(0, BT)
    p_M = M + i_bh * T + i_t * BT + tl.arange(0, BT)
    mask = i_t * BT + tl.arange(0, BT) < T
    b_l = tl.zeros([BT], dtype=tl.float32)
    b_m = tl.zeros([BT], dtype=tl.float32)
    b_l += tl.load(p_L, mask=mask)
    b_m += tl.load(p_M, mask=mask)
    # As opposed to Flashattention, this kernel requires scanning the KV blocks from right to left
    # Q block and K block have overlap.
    # masks required
    for offset in range((i_t + 1) * BT - 2 * BS, -BS, -BS):
        p_k = tl.make_block_ptr(k + i_bh * T * K, (K, T), (1, K), (0, offset), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v + i_bh * T * V, (T, V), (V, 1), (offset, 0), (BS, BV), (1, 0))
        # 要相反。
        p_h = tl.make_block_ptr(h + (i_bh * NT + tl.cdiv(offset, BS)) * K * K, (K, K), (K, 1), (0, 0), (BK, BK), (1, 0))
        # [BK, BS]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BS, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BK, BK]
        b_h = tl.load(p_h, boundary_check=(0, 1))
        # [BT, BS]
        m_s = tl.arange(0, BT) >= (offset - i_t*BT + BS)
        b_s = tl.dot(b_q.to(b_k.dtype), b_k)
        b_q_minus = tl.dot(b_q.to(b_h.dtype), b_h)
        b_q = tl.where(m_s[:, None], b_q - b_q_minus, b_q)
        
        if OUTPUT_ATTENTIONS:
            p_a = tl.make_block_ptr(attn + i_bh * T * T, (T, T), (T, 1), (i_t * BT, offset), (BT, BS), (1, 0))
            tl.store(p_a, (b_s * scale).to(p_a.dtype.element_ty), boundary_check=(0, 1))

        b_s = tl.where(m_s[:, None], b_s * sm_scale, float("-inf"))
        b_m_new = tl.maximum(b_m, tl.max(b_s, 1))
        alpha = tl.math.exp2((b_m - b_m_new))
        b_s = tl.math.exp2(b_s - b_m_new[:, None])
        b_s = tl.where(m_s[:, None], b_s, 0)
        if i_t >= 0:
            b_o *= alpha[:, None]
            b_l = b_l * alpha + tl.sum(b_s, 1)
            b_m = b_m_new
        b_o += tl.dot(b_s.to(b_v.dtype), b_v)

    b_o = b_o / b_l[:, None]
    p_o_new = tl.make_block_ptr(o_new + i_bh * T * V, (T, V), (V, 1), (i_t*BT, 0), (BT, BV), (1, 0))
    tl.store(p_o_new, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    
    b_l = tl.math.log2(b_l) + b_m
    p_L_new = L_new + i_bh * T + i_t * BT + tl.arange(0, BT)
    mask = i_t * BT + tl.arange(0, BT) < T
    tl.store(p_L_new, b_l.to(p_L_new.dtype.element_ty), mask=mask)


@triton.jit
def parallel_delta_rule_bwd_prepare_kernel(
    q,
    k,
    q_large,
    k_large,
    h,  
    attn,
    T: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    NT: tl.constexpr,
    BT_large: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    p_q = tl.make_block_ptr(q + i_bh * T * K, (T, K), (K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    # the Q block is kept in the shared memory throughout the whole kernel
    # [BT, BK]
    b_q = tl.zeros([BT, BK], dtype=tl.float32)
    b_q += tl.load(p_q, boundary_check=(0, 1))
    curr_end = (tl.floor(i_t * BT / BT_large).to(tl.int32) * BT_large).to(tl.int32)
    for offset in range((i_t + 1) * BT - 2 * BS, curr_end-BS, -BS):
        # p_k = tl.make_block_ptr(k + i_bh * T * K, (K, T), (1, K), (0, offset), (BK, BS), (0, 1))
        p_h = tl.make_block_ptr(h + (i_bh * NT + tl.cdiv(offset, BS)) * K * K, (K, K), (K, 1), (0, 0), (BK, BK), (1, 0))
        # [BK, BS]
        # b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        # [BS]
        # [BT, BS]
        # m_s = tl.arange(0, BT) >= (offset - i_t*BT + BS)
        # b_s = tl.dot(b_q.to(b_k.dtype), b_k)
        # b_s = tl.where(m_s[:, None], b_s, 0)
        # if OUTPUT_ATTENTIONS:
        # p_a = tl.make_block_ptr(attn + i_bh * T * BT_large, (T, BT_large), (BT_large, 1), (i_t * BT, offset - curr_end), (BT, BS), (1, 0))
        # tl.store(p_a, b_s.to(p_a.dtype.element_ty), boundary_check=(0, 1))
        b_q_minus = tl.dot(b_q.to(b_h.dtype), b_h)
        b_q -= b_q_minus

    p_q_last = tl.make_block_ptr(q_large + (tl.cdiv(curr_end, BT_large) * B * H + i_bh) * T * K, (T, K), (K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    tl.store(p_q_last, b_q.to(p_q_last.dtype.element_ty), boundary_check=(0, 1))
    
    for offset in range(curr_end-BS, BT_large-BS, -BS):
        p_h = tl.make_block_ptr(h + (i_bh * NT + tl.cdiv(offset, BS)) * K * K, (K, K), (K, 1), (0, 0), (BK, BK), (1, 0))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_q_minus = tl.dot(b_q.to(b_h.dtype), b_h)
        b_q = b_q - b_q_minus
        p_q_last = tl.make_block_ptr(q_large + ((tl.cdiv(offset, BT_large)) * B * H + i_bh) * T * K, (T, K), (K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
        if offset % BT_large == 0:
            tl.store(p_q_last, b_q.to(p_q_last.dtype.element_ty), boundary_check=(0, 1))

    # b_k2 = tl.zeros([BT, BK], dtype=tl.float32)
    # p_k2 = tl.make_block_ptr(k + i_bh * T * K, (T, K), (K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    # b_k2 += tl.load(p_k2, boundary_check=(0, 1))
    # curr_end = ((tl.floor(i_t * BT / BT_large) + 1) * BT_large).to(tl.int32)
    # for offset in range(i_t * BT + BS, curr_end, BS):
    #     p_h = tl.make_block_ptr(h + (i_bh * NT + tl.cdiv(offset, BS)) * K * K, (K, K), (K, 1), (0, 0), (BK, BK), (1, 0))
    #     b_h = tl.load(p_h, boundary_check=(0, 1))
    #     b_k2_minus = tl.dot(b_k2.to(b_h.dtype), tl.trans(b_h))
    #     b_k2 = b_k2 - b_k2_minus
    # p_k_large = tl.make_block_ptr(k_large + i_bh * T * K, (T, K), (K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    # tl.store(p_k_large, b_k2.to(p_k_large.dtype.element_ty), boundary_check=(0, 1))

# @triton.jit
# def parallel_delta_rule_bwd_prepare_k_kernel(
#     k,
#     k_large,
#     h,  
#     attn,
#     T: tl.constexpr,
#     K: tl.constexpr,
#     BT: tl.constexpr,
#     BS: tl.constexpr,
#     BK: tl.constexpr,
#     NT: tl.constexpr,
#     BT_large: tl.constexpr,
#     B: tl.constexpr,
#     H: tl.constexpr,
# ):
#     i_t, i_bh = tl.program_id(0), tl.program_id(1)
#     p_k = tl.make_block_ptr(q + i_bh * T * K, (T, K), (K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
#     # the Q block is kept in the shared memory throughout the whole kernel
#     # [BT, BK]
#     b_k = tl.zeros([BT, BK], dtype=tl.float32)
#     b_k += tl.load(p_k, boundary_check=(0, 1))
#     curr_end = (tl.cdiv(i_t * BT, BT_large) * BT_large).to(tl.int32) 
#     for offset in range((i_t + 1) * BT - 2 * BS, curr_end, BS):
#         p_h = tl.make_block_ptr(h + (i_bh * NT + tl.cdiv(offset, BS)) * K * K, (K, K), (K, 1), (0, 0), (BK, BK), (1, 0))
#         # [BK, BS]
#         b_h = tl.load(p_h, boundary_check=(0, 1))
#         # [BS]
#         # [BT, BS]
#         m_s = tl.arange(0, BT) >= (offset - i_t*BT + BS)
    
#     idx = (offset + BS) // BT_large
#     p_q_last = tl.make_block_ptr(q_large + (idx * B * H + i_bh) * T * K, (T, K), (K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
#     tl.store(p_q_last, b_q.to(p_q_last.dtype.element_ty), boundary_check=(0, 1))
        
#     for offset in range(curr_end, BT_large-BS, -BS):
#         # [BS]
#         # [BT, BS]
#         m_s = tl.arange(0, BT) >= (offset - i_t*BT + BS)
#         p_h = tl.make_block_ptr(h + (i_bh * NT + tl.cdiv(offset, BS)) * K * K, (K, K), (K, 1), (0, 0), (BK, BK), (1, 0))
#         b_h = tl.load(p_h, boundary_check=(0, 1))
#         b_q_minus = tl.dot(b_q.to(b_h.dtype), b_h)
#         b_q = tl.where(m_s[:, None], b_q - b_q_minus, b_q)
#         if offset % BT_large == 0:
#             idx = offset // BT_large
#             if idx > 0:
#                 p_q_last = tl.make_block_ptr(q_large + (idx * B * H + i_bh) * T * K, (T, K), (K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
#                 tl.store(p_q_last, b_q.to(p_q_last.dtype.element_ty), boundary_check=(0, 1))



@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
    ],
    key=["BT"],
)
@triton.jit
def save_intra_chunk_attn(
    A,
    A_local,
    T: tl.constexpr,
    BT: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    p_A = tl.make_block_ptr(A + i_bh * T * T, (T, T), (T, 1), (i_t * BT, i_t * BT), (BT, BT), (1, 0))
    p_A_local = tl.make_block_ptr(A_local + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_A_local = tl.load(p_A_local, boundary_check=(0, 1))
    tl.store(p_A, b_A_local.to(p_A.dtype.element_ty), boundary_check=(0, 1))



class ParallelDeltaRuleFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx, q, k, v, w, beta, scale):
        B, H, T, K, V = *k.shape, v.shape[-1]
        assert T % 64 == 0, 'The sequence length must be divisible by 64.'
        assert q.shape[-1] <= 128, 'The maximum supported sequence length is 128.'
        BT, BS = 128, 64
        BK = triton.next_power_of_2(k.shape[-1])
        BV = triton.next_power_of_2(v.shape[-1])
        assert BT % BS == 0

        A = fwd_prepare_T(w, beta, None,  None, True, BS)
        q_new, k_new, h, o, _, L, M, _, _ = chunk_transform_qk_fwd_fn(q, k, v, w, beta, A, scale, BS, False)
        num_stages = 3 if K <= 64 else 2
        num_warps = 4
        grid = (triton.cdiv(T, BT), B * H)
        o_new = torch.empty_like(o)
        L_new = torch.empty_like(L)
        # attn = torch.zeros(B, H, T, T, device=q.device)
        parallel_delta_rule_fwd_kernel[grid](
            q=q_new,
            k=k_new,
            v=v,
            o=o,
            o_new=o_new,
            h=h,
            attn=None,
            scale=scale,
            L=L,
            L_new=L_new,
            M=M,
            T=T,
            K=K,
            V=V,
            BT=BT,
            BS=BS,
            BK=BK,
            BV=BV,
            NT=triton.cdiv(T, BS),
            num_stages=num_stages,
            num_warps=num_warps
        )
        grid = (triton.cdiv(T, BS), B * H)
        # save_intra_chunk_attn[grid](
                # A=attn, A_local=A_local, T=T, BT=BS
        # )
        ctx.save_for_backward(q, k, v, w, o_new, beta, L_new)
        ctx.BT = BT
        ctx.BS = BS
        ctx.scale = scale
        return o_new


    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(ctx, do):
        q, k, v, w, o, beta, L_new = ctx.saved_tensors
        B, H, T, K, V = *k.shape, v.shape[-1]
        BT_large = T
        assert q.shape[-1] <= 128, 'The maximum supported sequence length is 128.'
        BT, BS = 64, 64
        BK = triton.next_power_of_2(k.shape[-1])
        BV = triton.next_power_of_2(v.shape[-1])
        scale = ctx.scale
        assert BT % BS == 0
        delta = torch.empty_like(L_new)
        grid = (triton.cdiv(T, BT), H*B)
        _bwd_preprocess_kernel[grid](
            o, do,
            delta,
            T, BT, V, BV
        )

        A = fwd_prepare_T(w, beta, None,  None, True, BS)
        q_new, k_new, h, _, A_local, _, _, _, _ = chunk_transform_qk_fwd_fn(q, k, v, w, beta, A, scale, BS, True)
        B, H, T, K = q.shape

        # should be something in this form here.
        grid = (triton.cdiv(T, BT), B * H)

        BT_large = 256
        NT_large = triton.cdiv(T, BT_large)

        q_large_new = torch.zeros(NT_large + 1, B, H, T, K, dtype=q.dtype, device=q.device)
        k_large_new = None
        A_qk = None

        parallel_delta_rule_bwd_prepare_kernel[grid](
            q=q_new,
            k=k_new,
            q_large=q_large_new,
            k_large=k_large_new,
            h=h,
            attn=A_qk,
            T=T,
            K=K,
            BT=BT,
            BS=BS,
            BK=BK,
            B=B,
            H=H,
            NT=triton.cdiv(T, BS),
            BT_large=BT_large
        )

        # debug = False

        # if debug:
        #     A_qk_ref = torch.zeros_like(A_qk)
        #     q_large_new_ref = torch.zeros_like(q_large_new)
        #     k_large_new_ref = torch.zeros_like(k_large_new)
        #     for i_start in range(0, T, BT):
        #         curr_end = math.floor((i_start) / BT_large) * BT_large
        #         q_i = q_new[:, :, i_start:i_start+BT].clone().float()
        #         q_i_shape = q_i.shape
        #         for j_start in range(i_start - BS, curr_end - BS, -BS):
        #             k_j = k_new[:, :, j_start:j_start+BS].clone()
        #             A_qk_ref[:, :, i_start:i_start+BT, j_start-curr_end:j_start+BS-curr_end] = q_i.to(k_j) @ k_j.transpose(-1, -2)
        #             h_idx = j_start // BS
        #             q_i = q_i - q_i.to(h) @ h[:, :, h_idx]
        #             assert q_i.shape == q_i_shape, breakpoint()

        #         q_large_new_ref[curr_end // BT_large, :, :, i_start:i_start+BT] = q_i
        #         for j_start in range(curr_end - BS,  BT_large-BS, -BS):
        #             h_idx = j_start // BS
        #             q_i = q_i - q_i.to(h) @ h[:, :, h_idx]
        #             if j_start % BT_large == 0:
        #                 q_large_new_ref[j_start // BT_large, :, :, i_start:i_start+BT] = q_i
        
        #     for i_start in range(0, T, BT):
        #         curr_end = (math.floor((i_start) / BT_large) + 1) * BT_large
        #         k_i = k_new[:, :, i_start:i_start+BT].clone().float()
        #         for j_start in range(i_start + BS, curr_end, BS):
        #             h_idx = j_start // BS
        #             k_i = k_i - k_i.to(h) @ h[:, :, h_idx].transpose(-1, -2)
        #         k_large_new_ref[:, :, i_start:i_start+BT] = k_i
            
        #     # (k_large_new-k_large_new_ref)[:,:,0:256]
        #     # get_err_ratio(k_large_new[:,:,:256], k_large_new_ref[:, :, :256])
        #     a1 = q_large_new[1,:,:, 256:512] @ k_large_new_ref[:,:,:256].transpose(-1, -2)
        #     a2 = q_large_new[1,:, :, 256:512] @ k_large_new[:,:,:256].transpose(-1, -2)
        #     # a1 = a1 * scale
        #     # a2 = a2 * scale
        #     a3 = attn_logit[:,:,256:512, 0:64]
        #     # breakpoint()
        
        grid = (triton.cdiv(T, BT_large) - 1, B*H)
        hc = torch.zeros_like(h)

        chunk_cumprod_householder_fwd_kernel[grid](
            h=h,
            hc=hc,
            BT=BT,
            BT_large=BT_large,
            K=K,
            NT=triton.cdiv(T, BS),
            BK=triton.next_power_of_2(K),
        )

        dq = torch.empty_like(q)
        dk = torch.zeros_like(k, dtype=torch.float32)
        dv = torch.zeros_like(v, dtype=torch.float32)
        dh = torch.zeros_like(h, dtype=torch.float32)
        grid = (triton.cdiv(T, BT), H*B)
        # dA = torch.empty_like(A_qk, dtype=q.dtype)

        parallel_softmax_delta_rule_bwd_kernel[grid](
            q_large=q_large_new,
            k_large=k_large_new,
            k=k_new,
            q=q_new,
            v=v,
            h=h,
            hc=hc,
            A=A_qk,
            dA=None,
            do=do,
            D=delta,
            L=L_new,
            dq=dq,
            dk=dk,
            dh=dh,
            dv=dv,
            scale=scale,
            B=B,
            H=H,
            T=T,
            K=K,
            V=V,
            BT=BT,
            BS=BS,
            BK=BK,
            BV=BV,
            BT_large=BT_large,
            NT=triton.cdiv(T, BS),
            NT_large=triton.cdiv(T, BT_large)
        )

        # breakpoint()
        # compute the gradient

        A, q, k, v, w, dq_new, dk_new, dv, A_local, do = map(lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=BS), [A, q, k, v, w, dq, dk, dv, A_local, do])
        beta, L_new, delta = map(lambda x: rearrange(x, 'b h (n c) -> b h n c', c=BS), [beta, L_new, delta])
        w_beta = w * beta[..., None]
        A_local_softmax = (torch.exp2(A_local * 1.44269504 - L_new[..., None])).tril()
        dv += A_local_softmax.transpose(-1, -2).to(do) @ do
        dA = do @ v.transpose(-1, -2)
        dA = ((dA - delta[..., None]) * A_local_softmax * scale).tril()

        # 确保输入tensor有requires_grad
        with torch.enable_grad():
            q = q.detach().requires_grad_(True)
            k = k.detach().requires_grad_(True) 
            w = w.detach().requires_grad_(True)
            w_beta = w_beta.detach().requires_grad_(True)
            A = A.detach().to(q).requires_grad_(True)

            mask = torch.triu(torch.ones(BS, BS, dtype=torch.bool, device=q.device), diagonal=0)
            # Scale q directly instead of creating a new tensor

            # Calculate intermediate values
            Twbk = A @ (w_beta @ k.transpose(-1, -2)).masked_fill(mask, 0)
            qw = (q @ w.transpose(-1, -2)).tril() 
            Twb = A @ w_beta

            # Calculate final values while maintaining gradient chain
            A_local = (q @ k.transpose(-1, -2)).tril() - qw @ Twbk
            q_new = q - qw @ Twb
            k_new = k - Twbk.transpose(-1, -2) @ w
            h = w.transpose(-1, -2) @ Twb
            # Backward passes
            ((q_new * dq_new).sum() + (k_new * dk_new).sum() + (h * dh).sum() + (A_local * dA).sum()).backward()
            dq, dk, dw, dw_beta, dA = q.grad, k.grad, w.grad, w_beta.grad, A.grad
        
        dA = -A.transpose(-1, -2) @ dA @ A.transpose(-1, -2)
        mask = torch.triu(torch.ones(BS, BS, dtype=torch.bool, device=q.device), diagonal=0)
        dA = dA.masked_fill(mask, 0)
        dw += dA.transpose(-1, -2) @ w_beta
        dw_beta += dA @ w
        dbeta = (dw_beta * w).sum(-1)
        dw += dw_beta * beta[..., None]

        # beta, L_new, delta = map(lambda x: rearrange(x, 'b h (n c) -> b h n c', c=BS), [beta, L_new, delta])
        # w_beta = w * beta[..., None]
        # dk_final = torch.zeros_like(dk)
        # dq_final = torch.zeros_like(dq)
        # dw = torch.zeros_like(w)
        # dT = torch.zeros_like(T)
        # wbk = (w_beta @ k.transpose(-1, -2)).masked_fill(mask, 0)
        # Twbk = T @ wbk
        # Twb = T @ w_beta
        # qw = (q @ w.transpose(-1, -2)).tril()
        # A_local_softmax = (torch.exp2(A_local * 1.44269504 - L_new[..., None])).tril()
        # dv += A_local_softmax.transpose(-1, -2).to(do) @ do
        # dA = do @ v.transpose(-1, -2)
        # dA = ((dA - delta[..., None]) * A_local_softmax).tril()
        # ## deal with dA
        # dk_final += dA.transpose(-1, -2) @ q.transpose(-1, -2) + dk
        # dq_final += dA @ k + dq
        # ## gradient for  qw @ Twbk. dqw and dTwbk will be used later so we do not preceed to compute dqw and dTwbk.
        # dqw = dA @ Twbk.transpose(-1, -2)
        # dTwbk = qw.transpose(-1, -2) @ dA
        # ### gradient for q = -qw @ Twb
        # # q [A, B], qw[A, C]. Twb[C, B]
        # dTwb = -qw.transpose(-1, -2) @ dq
        # dqw -= (dq @ Twb.transpose(-1, -2))
        # # 处理dqw
        # dqw = dqw.tril()
        # dq_final += dqw @ w
        # dw += dqw.transpose(-1, -2) @ q
        # ### gradient for k = -Twbk.transpose(-1, -2) @ w
        # # k [A, B], Twbk[C, A]. w[C, B]
        # dw += -Twbk @ dk
        # dTwbk = -w @ dk
        # ### gradient for dH. H = w.transpose(-1, -2) @ Twb
        # # H [A, B], w[C, A]. Twb[C, B]
        # dTwb += w @ dh
        # dw += Twb @ dh.transpose(-1, -2)
        # ### 处理dTwb 
        # dT += dTwbk @ wbk
        # dwbk += T.transpose(-1, -2) @ dTwbk
        # ## 处理dqw
        # dq_final += dqw @ w
        # dw += dqw.transpose(-1, -2) @ q
        # ## 处理dTw: Twbk = T @ wbk

        # dwbk = dwbk.masked_fill(mask, 0)
        # #处理dwbk
        # dwb = dwbk @ k.transpose(-1, -2)
        # dk += dwb.transpose(-1, -2) @ w

        # # matrix inverse.
        # dT = T.transpose(-1, -2) @ dT @ T.transpose(-1, -2)
        # dT = -dT.tril(-1)
        # dw += dT @ w_beta
        # dwb += dT @ w

        # dw += (dwb * beta[..., None]).sum(-1)
        # dbeta = (dwb * w).sum(-1)
  
        dq, dk, dv, dw = map(lambda x: rearrange(x, 'b h n c d -> b h (n c) d'), [dq, dk, dv, dw])
        dbeta = rearrange(dbeta, 'b h n c -> b h (n c)')
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), dw.to(w.dtype), dbeta.to(beta.dtype), None



def parallel_hope(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    head_first: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        beta (torch.Tensor):
            betas of shape `[B, H, T]` if `head_first=True` else `[B, T, H]`.
        scale (Optional[int]):
            Scale factor for attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        output_attentions (bool):
            Whether to output the materialized attention scores of shape [B, H, T, T]. Default: `False`.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format.
            Default: `True`.
    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        attn (torch.Tensor):
            Attention scores of shape `[B, H, T, T]` if `output_attentions=True` else `None`.
    """
    if scale is None:
        scale = k.shape[-1]**-0.5
    if not head_first:
        q, k, v, w, beta = map(lambda x: x.transpose(1, 2), (q, k, v, w, beta))
    o  = ParallelDeltaRuleFunction.apply(q, k, v, w, beta, scale)
    if not head_first:
        o = o.transpose(1, 2)
    return o


def get_abs_err(x, y):
    return (x-y).flatten().abs().max().item()


def get_err_ratio(x, y):
    err = (x-y).flatten().square().mean().sqrt().item()
    base = (x).flatten().square().mean().sqrt().item()
    return err / base


def assert_close(prefix, ref, tri, ratio):
    msg = f"{prefix} diff: {get_abs_err(ref, tri):.6f} ratio: {get_err_ratio(ref, tri):.6f}"
    print(msg)
    assert get_err_ratio(ref, tri) < ratio, msg



def naive_delta_rule_parallel(q, k, v, w, beta, scale, BM=64, BN=64):
    
    original_dtype = q.dtype
    q, k, v, w, beta = map(lambda x: x.to(torch.float32), [q, k, v, w, beta])
    b, h, l, d_k = q.shape
    if l % BM != 0:
        padding_size = BM - l % BM
        q, k, v, w = map(lambda x: torch.nn.functional.pad(x, (0, 0, 0, padding_size)), [q, k, v, w])
        beta = torch.nn.functional.pad(beta, (0, padding_size))
    l_origin = l
    l = q.shape[-2]

    q = q * scale
    w_beta = w * beta[..., None]
    # compute (I - tri(diag(beta) KK^T))^{-1}
    q, k, v, w, w_beta = map(lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=BN), [q, k, v, w, w_beta])
    mask = torch.triu(torch.ones(BN, BN, dtype=torch.bool, device=q.device), diagonal=0)
    T = -(w_beta @ w.transpose(-1, -2)).masked_fill(mask, 0)
    for i in range(1, BN):
        T[..., i, :i] = T[..., i, :i].clone() + (T[..., i, :, None].clone() * T[..., :, :i].clone()).sum(-2)
    T = T + torch.eye(BN, dtype=q.dtype, device=q.device)
    
    # mask2 = torch.triu(torch.ones(BN, BN, dtype=torch.bool, device=q.device), diagonal=1)
    Twbk = T @ (w_beta @ k.transpose(-1, -2)).masked_fill(mask, 0)
    qw = (q @ w.transpose(-1, -2)).tril()
    Twb = T @ w_beta
    
    A_local = (q @ k.transpose(-1, -2)).tril() - qw @ Twbk
    q = q - qw @ Twb
    k = k - Twbk.transpose(-1, -2) @ w
    H = w.transpose(-1, -2) @ Twb

    A = torch.zeros(b, h, l, l, device=q.device)
    q, k, v, w, w_beta = map(lambda x: rearrange(x, 'b h n c d -> b h (n c) d'), [q, k, v, w, w_beta])
    for i in range(0, l, BM):
        q_i = q[:, :, i:i+BM].clone()
        # o_i = o_intra[:, :, i:i+BM]
        # # intra block
        for j in range(i + BM - 2 * BN, i-BN, -BN):
            w_j = w[:, :, j:j+BN]
            k_j = k[:, :, j:j+BN]
            A_ij = q_i @ k_j.transpose(-1, -2)
            mask = torch.arange(i, i+BM) >= (j + BN)
            A_ij = A_ij.masked_fill_(~mask[:, None].to(A_ij.device), 0)
            # A2_ij = A2_ij.masked_fill_(~mask[:, None].to(A2_ij.device), 0)
            # A[:, :, i:i+BM, j:j+BN] = A_ij
            # q_i = q_i - q_i
        # inter block
        for j in range(i - BN, -BN, -BN):
            k_j = k[:, :, j:j+BN]
            A_ij = q_i @ k_j.transpose(-1, -2)
            A[:, :, i:i+BM, j:j+BN] = A_ij
            q_i = q_i - q_i @ H[:, :, j // BN]

    for i in range(0, l//BN):
        A[:, :, i*BN:i*BN+BN, i*BN:i*BN+BN] = A_local[:, :, i]
    
    A = A.masked_fill_(~torch.tril(torch.ones(l, l, device=q.device, dtype=torch.bool)), float("-inf"))
    A = A[:, :, :l_origin, :l_origin]
    return A

    


def parallel_attn(q, k, v, w, beta, scale):
    b, h, l, d_k = q.shape
    q, k, v, w, beta = map(lambda x: x.to(torch.float32), [q, k, v, w, beta])
    q = q * scale
    A = torch.zeros(b, h, l, l, device=q.device)
    for i in range(l):
        q_i = q[:, :, i].clone()
        for j in range(i, -1, -1):
            k_j = k[:, :, j]
            w_j = w[:, :, j]
            beta_j = beta[:, :, j]
            A[:, :, i, j] = (q_i * k_j).sum(-1)
    return A



if __name__ == "__main__":
    B, H, T, K, V = 1, 1, 2048, 64, 64
    torch.set_default_dtype(torch.bfloat16)

    q = torch.rand(B, H, T, K).cuda()
    k = torch.rand(B, H, T, K).cuda()
    w = torch.nn.functional.normalize(torch.rand(B, H, T, K).cuda(), p=2, dim=-1)
    # k = w.clone()
    v = torch.rand(B, H, T, V).cuda()
    beta = torch.rand(B, H, T).sigmoid().cuda()
    # beta = torch.ones(B, H, T).cuda()
    q, k, v, w, beta = map(lambda x: x.requires_grad_(True), [q, k, v, w, beta])
    output_attentions = True

    # attn2 = parallel_attn(q.clone(), k.clone(), v.clone(), w.clone(), beta.clone(), K**-0.5)
    o2 = parallel_hope(q.clone(), k.clone(), v.clone(), w.clone(), beta.clone(), K**-0.5)
    do2 = torch.randn_like(o2)
    o2.backward(do2)
    

    # ref_attn = naive_delta_rule_parallel(q.clone(), k.clone(), v.clone(), w.clone(), beta.clone(), K**-0.5)
    # ref_o = ref_attn.float().softmax(-1).to(v) @ v
    # print(get_err_ratio(o2, ref_o))
    # # breakpoint()

    # print(get_err_ratio(attn2[:,:,32:64,:32], ref_attn[:, :, 32:64,:32]))
    # print(get_err_ratio(attn2[:,:,64+32:64+64,:32], ref_attn[:, :, 64+32:64+64,:32]))
    # print(get_err_ratio(attn2[:,:,-128:,:32], ref_attn[:, :, -128:,:32]))
    # print(get_err_ratio(attn2.tril(), ref_attn.tril()))
    # # breakpoint()

    # do = torch.randn_like(ref_o)
    # ref_o.backward(do)
    # q_grad, q.grad = q.grad.clone(), None
    # k_grad, k.grad = k.grad.clone(), None
    # v_grad, v.grad = v.grad.clone(), None
    # beta_grad, beta.grad = beta.grad.clone(), None

    # o2, _ = delta_rule_recurrence(q, k, v, w, beta, K**-0.5)
    # breakpoint()
    # assert_close('o', ref_o, o2, 0.005)
    # breakpoint()

    # o2.backward(do, retain_graph=True)

    # print(get_err_ratio(q_grad, q.grad), (q_grad-q.grad).abs().max())
    # print(get_err_ratio(k_grad, k.grad), (k_grad-k.grad).abs().max())
    # print(get_err_ratio(v_grad, v.grad), (v_grad-v.grad).abs().max())
    # print(get_err_ratio(beta_grad, beta.grad), (beta_grad-beta.grad).abs().max())
    # print("changing dtype")
    # breakpoint()

    q_grad = q.grad.clone()
    v_grad = v.grad.clone()
    k_grad = k.grad.clone()
    beta_grad = beta.grad.clone()
    w_grad = w.grad.clone()
    scale = K**-0.5
    BN = 64
    BM = 64




    # with torch.no_grad():
    original_dtype = q.dtype
    q2, k2, v2, w2, beta2 = map(lambda x: x.to(torch.float32).detach().clone().requires_grad_(True), [q, k, v, w, beta])
    q = q2.clone()
    k = k2.clone()
    v = v2.clone()
    w = w2.clone()
    beta = beta2.clone()
    b, h, l, d_k = q.shape
    # q = q * scale\

    w_beta = w * beta[..., None]
    # compute (I - tri(diag(beta) KK^T))^{-1}
    q, k, v, w, w_beta = map(lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=BN), [q, k, v, w, w_beta])
    mask = torch.triu(torch.ones(BN, BN, dtype=torch.bool, device=q.device), diagonal=0)
    T = -(w_beta @ w.transpose(-1, -2)).masked_fill(mask, 0)
    for i in range(1, BN):
        T[..., i, :i] = T[..., i, :i].clone() + (T[..., i, :, None].clone() * T[..., :, :i].clone()).sum(-2)
    T = T + torch.eye(BN, dtype=q.dtype, device=q.device)
    # T = T.clone().detach()

    mask2 = torch.triu(torch.ones(BN, BN, dtype=torch.bool, device=q.device), diagonal=1)
    A_local = (q @ k.transpose(-1, -2)).tril() - (q @ w.transpose(-1, -2)).tril() @ T @ (w_beta @ k.transpose(-1, -2)).masked_fill(mask, 0)
    
    q = q - (q @ w.transpose(-1, -2)).tril() @ T @ w_beta
    k = k - (k @ w_beta.transpose(-1, -2)).triu(1) @ T.transpose(-1, -2) @ w
    # o_intra = A_local @ v
    
    Tw = T @ w_beta
    H = w.transpose(-1, -2) @ Tw
    
    A = torch.zeros(b, h, l, l, device=q.device)
    q, k, v, w, w_beta = map(lambda x: rearrange(x, 'b h n c d -> b h (n c) d'), [q, k, v, w, w_beta])
    # q2, k2, v2, w2, beta2 = map(lambda x: x.to(torch.float32).detach().clone().requires_grad_(True), [q, k, v, w, beta])
    # q = q2.clone()
    # k = k2.clone()
    # v = v2.clone()
    # w = w2.clone()
    # beta = beta2.clone()

    for i in range(0, l, BM):
        q_i = q[:, :, i:i+BM].clone()
        # o_i = o_intra[:, :, i:i+BM]
        # # intra block
        for j in range(i + BM - 2 * BN, i-BN, -BN):
            w_j = w[:, :, j:j+BN]
            k_j = k[:, :, j:j+BN]
            A_ij = q_i @ k_j.transpose(-1, -2)
            mask = torch.arange(i, i+BM) >= (j + BN)
            A_ij = A_ij.masked_fill_(~mask[:, None].to(A_ij.device), 0)
            # A2_ij = A2_ij.masked_fill_(~mask[:, None].to(A2_ij.device), 0)
            # A[:, :, i:i+BM, j:j+BN] = A_ij
            # q_i = q_i - q_i
        # inter block
        for j in range(i - BN, -BN, -BN):
            k_j = k[:, :, j:j+BN]
            A_ij = q_i @ k_j.transpose(-1, -2)
            A[:, :, i:i+BM, j:j+BN] = A_ij
            q_i = q_i - q_i @ H[:, :, j // BN]

    for i in range(0, l//BN):
        A[:, :, i*BN:i*BN+BN, i*BN:i*BN+BN] = A_local[:, :, i]

    A = A.masked_fill_(~torch.tril(torch.ones(l, l, device=q.device, dtype=torch.bool)), float("-inf"))
    o3 = (A.to(torch.float32) * K**-0.5).softmax(-1).to(v) @ v
    o3.backward(do2)

    print(get_err_ratio(o3, o2))
    # print(get_err_ratio(o3, ref_o))
    # print(get_err_ratio(o2, ref_o))
    # (q_grad - q_origin.grad)[:,:,64]
    print(get_err_ratio(q_grad[:,:,:256], q2.grad[:,:,:256]))
    print(get_err_ratio(q_grad[:,:,256:], q2.grad[:,:,256:]))
    print(get_err_ratio(v_grad, v2.grad))
    print(get_err_ratio(k_grad, k2.grad))
    print(get_err_ratio(w_grad, w2.grad))
    print(get_err_ratio(beta_grad, beta2.grad))    
    breakpoint()
