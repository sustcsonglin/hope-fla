

# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang

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
    k,
    k2,
    v,
    beta,
    A,
    dA,
    do,
    dq,
    dk2,
    dv,
    D, # delta
    L, # logsumexp
    scale,
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

    # curr_end = (tl.floor(i_t * BT / BT_large).to(tl.int32) * BT_large).to(tl.int32)
    # b_q = tl.zeros([BT, BK], dtype=q.dtype.element_ty)

    # for offset in range(0, curr_end, BS):
    #     if offset % BT_large == 0:
    #         idx = offset // BT_large
    #         p_q = tl.make_block_ptr(q + ((idx+1) * B * H + i_bh) * T * K, (T, K), (K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    #         b_q = tl.load(p_q, boundary_check=(0, 1))
    #     p_k = tl.make_block_ptr(k + i_bh * T * K, (K, T), (1, K), (0, offset), (BK, BS), (0, 1))
    #     p_v = tl.make_block_ptr(v + i_bh * T * K, (V, T), (1, V), (0, offset), (BK, BS), (0, 1))
    #     p_k2 = tl.make_block_ptr(k2 + i_bh * T * K, (K, T), (1, K), (0, offset), (BK, BS), (0, 1))
    #     p_beta = tl.make_block_ptr(beta + i_bh * T, (T, ), (1,), (offset,), (BS,), (0,))
    #     b_k = tl.load(p_k, boundary_check=(0, 1))
    #     b_k2 = tl.load(p_k2, boundary_check=(0, 1))
    #     b_beta = tl.load(p_beta, boundary_check=(0,))
    #     b_k2 = (b_k2 * b_beta[None, :]).to(b_k.dtype)
    #     b_v = tl.load(p_v, boundary_check=(0, 1))
    #     p_h = tl.make_block_ptr(h + (i_bh * NT + tl.cdiv(offset, BS)) * K * K, (K, K), (K, 1), (0, 0), (BK, BK), (1, 0))
    #     b_h = tl.load(p_h, boundary_check=(0, 1))
    #     b_qh = tl.dot(b_q, b_h.to(b_q.dtype)).to(b_k.dtype)
    #     b_s = tl.dot(b_qh.to(b_k.dtype), b_k)
    #     p_a = tl.make_block_ptr(A_global + i_bh * T * T, (T, T), (T, 1), (i_t * BT, offset), (BT, BS), (1, 0))
    #     tl.store(p_a, b_s, boundary_check=(0, 1))
    #     b_ds = tl.dot(b_do, b_v)
    #     b_p = tl.math.exp2(b_s * sm_scale - l[:, None])
    #     b_ds = (b_ds - delta[:, None]) * b_p * scale
    #     b_ds -= tl.dot(b_dq, b_k2.to(b_dq.dtype))
    #     b_dq += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k))
    
    for offset in range(0, (i_t + 1) * BT - BS, BS):
        p_k = tl.make_block_ptr(k + i_bh * T * K, (T, K), (K, 1), (offset, 0), (BS, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        m_s = tl.arange(0, BT) >= (offset - i_t*BT + BS)
        start = offset - 0
        p_A = tl.make_block_ptr(A + i_bh * T * BT_large, (T, BT_large), (BT_large, 1), (i_t * BT, start), (BT, BS), (1, 0))
        b_A = tl.load(p_A, boundary_check=(0, 1))

        b_A_softmax = tl.math.exp2(tl.where(m_s[:, None], b_A, float("-inf")) * sm_scale - l[:, None])
        b_dv = tl.dot(tl.trans(b_do), b_A_softmax.to(b_do.dtype))
        # mask = (tl.arange(0, BK) < V)[:, None] & ((offset + tl.arange(0, BS)) < T)[None, :]
        tl.atomic_add(dv + i_bh * T * V + (offset + tl.arange(0, BS))[None, :] * V + tl.arange(0, BK)[:, None], b_dv, sem='relaxed')

        # p_A = tl.make_block_ptr(A + i_bh * T * T, (T, T), (T, 1), (i_t * BT, offset), (BT, BS), (1, 0))
        # b_A_softmax = tl.load(p_A, boundary_check=(0, 1))
        p_v = tl.make_block_ptr(v + i_bh * T * K, (V, T), (1, V), (0, offset), (BK, BS), (0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_dp = tl.dot(b_do, b_v)
        b_dA = (b_dp - delta[:, None]) * b_A_softmax * scale
        p_k2 = tl.make_block_ptr(k2 + i_bh * T * K, (K, T), (1, K), (0, offset), (BK, BS), (0, 1))
        b_k2 = tl.load(p_k2, boundary_check=(0, 1))
        p_beta = tl.make_block_ptr(beta + i_bh * T, (T, ), (1,), (offset,), (BS,), (0,))
        b_beta = tl.load(p_beta, boundary_check=(0, ))
        b_k2 = (b_k2 * b_beta[None, :]).to(b_k.dtype)
        b_dA -= tl.dot(b_dq.to(b_k2.dtype), b_k2)
        b_dk2 = -tl.dot(tl.trans(b_A), b_dq.to(b_A.dtype))
        tl.atomic_add(dk2 + i_bh * T * K + (offset + tl.arange(0, BS))[:, None] * K + tl.arange(0, BK)[None, :], b_dk2, sem='relaxed')
        b_dA = tl.where(m_s[:, None], b_dA, 0)
        b_dq += tl.dot(b_dA.to(b_k.dtype), b_k)
        p_dA = tl.make_block_ptr(dA + i_bh * T * T, (T, T), (T, 1), (i_t * BT, offset), (BT, BS), (1, 0))
        tl.store(p_dA, b_dA.to(dA.dtype.element_ty), boundary_check=(0, 1))

    p_dq = tl.make_block_ptr(dq + i_bh * T * K, (T, K), (K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    tl.store(p_dq, b_dq.to(dq.dtype.element_ty), boundary_check=(0, 1))



# todo: to make k2 already k * beta[:, None]. 
@triton.jit
def parallel_softmax_delta_rule_bwd_kernel_dk(
    q,
    k,
    k2,
    v,
    beta,
    A,
    dA,
    do,
    dq,
    dk,
    dk2,
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
    # p_do = tl.make_block_ptr(do + i_bh * T * V, (T, V), (V, 1), (i_t * BT, 0), (BT, BV), (1, 0))
    # [BT, BV]
    # b_do = tl.load(p_do, boundary_check=(0, 1))
    # [BT, BK]
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    # sm_scale = scale * 1.44269504

    # p_v = tl.make_block_ptr(v + i_bh * T * K, (V, T), (1, V), (0, i_t * BT), (BK, BT), (0, 1))
    # b_v = tl.load(p_v, boundary_check=(0, 1))


    for offset in range(tl.cdiv(T, BS) * BS - BS, i_t * BT, -BS):
        # p_k = tl.make_block_ptr(k + i_bh * T * K, (T, K), (K, 1), (offset, 0), (BS, BK), (1, 0))
        # b_k = tl.load(p_k, boundary_check=(0, 1))
        m_s = tl.arange(0, BT) < (offset - i_t * BT)

        p_k = tl.make_block_ptr(k + i_bh * T * K, (K, T), (1, K), (0, offset), (BK, BS), (0, 1))
        p_k2 = tl.make_block_ptr(k2 + i_bh * T * K, (T, K), (K, 1), (offset, 0), (BS, BK), (1, 0))
        p_beta = tl.make_block_ptr(beta + i_bh * T, (T, ), (1,), (offset,), (BS,), (0,))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_k2 = tl.load(p_k2, boundary_check=(0, 1))
        b_beta = tl.load(p_beta, boundary_check=(0,))
        b_k2 = (b_k2 * b_beta[:, None]).to(b_k.dtype)
    
        b_s = tl.dot(b_dk, b_k.to(b_dk.dtype))
        b_s = tl.where(m_s[:, None], b_s, 0).to(b_k2.dtype)
        b_dk -= tl.dot(b_s, b_k2)

        # p_l = tl.make_block_ptr(L + i_bh * T, (T, ), (1, ), (offset, ), (BS, ), (0, ))
        # b_l = tl.load(p_l, boundary_check=(0, ))
        # p_delta = tl.make_block_ptr(D + i_bh * T, (T, ), (1, ), (offset, ), (BS, ), (0, ))
        # b_delta = tl.load(p_delta, boundary_check=(0, ))
        # b_A_softmax = tl.math.exp2(tl.where(m_s[None, :], b_A, float("-inf")) * sm_scale - b_l[:, None])
    
        p_q = tl.make_block_ptr(q + i_bh * T * K, (T, K), (K, 1), (offset, 0), (BS, BK), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))

        # [BQS, BKT]
        # b_dp = tl.dot(b_do, b_v)
        # b_dA = ((b_dp - b_delta[:, None]) * b_A_softmax * scale).to(b_q.dtype)
        # b_dk += tl.dot(tl.trans(b_dA), b_q)

        p_dA = tl.make_block_ptr(dA + i_bh * T * T, (T, T), (1, T), (i_t * BT, offset), (BT, BS), (0, 1))
        b_dA = tl.load(p_dA, boundary_check=(0, 1))
        b_dA = tl.where(m_s[:, None], b_dA, 0).to(b_q.dtype)
        b_dk += tl.dot(b_dA.to(b_q.dtype), b_q)

        # p_k2 = tl.make_block_ptr(k2 + i_bh * T * K, (K, T), (1, K), (0, offset), (BK, BS), (0, 1))
        # b_k2 = tl.load(p_k2, boundary_check=(0, 1))
        # p_beta = tl.make_block_ptr(beta + i_bh * T, (T, ), (1,), (offset,), (BS,), (0,))
        # b_beta = tl.load(p_beta, boundary_check=(0, ))
        # b_k2 = (b_k2 * b_beta[None, :]).to(b_k.dtype)
        # b_dq_tmp = tl.zeros([BT, BK], dtype=tl.float32)
        # b_dq_tmp += b_dq
        # b_dA -= tl.dot(b_dq, b_k2.to(b_dq.dtype))
        # b_dk2 = -tl.dot(tl.trans(b_A), b_dq)
        # b_dA = tl.where(m_s[:, None], b_dA, 0).to(b_k.dtype)
        # b_dq += tl.dot(b_dA, b_k)

    p_dk = tl.make_block_ptr(dk + i_bh * T * K, (T, K), (K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    tl.store(p_dk, b_dk.to(dk.dtype.element_ty), boundary_check=(0, 1))


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
    h2,
    h3, 
    BT: tl.constexpr, # previous small chunk size
    BT_large: tl.constexpr, # larger chunk size
    NT: tl.constexpr,
    NT_large: tl.constexpr, 
    K: tl.constexpr,
    BK: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    NT_small = triton.cdiv(BT_large, BT) # how many small chunks in a large chunk
    b_h = tl.zeros([BK, BK], dtype=tl.float32) + (tl.arange(0, BK)[:, None] == tl.arange(0, BK)[None, :]).to(tl.float32)

    for i_t_small in range(NT_small-1, -1, -1):
        p_h2 = tl.make_block_ptr(h2 + (i_bh * NT + i_t * NT_small + i_t_small) * K * K , (K, K), (K, 1), (0, 0), (BK, BK), (1, 0))
        tl.store(p_h2, b_h.to(h2.dtype.element_ty), boundary_check=(0, 1))
        p_h = tl.make_block_ptr(h + (i_bh * NT + i_t * NT_small + i_t_small) * K * K , (K, K), (K, 1), (0, 0), (BK, BK), (1, 0))
        b_h2 = tl.load(p_h, boundary_check=(0, 1))
        b_h = tl.dot(b_h, b_h2)    
    p_h3 = tl.make_block_ptr(h3 + (i_bh * NT_large + i_t) * K * K , (K, K), (K, 1), (0, 0), (BK, BK), (1, 0))
    tl.store(p_h3, b_h.to(h3.dtype.element_ty), boundary_check=(0, 1))



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
        triton.Config({}, num_warps=1),
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
    beta,
    o,
    A,
    L,
    M,  
    h,
    q_new,
    k_new,
    A_local,
    s_k_h,
    s_k_t,
    s_k_d,
    s_v_h,
    s_v_t,
    s_v_d,
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
    p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, 0), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, 0), (BT, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, 0), (BT, BV), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_v = tl.load(p_v, boundary_check=(0, 1))

    p_T = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_T = tl.load(p_T, boundary_check=(0, 1))

    o_i = tl.arange(0, BT)
    m_t = o_i[:, None] >= o_i[None, :]
    b_qk = tl.where(m_t, tl.dot(b_q, tl.trans(b_k)), 0)
    m_t = o_i[:, None] > o_i[None, :]
    b_kk = tl.where(m_t, tl.dot(b_k, tl.trans(b_k)), 0)

    p_beta = tl.make_block_ptr(beta + i_bh * T, (T, ), (1, ), (i_t * BT, ), (BT, ), (0, ))
    b_beta = tl.load(p_beta, boundary_check=(0, ))
    b_k_beta = (b_k * b_beta[:, None]).to(b_k.dtype)

    # 32x32 fp32
    b_qkT = tl.dot(b_qk, b_T)
    if OUTPUT_ATTENTIONS:
        p_a = tl.make_block_ptr(A_local + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
        tl.store(p_a, b_qkT.to(p_a.dtype.element_ty), boundary_check=(0, 1))

    b_qkT_softmax = tl.where(o_i[:, None] >= o_i[None, :], b_qkT * sm_scale, float("-inf"))
    m_i = tl.max(b_qkT_softmax, 1)
    b_qkT_softmax = tl.math.exp2(b_qkT_softmax - m_i[:, None])
    l_i = tl.sum(b_qkT_softmax, 1)
    b_o = tl.dot(b_qkT_softmax.to(b_v.dtype), b_v)
    p_o = tl.make_block_ptr(o + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, 0), (BT, BV), (1, 0))
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    p_l = L + i_bh * T + i_t * BT + tl.arange(0, BT)
    p_m = M + i_bh * T + i_t * BT + tl.arange(0, BT)
    mask = i_t * BT + tl.arange(0, BT) < T
    tl.store(p_m, m_i.to(p_m.dtype.element_ty), mask=mask)
    tl.store(p_l, l_i.to(p_l.dtype.element_ty), mask=mask)
    # 32x32 fp32
    b_kkT = tl.dot(b_kk, b_T).to(b_k.dtype)
    p_q_new = tl.make_block_ptr(q_new + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, 0), (BT, BK), (1, 0))
    tl.store(p_q_new, (b_q - tl.dot(b_qkT.to(b_k.dtype), b_k_beta)).to(p_q_new.dtype.element_ty), boundary_check=(0, 1))
    p_k_new = tl.make_block_ptr(k_new + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, 0), (BT, BK), (1, 0))
    b_k_new = (b_k - tl.dot(tl.trans(b_kkT), b_k_beta)).to(b_k.dtype)
    tl.store(p_k_new, b_k_new.to(p_k_new.dtype.element_ty), boundary_check=(0, 1))

    # b_h = (tl.arange(0, K)[:, None] == tl.arange(0, K)[None, :]).to(tl.float32) - tl.dot(tl.trans(b_k_new).to(tl.float32), b_k_beta.to(tl.float32))
    # p_h = tl.make_block_ptr(h + (i_bh * NT + i_t) * K * K, (K, K), (K, 1), (0, 0), (BK, BK), (1, 0))
    # tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))


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
    dq_new,
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
    p_T = tl.make_block_ptr(AT + i_bh * T * BT, (BT, T), (1, BT), (0, i_t * BT), (BT, BT), (0, 1))
    b_Tt = tl.load(p_T, boundary_check=(0, 1))

    # b_T = tl.load(p_T, boundary_check=(0, 1))

    # [m, n] x [n, k] = [m, k]
    # b_dT = tl.dot(tl.trans(b_qk), b_dqkT)
    # backward. [m, n] = [m, k] x [k, n]
    b_dqk = tl.dot(b_dqkT, b_Tt)
    # qkT[m, k] x T[k, n] = dqkT[m, n]
    b_dkk = tl.dot(b_dkkT, b_Tt)

    b_kk = tl.where(tl.arange(0, BT)[:, None] > tl.arange(0, BT)[None, :], tl.dot(b_k, tl.trans(b_k)), 0)
    b_dT = tl.dot(tl.trans(b_qk), b_dqkT) + tl.dot(tl.trans(b_kk), b_dkkT)
    b_dT = tl.where(tl.arange(0, BT)[:, None] > tl.arange(0, BT)[None, :], b_dT, 0)
    b_dT = tl.dot(b_dT, b_Tt)
    b_dT = tl.dot(b_Tt, b_dT)
    b_dT = tl.where(tl.arange(0, BT)[:, None] > tl.arange(0, BT)[None, :], -b_dT, 0).to(b_k.dtype)

    # b_dq, b_dk = tl.dot(b_T.to(b_dq.dtype), b_dq), tl.dot(b_T.to(b_dk.dtype), b_dk)
    # qk[m, n] x kbeta[n, k] = q[m, k]
    # [m, k] x [n, k].transpose = [m, n]
    # [m, k] = [m, n] x [n,]

    b_dq_final = tl.zeros([BT, K], dtype=tl.float32)
    b_dk_final = tl.zeros([BT, K], dtype=tl.float32)
    b_dbeta_final = tl.zeros([BT], dtype=tl.float32)
    b_dk_final += tl.dot(tl.trans(b_dT), b_k_beta)
    p_T = tl.make_block_ptr(AT + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_T = tl.load(p_T, boundary_check=(0, 1))
    b_qkT = tl.dot(b_qk, b_T)
    b_kkT = tl.dot(b_kk, b_T)
    b_dkbeta = tl.dot(b_dT, b_k) -tl.dot(tl.trans(b_qkT.to(b_dq.dtype)), b_dq) - tl.dot(b_kkT.to(b_dk.dtype), b_dk)
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
    ## HELP ME
    p_dq_new = tl.make_block_ptr(dq_new + i_bh * T * K, (T, K), (K, 1), (i_t * BT, 0), (BT, K), (1, 0))
    tl.store(p_dq_new, b_dq_final.to(p_dq_new.dtype.element_ty), boundary_check=(0, 1))
    p_dk = tl.make_block_ptr(dk + i_bh * T * K, (T, K), (K, 1), (i_t * BT, 0), (BT, K), (1, 0))
    b_dk_final += tl.load(p_dk, boundary_check=(0, 1))
    tl.store(p_dk, b_dk_final.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    p_dbeta = tl.make_block_ptr(dbeta + i_bh * T, (T, ), (1, ), (i_t * BT, ), (BT, ), (0, ))
    tl.store(p_dbeta, b_dbeta_final.to(p_dbeta.dtype.element_ty), boundary_check=(0, ))



def chunk_transform_qk_fwd_fn(q, k, v, beta, A, scale, BT, output_attentions):
    B, H, T, K = k.shape
    q_new = torch.empty_like(q)
    k_new = torch.empty_like(k)
    o = torch.empty_like(v)
    NT = triton.cdiv(T, BT)
    grid = (NT, B*H)
    V = v.shape[-1]
    A_local = torch.empty_like(A) if output_attentions else None
    L = torch.empty(B, H, T, dtype=torch.float32, device=q.device)
    M = torch.empty(B, H, T, dtype=torch.float32, device=q.device)
    h = torch.zeros(B, H, NT, K, K, dtype=torch.float32, device=q.device)
    chunk_transform_qk_fwd_kernel[grid](
        q,  k, v, beta, o, A, L, M, h, q_new, k_new, A_local,
        q.stride(1), q.stride(2), q.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        scale=scale,
        T=T,
        K=K,
        V=V,
        BT=BT,
        BK=triton.next_power_of_2(K),
        BV=triton.next_power_of_2(V),
        OUTPUT_ATTENTIONS=output_attentions
    )

    # h_new = torch.zeros_like(h)
    # # cache q_new every BT_large tokens 
    # BT_large = 256
    # grid = (triton.cdiv(T, BT_large) - 1, B*H)
    # h3 = torch.zeros(B, H, T//BT_large, K, K, dtype=h_new.dtype, device=q.device)
    # chunk_cumprod_householder_fwd_kernel[grid](
    #     h=h,
    #     h2=h_new,
    #     h3=h3,
    #     BT=BT,
    #     BT_large=BT_large,
    #     K=K,
    #     NT=NT,
    #     NT_large=triton.cdiv(T, BT_large),
    #     BK=triton.next_power_of_2(K),
    # )
    # h4 = torch.zeros(B, H, T//BT_large, T//BT_large, K, K, dtype=h3.dtype, device=q.device)
    # grid = (triton.cdiv(T, BT_large) - 1, B*H)
    # chunk_cumprod_householder_fwd2_kernel[grid](
    #     h=h3,
    #     h_pairwise=h4,
    #     NT=T//BT_large,
    #     K=K,
    #     BK=triton.next_power_of_2(K),
    # )
    return q_new, k_new, o, A_local, L, M, None, None

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
    k2,  # original k
    v,
    beta,
    o,
    o_new,
    attn,
    scale,
    L,
    L_new,
    M,
    s_k_h,
    s_k_t,
    s_v_h,
    s_v_t,
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
    p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    sm_scale = scale * 1.44269504

    # the Q block is kept in the shared memory throughout the whole kernel
    # [BT, BK]
    b_q = tl.zeros([BT, BK], dtype=tl.float32)
    b_q += tl.load(p_q, boundary_check=(0, 1))

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    p_o = tl.make_block_ptr(o + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, 0), (BT, BV), (1, 0))
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
    for offset in range((i_t + 1) * BT - 2 * BS, i_t * BT - BS, -BS):
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (K, T), (1, s_k_t), (0, offset), (BK, BS), (0, 1))
        p_k2 = tl.make_block_ptr(k2 + i_bh * s_k_h, (T, K), (s_k_t, 1), (offset, 0), (BS, BK), (1, 0))
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, 1), (offset, 0), (BS, BV), (1, 0))
        p_beta = tl.make_block_ptr(beta + i_bh * T, (T, ), (1, ), (offset, ), (BS, ), (0,))
        # [BK, BS]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BS, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BS]
        b_beta = tl.load(p_beta, boundary_check=(0,))
        # [BT, BS]
        m_s = tl.arange(0, BT) >= (offset - i_t*BT + BS)
        b_s = tl.dot(b_q.to(b_k.dtype), b_k)
        b_s = tl.where(m_s[:, None], b_s, 0)
        
        if OUTPUT_ATTENTIONS:
            p_a = tl.make_block_ptr(attn + i_bh * T * T, (T, T), (T, 1), (i_t * BT, offset), (BT, BS), (1, 0))
            tl.store(p_a, b_s.to(p_a.dtype.element_ty), boundary_check=(0, 1))

        b_k2 = (tl.load(p_k2, boundary_check=(0, 1)) * b_beta[:, None]).to(b_v.dtype)
        b_q -= tl.dot(b_s.to(b_k2.dtype), b_k2)
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

    # # Q block and K block have no overlap
    # # no need for mask, thereby saving flops
    for offset in range(i_t * BT - BS, -BS, -BS):
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (K, T), (1, s_k_t), (0, offset), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, 1), (offset, 0), (BS, BV), (1, 0))
        p_beta = tl.make_block_ptr(beta + i_bh * T, (T, ), (1, ), (offset, ), (BS, ), (0,))
        p_k2 = tl.make_block_ptr(k2 + i_bh * s_k_h, (T, K), (s_k_t, 1), (offset, 0), (BS, BK), (1, 0))

        # [BK, BS]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BS, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BS]
        b_beta = tl.load(p_beta, boundary_check=(0,))
        # [BT, BS]
        b_s = (tl.dot(b_q.to(b_k.dtype), b_k))

        if OUTPUT_ATTENTIONS:
            p_a = tl.make_block_ptr(attn + i_bh * T * T, (T, T), (T, 1), (i_t * BT, offset), (BT, BS), (1, 0))
            tl.store(p_a, b_s.to(p_a.dtype.element_ty), boundary_check=(0, 1))

        # update q first
        b_k2 = (tl.load(p_k2, boundary_check=(0, 1)) * b_beta[:, None]).to(b_v.dtype)
        b_q -= tl.dot(b_s.to(b_k2.dtype), b_k2).to(b_q.dtype)

        b_s = b_s * sm_scale
        b_m_new = tl.maximum(b_m, tl.max(b_s, 1))
        alpha = tl.math.exp2((b_m - b_m_new))
        b_s = tl.math.exp2(b_s - b_m_new[:, None])
        if i_t >= 0:
            b_o *= alpha[:, None]
            b_l = b_l * alpha + tl.sum(b_s, 1)
            b_m = b_m_new
        b_o += tl.dot(b_s.to(b_v.dtype), b_v)

    b_o = b_o / b_l[:, None]
    p_o_new = tl.make_block_ptr(o_new + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t*BT, 0), (BT, BV), (1, 0))
    tl.store(p_o_new, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    
    b_l = tl.math.log2(b_l) + b_m
    p_L_new = L_new + i_bh * T + i_t * BT + tl.arange(0, BT)
    mask = i_t * BT + tl.arange(0, BT) < T
    tl.store(p_L_new, b_l.to(p_L_new.dtype.element_ty), mask=mask)


@triton.jit
def parallel_delta_rule_bwd_prepare_kernel(
    q,
    # q_last,
    k,
    k2,  # original k
    beta,
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
    curr_end = (tl.floor(i_t * BT / BT_large).to(tl.int32) * BT_large).to(tl.int32) - BS
    for offset in range((i_t + 1) * BT - 2 * BS, curr_end, -BS):
        p_k = tl.make_block_ptr(k + i_bh * T * K, (K, T), (1, K), (0, offset), (BK, BS), (0, 1))
        p_k2 = tl.make_block_ptr(k2 + i_bh * T * K, (T, K), (K, 1), (offset, 0), (BS, BK), (1, 0))
        p_beta = tl.make_block_ptr(beta + i_bh * T, (T, ), (1, ), (offset, ), (BS, ), (0,))
        # [BK, BS]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BS]
        b_beta = tl.load(p_beta, boundary_check=(0,))
        # [BT, BS]
        m_s = tl.arange(0, BT) >= (offset - i_t*BT + BS)
        b_s = tl.dot(b_q.to(b_k.dtype), b_k)
        b_s = tl.where(m_s[:, None], b_s, 0)
        # if OUTPUT_ATTENTIONS:
        p_a = tl.make_block_ptr(attn + i_bh * T * BT_large, (T, BT_large), (BT_large, 1), (i_t * BT, offset - curr_end - BS), (BT, BS), (1, 0))
        tl.store(p_a, b_s.to(p_a.dtype.element_ty), boundary_check=(0, 1))

        b_k2 = (tl.load(p_k2, boundary_check=(0, 1)) * b_beta[:, None])
        b_q -= tl.dot(b_s.to(b_k2.dtype), b_k2)



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
    def forward(ctx, q, k, v, beta, scale):
        B, H, T, K, V = *k.shape, v.shape[-1]
        assert q.shape[-1] <= 128, 'The maximum supported sequence length is 128.'
        BT, BS = 128, 64
        BK = triton.next_power_of_2(k.shape[-1])
        BV = triton.next_power_of_2(v.shape[-1])
        assert BT % BS == 0

        A = fwd_prepare_T(k, beta, None,  None, True, BS)

        q_new, k_new, o, _, L, M, _, _ = chunk_transform_qk_fwd_fn(q, k, v, beta, A, scale, BS, False)
        num_stages = 3 if K <= 64 else 2
        num_warps = 8
        grid = (triton.cdiv(T, BT), B * H)
        o_new = torch.empty_like(o)
        L_new = torch.empty_like(L)
        parallel_delta_rule_fwd_kernel[grid](
            q=q_new,
            k=k_new,
            k2=k,
            v=v,
            beta=beta,
            o=o,
            o_new=o_new,
            attn=None,
            scale=scale,
            L=L,
            L_new=L_new,
            M=M,
            s_k_h=k.stride(1),
            s_k_t=k.stride(2),
            s_v_h=v.stride(1),
            s_v_t=v.stride(2),
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

        # if output_attentions:
        #     grid = (triton.cdiv(T, BS), B * H)
        #     save_intra_chunk_attn[grid](
        #         A=attn, A_local=A_local, T=T, BT=BS
        #     )
        # softmax_attn = torch.exp2(attn * scale * 1.4142135623730951 - L_new[..., None])
        ctx.save_for_backward(q, k, v, o_new, beta, L_new)
        # attn = attn.masked_fill_(~torch.tril(torch.ones(T, T, device=attn.device, dtype=torch.bool)), float("-inf"))
        ctx.BT = BT
        ctx.BS = BS
        ctx.scale = scale
        return o_new


    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(ctx, do):
        q, k, v, o, beta, L_new = ctx.saved_tensors
        B, H, T, K, V = *k.shape, v.shape[-1]
        BT_large = T
        assert q.shape[-1] <= 128, 'The maximum supported sequence length is 128.'
        BT, BS = 128, 64
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

        A = fwd_prepare_T(k, beta, None,  None, True, BS)
        q_new, k_new, _, A_local, _, _, _, _ = chunk_transform_qk_fwd_fn(q, k, v, beta, A, scale, BS, True)

        B, H, T, K = q.shape
        A_local_larger = torch.empty(B, H, T, T, dtype=torch.float32, device=q.device)
        # should be something in this form here.       
        grid = (triton.cdiv(T, BT), B * H)

        parallel_delta_rule_bwd_prepare_kernel[grid](
            q=q_new,
            k=k_new,
            k2=k,
            beta=beta,
            attn=A_local_larger,
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

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k, dtype=torch.float32)
        dk2 = torch.zeros_like(k, dtype=torch.float32)
        dv = torch.zeros_like(v, dtype=torch.float32)
        grid = (triton.cdiv(T, BT), H*B)
        dA = torch.zeros_like(A_local_larger, dtype=torch.float32)

        parallel_softmax_delta_rule_bwd_kernel[grid](
            k=k_new,
            k2=k,
            v=v,
            beta=beta,
            A=A_local_larger,
            dA=dA,
            do=do,
            D=delta,
            L=L_new,
            dq=dq,
            dk2=dk2,
            dv=dv,
            scale=scale,
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

        parallel_softmax_delta_rule_bwd_kernel_dk[grid](
            q=q_new,
            k=k_new,
            k2=k,
            v=v,
            beta=beta,
            A=A_local_larger,
            dA=dA,
            do=do,
            D=delta,
            L=L_new,
            dq=dq,
            dk=dk,
            dk2=dk2,
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
        grid = (triton.cdiv(T, BS), B * H)
        dbeta = torch.zeros_like(beta)
        dq_new = torch.zeros_like(dq).fill_(float("nan"))
        chunk_transform_qk_bwd_kernel[grid](
            q=q,
            k=k,
            v=v,
            beta=beta,
            A=A_local,
            AT=A,
            L=L_new,
            D=delta,
            do=do,
            dv=dv,
            dq=dq,
            dq_new=dq_new,
            dk=dk,
            dk2=dk2,
            dbeta=dbeta,
            scale=scale,
            H=H,
            T=T,
            K=K,
            BT=BS,
            BK=BK,
            BV=BV,
            V=V
        )
        return dq_new, dk, dv, dbeta, None, None


def parallel_hope(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
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
        q, k, v, beta = map(lambda x: x.transpose(1, 2), (q, k, v, beta))
    o  = ParallelDeltaRuleFunction.apply(q, k, v, beta, scale)
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




def naive_delta_rule_parallel(q, k, v, beta, scale, BM=128, BN=64):
    

    original_dtype = q.dtype
    q, k, v, beta = map(lambda x: x.to(torch.float32), [q, k, v, beta])
    b, h, l, d_k = q.shape
    if l % BM != 0:
        padding_size = BM - l % BM
        q, k, v = map(lambda x: torch.nn.functional.pad(x, (0, 0, 0, padding_size)), [q, k, v])
        beta = torch.nn.functional.pad(beta, (0, padding_size))
    l_origin = l
    l = q.shape[-2]

    k_beta = k * beta[..., None]
    # compute (I - tri(diag(beta) KK^T))^{-1}
    q, k, v, k_beta = map(lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=BN), [q, k, v, k_beta])
    mask = torch.triu(torch.ones(BN, BN, dtype=torch.bool, device=q.device), diagonal=0)
    T = -(k_beta @ k.transpose(-1, -2)).masked_fill(mask, 0)
    for i in range(1, BN):
        T[..., i, :i] = T[..., i, :i].clone() + (T[..., i, :, None].clone() * T[..., :, :i].clone()).sum(-2)
    T = T + torch.eye(BN, dtype=q.dtype, device=q.device)

    mask2 = torch.triu(torch.ones(BN, BN, dtype=torch.bool, device=q.device), diagonal=1)
    A_local = (q @ k.transpose(-1, -2)).masked_fill(mask2, 0) @ T
    o_intra = A_local @ v

    # apply cumprod transition matrices on k to the last position within the chunk
    k = k - ((k @ k.transpose(-1, -2)).masked_fill(mask, 0) @ T).transpose(-1, -2) @ k_beta
    # apply cumprod transition matrices on q to the first position within the chunk
    q = q - A_local @ k_beta

    A = torch.zeros(b, h, l, l, device=q.device)

    q, k, v, k_beta, o_intra = map(lambda x: rearrange(x, 'b h n c d -> b h (n c) d'), [q, k, v, k_beta, o_intra])
    o = torch.empty_like(v)
    for i in range(0, l, BM):
        q_i = q[:, :, i:i+BM].clone()
        # o_i = o_intra[:, :, i:i+BM]
        # intra block
        for j in range(i + BM - 2 * BN, i-BN, -BN):
            k_j = k[:, :, j:j+BN]
            A_ij = q_i @ k_j.transpose(-1, -2)
            mask = torch.arange(i, i+BM) >= (j + BN)
            A_ij = A_ij.masked_fill_(~mask[:, None].to(A_ij.device), 0)
            A[:, :, i:i+BM, j:j+BN] = A_ij
            q_i = q_i - A_ij @ k_beta[:, :, j:j+BN]
            # o_i += A_ij @ v[:, :, j:j+BN]
        # inter block
        for j in range(i - BN, -BN, -BN):
            k_j = k[:, :, j:j+BN]
            A_ij = q_i @ k_j.transpose(-1, -2)
            A[:, :, i:i+BM, j:j+BN] = A_ij
            q_i = q_i - A_ij @ k_beta[:, :, j:j+BN]
            # o_i += A_ij @ v[:, :, j:j+BN]

    for i in range(0, l//BN):
        A[:, :, i*BN:i*BN+BN, i*BN:i*BN+BN] = A_local[:, :, i]
    
    A = A.masked_fill_(~torch.tril(torch.ones(l, l, device=q.device, dtype=torch.bool)), float("-inf"))
    A = A[:, :, :l_origin, :l_origin]
    return A




if __name__ == "__main__":
    B, H, T, K, V = 2, 16, 2048, 64, 64
    torch.set_default_dtype(torch.bfloat16)

    q = torch.nn.functional.normalize(torch.randn(B, H, T, K).cuda(), p=2, dim=-1)
    k = torch.nn.functional.normalize(torch.randn(B, H, T, K).cuda(), p=2, dim=-1)
    v = torch.randn(B, H, T, V).cuda()
    beta = torch.rand(B, H, T).sigmoid().cuda() 
    # beta = torch.ones(B, H, T).cuda()

    q, k, v, beta = map(lambda x: x.requires_grad_(True), [q, k, v, beta])
    output_attentions = True
    # with torch.no_grad():
    ref_attn = naive_delta_rule_parallel(q.clone(), k.clone(), v.clone(), beta.clone(), K**-0.5)
    ref_o = (ref_attn.float() * K**-0.5).softmax(-1).to(v) @ v
    do = torch.randn_like(ref_o)
    ref_o.backward(do)

    q_grad, q.grad = q.grad.clone(), None
    k_grad, k.grad = k.grad.clone(), None
    v_grad, v.grad = v.grad.clone(), None
    beta_grad, beta.grad = beta.grad.clone(), None

    o2 = parallel_hope(q, k, v, beta, K**-0.5)
    assert_close('o', ref_o, o2, 0.005)
    o2.backward(do, retain_graph=True)
    print(get_err_ratio(q_grad, q.grad), (q_grad-q.grad).abs().max())
    print(get_err_ratio(k_grad, k.grad), (k_grad-k.grad).abs().max())
    print(get_err_ratio(v_grad, v.grad), (v_grad-v.grad).abs().max())
    print(get_err_ratio(beta_grad, beta.grad), (beta_grad-beta.grad).abs().max())
    breakpoint()

    # q_grad = q.grad.clone()
    # v_grad = v.grad.clone()
    # k_grad = k.grad.clone()
    # beta_grad = beta.grad.clone()
    # scale = K**-0.5
    # BN = 64
    # BM = 128

    # # with torch.no_grad():
    # original_dtype = q.dtype
    # q2, k2, v2, beta2 = map(lambda x: x.to(torch.float32).detach().clone().requires_grad_(True), [q, k, v, beta])
    # q = q2.clone()
    # k = k2.clone()
    # v = v2.clone()
    # beta = beta2.clone()
    # b, h, l, d_k = q.shape
    # # q = q * scale

    # k_beta = k * beta[..., None]
    # # compute (I - tri(diag(beta) KK^T))^{-1}
    # q, k, v, k_beta = map(lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=BN), [q, k, v, k_beta])
    # mask = torch.triu(torch.ones(BN, BN, dtype=torch.bool, device=q.device), diagonal=0)
    # T = -(k_beta @ k.transpose(-1, -2)).masked_fill(mask, 0)
    # for i in range(1, BN):
    #     T[..., i, :i] = T[..., i, :i].clone() + (T[..., i, :, None].clone() * T[..., :, :i].clone()).sum(-2)
    # T = T + torch.eye(BN, dtype=q.dtype, device=q.device)
    # # T = T.clone().detach()

    # mask2 = torch.triu(torch.ones(BN, BN, dtype=torch.bool, device=q.device), diagonal=1)
    # A_local = (q @ k.transpose(-1, -2)).masked_fill(mask2, 0) @ T
    # # o_intra = A_local @ v

    # # apply cumprod transition matrices on k to the last position within the chunk
    # k = k - ((k @ k.transpose(-1, -2)).masked_fill(mask, 0) @ T).transpose(-1, -2) @ k_beta
    # # apply cumprod transition matrices on q to the first position within the chunk
    # q = q - A_local @ k_beta
    # A = torch.zeros(b, h, l, l, device=q.device)

    # q_origin, k, v, k_beta = map(lambda x: rearrange(x, 'b h n c d -> b h (n c) d'), [q, k, v, k_beta])

    # for i in range(0, l, BM):
    #     q_i = q_origin[:, :, i:i+BM].clone()
    #     # o_i = o_intra[:, :, i:i+BM]
    #     # intra block
    #     for j in range(i + BM - 2 * BN, i-BN, -BN):
    #         k_j = k[:, :, j:j+BN]
    #         A_ij = q_i @ k_j.transpose(-1, -2)
    #         mask = torch.arange(i, i+BM) >= (j + BN)
    #         A_ij = A_ij.masked_fill_(~mask[:, None].to(A_ij.device), 0)
    #         A[:, :, i:i+BM, j:j+BN] = A_ij
    #         q_i = q_i - A_ij @ k_beta[:, :, j:j+BN]
    #         # o_i += A_ij @ v[:, :, j:j+BN]
    #     # inter block
    #     for j in range(i - BN, -BN, -BN):
    #         k_j = k[:, :, j:j+BN]
    #         A_ij = q_i @ k_j.transpose(-1, -2)
    #         A[:, :, i:i+BM, j:j+BN] = A_ij
    #         q_i = q_i - A_ij @ k_beta[:, :, j:j+BN]
    #         # o_i += A_ij @ v[:, :, j:j+BN]

    # for i in range(0, l//BN):
    #     A[:, :, i*BN:i*BN+BN, i*BN:i*BN+BN] = A_local[:, :, i]
    
    # A = A.masked_fill_(~torch.tril(torch.ones(l, l, device=q.device, dtype=torch.bool)), float("-inf"))
    # o3 = (A.to(torch.float32) * scale).softmax(-1).to(v) @ v
    # o3.backward(do)
    # print(get_err_ratio(o3, o2))
    # print(get_err_ratio(o3, ref_o))
    # print(get_err_ratio(o2, ref_o))
    # # (q_grad - q_origin.grad)[:,:,64]
    # print(get_err_ratio(q_grad, q2.grad))
    # print(get_err_ratio(v_grad, v2.grad))
    # print(get_err_ratio(k_grad, k2.grad))
    # print(get_err_ratio(beta_grad, beta2.grad))
    # breakpoint()
