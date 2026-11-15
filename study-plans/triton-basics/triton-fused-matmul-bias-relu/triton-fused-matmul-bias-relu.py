import torch
import triton
import triton.language as tl


@triton.jit
def fused_matmul_bias_relu_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    
    m_tile = tl.program_id(0)
    n_tile = tl.program_id(1)
    
    a_start = a_ptr + m_tile * BLOCK_M * stride_am
    b_start = b_ptr + n_tile * BLOCK_N * stride_bn

    a_offset = tl.arange(0, BLOCK_M)[:, None] * stride_am
    b_offset = tl.arange(0, BLOCK_N)[None, :] * stride_bn

    a_mask = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)[:, None] < M
    b_mask = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)[None, :] < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for chunk in tl.range(0, (BLOCK_K + K - 1) // BLOCK_K):
        
        k_indices = chunk * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = k_indices < K

        a_ptrs = a_start + a_offset + k_indices[None, :] * stride_ak
        b_ptrs = b_start + b_offset + k_indices[:, None] * stride_bk 

        a = tl.load(a_ptrs, mask = (a_mask & k_mask[None, :]), other = 0.0)
        b = tl.load(b_ptrs, mask = (b_mask & k_mask[:, None]), other = 0.0)

        acc = tl.dot(a,b,acc, input_precision = "ieee")

    bias_ptrs = bias_ptr + n_tile * BLOCK_N + tl.arange(0,BLOCK_N)
    bias_mask = n_tile * BLOCK_N + tl.arange(0,BLOCK_N) < N 
    bias = tl.load(bias_ptrs, mask = bias_mask, other = 0.0)

    acc = acc + bias[None, :]
    acc = tl.maximum(acc, 0)

    out_offset = m_tile * BLOCK_M * stride_cm + tl.arange(0, BLOCK_M)[:, None] * stride_cm + n_tile * BLOCK_N * stride_cn + tl.arange(0, BLOCK_M)[None] * stride_cn
    out_mask = (m_tile * BLOCK_M + tl.arange(0, BLOCK_M)[:, None] < M) & (n_tile * BLOCK_N + tl.arange(0, BLOCK_N)[None, :] < N)
    tl.store(c_ptr + out_offset, acc, mask = out_mask)
    
def solve(A: torch.Tensor, B: torch.Tensor, bias: torch.Tensor, out: torch.Tensor) -> None:
    """Launch fused kernel: out = relu(A @ B + bias)."""
    M, K = A.shape
    K2, N = B.shape
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    fused_matmul_bias_relu_kernel[grid](
        A, B, bias, out,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )