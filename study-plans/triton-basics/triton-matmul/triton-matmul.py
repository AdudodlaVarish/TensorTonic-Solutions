import torch
import triton
import triton.language as tl


import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    m = tl.program_id(0)
    n = tl.program_id(1)

    a_start = a_ptr + m * BLOCK_M * stride_am 
    b_start = b_ptr + n * BLOCK_N * stride_bn
    k_offset = tl.arange(0, BLOCK_K)

    a_ptrs = a_start + tl.arange(0, BLOCK_M)[:, None] * stride_am + k_offset[None, :] * stride_ak 
    b_ptrs = b_start + k_offset[:, None] * stride_bk + tl.arange(0, BLOCK_N)[None, :] * stride_bn

    m_mask = m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None] < M
    n_mask = n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :] < N       

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype = tl.float32)

    for chunk in tl.range(0, (K + BLOCK_K - 1) // BLOCK_K):
        
        k_indices = k_offset + chunk * BLOCK_K
        k_mask = k_indices < K

        a_mask = m_mask & k_mask[None, :]
        b_mask = k_mask[:, None] & n_mask

        a = tl.load(a_ptrs, mask = a_mask, other = 0.0)
        b = tl.load(b_ptrs, mask = b_mask, other = 0.0)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

        acc = tl.dot(a, b, acc, input_precision = "ieee")

    c_start = c_ptr + m * BLOCK_M * stride_cm + n * BLOCK_N * stride_cn
    c_offset = tl.arange(0, BLOCK_M)[:, None] * stride_cm + tl.arange(0, BLOCK_N)[None, :] * stride_cn
    c_mask = m_mask & n_mask
    tl.store(c_start + c_offset, acc, mask = c_mask)
       


def solve(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor) -> None:
    """Launch matmul_kernel: out = A @ B."""
    M, K = A.shape
    K2, N = B.shape
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_kernel[grid](
        A, B, out,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )


def solve(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor) -> None:
    """Launch matmul_kernel: out = A @ B."""
    M, K = A.shape
    K2, N = B.shape
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_kernel[grid](
        A, B, out,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )