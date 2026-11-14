import torch
import triton
import triton.language as tl


@triton.jit
def gemv_kernel(
    a_ptr, x_ptr, out_ptr,
    M, N,
    stride_am, stride_an,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    m_tile = tl.program_id(0)
    start_row = m_tile * BLOCK_M
    acc = tl.zeros((BLOCK_M,), dtype = tl.float32)
    m_mask = (m_tile * BLOCK_M + tl.arange(0, BLOCK_M)[:, None] < M)
    
    for n_chunk in tl.range(0, (BLOCK_N + N - 1) // BLOCK_N):
        
        n_indices = tl.arange(0, BLOCK_N)[None, :] + n_chunk * BLOCK_N
        n_mask = n_indices < N
        mask = m_mask & n_mask 
        a = tl.load(a_ptr + (start_row + tl.arange(0, BLOCK_M)[:, None]) * stride_am + n_indices * stride_an, mask = mask, other = 0.0)
        x = tl.load(x_ptr + n_chunk * BLOCK_N + tl.arange(0, BLOCK_N), mask = (n_chunk * BLOCK_N + tl.arange(0, BLOCK_N) < N), other = 0.0)

        acc += tl.sum(a * x, axis = -1)

    tl.store(out_ptr + start_row + tl.arange(0, BLOCK_M), acc, mask = tl.arange(0, BLOCK_M) + start_row < M)

    
    
def solve(A: torch.Tensor, x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch gemv_kernel: out = A @ x."""
    M, N = A.shape
    BLOCK_M = 32
    BLOCK_N = 64
    grid = (triton.cdiv(M, BLOCK_M),)
    gemv_kernel[grid](
        A, x, out,
        M, N,
        A.stride(0), A.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )