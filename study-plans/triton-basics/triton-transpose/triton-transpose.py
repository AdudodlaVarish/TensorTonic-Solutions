import torch
import triton
import triton.language as tl


@triton.jit
def transpose_kernel(
    a_ptr, out_ptr,
    M, N,
    stride_am, stride_an,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    m_tile = tl.program_id(0)
    n_tile = tl.program_id(1)

    start = a_ptr + m_tile * BLOCK_M * stride_am + n_tile * BLOCK_N * stride_an
    offset = tl.arange(0, BLOCK_M)[:, None] * stride_am + tl.arange(0, BLOCK_N)[None, :] * stride_an 
    mask = (m_tile * BLOCK_M + tl.arange(0, BLOCK_M)[:, None] < M) & (n_tile * BLOCK_N + tl.arange(0, BLOCK_N)[None, :] < N)
    tile = tl.load(start + offset, mask = mask, other = 0.0)

    tile_transpose = tile.T

    out_start = out_ptr + m_tile * BLOCK_M * stride_on + n_tile * BLOCK_N * stride_om
    out_offset = tl.arange(0, BLOCK_N)[:, None] * stride_om + tl.arange(0, BLOCK_M)[None, :] * stride_on
    out_mask = (n_tile * BLOCK_N + tl.arange(0, BLOCK_N)[:, None] < N) & (m_tile * BLOCK_M + tl.arange(0, BLOCK_M)[None, :] < M)
    tl.store(out_start + out_offset, tile_transpose, mask = out_mask)
    


def solve(A: torch.Tensor, out: torch.Tensor) -> None:
    """Launch transpose_kernel: out[j, i] = A[i, j]."""
    M, N = A.shape
    BLOCK_M = 32
    BLOCK_N = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    transpose_kernel[grid](
        A, out,
        M, N,
        A.stride(0), A.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )