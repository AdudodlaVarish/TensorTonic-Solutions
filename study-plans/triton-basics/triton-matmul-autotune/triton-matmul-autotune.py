import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_autotune_kernel(
    a_ptr, b_ptr, c_ptr,
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
    k_offset = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype = tl.float32)
    
    for k_chunk in tl.range(0, K, BLOCK_K):

        k_indices = k_offset + k_chunk 
        
        a_ptrs = a_start + tl.arange(0, BLOCK_M)[:, None] * stride_am + k_indices[None, :] * stride_ak
        b_ptrs = b_start + k_indices[:, None] * stride_bk + tl.arange(0, BLOCK_N)[None, :] * stride_bn

        a_mask = (m_tile * BLOCK_M + tl.arange(0, BLOCK_M)[:, None] < M) & (k_indices[None, :] < K)
        b_mask = (k_indices[:, None] < K) & (n_tile * BLOCK_N + tl.arange(0, BLOCK_N)[None, :] < N)

        a = tl.load(a_ptrs, mask = a_mask, other = 0.0)
        b = tl.load(b_ptrs, mask = b_mask, other = 0.0)

        acc = tl.dot(a,b,acc, input_precision="ieee")

    out_start = c_ptr + m_tile * BLOCK_M * stride_cm + n_tile * BLOCK_N * stride_cn
    out_offset = tl.arange(0, BLOCK_M)[:, None] * stride_cm + tl.arange(0, BLOCK_N)[None, :] * stride_cn
    out_mask = (m_tile * BLOCK_M + tl.arange(0, BLOCK_M)[:, None] < M) & (n_tile * BLOCK_N + tl.arange(0, BLOCK_N)[None, :] < N)
    tl.store(out_start + out_offset, acc, mask = out_mask)
        
def solve(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor) -> None:
    """Launch the autotuned matmul kernel: out = A @ B."""
    M, K = A.shape
    K2, N = B.shape
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))
    matmul_autotune_kernel[grid](
        A, B, out,
        M=M, N=N, K=K,
        stride_am=A.stride(0), stride_ak=A.stride(1),
        stride_bk=B.stride(0), stride_bn=B.stride(1),
        stride_cm=out.stride(0), stride_cn=out.stride(1),
    )