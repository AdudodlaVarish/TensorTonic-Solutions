import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_fwd_kernel(
    x_ptr, gamma_ptr, out_ptr,
    stride_x_row, stride_out_row,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    mask = tl.arange(0,BLOCK_SIZE) < N 
    x_ptrs = x_ptr + row * stride_x_row + tl.arange(0, BLOCK_SIZE)

    x = tl.load(x_ptrs, mask = mask, other = 0.0)
    squared_mean = tl.sum(x * x, axis = -1)

    gamma_ptrs = gamma_ptr + tl.arange(0, BLOCK_SIZE)
    gamma = tl.load(gamma_ptrs, mask = mask, other = 0.0)

    output = x / tl.sqrt((1 / N) * squared_mean + eps) * gamma

    out_ptrs = out_ptr + row * stride_out_row + tl.arange(0, BLOCK_SIZE)
    tl.store(out_ptrs, output, mask = mask)
    


def solve(x: torch.Tensor, gamma: torch.Tensor, out: torch.Tensor, eps: float) -> None:
    """Launch the RMSNorm forward kernel: one program per row."""
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    rmsnorm_fwd_kernel[grid](
        x, gamma, out,
        x.stride(0), out.stride(0),
        N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )