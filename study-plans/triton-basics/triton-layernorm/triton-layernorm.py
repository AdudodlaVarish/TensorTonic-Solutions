import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_fwd_kernel(
    x_ptr, gamma_ptr, beta_ptr, out_ptr,
    stride_x_row, stride_out_row,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    mask = tl.arange(0, BLOCK_SIZE) < N
    x_ptrs = x_ptr + row * stride_x_row + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptrs, mask = mask)
    mean = tl.sum(x, axis = -1) / N
    diff = tl.where(mask, x - mean, 0)
    var = tl.sum(diff * diff, axis = -1) / N
    gamma_ptrs = gamma_ptr + tl.arange(0, BLOCK_SIZE)
    beta_ptrs = beta_ptr + tl.arange(0, BLOCK_SIZE)

    gamma = tl.load(gamma_ptrs, mask = mask)
    beta = tl.load(beta_ptrs, mask = mask)

    output = (x - mean) / tl.sqrt(var + eps) * gamma + beta
    out_ptrs = out_ptr + row * stride_out_row + tl.arange(0, BLOCK_SIZE)
    tl.store(out_ptrs, output, mask = mask)
    
    


def solve(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, out: torch.Tensor, eps: float) -> None:
    """Launch the LayerNorm forward kernel: one program per row."""
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    layernorm_fwd_kernel[grid](
        x, gamma, beta, out,
        x.stride(0), out.stride(0),
        N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )