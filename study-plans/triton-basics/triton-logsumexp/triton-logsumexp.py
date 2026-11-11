import torch
import triton
import triton.language as tl


@triton.jit
def logsumexp_kernel(x_ptr, out_ptr, x_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):

    pid = tl.program_id(0)
    offset = pid * x_row_stride + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < n_cols 

    x = tl.load(x_ptr + offset, 
                mask = mask, 
                other = float("-inf"))
    max = tl.max(x)

    output = max + tl.log(tl.sum(tl.exp(x - max)))
    tl.store(out_ptr + pid, output)

def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch logsumexp_kernel with one program per row."""
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    logsumexp_kernel[grid](
        x, out, x.stride(0), N, BLOCK_SIZE=BLOCK_SIZE,
    )