import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(x_ptr, out_ptr, x_row_stride, out_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):

    row = tl.program_id(0)
    offsets = row * x_row_stride + tl.arange(0, BLOCK_SIZE)
    mask = offsets < row * x_row_stride + n_cols
    x = tl.load(x_ptr + offsets, mask = mask, other = float("-inf"))

    max = tl.max(x)
    numerator = tl.exp(x - max)
    denominator = tl.sum(tl.exp(x - max))
    output = numerator / denominator 
    tl.store(out_ptr + offsets, output, mask = mask)

def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch softmax_kernel with one program per row."""
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    softmax_kernel[grid](
        x, out, x.stride(0), out.stride(0), N, BLOCK_SIZE=BLOCK_SIZE,
    )