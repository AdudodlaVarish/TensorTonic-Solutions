import torch
import triton
import triton.language as tl


@triton.jit
def dropout_kernel(
    x_ptr, mask_ptr, out_ptr,
    n, p,
    BLOCK_SIZE: tl.constexpr,
):
    chunk = tl.program_id(0)
    offset = chunk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x_ptrs = x_ptr + offset 
    mask_ptrs = mask_ptr + offset

    mask = chunk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) < n
    
    x = tl.load(x_ptrs, mask = mask, other = 0.0)
    dropout_mask = tl.load(mask_ptrs, mask = mask, other = 0.0)

    output = (x * dropout_mask) * (1 / (1-p))
    out_ptrs = out_ptr + offset

    tl.store(out_ptrs, output, mask = mask)
    


def solve(x: torch.Tensor, mask: torch.Tensor, out: torch.Tensor, p: float) -> None:
    """Launch the dropout kernel: 1D grid over the input vector."""
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    dropout_kernel[grid](
        x, mask, out,
        n, p,
        BLOCK_SIZE=BLOCK_SIZE,
    )