import torch
import triton
import triton.language as tl


@triton.jit
def sum_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    valid_mask = offs < n
    
    x = tl.load(x_ptr + offs, mask=valid_mask, other=0.0)
    
    block_sum = tl.sum(x, axis=0)
    
    tl.atomic_add(out_ptr, block_sum)


def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    n = x.numel()
    out.zero_()
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    sum_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)