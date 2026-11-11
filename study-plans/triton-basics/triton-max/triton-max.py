import torch
import triton
import triton.language as tl

@triton.jit
def max_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    
    offset = tl.arange(0, BLOCK_SIZE)
    mask = offset < n
    
    x = tl.load(x_ptr + offset, 
                mask = mask, 
                other = -float('inf'))
    max = tl.max(x)
    tl.store(out_ptr, max)

    

def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch max_kernel on the provided tensor with a single-program reduction."""
    n = x.numel()
    BLOCK_SIZE = triton.next_power_of_2(n)
    grid = (1,)
    max_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)