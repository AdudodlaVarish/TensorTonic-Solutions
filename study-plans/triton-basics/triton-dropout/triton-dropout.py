import torch
import triton
import triton.language as tl


@triton.jit
def dropout_kernel(
    x_ptr, mask_ptr, out_ptr,
    n, p,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    valid_mask = offs < n
    
    x = tl.load(x_ptr + offs, mask=valid_mask, other=0.0)
    m = tl.load(mask_ptr + offs, mask=valid_mask, other=0.0)
    
    scale = 1.0 / (1.0 - p)
    
    out = x * m * scale
    
    tl.store(out_ptr + offs, out, mask=valid_mask)


def solve(x: torch.Tensor, mask: torch.Tensor, out: torch.Tensor, p: float) -> None:
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    dropout_kernel[grid](
        x, mask, out,
        n, p,
        BLOCK_SIZE=BLOCK_SIZE,
    )