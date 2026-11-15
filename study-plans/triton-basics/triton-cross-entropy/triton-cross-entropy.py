import torch
import triton
import triton.language as tl


@triton.jit
def cross_entropy_kernel(
    logits_ptr, target_ptr, loss_out_ptr,
    stride_logits_row,
    B, C,
    BLOCK_SIZE: tl.constexpr,
):
    batch = tl.program_id(0)
    logits_offsets = logits_ptr + batch * stride_logits_row + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < C

    logits = tl.load(logits_offsets, mask = mask, other = float("-inf"))
    max = tl.max(logits)
    logit_sum = tl.sum(tl.exp(logits - max), axis = -1)

    correct_idx = tl.load(target_ptr + batch)
    correct = tl.load(logits_ptr +  batch * stride_logits_row + correct_idx)
    
    final = tl.log(logit_sum) - correct + max
    tl.atomic_add(loss_out_ptr, final)
    


def solve(logits: torch.Tensor, target: torch.Tensor, loss_out: torch.Tensor) -> None:
    """Launch the cross-entropy kernel: one program per row, atomic accumulate, then divide by B."""
    B, C = logits.shape
    BLOCK_SIZE = triton.next_power_of_2(C)
    loss_out.zero_()
    grid = (B,)
    cross_entropy_kernel[grid](
        logits, target, loss_out,
        logits.stride(0),
        B, C,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    loss_out.div_(B)