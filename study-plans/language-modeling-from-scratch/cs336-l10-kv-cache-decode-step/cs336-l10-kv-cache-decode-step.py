import math
import torch

def kv_cache_decode_step(
    query: torch.Tensor, new_key: torch.Tensor, new_value: torch.Tensor,
    key_cache: torch.Tensor, value_cache: torch.Tensor,
    num_query_heads: int, num_kv_heads: int,
) -> dict:
    B, Hq, D = query.shape
    G = num_query_heads // num_kv_heads

    new_key_cache = torch.cat(
        (key_cache, new_key.unsqueeze(2)), dim=2
    )
    new_value_cache = torch.cat(
        (value_cache, new_value.unsqueeze(2)), dim=2
    )

    q = query.reshape(B, num_kv_heads, G, D).float()

    k = new_key_cache.float()
    v = new_value_cache.float()

    # (B, Hkv, G, D) @ (B, Hkv, D, S+1) -> (B, Hkv, G, S+1)
    scores = torch.matmul(q, k.transpose(-1, -2))
    scores = scores / math.sqrt(D)

    attn = torch.softmax(scores, dim=-1)

    # (B, Hkv, G, S+1) @ (B, Hkv, S+1, D) -> (B, Hkv, G, D)
    output = torch.matmul(attn, v)
    output = output.reshape(B, Hq, D).to(dtype=query.dtype)

    return {
        "output": output,
        "new_key_cache": new_key_cache,
        "new_value_cache": new_value_cache,
    }