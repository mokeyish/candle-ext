import torch
import safetensors.torch as safetensors

q = torch.randn((1, 3, 2, 4))
k = torch.randn((1, 3, 3, 4))
v = torch.randn((1, 3, 3, 4))
m = torch.tril(torch.ones(q.size(-2), k.size(-2), dtype=torch.uint8))
o = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=m.bool())


safetensors.save_file({
    "q": q,
    "k": k,
    "v": v,
    "m": m,
    "o": o
}, "scaled_dot_product_attention-qkvmo.safetensors")

print(o)