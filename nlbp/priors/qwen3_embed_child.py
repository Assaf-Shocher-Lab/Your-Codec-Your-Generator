"""Child process of nlbp/priors/text_embedding.py (run in the prior environment): the Qwen3TextEncoder steps of the T2I
prior's own conditioner: Qwen2Tokenizer (max_length, right padding, truncation), Qwen3Model in bf16, the last hidden
state. Saves {"y": [1, L, D], "valid": n_tokens, "attn": implementation}.
Usage: python qwen3_embed_child.py <qwen_dir> <prompt> <out.pt> [max_length]"""
import sys

import torch
from transformers import Qwen2Tokenizer, Qwen3Model


def main():
    qwen_dir, prompt, out = sys.argv[1], sys.argv[2], sys.argv[3]
    L = int(sys.argv[4]) if len(sys.argv) > 4 else 128
    dev = torch.device("cuda")
    tok = Qwen2Tokenizer.from_pretrained(qwen_dir, max_length=L, padding_side="right")
    ids = tok([prompt], truncation=True, max_length=L, padding="max_length", return_tensors="pt")
    y, used = None, None
    with torch.no_grad():
        for impl in ("sdpa", "eager"):
            m = Qwen3Model.from_pretrained(qwen_dir, attn_implementation=impl).to(torch.bfloat16).to(dev).eval()
            y = m(input_ids=ids.input_ids.to(dev), attention_mask=ids.attention_mask.to(dev))[0]
            del m
            torch.cuda.empty_cache()
            ok = bool(torch.isfinite(y).all())
            used = impl
            if ok:
                break
    if not torch.isfinite(y).all():
        sys.exit("Qwen3 embedding not finite")
    torch.save(dict(y=y.cpu(), valid=int(ids.attention_mask.sum()), attn=used), out)


if __name__ == "__main__":
    main()
