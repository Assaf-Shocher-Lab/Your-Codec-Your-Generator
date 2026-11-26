"""Child process of nlbp/priors/latent_client.py, run only in the prior environment (python -I latent_server.py --prior
zimage | flux2_dev --models <dir>). One convention for both priors: z_s = (1 - s) z_0 + s eps, v(z_s, s) = eps - z_0;
prompts "empty" ("") and "generic" ("a high-quality photograph") are embedded at load.
  zimage     Z-Image (diffusers ZImagePipeline, bf16 transformer, float32 VAE); latent = (VAE mode - shift) * scale;
             cfg c != 1 with a prompt: v = v_empty + c (v_prompt - v_empty), else one pass
  flux2_dev  FLUX.2 [dev] (components loaded separately: Flux2Transformer2DModel bf16, AutoencoderKLFlux2 float32);
             latent = the batch-norm-normalised patchified VAE latent; guidance-distilled: one pass with the embedded
             guidance FD_GUIDANCE (default 4.0); the Mistral-3 text encoder embeds the prompts once
             (cached in NLBP_WORK/cache/fd_emb.pt) and is freed before the transformer loads."""
import argparse
import io
import os
import struct
import sys
import time
import traceback

import torch

OUT = os.fdopen(os.dup(1), "wb")          # replies go to the real stdout; prints go to stderr
os.dup2(2, 1)
sys.stdout = sys.stderr
INP = os.fdopen(0, "rb")

GENERIC = "a high-quality photograph"
DEFAULT_CFG = {"zimage": 6.0, "flux2_dev": 1.0}


def send(obj):
    buf = io.BytesIO()
    torch.save(obj, buf)
    b = buf.getvalue()
    OUT.write(struct.pack("<Q", len(b)))
    OUT.write(b)
    OUT.flush()


def recv():
    h = INP.read(8)
    if len(h) < 8:
        return None
    n = struct.unpack("<Q", h)[0]
    return torch.load(io.BytesIO(INP.read(n)), map_location="cpu", weights_only=False)


class Flux2Dev:
    def __init__(self, root, dev):
        import gc
        from diffusers import AutoencoderKLFlux2, Flux2Transformer2DModel
        from diffusers.pipelines.flux2.pipeline_flux2 import Flux2Pipeline
        self.FP = Flux2Pipeline
        t = time.time()
        self.dev = dev
        self.g = float(os.environ.get("FD_GUIDANCE", "4.0"))
        cache = os.path.join(os.environ.get("NLBP_WORK", "."), "cache", "fd_emb.pt")
        if os.path.exists(cache):
            raw = torch.load(cache, map_location="cpu")
        else:
            from transformers import AutoProcessor, Mistral3ForConditionalGeneration
            from diffusers.pipelines.flux2.system_messages import SYSTEM_MESSAGE
            tok = AutoProcessor.from_pretrained(os.path.join(root, "tokenizer"), local_files_only=True)
            te = Mistral3ForConditionalGeneration.from_pretrained(os.path.join(root, "text_encoder"),
                                                                  torch_dtype=torch.bfloat16,
                                                                  local_files_only=True).to(dev).eval()
            raw = {}
            with torch.no_grad():
                for key, txt in (("empty", ""), ("generic", GENERIC)):
                    pe = Flux2Pipeline._get_mistral_3_small_prompt_embeds(
                        text_encoder=te, tokenizer=tok, prompt=txt, dtype=torch.bfloat16, device=dev,
                        max_sequence_length=512, system_message=SYSTEM_MESSAGE, hidden_states_layers=(10, 20, 30))
                    raw[key] = pe.cpu()
            del te, tok
            gc.collect(); torch.cuda.empty_cache()
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            torch.save(raw, cache + ".tmp"); os.replace(cache + ".tmp", cache)
        self.emb = {}
        for key, pe in raw.items():
            pe = pe.to(dev, torch.bfloat16)
            self.emb[key] = (pe, Flux2Pipeline._prepare_text_ids(pe).to(dev))
        self.tr = Flux2Transformer2DModel.from_pretrained(os.path.join(root, "transformer"), torch_dtype=torch.bfloat16,
                                                          local_files_only=True).to(dev).eval()
        self.vae = AutoencoderKLFlux2.from_pretrained(os.path.join(root, "vae"), torch_dtype=torch.float32,
                                                      local_files_only=True).to(dev).eval()
        self.bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).float()
        self.bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1).float() + self.vae.config.batch_norm_eps)

    def add_prompt(self, key, text):
        if key not in self.emb:
            raise RuntimeError(f"flux2_dev: text encoder freed; only cached prompts {list(self.emb)} (asked {key!r})")

    @torch.no_grad()
    def encode(self, x):
        lat = self.vae.encode(x.to(self.dev, torch.float32)).latent_dist.mode()
        lat = self.FP._patchify_latents(lat)
        return ((lat - self.bn_mean) / self.bn_std).float()

    @torch.no_grad()
    def decode(self, z):
        z = z.to(self.dev, torch.float32)
        lat = self.FP._unpatchify_latents(z * self.bn_std + self.bn_mean)
        return self.vae.decode(lat, return_dict=False)[0].float().clamp(-1, 1)

    @torch.no_grad()
    def velocity(self, z, s, prompt, cfg):
        z = z.to(self.dev, torch.float32)
        key = "empty" if prompt == "empty" else prompt
        pe, tids = self.emb[key]
        B, C, H, W = z.shape
        ids = self.FP._prepare_latent_ids(z).to(self.dev)
        hs = self.FP._pack_latents(z).to(torch.bfloat16)
        g = float(cfg) if float(cfg) != 1.0 else self.g
        out = self.tr(hidden_states=hs, timestep=torch.full((B,), float(s), device=self.dev, dtype=torch.bfloat16),
                      guidance=torch.full((B,), g, device=self.dev, dtype=torch.float32),
                      encoder_hidden_states=pe.expand(B, -1, -1), txt_ids=tids.expand(B, -1, -1), img_ids=ids,
                      return_dict=False)[0]
        out = out[:, :hs.size(1)].float()
        return out.permute(0, 2, 1).reshape(B, C, H, W)


class ZImage:
    def __init__(self, root, dev):
        from diffusers import ZImagePipeline, AutoencoderKL
        t = time.time()
        self.pipe = ZImagePipeline.from_pretrained(root, torch_dtype=torch.bfloat16, local_files_only=True)
        self.pipe.vae = AutoencoderKL.from_pretrained(os.path.join(root, "vae"), torch_dtype=torch.float32,
                                                      local_files_only=True)
        self.pipe.to(dev)
        self.dev, self.P = dev, self.pipe
        self.tr = self.pipe.transformer
        self.sf, self.sh = float(self.P.vae.config.scaling_factor), float(self.P.vae.config.shift_factor)
        self.emb = {}
        for key, txt in (("empty", ""), ("generic", GENERIC)):
            self.emb[key] = self._embed(txt)

    @torch.no_grad()
    def _embed(self, text):
        e = self.P._encode_prompt(prompt=[text], device=self.dev)
        return [t_.to(torch.bfloat16) for t_ in e]

    def add_prompt(self, key, text):
        if key not in self.emb:
            self.emb[key] = self._embed(text)

    @torch.no_grad()
    def encode(self, x):
        lat = self.P.vae.encode(x.to(self.dev, torch.float32)).latent_dist.mode()
        return ((lat - self.sh) * self.sf).float()

    @torch.no_grad()
    def decode(self, z):
        lat = z.to(self.dev, torch.float32) / self.sf + self.sh
        return self.P.vae.decode(lat, return_dict=False)[0].float().clamp(-1, 1)

    def _out(self, z, s, embs):
        B = len(embs)
        inp = z.to(torch.bfloat16).repeat(B, 1, 1, 1).unsqueeze(2)
        ts = torch.full((B,), 1.0 - float(s), device=self.dev)
        out = self.tr(list(inp.unbind(dim=0)), ts, embs, return_dict=False)[0]
        return [-(o.float().squeeze(1)) for o in out]

    @torch.no_grad()
    def velocity(self, z, s, prompt, cfg):
        z = z.to(self.dev, torch.float32)
        key = "empty" if prompt == "empty" else prompt
        if cfg != 1.0 and key != "empty":
            vp, vn = self._out(z, s, self.emb[key] + self.emb["empty"])
            return (vn + cfg * (vp - vn))[None]
        return self._out(z, s, self.emb[key])[0][None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", required=True, choices=["zimage", "flux2_dev"])
    ap.add_argument("--models", required=True)
    ap.add_argument("--upsample", type=int, default=1)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device("cuda")
    root = os.path.join(a.models, a.prior)
    if a.prior == "flux2_dev":
        M = Flux2Dev(root, dev)
    else:
        M = ZImage(root, dev)
    print(f"[server] {a.prior} mem {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    while True:
        msg = recv()
        if msg is None or msg.get("cmd") == "quit":
            break
        try:
            c = msg["cmd"]
            if c == "hello":
                send(dict(prior=a.prior, default_cfg=DEFAULT_CFG[a.prior], upsample=a.upsample,
                          mem_gib=round(torch.cuda.memory_allocated() / 2**30, 2), torch=torch.__version__))
            elif c == "encode":
                send(dict(z=M.encode(msg["x"]).cpu()))
            elif c == "decode":
                send(dict(x=M.decode(msg["z"]).cpu()))
            elif c == "velocity":
                v = M.velocity(msg["z"], msg["s"], msg["prompt"], msg["cfg"])
                assert v.shape == msg["z"].shape and torch.isfinite(v).all(), (tuple(v.shape), tuple(msg["z"].shape))
                send(dict(v=v.cpu()))
            elif c == "add_prompt":
                M.add_prompt(msg["key"], msg["text"])
                send(dict(ok=True))
            else:
                send(dict(error=f"unknown cmd {c}"))
        except Exception as e:
            traceback.print_exc()
            send(dict(error=f"{type(e).__name__}: {e}"))
    print("[server] bye", flush=True)


if __name__ == "__main__":
    main()
