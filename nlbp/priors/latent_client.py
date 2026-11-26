"""The parent side of a latent-space prior (Z-Image, FLUX.2 [dev]) of the NLBP sampler (nlbp/sampler.nlbp_decode_latent):
the prior runs in a child process in the prior environment (NLBP_PRIOR_PY, its diffusers / transformers versions;
nlbp/priors/latent_server.py), on the same GPU, with a clean PYTHONPATH; the two talk over the child's stdin / stdout
(frame = 8-byte little-endian length + torch.save of a dict of CPU tensors). Calls: encode(x in [-1, 1]) -> z,
decode(z) -> x, velocity(z, s, prompt, cfg) -> v (flow convention z_s = (1 - s) z_0 + s eps, v = eps - z_0). The
parent keeps every NLBP step and the exact write."""
import io
import os
import struct
import subprocess
import sys
import time

import torch

PRIOR_PY = os.environ.get("NLBP_PRIOR_PY", sys.executable)
HERE = os.path.dirname(os.path.abspath(__file__))


class PriorClient:
    def __init__(self, prior, models_dir, extra_args=(), log_path=None, timeout_s=1800):
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP")}
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
                   TOKENIZERS_PARALLELISM="false")
        env.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
        self.log = open(log_path, "a") if log_path else sys.stderr
        srv = "latent_server.py"
        cmd = [PRIOR_PY, "-u", "-I", os.path.join(HERE, srv), "--prior", prior,
               "--models", models_dir] + list(extra_args)
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, env=env,
                                  cwd=HERE)
        self.prior = prior
        self.stats = dict(n_vel=0, vel_s=0.0, n_enc=0, enc_s=0.0, n_dec=0, dec_s=0.0, ipc_s=0.0)
        t = time.time()
        self.info = self._call(dict(cmd="hello"))

    def _send(self, obj):
        buf = io.BytesIO()
        torch.save(obj, buf)
        b = buf.getvalue()
        self.p.stdin.write(struct.pack("<Q", len(b)))
        self.p.stdin.write(b)
        self.p.stdin.flush()

    def _recv(self):
        h = self.p.stdout.read(8)
        if len(h) < 8:
            raise RuntimeError(f"prior server ({self.prior}) died (exit {self.p.poll()}); see its log")
        n = struct.unpack("<Q", h)[0]
        b = self.p.stdout.read(n)
        return torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)

    def _call(self, msg):
        self._send(msg)
        r = self._recv()
        if isinstance(r, dict) and r.get("error"):
            raise RuntimeError(f"prior server error: {r['error']}")
        return r

    def _t(self, x):
        return x.detach().float().cpu().contiguous()

    def encode(self, x):
        t = time.time()
        r = self._call(dict(cmd="encode", x=self._t(x)))
        self.stats["n_enc"] += 1; self.stats["enc_s"] += time.time() - t
        return r["z"].to(x.device)

    def decode(self, z):
        t = time.time()
        r = self._call(dict(cmd="decode", z=self._t(z)))
        self.stats["n_dec"] += 1; self.stats["dec_s"] += time.time() - t
        return r["x"].to(z.device)

    def velocity(self, z, s, prompt="empty", cfg=1.0):
        t = time.time()
        r = self._call(dict(cmd="velocity", z=self._t(z), s=float(s), prompt=prompt, cfg=float(cfg)))
        self.stats["n_vel"] += 1; self.stats["vel_s"] += time.time() - t
        return r["v"].to(z.device)

    def close(self):
        try:
            self._send(dict(cmd="quit"))
            self.p.wait(timeout=60)
        except Exception:
            self.p.kill()


def start_prior(prior, models_dir, log_path=None):
    """The child process of one latent prior ("zimage" | "flux2_dev", weights in models_dir/<prior>/) at the image's
    own resolution (upsample 1, as every reported configuration)."""
    client = PriorClient(prior, models_dir, log_path=log_path)
    client.upsample = 1
    return client
