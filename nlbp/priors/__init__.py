"""Clean-estimate adapters of the frozen priors. A prior is a callable prior(x_s, s) -> v in the flow convention
x_s = (1 - s) x_0 + s eps (images in [-1, 1]), with the clean estimate x_hat_0 = x_s - s v.

| prior (paper) | runs | adapter | weights |
|---|---|---|---|
| ffhq_10m (FFHQ 256 DDPM) | ELIC on FFHQ | ffhq_dps.py (+ ddpm.py) | external, third_party/xu/UPSTREAM.md |
| PixNerd-XXL/16 T2I (empty prompt) | HPCM on Kodak | pixnerd_t2i.py (+ tiles.py, text_embedding.py, qwen3_embed_child.py) | external network, third_party/nlbp_nets/UPSTREAM.md |
| DeCo-512 (null class) | AEIC on Kodak (controls) | deco512.py | external network, third_party/nlbp_nets/UPSTREAM.md |
| Z-Image, FLUX.2 [dev] (latent) | HPCM / AEIC on Kodak | latent_client.py (parent) + latent_server.py (child process in the prior environment), used by nlbp/sampler.nlbp_decode_latent | Z-Image public; FLUX.2 [dev] gated (licence acceptance on its model page) |
"""
