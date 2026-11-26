"""Run recompression chains of one reported variant (configs/<variant>.yaml).

    python scripts/run_chain.py --config configs/hpcm_kodak_8bit_stabilized.yaml --rate 0.013 \
        --images kodim07 --rounds 3 --data data/kodak --out runs/hpcm_8bit

8-bit protocol (file chain, adapted_codec/chain.py): per (rate, image) rows.csv, the compressed file of every round
(F/g###.bin), the stored image (png/), the symbol md5 (sym/).
Float / clipped-float protocols (in memory, adapted_codec/memory_chain.py): rows.csv, F/g###.bin, recon/g###.png.
AEIC (adapters/aeic/chain.py): in-memory chains (float, clipped float, 8-bit in memory) and the 8-bit canvas file chain
of the high-resolution sets; one rate per process (AEIC models share the UNet).
Weights and data are not part of the repository (README, "Weights and data")."""
import argparse
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from adapted_codec.settings import S  # noqa: E402


def weights_of(cfg, rate):
    """The weights file of a rate: a path pattern with {} for the rate, or a table rate -> path."""
    w = cfg["weights"]
    return os.path.join(ROOT, w[str(rate)] if isinstance(w, dict) else w)


def build_backend(cfg, rate):
    if cfg["codec"] == "hpcm":
        from adapters.hpcm.adapter import HpcmBackend
        return HpcmBackend(rate, os.path.join(ROOT, cfg["pca"]), float(cfg["gain"]), weights_of(cfg, rate))
    if cfg["codec"] == "elic":
        from adapters.elic.backend import ElicBackend
        return ElicBackend(rate, os.path.join(ROOT, cfg["pca"]), float(cfg["gain"]), weights_of(cfg, rate))
    raise SystemExit(f"codec {cfg['codec']!r}: no backend")


def build_file_chain_adapter(cfg, rate, method):
    if cfg["codec"] == "hpcm":
        from adapters.hpcm.adapter import HpcmAdapter
        return HpcmAdapter(build_backend(cfg, rate), method or cfg["method"])
    if cfg["codec"] == "elic":
        from adapters.elic.eight_bit import ElicFileAdapter
        return ElicFileAdapter(build_backend(cfg, rate), method or cfg["method"])
    pca, c = os.path.join(ROOT, cfg["pca"]), float(cfg["gain"])
    if cfg["codec"] == "hyperprior":
        from adapters.hyperprior.adapter import HyperpriorAdapter, HyperpriorBackend
        return HyperpriorAdapter(HyperpriorBackend(rate, pca, c), method or cfg["method"])
    if cfg["codec"] == "mean_scale":
        from adapters.mean_scale.adapter import MeanScaleAdapter, MeanScaleBackend
        return MeanScaleAdapter(MeanScaleBackend(rate, pca, c), method or cfg["method"])
    if cfg["codec"] == "tcm":
        from adapters.tcm.adapter import TcmAdapter, TcmBackend
        return TcmAdapter(TcmBackend(rate, pca, c, weights_of(cfg, rate)), method or cfg["method"])
    if cfg["codec"] == "msillm":
        from adapters.msillm.adapter import MsillmAdapter, MsillmBackend
        return MsillmAdapter(MsillmBackend(rate, pca, c, weights_of(cfg, rate)), method or cfg["method"])
    raise SystemExit(f"codec {cfg['codec']!r}: no file-chain adapter")


def protocol_decoder(protocol):
    if protocol == "float":
        from adapted_codec.protocols.float_chain import decode_float
        return decode_float
    if protocol == "clipped_float":
        from adapted_codec.protocols.clipped_float.range_repair import decode_clipped
        return decode_clipped
    raise SystemExit(f"protocol {protocol!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--rate", required=True, help="rate name as in the config (e.g. 0.013)")
    ap.add_argument("--images", required=True, help="comma-separated image stems")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--data", required=True, help="folder with <stem>.png")
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", default=None, help="override the config's method (e.g. the plain codec)")
    ap.add_argument("--metrics", default="psnr,ms_ssim,lpips,dists", help="in-memory chains: metrics per round")
    ap.add_argument("--save-rounds", default="1,2,5,10,20,30,40,50", help="in-memory chains: rounds saved as PNG16")
    a = ap.parse_args()
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    if a.rate not in [str(r) for r in cfg["rates"]]:
        raise SystemExit(f"rate {a.rate} not in {cfg['rates']}")
    S.load(cfg.get("settings"))
    stems = a.images.split(",")
    if cfg["codec"] == "aeic":
        from adapters.aeic.chain import AeicBackend, run_canvas_chain, run_memory_chain
        backend = AeicBackend(cfg, a.rate, ROOT)
        method = a.method or cfg["method"]
        for stem in stems:
            out = os.path.join(a.out, f"aeic_{method}", a.rate, stem)
            png = os.path.join(a.data, stem + ".png")
            if cfg["protocol"] == "canvas_file_chain":
                run_canvas_chain(backend, png, out, stem, a.rounds, a.rate, method=method, log=print)
            else:
                run_memory_chain(backend, png, out, stem, a.rounds, a.rate, cfg["protocol"],
                                 canvas=bool(cfg.get("canvas", False)),
                                 freeze_from=int(cfg.get("colour_freeze_from", 0)), log=print)
        return
    if cfg["protocol"] == "file_chain":
        from adapted_codec.chain import run_chain
        adapter = build_file_chain_adapter(cfg, a.rate, a.method)
        for stem in stems:
            out = os.path.join(a.out, adapter.name, a.rate, stem)
            run_chain(adapter, os.path.join(a.data, stem + ".png"), out, stem, a.rounds, save_png=None, log=print)
        return
    from evaluation.metrics import build
    if cfg["codec"] == "glc":
        from adapters.glc.chain import GlcBackend, run_glc_chain
        backend = GlcBackend(a.rate, os.path.join(ROOT, cfg["pca"]), float(cfg["gain"]), weights_of(cfg, a.rate))
        metrics = build([m for m in a.metrics.split(",") if m and m != "psnr"], backend.dev)
        method = a.method or cfg["method"]
        save = {int(r) for r in a.save_rounds.split(",") if r}
        for stem in stems:
            out = os.path.join(a.out, f"glc_{method}", a.rate, stem)
            run_glc_chain(backend, os.path.join(a.data, stem + ".png"), out, stem, a.rounds, method, metrics,
                          save_rounds=save, log=print)
        return
    from adapted_codec.memory_chain import run_memory_chain
    decode = protocol_decoder(cfg["protocol"])
    backend = build_backend(cfg, a.rate)
    metrics = build([m for m in a.metrics.split(",") if m], backend.dev)
    method = a.method or cfg["method"]
    save = {int(r) for r in a.save_rounds.split(",") if r}
    for stem in stems:
        out = os.path.join(a.out, f"{cfg['codec']}_{method}", a.rate, stem)
        run_memory_chain(backend, decode, os.path.join(a.data, stem + ".png"), out, stem, a.rounds, method, metrics,
                         cfg["protocol"], save_rounds=save, log=print)


if __name__ == "__main__":
    main()
