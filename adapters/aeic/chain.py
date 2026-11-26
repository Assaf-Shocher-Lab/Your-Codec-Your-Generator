"""AEIC recompression chains (App. A.2): the in-memory chain (Kodak float / clipped float / 8-bit, DIV2K float with the
padded canvas and frozen colour side information) and the high-resolution 8-bit canvas file chain (DIV2K, CLIC 2K).

In-memory chain, round g = 1 .. G (x_0 = the original, AEIC's transform to [-1, 1]; nothing is written between rounds
except the files that are checked):
  record   s_g = AdaptedAeic.encode(pad64(x_{g-1}), g): round 1 AEIC's own record Q(g_a(x)), rounds >= 2 Q(E'(x))
  side     AEIC's 16-bit colour statistics of the crop (images > 768^2); with freeze_from = n, rounds > n send round n's
  decode   x_raw = AEIC's decoder output for y_hat; y* = select_target; x_g = consistent_decode(...) then the storage
           S of the chain format (none: float; clamp: clipped float; png8: 8-bit, S_8 in memory)
  file     AEIC's rANS file of (s_g, side), written to F/g###.bin and decoded back (roundtrip = 1: same record)
  next     the crop x_g (Kodak), or the padded canvas (canvas = true: DIV2K float)
Canvas file chain (8-bit, metadata preserving), round g:
  g = 1: the original PNG, AEIC's padding to a multiple of 64, AEIC's encoder, the colour statistics of the original;
  g >= 2: the canvas PNG of round g - 1 and its sidecar (h, w, the six colour integers of round 1), encoded as read.
  F_g (header + rANS stream + colour integers) is written; a fresh decode reads only F_g; the canvas is stored as an
  8-bit PNG (S_8) and read back; the h x w crop is the displayed image. With escalation_gamma_z set, a closed-loop write
  that does not pass is repeated once with that wider hyper guard (EscalatedAeic; AEIC DIV2K).
Outputs per chain: rows.csv, F/g###.bin (+ png/g###.png and meta/g###.bin on the canvas file chain), DONE."""
import csv
import hashlib
import os
import struct

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from adapted_codec.protocols.storage import store11
from adapted_codec.settings import S
from adapters.aeic.codec import (AdaptedAeic, AeicSettings, Pair, colour_side_info, crop_and_colour_fix,
                                 decode_symbols, encode_pair, pad64)
from adapters.aeic.file_format import HEADER, Coder
from adapters.aeic.model import load_aeic, load_wrapped_encoder
from adapters.aeic.upstream_copies import aeic_transform, preprocess_image
from evaluation.metrics import PAIRED, PairedMetrics

META = ">IIB6H"                      # canvas sidecar: h, w, has_side, six colour integers (21 bytes)


def md5(*chunks):
    m = hashlib.md5()
    for c in chunks:
        m.update(c)
    return m.hexdigest()


def symbols_md5(z_hat, y_hat):
    return md5(z_hat.float().cpu().numpy().tobytes(), y_hat.float().cpu().numpy().tobytes())


class EscalatedAeic(AdaptedAeic):
    """Escalation of the closed-loop write (AEIC DIV2K, App. B.3): if the write at the preset guard does not pass, the
    same write is repeated from the same start with the wider hyper guard gamma_z = S.escalation_gamma_z; the passing
    one, else the one with fewer mismatched symbols (tie: the first), is returned. A write that passes first is returned
    unchanged."""

    def corrected_write(self, xp0, pair, yt):
        xq = super().corrected_write(xp0, pair, yt)
        first = dict(self.last_write)
        if first["ok"] == 1 or not S.escalation_gamma_z:
            return xq
        gz0 = S.gamma_z
        try:
            S.gamma_z = float(S.escalation_gamma_z)
            xq2 = super().corrected_write(xp0, pair, yt)
        finally:
            S.gamma_z = gz0
        second = dict(self.last_write)
        use2 = second["ok"] == 1 or second["best_bad"] < first["best_bad"]
        self.last_write = second if use2 else first
        return xq2 if use2 else xq


class AeicBackend:
    """One AEIC-ME rate: model, wrapped encoder E', decoder settings, file coder and the paired metrics."""

    def __init__(self, cfg, rate, root):
        def path(p):
            return p if os.path.isabs(p) else os.path.join(root, p)
        c = cfg["codec_model"]
        self.net = load_aeic(path(cfg["weights"][rate]), path(cfg["sd_turbo"]), path(cfg["vae_decoder"]), c)
        self.we = load_wrapped_encoder(self.net, path(cfg["pca"]), cfg["wrapper"])
        self.settings = AeicSettings(**cfg["aeic"])
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.metrics = PairedMetrics(self.dev)
        self.coder = Coder(self.net)
        self.decoder_class = EscalatedAeic if S.escalation_gamma_z else AdaptedAeic


def _writer(path, fields):
    fh = open(path, "w", newline="")
    wr = csv.DictWriter(fh, fieldnames=fields)
    wr.writeheader()
    return fh, wr


MEM_FIELDS = ["protocol", "method", "rate", "image", "gen", "h", "w", "enc_tag", "side", "bpp", "file_bytes",
              "md5_sym", "md5_file", "roundtrip", "same_file_prev", "diag"] + list(PAIRED)


def run_memory_chain(b, png_path, out_dir, stem, rounds, rate, protocol, canvas=False, freeze_from=0, log=print):
    """The in-memory chain of one image (module docstring)."""
    os.makedirs(os.path.join(out_dir, "F"), exist_ok=True)
    net, we, coder = b.net, b.we, b.coder
    dec = b.decoder_class(net, we, b.settings)
    transform = aeic_transform()
    fh, wr = _writer(os.path.join(out_dir, "rows.csv"), MEM_FIELDS)
    with torch.inference_mode():
        x0 = preprocess_image(png_path, transform).to(b.dev).unsqueeze(0)
        h, w = x0.shape[2:]
        gt01 = preprocess_image(png_path, transforms.ToTensor()).unsqueeze(0).to(b.dev)
        x_in, prev, frozen = x0, None, None
        for g in range(1, rounds + 1):
            q = dec.encode(pad64(x_in), h, w, g)
            side = colour_side_info(x_in[:, :, :h, :w], h, w)
            if freeze_from and g > freeze_from:
                side = frozen
            x_hat = decode_symbols(net, q.y_hat)
            yt, it, tag = dec.select_target(q.z_hat, q.y_hat, x_raw=x_hat)
            diag = f"zs{it}{tag}" if yt is not None else "zsNONE"
            yt = q.y_hat if yt is None else yt
            res = dec.consistent_decode(x_hat, q, yt, side, h, w, canvas=canvas)
            x_next = store11(res[0])
            x_canvas = store11(res[4]) if canvas else None
            q.side_info = side
            data = coder.encode("adapted", q, pad64(x_in), h, w)
            z2, y2, s2 = coder.decode(data)
            ok = torch.equal(z2, q.z_hat) and torch.equal(y2, q.y_hat) and s2 == (tuple(side) if side else None)
            with open(os.path.join(out_dir, "F", f"g{g:03d}.bin"), "wb") as f:
                f.write(data)
            if freeze_from and g == freeze_from:
                frozen = side
            sc = b.metrics.score(x_next, gt01)
            r = dict(protocol=protocol, method="adapted", rate=rate, image=stem, gen=g, h=h, w=w, enc_tag=q.tag,
                     side="" if side is None else "_".join(str(v) for v in side), bpp=round(q.bpp, 7),
                     file_bytes=len(data), md5_sym=symbols_md5(q.z_hat, q.y_hat), md5_file=md5(data),
                     roundtrip=int(ok), same_file_prev="" if prev is None else int(data == prev),
                     diag=f"{diag}/rng{res[1]}/{res[2]:.1e}" + (f"/pad{res[3]}" if res[3] else ""), **sc)
            wr.writerow(r)
            fh.flush()
            log(f"{stem} round {g}: bpp {r['bpp']:.5f} file {r['md5_file'][:8]} psnr {r['psnr']} {q.tag}")
            prev, x_in = data, (x_canvas if canvas else x_next)
    fh.close()
    open(os.path.join(out_dir, "DONE"), "w").close()


CANVAS_FIELDS = ["protocol", "method", "rate", "image", "h", "w", "H", "W", "gen", "enc_tag", "side", "side_from",
                 "file_bytes", "stream_bytes", "bpp", "bpp_payload", "bpp_est", "md5_sym", "md5_bits", "md5_file",
                 "md5_img", "md5_crop", "md5_meta", "enc_roundtrip", "dec_ok", "dec_rounds", "dec_bad", "fmt_same",
                 "diag"] + list(PAIRED)


def run_canvas_chain(b, png_path, out_dir, stem, rounds, rate, method="adapted", log=print):
    """The 8-bit canvas file chain of one image (module docstring)."""
    if S.chain_format != "png8" or S.correction_budget <= 0:
        raise SystemExit("the canvas file chain needs chain_format png8 and correction_budget > 0")
    if b.settings.encoder != "variant_a" or b.settings.pad_repair:
        raise SystemExit("the canvas file chain needs the round-selected encoder and no border repair")
    net, we, coder, dev = b.net, b.we, b.coder, b.dev
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    for sub in ("F", "png", "meta"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    def load_png(path):
        return transform(Image.open(path).convert("RGB")).to(dev).unsqueeze(0)

    def store_u8(x_m11):
        im = ((x_m11 + 1) / 2).detach().float().clamp(0, 1).cpu()[0]
        return np.rint(im.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)

    def u8_to_m11(u8):
        return transform(Image.fromarray(u8)).unsqueeze(0).to(dev)

    def canvas_size(h, w):
        return -(-h // 64) * 64, -(-w // 64) * 64

    def encode_to_file(g, src_png, src_meta, f_path):
        if g == 1:
            x_in = load_png(src_png)
            h, w = x_in.shape[2:]
            xc = pad64(x_in)
            side, side_from = colour_side_info(x_in, h, w), "measured_on_original"
            q = encode_pair(net, xc, h, w)
            tag = "plain" if method == "plain" else "zsA_plain"
        else:
            with open(src_meta, "rb") as f:
                h, w, has, *sv = struct.unpack(META, f.read())
            side, side_from = (tuple(sv) if has else None), "sidecar_frozen_g1"
            xc = load_png(src_png)
            if tuple(xc.shape[2:]) != canvas_size(h, w):
                raise RuntimeError(f"canvas size {tuple(xc.shape[2:])} is not the multiple-of-64 size of {(h, w)}")
            if method == "plain":
                q, tag = encode_pair(net, xc, h, w), "plain"
            else:
                q, tag = encode_pair(net, xc, h, w, latent=we(xc)), "zsA_wrapped"
        q.side_info = side
        data = coder.encode(method, q, xc, h, w)
        with open(f_path, "wb") as f:
            f.write(data)
        return dict(tag=tag, bpp_est=q.bpp, enc_sym=symbols_md5(q.z_hat, q.y_hat), side_from=side_from,
                    canvas=tuple(xc.shape[2:]))

    def decode_from_file(f_path):
        with open(f_path, "rb") as f:
            data = f.read()
        h, w, n = struct.unpack(HEADER, data[:12])
        z_hat, y_hat, side = coder.decode(data)
        md = dict(md5_sym=symbols_md5(z_hat, y_hat), md5_bits=md5(data[12:12 + n]), md5_file=md5(data),
                  file_bytes=len(data), stream_bytes=n, side=side, h=h, w=w)
        x_hat = decode_symbols(net, y_hat)
        if method == "plain":
            return store_u8(pad64(crop_and_colour_fix(x_hat, h, w, side).float())), md, dict(
                ok="", rounds="", bad="", fmt_same="", diag="")
        dec = b.decoder_class(net, we, b.settings)                     # fresh decoder per decode
        yt, it, tag = dec.select_target(z_hat, y_hat, x_raw=x_hat)
        zs_tag = f"zs{it}{tag}" if yt is not None else "zsNONE"
        yt = y_hat if yt is None else yt
        res = dec.consistent_decode(x_hat, Pair(z_hat=z_hat, y_hat=y_hat, y=None, bpp=0.0), yt, side, h, w,
                                    canvas=True)
        u8 = store_u8(res[4])
        fe = dec.last_write
        return u8, md, dict(ok=fe["ok"], rounds=fe["rounds"], bad=fe["best_bad"],
                            fmt_same=int(torch.equal(u8_to_m11(u8), res[4].float())),
                            diag=f"{zs_tag}|best{fe['best_round']}|{fe['hist']}|rng{res[1]}/{res[2]:.1e}")

    rows = []
    fh, wr = _writer(os.path.join(out_dir, "rows.csv"), CANVAS_FIELDS)
    f_of = lambda g: os.path.join(out_dir, "F", f"g{g:03d}.bin")      # noqa: E731
    png_of = lambda g: os.path.join(out_dir, "png", f"g{g:03d}.png")  # noqa: E731
    meta_of = lambda g: os.path.join(out_dir, "meta", f"g{g:03d}.bin")  # noqa: E731
    with torch.inference_mode():
        gt01 = transforms.ToTensor()(Image.open(png_path).convert("RGB")).unsqueeze(0).to(dev)
        for g in range(1, rounds + 1):
            enc = encode_to_file(g, png_path if g == 1 else png_of(g - 1), "" if g == 1 else meta_of(g - 1), f_of(g))
            u8, md, info = decode_from_file(f_of(g))
            h, w = md["h"], md["w"]
            H, W = u8.shape[:2]
            assert (h, w) == tuple(gt01.shape[2:]) and (H, W) == canvas_size(h, w) == enc["canvas"]
            Image.fromarray(u8).save(png_of(g), format="PNG")
            back = np.array(Image.open(png_of(g)).convert("RGB"), dtype=np.uint8)
            if not np.array_equal(back, u8):
                raise RuntimeError("PNG round trip changed pixels")
            side = md["side"]
            meta = struct.pack(META, h, w, int(side is not None), *(side if side is not None else (0,) * 6))
            with open(meta_of(g), "wb") as f:
                f.write(meta)
            with open(meta_of(g), "rb") as f:
                if f.read() != meta:
                    raise RuntimeError("sidecar round trip changed bytes")
            crop = np.ascontiguousarray(back[:h, :w])
            sc = b.metrics.score(u8_to_m11(crop), gt01)
            r = dict(protocol="8bit-canvas", method=method, rate=rate, image=stem, h=h, w=w, H=H, W=W, gen=g,
                     enc_tag=enc["tag"], side="" if side is None else "_".join(str(v) for v in side),
                     side_from=enc["side_from"], file_bytes=md["file_bytes"], stream_bytes=md["stream_bytes"],
                     bpp=round(8 * md["file_bytes"] / (h * w), 7),
                     bpp_payload=round(8 * (md["file_bytes"] - 12) / (h * w), 7), bpp_est=round(enc["bpp_est"], 7),
                     md5_sym=md["md5_sym"], md5_bits=md["md5_bits"], md5_file=md["md5_file"],
                     md5_img=md5(str(back.shape).encode(), back.tobytes()),
                     md5_crop=md5(str(crop.shape).encode(), crop.tobytes()), md5_meta=md5(meta),
                     enc_roundtrip=int(enc["enc_sym"] == md["md5_sym"]), dec_ok=info["ok"], dec_rounds=info["rounds"],
                     dec_bad=info["bad"], fmt_same=info["fmt_same"], diag=info["diag"], **sc)
            rows.append(r)
            wr.writerow(r)
            fh.flush()
            log(f"{stem} {method} {rate} round {g}: bpp {r['bpp']:.5f} file {r['md5_file'][:8]} crop "
                f"{r['md5_crop'][:8]} psnr {r['psnr']} ok {info['ok']} rounds {info['rounds']}")
    fh.close()
    open(os.path.join(out_dir, "DONE"), "w").close()
