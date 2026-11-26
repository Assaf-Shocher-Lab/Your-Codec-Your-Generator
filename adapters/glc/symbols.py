"""The GLC symbol file (no entropy coder): header "<4sBBHHHHHHHHH" (magic b"GLCS", version 1, q, h, w, H, W, zh, zw,
C, yh, yw), the VQ indices as uint16, y_q as int16 (little-endian). Equality of rounds is checked on this file.
y_hat_from_symbols is GLC's forward_four_part_prior (src/models/common_model.py) with each part's y_q read from the
symbols instead of computed from y."""
import hashlib
import struct

import numpy as np
import torch

MAGIC = b"GLCS"
VERSION = 1
_HEAD = struct.Struct("<4sBBHHHHHHHHH")
CODEBOOK = 16384


def pack(idx, y_q, q, h, w, H, W, zshape):
    i = idx.detach().reshape(-1).cpu().numpy()
    yq = y_q.detach().cpu().numpy()
    if yq.ndim != 4 or yq.shape[0] != 1:
        raise ValueError(f"y_q must be (1, C, yh, yw), got {yq.shape}")
    if i.size != zshape[2] * zshape[3] or i.min() < 0 or i.max() >= CODEBOOK:
        raise ValueError("VQ indices out of range or of the wrong count")
    yi = np.rint(yq).astype(np.int64)
    if not np.array_equal(yi.astype(yq.dtype), yq):
        raise ValueError("y_q holds non-integer values")
    if yi.min() < -32768 or yi.max() > 32767:
        raise ValueError(f"y_q out of the int16 range: {yi.min()} .. {yi.max()}")
    _, C, yh, yw = yq.shape
    head = _HEAD.pack(MAGIC, VERSION, int(q), h, w, H, W, int(zshape[2]), int(zshape[3]), C, yh, yw)
    return head + i.astype("<u2").tobytes() + yi.astype("<i2").tobytes()


def unpack(data, device="cpu"):
    magic, ver, q, h, w, H, W, zh, zw, C, yh, yw = _HEAD.unpack_from(data, 0)
    if magic != MAGIC or ver != VERSION:
        raise ValueError("not a GLC symbol file (version 1)")
    o = _HEAD.size
    n_i, n_y = zh * zw, C * yh * yw
    if len(data) != o + 2 * n_i + 2 * n_y:
        raise ValueError("symbol file has the wrong length")
    i = np.frombuffer(data, dtype="<u2", count=n_i, offset=o).astype(np.int64)
    y = np.frombuffer(data, dtype="<i2", count=n_y, offset=o + 2 * n_i).astype(np.float32)
    return dict(q=q, h=h, w=w, H=H, W=W, zshape=(1, None, zh, zw),
                idx=torch.from_numpy(i.copy()).view(-1, 1).to(device),
                y_q=torch.from_numpy(y.copy()).view(1, C, yh, yw).to(device))


def sha256(data):
    return hashlib.sha256(data).hexdigest()


@torch.no_grad()
def y_hat_from_symbols(net, idx, zshape, y_q):
    self = net
    cz = net.z_vq.emb_dim if zshape[1] is None else zshape[1]
    z_hat = net.z_vq.get_quan_feat(idx, (1, zshape[2], zshape[3], cz))
    common_params = net.y_prior_fusion(net.hyper_dec(z_hat))
    y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2, y_spatial_prior_adaptor_3, y_spatial_prior = (
        net.y_spatial_prior_adaptor_1, net.y_spatial_prior_adaptor_2, net.y_spatial_prior_adaptor_3,
        net.y_spatial_prior)
    y_spatial_prior_reduction = net.y_spatial_prior_reduction
    y = y_q

    q_enc, q_dec, scales, means = self.separate_prior(common_params)
    if y_spatial_prior_reduction is not None:
        common_params = y_spatial_prior_reduction(common_params)
    dtype = y.dtype
    device = y.device
    B, C, H, W = y.size()
    mask_0, mask_1, mask_2, mask_3 = self.get_mask_four_parts(B, C, H, W, dtype, device)

    y_hat_0 = y_q * mask_0 + means * mask_0

    y_hat_so_far = y_hat_0
    params = torch.cat((y_hat_so_far, common_params), dim=1)
    scales, means = y_spatial_prior(y_spatial_prior_adaptor_1(params)).chunk(2, 1)

    y_hat_1 = y_q * mask_1 + means * mask_1

    y_hat_so_far = y_hat_so_far + y_hat_1
    params = torch.cat((y_hat_so_far, common_params), dim=1)
    scales, means = y_spatial_prior(y_spatial_prior_adaptor_2(params)).chunk(2, 1)

    y_hat_2 = y_q * mask_2 + means * mask_2

    y_hat_so_far = y_hat_so_far + y_hat_2
    params = torch.cat((y_hat_so_far, common_params), dim=1)
    scales, means = y_spatial_prior(y_spatial_prior_adaptor_3(params)).chunk(2, 1)

    y_hat_3 = y_q * mask_3 + means * mask_3

    y_hat = y_hat_so_far + y_hat_3

    y_hat = y_hat * q_dec

    return y_hat, q_dec
