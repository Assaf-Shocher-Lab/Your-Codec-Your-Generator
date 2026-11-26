"""The codec interface the adapted codec needs (adapters/<codec>/ subclass it):

    z_offset()             hyper offset (EntropyBottleneck medians or the codec's hyper means)
    hyper(y)               (h_a(y), z_hat = Q_z(h_a(y)))
    quantize_main(y, z_hat)  (y_hat = Q_y(y; z_hat), scales, means)
    base_encode(x)         E(x), the codec's own analysis transform
    wrapped_encode(x)      E'(x)
    base_decode(y_hat, z_hat)  D(y_hat), the codec's own synthesis and output steps
    write(x, v)            W(v, b(x)): the exact latent write on the complement of x
    gaussian_conditional, entropy_bottleneck   likelihood models (estimated bits)

HyperpriorOps / MeanScaleOps: CompressAI bmshj2018-hyperprior / mbt2018-mean, images in [0, 1]; encode_bytes: the
record's entropy coding with the CompressAI coders."""
import torch


class HyperpriorOps:
    def __init__(self, net, wrapper):
        self.net, self.wrapper = net, wrapper
        self.gaussian_conditional, self.entropy_bottleneck = net.gaussian_conditional, net.entropy_bottleneck

    def z_offset(self):
        return self.entropy_bottleneck._get_medians()

    def hyper(self, y):
        z = self.net.h_a(torch.abs(y))
        off = self.z_offset()
        return z, torch.round(z - off) + off

    def quantize_main(self, y, z_hat):
        return torch.round(y), self.net.h_s(z_hat), None

    def wrapped_encode(self, x):
        return self.wrapper.wrapped_encoder(x)

    def decode(self, y_hat):
        return self.write(self.base_decode(y_hat), y_hat)

    def base_decode(self, y_hat, z_hat=None):
        return self.net.g_s(y_hat).clamp(0, 1)

    def write(self, x_hat, y_t):
        d = self.wrapper
        _, u1 = d.split(x_hat.float())
        y1 = d.join(torch.zeros(len(y_t), d.k, *u1.shape[2:], device=y_t.device), u1)
        return d.join((y_t - d.E(y1)) / d.c.view(1, -1, 1, 1), u1)

    def base_encode(self, x):
        return self.net.g_a(x)


class MeanScaleOps(HyperpriorOps):
    def hyper(self, y):
        z = self.net.h_a(y)
        off = self.z_offset()
        return z, torch.round(z - off) + off

    def quantize_main(self, y, z_hat):
        scales, means = self.net.h_s(z_hat).chunk(2, 1)
        return torch.round(y - means) + means, scales, means


@torch.no_grad()
def encode_bytes(ops, z_hat, y_hat):
    """CompressAI rANS of the record (z strings, then y strings with the record own scales and means) and the decode-back check. Returns (strings, payload bytes, decode_ok)."""
    z_strings = ops.entropy_bottleneck.compress(z_hat)
    zd = ops.entropy_bottleneck.decompress(z_strings, z_hat.shape[-2:])
    _, sc, mu = ops.quantize_main(y_hat, zd)
    idx = ops.gaussian_conditional.build_indexes(sc)
    y_strings = ops.gaussian_conditional.compress(y_hat, idx, means=mu)
    yd = ops.gaussian_conditional.decompress(y_strings, idx, zd.dtype, means=mu)
    ok = bool(torch.equal(zd, z_hat) and torch.equal(yd, y_hat))
    return [y_strings, z_strings], sum(len(s[0]) for s in (y_strings, z_strings)), ok
