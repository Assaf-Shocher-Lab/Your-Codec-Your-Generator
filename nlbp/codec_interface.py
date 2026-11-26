"""The NLBP codec interface (Sec. 3.4, App. E.1) of a codec whose wrapper works on images in [0, 1] (ELIC, HPCM); the
sampler works on images in [-1, 1]:

    base_decode(y)            D_ref(y) = 2 g_s(y).clamp(0, 1) - 1      (the fixed reference decoder with its output step)
    complement(x)             b((x + 1) / 2)
    write_step(y, b)          W(y, b) = J((y - f(b)) / c, b), mapped to [-1, 1]     (E'(W(y, b)) = y)
    wrapper.wrapped_encoder   E'((x + 1) / 2)
wrapper01 is the codec's PCAWrapper (adapted_codec/wrapper.py) on [0, 1] images."""
import torch


def to01(x):
    return (x.float() + 1) / 2


class SignedWrapper:
    """The wrapper seen on [-1, 1] images: wrapped_encoder(x) = E'((x + 1) / 2)."""
    def __init__(self, wrapper01):
        self.wrapper01 = wrapper01

    def wrapped_encoder(self, x):
        return self.wrapper01.wrapped_encoder(to01(x))


class NlbpCodec:
    """D_ref, b, W and E' of one codec (module docstring)."""
    def __init__(self, net, wrapper01):
        self.net, self.wrapper01, self.wrapper = net, wrapper01, SignedWrapper(wrapper01)

    def complement(self, x):
        return self.wrapper01.split(to01(x))[1]

    def write_step(self, y, complement):
        wrapper01 = self.wrapper01
        x_b = wrapper01.join(torch.zeros(len(y), wrapper01.k, *complement.shape[2:], device=y.device), complement)
        return wrapper01.join((y - wrapper01.E(x_b)) / wrapper01.c.view(1, -1, 1, 1), complement) * 2 - 1

    def base_decode(self, y):
        return self.net.g_s(y).clamp(0, 1) * 2 - 1
