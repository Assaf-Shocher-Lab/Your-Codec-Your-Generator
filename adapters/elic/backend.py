"""One ELIC rate (lambda): exact math, PCA basis, model, wrapper, codec interface and file coder."""
import torch

from adapted_codec.determinism import set_exact_math
from adapted_codec.wrapper import build_wrapper
from adapters.elic import codec
from adapters.elic.file_format import PAD_MULTIPLE, Coder


class ElicBackend:
    def __init__(self, rate, pca, c, weights):
        set_exact_math()
        self.rate = rate
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        p = torch.load(pca, map_location="cpu", weights_only=False)
        codec.WEIGHTS = weights
        self.net = codec.load_elic(rate, self.dev)
        self.wrapper = build_wrapper(self.net, p, c)
        self.ops = codec.ElicOps(self.net, self.wrapper)
        self.coder = Coder(self.net)
        self.pm = PAD_MULTIPLE
