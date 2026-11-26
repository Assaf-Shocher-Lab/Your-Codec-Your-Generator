# Stand-in for the package root of NeuralCompression: only the two output containers the model files import
# (upstream neuralcompression/_outputs.py is not included). The model files under models/ and layers.py are upstream.
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple


@dataclass
class HyperpriorOutput:
    image: Any = None
    latent: Any = None
    latent_likelihoods: Any = None
    quantized_latent_likelihoods: Any = None
    hyper_latent: Any = None
    hyper_latent_likelihoods: Any = None
    quantized_hyper_latent_likelihoods: Any = None
    quantized_latent: Any = None
    quantized_hyper_latent: Any = None


@dataclass
class HyperpriorCompressedOutput:
    latent_strings: List[bytes]
    hyper_latent_strings: List[bytes]
    image_size: Tuple[int, int]
    padded_size: Tuple[int, int]
    hyper_image_size: Optional[Tuple[int, int]] = None
