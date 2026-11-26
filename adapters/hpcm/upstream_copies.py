"""Upstream HPCM functions in which the adapted codec acts, copied verbatim (third_party/hpcm, LIC-HPCM c271cfe2);
our lines are between "adapted codec" fences and every upstream line they replace is kept as "# upstream: <line>".
tests/test_upstream_copies.py restores the upstream text from this file and compares it with third_party/hpcm.

  HPCM           the model with the constructor of the reported runs (two fenced changes, no numerical effect on GPU)
  HPCMCoderCopy  HPCM.decompress, returning the decoded record for the file check
Import after adapters.hpcm.codec (its shims bind the upstream `src` package and the rANS coder)."""
import torch
from torch import nn

from src.models.HPCM_Large import HPCM as HPCMUpstream                       # third_party/hpcm, unchanged
from src.models.HPCM_Large import (CrossAttentionCell, basemodel, conv1x1, g_a, g_s, h_a, h_s,  # noqa: F401
                                   y_spatial_prior_s1_s2, y_spatial_prior_s3)


class HPCM(HPCMUpstream):
    """HPCM-Large with the constructor the reported runs used (load_hpcm calls HPCM())."""

# upstream copy: src/models/HPCM_Large.py L185-208 (sha256 6b62db5d...)
    def __init__(self, M=320, N=256):
        # --- adapted codec: begin --- the subclass calls the base model constructor itself (super() here would be upstream HPCM.__init__)
        basemodel.__init__(self, N)
        # upstream: super().__init__(N)
        # --- adapted codec: end ---
        
        self.g_a = g_a()
        self.g_s = g_s()

        self.h_a = h_a()
        self.h_s = h_s()

        self.y_spatial_prior_adaptor_list_s1 = nn.ModuleList(conv1x1(3*M,3*M) for _ in range(1))
        self.y_spatial_prior_adaptor_list_s2 = nn.ModuleList(conv1x1(3*M,3*M) for _ in range(3))
        self.y_spatial_prior_adaptor_list_s3 = nn.ModuleList(conv1x1(3*M,3*M) for _ in range(6))
        self.y_spatial_prior_s1_s2 = y_spatial_prior_s1_s2(M)
        self.y_spatial_prior_s3 = y_spatial_prior_s3(M)

        self.adaptive_params_list = nn.ParameterList([
            # --- adapted codec: begin --- parameters on the default device (moved with the model by load_hpcm); upstream creates them on cuda
            torch.nn.Parameter(torch.ones((1, M*3, 1, 1)), requires_grad=True) for _ in range(10)
            # upstream: torch.nn.Parameter(torch.ones((1, M*3, 1, 1), device='cuda'), requires_grad=True) for _ in range(10)
            # --- adapted codec: end ---
        ])

        self.attn_s1 = CrossAttentionCell(320*2, 320*2, window_size=4, kernel_size=1)
        self.attn_s2 = CrossAttentionCell(320*2, 320*2, window_size=8, kernel_size=1)
        self.attn_s3 = CrossAttentionCell(320*2, 320*2, window_size=8, kernel_size=1)
        
        self.context_net = nn.ModuleList(conv1x1(2*M,2*M) for _ in range(2))
# end of upstream copy


class HPCMCoderCopy:
    """HPCM.decompress: HPCMCoderCopy.decompress(net, [y_string, z_string], z_shape); with net.pair_only = True it
    returns {"z_hat", "y_hat"}, otherwise HPCM's {"x_hat"}."""

# upstream copy: src/models/HPCM_Large.py L433-456 (sha256 6b62db5d...)
    def decompress(self, strings, shape):
        from src.entropy_models import ubransDecoder
        device = self.quantized_cdf_z.device
        output_size = (1, self.scales_hyper.size(1), *shape)
        indexes_z = self.build_indexes_z(output_size).to(device)
        
        decoder_z = ubransDecoder()
        decoder_z.set_stream(strings[1])
        z_res_hat = self.decompress_symbols(indexes_z, self.quantized_cdf_z.cpu().numpy(), self.cdf_length_z.cpu().numpy(), self.offset_z.cpu().numpy(), decoder_z)
        z_hat = z_res_hat+self.means_hyper
        
        params = self.h_s(z_hat)
        decoder_y = ubransDecoder()
        decoder_y.set_stream(strings[0])
        y_hat = self.decompress_hpcm(params, 
                                            self.y_spatial_prior_adaptor_list_s1, self.y_spatial_prior_s1_s2, 
                                            self.y_spatial_prior_adaptor_list_s2, self.y_spatial_prior_s1_s2, 
                                            self.y_spatial_prior_adaptor_list_s3, self.y_spatial_prior_s3, 
                                            self.adaptive_params_list, self.context_net, decoder_y
                                            )
    
        # --- adapted codec: begin --- return the decoded record (z_hat, y_hat) instead of the image when the file check asks for it
        if getattr(self, "pair_only", False):
            return {"z_hat": z_hat, "y_hat": y_hat}
        # --- adapted codec: end ---
        x_hat = self.g_s(y_hat).clamp_(0,1)
        
        return {"x_hat": x_hat}
# end of upstream copy
