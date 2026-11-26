"""NLBP (nonlinear back-projection, Sec. 3.4 and App. E.1): a frozen generative prior decodes the received symbols s,
and every output keeps E'(x_out) = y* in F(s), so the file decodes to the same record.

| symbol | meaning | code |
|---|---|---|
| y_hat, y* | received main latent; the selected target (adapted_codec/target_search.py) | geo.y_hat = y* |
| x_base | W(y*, b(D(y_hat))), the adapted decoder's output | state["Dp"] |
| s, omega | flow time; start noise weight = s of the first step | nlbp/schedule.py |
| u_t | the prior's clean estimate x_hat_{0,theta}(x_t, t) = x_s - s v | nlbp/sampler.py |
| lambda_t, kappa | pull toward y* (early -> late value), complement correction weight | lambda2 -> lambda1, kz |
| v_ref, r_ref | reference latent (y_hat or y*) and r_ref = b(D_ref(v_ref)) | target, geo.r_t / r_ys |
| K | Adam steps of the final b refinement | opt_K |

Files: update.py (the three update equations), sampler.py (schedule, start, loop, final write; pixel and latent
priors), final_write.py (W(y*, b) and the bounded b refinement), receiver.py (the per-image state), codec_interface.py
(D_ref, b, W, E' on [-1, 1] images), schedule.py (time grid, bitstream seed), priors/ (clean-estimate adapters),
configs/ (the reported configurations; scripts/run_nlbp.py runs the ELIC FFHQ and HPCM Kodak ones)."""
