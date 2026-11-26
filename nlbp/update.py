"""The NLBP update of one sampling step (Sec. 3.4, Algorithm "NLBP decoding"), applied to the prior's clean estimate
u_t = x_hat_{0,theta}(x_t, t):

    y_t    = E'(u_t)
    v_t^+  = (1 - lambda_t) y_t + lambda_t y*                  (written y_t + lambda_t (y* - y_t))
    b_t^+  = b(u_t) + kappa lambda_t [r_ref - b(D_ref(y_t))]   (r_ref = b(D_ref(v_ref)), v_ref in {y_hat, y*};
                                                                 kappa = 0 keeps the proposal's complement)
    out    = W(v_t^+, b_t^+)

codec: base_decode = D_ref (the fixed reference decoder with its preprocessing), complement = b, write_step = W,
wrapper.wrapped_encoder = E'."""


def nlbp_update(u, lam, kz, codec, y_target, r_ref):
    """W(v_t^+, b_t^+) for the clean estimate u (lam = lambda_t, kz = kappa, y_target = y*)."""
    y = codec.wrapper.wrapped_encoder(u.float())
    b = codec.complement(u)
    if kz:
        b = b + kz * lam * (r_ref - codec.complement(codec.base_decode(y)))
    return codec.write_step(y + lam * (y_target - y), b)
