"""NLBP final write (App. E.1, after sampling): with the target y* fixed, W_b = W(y*, b) is the exact write for a
complement b. raw(u) = W(y*, b(u)) is the exact write of an image; opt(u, K, lr) runs K Adam steps on b for
||W(y*, b) - u||^2 from b(u) and keeps the best iterate (K = 0: raw). Every output satisfies E'(out) = y* (real
arithmetic).
codec: the NLBP codec interface (base_decode = D, complement = b, write_step(v, b) = J((v - f(b)) / c, b),
wrapper.wrapped_encoder = E')."""
import torch


class TargetWrite:
    """The writes for one fixed target y_hat (here: y*); r_t = b(D(y_hat)) is cached."""
    def __init__(self, codec, y_hat):
        self.codec, self.y_hat = codec, y_hat
        self.x_dec = codec.base_decode(y_hat)
        self.r_t = codec.complement(self.x_dec)
        self.n_dec = 0

    def write_b(self, b):
        return self.codec.write_step(self.y_hat, b)

    def raw(self, u):
        return self.write_b(self.codec.complement(u))

    def opt(self, u, K, lr):
        u = u.detach().float()
        with torch.enable_grad():
            b = self.codec.complement(u).detach().clone().requires_grad_(True)
            o = torch.optim.Adam([b], lr=lr)
            best_l, best_b = None, b.detach().clone()
            for k in range(K + 1):
                o.zero_grad(set_to_none=True)
                L = (self.write_b(b) - u).pow(2).sum()
                if best_l is None or float(L) < best_l:
                    best_l, best_b = float(L), b.detach().clone()
                if k == K:
                    break
                L.backward()
                o.step()
        with torch.no_grad():
            return self.write_b(best_b)
