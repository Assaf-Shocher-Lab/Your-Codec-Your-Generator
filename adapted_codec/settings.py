"""Settings of the adapted codec, shared by all stages (one namespace, filled from a variant's yaml).

Defaults switch the optional stages off (correction_budget = 0: plain 8-bit write; ladder, box step and stabilization
off); the reported configurations set them in configs/*.yaml. Values are coerced to the type of the default. Paper
symbols: m_y = margin_y, m_z = margin_z, gamma_y / gamma_z = guards of the target update, eta_y / eta_z = guards of the
lattice merit.

Fixed parts of the method (one value in every reported configuration, so not settings): the target update moves t by
the violation-driven step only; the lattice search pulls dM/dy back through the full E', never accepts a move that
increases the number of wrong hyper symbols, tries group moves after the single-pixel scan, and ends with the
pair-aware per-symbol hyper repair.
"""
import copy

DEFAULTS = dict(
    # encoder (round 1: base record Q(E(x)); rounds >= 2: wrapped record Q(E'(x)))
    encoder="variant_a",
    # target search and refinement (Sec. 3.2, App. B.2)
    search_start="centre",          # "proj": start at the projection of E'(D(y_hat)); "centre": start at y_hat
    search_steps=400,
    search_lr=0.01,
    margin_y=0.02,                  # m_y, main-cell margin
    margin_z=0.02,                  # m_z, hyper-cell margin
    float_margin_y=None,            # 8-bit chains whose repair starts from the float D' output (ELIC): m_y of y*_F
    refine_steps=0,                 # bounded Adam steps toward the projected start (0: no refinement)
    refine_lr=0.003,
    refine_weight=100.0,
    image_hw=None,                  # (h, w) of the unpadded image, set per image by the chain
    pad_rows=96,                    # rows / columns of the border repair strip of padded images
    pad_iters=1000,                 # border repair: maximum steps
    pad_lr=2e-3,
    pad_cell_margin=0.08,
    # storage of each round (App. A.2)
    chain_format="none",            # none (float) | clamp (clipped float) | png8 (8-bit)
    s8_signed_path=False,           # S_8 float path through [-1, 1] (the ELIC encoder's own normalization)
    # clipped-float range repair (App. B.3)
    range_repair_steps=0,
    range_guard=1e-4,
    # 8-bit closed-loop write and target correction (Sec. 3.3, App. B.3)
    correction_budget=0,            # maximum target updates (0: plain write)
    gamma_y=0.02,
    gamma_z=0.02,
    context_free_count=0,           # 1: count mismatches as nz + #(|v - y_hat| >= 1/2)
    escalation_gamma_z=None,        # AEIC DIV2K: a failed closed-loop write is repeated once with this hyper guard
    # lattice search (App. B.3)
    lattice_steps=9056,             # maximum lattice moves per call
    lattice_trials=9056,            # budget of exactly evaluated trials per call
    lattice_time_cap=86000.0,       # seconds; safety abort only
    eta_y=0.005,
    eta_z=0.003,
    lattice_scan=64,                # single-pixel ranks tried before the descent is stuck
    lattice_group=32,               # coupled group moves (2, 4, 8 pixels) on the top ranks
    # S_8 inside the search trials
    fast_s8=0,                      # 1: device lookup table with the same values as the PNG path
    fast_s8_audit=0,
    # outer stabilization (HPCM, App. B.3)
    stabilization_states=0,
    stabilization_candidates=[1],
    # escalation ladder (App. B.3)
    ladder=0,
    ladder_stages=[(None, None), (60, None), (60, 0.10)],
    box_step=0,
    box_steps=600,
)


class Settings:
    """Attribute namespace of the settings."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.__dict__.update(copy.deepcopy(DEFAULTS))

    def load(self, values):
        self.reset()
        for k, v in (values or {}).items():
            if k not in DEFAULTS:
                raise KeyError(f"unknown setting {k!r}")
            d = DEFAULTS[k]
            if isinstance(d, bool):
                v = bool(v)
            elif isinstance(d, int) and not isinstance(d, bool):
                v = int(v)
            elif isinstance(d, float):
                v = float(v)
            elif isinstance(d, list) and k == "ladder_stages":
                v = [tuple(s) for s in v]
            setattr(self, k, v)
        if self.encoder != "variant_a":
            raise ValueError(f"encoder must be variant_a, not {self.encoder!r}")
        if self.chain_format not in ("none", "clamp", "png8"):
            raise ValueError(f"chain_format must be none, clamp or png8, not {self.chain_format!r}")
        for k in ("lattice_steps", "lattice_trials", "lattice_group"):
            if getattr(self, k) <= 0:
                raise ValueError(f"{k} must be > 0 (the lattice search has no off switch)")
        return self

    def as_dict(self):
        return {k: getattr(self, k) for k in DEFAULTS}


S = Settings()
