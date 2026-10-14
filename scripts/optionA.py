"""Option A: train, evaluate and apply the four-parameter model (RESULTS.md 35).

Data: ``data/processed/optionA_card_dataset_shards`` -- 32,768 comphs draws of
(beta, delta, rho, R_gamma) from the switched edge-mixture proposal, card
access 50/50, 16 households per draw (RESULTS 30, 32). Configuration: the
adopted one (RESULTS 29) -- anchor + level inputs, ``log(1 - delta)`` target,
wide embedder, static standardisation on age and log mean income only -- with
R_gamma as a fourth target. ``train --beta_transform log1m|logit`` swaps beta's
flow target (RESULTS 37); evaluation and PSID read it from the checkpoints'
config.

Three things differ from every earlier training script, and each would be
silent if got wrong:

* **theta comes from the shards**, never from a sampler, and is checked against
  the proposal the generation recorded: the importance weights are only right
  for draws from that proposal.
* **Every posterior summary is importance-weighted** back to the uniform prior
  (``prior.proposal_log_weight``). The checkpoint records its proposal, and
  ``load_posterior`` refuses it to any caller that does not weight.
* **Card type is marginalised**: the network never sees it, so its posterior
  integrates over cardholder / no card at the 50/50 generation share. PSID has
  no card-access variable. Its proxy (any reported card debt) adds nothing the
  network cannot already see -- a household that borrows shows negative liquid
  wealth, which no no-card household can -- and it misclassifies cardholders
  who never borrow, so conditioning on it would only add bias.

``evaluate``:

* **SBC** on the 1,000 uniform-prior simulations, with weighted ranks:
  coverage, rank uniformity and recovery of the corrected posterior. Uniform
  draws, so recovery is comparable to earlier models' held-out numbers.
* **Coverage by posterior-mean region**: the valid conditional check. Coverage
  by TRUE-theta region (RESULTS 30.1) is printed for continuity but is not a
  calibration test: an exact posterior also under-covers truths at a prior
  edge whenever the likelihood is wide. In a uniform-prior Gaussian toy at
  beta's width an exact posterior covers 0.56-0.76 of truths above 0.95, and
  0.90 in every bin of its own posterior mean.
* **Held-out proposal draws** (the last shards, one household per draw so
  start ages are not clumped): the network's unweighted ``q``, comparable to
  the R_gamma pilot (RESULTS 32.1), and held-out ``log q``.

Usage::

    PYTHONPATH=. .venv/bin/python scripts/optionA.py verify
    PYTHONPATH=. .venv/bin/python scripts/optionA.py train --seed 0
    PYTHONPATH=. .venv/bin/python scripts/optionA.py evaluate \
        --run_dirs outputs/optionA/s0 outputs/optionA/s1 ...
    PYTHONPATH=. .venv/bin/python scripts/optionA.py psid \
        --run_dirs outputs/optionA/s0 outputs/optionA/s1 ...
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

from hh_npe.data.waves import FEATURES_TWOASSET_AGE
from hh_npe.data.windows import build_windowed, window_panel
from hh_npe.evaluation.weighted import sample_weighted, sbc_from_u
from hh_npe.npe.embedder import TrajectoryTransformer
from hh_npe.npe.prior import (
    A2_STRIP,
    BETA_TRANSFORMS,
    PHASE3,
    PHASE3_RGAMMA,
    RGAMMA_TRANSFORMS,
    SWITCHED_A2,
    EdgeMixture,
    PriorBox,
    SwitchedProposal,
    beta_flow_bounds,
    beta_from_flow,
    beta_log_jacobian,
    beta_to_flow,
    check_rgamma_tiling,
    from_log1m,
    log1m_box,
    proposal_log_weight,
    rgamma_flow_bounds,
    rgamma_from_flow,
    rgamma_log_jacobian,
    rgamma_to_flow,
    sample_sobol,
    to_log1m,
)
from hh_npe.npe.train import load_posterior, save_posterior, train_npe
from scripts.compare_windows import EMBEDDER, TRAINING, anchor_log, mean_income_channel

SHARDS = Path("data/processed/optionA_card_dataset_shards")
SBC_CACHE = Path("outputs/optionA/sbc_sims.pt")
PSID_X = Path("data/processed/psid_x_comphs_optionA.pt")
OUT = Path("outputs/optionA")
BOX = PHASE3_RGAMMA
N_TOTAL = 32768
N_HELDOUT_SHARDS = 2
#: Runs that are not 32,768 draws with 2 held-out shards: the R_gamma strip
#: (RESULTS 43.5) holds out 1 of its 6 shards, leaving the main run's training
#: density per unit of R_gamma.
RUN_SIZE = {"a2_strip": (3072, 1)}
#: Settings that may differ between runs trained together; every other
#: solver_config key must agree, or two simulators would be mixed.
PER_RUN_KEYS = ("proposal", "seed", "rgamma_range")
WINDOW = dict(n_waves=7, start_low=24, start_high=45, wave_years=2, with_age=True)
ARCH = dict(d_model=128, n_layers=3, output_dim=64)
STATIC = ("age", "log_mean_income")
#: Window seeds: SBC as in ensemble_eval (4242), held-out as everywhere (999).
SBC_SEED, HELD_SEED = 4242, 999
PRIOR_SD = (BOX.high - BOX.low) / np.sqrt(12)
#: Region cut points for the coverage tables: RESULTS 30.1's, plus R_gamma thirds.
CUTS = {"beta": (0.30, 0.80, 0.95, 1.00), "delta": (0.85, 0.95, 0.98, 1.00),
        "crra": (0.5, 3.5, 4.5, 5.0), "R_gamma": (1.025, 1.0417, 1.0583, 1.075)}


# --------------------------------------------------------------------------- data

class Shard(NamedTuple):
    """A shard file and the offset of its run's draws in the combined index:
    0 for the first run, the first run's size for the next (RESULTS 43.5)."""
    path: Path
    offset: int


def _load_run(shards: Path, train_shards: int | None, need_complete: bool):
    """One generation run: its draws, checked against the proposal it recorded,
    and its train / held-out shards."""
    cfg = json.loads((shards / "solver_config.json").read_text())
    if not cfg.get("rgamma_range"):
        raise SystemExit(f"{shards}: not an R_gamma run")
    if not cfg.get("card_types") or cfg.get("educ") != "comphs":
        raise SystemExit(f"{shards}: expected a comphs card-types run, got {cfg}")
    files = sorted(shards.glob("shard_*.npz"))
    meta = []
    for f in files:
        d = np.load(f)
        meta.append((int(d["lo"]), int(d["hi"]), d["theta"], d["card"], f))
    meta.sort(key=lambda t: t[0])
    if not meta or [m[0] for m in meta] != [0] + [m[1] for m in meta[:-1]]:
        raise SystemExit(f"{shards}: shards do not tile a contiguous range from 0")
    n = meta[-1][1]
    name = cfg["proposal"]
    total, n_held = RUN_SIZE.get(name, (N_TOTAL, N_HELDOUT_SHARDS))
    if need_complete and n != total:
        raise SystemExit(f"{shards}: {n} of {total} draws generated")
    theta_all = np.concatenate([m[2] for m in meta])
    card_all = np.concatenate([m[3] for m in meta])

    # The weights are only right for these exact draws: regenerate the
    # proposal's sequence and compare.
    # "uniform": the option A2 pilot (RESULTS 43). generate_dataset draws the
    # whole run's Sobol sequence and the pilot is its prefix; Sobol points are
    # sequential, so the prefix is sample_sobol(n).
    ref = {"edge_mixture": lambda: EdgeMixture().sample(n, seed=cfg["seed"]),
           "edge_mixture_switched": lambda: SwitchedProposal().sample(n, seed=cfg["seed"]),
           "uniform": lambda: sample_sobol(n, PHASE3, seed=cfg["seed"]),
           # The rest of the option A2 run: the uniform pilot, then the mixture
           # re-aimed at PSID couples (RESULTS 43.3).
           "a2_switched": lambda: SWITCHED_A2.sample(n, seed=cfg["seed"]),
           # The R_gamma strip below it (RESULTS 43.5).
           "a2_strip": lambda: A2_STRIP.sample(n, seed=cfg["seed"])}
    if name not in ref:
        raise SystemExit(f"proposal {name!r} not handled here")
    if not np.allclose(theta_all[:, :3], ref[name](), rtol=0, atol=1e-12):
        raise SystemExit(f"{shards}: stored theta is not the {name} sequence")
    lo, hi = cfg["rgamma_range"]
    if not ((theta_all[:, 3] >= lo) & (theta_all[:, 3] <= hi)).all():
        raise SystemExit(f"{shards}: R_gamma outside the recorded range {lo}-{hi}")

    if len(meta) <= n_held:
        raise SystemExit(f"{shards}: not enough shards for a held-out set")
    held, train = meta[-n_held:], meta[:-n_held]
    if train_shards is not None:
        train = train[:train_shards]
    return (theta_all, card_all, [m[4] for m in train], [m[4] for m in held],
            {"name": name, "n_train": int(train[-1][1])}, cfg)


def load_shards(shards: Path | list[Path], train_shards: int | None = None,
                need_complete: bool = False):
    """theta and card per draw from the shards, the train / held-out split, and
    the proposal the training draws came from.

    Held out: the last ``N_HELDOUT_SHARDS`` shards of a run (``RUN_SIZE`` for
    others). Training: every shard before them, or only the first
    ``train_shards`` of each run (smoke tests). Training draws are always a
    prefix ``[0, n_train)`` of the generation sequence, which is what
    ``SwitchedProposal.log_weight(theta, n_train)`` assumes.

    **Several runs** (RESULTS 43.5: the main A2 run, then the R_gamma strip
    below it) are combined in the order given: their draws are concatenated,
    each shard carries its run's offset (:class:`Shard`), their R_gamma ranges
    must tile one interval, and every solver setting except
    ``PER_RUN_KEYS`` must agree. The proposal is then ``{"parts": [...]}``, one
    per run with its ``rgamma_range``, which ``proposal_log_weight`` reads; one
    run keeps the ``{"name", "n_train"}`` its checkpoints always recorded.
    The returned config is the first run's, with ``rgamma_range`` the union.
    """
    dirs = [shards] if isinstance(shards, (str, Path)) else list(shards)
    runs = [_load_run(Path(d), train_shards, need_complete) for d in dirs]
    base = {k: v for k, v in runs[0][5].items()
            if not k.startswith("_") and k not in PER_RUN_KEYS}
    for d, r in zip(dirs[1:], runs[1:]):
        other = {k: v for k, v in r[5].items() if not k.startswith("_") and k not in PER_RUN_KEYS}
        if other != base:
            diff = sorted(k for k in set(base) | set(other) if base.get(k) != other.get(k))
            raise SystemExit(f"{d}: generated under different settings than {dirs[0]}: {diff}")
    lo, hi = check_rgamma_tiling([r[5]["rgamma_range"] for r in runs])
    offsets = np.cumsum([0] + [len(r[0]) for r in runs[:-1]])
    train = [Shard(f, int(o)) for r, o in zip(runs, offsets) for f in r[2]]
    held = [Shard(f, int(o)) for r, o in zip(runs, offsets) for f in r[3]]
    if len(runs) == 1:
        proposal = runs[0][4]
    else:
        proposal = {"parts": [{**r[4], "rgamma_range": [float(x) for x in r[5]["rgamma_range"]]}
                              for r in runs]}
    cfg = {**runs[0][5], "rgamma_range": [lo, hi]}
    if len(runs) > 1:
        cfg["runs"] = [{"shards": str(d), "n_draws": len(r[0]),
                        **{k: r[5][k] for k in PER_RUN_KEYS}} for d, r in zip(dirs, runs)]
    return (np.concatenate([r[0] for r in runs]), np.concatenate([r[1] for r in runs]),
            train, held, proposal, cfg)


def n_train(proposal: dict) -> int:
    return sum(p["n_train"] for p in proposal.get("parts", [proposal]))


def box_of(cfg: dict) -> PriorBox:
    """The prior box of a run, from the R_gamma range its data cover (the A box
    for every run before RESULTS 43.5)."""
    lo, hi = cfg.get("rgamma_range") or (BOX.rgamma_low, BOX.rgamma_high)
    return dataclasses.replace(BOX, rgamma_low=float(lo), rgamma_high=float(hi))


def transform(x: torch.Tensor, masked: bool = False):
    """The adopted input chain: level channel from dollar levels, then anchor.

    ``masked`` (RESULTS 50): consumption, liquid and illiquid values marked
    missing (NaN, :func:`add_imputation_mask`) are set to 0, and one 0/1
    channel per such feature marks them. Income, the anchor, is never missing.
    Without ``masked`` the chain is exactly as before."""
    x, feats = mean_income_channel(x, FEATURES_TWOASSET_AGE)
    x = anchor_log(x, feats)
    if not masked:
        return x, feats
    miss = torch.stack([torch.isnan(x[..., feats.index(f)]) for f in MASKABLE], -1)
    if int(torch.isnan(x).sum()) != int(miss.sum()):
        raise ValueError("missing values outside consumption, liquid and illiquid wealth")
    x = torch.nan_to_num(x, nan=0.0)
    return (torch.cat([x, miss.to(x.dtype)], -1),
            tuple(feats) + tuple(f"missing_{f}" for f in MASKABLE))


#: Dollar features that may carry observation noise (RESULTS 43.7); age is exact.
NOISY = ("income", "consumption", "liquid_assets", "illiquid_assets")
#: Added to a window seed for its noise stream, so noise never reuses it.
NOISE_SEED = 437_000
#: Features PSID's accuracy codes can mark as imputed (RESULTS 49, 50). Income
#: and age carry no codes and are never masked.
MASKABLE = ("consumption", "liquid_assets", "illiquid_assets")
#: Added to a window seed for its imputation-mask stream.
MASK_SEED = 529_000


def load_mask_patterns(path, sha256: str | None = None) -> dict:
    """PSID couples' imputation masks (``psid_accuracy_codes.py --write_mask``):
    ``mask`` (couples, 7, len(MASKABLE)) bool, ``psid_row``, ``tau``. With
    ``sha256``, the file must be the one a checkpoint was trained with."""
    import hashlib
    path = Path(path)
    if sha256 is not None and hashlib.sha256(path.read_bytes()).hexdigest() != sha256:
        raise SystemExit(f"{path} is not the mask file this model was trained with")
    d = torch.load(path, weights_only=False)
    if tuple(d["features"]) != MASKABLE:
        raise SystemExit(f"{path}: mask features {d['features']} are not {MASKABLE}")
    return d


def psid_missing(spec, psid_rows: np.ndarray) -> np.ndarray | None:
    """PSID's own masks for rows ``psid_rows`` (couples, 7, len(MASKABLE)),
    from the file the run was trained with; None for an unmasked run. The
    file must cover exactly these rows, in this order."""
    if not spec.masked:
        return None
    md = load_mask_patterns(spec.imputation_mask[0], spec.imputation_mask[2])
    if not np.array_equal(md["psid_row"].numpy(), np.asarray(psid_rows)):
        raise SystemExit(f"{spec.imputation_mask[0]} covers other PSID rows")
    return md["mask"].numpy()


def set_missing(x_raw: np.ndarray, miss: np.ndarray | None) -> np.ndarray:
    """Raw (dollar) windows with masked feature-waves set to NaN."""
    if miss is None:
        return x_raw
    out = np.asarray(x_raw, dtype=float).copy()
    for k, f in enumerate(MASKABLE):
        j = FEATURES_TWOASSET_AGE.index(f)
        out[..., j] = np.where(miss[..., k], np.nan, out[..., j])
    return out


def add_imputation_mask(x: torch.Tensor, patterns, seed: int) -> torch.Tensor:
    """Mark simulated feature-waves missing (NaN) the way PSID's are (RESULTS
    50). Each household (row) takes the whole mask of one PSID couple drawn
    uniformly, wave for wave, so the rate by feature and wave, and how masks
    cluster within a household, are PSID couples'. Masks are drawn
    independently of theta and of the values, so the network learns to read a
    masked value as missing, never as evidence about theta."""
    m = torch.as_tensor(np.asarray(patterns), dtype=torch.bool)
    if x.shape[1] != m.shape[1]:
        raise ValueError(f"windows have {x.shape[1]} waves, the masks {m.shape[1]}")
    g = torch.Generator().manual_seed(seed)
    pick = torch.randint(0, len(m), (x.shape[0],), generator=g)
    out = x.clone()
    for k, f in enumerate(MASKABLE):
        j = FEATURES_TWOASSET_AGE.index(f)
        out[..., j] = torch.where(m[pick, :, k], torch.full_like(x[..., j], torch.nan),
                                  x[..., j])
    return out


def _noise_range(v) -> tuple[float, float]:
    """A feature's noise as ``(low, high)``: a number is a fixed sigma."""
    lo, hi = (float(v), float(v)) if np.isscalar(v) else map(float, v)
    if not 0 <= lo <= hi:
        raise ValueError(f"observation noise range {v!r}: need 0 <= low <= high")
    return lo, hi


def add_obs_noise(x: torch.Tensor, noise, seed: int) -> torch.Tensor:
    """Classical measurement error on raw (dollar) windows (RESULTS 43.7).

    ``noise`` maps features to a sigma or a ``(low, high)`` range (a dict or
    its items; :class:`Spec`'s ``obs_noise`` holds ``(feature, low, high)``).
    Each household (row) draws its own sigma per feature, uniformly on the
    range, since how noisy a household's data are is unknown and the network
    should marginalise over it; then every wave is multiplied by
    ``exp(sigma * z)``, ``z`` iid N(0, 1). The noise has mean zero on the log
    scale, so medians are kept; zero stays zero and signs are kept. Age is
    never touched. Unlike ``compare_windows.add_obs_noise`` (§9.4: one sigma
    on every feature), each feature has its own range."""
    items = noise.items() if isinstance(noise, dict) else noise
    rng = {e[0]: _noise_range(e[1] if len(e) == 2 else e[1:]) for e in items}
    if bad := set(rng) - set(NOISY):
        raise ValueError(f"no observation noise on {sorted(bad)}")
    if not any(hi for _lo, hi in rng.values()):
        return x
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(x.shape, generator=g, dtype=x.dtype)
    u = torch.rand((x.shape[0], x.shape[-1]), generator=g, dtype=x.dtype)
    out = x.clone()
    for f, (lo, hi) in rng.items():
        j = FEATURES_TWOASSET_AGE.index(f)
        s = lo + (hi - lo) * u[:, j]
        out[..., j] = x[..., j] * torch.exp(s[:, None] * z[..., j])
    return out


def windows_raw(files, theta_all, seed, n_waves=WINDOW["n_waves"], noise=(), mask=None):
    """``build_windowed`` over shards of one or more runs, untransformed.
    ``files`` are :class:`Shard`\\ s or plain paths (one run, offset 0). Draw
    ids, and the window seeds (``seed + lo``), are offset per run, so ids stay
    unique for the grouped validation split; one run is exactly as before.
    ``noise`` (a :class:`Spec`'s ``obs_noise``) adds observation noise, seeded
    by the window seed; ``mask`` (PSID couples' mask patterns,
    :meth:`Spec.mask_patterns`) then marks feature-waves missing, from its own
    stream, so the noise is the same with or without it."""
    groups: dict[int, list[Path]] = {}
    for f in files:
        path, off = (f.path, f.offset) if isinstance(f, Shard) else (Path(f), 0)
        groups.setdefault(off, []).append(path)
    out = []
    for off, paths in groups.items():
        th, x, pid = build_windowed(paths, theta_all[off:], k=1, seed=seed + off,
                                    **{**WINDOW, "n_waves": n_waves})
        x = add_obs_noise(x, noise, seed=NOISE_SEED + seed + off)
        if mask is not None:
            x = add_imputation_mask(x, mask, seed=MASK_SEED + seed + off)
        out.append((th, x, pid + off))
    return tuple(torch.cat(t) for t in zip(*out))


def windows(files, theta_all, seed, n_waves=WINDOW["n_waves"], noise=(), mask=None):
    th, x, pid = windows_raw(files, theta_all, seed, n_waves, noise, mask)
    x, feats = transform(x, masked=mask is not None)
    return th, x, pid, feats


def one_per_draw(pid: torch.Tensor, seed: int) -> np.ndarray:
    """One random household row per draw. Rows come out grouped by window start
    age, so a prefix of them is NOT a random sample of start ages."""
    perm = np.random.default_rng(seed).permutation(len(pid))
    _, first = np.unique(pid.numpy()[perm], return_index=True)
    return np.sort(perm[first])


# --------------------------------------------------------------- flow targets

@dataclasses.dataclass(frozen=True)
class Spec:
    """A run's prior box and flow targets: ``log(1 - delta)`` always, beta's and
    R_gamma's columns as their transforms (RESULTS 37, 43.5). Checkpoints from
    before 43.5 record no R_gamma settings and get the defaults: the A box and
    a linear R_gamma (:func:`spec_of`).

    ``obs_noise``: the observation noise its simulated training windows carried
    (RESULTS 43.7), ``((feature, low, high), ...)``, each household's sigma
    uniform on [low, high]; every simulated window scored against the run must
    carry the same (``windows(..., noise=spec.obs_noise)``). PSID never gets
    any.

    ``imputation_mask``: ``(patterns file, tau, sha256)`` when the run was
    trained with PSID's imputed values treated as missing (RESULTS 50), else
    empty. Simulated windows scored against it take masks from the same file
    (``windows(..., mask=spec.mask_patterns())``), and PSID's own masked
    values are set missing."""

    box: PriorBox = BOX
    beta_transform: str = "linear"
    rgamma_transform: str = "linear"
    obs_noise: tuple = ()
    imputation_mask: tuple = ()

    @property
    def masked(self) -> bool:
        return bool(self.imputation_mask)

    def mask_patterns(self):
        """The mask patterns to draw simulated masks from (None if unmasked)."""
        if not self.masked:
            return None
        path, _tau, sha = self.imputation_mask
        return load_mask_patterns(path, sha)["mask"]

    def __post_init__(self) -> None:
        if self.beta_transform not in BETA_TRANSFORMS:
            raise ValueError(f"beta transform {self.beta_transform!r}")
        if self.rgamma_transform not in RGAMMA_TRANSFORMS:
            raise ValueError(f"R_gamma transform {self.rgamma_transform!r}")
        if bad := {e[0] for e in self.obs_noise} - set(NOISY):
            raise ValueError(f"no observation noise on {sorted(bad)}")
        for e in self.obs_noise:
            _noise_range(e[1:])

    @property
    def prior_sd(self) -> np.ndarray:
        return (self.box.high - self.box.low) / np.sqrt(12)

    @property
    def cuts(self) -> dict:
        """``CUTS``, with R_gamma's thirds of this box's range."""
        return {**CUTS, "R_gamma": tuple(np.linspace(self.box.rgamma_low,
                                                     self.box.rgamma_high, 4).round(4))}

    def flow_box(self) -> PriorBox:
        """The box the flow lives in."""
        blo, bhi = beta_flow_bounds(self.box, self.beta_transform)
        rlo, rhi = rgamma_flow_bounds(self.box, self.rgamma_transform)
        return dataclasses.replace(log1m_box(self.box), beta_low=blo, beta_high=bhi,
                                   rgamma_low=rlo, rgamma_high=rhi)

    def to_flow(self, theta):
        t = beta_to_flow(to_log1m(theta, self.box), self.box, self.beta_transform)
        return rgamma_to_flow(t, self.box, self.rgamma_transform)

    def from_flow(self, t):
        t = rgamma_from_flow(t, self.box, self.rgamma_transform)
        return from_log1m(beta_from_flow(t, self.box, self.beta_transform), self.box)

    def log_jacobian(self, theta: np.ndarray) -> np.ndarray:
        """beta's and R_gamma's ``log |dt / dtheta|`` (delta stays in log space,
        see ``_log_q``)."""
        th = np.asarray(theta, dtype=float)
        return (beta_log_jacobian(th[:, 0], self.box, self.beta_transform)
                + rgamma_log_jacobian(th[:, 3], self.box, self.rgamma_transform))

    def describe(self) -> str:
        noise = ("; observation noise sigma " + ", ".join(
            f"{f} {lo:g}" + (f"-{hi:g}" if hi != lo else "") for f, lo, hi in self.obs_noise)
            if self.obs_noise else "")
        mask = (f"; imputed PSID values missing (tau {self.imputation_mask[1]:g})"
                if self.masked else "")
        return (f"box R_gamma {self.box.rgamma_low}-{self.box.rgamma_high}; beta target "
                f"{self.beta_transform}; R_gamma target {self.rgamma_transform}{noise}{mask}")


def spec_of(cfg: dict) -> Spec:
    """The Spec a run's ``results.json`` config describes."""
    return Spec(box=box_of(cfg), beta_transform=cfg.get("beta_transform", "linear"),
                rgamma_transform=cfg.get("rgamma_transform", "linear"),
                obs_noise=tuple((f, *_noise_range(v))
                                for f, v in cfg.get("obs_noise", {}).items()),
                imputation_mask=tuple(cfg["imputation_mask"][k] for k in
                                      ("patterns", "tau", "sha256"))
                if "imputation_mask" in cfg else ())


def _spec(s) -> Spec:
    """A Spec, or a beta transform name meaning the pre-43.5 Spec with it."""
    return s if isinstance(s, Spec) else Spec(beta_transform=s)


def flow_box(spec="linear"):
    """The box the flow lives in (``spec``: a :class:`Spec` or, for runs before
    RESULTS 43.5, beta's transform name)."""
    return _spec(spec).flow_box()


def to_flow(theta, spec="linear"):
    return _spec(spec).to_flow(theta)


def from_flow(t, spec="linear"):
    return _spec(spec).from_flow(t)


# --------------------------------------------------------------------------- train

def train(args) -> None:
    theta_all, card_all, train_f, held_f, proposal, cfg = load_shards(
        args.shards, args.train_shards)
    noise = tuple((f, *_noise_range([float(v) for v in r.split(":")] if ":" in r else float(r)))
                  for f, r in (e.split("=") for e in args.obs_noise))
    mask = ()
    if args.mask_patterns:
        import hashlib
        d = load_mask_patterns(args.mask_patterns)
        mask = (str(args.mask_patterns), float(d["tau"]),
                hashlib.sha256(Path(args.mask_patterns).read_bytes()).hexdigest())
    spec = Spec(box=box_of(cfg), beta_transform=args.beta_transform,
                rgamma_transform=args.rgamma_transform, obs_noise=noise, imputation_mask=mask)
    th, x, pid, feats = windows(train_f, theta_all, seed=0, n_waves=args.n_waves,
                                noise=spec.obs_noise, mask=spec.mask_patterns())
    drawn = np.unique(pid.numpy())
    train_ids = np.concatenate([f.offset + np.arange(int(np.load(f.path)["lo"]),
                                                     int(np.load(f.path)["hi"]))
                                for f in train_f])
    assert np.isin(drawn, train_ids).all()
    print(f"train: {len(th)} rows from {len(drawn)} draws (n_train "
          f"{n_train(proposal)}, proposal {proposal}); cardholder share "
          f"{card_all[drawn].mean():.3f}; {spec.describe()}; channels {list(feats)}",
          flush=True)
    keep = torch.tensor([f in STATIC for f in feats])
    f_mean = torch.where(keep, x.mean((0, 1)), torch.zeros(len(feats)))
    f_std = torch.where(keep, x.std((0, 1)), torch.ones(len(feats)))
    emb = TrajectoryTransformer(n_features=len(feats), seq_len=args.n_waves,
                                feature_mean=f_mean, feature_std=f_std,
                                per_sequence=False, **{**EMBEDDER, **ARCH})
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    over = {"max_num_epochs": args.max_epochs} if args.max_epochs else {}
    torch.manual_seed(args.seed)
    post, _d, _i = train_npe(spec.to_flow(th).float(), x, embedder=emb,
                             box=spec.flow_box(), device=dev, group_ids=pid,
                             **{**TRAINING, "batch_size": 1024,
                                "learning_rate": 1e-3, **over})
    out = args.out / f"s{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    save_posterior(post, emb, spec.flow_box(), out / f"posterior_{args.n_waves}w.pt",
                   proposal=proposal)
    # Keys runs before RESULTS 43.5 never recorded, written only when they
    # differ from the defaults, so a run like those writes the same config.
    extra = {}
    if (spec.box.rgamma_low, spec.box.rgamma_high) != (BOX.rgamma_low, BOX.rgamma_high):
        extra["rgamma_range"] = cfg["rgamma_range"]
    if spec.rgamma_transform != "linear":
        extra["rgamma_transform"] = spec.rgamma_transform
    if spec.obs_noise:
        extra["obs_noise"] = {f: [lo, hi] for f, lo, hi in spec.obs_noise}
    if spec.masked:
        extra["imputation_mask"] = dict(zip(("patterns", "tau", "sha256"),
                                            spec.imputation_mask, strict=True))
    (out / "results.json").write_text(json.dumps({"_config": {
        "shards": str(args.shards[0]) if len(args.shards) == 1 else list(map(str, args.shards)),
        "train_shards": args.train_shards,
        "proposal": proposal, "box": list(BOX.names), "delta_transform": True,
        "beta_transform": spec.beta_transform, **extra,
        "window": {**WINDOW, "n_waves": args.n_waves}, "features": list(feats),
        "anchor_log": True,
        "mean_income_channel": True, "static_norm_channels": list(STATIC),
        "card": "marginalised", "arch": ARCH, "max_epochs": args.max_epochs,
        "train_seed": args.seed}}, indent=2))
    print(f"saved {out}")


# ------------------------------------------------------------------------ evaluate

def load_members(run_dirs, device=None, n_waves=WINDOW["n_waves"]):
    """Members of one ensemble: same configuration except the seed. Refuses
    members trained on another window length (the 5-wave models of the
    out-of-sample test, ``train --n_waves 5``)."""
    posts, cfgs, props = [], [], []
    for d in run_dirs:
        cfg = json.loads((Path(d) / "results.json").read_text())["_config"]
        if cfg["window"]["n_waves"] != n_waves:
            raise SystemExit(f"{d}: trained on {cfg['window']['n_waves']} waves, "
                             f"expected {n_waves}")
        ck = load_posterior(Path(d) / f"posterior_{n_waves}w.pt", map_location=device,
                            weighted_ok=True)
        p = ck["posterior"]
        if device:
            p.posterior_estimator.to(device)
            p._device = device
        posts.append(p)
        props.append(ck.get("proposal"))
        cfg.setdefault("beta_transform", "linear")     # runs before RESULTS 37
        cfg.setdefault("rgamma_transform", "linear")   # runs before RESULTS 43.5
        cfgs.append({k: v for k, v in cfg.items() if k != "train_seed"})
    if any(c != cfgs[0] for c in cfgs) or any(p != props[0] for p in props):
        raise SystemExit("ensemble members differ in configuration or proposal")
    if props[0] is None:
        raise SystemExit("checkpoints record no proposal; not an option A run")
    return posts, props[0], cfgs[0]


def _weighting(proposal, spec):
    """Inversion from flow space and the log importance weight. ``spec`` is the
    run's :class:`Spec` (``spec_of(cfg)``); a beta transform name still works
    for runs before RESULTS 43.5, and is refused for later ones, whose box and
    R_gamma target it cannot describe."""
    if not isinstance(spec, Spec) and "parts" in proposal:
        raise SystemExit("this ensemble was trained on several runs (RESULTS 43.5); "
                         "pass spec_of(cfg), not a beta transform name")
    s = _spec(spec)
    inv = lambda t: s.from_flow(t)
    lw = lambda th: proposal_log_weight(proposal, th)
    return inv, lw


def _recovery(r: dict, truth: np.ndarray, spec: Spec = Spec()) -> dict:
    ok = np.isfinite(r["mean"][:, 0])
    m, s, t = r["mean"][ok], r["sd"][ok], truth[ok]
    cov = (r["lo"][ok] <= t) & (t <= r["hi"][ok])
    sd = spec.prior_sd
    out = {}
    for j, n in enumerate(BOX.names):
        e = np.abs(m[:, j] - t[:, j]).mean()
        out[n] = {"corr": float(np.corrcoef(m[:, j], t[:, j])[0, 1]),
                  "mae": float(e), "mae_over_prior_sd": float(e / sd[j]),
                  "contraction": float(1 - (s[:, j] ** 2).mean() / sd[j] ** 2),
                  "coverage_90_interval": float(cov[:, j].mean())}
    out["n"] = int(ok.sum())
    out["median_in_box"] = float(np.median(r["in_box_frac"]))
    out["median_ess_frac"] = float(np.nanmedian(r["ess"] / np.maximum(r["n_kept"], 1)))
    return out


def _by_region(v: np.ndarray, u: np.ndarray, spec: Spec = Spec()) -> dict:
    """Coverage within each region of ``v`` (one column per parameter), and
    which side the misses fall on: ``above_95`` is the share of truths above
    the 95th percentile (an upper tail too short), ``below_5`` below the 5th."""
    out = {}
    for j, n in enumerate(BOX.names):
        c = spec.cuts[n]
        out[n] = []
        for a, b in zip(c[:-1], c[1:]):
            m = (v[:, j] >= a) & ((v[:, j] < b) if b < c[-1] else (v[:, j] <= b))
            uj = u[m, j]
            out[n].append({"lo": a, "hi": b, "n": int(m.sum()),
                           "coverage_90": float(((uj >= 0.05) & (uj <= 0.95)).mean())
                           if m.any() else None,
                           "below_5": float((uj < 0.05).mean()) if m.any() else None,
                           "above_95": float((uj > 0.95).mean()) if m.any() else None})
    return out


def _sbc(r: dict, truth: np.ndarray, spec: Spec = Spec()) -> dict:
    ok = np.isfinite(r["u"]).all(1)
    cov, ks = sbc_from_u(r["u"])
    u = r["u"][ok]
    return {"coverage_90": dict(zip(BOX.names, map(float, cov))),
            "ks_p": dict(zip(BOX.names, map(float, ks))),
            "by_posterior_mean": _by_region(r["mean"][ok], u, spec),
            "by_true_theta": _by_region(truth[ok], u, spec),
            "recovery": _recovery(r, truth, spec)}


def _log_q(members, theta: torch.Tensor, x: torch.Tensor, spec, batch=1024,
           per_row: bool = False):
    """Mean held-out log density of the equal-weight mixture, with delta as
    ``log(1 - delta)`` and beta and R_gamma on their own scales (their
    transforms' Jacobians added back, so arms with different targets compare),
    and without the box-truncation renormalisation sbi's ``log_prob`` adds --
    comparable between option A runs, not to earlier models. ``per_row``: one
    value per household instead of the mean."""
    spec = _spec(spec)
    dev = torch.device(getattr(members[0], "_device", "cpu"))
    th_flow = spec.to_flow(theta)
    jac = torch.from_numpy(spec.log_jacobian(theta.numpy()))
    out = []
    with torch.no_grad():
        for b0 in range(0, len(x), batch):
            tb, xb = th_flow[b0:b0 + batch].to(dev), x[b0:b0 + batch].to(dev)
            lp = torch.stack([m.posterior_estimator.log_prob(tb[None], condition=xb)[0]
                              for m in members])
            out.append((torch.logsumexp(lp, 0) - np.log(len(members))).cpu())
    v = torch.cat(out).double() + jac
    return v.numpy() if per_row else float(v.mean())


def evaluate(args) -> None:
    members, proposal, cfg = load_members(args.run_dirs, args.device)
    spec = spec_of(cfg)
    inv, lw = _weighting(proposal, spec)
    fb = spec.flow_box()
    print(f"{len(members)} members; proposal {proposal}; {spec.describe()}")

    cache = torch.load(args.sbc_cache, weights_only=False)
    if (tuple(cache["rgamma_range"]) != (spec.box.rgamma_low, spec.box.rgamma_high)
            or not cache["card_types"] or cache["educ"] != "comphs"):
        raise SystemExit(f"{args.sbc_cache} was not simulated for option A")
    th_sbc, x_sbc, _ids = window_panel(cache["panels"], cache["thetas"].numpy(),
                                       k=1, seed=SBC_SEED, **WINDOW)
    x_sbc = add_obs_noise(x_sbc, spec.obs_noise, seed=NOISE_SEED + SBC_SEED)
    if spec.masked:
        x_sbc = add_imputation_mask(x_sbc, spec.mask_patterns(), seed=MASK_SEED + SBC_SEED)
    x_sbc, _ = transform(x_sbc, masked=spec.masked)
    th_sbc = th_sbc.numpy().astype(float)
    if args.n_sbc:
        th_sbc, x_sbc = th_sbc[:args.n_sbc], x_sbc[:args.n_sbc]

    theta_all, _card, _tr, held_f, _p, sh_cfg = load_shards(args.shards)
    if box_of(sh_cfg) != spec.box:
        raise SystemExit(f"{args.shards} cover R_gamma {sh_cfg['rgamma_range']}, the "
                         f"ensemble's box {spec.box.rgamma_low}-{spec.box.rgamma_high}")
    th_ho, x_ho, pid_ho, _f = windows(held_f, theta_all, seed=HELD_SEED, noise=spec.obs_noise,
                                      mask=spec.mask_patterns())
    pick = one_per_draw(pid_ho, seed=0)[:args.n_heldout]
    th_ho, x_ho = th_ho[pick], x_ho[pick]
    print(f"SBC: {len(th_sbc)} uniform-prior draws; held-out: {len(th_ho)} proposal "
          f"draws, one household each")

    sets = {"ensemble": members} if len(members) > 1 else {}
    sets.update({f"member_{i}": [m] for i, m in enumerate(members)})
    res = {"proposal": proposal, "config": cfg, "run_dirs": list(map(str, args.run_dirs))}
    torch.manual_seed(0)
    for name, ms in sets.items():
        r = sample_weighted(ms, x_sbc, args.n_draws, fb.low, fb.high,
                            invert=inv, log_weight=lw, truth=th_sbc)
        q = sample_weighted(ms, x_ho, args.n_draws, fb.low, fb.high,
                            invert=inv, log_weight=None, truth=th_ho.numpy())
        res[name] = {"sbc": _sbc(r, th_sbc, spec),
                     "heldout_q": {**_recovery(q, th_ho.numpy(), spec),
                                   "log_q": _log_q(ms, th_ho, x_ho, spec)}}
        if name == "ensemble" or len(sets) == 1:
            np.savez(args.out / f"sbc_{name}.npz", u=r["u"], mean=r["mean"],
                     sd=r["sd"], truth=th_sbc)
        print(f"  {name} done", flush=True)
    (args.out / "evaluation.json").write_text(json.dumps(res, indent=2))
    report(res)


def report(res: dict) -> None:
    names = BOX.names
    print("\n=== SBC (uniform prior, importance-weighted) ===")
    print(f"{'':12s}" + "".join(f"{n:>22s}" for n in names))
    for k in [k for k in res if k.startswith(("ensemble", "member"))]:
        s = res[k]["sbc"]
        print(f"{k:12s}" + "".join(
            f"   cov {s['coverage_90'][n]:.3f} ks {s['ks_p'][n]:.3f}" for n in names))
    key = "ensemble" if "ensemble" in res else "member_0"
    s = res[key]["sbc"]
    print(f"\n--- {key}: recovery on the SBC draws (corrected posterior) ---")
    print(f"{'':10s}{'corr':>8s}{'mae':>10s}{'mae/prior sd':>14s}{'contraction':>13s}")
    for n in names:
        v = s["recovery"][n]
        print(f"{n:10s}{v['corr']:8.3f}{v['mae']:10.4f}{v['mae_over_prior_sd']:14.3f}"
              f"{v['contraction']:13.3f}")
    print(f"in-box median {s['recovery']['median_in_box']:.3f}; median ESS / kept "
          f"{s['recovery']['median_ess_frac']:.3f}")
    for title, k in (("by POSTERIOR-MEAN region (the calibration check)", "by_posterior_mean"),
                     ("by TRUE-theta region (continuity with RESULTS 30.1; an exact "
                      "posterior also under-covers at the edges)", "by_true_theta")):
        print(f"\n--- {key}: 90% coverage {title} ---")
        for n in names:
            print(f"{n:8s}" + "".join(
                f"  [{b['lo']:g}, {b['hi']:g}) "
                + (f"{b['coverage_90']:.3f}" if b["coverage_90"] is not None else "  -  ")
                + f" (n={b['n']})" for b in s[k][n]))
    print(f"\n--- {key}: which side the misses fall on, by POSTERIOR-MEAN region "
          f"(5% each if calibrated) ---")
    for n in names:
        print(f"{n:8s}" + "".join(
            f"  [{b['lo']:g}, {b['hi']:g}) "
            + (f"below {b['below_5']:.3f} above {b['above_95']:.3f}"
               if b["coverage_90"] is not None else "  -  ")
            for b in s["by_posterior_mean"][n]))
    h = res[key]["heldout_q"]
    print(f"\n--- {key}: held-out proposal draws, unweighted q (cf. RESULTS 32.1) ---")
    print(f"{'':10s}{'corr':>8s}{'mae':>10s}{'mae/prior sd':>14s}{'cov90':>8s}")
    for n in names:
        v = h[n]
        print(f"{n:10s}{v['corr']:8.3f}{v['mae']:10.4f}{v['mae_over_prior_sd']:14.3f}"
              f"{v['coverage_90_interval']:8.3f}")
    print(f"held-out log q (beta linear, delta as log(1-delta)): {h['log_q']:.3f}"
          + "".join(f"   {k} {res[k]['heldout_q']['log_q']:.3f}"
                    for k in res if k.startswith("member")))


# ---------------------------------------------------------------------------- PSID

def psid(args) -> None:
    from scripts.literature_ranges import LAIBSON, META
    from scripts.psid_posterior import graded_correction

    members, proposal, cfg = load_members(args.run_dirs, args.device)
    spec = spec_of(cfg)
    box = spec.box
    inv, lw = _weighting(proposal, spec)
    fb = spec.flow_box()
    d = torch.load(args.x, weights_only=False)
    if list(d["features"]) != list(FEATURES_TWOASSET_AGE):
        raise SystemExit(f"{args.x}: features {d['features']} are not "
                         f"{list(FEATURES_TWOASSET_AGE)}")
    from hh_npe.simulator.laibson_calibration import EDUC_GROUPS
    keep = np.flatnonzero(d["educ"].numpy() == EDUC_GROUPS.index("comphs"))
    x_raw = d["x"].numpy()[keep]
    if args.debt_scale != 1.0:
        # The alpha check (RESULTS 42.3, 43.11): card debt (negative liquid
        # wealth) scaled as Laibson et al. scale SCF's under-reported balances.
        li = FEATURES_TWOASSET_AGE.index("liquid_assets")
        x_raw = x_raw.copy()
        x_raw[..., li] = np.where(x_raw[..., li] < 0, args.debt_scale * x_raw[..., li],
                                  x_raw[..., li])
    print(f"{args.x}: {len(keep)} comphs households; {len(members)} members; "
          f"proposal {proposal}; {spec.describe()}"
          + (f"; card debt x {args.debt_scale}" if args.debt_scale != 1.0 else ""))
    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "household_index.npy", keep)

    ci = FEATURES_TWOASSET_AGE.index("consumption")
    arms = {"uncorrected": x_raw, "corrected": graded_correction(x_raw, cons_idx=ci)}
    names = list(box.names)
    lit_lo = np.append(META[0], box.rgamma_low)      # R_gamma: no literature band
    lit_hi = np.append(META[1], box.rgamma_high)
    ref = np.append(LAIBSON, 1.05)                   # their calibrated R_gamma
    summary = {"proposal": proposal, "beta_transform": spec.beta_transform,
               "rgamma_transform": spec.rgamma_transform,
               "rgamma_range": [box.rgamma_low, box.rgamma_high],
               "n_households": int(len(keep))}
    if spec.obs_noise:     # the training windows' noise; PSID itself gets none
        summary["training_obs_noise"] = {f: [lo, hi] for f, lo, hi in spec.obs_noise}
    if args.debt_scale != 1.0:
        summary["debt_scale"] = args.debt_scale
    # PSID's own imputed values, missing exactly where the accuracy codes put
    # them (RESULTS 50), after the consumption correction so it sees no NaN.
    miss = psid_missing(spec, d["psid_row"].numpy()[keep])
    if miss is not None:
        summary["imputation_mask"] = {"tau": spec.imputation_mask[1],
                                      "share_missing": dict(zip(
                                          MASKABLE, miss.mean((0, 1)).tolist(), strict=True))}
        arms = {arm: set_missing(xa, miss) for arm, xa in arms.items()}
    for arm, xa in arms.items():
        torch.manual_seed(0)
        xt, _ = transform(torch.from_numpy(xa).float(), masked=spec.masked)
        r = sample_weighted(members, xt, args.n_draws, fb.low, fb.high,
                            invert=inv, log_weight=lw)
        np.savez(args.out / f"posterior_{arm}.npz", mean=r["mean"], sd=r["sd"],
                 lo=r["lo"], hi=r["hi"], in_box_frac=r["in_box_frac"],
                 ess=r["ess"], n_kept=r["n_kept"], names=np.array(names))
        ok = np.isfinite(r["mean"][:, 0])
        m, lo, hi = r["mean"][ok], r["lo"][ok], r["hi"][ok]
        inside = (m >= lit_lo) & (m <= lit_hi)
        covers = (lo <= ref) & (ref <= hi)
        summary[arm] = {
            "median_of_means": dict(zip(names, np.median(m, 0).tolist())),
            "sd_across_households": dict(zip(names, m.std(0).tolist())),
            "median_posterior_sd": dict(zip(names, np.median(r["sd"][ok], 0).tolist())),
            "median_90_width": dict(zip(names, np.median(hi - lo, 0).tolist())),
            "share_ci_covers_reference": dict(zip(names, covers.mean(0).tolist())),
            "share_mean_in_meta_range": dict(zip(names[:3], inside[:, :3].mean(0).tolist())),
            "share_all_three_in_meta_range": float(inside[:, :3].all(1).mean()),
            "in_box_median": float(np.median(r["in_box_frac"])),
            "in_box_p10": float(np.percentile(r["in_box_frac"], 10)),
            "median_ess_frac": float(np.median(r["ess"][ok] / r["n_kept"][ok])),
            # Weights vary most for posteriors near delta = 1, where the
            # proposal is densest; the low tail of the ESS is the one to watch.
            "ess_p10": float(np.percentile(r["ess"][ok], 10)),
            "unrepresentable": int((~ok).sum()),
        }
        s = summary[arm]
        print(f"\n--- {arm} ---")
        print(f"{'':24s}" + "".join(f"{n:>11s}" for n in names))
        for k in ("median_of_means", "sd_across_households", "median_posterior_sd",
                  "median_90_width", "share_ci_covers_reference"):
            print(f"{k:24s}" + "".join(f"{s[k][n]:11.4f}" for n in names))
        print(f"{'reference':24s}" + "".join(f"{v:11.4f}" for v in ref)
              + "   (Laibson et al.; R_gamma their calibrated 1.05)")
        print("households whose mean is in the meta-analytic range: "
              + "  ".join(f"{n} {s['share_mean_in_meta_range'][n]:.1%}" for n in names[:3])
              + f"   all three {s['share_all_three_in_meta_range']:.1%}")
        print(f"in-box median {s['in_box_median']:.3f} (p10 {s['in_box_p10']:.3f}); "
              f"median ESS / kept {s['median_ess_frac']:.3f}, ESS p10 {s['ess_p10']:.0f}; "
              f"unrepresentable "
              f"{s['unrepresentable']}")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))

    # Standing requirement: every estimation result ships a contour figure.
    from hh_npe.evaluation.plots import contour_corner
    m = np.load(args.out / "posterior_uncorrected.npz")["mean"]
    m = m[np.isfinite(m[:, 0])]
    contour_corner(
        {f"household posterior means  N={len(m)}": m}, box,
        truth={"Laibson et al. MSM (R_gamma: calibrated 1.05)": ref},
        truth_markers=("*",), truth_colors=("#d62728",),
        bands={"meta-analytic range (lit.; R_gamma none)": (lit_lo, lit_hi)},
        reflect_axes=("delta",),
        axis_limits=(box.low, np.where(np.array(names) == "delta", 1.04, box.high)),
        path=args.out / "psid_household_means.png",
        title=f"Per-household posterior means, PSID comphs, {args.out.name} (4 parameters)\n"
              f"{spec.describe()}; importance-weighted to the uniform prior; "
              "card type marginalised")
    print(f"\nwrote {args.out}")


# ---------------------------------------------------------------------------- verify

def verify(args) -> None:
    """Every check the reboot audit ran, before anything trains on the shards."""
    theta_all, card_all, train_f, held_f, proposal, cfg = load_shards(
        args.shards, need_complete=not args.partial)
    tb = int(cfg["theta_batch"])
    # Per-block R_gamma and card, as generated (RESULTS 32).
    blocks = theta_all[:, 3].reshape(-1, tb) if len(theta_all) % tb == 0 else None
    if blocks is None or not (blocks == blocks[:, :1]).all():
        raise SystemExit("R_gamma is not constant within blocks")
    if not (card_all.reshape(-1, tb) == card_all.reshape(-1, tb)[:, :1]).all():
        raise SystemExit("card type is not constant within blocks")
    box = box_of(cfg)
    if not ((theta_all >= box.low) & (theta_all <= box.high)).all():
        raise SystemExit("theta outside the prior box")
    bad = []
    for f in (s.path for s in train_f + held_f):
        d = np.load(f)
        for k in d.files:
            if (k.startswith("panel_") or k == "x") and not np.isfinite(d[k]).all():
                bad.append(f"{f.name}:{k}")
        if "init_pool_sha256" in cfg:
            # RESULTS 43: every household's age-20 start, from the pool.
            w = d["init_wealth"] if "init_wealth" in d.files else None
            if w is None or w.shape != (len(d["x"]), 2) or (w < 0).any():
                bad.append(f"{f.name}:init_wealth")
            if "init_state" in d.files and not np.array_equal(
                    d["init_state"], d["panel_income_state"][:, 0]):
                bad.append(f"{f.name}:init_state")
    if bad:
        raise SystemExit(f"non-finite values or bad initial conditions: {bad}")
    names = [p["name"] for p in proposal.get("parts", [proposal])]
    print(f"OK: {len(theta_all)} draws in {len(train_f) + len(held_f)} shards, "
          f"theta matches {' + '.join(names)}, R_gamma and card per block of {tb}, "
          f"all panels finite. Training {n_train(proposal)} draws "
          f"({len(train_f)} shards), held out {len(held_f)} shards. Cardholder share "
          f"{card_all.mean():.3f}; R_gamma {theta_all[:, 3].min():.4f}-"
          f"{theta_all[:, 3].max():.4f}.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify", help="Check the shards before training.")
    v.add_argument("--shards", type=Path, nargs="+", default=[SHARDS],
                   help="Shard directories; several are combined in order (RESULTS 43.5).")
    v.add_argument("--partial", action="store_true",
                   help="Accept an unfinished run (checks the shards so far).")
    t = sub.add_parser("train", help="Train one ensemble member.")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--shards", type=Path, nargs="+", default=[SHARDS],
                   help="Shard directories; several are combined in order (RESULTS 43.5).")
    t.add_argument("--train_shards", type=int, default=None,
                   help="Use only the first N training shards (smoke tests).")
    t.add_argument("--max_epochs", type=int, default=None,
                   help="Cap epochs (smoke tests); default TRAINING's.")
    t.add_argument("--device", default=None)
    t.add_argument("--rgamma_transform", choices=RGAMMA_TRANSFORMS, default="linear",
                   help="R_gamma's flow target (RESULTS 43.5).")
    t.add_argument("--beta_transform", choices=BETA_TRANSFORMS, default="linear",
                   help="beta's flow target (RESULTS 37); delta is always log(1 - delta).")
    t.add_argument("--n_waves", type=int, default=WINDOW["n_waves"],
                   help="Window length; 5 for the out-of-sample test (oos_optionA.py).")
    t.add_argument("--obs_noise", nargs="*", default=[], metavar="FEATURE=LOW:HIGH",
                   help="Observation noise on the simulated windows, per dollar feature: "
                        "each household's sigma uniform on [LOW, HIGH], or a fixed SIGMA "
                        "(RESULTS 43.7; noise_calibration.py's sigma_range).")
    t.add_argument("--mask_patterns", type=Path, default=None,
                   help="PSID couples' imputation masks (psid_accuracy_codes.py --write_mask): "
                        "simulated windows take them, and the model gets missing-value "
                        "channels (RESULTS 50).")
    t.add_argument("--out", type=Path, default=OUT)
    e = sub.add_parser("evaluate", help="SBC and held-out scores, ensemble and members.")
    e.add_argument("--run_dirs", type=Path, nargs="+", required=True)
    e.add_argument("--shards", type=Path, nargs="+", default=[SHARDS],
                   help="Shard directories; several are combined in order (RESULTS 43.5).")
    e.add_argument("--sbc_cache", type=Path, default=SBC_CACHE)
    e.add_argument("--n_draws", type=int, default=2000,
                   help="Raw posterior draws per household, split across members.")
    e.add_argument("--n_sbc", type=int, default=None, help="Use only the first N SBC draws.")
    e.add_argument("--n_heldout", type=int, default=1024)
    e.add_argument("--device", default=None)
    e.add_argument("--out", type=Path, default=OUT / "evaluation")
    p = sub.add_parser("psid", help="Per-household posteriors for PSID comphs.")
    p.add_argument("--run_dirs", type=Path, nargs="+", required=True)
    p.add_argument("--x", type=Path, default=PSID_X)
    p.add_argument("--n_draws", type=int, default=40000,
                   help="Raw draws per household; the weights cost effective "
                        "sample size, most for posteriors near delta = 1 "
                        "(at 4,000 the ESS p10 was 106; RESULTS 36).")
    p.add_argument("--device", default=None)
    p.add_argument("--debt_scale", type=float, default=1.0,
                   help="Multiply PSID's negative liquid wealth (card debt) by this: "
                        "the alpha = 2.02 check (RESULTS 42.3, 43.11).")
    p.add_argument("--out", type=Path, default=Path("outputs/psid_optionA"))
    args = ap.parse_args()
    if args.cmd == "evaluate":
        args.out.mkdir(parents=True, exist_ok=True)
    {"verify": verify, "train": train, "evaluate": evaluate, "psid": psid}[args.cmd](args)


if __name__ == "__main__":
    main()
