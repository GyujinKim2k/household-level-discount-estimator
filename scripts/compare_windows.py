"""Compare observation windows on both estimation *and* calibration.

Shards store the annual panel, so the observation window is a post-hoc
aggregation choice: 5, 10 and 15 biennial waves are three views of one dataset,
not three datasets. This trains one NPE per window on identical draws and then
asks two different questions of each:

**Estimation** -- posterior contraction, correlation between truth and
posterior mean, mean absolute error, and held-out ``log q(theta_true | x)``,
all on draws the training prefix never saw.

**Calibration** -- SBC rank uniformity (Talts et al. 2018) and empirical
coverage of the 90% credible interval. A posterior can contract beautifully and
still be wrong; contraction without calibration is confidence, not accuracy,
and only the second question catches a posterior that is confidently mistaken.

The SBC simulations are run **once** and shared across windows -- the same
stored-panel trick that makes the comparison cheap in the first place. They use
i.i.d. prior draws rather than the dataset's Sobol points, because SBC's rank
argument assumes i.i.d. sampling from the prior and a low-discrepancy sequence
is deliberately not that.

Usage::

    uv run python scripts/compare_windows.py --windows 5 10 15
    uv run python scripts/compare_windows.py --n_sbc 200 --train_n 8192
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from hh_npe.data.dataset import read_solver_config
from hh_npe.data.waves import (
    FEATURE_SETS,
    FEATURES_TWOASSET,
    FEATURES_TWOASSET_AGE,
)
from hh_npe.data.windows import build_windowed, max_start_age, window_panel
from hh_npe.evaluation.scoring import calibration_scores, estimation_scores
from hh_npe.npe.embedder import TrajectoryTransformer
from hh_npe.npe.prior import PHASE3, make_sbi_prior, sample_sobol
from hh_npe.npe.train import save_posterior, train_npe
from hh_npe.simulator import laibson_calibration as cal
from hh_npe.simulator.dispatch import simulate_batch_twoasset_gpu
from hh_npe.utils.seeding import seed_all

log = logging.getLogger("compare_windows")

WAVE_YEARS = 2
AGE_RETIRE = 64  # laibson_calibration.AGE_RETIRE, for the end-age warning
EMBEDDER = dict(d_model=64, n_heads=4, n_layers=2, output_dim=32)
# hidden_features and num_transforms are sbi's own defaults, carried unchanged
# since Phase 1 and never swept -- they size the flow, not the embedder, and
# were chosen for a dataset two orders of magnitude smaller than this one.
TRAINING = dict(flow="nsf", max_num_epochs=200, stop_after_epochs=20,
                learning_rate=5e-4, batch_size=256, validation_fraction=0.1,
                hidden_features=50, num_transforms=5)


def add_obs_noise(x: torch.Tensor, sigma: float, features, seed: int):
    """Multiplicative lognormal noise on dollar features only.

    Median-preserving (exp(-sigma^2/2) centring), so this widens the observation
    distribution without shifting its level -- a level shift would be a
    different experiment.
    """
    if sigma <= 0:
        return x
    idx = [i for i, f in enumerate(features) if f != "age"]
    g = torch.Generator().manual_seed(seed)
    noise = torch.exp(torch.randn(x.shape, generator=g) * sigma - sigma ** 2 / 2)
    out = x.clone()
    out[..., idx] = x[..., idx] * noise[..., idx]
    return out


def educ_by_draw(shard_files: list[Path]) -> np.ndarray | None:
    """Education index per Sobol draw, indexed by the global draw number.

    ``panel_id`` from ``build_windowed`` *is* the draw index, so
    ``educ_by_draw(shards)[panel_id]`` gives the group of every row without any
    change to the windowing code.

    Returns ``None`` for a comphs-only dataset, whose shards carry no ``educ``.
    """
    lo_hi = []
    for sf in shard_files:
        d = np.load(sf)
        if "educ" not in d.files:
            return None
        lo_hi.append((int(d["lo"]), int(d["hi"]), d["educ"]))
    n = max(hi for _, hi, _ in lo_hi)
    out = np.full(n, -1, dtype=np.int64)
    for lo, hi, e in lo_hi:
        out[lo:hi] = e
    return out


def one_hot_educ(x: torch.Tensor, educ_rows: np.ndarray) -> torch.Tensor:
    """Append a one-hot education block, constant across waves.

    One-hot rather than a single ordinal column: the calibration blocks are not
    monotone in education (``C0_CREDIT`` is 0.167 / 0.00057 / 0.422 across
    comphs / somehs / compco), so an ordering would be a fiction the network
    could not undo.

    These columns are constant *within* a sequence but vary *across* households,
    so the global per-feature normalisation the embedder uses is well defined.
    Per-sequence normalisation would divide by a zero within-sequence spread,
    which is why they join ``age`` in the skip list.
    """
    n, w, _ = x.shape
    e = torch.from_numpy(np.asarray(educ_rows)).long().clamp(min=0)
    oh = torch.nn.functional.one_hot(e, num_classes=3).float()   # (n, 3)
    return torch.cat([x, oh[:, None, :].expand(n, w, 3)], dim=-1)


#: Features measured in dollars. Only these are log-transformed or perturbed by
#: observation noise -- `age`, the education one-hot block and the derived
#: ratios below are on their own scales and must be left alone.
DOLLAR_FEATURES = frozenset(FEATURES_TWOASSET)

#: Appended by :func:`derived_features`, in this order.
DERIVED_FEATURES: tuple[str, ...] = ("card_debt", "liq_to_inc", "illiq_to_inc")


def derived_features(x: torch.Tensor, features) -> tuple[torch.Tensor, tuple]:
    """Append Laibson et al.'s own moment definitions as explicit channels.

    Their sixteen target moments are ``[%Visa, meanVisa, wealth|debt,
    wealth|no debt] x 4 age bands``: a borrowing *indicator*, and wealth
    *conditional on* that indicator, normalised by income. All three are
    non-linear functions of the level features the network already sees, so
    nothing here is new information -- the question is whether the flow has to
    spend capacity rediscovering them.

    The indicator is the one with a real claim. ``1{liquid < 0}`` is a
    discontinuity at exactly zero, and after the embedder's global
    standardisation a household at -$200 and one at +$200 differ by a hair; the
    borrowing margin beta rides on is precisely the sign, not the magnitude.

    The two ratios divide by the household's own income, so they survive the
    global standardisation that flattens small signed features against
    six-figure income.

    Returns the widened ``x`` and the widened feature-name tuple.
    """
    f = list(features)
    liq, illiq, inc = (f.index("liquid_assets"), f.index("illiquid_assets"),
                       f.index("income"))
    # Income is positive by construction in the simulator and floored in the
    # PSID tensor, but a zero would make both ratios infinite; clamp rather
    # than fail, since the ratio of a zero-income household is just undefined.
    denom = x[..., inc].clamp_min(1.0)
    cols = torch.stack([(x[..., liq] < 0).to(x.dtype),
                        x[..., liq] / denom,
                        x[..., illiq] / denom], dim=-1)
    return torch.cat([x, cols], dim=-1), tuple(f) + DERIVED_FEATURES


#: Appended by :func:`household_ratios`, in this order.
RATIO_FEATURES: tuple[str, ...] = ("ratio_cons", "ratio_liq", "ratio_illiq")


def _mean_income(x: torch.Tensor, features) -> torch.Tensor:
    """The household's own mean income over its observed waves, (N, 1).

    Computed from the input window itself, so it is available identically for
    simulated and PSID households. Floored at $1,000: a household reporting no
    income in every wave has no meaningful scale, and dividing by ~0 would send
    every ratio to infinity.
    """
    return x[..., list(features).index("income")].mean(1, keepdim=True).clamp_min(1000.0)


def household_ratios(x: torch.Tensor, features) -> tuple[torch.Tensor, tuple]:
    """Append each household's mean consumption, liquid and illiquid wealth
    relative to its mean income, as constants repeated across waves.

    Per-household, per-feature z-scoring (``--per_sequence``) removes each
    feature's level and scale separately, which destroys the ratios between
    features -- saving rates and wealth-to-income -- that identify theta. These
    three channels put the between-feature ratios back. Computed from dollar
    LEVELS, so call before any log transform. asinh scale: illiquid wealth
    relative to income runs from 0 to above 20 across households. Constant
    within a household, so they must be kept out of per-sequence normalisation
    (the skip list matches the ``ratio_`` prefix).
    """
    f = list(features)
    ybar = _mean_income(x, f)
    cols = [torch.asinh(x[..., f.index(k)].mean(1, keepdim=True) / ybar)
            for k in ("consumption", "liquid_assets", "illiquid_assets")]
    block = torch.cat(cols, dim=-1)[:, None, :].expand(-1, x.shape[1], -1)
    return torch.cat([x, block], dim=-1), tuple(f) + RATIO_FEATURES


def mean_income_channel(x: torch.Tensor, features) -> tuple[torch.Tensor, tuple]:
    """Append ``log(mean income)`` as one per-household constant channel.

    Paired with :func:`anchor_log`, it restores the level that anchoring
    removes: anchored ratio x mean income recovers every dollar value, so
    anchor + this channel carries exactly the information of the levels input,
    only presented differently (RESULTS.md 28).

    Log, because income is roughly log-normal and a raw value would be dominated
    by a few high earners. NOT normalised here: the embedder standardises every
    channel with fixed statistics computed once from the training data and
    stored with the model, so on PSID this channel is scaled by the training
    set's mean and sd, never re-estimated. Computed from dollar LEVELS, so call
    before any log or anchor transform. Constant within a household, so it is
    kept out of per-sequence normalisation.
    """
    f = list(features)
    v = torch.log(_mean_income(x, f))[:, None, :].expand(-1, x.shape[1], -1)
    return torch.cat([x, v], dim=-1), tuple(f) + ("log_mean_income",)


def anchor_log(x: torch.Tensor, features) -> torch.Tensor:
    """Divide the four dollar features by the household's mean income, then log.

    Removes only the absolute dollar scale while keeping every ratio between
    features and the shape of each series over time: the anchor is ONE number
    per household, shared by all features, unlike per-feature z-scoring.

    Income and consumption are positive, so they enter as ``log(v / anchor)``
    (floored at 1% of the anchor). Liquid and illiquid wealth can be zero or
    negative, so they enter as ``asinh(v / anchor)``, the signed log. Age and
    any appended channels are untouched.
    """
    f = list(features)
    ybar = _mean_income(x, f)
    out = x.clone()
    # Only the dollar features present: a feature set may drop one (e.g.
    # noilliq_age), and income must be there to be the anchor.
    for k in ("income", "consumption"):
        if k in f:
            i = f.index(k)
            out[..., i] = torch.log((x[..., i] / ybar).clamp_min(0.01))
    for k in ("liquid_assets", "illiquid_assets"):
        if k in f:
            i = f.index(k)
            out[..., i] = torch.asinh(x[..., i] / ybar)
    return out


def log_features(x: torch.Tensor, features) -> torch.Tensor:
    """Signed log1p on the dollar features; ``age`` and one-hots untouched.

    Two independent reasons, and the first is the one that matters for PSID:

    1. **It makes extrapolation visible.** The embedder normalises each feature
       globally. In levels, a PSID household at the 1st percentile of income
       sits 1.6 sd below the simulated mean -- unremarkable -- so the network
       answers confidently on an input far outside its training support. In
       logs the same household is several sd out, which is what it actually is.
    2. **It is the scale the simulator is Gaussian on.** Income is generated
       lognormal, so simulated log income has skew -0.11 against +1.24 in
       levels.

    Signed rather than plain log1p because ``liquid_assets`` is negative
    whenever a household is borrowing on the card, which is the margin beta
    rides on.
    """
    idx = [i for i, f in enumerate(features) if f in DOLLAR_FEATURES]
    out = x.clone()
    v = x[..., idx]
    out[..., idx] = torch.sign(v) * torch.log1p(v.abs())
    return out


def split_shards(shard_files: list[Path], train_n: int):
    """Shards below the panel cutoff train; the rest are held out."""
    train, held = [], []
    for sf in shard_files:
        (train if int(np.load(sf)["lo"]) < train_n else held).append(sf)
    if not train or not held:
        raise SystemExit(
            f"need shards on both sides of panel {train_n}; got {len(train)} "
            f"train and {len(held)} held out"
        )
    return train, held


def simulate_sbc_once(n_sbc: int, seed: int, solver_config: dict,
                      cache: Path | None = None, educ: str = "comphs",
                      card_types: bool = False,
                      rgamma_range: tuple[float, float] | None = None,
                      init_pool: np.ndarray | None = None):
    """One GPU pass; the panels are re-windowed per window afterwards.

    Cached to disk because this is hours of GPU and everything downstream is
    seconds: a crash after the solves should cost a rerun of the seconds, not
    of the hours. It did once.

    ``educ`` must match how the training set was generated. SBC asks whether the
    posterior is calibrated for *its own* generative process; validating a
    marginalised posterior against comphs-only simulations would measure the
    mismatch instead of the calibration, and would look like miscalibration that
    no amount of training could fix. It is part of the cache key for the same
    reason -- silently reusing a comphs cache under ``mixed`` is the exact trap.

    ``card_types`` likewise must match generation (``generate_dataset.py
    --card_types``): 50/50 cardholder / no card, from its own stream
    (seed + 1). Also part of the cache key. The per-draw type is stored in the
    cache as ``card_idx`` (1 = cardholder).

    ``rgamma_range`` adds R_gamma as a fourth parameter (RESULTS.md 32), uniform
    on the range, drawn per block of ``theta_batch`` draws exactly as generation
    does -- the solver needs one R_gamma per batch -- with the card type then
    also per block. Returned thetas have four columns. Draws within a block share
    R_gamma, so the R_gamma rank test has ~n_sbc / theta_batch independent
    values; its coverage estimate is unaffected, its KS p-value optimistic.

    ``init_pool`` draws each household's age-20 wealth from a seed pool, as
    ``generate_dataset.py --init_pool`` does (RESULTS 42-43). Part of the cache
    key by hash, for the same reason as the others.
    """
    pool_key = None
    if init_pool is not None:
        import hashlib
        init_pool = np.ascontiguousarray(init_pool, dtype=np.float64)
        pool_key = hashlib.sha256(init_pool.tobytes()).hexdigest()
    if cache is not None and cache.exists():
        d = torch.load(cache, weights_only=False)
        cached_educ = d.get("educ", "comphs")
        cached_card = d.get("card_types", False)
        cached_rg = d.get("rgamma_range")
        if (d["n_sbc"] == n_sbc and d["seed"] == seed and cached_educ == educ
                and cached_card == card_types
                and (None if cached_rg is None else tuple(cached_rg))
                == (None if rgamma_range is None else tuple(rgamma_range))
                and d.get("init_pool_sha256") == pool_key):
            log.info(f"Reusing {n_sbc} cached SBC simulations from {cache}")
            return d["thetas"], d["panels"], d.get("educ_idx")
        log.warning(f"{cache} holds n_sbc={d['n_sbc']} seed={d['seed']} "
                    f"educ={cached_educ} card_types={cached_card}; need "
                    f"{n_sbc}/{seed}/{educ}/{card_types}. "
                    f"Re-simulating.")
    prior = make_sbi_prior(PHASE3)
    torch.manual_seed(seed)
    thetas = prior.sample((n_sbc,))
    educ_idx = (None if educ == "comphs" else
                np.random.default_rng(seed).integers(
                    0, len(cal.EDUC_GROUPS), size=n_sbc))
    card_idx = (np.random.default_rng(seed + 1).integers(0, 2, size=n_sbc)
                if card_types else None)
    r_gamma = None
    if rgamma_range is not None:
        tb = int(solver_config["theta_batch"])
        n_blocks = -(-n_sbc // tb)
        rng = np.random.default_rng(seed + 2)
        lo, hi = rgamma_range
        r_gamma = np.repeat(rng.uniform(lo, hi, size=n_blocks), tb)[:n_sbc]
        if card_types:
            card_idx = np.repeat(np.random.default_rng(seed + 1).integers(
                0, 2, size=n_blocks), tb)[:n_sbc]
    t0 = time.time()
    # n_waves here only sizes the throwaway `x`; the panels are what we keep,
    # and they get windowed per arm afterwards.
    _x, _alive, panels = simulate_batch_twoasset_gpu(
        thetas.numpy(), seed, 30, n_waves=5, wave_years=WAVE_YEARS,
        grid=solver_config.get("grid", "full"),
        theta_batch=solver_config["theta_batch"],
        chunk=solver_config["chunk"],
        return_panels=True, educ=educ_idx, card=card_idx, r_gamma=r_gamma,
        init_pool=init_pool,
    )
    if r_gamma is not None:
        thetas = torch.cat([thetas, torch.as_tensor(r_gamma, dtype=thetas.dtype)[:, None]], 1)
    log.info(f"SBC simulations done in {(time.time() - t0) / 3600:.2f} h")
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"thetas": thetas, "panels": panels, "n_sbc": n_sbc,
                    "seed": seed, "solver_config": solver_config,
                    "educ": educ, "educ_idx": educ_idx,
                    "card_types": card_types, "card_idx": card_idx,
                    "rgamma_range": rgamma_range,
                    "init_pool_sha256": pool_key}, cache)
        log.info(f"Cached SBC simulations to {cache}")
    return thetas, panels, educ_idx


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--windows", type=int, nargs="+", default=[5, 10, 15])
    p.add_argument("--shards", type=Path,
                   default=Path("data/processed/phase3_dataset_shards"))
    p.add_argument("--dataset", type=Path,
                   default=Path("data/processed/phase3_dataset.pt"),
                   help="Only used to locate the recorded solver config.")
    p.add_argument("--train_n", type=int, default=57344,
                   help="Draws used for training; the rest are held out. "
                        "57344 = 65536 - 8192: both sides are unions of "
                        "power-of-2 Sobol blocks, so both stay balanced. Was "
                        "65536 while generation was still running and later "
                        "shards supplied free held-out draws; with the dataset "
                        "complete that leaves nothing held out.")
    p.add_argument("--n_total", type=int, default=65536)
    p.add_argument("--n_sbc", type=int, default=1000,
                   help="SBC simulations. These are fresh GPU solves and the "
                        "dominant cost: ~12.4 s each, shared across windows.")
    p.add_argument("--n_post", type=int, default=1000)
    p.add_argument("--n_heldout_eval", type=int, default=2048,
                   help="Held-out draws scored for estimation metrics.")
    p.add_argument("--start_low", type=int, default=25)
    p.add_argument("--start_high", type=int, default=40,
                   help="Start ages are drawn from [low, high] for every arm, "
                        "so the age distribution is identical and only the "
                        "window length differs. Note this makes the longer "
                        "arms reach further past retirement -- reported.")
    p.add_argument("--k", type=int, default=8,
                   help="Windows per panel (augmentation).")
    p.add_argument("--no_age", action="store_true",
                   help="Drop the per-wave age channel.")
    p.add_argument("--obs_noise", type=float, default=0.0,
                   help="Multiplicative lognormal observation noise on the four "
                        "dollar features, sigma in logs. Models PSID reporting "
                        "and imputation error, which the simulated data has "
                        "none of. Applied to the TRAINING, HELD-OUT and SBC "
                        "windows alike -- the observation model must be the "
                        "same everywhere or calibration measures the mismatch "
                        "instead of the posterior. `age` is exact in PSID and "
                        "is never perturbed.")
    p.add_argument("--per_sequence", action="store_true",
                   help="Normalise each dollar feature within a household, "
                        "across its waves: (x - hh mean) / hh sd. A "
                        "proportional measurement bias then cancels exactly, "
                        "which is the PSID consumption problem -- but levels "
                        "are removed, and levels are what Laibson et al.'s "
                        "moments are made of. `age` is EXCLUDED and keeps the "
                        "global scale: it advances by wave_years every wave, so "
                        "per-household normalisation maps every household to "
                        "the same ramp and destroys the channel outright.")
    p.add_argument("--features", type=str, default=None,
                   choices=sorted(FEATURE_SETS),
                   help="Named feature set (hh_npe.data.waves.FEATURE_SETS). "
                        "Default follows --no_age. 'nocons_age' drops "
                        "consumption, which is Laibson et al.'s own "
                        "information set -- their 16 moments use only "
                        "credit-card borrowing and wealth. Applied to the "
                        "training, held-out AND SBC windows together; scoring "
                        "a posterior on a different feature set than it was "
                        "trained on fails on shape, but scoring it on the same "
                        "features cut a different way would not.")
    p.add_argument("--batch_size", type=int, default=256,
                   help="Training minibatch. The default starves the GPU: the "
                        "model is ~200k parameters on (batch, waves, 5) inputs, "
                        "so a V100 sits near 11%% and ~98%% of wall time is "
                        "kernel-launch latency, not compute. Raising this cuts "
                        "step count proportionally -- scale --learning_rate with "
                        "it, and keep both fixed across every arm of a "
                        "comparison.")
    p.add_argument("--learning_rate", type=float, default=5e-4)
    p.add_argument("--derived_features", action="store_true",
                   help="Append card_debt = 1{liquid<0}, liquid/income and "
                        "illiquid/income as extra channels. These are Laibson "
                        "et al.'s own moment definitions and are exact "
                        "functions of features already present, so this adds "
                        "no information -- it tests whether the flow is "
                        "spending capacity rediscovering them.")
    p.add_argument("--anchor_log", action="store_true",
                   help="Divide the four dollar features by the household's "
                        "mean income, then log (income, consumption) or asinh "
                        "(liquid, illiquid). Removes only the dollar scale; "
                        "keeps ratios between features. Not with "
                        "--log_features. RESULTS 27.")
    p.add_argument("--static_norm_channels", nargs="+", default=None,
                   help="Standardise ONLY these channels with the training "
                        "set's fixed mean/sd; all others pass through as "
                        "computed. Default: every channel (the behaviour of "
                        "every run before RESULTS 28). E.g. "
                        "--static_norm_channels age log_mean_income.")
    p.add_argument("--mean_income_channel", action="store_true",
                   help="Append log(household mean income) as one constant "
                        "channel, standardised by the embedder's fixed "
                        "training-set statistics. With --anchor_log it restores "
                        "the level anchoring removes. RESULTS 28.")
    p.add_argument("--household_ratios", action="store_true",
                   help="Append asinh(mean consumption / mean income), and the "
                        "same for liquid and illiquid wealth, as per-household "
                        "constants. Restores the between-feature ratios that "
                        "--per_sequence removes. RESULTS 27.")
    p.add_argument("--hidden_features", type=int, default=50,
                   help="Width of each flow transform's conditioner. sbi's "
                        "default, never swept here.")
    p.add_argument("--num_transforms", type=int, default=5,
                   help="Spline transforms stacked in the flow. sbi's default, "
                        "never swept here.")
    p.add_argument("--d_model", type=int, default=64,
                   help="Embedder width. --n_heads must divide it.")
    p.add_argument("--n_layers", type=int, default=2,
                   help="Embedder encoder layers.")
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--embed_dim", type=int, default=32,
                   help="Embedder output width, i.e. how many numbers the flow "
                        "sees in place of the raw trajectory.")
    p.add_argument("--train_seed", type=int, default=0,
                   help="Seeds network init and batch order only. The dataset, "
                        "the panel split and the SBC draws have their own fixed "
                        "seeds, so this isolates optimization noise -- use it to "
                        "tell a real calibration difference from scatter.")
    p.add_argument("--out", type=Path, default=Path("outputs/window_comparison"))
    p.add_argument("--educ", choices=["comphs", "mixed"], default="comphs",
                   help="How the training shards were generated. SBC "
                        "simulations must use the same process, or calibration "
                        "measures the mismatch rather than the posterior. Part "
                        "of the --sbc_cache key.")
    p.add_argument("--log_features", action="store_true",
                   help="Signed log1p the dollar features. Changes only how x "
                        "is presented to the network -- theta, the shards and "
                        "the simulator are untouched -- so recovery metrics "
                        "stay directly comparable to a levels run.")
    p.add_argument("--educ_group", choices=["comphs", "somehs", "compco"],
                   default=None,
                   help="Train on one education group only, for the per-group "
                        "posteriors. Requires shards generated with "
                        "--educ mixed.")
    p.add_argument("--condition_educ", action="store_true",
                   help="Append a one-hot education block to x, so the "
                        "posterior is q(theta | x, edu) rather than the "
                        "marginalised q(theta | x). We observe education in "
                        "PSID, so conditioning is legitimate and strictly "
                        "sharper -- the same treatment `age` already gets. "
                        "Marginalising an observed variable buys width by "
                        "discarding information (RESULTS.md 10.10).")
    p.add_argument("--sbc_cache", type=Path,
                   default=Path("outputs/window_comparison/sbc_sims.pt"),
                   help="Where the SBC panels are cached. Reused if it matches "
                        "--n_sbc and the seed.")
    p.add_argument("--skip_sbc", action="store_true",
                   help="Estimation metrics only; skips the GPU simulations.")
    args = p.parse_args()
    if args.anchor_log and args.log_features:
        raise SystemExit("--anchor_log and --log_features are alternatives; "
                         "anchor_log already takes the log.")

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)

    # Frozen once: generation may still be running and writing new shards, and
    # every window must see the same data for the comparison to mean anything.
    shard_files = sorted(args.shards.glob("shard_*.npz"))
    if not shard_files:
        raise SystemExit(f"no shards in {args.shards}")
    log.info(f"Shard list frozen at {len(shard_files)} shards "
             f"({shard_files[0].name}..{shard_files[-1].name})")

    # Before the SBC solves, not after: this is a millisecond check guarding
    # hours of GPU, and running it second once cost 3.46 h of simulations that
    # were still only in memory.
    train_sh, held_sh = split_shards(shard_files, args.train_n)
    log.info(f"{len(train_sh)} shards train (< panel {args.train_n}), "
             f"{len(held_sh)} held out")

    sbc_thetas = sbc_panels = sbc_educ = None
    if not args.skip_sbc:
        cfg = read_solver_config(args.dataset)
        if not cfg or cfg.get("device") != "cuda":
            raise SystemExit(
                "No CUDA solver config beside the dataset. SBC must reproduce "
                "the training set's simulator exactly; refusing to guess."
            )
        log.info(f"Simulating {args.n_sbc} SBC draws once, shared across "
                 f"windows (solver config: {cfg})")
        sbc_thetas, sbc_panels, sbc_educ = simulate_sbc_once(
            args.n_sbc, 20260822, cfg, cache=args.sbc_cache, educ=args.educ)

    theta_all = sample_sobol(args.n_total, PHASE3, seed=0)
    win = dict(start_low=args.start_low, start_high=args.start_high,
               wave_years=WAVE_YEARS, with_age=not args.no_age,
               features=FEATURE_SETS[args.features] if args.features else None)

    results = {}
    for k in args.windows:
        limit = max_start_age(k, WAVE_YEARS)
        if args.start_high > limit:
            raise SystemExit(
                f"{k} waves cannot start as late as {args.start_high}; the "
                f"panel runs out at {limit}."
            )
        end_lo = args.start_low + WAVE_YEARS * k - 1
        end_hi = args.start_high + WAVE_YEARS * k - 1
        ages = f"{args.start_low}-{args.start_high} start, ends {end_lo}-{end_hi}"

        th_tr, x_tr, pid_tr = build_windowed(
            train_sh, theta_all, k=args.k, n_waves=k, seed=0, **win
        )
        th_ho, x_ho, pid_ho = build_windowed(
            held_sh, theta_all, k=1, n_waves=k, seed=999, **win
        )
        feats = (FEATURE_SETS[args.features] if args.features
                 else (FEATURES_TWOASSET if args.no_age else FEATURES_TWOASSET_AGE))
        base_feats = feats
        x_tr = add_obs_noise(x_tr, args.obs_noise, feats, seed=11)
        x_ho = add_obs_noise(x_ho, args.obs_noise, feats, seed=22)
        if args.derived_features:
            x_tr, feats = derived_features(x_tr, feats)
            x_ho, _ = derived_features(x_ho, base_feats)
            log.info(f"derived features appended: {x_tr.shape[-1]} features")
        if args.mean_income_channel:
            # From dollar levels, so before any log or anchor.
            x_tr, feats_m = mean_income_channel(x_tr, feats)
            x_ho, _ = mean_income_channel(x_ho, feats)
            feats = feats_m
            log.info(f"log mean income appended: {x_tr.shape[-1]} features")
        if args.household_ratios:
            # From dollar levels, so before any log.
            x_tr, feats_r = household_ratios(x_tr, feats)
            x_ho, _ = household_ratios(x_ho, feats)
            feats = feats_r
            log.info(f"household ratios appended: {x_tr.shape[-1]} features")
        if args.log_features:
            x_tr, x_ho = log_features(x_tr, feats), log_features(x_ho, feats)
        if args.anchor_log:
            x_tr, x_ho = anchor_log(x_tr, feats), anchor_log(x_ho, feats)

        if args.educ_group or args.condition_educ:
            by_draw = educ_by_draw(shard_files)
            if by_draw is None:
                raise SystemExit(
                    f"{args.shards} holds no `educ` field, so it was generated "
                    f"comphs-only. --educ_group/--condition_educ need shards "
                    f"from a --educ mixed run."
                )
            e_tr = by_draw[pid_tr.numpy()]
            e_ho = by_draw[pid_ho.numpy()]
        if args.educ_group:
            g = cal.EDUC_GROUPS.index(args.educ_group)
            # Filter by draw, which keeps every household of a kept draw
            # together -- the grouped split still sees whole panels.
            m_tr, m_ho = e_tr == g, e_ho == g
            th_tr, x_tr, pid_tr = th_tr[m_tr], x_tr[m_tr], pid_tr[m_tr]
            th_ho, x_ho, pid_ho = th_ho[m_ho], x_ho[m_ho], pid_ho[m_ho]
            e_tr, e_ho = e_tr[m_tr], e_ho[m_ho]
            log.info(f"education filter {args.educ_group}: "
                     f"{len(th_tr)} train rows from "
                     f"{len(pid_tr.unique())} draws, {len(th_ho)} held out")
        if args.condition_educ:
            x_tr = one_hot_educ(x_tr, e_tr)
            x_ho = one_hot_educ(x_ho, e_ho)
            feats = tuple(feats) + ("educ_comphs", "educ_somehs", "educ_compco")
            log.info("conditioning on education: x gains a one-hot block, "
                     f"{x_tr.shape[-1]} features")
        th_ho, x_ho = th_ho[: args.n_heldout_eval], x_ho[: args.n_heldout_eval]
        n_panels = len(pid_tr.unique())
        log.info(f"=== {k} waves ({ages}) | train {len(th_tr)} windows from "
                 f"{n_panels} panels | held-out scored {len(th_ho)} ===")
        if end_hi > AGE_RETIRE:
            # Stated per arm rather than assumed: the forward pass applies no
            # mortality, so windows reaching past retirement describe a cohort
            # in which everyone survives. That flatters the longer arms.
            log.warning(
                f"  {k}w windows reach age {end_hi}, past retirement at "
                f"{AGE_RETIRE}. The forward pass has no mortality, so this arm "
                f"gets a survivorship advantage the shorter arms do not."
            )

        # Identical init and batch order for every arm at a given --train_seed.
        # Varying only this seed holds the data, the panel split and the SBC
        # draws fixed (their seeds are passed explicitly above), so a spread in
        # calibration across seeds is optimization noise and nothing else.
        seed_all(args.train_seed)
        # `feats` is NOT recomputed here. It has been widened above by
        # --derived_features and --condition_educ, and recomputing it from the
        # CLI arguments silently drops those names -- which would leave `skip`
        # without the education one-hot indices and let per-sequence
        # normalisation divide a constant block by its zero spread.
        # Derived, not hardcoded: the age column moves with the feature set.
        # `age` and the one-hot education block are constant within a
        # sequence; per-sequence normalisation would divide by a zero spread.
        skip = tuple(i for i, f in enumerate(feats)
                     if f == "age" or f.startswith("educ_")
                     or f == "card_debt" or f.startswith("ratio_")
                     or f == "log_mean_income")
        f_mean, f_std = x_tr.mean(dim=(0, 1)), x_tr.std(dim=(0, 1))
        if args.static_norm_channels is not None:
            # Only the listed channels are standardised by the training set's
            # fixed statistics; every other channel passes through exactly as
            # computed (mean 0, sd 1 is the identity). For scale-free inputs
            # this keeps training-set constants out of them entirely, so a
            # household outside the training range is not rescaled by numbers
            # fitted to it. RESULTS 28.
            unknown = set(args.static_norm_channels) - set(feats)
            if unknown:
                raise SystemExit(f"--static_norm_channels names unknown "
                                 f"channels {sorted(unknown)}; have {list(feats)}")
            keep = torch.tensor([f in args.static_norm_channels for f in feats])
            f_mean = torch.where(keep, f_mean, torch.zeros_like(f_mean))
            f_std = torch.where(keep, f_std, torch.ones_like(f_std))
            log.info(f"static normalisation only for {args.static_norm_channels}")
        embedder = TrajectoryTransformer(
            n_features=x_tr.shape[-1], seq_len=k,
            feature_mean=f_mean, feature_std=f_std,
            per_sequence=args.per_sequence, per_sequence_skip=skip,
            **{**EMBEDDER, "d_model": args.d_model, "n_heads": args.n_heads,
               "n_layers": args.n_layers, "output_dim": args.embed_dim},
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        training = {**TRAINING, "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate,
                    "hidden_features": args.hidden_features,
                    "num_transforms": args.num_transforms}
        post, _de, _inf = train_npe(
            th_tr, x_tr, embedder=embedder, box=PHASE3, device=device,
            group_ids=pid_tr, **training
        )
        save_posterior(post, embedder, PHASE3, args.out / f"posterior_{k}w.pt")

        torch.manual_seed(0)
        per_param, log_q = estimation_scores(post, PHASE3, th_ho, x_ho, n_post=400)
        entry = {"ages": ages, "n_train": len(th_tr), "n_panels": n_panels,
                 "n_heldout": len(th_ho), "ends_past_retirement": end_hi > AGE_RETIRE,
                 "estimation": per_param, "held_out_log_q": log_q}

        if sbc_panels is not None:
            # One window per SBC draw, cut the way a training window was.
            th_sbc, x_sbc, ids = window_panel(
                sbc_panels, sbc_thetas.numpy(), k=1, n_waves=k, seed=4242, **win
            )
            x_sbc = add_obs_noise(x_sbc, args.obs_noise, base_feats, seed=33)
            if args.derived_features:
                x_sbc, _ = derived_features(x_sbc, base_feats)
            if args.mean_income_channel:
                x_sbc, _ = mean_income_channel(x_sbc, base_feats)
            if args.household_ratios:
                x_sbc, _ = household_ratios(x_sbc, base_feats)
            if args.log_features:
                x_sbc = log_features(x_sbc, base_feats)
            if args.anchor_log:
                x_sbc = anchor_log(x_sbc, base_feats)
            # SBC x must go through exactly the transformations training x did,
            # or the posterior is handed a different feature count than it was
            # built for. One SBC draw is one household, so `ids` is the draw
            # index and indexes `sbc_educ` directly.
            if (args.educ_group or args.condition_educ) and sbc_educ is None:
                raise SystemExit(
                    f"{args.sbc_cache} carries no per-draw education, so SBC "
                    f"cannot be filtered or conditioned to match training. "
                    f"Re-simulate it with --educ mixed."
                )
            if args.educ_group:
                m = np.asarray(sbc_educ)[ids.numpy()] == cal.EDUC_GROUPS.index(
                    args.educ_group)
                th_sbc, x_sbc, ids = th_sbc[m], x_sbc[m], ids[m]
            if args.condition_educ:
                x_sbc = one_hot_educ(x_sbc, np.asarray(sbc_educ)[ids.numpy()])
            entry["calibration"] = calibration_scores(
                post, PHASE3, th_sbc, x_sbc,
                n_post=args.n_post, out_dir=args.out, tag=f"{k}w",
            )
            entry["n_sbc"] = len(th_sbc)
        results[k] = entry
        log.info(f"  {k}w done: log q = {log_q:.3f}")

    # Self-describing: arms are now compared across age envelopes, batch sizes
    # and training seeds, and a results.json that records only the numbers is
    # one filename away from being read as something it is not.
    results["_config"] = {
        "start_low": args.start_low, "start_high": args.start_high,
        "k_windows_per_panel": args.k, "wave_years": WAVE_YEARS,
        "train_n": args.train_n, "train_seed": args.train_seed,
        "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "hidden_features": args.hidden_features,
        "num_transforms": args.num_transforms,
        "d_model": args.d_model, "n_heads": args.n_heads,
        "n_layers": args.n_layers, "embed_dim": args.embed_dim,
        "per_sequence": args.per_sequence,
        "obs_noise": args.obs_noise,
        "educ_group": args.educ_group,
        "condition_educ": args.condition_educ,
        "log_features": args.log_features,
        "derived_features": args.derived_features,
        "anchor_log": args.anchor_log,
        "household_ratios": args.household_ratios,
        "mean_income_channel": args.mean_income_channel,
        "static_norm_channels": args.static_norm_channels,
        # Provenance, so a later scorer can verify it is rebuilding the
        # evaluation windows from the data the members actually saw. Omitting
        # this is how a Phase 4 ensemble came to be scored on Phase 3 shards.
        "shards": str(args.shards),
        "sbc_cache": str(args.sbc_cache),
        "features": list(FEATURE_SETS[args.features]) if args.features else None,
        "n_sbc": args.n_sbc, "n_post": args.n_post,
        "n_heldout_eval": args.n_heldout_eval,
        "age_envelope": [args.start_low,
                         max(args.start_high + WAVE_YEARS * k - 1
                             for k in args.windows)],
    }
    (args.out / "results.json").write_text(json.dumps(results, indent=2))
    _report(results, args.windows)


def _report(results: dict, windows: list[int]) -> None:
    hdr = "".join(f"{f'{k}w':>12s}" for k in windows)
    print(f"\n{'':10s}{hdr}")
    print(f"{'ends':10s}" + "".join(
        f"{results[k]['ages'].split('ends ')[-1]:>12s}" for k in windows))
    print(f"{'windows':10s}" + "".join(f"{results[k]['n_train']:12d}"
                                       for k in windows))
    print(f"{'panels':10s}" + "".join(f"{results[k]['n_panels']:12d}"
                                      for k in windows))

    for metric, fmt in (("contraction", "12.3f"), ("corr", "12.3f"),
                        ("mae", "12.4f")):
        print(f"\n=== {metric} ===")
        for n in PHASE3.names:
            row = "".join(
                f"{results[k]['estimation'][n][metric]:{fmt}}" for k in windows
            )
            print(f"{n:10s}{row}")

    print("\n=== held-out log q(theta_true | x) ===")
    print(f"{'':10s}" + "".join(f"{results[k]['held_out_log_q']:12.3f}"
                                for k in windows))

    flagged = [k for k in windows if results[k]["ends_past_retirement"]]
    if flagged:
        print(f"\nNOTE: arms {flagged} reach past retirement at {AGE_RETIRE}. "
              f"The forward pass applies no mortality, so those windows train\n"
              f"      on a cohort where everyone survives -- an advantage the "
              f"shorter arms do not get. Discount accordingly.")
    print("\nEffective independent sample is the panel count, not the window "
          "count: augmentation teaches\nthe age mapping, it adds no information "
          "about theta.")

    if "calibration" not in results[windows[0]]:
        return
    for metric, fmt, target in (("coverage_90", "12.3f", " (target 0.900)"),
                                ("ks_p", "12.4f", " (>0.05 = uniform)")):
        print(f"\n=== SBC {metric}{target} ===")
        for n in PHASE3.names:
            row = "".join(
                f"{results[k]['calibration'][n][metric]:{fmt}}" for k in windows
            )
            print(f"{n:10s}{row}")


if __name__ == "__main__":
    main()
