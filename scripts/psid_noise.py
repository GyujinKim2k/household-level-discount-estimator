"""The measurement-error model of "psid noise methodology.md", as written (RESULTS 67).

The file (untracked, not ours) describes a module ``psid_noise.py`` that is
not on this machine; this is that model built from its text, for simulated
windows in ``FEATURES_TWOASSET_AGE`` order. Every error multiplies the
magnitude, ``x_obs = sign(x*) |x*| exp(u)``, so zeros stay zero and signs are
kept; age is never touched; the four errors are independent. Each household
(row) draws its own error parameters from the file's priors (its section 0:
"draw the error parameters from the priors on every simulation and
marginalize over them"):

* **income**: ``u_t = b (log y*_t - ref_t) + sigma_v z_t``, z an AR(1) across
  waves (``z_t = rho z_{t-1} + sqrt(1 - rho^2) eps_t``, stationary), with
  sigma_v ~ U(0.08, 0.14), b ~ U(-0.20, 0), rho ~ U(0, 0.3). ``ref_t`` is the
  file's "wave's cross-sectional mean of log y*": for windows, the mean clean
  log income at each window position over the training windows, fixed once
  (:func:`income_reference`) so every scoring uses the same reference;
* **consumption**: ``u ~ N(0, sigma_c^2)`` iid across waves, sigma_c ~ U(0.12, 0.25);
* **liquid and illiquid wealth**: ``u = sigma t_nu``, a unit-variance Student
  t clipped at ``|u| <= 3`` (the file's "(~20x)": log points), iid across
  waves, nu ~ U(3, 8); ``sigma^2 = s V / 2`` with share s ~ U(0.3, 0.7) and V
  the variance of one-wave changes of log wealth in PSID couples
  (:func:`calibrate_asset_sigma` with ``observed=True``, the file's section
  4.2 and open item 2).

Not built, as in the file: transfers' non-reporting and income timing (its
TODOs), illiquid wealth by component (the model has one illiquid asset), and
no conversion to the network's input scale (RESULTS 57.4's; the file has none).

Commands:

``calibrate``
    PSID couples' V for liquid and illiquid wealth, and the income reference
    from the training windows, to ``outputs/noise_copies/psid_file_noise.json``
    (aggregate numbers only).

``check``
    Diagnostic: simulated couples' windows under v2's noise and under the
    file's, read with ``psid_iid_error``'s estimator beside PSID couples'
    readings (RESULTS 57.2), on log|value|.

Usage::

    PYTHONPATH=.:src .venv/bin/python scripts/psid_noise.py calibrate
    PYTHONPATH=.:src .venv/bin/python scripts/psid_noise.py check
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from hh_npe.data.waves import FEATURES_TWOASSET_AGE

#: The file's priors (its section 0).
PRIORS = {"sigma_v": (0.08, 0.14), "b": (-0.20, 0.0), "rho": (0.0, 0.3),
          "sigma_c": (0.12, 0.25), "share": (0.3, 0.7), "nu": (3.0, 8.0)}
#: |u| <= CLIP on wealth errors, in log points (exp(3) ~ 20x).
CLIP = 3.0
#: ``calibrate_asset_sigma`` refuses fewer wave-to-wave pairs (the file's 4.2).
MIN_PAIRS = 50
ASSETS = ("liquid_assets", "illiquid_assets")
INC = FEATURES_TWOASSET_AGE.index("income")
CONS = FEATURES_TWOASSET_AGE.index("consumption")
OUT = Path("outputs/noise_copies/psid_file_noise.json")
PSID_X = Path("data/processed/psid_x_comphs_couples.pt")
SHARDS = (Path("data/processed/couples_v2_dataset_shards"),
          Path("data/processed/couples_v2_strip_dataset_shards"),
          Path("data/processed/couples_v2_betastrip_dataset_shards"))


@dataclasses.dataclass(frozen=True)
class FileNoise:
    """The file's error model with its two PSID-calibrated inputs: ``var``,
    PSID couples' variance of one-wave changes of log wealth per asset, and
    ``income_ref``, the reference log income at each window position.
    Hashable, so a :class:`optionA.Spec` can hold it as its ``obs_noise``."""

    var_liquid: float
    var_illiquid: float
    income_ref: tuple
    priors: tuple = tuple(sorted(PRIORS.items()))
    clip: float = CLIP
    sha256: str = ""

    def prior(self, name: str) -> tuple[float, float]:
        return dict(self.priors)[name]

    def var(self, f: str) -> float:
        return {"liquid_assets": self.var_liquid, "illiquid_assets": self.var_illiquid}[f]

    def record(self) -> dict:
        """The JSON a run's config stores (``noise_model``)."""
        return {"model": "psid_file", "var_liquid": self.var_liquid,
                "var_illiquid": self.var_illiquid, "income_ref": list(self.income_ref),
                "priors": {k: list(v) for k, v in self.priors}, "clip": self.clip,
                "sha256": self.sha256}

    @classmethod
    def from_record(cls, d: dict) -> FileNoise:
        if d.get("model") != "psid_file":
            raise ValueError(f"not the file's noise model: {d.get('model')!r}")
        return cls(var_liquid=float(d["var_liquid"]), var_illiquid=float(d["var_illiquid"]),
                   income_ref=tuple(float(v) for v in d["income_ref"]),
                   priors=tuple(sorted((k, tuple(map(float, v)))
                                       for k, v in d["priors"].items())),
                   clip=float(d["clip"]), sha256=d.get("sha256", ""))

    @classmethod
    def load(cls, path) -> FileNoise:
        """From ``calibrate``'s JSON, recording the file's sha256."""
        raw = Path(path).read_bytes()
        return dataclasses.replace(cls.from_record(json.loads(raw)),
                                   sha256=hashlib.sha256(raw).hexdigest())

    def describe(self) -> str:
        return (f"the methodology file's noise (income sigma_v {self.prior('sigma_v')}, "
                f"b {self.prior('b')}, rho {self.prior('rho')}; consumption "
                f"{self.prior('sigma_c')}; wealth t, nu {self.prior('nu')}, share "
                f"{self.prior('share')} of V liquid {self.var_liquid:.3f}, illiquid "
                f"{self.var_illiquid:.3f})")


def sample_noise_params(rng: np.random.Generator, n: int, model: FileNoise) -> dict:
    """One draw of every error parameter per row, uniform on the priors, in a
    fixed order."""
    out = {}
    for name in ("sigma_v", "b", "rho", "sigma_c"):
        lo, hi = model.prior(name)
        out[name] = rng.uniform(lo, hi, n)
    for f in ASSETS:
        lo, hi = model.prior("share")
        out[f"share_{f}"] = rng.uniform(lo, hi, n)
        lo, hi = model.prior("nu")
        out[f"nu_{f}"] = rng.uniform(lo, hi, n)
        out[f"sigma_{f}"] = np.sqrt(out[f"share_{f}"] * model.var(f) / 2)
    return out


def unit_t(rng: np.random.Generator, nu: np.ndarray, shape: tuple) -> np.ndarray:
    """Student t draws rescaled to unit variance, ``nu`` per row (> 2)."""
    nu = np.asarray(nu, dtype=float)[:, None]
    return rng.standard_t(np.broadcast_to(nu, shape)) * np.sqrt((nu - 2) / nu)


def errors(x: np.ndarray, model: FileNoise, rng: np.random.Generator) -> dict:
    """The log errors u per noisy feature, (rows, waves) each, and the
    parameters drawn."""
    n, waves = x.shape[:2]
    if len(model.income_ref) != waves:
        raise ValueError(f"income reference has {len(model.income_ref)} waves, "
                         f"the windows {waves}")
    p = sample_noise_params(rng, n, model)
    y = x[..., INC]
    pos = y > 0
    logy = np.log(np.where(pos, y, 1.0))
    eps = rng.standard_normal((n, waves))
    z = np.empty_like(eps)
    z[:, 0] = eps[:, 0]
    r = p["rho"][:, None]
    for t in range(1, waves):
        z[:, t] = r[:, 0] * z[:, t - 1] + np.sqrt(1 - r[:, 0] ** 2) * eps[:, t]
    ref = np.asarray(model.income_ref)[None]
    u = {"income": np.where(pos, p["b"][:, None] * (logy - ref), 0.0) + p["sigma_v"][:, None] * z,
         "consumption": p["sigma_c"][:, None] * rng.standard_normal((n, waves))}
    for f in ASSETS:
        t = unit_t(rng, p[f"nu_{f}"], (n, waves))
        u[f] = np.clip(p[f"sigma_{f}"][:, None] * t, -model.clip, model.clip)
    return {"u": u, "params": p}


def add_psid_noise(x: torch.Tensor, model: FileNoise, seed: int) -> torch.Tensor:
    """``x`` (rows, waves, features) raw windows with the file's errors; the
    same seed gives the same draw (``np.random.default_rng(seed)``)."""
    xn = x.detach().cpu().numpy().astype(np.float64)
    e = errors(xn, model, np.random.default_rng(seed))
    out = xn.copy()
    for f, u in e["u"].items():
        j = FEATURES_TWOASSET_AGE.index(f)
        out[..., j] = xn[..., j] * np.exp(u)
    return torch.from_numpy(out).to(x.dtype)


def calibrate_asset_sigma(panel: np.ndarray, observed: bool = True,
                          share: float = 0.5) -> dict:
    """The file's section 4.2: V = ((p75 - p25) / 1.349)^2 of one-wave changes
    of log A over pairs where both waves are positive, and sigma^2 = s V / 2
    (``observed``: V is of observed data) or s / (1 - s) V / 2 (V of the
    truth). Raises with fewer than MIN_PAIRS pairs."""
    a = np.asarray(panel, dtype=float)
    both = (a[:, 1:] > 0) & (a[:, :-1] > 0)
    d = (np.log(np.where(a[:, 1:] > 0, a[:, 1:], 1.0))
         - np.log(np.where(a[:, :-1] > 0, a[:, :-1], 1.0)))[both]
    if len(d) < MIN_PAIRS:
        raise ValueError(f"{len(d)} wave-to-wave pairs with both waves positive; "
                         f"need {MIN_PAIRS}")
    q25, q75 = np.percentile(d, [25, 75])
    v = float(((q75 - q25) / 1.349) ** 2)
    s2 = share * v / 2 if observed else share / (1 - share) * v / 2
    return {"var": v, "n_pairs": int(len(d)), "sigma2": float(s2), "sigma": float(np.sqrt(s2))}


def income_reference(x_clean: torch.Tensor) -> list[float]:
    """Mean clean log income at each window position, over rows with income > 0."""
    y = x_clean[..., INC].double()
    pos = y > 0
    logy = torch.where(pos, torch.log(torch.where(pos, y, torch.ones_like(y))),
                       torch.zeros_like(y))
    return (logy.sum(0) / pos.sum(0)).tolist()


def psid_couples(path: Path = PSID_X) -> np.ndarray:
    """PSID couples' raw x, comphs only as ``optionA.psid`` keeps them."""
    from hh_npe.simulator.laibson_calibration import EDUC_GROUPS

    d = torch.load(path, weights_only=False)
    if list(d["features"]) != list(FEATURES_TWOASSET_AGE):
        raise SystemExit(f"{path}: features {d['features']}")
    keep = np.flatnonzero(d["educ"].numpy() == EDUC_GROUPS.index("comphs"))
    return d["x"].numpy()[keep].astype(np.float64)


# ------------------------------------------------------------------ commands

def calibrate(args) -> None:
    from scripts import optionA as oa

    x = psid_couples(args.psid)
    res = {"psid": str(args.psid), "n_couples": int(len(x))}
    for f in ASSETS:
        c = calibrate_asset_sigma(x[..., FEATURES_TWOASSET_AGE.index(f)], observed=True)
        lo, hi = PRIORS["share"]
        res[f] = {"var": c["var"], "n_pairs": c["n_pairs"],
                  "sigma_range": [float(np.sqrt(lo * c["var"] / 2)),
                                  float(np.sqrt(hi * c["var"] / 2))]}
    theta_all, _card, train_f, _held, _p, cfg = oa.load_shards(list(args.shards))
    win = {**oa.WINDOW, "n_waves": args.n_waves, "start_low": args.start_low,
           "start_high": args.start_high, "wave_years": args.wave_years}
    _th, xc, _pid = oa.windows_raw(train_f, theta_all, seed=0, window=win)
    ref = income_reference(xc)
    res["income_ref"] = {"window": win, "rows": int(len(xc)), "ref": ref}
    model = FileNoise(var_liquid=res["liquid_assets"]["var"],
                      var_illiquid=res["illiquid_assets"]["var"], income_ref=tuple(ref))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({**model.record(), "calibration": res}, indent=1))
    print(f"PSID couples: {res['n_couples']}")
    for f in ASSETS:
        r = res[f]
        print(f"  {f:16s} V {r['var']:.4f} over {r['n_pairs']} pairs; sigma "
              f"{r['sigma_range'][0]:.3f}-{r['sigma_range'][1]:.3f} (share "
              f"{PRIORS['share'][0]}-{PRIORS['share'][1]})")
    print(f"income reference (mean clean log income by window position, {len(xc)} rows): "
          + " ".join(f"{v:.3f}" for v in ref))
    print(f"wrote {args.out}")


def check(args) -> None:
    """Simulated couples' windows (one household per draw, couples' region of
    RESULTS 43.9) under each noise, read as RESULTS 57.2 reads PSID."""
    from scripts import optionA as oa
    from scripts import psid_iid_error as pie
    from scripts.noise_calibration import cells, region

    model = FileNoise.load(args.noise_model)
    v2 = oa.spec_of(json.loads((args.v2_run / "results.json").read_text())["_config"]).obs_noise
    theta_all, card, train_f, held_f, _p, cfg = oa.load_shards(list(args.shards))
    box = oa.box_of(cfg)
    win = {**oa.WINDOW, "n_waves": len(model.income_ref), "start_low": args.start_low,
           "start_high": args.start_high}
    th, x, pid = oa.windows_raw(held_f, theta_all, seed=oa.HELD_SEED, window=win)
    pick = oa.one_per_draw(pid, seed=0)
    th, x, pid = th[pick], x[pick], pid[pick]
    keep = cells(th.numpy(), np.asarray(card)[pid.numpy()], box, region("couples", box)) >= 0
    x = x[torch.from_numpy(keep)]
    noisy = {"none": x, "v2": oa.add_obs_noise(x, v2, seed=oa.NOISE_SEED + oa.HELD_SEED),
             "file": add_psid_noise(x, model, seed=oa.NOISE_SEED + oa.HELD_SEED)}
    psid_res = json.loads(args.psid_iid.read_text())["psid"]
    res = {"n_windows": int(len(x)), "window": win, "features": {}}
    print(f"{len(x)} simulated couples' windows (held-out, couples' region); sigma on "
          f"log|value|, IQR / winsorised, rw+drift")
    for f in pie.FEATURES:
        j = FEATURES_TWOASSET_AGE.index(f)
        row = {}
        for name, xn in noisy.items():
            e = pie.estimate(xn[..., j].double().numpy())
            row[name] = {k: {"V1": e[k]["V"][0], "sigma_rw_drift": e[k]["rw+drift"]["sigma"]}
                         for k in pie.VARIANCES}
        p = psid_res[f]["all"]
        row["psid"] = {k: {"V1": p[k]["V"][0], "sigma_rw_drift": p[k]["rw+drift"]["sigma"]}
                       for k in pie.VARIANCES}
        res["features"][f] = row
        print(f"  {f:16s}" + "  ".join(
            f"{name} {r['iqr']['sigma_rw_drift']:.3f}/{r['winsorised']['sigma_rw_drift']:.3f}"
            for name, r in row.items()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    print(f"wrote {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate")
    c.add_argument("--psid", type=Path, default=PSID_X)
    c.add_argument("--shards", type=Path, nargs="+", default=list(SHARDS))
    c.add_argument("--n_waves", type=int, default=7)
    c.add_argument("--wave_years", type=int, default=2)
    c.add_argument("--start_low", type=int, default=24)
    c.add_argument("--start_high", type=int, default=51)
    c.add_argument("--out", type=Path, default=OUT)
    k = sub.add_parser("check")
    k.add_argument("--noise_model", type=Path, default=OUT)
    k.add_argument("--v2_run", type=Path, default=Path("outputs/waves/w07/s0"),
                   help="A member whose training noise is v2's (RESULTS 64's w07).")
    k.add_argument("--psid_iid", type=Path,
                   default=Path("outputs/psid_iid_error/psid_iid_error.json"))
    k.add_argument("--shards", type=Path, nargs="+", default=list(SHARDS))
    k.add_argument("--start_low", type=int, default=24)
    k.add_argument("--start_high", type=int, default=51)
    k.add_argument("--out", type=Path, default=OUT.parent / "noise_check.json")
    args = ap.parse_args()
    {"calibrate": calibrate, "check": check}[args.cmd](args)


if __name__ == "__main__":
    main()
