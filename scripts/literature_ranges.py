"""Published (beta, delta, rho) ranges for US households, to plot behind our own.

**Why not just compare to Laibson et al.** Their single MSM estimate is a
*representative-agent* fit to population moments. Our posteriors are a
distribution over households. Those are different objects, and §6's deviation 3
already flags that "same information set" is never literally true. A band of
what the literature finds plausible is a fairer benchmark than one number.

Every value below is sourced. Where a figure is a meta-analytic pooled estimate
its confidence interval is given, because the spread across studies is itself
the point.

Sources
-------
beta, present bias — two meta-analyses of the Convex Time Budget literature:

  Imai, Rutter & Camerer, "Meta-Analysis of Present-Bias Estimation using
  Convex Time Budgets", Economic Journal 131(636), 2021. 220 estimates from 28
  articles. Monetary rewards beta = 0.82, 95% CI [0.74, 0.90]; non-monetary
  beta = 0.66, 95% CI [0.51, 0.85].
  https://academic.oup.com/ej/article/131/636/1788/5912830

  Cheung, Tymula & Wang, "A Meta-Analysis of Quasi-Hyperbolic Discounting",
  Management Science, 2023 (IZA DP 14625). Monetary beta = 0.94, 95% CI
  [0.90, 0.97]; non-monetary beta = 0.68, 95% CI [0.57, 0.82].
  https://docs.iza.org/dp14625.pdf

delta, long-run discount factor with heterogeneity across households:

  Carroll, Slacalek, Tokuoka & White, "The distribution of wealth and the
  marginal propensity to consume", Quantitative Economics 8(3), 2017. Estimates
  a *distribution* of annual discount factors -- the closest published object to
  what this project estimates -- with agents differing from the mean by about
  0.02. Their European companion reports beta ~ 0.99 against net wealth and
  ~ 0.97 against liquid assets.
  https://onlinelibrary.wiley.com/doi/abs/10.3982/QE694

rho, relative risk aversion:

  Elminejad, Havranek & Irsova, "Relative Risk Aversion: A Meta-Analysis",
  Journal of Economic Surveys, 2025. 1021 estimates from 92 studies using the
  consumption Euler equation. After correcting for publication bias the mean is
  about 1 in economics contexts and 2-7 in finance contexts, and calibrations
  are systematically larger than estimates.
  https://onlinelibrary.wiley.com/doi/full/10.1111/joes.12689

Verifiable in this repository, transcribed from their table 3:
``laibson_calibration.BENCHMARK_PREFS`` and ``EXPONENTIAL_PREFS``.
"""

from __future__ import annotations

import numpy as np

#: Their naive quasi-hyperbolic estimate and its standard errors.
LAIBSON = np.array([0.5305, 0.9891, 1.9355])
LAIBSON_SE = np.array([0.1140, 0.0051, 0.4350])

#: Their own beta == 1 (exponential) restriction, same paper.
LAIBSON_EXPONENTIAL = np.array([1.0, 0.9600, 1.4663])

#: Spanning the two CTB meta-analyses: Imai et al.'s non-monetary lower bound
#: (0.66) to Cheung et al.'s monetary estimate (0.94). Experimental present bias
#: is consistently *milder* than the structural lifecycle estimate of 0.53.
BETA_META = (0.66, 0.94)

#: Carroll et al.'s heterogeneous annual discount factors: centred near
#: 0.97-0.99 with agents spread about 0.02 either side.
DELTA_META = (0.95, 1.00)

#: Elminejad et al.: ~1 in economics, 2-7 in finance. The band spans both; our
#: own estimates land in the finance part, which is worth saying out loud.
CRRA_META = (1.0, 7.0)

META = (np.array([BETA_META[0], DELTA_META[0], CRRA_META[0]]),
        np.array([BETA_META[1], DELTA_META[1], CRRA_META[1]]))


def laibson_ci(n_se: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    """Their estimate +- ``n_se`` SEs. delta is clipped at 1.0, since the upper
    arm (0.9891 + 1.96 x 0.0051) runs past what is a discount factor."""
    lo, hi = LAIBSON - n_se * LAIBSON_SE, LAIBSON + n_se * LAIBSON_SE
    return lo, np.minimum(hi, [np.inf, 1.0, np.inf])


if __name__ == "__main__":
    lo, hi = laibson_ci()
    print(f"Laibson et al. MSM        {LAIBSON}")
    print(f"  95% CI                  {lo.round(3)} .. {hi.round(3)}")
    print(f"Laibson et al. beta==1    {LAIBSON_EXPONENTIAL}")
    print()
    print("meta-analytic bands (sourced, see module docstring)")
    print(f"  beta  {BETA_META}   Imai 0.82 [0.74,0.90], Cheung 0.94 [0.90,0.97]")
    print(f"  delta {DELTA_META}  Carroll et al. heterogeneous, spread ~0.02")
    print(f"  crra  {CRRA_META}   Elminejad ~1 economics, 2-7 finance")
