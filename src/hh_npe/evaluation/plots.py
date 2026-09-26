"""Posterior contour figures in the cosmology constraint-plot style.

Every estimation result gets one of these alongside its numbers. The reason is
specific to this project: twice now a model has been selected on a scalar
summary that turned out to be misleading -- contraction under LayerNorm, and
single-draw KS p-values in the wave comparison -- and both times the joint
shape of the posterior was where the problem was visible. A table of marginal
contraction and coverage cannot show a degeneracy direction, or that two arms
which look different on paper actually overlap.

Style: 68% contour solid and filled, 95% dashed and open, one hue per series,
serif labels, faint dotted grid, nothing else.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

#: Colour-blind-safe and matching the reference figure's blue/orange/green.
PALETTE = ["#3B75AF", "#EF8636", "#519E3E", "#C53A32", "#8D69B8", "#84584E"]

_LABEL = {"beta": r"$\beta$", "delta": r"$\delta$", "crra": r"$\rho$"}


def _hpd_levels(density: np.ndarray, probs=(0.68, 0.95)) -> list[float]:
    """Density levels enclosing ``probs`` of the total mass.

    Sort the grid cells by density and walk down until the cumulative mass
    crosses each target; the crossing density is the contour level. This is the
    2-D highest-posterior-density region, not an ellipse fit, so it stays
    correct for the banana-shaped posteriors this model produces.
    """
    flat = np.sort(density.ravel())[::-1]
    csum = np.cumsum(flat)
    csum /= csum[-1]
    return [float(flat[np.searchsorted(csum, p)]) for p in probs][::-1]


def _kde_grid(x: np.ndarray, y: np.ndarray, lo, hi, n: int = 140,
              reflect=(False, False)):
    """KDE on a grid, with optional reflection at an axis's upper bound.

    Reflection (Schuster 1985; Silverman 1986, sec. 2.10) is the classic
    boundary correction for a density on a bounded range. Near a hard wall a
    Gaussian kernel spills mass past it; cutting the grid at the wall then draws
    a contour along the bound that no data point produced. Here: households
    piled at delta 0.995-0.9996, narrower than the kernel, against delta = 1.

    The estimate is ``f(v) + f(2 * hi - v)``: the kernel is fitted on the REAL
    points only and evaluated at each grid point and at its mirror image.
    Fitting on points plus mirrored copies would instead double the axis's
    spread and so inflate the automatically chosen bandwidth.

    Known limitation: reflection forces zero slope at the bound, so where the
    true density rises steeply into the wall the very edge is slightly
    underestimated. The body and far tail are exactly as without it.

    An earlier version smoothed in log(1 - v) instead (the transformation
    method). It closed the contours below 1 but, with one fixed bandwidth in u,
    made the kernel very wide in v away from the wall and dragged the 95%
    region down to the bottom of the axis.
    """
    from scipy.stats import gaussian_kde

    X, Y = np.meshgrid(np.linspace(lo[0], hi[0], n), np.linspace(lo[1], hi[1], n))
    try:
        k = gaussian_kde(np.vstack([x, y]))
    except np.linalg.LinAlgError:
        # A posterior that has collapsed to a point has a singular covariance;
        # nothing to contour, and silently drawing nothing is better than
        # aborting a whole figure over one degenerate series.
        return X, Y, None
    xs, ys = X.ravel(), Y.ravel()
    mx, my = 2 * hi[0] - xs, 2 * hi[1] - ys
    Z = k(np.vstack([xs, ys]))
    if reflect[0]:
        Z += k(np.vstack([mx, ys]))
    if reflect[1]:
        Z += k(np.vstack([xs, my]))
    if reflect[0] and reflect[1]:
        Z += k(np.vstack([mx, my]))
    return X, Y, Z.reshape(X.shape)


#: Muted hues for external ranges, chosen to sit behind the series palette.
BAND_COLORS = ("0.45", "#8c6d31", "#7b4173")


def contour_corner(
    series: dict[str, np.ndarray],
    box,
    truth: dict[str, np.ndarray] | None = None,
    bands: dict[str, tuple] | None = None,
    path: str | Path | None = None,
    title: str | None = None,
    probs=(0.68, 0.95),
    axis_limits: tuple | None = None,
    reflect_axes: tuple[str, ...] = (),
    truth_markers: tuple[str, ...] | None = None,
    truth_colors: tuple[str, ...] | None = None,
):
    """Lower-triangle pairwise contour plot of posterior samples.

    Parameters
    ----------
    series
        ``{label: samples}``, each ``(n_samples, n_params)`` in ``box.names``
        order. One hue per entry, overlaid.
    box
        :class:`PriorBox` — supplies parameter names and the axis ranges, so
        every figure is drawn on the prior's scale and posteriors from
        different runs are visually comparable.
    truth
        Optional ``{label: theta_true}`` markers, for simulated data.
    bands
        Optional ``{label: (low, high)}`` shaded rectangles, each a length-``d``
        array pair. For plotting an external plausible range -- a literature
        interval, a published confidence region -- behind the posteriors.
        Drawn first and unfilled at the edges so they never obscure a contour.
    probs
        Enclosed-mass levels. Default 68% (filled) and 95% (dashed).
    axis_limits
        Optional ``(low, high)`` arrays that widen the *axes only*. The density
        is still evaluated on ``box``, so the region outside it stays empty
        rather than filling with kernel smoothing: a KDE of households piled at
        delta ~ 0.998 would otherwise draw contours above 1.00 that no
        household occupies. Use it to show that a bound is empty, not to
        pretend it is not there.
    truth_markers, truth_colors
        Optional per-entry marker symbols and colours for ``truth``; default a
        star in the series palette. Lets a true value and a point estimate be
        told apart on the same panel.
    reflect_axes
        Parameter names (e.g. ``("delta",)``) whose density is reflected at the
        box's upper bound, so a pile of samples against a hard wall does not
        draw a contour along the wall. See ``_kde_grid``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(box.names)
    lo, hi = np.asarray(box.low, float), np.asarray(box.high, float)
    d = len(names)
    pairs = [(i, j) for j in range(d) for i in range(j)]  # lower triangle

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
        "axes.linewidth": 0.9,
        "xtick.direction": "in",
        "ytick.direction": "in",
    })
    n = d - 1
    fig, axes = plt.subplots(n, n, figsize=(3.4 * n, 3.4 * n), squeeze=False)
    for a in axes.ravel():
        a.set_visible(False)

    for (i, j) in pairs:
        ax = axes[j - 1][i]
        ax.set_visible(True)
        if bands:
            for bc, (blab, (blo, bhi)) in zip(BAND_COLORS, bands.items()):
                ax.add_patch(plt.Rectangle(
                    (blo[i], blo[j]), bhi[i] - blo[i], bhi[j] - blo[j],
                    facecolor=bc, alpha=0.13, edgecolor=bc, lw=1.2,
                    ls=(0, (4, 3)), zorder=0))
        for c, (label, s) in zip(PALETTE, series.items()):
            s = np.asarray(s)
            X, Y, Z = _kde_grid(s[:, i], s[:, j], (lo[i], lo[j]), (hi[i], hi[j]),
                                reflect=(names[i] in reflect_axes,
                                         names[j] in reflect_axes))
            if Z is None:
                continue
            lv = _hpd_levels(Z, probs)
            # 95% dashed and open, 68% solid and tinted -- the outer contour is
            # the honest one, so it must not be hidden behind a fill.
            ax.contour(X, Y, Z, levels=[lv[0]], colors=[c],
                       linestyles="dashed", linewidths=1.6)
            ax.contourf(X, Y, Z, levels=[lv[1], Z.max()], colors=[c], alpha=0.18)
            ax.contour(X, Y, Z, levels=[lv[1]], colors=[c],
                       linestyles="solid", linewidths=2.0)
        if truth:
            for c, mk, t in zip(truth_colors or PALETTE,
                                truth_markers or ("*",) * len(truth),
                                truth.values()):
                ax.plot(t[i], t[j], marker=mk, ms=15 if mk == "*" else 11,
                        color=c, mec="0.15", mew=0.8, zorder=5, ls="none")
        alo, ahi = (lo, hi) if axis_limits is None else map(np.asarray, axis_limits)
        ax.set_xlim(alo[i], ahi[i])
        ax.set_ylim(alo[j], ahi[j])
        if axis_limits is not None:
            # Mark the prior bound wherever the axes run past it.
            for k, setter in ((i, ax.axvline), (j, ax.axhline)):
                if ahi[k] > hi[k]:
                    setter(hi[k], color="0.3", lw=0.9, ls=":", zorder=1)
        ax.grid(True, ls=":", lw=0.6, color="0.85")
        ax.set_axisbelow(True)
        if j == d - 1:
            ax.set_xlabel(_LABEL.get(names[i], names[i]), fontsize=17)
        else:
            ax.set_xticklabels([])
        if i == 0:
            ax.set_ylabel(_LABEL.get(names[j], names[j]), fontsize=17)
        else:
            ax.set_yticklabels([])

    handles = [plt.Line2D([], [], color=c, lw=2.0, label=k)
               for c, k in zip(PALETTE, series)]
    if bands:
        handles += [plt.Line2D([], [], color=bc, lw=6, alpha=0.35, label=k)
                    for bc, k in zip(BAND_COLORS, bands)]
    if truth:
        handles += [plt.Line2D([], [], marker=mk, ms=13 if mk == "*" else 10,
                               color=c, mec="0.15", mew=0.8, ls="none", label=k)
                    for c, mk, k in zip(truth_colors or PALETTE,
                                        truth_markers or ("*",) * len(truth),
                                        truth)]
    handles += [
        plt.Line2D([], [], color="0.35", lw=2.0, ls="solid", label="68%"),
        plt.Line2D([], [], color="0.35", lw=1.6, ls="dashed", label="95%"),
    ]
    # The upper-right of a corner plot is empty by construction; putting the
    # legend there keeps it off the contours and off the title.
    leg_ax = axes[0][n - 1] if n > 1 else axes[0][0]
    if n > 1:
        leg_ax.set_visible(True)
        leg_ax.axis("off")
        leg_ax.legend(handles=handles, loc="center", frameon=False,
                      fontsize=11, handlelength=2.4)
    else:
        leg_ax.legend(handles=handles, loc="best", frameon=False, fontsize=11)
    fig.tight_layout()
    if title:
        fig.suptitle(title, fontsize=13)
        # Reserve room per title line. A fixed 0.93 fits one line and lets a
        # two-line title sit on top of the first row's axis.
        fig.subplots_adjust(top=1.0 - 0.045 * (title.count("\n") + 1.5))
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
    return fig


def posterior_samples(post, x, n: int = 4000) -> np.ndarray:
    """Draw ``n`` posterior samples at one observation, as a numpy array."""
    import torch

    from hh_npe.evaluation.sbc import posterior_device

    dev = posterior_device(post)
    s = post.sample((n,), x=torch.as_tensor(x).to(dev),
                    show_progress_bars=False)
    return s.detach().cpu().numpy()
