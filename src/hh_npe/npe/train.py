"""Amortized SNPE-C training for the household-level NPE.

Wraps sbi's ``SNPE_C`` with our :class:`TrajectoryTransformer` embedder and the
Sobol-uniform prior. Single-round training (no proposal adaptation) so the
trained posterior is amortized — usable on any new ``x`` without re-training,
which is what the headline contribution requires.

We pass ``z_score_x='none'`` because our embedder handles input normalization
via its leading ``LayerNorm``, avoiding sbi's flatten-first-then-z-score
assumption for 3D ``x``.
"""

from __future__ import annotations

import contextlib
import csv
import logging
import time
from pathlib import Path

import torch
from sbi.inference import SNPE_C
from sbi.neural_nets import posterior_nn

from hh_npe.npe.embedder import TrajectoryTransformer
from hh_npe.npe.prior import PriorBox, make_sbi_prior

log = logging.getLogger(__name__)


def _use_grouped_split(inference, group_ids: torch.Tensor, seed: int = 0) -> None:
    """Make sbi split train/validation by group instead of by row.

    sbi assigns rows to train and validation with a random index permutation
    (``sbi/inference/trainers/base.py``). When several rows come from the same
    simulated panel -- which is exactly what random-age augmentation produces --
    that puts windows of one household on both sides of the split. The
    validation loss is then measured on panels the model trained on, early
    stopping fires late, and every reported number is optimistic.

    ``resume_training`` inside ``get_dataloaders`` gates *only* the permutation;
    the optimizer and epoch counter are initialised separately in the trainer's
    ``train``. So pre-computing the indices and delegating with
    ``resume_training=True`` reuses sbi's own machinery rather than forking it.
    """
    original = inference.get_dataloaders

    def grouped(starting_round=0, training_batch_size=200,
                validation_fraction=0.1, resume_training=False,
                dataloader_kwargs=None):
        if not resume_training:
            groups = group_ids.to("cpu")
            # The row count is not an argument -- sbi reads it back out of the
            # stored simulations, so take it from the same place.
            n_rows = len(inference.get_simulations(starting_round)[0])
            if len(groups) != n_rows:
                raise ValueError(
                    f"group_ids has {len(groups)} entries but the dataset has "
                    f"{n_rows} rows"
                )
            uniq = torch.unique(groups)
            g = torch.Generator().manual_seed(seed)
            shuffled = uniq[torch.randperm(len(uniq), generator=g)]
            n_val = max(1, int(validation_fraction * len(uniq)))
            is_val = torch.isin(groups, shuffled[:n_val])
            inference.val_indices = torch.where(is_val)[0]
            inference.train_indices = torch.where(~is_val)[0]
            log.info(
                f"Grouped split: {len(uniq) - n_val} panels "
                f"({len(inference.train_indices)} rows) train, {n_val} panels "
                f"({len(inference.val_indices)} rows) validation"
            )
        # sbi sizes each loader's batch from the split *it* would have made, so
        # a grouped split that lands a few rows short of that leaves the
        # validation loader with drop_last=True and zero batches -- which
        # surfaces as a ZeroDivisionError when the epoch averages its loss.
        # Keeping the partial batch is the standard fix and costs nothing.
        kwargs = dict(dataloader_kwargs or {})
        kwargs.setdefault("drop_last", False)
        return original(starting_round, training_batch_size,
                        validation_fraction, True, kwargs)

    inference.get_dataloaders = grouped


#: Columns of the per-epoch training CSV (RESULTS 67).
METRIC_COLUMNS = ("epoch", "passes", "train_loss", "val_loss", "best_val_loss", "is_best",
                  "epochs_since_best", "epoch_sec", "elapsed_min", "grad_norm_mean",
                  "grad_norm_max", "lr", "n_train_rows", "n_val_rows", "gpu_mem_gb")


class EpochCSV:
    """One row per epoch to each of ``paths``, written and flushed as the epoch
    ends, so a run can be watched while it trains.

    sbi's ``train`` loop has no callback; it calls ``_converged(epoch, ...)``
    once before the first epoch (epoch 0) and once after each, then once more
    at the end with the same epoch. :meth:`attach` wraps that method on one
    instance and logs each epoch the first time it is seen, after sbi has
    updated its best loss. Gradient norms are the values sbi's own
    ``clip_grad_norm_`` returns, recorded without changing the call.
    Nothing here draws random numbers or touches the optimisation."""

    def __init__(self, paths, passes_per_epoch: int = 1):
        self.paths = [Path(p) for p in paths]
        self.passes = passes_per_epoch
        self.files = []
        self.norms: list[torch.Tensor] = []
        self.logged = 0
        self.t0 = time.time()

    def __enter__(self):
        for path in self.paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            f = path.open("w", newline="")
            csv.writer(f).writerow(METRIC_COLUMNS)
            f.flush()
            self.files.append(f)
        return self

    def __exit__(self, *exc):
        for f in self.files:
            f.close()
        return False

    @contextlib.contextmanager
    def grad_norms(self):
        """Record every gradient norm sbi's NPE loop clips, for the epoch."""
        from sbi.inference.trainers.npe import npe_base

        orig = npe_base.clip_grad_norm_

        def recording(*a, **k):
            n = orig(*a, **k)
            self.norms.append(n.detach())
            return n

        npe_base.clip_grad_norm_ = recording
        try:
            yield
        finally:
            npe_base.clip_grad_norm_ = orig

    def attach(self, inference) -> None:
        orig = inference._converged

        def converged(epoch, stop_after_epochs):
            out = orig(epoch, stop_after_epochs)
            if epoch > self.logged:
                self.logged = epoch
                self.write(inference, epoch)
            return out

        inference._converged = converged

    def write(self, inference, epoch: int) -> None:
        s = inference._summary
        g = torch.stack(self.norms).float() if self.norms else None
        self.norms = []
        dev = torch.device(inference._device)
        mem = (torch.cuda.max_memory_allocated(dev) / 1e9
               if dev.type == "cuda" and torch.cuda.is_available() else float("nan"))
        since = inference._epochs_since_last_improvement
        row = (epoch, epoch * self.passes, s["training_loss"][-1], s["validation_loss"][-1],
               inference._best_val_loss, int(since == 0), since,
               s["epoch_durations_sec"][-1], (time.time() - self.t0) / 60,
               float(g.mean()) if g is not None else float("nan"),
               float(g.max()) if g is not None else float("nan"),
               inference.optimizer.param_groups[0]["lr"], len(inference.train_indices),
               len(inference.val_indices), mem)
        for f in self.files:
            csv.writer(f).writerow([f"{v:.6g}" if isinstance(v, float) else v for v in row])
            f.flush()


class FreshNoise:
    """A new noise draw on the training rows before every epoch (RESULTS 68).

    ``redraw(e)`` returns the whole ``x`` (every row, in the order handed to
    sbi) with noise draw ``e``; the rows passed to sbi are draw 0. sbi builds
    its loaders over one concatenated copy of ``x``: :meth:`attach` keeps a
    handle on it (wrapping ``get_dataloaders``, on top of any grouped split),
    and wraps ``_converged``, which sbi calls before each epoch's training.
    When training goes on after epoch ``e`` (1 <= ``e`` <= the cap), the training rows
    are overwritten in place with draw ``e``, so epoch ``e + 1`` trains on
    it; the validation rows keep draw 0. The redraw falls before sbi starts
    its epoch timer and draws nothing from torch's global generator.

    sbi keeps the last weights when ``max_num_epochs`` is reached;
    :meth:`restore_best` puts the best ones back (``restored_best_at_cap``)."""

    def __init__(self, redraw, max_num_epochs: int):
        self.redraw = redraw
        self.max_epochs = max_num_epochs
        self.dataset_x: torch.Tensor | None = None
        self.drawn = 0
        self.restored = False

    def attach(self, inference) -> None:
        orig_loaders = inference.get_dataloaders

        def loaders(*a, **k):
            train_loader, val_loader = orig_loaders(*a, **k)
            self.dataset_x = train_loader.dataset.tensors[1]
            return train_loader, val_loader

        inference.get_dataloaders = loaders
        orig_converged = inference._converged

        def converged(epoch, stop_after_epochs):
            out = orig_converged(epoch, stop_after_epochs)
            # sbi trains epoch e + 1 after this call only while e <= the cap;
            # its closing call (one past the cap) needs no draw
            if not out and self.drawn < epoch <= self.max_epochs:
                self.drawn = epoch
                new = self.redraw(epoch)
                if new.shape != self.dataset_x.shape:
                    raise ValueError(f"draw {epoch} has shape {tuple(new.shape)}, the "
                                     f"training data {tuple(self.dataset_x.shape)}")
                idx = inference.train_indices
                self.dataset_x[idx] = new[idx].to(self.dataset_x.device, self.dataset_x.dtype)
            return out

        inference._converged = converged

    def restore_best(self, inference, density_estimator) -> None:
        """At the epoch cap sbi keeps the last weights; take the best. (When
        patience stops training, sbi has already loaded them, and this finds
        the weights equal.)"""
        best = inference._best_model_state_dict
        now = density_estimator.state_dict()
        if any(not torch.equal(now[k], v.to(now[k].device)) for k, v in best.items()):
            density_estimator.load_state_dict(best)
            inference._neural_net.load_state_dict(best)
            self.restored = True


def train_npe(
    theta: torch.Tensor,
    x: torch.Tensor,
    *,
    embedder: TrajectoryTransformer | None = None,
    box: PriorBox | None = None,
    flow: str = "nsf",
    hidden_features: int = 50,
    num_transforms: int = 5,
    max_num_epochs: int = 200,
    stop_after_epochs: int = 20,
    learning_rate: float = 5e-4,
    batch_size: int = 256,
    validation_fraction: float = 0.1,
    device: str = "cpu",
    show_progress: bool = False,
    group_ids: torch.Tensor | None = None,
    metrics_csv=None,
    passes_per_epoch: int = 1,
    redraw_x=None,
):
    """Train SNPE-C on ``(theta, x)`` and return ``(posterior, density_estimator, inference)``.

    ``group_ids`` marks rows that share a simulated panel, so the train/
    validation split can keep them together -- see :func:`_use_grouped_split`.
    Required whenever one panel contributes several rows; harmless otherwise.

    ``metrics_csv``: a path or paths, each given one row per epoch
    (:class:`EpochCSV`, :data:`METRIC_COLUMNS`); ``passes_per_epoch`` is the
    number of passes over the clean windows an epoch makes (K noisy copies
    make K, RESULTS 67). Logging changes nothing in training.

    ``redraw_x`` (RESULTS 68): ``e -> x`` with noise draw ``e`` on every row;
    the training rows take draw ``e`` before epoch ``e + 1`` (:class:`FreshNoise`),
    and at the epoch cap the best weights are restored. The returned
    ``inference`` then carries ``fresh_noise`` (the :class:`FreshNoise`) and
    ``restored_best_at_cap``.
    """
    if embedder is None:
        embedder = TrajectoryTransformer()
    if box is None:
        box = PriorBox()

    prior = make_sbi_prior(box, device=device)
    de_builder = posterior_nn(
        model=flow,
        embedding_net=embedder,
        hidden_features=hidden_features,
        num_transforms=num_transforms,
        z_score_x="none",
        z_score_theta="independent",
    )

    inference = SNPE_C(
        prior=prior,
        density_estimator=de_builder,
        device=device,
        show_progress_bars=show_progress,
    )
    inference.append_simulations(theta.float(), x.float())
    if group_ids is not None:
        _use_grouped_split(inference, group_ids)

    n_groups = len(torch.unique(group_ids)) if group_ids is not None else theta.shape[0]
    log.info(
        f"Training SNPE-C: n_samples={theta.shape[0]} from {n_groups} panels, "
        f"x_shape={tuple(x.shape[1:])}, device={device}, "
        f"max_epochs={max_num_epochs}, batch_size={batch_size}"
    )
    paths = ([] if metrics_csv is None else
             [metrics_csv] if isinstance(metrics_csv, (str, Path)) else list(metrics_csv))
    with contextlib.ExitStack() as stack:
        if paths:
            logger = stack.enter_context(EpochCSV(paths, passes_per_epoch))
            stack.enter_context(logger.grad_norms())
            logger.attach(inference)
        fresh = None
        if redraw_x is not None:
            fresh = FreshNoise(redraw_x, max_num_epochs)
            fresh.attach(inference)
        density_estimator = inference.train(
            max_num_epochs=max_num_epochs,
            stop_after_epochs=stop_after_epochs,
            learning_rate=learning_rate,
            training_batch_size=batch_size,
            validation_fraction=validation_fraction,
        )
    if fresh is not None:
        fresh.restore_best(inference, density_estimator)
        inference.fresh_noise, inference.restored_best_at_cap = fresh, fresh.restored
    posterior = inference.build_posterior(density_estimator)
    return posterior, density_estimator, inference


def save_posterior(
    posterior,
    embedder: TrajectoryTransformer,
    box: PriorBox,
    path: str | Path,
    proposal: dict | None = None,
) -> None:
    """Persist the trained posterior + embedder state + prior box.

    ``proposal`` (``{"name", "n_train"}``) records a non-uniform training
    proposal, so the checkpoint itself says its draws need importance weights
    (:func:`hh_npe.npe.prior.proposal_log_weight`); see :func:`load_posterior`.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ck = {
        "posterior": posterior,
        "embedder_state_dict": embedder.state_dict(),
        "box": box,
    }
    if proposal is not None:
        ck["proposal"] = dict(proposal)
    torch.save(ck, p)


def load_posterior(path: str | Path, map_location=None,
                   weighted_ok: bool = False) -> dict:
    """Load a checkpoint saved by :func:`save_posterior` (returns the raw dict).

    ``map_location="cpu"`` loads a GPU-trained posterior onto the CPU, for use
    while the GPU is occupied (e.g. by a long generation run).

    A network trained on draws from a non-uniform proposal (RESULTS.md 30.2)
    learns the posterior under that proposal; its raw draws are NOT the
    posterior under the uniform prior, and every sampler that does not apply
    the weights reports biased numbers without failing. So loading one is
    refused unless the caller says it weights (``weighted_ok=True``), as
    ``hh_npe.evaluation.weighted.sample_weighted`` does.
    """
    ck = torch.load(Path(path), weights_only=False, map_location=map_location)
    prop = ck.get("proposal")
    # Several runs (RESULTS 43.5) record {"parts": [...]}, one per run; their
    # R_gamma ranges differ, so even all-uniform parts need weights.
    weighted = prop is not None and ("parts" in prop or prop["name"] != "uniform")
    if weighted and not weighted_ok:
        raise SystemExit(
            f"{path} was trained on draws from the {prop.get('name', prop)!r} proposal "
            f"(n_train={prop.get('n_train')}). Its raw draws are not the "
            f"posterior under the uniform prior. Sample it through "
            f"hh_npe.evaluation.weighted.sample_weighted, which applies the "
            f"importance weights, and load with weighted_ok=True.")
    return ck
