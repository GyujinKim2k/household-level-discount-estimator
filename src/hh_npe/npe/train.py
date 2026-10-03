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

import logging
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
):
    """Train SNPE-C on ``(theta, x)`` and return ``(posterior, density_estimator, inference)``.

    ``group_ids`` marks rows that share a simulated panel, so the train/
    validation split can keep them together -- see :func:`_use_grouped_split`.
    Required whenever one panel contributes several rows; harmless otherwise.
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
    density_estimator = inference.train(
        max_num_epochs=max_num_epochs,
        stop_after_epochs=stop_after_epochs,
        learning_rate=learning_rate,
        training_batch_size=batch_size,
        validation_fraction=validation_fraction,
    )
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
    if prop is not None and prop["name"] != "uniform" and not weighted_ok:
        raise SystemExit(
            f"{path} was trained on draws from the {prop['name']!r} proposal "
            f"(n_train={prop.get('n_train')}). Its raw draws are not the "
            f"posterior under the uniform prior. Sample it through "
            f"hh_npe.evaluation.weighted.sample_weighted, which applies the "
            f"importance weights, and load with weighted_ok=True.")
    return ck
