"""Scoring an ensemble on data its members never saw must fail loudly.

This is the bug the guard exists for, and it is worth stating because the
failure mode is silence. `ensemble_eval` rebuilds the evaluation windows from
`--shards` rather than reusing the members', so every argument that shaped
training has to be repeated. Its defaults point at the Phase 3 dataset, which is
correct for Phase 3 and silently wrong after it.

A Phase 4 ensemble was in fact scored against Phase 3 shards with the wrong
window and the wrong education handling. Nothing aborted -- a feature-count
mismatch would have, but the count happened to match -- and the resulting
numbers were plausible enough to be reported before the error was noticed.
"""

import json

import pytest


_N = iter(range(1000))


def _run(tmp_path, cfg, **args):
    """One member directory carrying `cfg`, checked against `args`."""
    import types

    from scripts.ensemble_eval import _check_provenance

    # Fresh directory per call: a test may invoke this twice.
    d = tmp_path / f"w7_s{next(_N)}"
    d.mkdir()
    (d / "results.json").write_text(json.dumps({"7": {}, "_config": cfg}))
    ns = types.SimpleNamespace(start_low=25, shards="phase3", sbc_cache="old",
                               educ_group=None, condition_educ=False,
                               log_features=False)
    for k, v in args.items():
        setattr(ns, k, v)
    _check_provenance(ns, [d], 7)


BASE = {"start_low": 25, "shards": "phase3", "sbc_cache": "old",
        "educ_group": None, "condition_educ": False, "log_features": False}


def test_matching_provenance_passes(tmp_path):
    _run(tmp_path, BASE)


@pytest.mark.parametrize("field,trained,scoring", [
    ("start_low", 24, 25),                 # the off-by-one window of RESULTS 10.8
    ("shards", "phase4", "phase3"),        # the actual Phase 4 bug
    ("sbc_cache", "phase4_sbc", "old"),
    ("educ_group", "comphs", None),        # per-group model, pooled evaluation
    ("condition_educ", True, False),       # conditioned model, unconditioned x
    ("log_features", True, False),         # log-trained model, level-scaled x
])
def test_mismatched_provenance_aborts(tmp_path, field, trained, scoring):
    cfg = {**BASE, field: trained}
    with pytest.raises(SystemExit, match="different data"):
        _run(tmp_path, cfg, **{field: scoring})


def test_a_member_without_a_config_is_skipped_not_guessed(tmp_path):
    """Older runs predate these fields. Absent must mean 'cannot check', never
    'matches' -- and never block scoring an otherwise valid ensemble."""
    _run(tmp_path, {})


def test_fields_are_checked_independently(tmp_path):
    """A run recording only some fields is checked on those, not skipped whole."""
    with pytest.raises(SystemExit):
        _run(tmp_path, {"start_low": 24}, start_low=25)
    _run(tmp_path, {"start_low": 25}, start_low=25)
