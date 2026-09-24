"""Replay-row numeric-answer reward for Stage-4 training (pure, import-safe).

PROVENANCE: :func:`official_stage3_na_reward` is a bug-for-bug port of the
vendored official Stage-3 numeric-answer reward
(``vendor/SpatialLadder/VLM-R1/src/open-r1-multimodal/src/open_r1/grpo_spld_stage3.py``,
``accuracy_reward`` closure ``na_reward`` -> ``mean_relative_accuracy``),
transitively via the evaluation-side port
``scripts/e3_rollout_eval.py:official_stage3_na_reward`` (read-only,
hash-bound).  It is re-homed here because ``src`` modules must not import
from ``scripts`` and the training entry point (``scripts/train_stage4.py``)
needs the same official semantics for the ``row_kind == "replay"`` rows of
the merged AReG manifest (amendment
``asymmetric_directional_credit_revision_20260826``: replay contributes
GRPO/KL gradients with the official numeric reward, aux weight zero).
Logical equivalence with the evaluation-side port is pinned by
``tests/test_training_rewards.py`` on a fixed fixture grid.
"""

from __future__ import annotations

import numpy as np

from regcfpo.audit import clean_stage3_mca_text


def _to_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def official_stage3_na_reward(completion: str, answer: str) -> float:
    """Bug-for-bug port of the official stage3 ``na_reward``.

    Mirrors ``mean_relative_accuracy(start=.5, end=.95, interval=.05)`` from
    grpo_spld_stage3.py, including the ``num_pts = ... + 2`` quirk: in binary
    floating point the expression evaluates to 10.999..., so ``int`` yields
    10 linspace points (thresholds 0.5, 0.45, ..., 0.05).  The division uses
    the raw target.  The official corpus has no zero targets and the official
    code would raise on one; here a zero target is scored as exact match
    (documented deviation, identical to the evaluation-side port).
    """

    pred = _to_float(clean_stage3_mca_text(completion))
    target = _to_float(clean_stage3_mca_text(answer))
    if pred is None or target is None:
        return 0.0
    if target == 0.0:
        return 1.0 if pred == 0.0 else 0.0
    num_pts = (0.95 - 0.5) / 0.05 + 2
    conf_intervs = np.linspace(0.5, 0.95, int(num_pts))
    rel_err = abs(pred - target) / target
    return float((rel_err <= 1 - conf_intervs).mean())


__all__ = ["official_stage3_na_reward"]
