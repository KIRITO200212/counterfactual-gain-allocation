"""Global-batch gate computation glue for the Stage-4 trainer.

The saturation-aware gate (``auxiliary.py``) has a strict input contract:
accuracy-only group means, computed after the global reward gather, with
weight normalization/clipping applied on the global batch.  This module is
the single, unit-testable implementation of that global semantics; the
trainer subclass only orchestrates collectives around it.

MIX credits use the batch-independent closed form of plan7 section 1.1 R1
(``closed_form_success_credit``): a group's credit depends only on its own
``k/G``, never on the other groups in the batch, so the
one-prompt-group-per-step configuration no longer degenerates the
continuous credit to exactly 1.

Execution rule (plan5 section 5.3, updated to the shipped trainer): ranks
never skip auxiliary forwards based on LOCAL weights, and the current
trainer performs NO local slicing at all — every rank scores EVERY global
group redundantly (see ``trainer.py``), so the executed weights are the
full global weight vector on every rank.  The only skip decision is the
GLOBAL active count, which is identical on every rank because it derives
from the gathered global batch.  ``local_group_slice`` below is retained
only for a future sharded variant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor

from regcfpo.training.auxiliary import (
    GateConfig,
    LEGACY_GATE_CREDIT_MODE,
    MONOTONIC_GATE_CREDIT_MODE,
    closed_form_success_credit,
    monotonic_success_credit,
    normalize_gate_weights,
)

#: The vendor trainer's group-relative advantage epsilon (grpo_trainer.py:763).
ADVANTAGE_EPSILON = 1e-4

#: Fields that identify a prompt group within a batch of G-repeated rollouts.
DEFAULT_GROUP_KEY_FIELDS = ("sample_id", "question_id")


@dataclass(frozen=True)
class GateStrataMetrics:
    """Global-batch coverage of the three saturation strata."""

    sat_fraction: float
    mix_fraction: float
    all_wrong_fraction: float


@dataclass(frozen=True)
class GlobalGatePayload:
    """Gate quantities computed once per generation on the GLOBAL batch.

    ``weights`` are the normalized, clipped, stop-gradient credits
    (``w_tilde``); they are detached by construction in
    ``normalize_gate_weights`` and must never carry gradients.
    ``group_max_positive_advantage`` is the vendored-trainer advantage
    mirror (ddof=1); since plan7 section 1.1 R1 it is DIAGNOSTIC ONLY —
    the credit derives from the closed form in ``k/G`` and never consumes
    the empirical advantage.
    """

    group_rewards: Tensor
    group_max_positive_advantage: Tensor
    credits: Tensor
    weights: Tensor
    active_count: int
    global_active: bool
    strata: GateStrataMetrics
    w_tilde_mean: float
    clip_hit_fraction: float
    num_generations: int
    credit_mode: str
    normalization_applied: bool

    @property
    def num_groups(self) -> int:
        return int(self.group_rewards.shape[0])


@dataclass(frozen=True)
class AuxiliaryExecutionPlan:
    """Rank-consistent auxiliary execution decision for one compute_loss.

    ``run_forwards`` is identical on every rank (derived from the global
    batch).  In the current trainer the executed set is the full global
    group list on every rank (redundant scoring, no slicing), so
    ``forward_count`` and ``active_local_groups`` are also identical on
    every rank; groups with weight zero still run their forwards, masked by
    the zero weight, so the forward count depends only on the group count.
    """

    run_forwards: bool
    forward_count: int
    active_local_groups: int
    total_local_groups: int


def group_accuracy_rewards(
    accuracy_rewards: Tensor, num_generations: int
) -> tuple[Tensor, Tensor]:
    """Fold a flat accuracy-reward vector into per-group summaries.

    ``accuracy_rewards`` must be the accuracy-only reward component (not the
    vendored multi-component sum) of shape ``[n_groups * num_generations]`` in
    group-major order (consecutive ``num_generations`` entries per prompt).
    Returns ``(group_means, group_max_positive_advantage)`` where the latter
    is ``max_j [A_ij]^+`` with ``A`` the group-relative advantage computed
    from the accuracy-only rewards using the vendor's epsilon.
    """

    if not isinstance(num_generations, int) or isinstance(num_generations, bool):
        raise ValueError("num_generations must be an integer")
    if num_generations < 2:
        raise ValueError("num_generations must be >= 2 for group-relative gating")
    if not isinstance(accuracy_rewards, Tensor) or accuracy_rewards.ndim != 1:
        raise ValueError("accuracy_rewards must be a 1-D torch.Tensor")
    if not bool(torch.isfinite(accuracy_rewards).all()):
        raise FloatingPointError("accuracy_rewards contains NaN or Inf; refusing to continue")
    total = accuracy_rewards.shape[0]
    if total == 0 or total % num_generations != 0:
        raise ValueError(
            f"accuracy_rewards length {total} is not a positive multiple of "
            f"num_generations={num_generations}"
        )
    rewards = accuracy_rewards.float()
    if bool(((rewards < 0.0) | (rewards > 1.0)).any()):
        raise ValueError("accuracy rewards must lie in [0, 1]")
    grouped = rewards.view(-1, num_generations)
    means = grouped.mean(dim=1)
    stds = grouped.std(dim=1)
    advantages = (grouped - means.unsqueeze(1)) / (stds.unsqueeze(1) + ADVANTAGE_EPSILON)
    max_positive = advantages.clamp_min(0.0).max(dim=1).values
    return means, max_positive


def compute_global_gate_payload(
    accuracy_rewards_global: Tensor,
    num_generations: int,
    config: GateConfig | None = None,
    *,
    group_aux_mask: Tensor | None = None,
) -> GlobalGatePayload:
    """Compute the saturation-aware gate on the GLOBAL gathered batch.

    ``c_loc``/``c_op`` are preregistered multipliers pre-applied by the
    caller (1.0 for the GT-box Stage-4 configuration), so the raw weights
    are exactly the success credits.  Credits come from the
    batch-independent closed form (``closed_form_success_credit``, plan7
    section 1.1 R1); normalization and clipping
    (``normalize_gate_weights``, plan.md section 1.1) still happen here, on
    the global batch.

    ``group_aux_mask`` is the per-group auxiliary-eligibility multiplier
    (1.0 = eligible, 0.0 = aux-disabled), applied to the credits BEFORE
    normalization.  The AReG-CFPO 348-step diagnostic uses it to force the
    merged manifest's ``row_kind == "replay"`` groups to aux weight exactly
    zero (amendment ``asymmetric_directional_credit_revision_20260826``):
    replay rows contribute GRPO/KL gradients only.  Because the masked
    credits feed ``active_count``/``global_active``, a batch whose groups are
    ALL replay makes every rank skip the auxiliary forwards together (the
    only collective-consistent skip; see ``trainer.py``).  The strata
    fractions stay unmasked (descriptive accuracy coverage).  ``None``
    preserves the historical all-eligible behavior exactly.
    """

    cfg = config or GateConfig()
    means, max_positive = group_accuracy_rewards(accuracy_rewards_global, num_generations)
    if cfg.credit_mode == LEGACY_GATE_CREDIT_MODE:
        credits = closed_form_success_credit(means, num_generations, epsilon=cfg.epsilon)
    elif cfg.credit_mode == MONOTONIC_GATE_CREDIT_MODE:
        credits = monotonic_success_credit(means, num_generations, epsilon=cfg.epsilon)
    else:  # pragma: no cover - GateConfig validates this at construction.
        raise RuntimeError(f"unhandled gate credit mode {cfg.credit_mode!r}")
    if group_aux_mask is not None:
        if not isinstance(group_aux_mask, Tensor) or group_aux_mask.ndim != 1:
            raise ValueError("group_aux_mask must be a 1-D torch.Tensor")
        if group_aux_mask.shape != means.shape:
            raise ValueError("group_aux_mask must have one entry per group")
        if not bool(torch.isfinite(group_aux_mask).all()):
            raise FloatingPointError("group_aux_mask contains NaN or Inf; refusing to continue")
        if bool((group_aux_mask < 0.0).any()):
            raise ValueError("group_aux_mask must be non-negative")
        credits = credits * group_aux_mask.float()
    normalization_applied = cfg.credit_mode == LEGACY_GATE_CREDIT_MODE
    weights = (
        normalize_gate_weights(credits, clip=cfg.weight_clip, epsilon=cfg.epsilon)
        if normalization_applied
        else credits.detach()
    )

    is_sat = means >= 1.0 - cfg.epsilon
    is_all_wrong = means <= cfg.epsilon
    is_mix = ~is_sat & ~is_all_wrong
    num_groups = means.shape[0]
    strata = GateStrataMetrics(
        sat_fraction=is_sat.float().mean().item(),
        mix_fraction=is_mix.float().mean().item(),
        all_wrong_fraction=is_all_wrong.float().mean().item(),
    )
    active_count = int((credits > 0.0).sum().item())
    clip_hits = (
        (credits > 0.0) & (weights >= cfg.weight_clip)
        if normalization_applied
        else torch.zeros_like(credits, dtype=torch.bool)
    )
    return GlobalGatePayload(
        group_rewards=means,
        group_max_positive_advantage=max_positive,
        credits=credits,
        weights=weights,
        active_count=active_count,
        global_active=active_count > 0,
        strata=strata,
        w_tilde_mean=weights.mean().item() if num_groups else 0.0,
        clip_hit_fraction=clip_hits.float().mean().item() if num_groups else 0.0,
        num_generations=num_generations,
        credit_mode=cfg.credit_mode,
        normalization_applied=normalization_applied,
    )


def local_group_slice(
    num_groups_global: int, process_index: int, num_processes: int
) -> slice:
    """Map a rank to its contiguous slice of global groups.

    RESERVED FOR A FUTURE SHARDED VARIANT — no production caller exists:
    the current trainer scores every global group redundantly on every rank
    (module docstring), so no slicing is performed.  Retained, with tests,
    because a sharded auxiliary variant (which would need a differentiable
    cross-rank score gather) needs exactly this mapping.  Mirrors the
    vendored trainer's completion-level ``process_slice``
    (grpo_trainer.py:766-769): with equal per-rank batch sizes, rank ``r``
    owns global groups ``[r * n_local, (r + 1) * n_local)``.
    """

    if not (0 <= process_index < num_processes):
        raise ValueError(
            f"process_index {process_index} out of range for {num_processes} processes"
        )
    if num_groups_global % num_processes != 0:
        raise ValueError(
            f"global group count {num_groups_global} is not divisible by "
            f"num_processes={num_processes}; the gate cannot be sliced consistently"
        )
    per_rank = num_groups_global // num_processes
    return slice(process_index * per_rank, (process_index + 1) * per_rank)


def plan_auxiliary_execution(
    *,
    global_active: bool,
    executed_weights: Tensor,
    forwards_per_group: int,
) -> AuxiliaryExecutionPlan:
    """Decide auxiliary execution for this rank, consistently across ranks.

    The skip decision uses ONLY ``global_active`` (identical on all ranks
    because it derives from the gathered global batch).
    ``executed_weights`` are the gate weights of the groups this rank will
    execute; groups with weight zero still get their forwards (masked by the
    zero weight), so the forward count depends only on the group count, which
    is rank-invariant by construction in the caller.
    """

    if not isinstance(executed_weights, Tensor) or executed_weights.ndim != 1:
        raise ValueError("executed_weights must be a 1-D torch.Tensor")
    if not bool(torch.isfinite(executed_weights).all()):
        raise FloatingPointError("executed_weights contains NaN or Inf; refusing to continue")
    if forwards_per_group <= 0:
        raise ValueError("forwards_per_group must be positive")
    total = int(executed_weights.shape[0])
    active = int((executed_weights > 0.0).sum().item())
    return AuxiliaryExecutionPlan(
        run_forwards=global_active,
        forward_count=forwards_per_group * total if global_active else 0,
        active_local_groups=active,
        total_local_groups=total,
    )


def collapse_generation_groups(
    rows: Sequence[Mapping[str, object]],
    num_generations: int,
    *,
    group_key_fields: Sequence[str] = DEFAULT_GROUP_KEY_FIELDS,
) -> list[Mapping[str, object]]:
    """Collapse G-repeated rollout rows to one metadata row per unique prompt.

    The vendored ``RepeatRandomSampler`` emits each dataset index
    ``num_generations`` times consecutively, so each batch of rows is a
    sequence of homogeneous groups.  This function fail-fast verifies that
    layout (group size, per-group key homogeneity, per-batch key uniqueness)
    instead of silently trusting it, and returns one representative row per
    group — the deduplication that keeps the auxiliary branch from paying
    G times the pixel/forward cost.
    """

    if not isinstance(num_generations, int) or num_generations < 2:
        raise ValueError("num_generations must be an integer >= 2")
    if not group_key_fields:
        raise ValueError("group_key_fields must be non-empty")
    rows = list(rows)
    if not rows or len(rows) % num_generations != 0:
        raise ValueError(
            f"row count {len(rows)} is not a positive multiple of "
            f"num_generations={num_generations}"
        )
    groups: list[Mapping[str, object]] = []
    seen_keys: set[tuple[object, ...]] = set()
    for start in range(0, len(rows), num_generations):
        chunk = rows[start : start + num_generations]
        for field in group_key_fields:
            if field not in chunk[0]:
                raise ValueError(f"group key field {field!r} missing from batch rows")
        key = tuple(chunk[0][field] for field in group_key_fields)
        for offset, row in enumerate(chunk[1:], start=1):
            other = tuple(row.get(field) for field in group_key_fields)
            if other != key:
                raise ValueError(
                    f"rollout group layout violated at rows "
                    f"{start}..{start + num_generations - 1}: row {start + offset} "
                    f"has {group_key_fields}={other}, expected {key}"
                )
        if key in seen_keys:
            raise ValueError(
                f"prompt group {key} appears twice in one batch; the auxiliary "
                "branch requires one group per unique prompt"
            )
        seen_keys.add(key)
        groups.append(chunk[0])
    return groups


__all__ = [
    "ADVANTAGE_EPSILON",
    "DEFAULT_GROUP_KEY_FIELDS",
    "AuxiliaryExecutionPlan",
    "GateStrataMetrics",
    "GlobalGatePayload",
    "collapse_generation_groups",
    "compute_global_gate_payload",
    "group_accuracy_rewards",
    "local_group_slice",
    "plan_auxiliary_execution",
]
