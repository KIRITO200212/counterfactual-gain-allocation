"""Reward-saturation diagnostics for G=4 factual rollout groups."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


DEFAULT_GROUP_SIZE = 4
OFFICIAL_GRPO_EPSILON = 1e-4
OFFICIAL_GRPO_STD_CORRECTION = 1
OFFICIAL_REWARD_WEIGHTS = (1.0, 1.0)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_FORMAT_RE = re.compile(r"^<think>.*?</think>\s*<answer>.*?</answer>$", re.DOTALL)


class RewardAuditError(ValueError):
    """Raised when reward or gate inputs violate the audit contract."""


class SaturationStratum(str, Enum):
    """Mutually exclusive factual-reward strata."""

    SAT = "SAT"
    MIX = "MIX"
    ALL_WRONG = "ALL_WRONG"


def clean_stage3_mca_text(text: str) -> str:
    """Mirror the official Stage-3 multiple-choice normalization exactly."""

    if not isinstance(text, str):
        raise RewardAuditError("multiple-choice text must be a string")
    matches = _ANSWER_RE.findall(text)
    value = matches[-1] if matches else text
    for char in ("\n", "\r"):
        value = re.sub(r"(?<=\s)" + re.escape(char), "", value)
        value = re.sub(r"(?<!\s)" + re.escape(char), " ", value)
    return value.strip().rstrip(".").lower()


def stage3_format_valid(text: str) -> bool:
    """Apply the official Stage-3 think/answer format predicate."""

    if not isinstance(text, str):
        raise RewardAuditError("completion text must be a string")
    stripped = text.strip()
    if _FORMAT_RE.fullmatch(stripped) is None:
        return False
    think_match = re.search(r"<think>(.*?)</think>", stripped, re.DOTALL)
    if think_match is not None and "<think>" in think_match.group(1):
        return False
    return stripped.count("<think>") == 1 and stripped.count("<answer>") == 1


def _validate_reward_matrix(
    rewards: ArrayLike,
    *,
    expected_group_size: int = DEFAULT_GROUP_SIZE,
    name: str = "rewards",
) -> NDArray[np.float32]:
    array = np.asarray(rewards, dtype=np.float32)
    if array.ndim != 2:
        raise RewardAuditError(
            f"{name} must have shape [num_groups, {expected_group_size}]"
        )
    if array.shape[0] == 0:
        raise RewardAuditError(f"{name} must contain at least one group")
    if array.shape[1] != expected_group_size:
        raise RewardAuditError(
            f"{name}: expected G={expected_group_size}, received G={array.shape[1]}"
        )
    if not np.all(np.isfinite(array)):
        raise RewardAuditError(f"{name} must be finite")
    return array


def _validate_binary_rewards(
    rewards: ArrayLike,
    *,
    expected_group_size: int = DEFAULT_GROUP_SIZE,
    name: str = "rewards",
    atol: float = 1e-8,
) -> NDArray[np.float32]:
    array = _validate_reward_matrix(
        rewards,
        expected_group_size=expected_group_size,
        name=name,
    )
    is_zero = np.isclose(array, 0.0, rtol=0.0, atol=atol)
    is_one = np.isclose(array, 1.0, rtol=0.0, atol=atol)
    if not np.all(is_zero | is_one):
        invalid = array[~(is_zero | is_one)][0]
        raise RewardAuditError(
            f"{name} must be binary (0/1); found {float(invalid)!r}"
        )
    # Canonicalizing near-binary values makes downstream equality deterministic.
    return np.where(is_one, np.float32(1.0), np.float32(0.0)).astype(
        np.float32, copy=False
    )


def official_total_rewards(
    accuracy_rewards: ArrayLike,
    format_rewards: ArrayLike,
    *,
    expected_group_size: int = DEFAULT_GROUP_SIZE,
) -> NDArray[np.float32]:
    """Compose the official equal-weight accuracy-plus-format reward."""

    accuracy = _validate_binary_rewards(
        accuracy_rewards,
        expected_group_size=expected_group_size,
        name="accuracy_rewards",
    )
    formatting = _validate_binary_rewards(
        format_rewards,
        expected_group_size=expected_group_size,
        name="format_rewards",
    )
    if formatting.shape != accuracy.shape:
        raise RewardAuditError(
            "format_rewards shape "
            f"{formatting.shape} does not match accuracy_rewards shape {accuracy.shape}"
        )
    accuracy_weight, format_weight = (
        np.float32(weight) for weight in OFFICIAL_REWARD_WEIGHTS
    )
    return (accuracy_weight * accuracy + format_weight * formatting).astype(
        np.float32, copy=False
    )


def classify_reward_group(
    rewards: Sequence[float],
    *,
    expected_group_size: int = DEFAULT_GROUP_SIZE,
) -> SaturationStratum:
    """Classify one factual rollout group as SAT, MIX, or ALL_WRONG."""

    array = _validate_binary_rewards(
        np.asarray(rewards, dtype=np.float32).reshape(1, -1),
        expected_group_size=expected_group_size,
    )[0]
    correct = int(array.sum())
    if correct == expected_group_size:
        return SaturationStratum.SAT
    if correct == 0:
        return SaturationStratum.ALL_WRONG
    return SaturationStratum.MIX


def stratify_reward_groups(
    rewards: ArrayLike,
    *,
    expected_group_size: int = DEFAULT_GROUP_SIZE,
) -> tuple[SaturationStratum, ...]:
    """Classify every row of a binary reward matrix."""

    array = _validate_binary_rewards(
        rewards, expected_group_size=expected_group_size
    )
    totals = array.sum(axis=1)
    return tuple(
        SaturationStratum.SAT
        if total == expected_group_size
        else SaturationStratum.ALL_WRONG
        if total == 0
        else SaturationStratum.MIX
        for total in totals
    )


def stratum_indices(
    rewards: ArrayLike,
    *,
    expected_group_size: int = DEFAULT_GROUP_SIZE,
) -> dict[SaturationStratum, NDArray[np.int64]]:
    """Return stable group indices for each saturation stratum."""

    strata = stratify_reward_groups(rewards, expected_group_size=expected_group_size)
    return {
        stratum: np.asarray(
            [index for index, value in enumerate(strata) if value is stratum],
            dtype=np.int64,
        )
        for stratum in SaturationStratum
    }


def grpo_advantages(
    rewards: ArrayLike,
    *,
    expected_group_size: int = DEFAULT_GROUP_SIZE,
    eps: float = OFFICIAL_GRPO_EPSILON,
) -> NDArray[np.float32]:
    """Compute the official group-relative standardized total reward.

    The released trainer uses ``torch.std`` with its default correction of one
    and divides by ``std + 1e-4``.  This function mirrors that computation in
    FP32 and intentionally accepts non-binary totals such as 0, 1, and 2.
    """

    if not isinstance(eps, (int, float)) or isinstance(eps, bool) or eps <= 0:
        raise RewardAuditError("eps must be a positive real number")
    array = _validate_reward_matrix(
        rewards,
        expected_group_size=expected_group_size,
        name="total_rewards",
    )
    if expected_group_size <= OFFICIAL_GRPO_STD_CORRECTION:
        raise RewardAuditError("official GRPO advantage requires group size at least two")
    centered = array - array.mean(axis=1, keepdims=True, dtype=np.float32)
    scale = array.std(
        axis=1,
        ddof=OFFICIAL_GRPO_STD_CORRECTION,
        keepdims=True,
        dtype=np.float32,
    )
    return (centered / (scale + np.float32(eps))).astype(np.float32, copy=False)


def _validate_advantages(
    advantages: ArrayLike,
    *,
    expected_shape: tuple[int, int] | None = None,
) -> NDArray[np.float32]:
    array = np.asarray(advantages, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0:
        raise RewardAuditError("advantages must be a non-empty two-dimensional array")
    if expected_shape is not None and array.shape != expected_shape:
        raise RewardAuditError(
            f"advantages shape {array.shape} does not match rewards shape {expected_shape}"
        )
    if not np.all(np.isfinite(array)):
        raise RewardAuditError("advantages must be finite")
    return array


def positive_advantage_gate(advantages: ArrayLike) -> NDArray[np.float32]:
    """Return the original rollout-level ``max(A, 0)`` gate."""

    array = _validate_advantages(advantages)
    return np.maximum(array, 0.0)


def positive_advantage_group_gate(advantages: ArrayLike) -> NDArray[np.float32]:
    """Reduce the original gate to one weight per rollout group."""

    return positive_advantage_gate(advantages).max(axis=1)


def _validate_confidence(
    name: str, confidence: ArrayLike | float, num_groups: int
) -> NDArray[np.float32]:
    array = np.asarray(confidence, dtype=np.float32)
    if array.ndim == 0:
        array = np.full(num_groups, np.float32(array), dtype=np.float32)
    if array.shape != (num_groups,):
        raise RewardAuditError(f"{name} must be a scalar or shape ({num_groups},)")
    if not np.all(np.isfinite(array)) or np.any((array < 0.0) | (array > 1.0)):
        raise RewardAuditError(f"{name} must contain finite values in [0, 1]")
    return array


def saturation_aware_success_gate(
    accuracy_rewards: ArrayLike,
    total_advantages: ArrayLike,
    *,
    locator_confidence: ArrayLike | float = 1.0,
    operator_confidence: ArrayLike | float = 1.0,
    expected_group_size: int = DEFAULT_GROUP_SIZE,
    eps: float = 1e-8,
) -> NDArray[np.float32]:
    """Compute the preregistered group-level saturation-aware gate.

    Accuracy alone defines SAT/MIX/ALL_WRONG.  SAT groups receive base weight
    one, ALL_WRONG groups receive zero, and MIX groups use maximum positive
    *total-reward* advantage normalized by the MIX-group mean.  Keeping these
    two inputs separate prevents format reward from changing stratum identity.

    The base weight is multiplied by frozen locator/operator confidences.  No
    batch mean normalization or clipping is performed here; use
    :func:`normalize_gate_weights` for the final training weight.
    """

    if not isinstance(eps, (int, float)) or isinstance(eps, bool) or eps <= 0:
        raise RewardAuditError("eps must be a positive real number")
    accuracy_array = _validate_binary_rewards(
        accuracy_rewards,
        expected_group_size=expected_group_size,
        name="accuracy_rewards",
    )
    advantage_array = _validate_advantages(
        total_advantages, expected_shape=accuracy_array.shape
    )

    totals = accuracy_array.sum(axis=1, dtype=np.float32)
    sat_mask = totals == expected_group_size
    mix_mask = (totals > 0) & (totals < expected_group_size)
    max_positive = positive_advantage_group_gate(advantage_array)

    base = np.zeros(accuracy_array.shape[0], dtype=np.float32)
    base[sat_mask] = 1.0
    if np.any(mix_mask):
        denominator = np.float32(max_positive[mix_mask].mean(dtype=np.float32)) + np.float32(
            eps
        )
        base[mix_mask] = max_positive[mix_mask] / denominator

    locator = _validate_confidence(
        "locator_confidence", locator_confidence, accuracy_array.shape[0]
    )
    operator = _validate_confidence(
        "operator_confidence", operator_confidence, accuracy_array.shape[0]
    )
    return base * locator * operator


def normalize_gate_weights(
    weights: ArrayLike,
    *,
    eps: float = 1e-8,
    max_weight: float | None = None,
) -> NDArray[np.float32]:
    """Mean-normalize non-negative gate weights and optionally clip them."""

    array = np.asarray(weights, dtype=np.float32)
    if array.ndim != 1 or array.size == 0:
        raise RewardAuditError("weights must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise RewardAuditError("weights must be finite and non-negative")
    if not isinstance(eps, (int, float)) or isinstance(eps, bool) or eps <= 0:
        raise RewardAuditError("eps must be a positive real number")
    if max_weight is not None and (
        not isinstance(max_weight, (int, float))
        or isinstance(max_weight, bool)
        or not np.isfinite(max_weight)
        or max_weight <= 0
    ):
        raise RewardAuditError("max_weight must be a positive finite number")

    normalized = array / (
        np.float32(array.mean(dtype=np.float32)) + np.float32(eps)
    )
    if max_weight is not None:
        normalized = np.minimum(normalized, np.float32(max_weight))
    return normalized.astype(np.float32, copy=False)


def gate_coverage(gate: ArrayLike, *, threshold: float = 0.0) -> float:
    """Return the fraction of gate entries strictly above ``threshold``."""

    array = np.asarray(gate, dtype=np.float32)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise RewardAuditError("gate must contain at least one finite value")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise RewardAuditError("threshold must be a real number")
    return float(np.mean(array > float(threshold), dtype=np.float32))


@dataclass(frozen=True, slots=True)
class RewardSaturationAudit:
    """Serializable summary of a frozen G=4 reward audit."""

    reward_weights: tuple[float, float]
    advantage_epsilon: float
    advantage_std_correction: int
    num_groups: int
    group_size: int
    num_rollouts: int
    rollout_accuracy: float
    rollout_format_reward: float
    rollout_total_reward: float
    sat_groups: int
    mix_groups: int
    all_wrong_groups: int
    sat_fraction: float
    mix_fraction: float
    all_wrong_fraction: float
    mean_group_accuracy_variance: float
    mean_group_total_reward_variance: float
    positive_advantage_gate_coverage: float
    positive_advantage_rollout_coverage: float
    nonzero_advantage_group_coverage: float
    nonzero_advantage_rollout_coverage: float
    saturation_aware_gate_coverage: float
    saturation_aware_mean_weight: float

    @property
    def stratum_counts(self) -> Mapping[str, int]:
        return {
            SaturationStratum.SAT.value: self.sat_groups,
            SaturationStratum.MIX.value: self.mix_groups,
            SaturationStratum.ALL_WRONG.value: self.all_wrong_groups,
        }

    @property
    def stratum_fractions(self) -> Mapping[str, float]:
        return {
            SaturationStratum.SAT.value: self.sat_fraction,
            SaturationStratum.MIX.value: self.mix_fraction,
            SaturationStratum.ALL_WRONG.value: self.all_wrong_fraction,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "reward_weights": list(self.reward_weights),
            "advantage_epsilon": self.advantage_epsilon,
            "advantage_std_correction": self.advantage_std_correction,
            "num_groups": self.num_groups,
            "group_size": self.group_size,
            "num_rollouts": self.num_rollouts,
            "rollout_accuracy": self.rollout_accuracy,
            "rollout_format_reward": self.rollout_format_reward,
            "rollout_total_reward": self.rollout_total_reward,
            "stratum_counts": dict(self.stratum_counts),
            "stratum_fractions": dict(self.stratum_fractions),
            "variance_ddof": 0,
            "mean_group_accuracy_variance": self.mean_group_accuracy_variance,
            "mean_group_total_reward_variance": self.mean_group_total_reward_variance,
            "positive_advantage_gate_coverage": self.positive_advantage_gate_coverage,
            "positive_advantage_rollout_coverage": self.positive_advantage_rollout_coverage,
            "nonzero_advantage_group_coverage": self.nonzero_advantage_group_coverage,
            "nonzero_advantage_rollout_coverage": self.nonzero_advantage_rollout_coverage,
            "saturation_aware_gate_coverage": self.saturation_aware_gate_coverage,
            "saturation_aware_mean_weight": self.saturation_aware_mean_weight,
        }


def audit_reward_saturation(
    accuracy_rewards: ArrayLike,
    format_rewards: ArrayLike,
    *,
    locator_confidence: ArrayLike | float = 1.0,
    operator_confidence: ArrayLike | float = 1.0,
    expected_group_size: int = DEFAULT_GROUP_SIZE,
) -> RewardSaturationAudit:
    """Audit accuracy strata while preserving official total-reward advantage."""

    accuracy = _validate_binary_rewards(
        accuracy_rewards,
        expected_group_size=expected_group_size,
        name="accuracy_rewards",
    )
    formatting = _validate_binary_rewards(
        format_rewards,
        expected_group_size=expected_group_size,
        name="format_rewards",
    )
    if formatting.shape != accuracy.shape:
        raise RewardAuditError(
            "format_rewards shape "
            f"{formatting.shape} does not match accuracy_rewards shape {accuracy.shape}"
        )
    total_rewards = official_total_rewards(
        accuracy,
        formatting,
        expected_group_size=expected_group_size,
    )
    total_advantages = grpo_advantages(
        total_rewards, expected_group_size=expected_group_size
    )
    positive_rollout = positive_advantage_gate(total_advantages)
    positive_group = positive_rollout.max(axis=1)
    saturation_gate = saturation_aware_success_gate(
        accuracy,
        total_advantages,
        locator_confidence=locator_confidence,
        operator_confidence=operator_confidence,
        expected_group_size=expected_group_size,
    )
    accuracy_totals = accuracy.sum(axis=1, dtype=np.float32)
    sat = accuracy_totals == expected_group_size
    all_wrong = accuracy_totals == 0
    mix = ~(sat | all_wrong)
    nonzero = np.abs(total_advantages) > 0.0
    num_groups = accuracy.shape[0]

    return RewardSaturationAudit(
        reward_weights=OFFICIAL_REWARD_WEIGHTS,
        advantage_epsilon=OFFICIAL_GRPO_EPSILON,
        advantage_std_correction=OFFICIAL_GRPO_STD_CORRECTION,
        num_groups=num_groups,
        group_size=expected_group_size,
        num_rollouts=int(accuracy.size),
        rollout_accuracy=float(accuracy.mean(dtype=np.float32)),
        rollout_format_reward=float(formatting.mean(dtype=np.float32)),
        rollout_total_reward=float(total_rewards.mean(dtype=np.float32)),
        sat_groups=int(sat.sum()),
        mix_groups=int(mix.sum()),
        all_wrong_groups=int(all_wrong.sum()),
        sat_fraction=float(sat.mean(dtype=np.float32)),
        mix_fraction=float(mix.mean(dtype=np.float32)),
        all_wrong_fraction=float(all_wrong.mean(dtype=np.float32)),
        mean_group_accuracy_variance=float(
            accuracy.var(axis=1, ddof=0, dtype=np.float32).mean(dtype=np.float32)
        ),
        mean_group_total_reward_variance=float(
            total_rewards.var(axis=1, ddof=0, dtype=np.float32).mean(dtype=np.float32)
        ),
        positive_advantage_gate_coverage=gate_coverage(positive_group),
        positive_advantage_rollout_coverage=gate_coverage(positive_rollout),
        nonzero_advantage_group_coverage=float(
            np.mean(nonzero.any(axis=1), dtype=np.float32)
        ),
        nonzero_advantage_rollout_coverage=float(
            nonzero.mean(dtype=np.float32)
        ),
        saturation_aware_gate_coverage=gate_coverage(saturation_gate),
        saturation_aware_mean_weight=float(saturation_gate.mean(dtype=np.float32)),
    )


__all__ = [
    "DEFAULT_GROUP_SIZE",
    "OFFICIAL_GRPO_EPSILON",
    "OFFICIAL_GRPO_STD_CORRECTION",
    "OFFICIAL_REWARD_WEIGHTS",
    "RewardAuditError",
    "RewardSaturationAudit",
    "SaturationStratum",
    "audit_reward_saturation",
    "clean_stage3_mca_text",
    "classify_reward_group",
    "gate_coverage",
    "grpo_advantages",
    "normalize_gate_weights",
    "official_total_rewards",
    "positive_advantage_gate",
    "positive_advantage_group_gate",
    "saturation_aware_success_gate",
    "stratify_reward_groups",
    "stratum_indices",
    "stage3_format_valid",
]
