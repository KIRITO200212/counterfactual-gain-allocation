"""Scene-level statistical utilities for paired ReG-CFPO comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


class BootstrapError(ValueError):
    """Raised when a paired scene-bootstrap input is invalid."""


def _validate_numeric_vector(name: str, values: ArrayLike) -> NDArray[np.float32]:
    try:
        array = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise BootstrapError(f"{name} must be numeric") from exc
    if array.ndim != 1 or array.size == 0:
        raise BootstrapError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise BootstrapError(f"{name} must contain only finite values")
    return array


def _validate_scene_ids(scene_ids: Sequence[str], expected_size: int) -> tuple[str, ...]:
    if isinstance(scene_ids, (str, bytes)):
        raise BootstrapError("scene_ids must be a sequence, not one string")
    try:
        ids = tuple(scene_ids)
    except TypeError as exc:
        raise BootstrapError("scene_ids must be a sequence") from exc
    if len(ids) != expected_size:
        raise BootstrapError(
            f"scene_ids has length {len(ids)}, expected {expected_size}"
        )
    for scene_id in ids:
        if not isinstance(scene_id, str) or not scene_id.strip():
            raise BootstrapError("every scene_id must be a non-empty string")
    return ids


def aggregate_paired_differences_by_scene(
    treatment: ArrayLike,
    control: ArrayLike,
    scene_ids: Sequence[str],
) -> tuple[tuple[str, ...], NDArray[np.float32], NDArray[np.int64]]:
    """Aggregate paired ``treatment - control`` differences within each scene.

    Each returned scene mean is one statistical unit.  Scene identifiers are
    sorted so results are invariant to input row order.  The third return value
    records the number of paired observations contributing to each scene.
    """

    treatment_array = _validate_numeric_vector("treatment", treatment)
    control_array = _validate_numeric_vector("control", control)
    if treatment_array.shape != control_array.shape:
        raise BootstrapError(
            "treatment and control must have the same shape for a paired analysis"
        )
    ids = _validate_scene_ids(scene_ids, treatment_array.size)
    differences = treatment_array - control_array

    by_scene: dict[str, list[np.float32]] = {}
    for scene_id, difference in zip(ids, differences, strict=True):
        by_scene.setdefault(scene_id, []).append(np.float32(difference))
    ordered_ids = tuple(sorted(by_scene))
    if len(ordered_ids) < 2:
        raise BootstrapError("scene-level bootstrap requires at least two scenes")
    means = np.asarray(
        [np.mean(by_scene[scene_id], dtype=np.float32) for scene_id in ordered_ids],
        dtype=np.float32,
    )
    counts = np.asarray(
        [len(by_scene[scene_id]) for scene_id in ordered_ids], dtype=np.int64
    )
    return ordered_ids, means, counts


@dataclass(frozen=True, slots=True)
class PairedBootstrapResult:
    """Percentile confidence interval from a paired scene bootstrap."""

    estimate: float
    ci_low: float
    ci_high: float
    standard_error: float
    confidence_level: float
    num_resamples: int
    num_scenes: int
    num_pairs: int
    min_pairs_per_scene: int
    max_pairs_per_scene: int
    seed: int | None

    @property
    def confidence_interval(self) -> tuple[float, float]:
        return (self.ci_low, self.ci_high)

    @property
    def excludes_zero(self) -> bool:
        return self.ci_low > 0.0 or self.ci_high < 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimate": self.estimate,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "standard_error": self.standard_error,
            "confidence_level": self.confidence_level,
            "num_resamples": self.num_resamples,
            "num_scenes": self.num_scenes,
            "num_pairs": self.num_pairs,
            "min_pairs_per_scene": self.min_pairs_per_scene,
            "max_pairs_per_scene": self.max_pairs_per_scene,
            "seed": self.seed,
            "excludes_zero": self.excludes_zero,
        }


def scene_level_paired_bootstrap(
    treatment: ArrayLike,
    control: ArrayLike,
    scene_ids: Sequence[str],
    *,
    num_resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int | None = 0,
    chunk_size: int = 4_096,
) -> PairedBootstrapResult:
    """Estimate a paired effect with a scene-level percentile bootstrap.

    Pairwise differences are first averaged within scene.  Scenes are then
    sampled uniformly with replacement, so a scene with many frames/images
    cannot masquerade as many independent observations.  The reported effect
    is therefore the mean per-scene ``treatment - control`` difference.
    """

    if (
        isinstance(num_resamples, bool)
        or not isinstance(num_resamples, (int, np.integer))
        or num_resamples < 2
    ):
        raise BootstrapError("num_resamples must be an integer of at least 2")
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, (int, np.integer))
        or chunk_size < 1
    ):
        raise BootstrapError("chunk_size must be a positive integer")
    if (
        isinstance(confidence_level, bool)
        or not isinstance(confidence_level, (int, float, np.floating))
        or not 0.0 < float(confidence_level) < 1.0
    ):
        raise BootstrapError("confidence_level must lie strictly between 0 and 1")
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, (int, np.integer))
    ):
        raise BootstrapError("seed must be an integer or None")

    _, scene_effects, counts = aggregate_paired_differences_by_scene(
        treatment, control, scene_ids
    )
    num_scenes = scene_effects.size
    rng = np.random.default_rng(seed)
    bootstrap_estimates = np.empty(int(num_resamples), dtype=np.float32)

    # Chunking avoids allocating num_resamples * num_scenes indices at once for
    # the larger Phase-1 manifests.
    for start in range(0, int(num_resamples), int(chunk_size)):
        stop = min(start + int(chunk_size), int(num_resamples))
        sampled_indices = rng.integers(
            0, num_scenes, size=(stop - start, num_scenes), endpoint=False
        )
        bootstrap_estimates[start:stop] = scene_effects[sampled_indices].mean(
            axis=1, dtype=np.float32
        )

    alpha = (1.0 - float(confidence_level)) / 2.0
    ci_low, ci_high = np.asarray(
        np.quantile(bootstrap_estimates, [alpha, 1.0 - alpha], method="linear"),
        dtype=np.float32,
    )
    return PairedBootstrapResult(
        estimate=float(scene_effects.mean(dtype=np.float32)),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        standard_error=float(bootstrap_estimates.std(ddof=1, dtype=np.float32)),
        confidence_level=float(confidence_level),
        num_resamples=int(num_resamples),
        num_scenes=int(num_scenes),
        num_pairs=int(counts.sum()),
        min_pairs_per_scene=int(counts.min()),
        max_pairs_per_scene=int(counts.max()),
        seed=None if seed is None else int(seed),
    )


__all__ = [
    "BootstrapError",
    "PairedBootstrapResult",
    "aggregate_paired_differences_by_scene",
    "scene_level_paired_bootstrap",
]
