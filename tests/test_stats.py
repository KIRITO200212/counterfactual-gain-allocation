from __future__ import annotations

import numpy as np
import pytest

from regcfpo.stats import (
    BootstrapError,
    aggregate_paired_differences_by_scene,
    scene_level_paired_bootstrap,
)


def test_scene_aggregation_preserves_pairing_and_equal_scene_weight() -> None:
    scene_ids, effects, counts = aggregate_paired_differences_by_scene(
        treatment=[10.0, 12.0, -1.0],
        control=[0.0, 2.0, 0.0],
        scene_ids=["scene-a", "scene-a", "scene-b"],
    )

    assert scene_ids == ("scene-a", "scene-b")
    assert effects.dtype == np.float32
    np.testing.assert_allclose(effects, [10.0, -1.0])
    np.testing.assert_array_equal(counts, [2, 1])
    # Equal scene weighting gives 4.5, not the observation-weighted 19 / 3.
    result = scene_level_paired_bootstrap(
        [10.0, 12.0, -1.0],
        [0.0, 2.0, 0.0],
        ["scene-a", "scene-a", "scene-b"],
        num_resamples=1_000,
        seed=7,
    )
    assert result.estimate == pytest.approx(4.5)
    assert result.num_scenes == 2
    assert result.num_pairs == 3
    assert result.min_pairs_per_scene == 1
    assert result.max_pairs_per_scene == 2


def test_bootstrap_is_deterministic_and_invariant_to_row_order() -> None:
    treatment = np.asarray([1.2, 0.7, 2.0, 1.3, 0.2, 1.9])
    control = np.asarray([0.1, 0.4, 1.1, 0.9, 0.0, 1.0])
    scenes = np.asarray(["c", "a", "b", "c", "a", "b"])
    permutation = np.asarray([4, 2, 0, 5, 1, 3])

    first = scene_level_paired_bootstrap(
        treatment, control, scenes, num_resamples=2_000, seed=123, chunk_size=127
    )
    second = scene_level_paired_bootstrap(
        treatment[permutation],
        control[permutation],
        scenes[permutation],
        num_resamples=2_000,
        seed=123,
        chunk_size=127,
    )

    assert first == second
    assert first.to_dict() == second.to_dict()


def test_positive_paired_effect_has_ci_excluding_zero() -> None:
    scene_ids = [f"scene-{index:02d}" for index in range(30)]
    control = np.linspace(-2.0, 2.0, num=30)
    treatment = control + np.linspace(0.6, 1.4, num=30)

    result = scene_level_paired_bootstrap(
        treatment,
        control,
        scene_ids,
        num_resamples=5_000,
        confidence_level=0.95,
        seed=2027,
    )

    assert result.estimate == pytest.approx(1.0)
    assert result.ci_low > 0.0
    assert result.excludes_zero
    assert result.confidence_interval == (result.ci_low, result.ci_high)
    assert result.standard_error > 0.0


def test_pairing_removes_large_shared_scene_baseline() -> None:
    baselines = np.asarray([1000.0, -500.0, 300.0, -900.0])
    result = scene_level_paired_bootstrap(
        treatment=baselines + 0.25,
        control=baselines,
        scene_ids=["a", "b", "c", "d"],
        num_resamples=500,
        seed=1,
    )
    assert result.estimate == pytest.approx(0.25)
    assert result.ci_low == pytest.approx(0.25)
    assert result.ci_high == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("treatment", "control", "scenes", "message"),
    [
        ([1.0, 2.0], [1.0], ["a", "b"], "same shape"),
        ([1.0, np.nan], [0.0, 0.0], ["a", "b"], "finite"),
        ([1.0, 2.0], [0.0, 0.0], ["a"], "length"),
        ([1.0, 2.0], [0.0, 0.0], ["a", "a"], "at least two scenes"),
        ([1.0, 2.0], [0.0, 0.0], ["a", ""], "non-empty string"),
    ],
)
def test_bootstrap_rejects_invalid_paired_inputs(
    treatment: list[float],
    control: list[float],
    scenes: list[str],
    message: str,
) -> None:
    with pytest.raises(BootstrapError, match=message):
        scene_level_paired_bootstrap(
            treatment, control, scenes, num_resamples=100, seed=0
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_resamples": 1}, "at least 2"),
        ({"confidence_level": 1.0}, "strictly between"),
        ({"chunk_size": 0}, "positive integer"),
        ({"seed": 1.5}, "integer or None"),
    ],
)
def test_bootstrap_rejects_invalid_configuration(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(BootstrapError, match=message):
        scene_level_paired_bootstrap(
            [1.0, 2.0], [0.0, 0.0], ["a", "b"], **kwargs
        )
