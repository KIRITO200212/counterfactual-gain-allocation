"""Outcome-blind deterministic selection helpers for frozen manifests."""

from __future__ import annotations

import hashlib


def stable_scene_subset(
    scene_sizes: dict[str, int], target_rows: int, selection_seed: int
) -> tuple[str, ...]:
    """Choose an exact-size scene subset without using model outcomes.

    Scenes are first ordered by a seeded hash.  Dynamic programming retains
    the first path reaching each row total, so ties are deterministic.  If an
    exact target is impossible, the smallest reachable total above it is used.
    """

    if not scene_sizes or any(
        not isinstance(scene, str) or not scene or size < 1
        for scene, size in scene_sizes.items()
    ):
        raise ValueError("scene_sizes must map non-empty scene ids to positive counts")
    if isinstance(target_rows, bool) or not isinstance(target_rows, int) or target_rows < 1:
        raise ValueError("target_rows must be a positive integer")
    if target_rows > sum(scene_sizes.values()):
        raise ValueError("target_rows exceeds the available operator-valid rows")
    ordered = sorted(
        scene_sizes,
        key=lambda scene: hashlib.sha256(
            f"{selection_seed}:{scene}".encode()
        ).hexdigest(),
    )
    paths: dict[int, tuple[str, ...]] = {0: ()}
    for scene in ordered:
        size = scene_sizes[scene]
        additions: dict[int, tuple[str, ...]] = {}
        for total, path in sorted(paths.items(), reverse=True):
            new_total = total + size
            if new_total not in paths and new_total not in additions:
                additions[new_total] = (*path, scene)
        paths.update(additions)
    selected_total = target_rows if target_rows in paths else min(
        total for total in paths if total > target_rows
    )
    return paths[selected_total]


__all__ = ["stable_scene_subset"]
