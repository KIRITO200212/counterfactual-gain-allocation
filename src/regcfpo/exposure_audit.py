"""Exposure-schedule audit for cycle-position manifests (plan8 A++ contract).

Amendment ``a_plus_plus_screening_authorization_20260831``
``exposure_audit_contract``: every 348-step run contract records axis
exposures, diagonal exposures, unique directional prompts/scenes, the
per-prompt repeat histogram, unseen prompts, replay unique prompts/scenes,
and the AReG/continued schedule identity.  The ``CycleOrderSampler`` layout
is seed-independent by construction, so one pure computation over the
cycle-position-sorted manifest IS the exact schedule every objective leg
(AReG / continued / local) will see -- the schedule-identity field is true
exactly when the manifest layout is a valid single-cycle layout.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Mapping, Sequence


def _repeat_histogram(
    rows: Sequence[Mapping[str, Any]], key_of: Callable[[Mapping[str, Any]], Any]
) -> dict[str, int]:
    """Map "visits per prompt" -> number of prompts with that visit count."""

    visits = Counter(key_of(row) for row in rows)
    histogram = Counter(visits.values())
    return {str(count): histogram[count] for count in sorted(histogram)}


def compute_exposure_audit(
    rows: Sequence[Mapping[str, Any]],
    pool_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compute the exposure-audit contract fields over manifest rows (pure).

    ``rows`` must be in the order the trainer consumes them (sorted by
    ``cycle_position``; ``train_stage4.load_data_rows`` does this for merged
    manifests).  ``pool_rows`` is the optional registered directional pool
    the manifest's directional half was drawn from; when given, unseen-pool
    and off-pool integrity fields are added.
    """

    positions = [row.get("cycle_position") for row in rows]
    contiguous = (
        bool(rows)
        and all(isinstance(position, int) for position in positions)
        and list(positions) == sorted(positions)
        and sorted(positions) == list(range(len(positions)))
    )
    directional = [row for row in rows if row.get("row_kind") == "directional"]
    replay = [row for row in rows if row.get("row_kind") == "replay"]
    axis = [row for row in directional if row.get("relation_class") == "axis"]
    diagonal = [row for row in directional if row.get("relation_class") == "diagonal"]
    replay_key = lambda row: (row.get("scene_id"), row.get("question"))  # noqa: E731
    audit: dict[str, Any] = {
        "total_rows": len(rows),
        "directional_rows": len(directional),
        "replay_rows": len(replay),
        "axis_exposures": len(axis),
        "diagonal_exposures": len(diagonal),
        "unique_directional_prompts": len({row.get("sample_id") for row in directional}),
        "unique_directional_scenes": len({row.get("scene_id") for row in directional}),
        "directional_prompt_repeat_histogram": _repeat_histogram(
            directional, lambda row: row.get("sample_id")
        ),
        "replay_unique_prompts": len({replay_key(row) for row in replay}),
        "replay_unique_scenes": len({row.get("scene_id") for row in replay}),
        "replay_prompt_repeat_histogram": _repeat_histogram(replay, replay_key),
        "cycle_positions_contiguous_sorted": contiguous,
        "sampler": "CycleOrderSampler",
        "schedule_seed_independent": True,
        "areg_continued_schedule_identity": bool(contiguous),
    }
    if pool_rows is not None:
        pool_ids = {row.get("sample_id") for row in pool_rows}
        visited_ids = {row.get("sample_id") for row in directional}
        audit["directional_pool_prompts"] = len(pool_ids)
        audit["unseen_pool_prompts"] = sorted(pool_ids - visited_ids)
        audit["off_pool_visited_prompts"] = sorted(visited_ids - pool_ids)
    return audit


__all__ = ["compute_exposure_audit"]
