#!/usr/bin/env python3
"""One-shot freeze of the final RelationPair-v3 split.

Implements ``final_split_configuration`` from configs/phase1_prereg.yaml
(amendment capacity_diagonal_reserve_revision_20260820, plan4 section 6).
All three preconditions are satisfied (composition PASS, distortion PASS,
contract hardening complete), so this script performs the single freeze:

- directional train pool: 174 unseen eligible prompts / 76 scenes, with the
  preregistered exposure-scheduler plan (axis:diagonal 1:2, scene-uniform,
  family alternation) recorded as metadata, never as duplicated rows;
- final internal holdout: the 8 scenes / 52 prompts from the metadata-only
  reserve reallocation, never reallocated again;
- replay pool: 174 rows in four buckets (multi-view 61, video 26, counting
  26 cross-modality stratified, other single-image 61), fixed selection
  priority, scene-uniform seeded sampling;
- saturation dual-cohort manifests: train-distribution 174 groups and
  confirmation-distribution 291 groups, reported separately;
- fail-closed lineage assertions verified against raw manifests, not row
  fields; atomic writes; contract hash frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from regcfpo.data_v3 import RelationPairV3
from regcfpo.eligibility import (
    assert_scene_lineage_disjoint,
    evaluate_pixel_pair_eligibility,
)

SEED = 20270821
PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPLAY_BUCKETS = [
    ("multi_view_non_counting_non_direction", 61, ("multi_view",)),
    ("video_non_counting_non_direction", 26, ("video",)),
    ("counting_cross_modality_stratified", 26, ("video", "multi_view", "single_image")),
    ("other_single_image_non_counting_non_direction", 61, ("single_image",)),
]
# Preregistered strata were video/multi_view/single_image 9/9/8, but the
# released single_image split contains zero object-count rows (verified by
# read-only enumeration before any freeze took effect).  The 26-row counting
# quota is therefore met by the two feasible modalities only; this
# feasibility correction is recorded in the freeze report.
COUNTING_STRATA = {"video": 13, "multi_view": 13}
SELECTION_PRIORITY = [
    "counting",
    "video_non_counting",
    "multi_view_non_counting",
    "single_image_non_counting",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        raise SystemExit(f"partial output {tmp} exists; refusing to continue")
    with tmp.open("w", encoding="utf-8") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def base_scene(image: str) -> str:
    return image.partition("/")[0].rsplit("_", 1)[0]


def scene_uniform_pick(
    rows_by_scene: dict[str, list[dict[str, Any]]],
    count: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Round-robin over a seeded scene order; one row per scene per cycle."""

    scenes = sorted(rows_by_scene)
    rng.shuffle(scenes)
    per_scene = {scene: list(rows_by_scene[scene]) for scene in scenes}
    for scene in scenes:
        rng.shuffle(per_scene[scene])
    picked: list[dict[str, Any]] = []
    while len(picked) < count:
        progressed = False
        for scene in scenes:
            if len(picked) >= count:
                break
            if per_scene[scene]:
                picked.append(per_scene[scene].pop())
                progressed = True
        if not progressed:
            raise SystemExit("replay pool exhausted before quota was met")
    return picked


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3-candidates", required=True, type=Path)
    parser.add_argument("--capacity-rows", required=True, type=Path)
    parser.add_argument("--reserve-realloc", required=True, type=Path)
    parser.add_argument("--viewed-diagnostic", required=True, type=Path)
    parser.add_argument("--spatial", required=True, type=Path)
    parser.add_argument("--confirmation-run", required=True, type=Path)
    parser.add_argument("--confirmation-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    rng = random.Random(SEED)
    realloc = json.loads(args.reserve_realloc.read_text(encoding="utf-8"))
    holdout_scenes = set(realloc["holdout_scenes"])

    # --- viewed lineage from raw manifests (never row fields) ---
    viewed_scenes = {
        row["scene_id"] for row in read_jsonl(args.viewed_diagnostic)
    }

    # --- directional train pool ---
    capacity = {row["sample_id"]: row for row in read_jsonl(args.capacity_rows)}
    train_rows: list[dict[str, Any]] = []
    for row in read_jsonl(args.v3_candidates):
        pair = RelationPairV3.from_dict(row)  # fail-closed validation
        cap = capacity.get(pair.sample_id)
        if cap is None or cap["lineage"] != "unseen_lineage" or not cap["eligible"]:
            continue
        if pair.scene_id in holdout_scenes:
            continue
        result = evaluate_pixel_pair_eligibility(
            pair.gt_box_a, pair.gt_box_b, tuple(cap["image_size_hw"])
        )
        if not result.eligible:
            raise SystemExit(
                f"shared eligibility disagrees with capacity audit on {pair.sample_id}"
            )
        train_rows.append(row)
    train_rows.sort(key=lambda r: r["sample_id"])
    axis_rows = [r for r in train_rows if r["relation_class"] == "axis"]
    diagonal_rows = [r for r in train_rows if r["relation_class"] == "diagonal"]
    families = Counter(r["answer_relation"] for r in diagonal_rows)

    # --- holdout rows ---
    holdout_rows = [
        row
        for row in read_jsonl(args.v3_candidates)
        if row["scene_id"] in holdout_scenes
        and capacity.get(row["sample_id"], {}).get("eligible")
        and capacity.get(row["sample_id"], {}).get("lineage") == "unseen_lineage"
    ]
    holdout_rows.sort(key=lambda r: r["sample_id"])
    for row in holdout_rows:
        RelationPairV3.from_dict(row)

    # --- replay pool ---
    excluded_scenes = viewed_scenes | holdout_scenes | {
        row["scene_id"] for row in train_rows
    }
    spatial = read_jsonl(args.spatial)
    bucket_candidates: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for name, _quota, modalities in REPLAY_BUCKETS:
        by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in spatial:
            if row.get("data_type") not in modalities:
                continue
            qt = row.get("question_type")
            if name == "counting_cross_modality_stratified":
                if qt != "object count":
                    continue
            else:
                if qt == "object count" or qt == "relative direction":
                    continue
                # room size stays included: it is the available non-counting
                # video task and carries no left/right answer vocabulary
            scene = base_scene(row["image"][0])
            if scene in excluded_scenes:
                continue
            by_scene[scene].append(row)
        bucket_candidates[name] = by_scene

    replay_rows: list[dict[str, Any]] = []
    replay_manifest_rows: list[dict[str, Any]] = []
    used_ids: set[int] = set()
    for name, quota, modalities in REPLAY_BUCKETS:
        if name == "counting_cross_modality_stratified":
            picked: list[dict[str, Any]] = []
            for modality, stratum_quota in COUNTING_STRATA.items():
                by_scene = {
                    scene: rows
                    for scene, rows in bucket_candidates[name].items()
                    if rows and rows[0].get("data_type") == modality
                }
                picked.extend(scene_uniform_pick(by_scene, stratum_quota, rng))
        else:
            picked = scene_uniform_pick(bucket_candidates[name], quota, rng)
        for row in picked:
            if row["question_id"] in used_ids:
                raise SystemExit("duplicate question_id in replay pool")
            used_ids.add(row["question_id"])
            replay_rows.append(row)
            replay_manifest_rows.append(
                {
                    "question_id": row["question_id"],
                    "scene_id": base_scene(row["image"][0]),
                    "image": row["image"][0],
                    "data_type": row["data_type"],
                    "question_type": row["question_type"],
                    "replay_bucket": name,
                    "question": row["question"],
                    "options": row["options"],
                    "answer": row["answer"],
                }
            )
    replay_scenes = {row["scene_id"] for row in replay_manifest_rows}

    # --- lineage assertions against raw scene sets ---
    assert_scene_lineage_disjoint(
        {
            "train": {row["scene_id"] for row in train_rows},
            "replay": replay_scenes,
            "holdout": holdout_scenes,
        },
        viewed_scenes=viewed_scenes,
    )

    # --- saturation dual-cohort manifests ---
    train_cohort = [
        {
            "sample_id": row["sample_id"],
            "scene_id": row["scene_id"],
            "image": row["image"],
            "question": row["question"],
            "options": row["options"],
            "answer_letter": row["answer_letter"],
            "relation_class": row["relation_class"],
            "cohort": "train_distribution",
        }
        for row in train_rows
    ]
    v2_confirmation = {
        row["sample_id"]: row for row in read_jsonl(args.confirmation_manifest)
    }
    confirmation_cohort = []
    for row in read_jsonl(args.confirmation_run):
        if row["branches"]["pixel_pair_slot_swap"]["accepted"]:
            source = v2_confirmation[row["sample_id"]]
            confirmation_cohort.append(
                {
                    "sample_id": row["sample_id"],
                    "scene_id": row["scene_id"],
                    "image": row["image"],
                    "question": source["question"],
                    "options": source["options"],
                    "answer_letter": source["answer_letter"],
                    "cohort": "confirmation_distribution",
                }
            )

    # --- exposure scheduler plan (metadata only, no row duplication) ---
    scheduler = {
        "axis_to_diagonal": "1:2",
        "rule": (
            "per 3 directional slots take 1 axis and 2 diagonal; scene-uniform "
            "within each class; at most one prompt per scene per cycle; the "
            "two diagonal families alternate 1:1; record unique-scene exposure "
            "and per-prompt repetition counts at training time"
        ),
        "natural_pool": {
            "axis": len(axis_rows),
            "diagonal": len(diagonal_rows),
            "diagonal_families": dict(sorted(families.items())),
        },
    }

    outputs = {
        "directional_train": (args.output_dir / "v3_directional_train.jsonl", train_rows),
        "final_holdout": (args.output_dir / "v3_final_holdout.jsonl", holdout_rows),
        "replay": (args.output_dir / "v3_replay.jsonl", replay_manifest_rows),
        "saturation_train_cohort": (
            args.output_dir / "saturation_manifest_train_distribution.jsonl",
            train_cohort,
        ),
    }
    hashes = {}
    for key, (path, rows) in outputs.items():
        write_jsonl_atomic(path, rows)
        hashes[key] = sha256_file(path)
    sat_conf_path = args.output_dir / "saturation_manifest_confirmation_distribution.jsonl"
    write_jsonl_atomic(sat_conf_path, confirmation_cohort)
    hashes["saturation_confirmation_cohort"] = sha256_file(sat_conf_path)

    freeze_record = {
        "schema_version": 1,
        "kind": "final_v3_split_freeze",
        "one_shot": True,
        "seed": SEED,
        "selection_priority": SELECTION_PRIORITY,
        "replay_buckets": {name: quota for name, quota, _ in REPLAY_BUCKETS},
        "counting_strata": COUNTING_STRATA,
        "counting_strata_feasibility_note": (
            "preregistered strata were video/multi_view/single_image 9/9/8; "
            "the released single_image data contains no object-count rows, so "
            "the quota is met by video 13 + multi_view 13 (cross-modality "
            "stratification retained over the feasible modalities)"
        ),
        "directional_train": {
            "rows": len(train_rows),
            "scenes": len({row["scene_id"] for row in train_rows}),
            "axis": len(axis_rows),
            "diagonal": len(diagonal_rows),
            "diagonal_families": dict(sorted(families.items())),
        },
        "final_holdout": {
            "scenes": len(holdout_scenes),
            "rows": len(holdout_rows),
            "axis": sum(1 for r in holdout_rows if r["relation_class"] == "axis"),
            "diagonal": sum(1 for r in holdout_rows if r["relation_class"] == "diagonal"),
            "no_reallocation": True,
        },
        "replay_pool": {
            "rows": len(replay_manifest_rows),
            "scenes": len(replay_scenes),
            "buckets": dict(
                sorted(Counter(row["replay_bucket"] for row in replay_manifest_rows).items())
            ),
            "question_types": dict(
                sorted(Counter(row["question_type"] for row in replay_manifest_rows).items())
            ),
        },
        "saturation_dual_cohort": {
            "train_distribution_groups": len(train_cohort),
            "confirmation_distribution_groups": len(confirmation_cohort),
            "reporting": "separate, never pooled; dual-cohort 465-group audit",
        },
        "exposure_scheduler": scheduler,
        "lineage_assertions": {
            "train_intersect_viewed_empty": True,
            "replay_intersect_viewed_empty": True,
            "holdout_intersect_viewed_empty": True,
            "pairwise_disjoint": True,
            "verified_against_raw_manifests": True,
        },
        "output_sha256": hashes,
        "input_sha256": {
            "v3_candidates": sha256_file(args.v3_candidates),
            "capacity_rows": sha256_file(args.capacity_rows),
            "reserve_reallocation": sha256_file(args.reserve_realloc),
            "viewed_diagnostic": sha256_file(args.viewed_diagnostic),
            "spatial": sha256_file(args.spatial),
            "confirmation_run": sha256_file(args.confirmation_run),
            "confirmation_manifest": sha256_file(args.confirmation_manifest),
        },
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }
    freeze_path = args.output_dir / "v3_final_split_freeze.json"
    tmp = freeze_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(freeze_record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(freeze_path)
    print(json.dumps(freeze_record, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
