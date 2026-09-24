#!/usr/bin/env python3
"""Join released SpatialLadder annotations into strict RelationPair-v3 rows.

Implements the preregistered ``relation_pair_v3`` protocol
(configs/phase1_prereg.yaml, amendment relation_pair_v3_protocol_20260820):

- six swap-flippable relations with the point-reflection mapping;
- composite join key (question_id, image, data_type) with hard uniqueness;
- explicit rejection of equal x-centers (relation undefined);
- boundary-aware unique entity spans (frozen v2 rule);
- operator_valid requires the horizontal center order to match the answer's
  horizontal component; diagonal depth components are not verifiable from
  2D boxes and are recorded as hypotheses for unseen-scene validation.

Read-only with respect to model outcomes.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from regcfpo.data import spatialladder_base_scene_id
from regcfpo.data_v3 import (
    AXIS_RELATIONS,
    DIAGONAL_RELATIONS,
    OptionRelationV3Error,
    RelationPairV3,
    horizontal_component,
    resolve_option_relations_v3,
    swap_mapped_relation_v3,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_relation_pair_v2 import ordered_entities  # noqa: E402  (frozen v2 span rule)


def read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield line_number, value


def index_by_composite_key(path: Path) -> dict[tuple[Any, str, str], dict[str, Any]]:
    result: dict[tuple[Any, str, str], dict[str, Any]] = {}
    for line_number, row in read_jsonl(path):
        question_id = row.get("question_id")
        if not isinstance(question_id, int) or isinstance(question_id, bool):
            raise ValueError(f"{path}:{line_number}: question_id must be an integer")
        images = row.get("image")
        data_type = row.get("data_type")
        if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], str):
            continue
        key = (question_id, images[0], data_type)
        if key in result:
            raise ValueError(
                f"{path}:{line_number}: duplicate composite key {key!r}"
            )
        result[key] = row
    return result


def answer_relation_v3(row: dict[str, Any]) -> str | None:
    answer = row.get("answer")
    for index, option in enumerate(row.get("options") or []):
        if chr(ord("A") + index) == answer:
            text = option.split(".", maxsplit=1)[1].strip().lower() if "." in option else ""
            try:
                from regcfpo.data_v3 import normalize_relation_v3

                return normalize_relation_v3(text)
            except ValueError:
                return None
    return None


def center_x(box: list[float]) -> float:
    return (float(box[0]) + float(box[2])) / 2.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spatial", required=True, type=Path)
    parser.add_argument("--grounding", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit-json", required=True, type=Path)
    args = parser.parse_args()

    spatial = index_by_composite_key(args.spatial)
    grounding = index_by_composite_key(args.grounding)
    counts: Counter = Counter()
    candidates: list[RelationPairV3] = []

    for key in sorted(spatial, key=lambda k: k[0]):
        row = spatial[key]
        counts["spatial_total"] += 1
        if row.get("data_type") != "single_image":
            counts["reject_not_single_image"] += 1
            continue
        if row.get("question_type") != "relative direction":
            counts["reject_not_relative_direction"] += 1
            continue
        counts["single_image_relative_direction"] += 1

        relation = answer_relation_v3(row)
        if relation is None:
            counts["reject_answer_not_six_relations"] += 1
            continue
        counts[f"answer_{relation.replace('-', '_')}"] += 1
        relation_class = "axis" if relation in AXIS_RELATIONS else "diagonal"

        grounding_row = grounding.get(key)
        if grounding_row is None:
            counts["reject_missing_grounding_join"] += 1
            continue
        if grounding_row.get("question") != row.get("question"):
            counts["reject_grounding_question_mismatch"] += 1
            continue
        if grounding_row.get("image") != row.get("image"):
            counts["reject_grounding_image_mismatch"] += 1
            continue

        try:
            relation_letters = resolve_option_relations_v3(row.get("options") or [])
        except OptionRelationV3Error:
            counts["reject_nonunique_option_relation_mapping"] += 1
            continue
        mapped_relation = swap_mapped_relation_v3(relation)
        if relation not in relation_letters.values():
            counts["reject_answer_letter_not_mapped"] += 1
            continue
        if mapped_relation not in relation_letters.values():
            counts["reject_missing_mapped_option"] += 1
            continue
        counts["unique_mapped_answer"] += 1
        answer_letter = row["answer"]
        mapped_letter = next(
            letter for letter, rel in relation_letters.items() if rel == mapped_relation
        )

        entities = ordered_entities(row["question"], grounding_row.get("answer"))
        if entities is None:
            counts["reject_entity_or_box_parse"] += 1
            continue
        counts["unique_boundary_entity_spans"] += 1
        entity_a, entity_b = entities
        box_a = entity_a["bbox_2d"]
        box_b = entity_b["bbox_2d"]
        center_a = center_x(box_a)
        center_b = center_x(box_b)
        if center_a == center_b:
            counts["reject_equal_x_centers"] += 1
            continue
        box_horizontal = "left" if center_a < center_b else "right"
        operator_valid = box_horizontal == horizontal_component(relation)
        if not operator_valid:
            counts["reject_box_answer_inconsistent"] += 1
            continue
        if relation in DIAGONAL_RELATIONS:
            counts["diagonal_depth_component_unverifiable_from_boxes"] += 1

        image = row["image"][0]
        pair = RelationPairV3(
            sample_id=f"spld-{key[0]:05d}",
            scene_id=spatialladder_base_scene_id(image),
            image=image,
            question=row["question"],
            options=tuple(row["options"]),
            answer_relation=relation,
            mapped_relation=mapped_relation,
            answer_letter=answer_letter,
            mapped_answer_letter=mapped_letter,
            entity_a=entity_a["label"],
            entity_b=entity_b["label"],
            gt_box_a=tuple(float(v) for v in box_a),
            gt_box_b=tuple(float(v) for v in box_b),
            relation_class=relation_class,
            operator_valid=True,
        )
        candidates.append(pair)
        counts["accepted"] += 1
        counts[f"accepted_{relation_class}"] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for pair in candidates:
            handle.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")

    scenes = sorted({pair.scene_id for pair in candidates})
    axis_scenes = {p.scene_id for p in candidates if p.relation_class == "axis"}
    diagonal_scenes = {p.scene_id for p in candidates if p.relation_class == "diagonal"}
    audit = {
        "schema_version": 1,
        "counts": dict(sorted(counts.items())),
        "accepted_rows": len(candidates),
        "accepted_scenes": len(scenes),
        "axis_rows": sum(1 for p in candidates if p.relation_class == "axis"),
        "diagonal_rows": sum(1 for p in candidates if p.relation_class == "diagonal"),
        "axis_scenes": len(axis_scenes),
        "diagonal_scenes": len(diagonal_scenes),
        "relation_counts": dict(
            sorted(Counter(p.answer_relation for p in candidates).items())
        ),
    }
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
