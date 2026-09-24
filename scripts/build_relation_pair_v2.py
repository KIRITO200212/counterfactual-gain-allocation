#!/usr/bin/env python3
"""Join released SpatialLadder annotations into strict RelationPair-v2 rows."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from regcfpo.data import (
    OptionRelationMappingError,
    RelationPairV2,
    inverse_relation,
    relation_to_option_letters,
    spatialladder_base_scene_id,
)


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


def index_by_question_id(path: Path) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for line_number, row in read_jsonl(path):
        question_id = row.get("question_id")
        if not isinstance(question_id, int) or isinstance(question_id, bool):
            raise ValueError(f"{path}:{line_number}: question_id must be an integer")
        if question_id in result:
            raise ValueError(f"{path}:{line_number}: duplicate question_id={question_id}")
        result[question_id] = row
    return result


def option_text(option: str) -> str:
    return option.split(".", maxsplit=1)[1].strip().casefold() if "." in option else ""


def answer_relation(row: dict[str, Any]) -> str | None:
    answer = row.get("answer")
    for index, option in enumerate(row.get("options") or []):
        if chr(ord("A") + index) == answer:
            value = option_text(option)
            return value if value in {"left", "right"} else None
    return None


def ordered_entities(
    question: str, grounding_answer: Any
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    if not isinstance(grounding_answer, list) or len(grounding_answer) != 2:
        return None
    spans: list[tuple[int, int]] = []
    for object_record in grounding_answer:
        if not isinstance(object_record, dict):
            return None
        label = object_record.get("label")
        box = object_record.get("bbox_2d")
        if not isinstance(label, str) or not label.strip():
            return None
        if not isinstance(box, list) or len(box) != 4:
            return None
        matches = list(
            re.finditer(
                rf"(?<!\w){re.escape(label.strip())}(?!\w)",
                question,
                flags=re.IGNORECASE,
            )
        )
        if len(matches) != 1:
            return None
        spans.append(matches[0].span())
    if spans[0][0] < spans[1][1] and spans[1][0] < spans[0][1]:
        return None
    return (
        (grounding_answer[0], grounding_answer[1])
        if spans[0][0] < spans[1][0]
        else (grounding_answer[1], grounding_answer[0])
    )


def center_x(box: list[float]) -> float:
    return (float(box[0]) + float(box[2])) / 2.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spatial", required=True, type=Path)
    parser.add_argument("--grounding", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit-json", required=True, type=Path)
    args = parser.parse_args()

    spatial = index_by_question_id(args.spatial)
    grounding = index_by_question_id(args.grounding)
    counts: Counter[str] = Counter()
    scene_counts: Counter[str] = Counter()
    candidates: list[RelationPairV2] = []

    for question_id in sorted(spatial):
        row = spatial[question_id]
        counts["spatial_total"] += 1
        if row.get("data_type") != "single_image":
            counts["reject_not_single_image"] += 1
            continue
        counts["single_image_total"] += 1
        if row.get("question_type") != "relative direction":
            counts["reject_not_relative_direction"] += 1
            continue
        counts["single_image_relative_direction"] += 1

        relation = answer_relation(row)
        if relation is None:
            counts["reject_answer_not_exact_left_right"] += 1
            continue
        counts["exact_left_right_answer"] += 1

        grounding_row = grounding.get(question_id)
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
            relation_letters = relation_to_option_letters(row.get("options") or [])
        except OptionRelationMappingError:
            counts["reject_nonunique_option_relation_mapping"] += 1
            continue
        counts["unique_option_relation_mapping"] += 1
        mapped_relation = inverse_relation(relation)
        if relation not in relation_letters or mapped_relation not in relation_letters:
            counts["reject_missing_inverse_option"] += 1
            continue
        counts["unique_mapped_answer"] += 1

        entities = ordered_entities(row["question"], grounding_row.get("answer"))
        if entities is None:
            counts["reject_entity_or_box_parse"] += 1
            continue
        counts["unique_boundary_entity_spans"] += 1
        entity_a, entity_b = entities
        box_a = entity_a["bbox_2d"]
        box_b = entity_b["bbox_2d"]
        box_relation = "left" if center_x(box_a) < center_x(box_b) else "right"
        operator_valid = box_relation == relation
        if not operator_valid:
            counts["reject_box_answer_inconsistent"] += 1
            continue

        images = row.get("image")
        if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], str):
            counts["reject_image_schema"] += 1
            continue
        image = images[0]
        scene_id = spatialladder_base_scene_id(image)
        pair = RelationPairV2(
            sample_id=f"spld-{question_id:05d}",
            scene_id=scene_id,
            image=image,
            question=row["question"],
            options=tuple(row["options"]),
            answer_relation=relation,
            mapped_relation=mapped_relation,
            answer_letter=row["answer"],
            mapped_answer_letter=relation_letters[mapped_relation],
            entity_a=entity_a["label"],
            entity_b=entity_b["label"],
            gt_box_a=tuple(box_a),
            gt_box_b=tuple(box_b),
            operator_valid=True,
        )
        candidates.append(pair)
        scene_counts[scene_id] += 1
        counts["accepted"] += 1
        counts[f"accepted_{relation}"] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for pair in candidates:
            handle.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")

    audit = {
        "schema_version": 1,
        "source": {
            "spatial": str(args.spatial),
            "grounding": str(args.grounding),
        },
        "counts": dict(sorted(counts.items())),
        "accepted_scenes": len(scene_counts),
        "min_examples_per_scene": min(scene_counts.values(), default=0),
        "max_examples_per_scene": max(scene_counts.values(), default=0),
        "option_relation_parse_rate": (
            counts["unique_option_relation_mapping"] / counts["exact_left_right_answer"]
            if counts["exact_left_right_answer"]
            else 0.0
        ),
        "mapped_answer_unique_rate": (
            counts["unique_mapped_answer"] / counts["exact_left_right_answer"]
            if counts["exact_left_right_answer"]
            else 0.0
        ),
        "unique_boundary_entity_span_rate": (
            counts["unique_boundary_entity_spans"] / counts["exact_left_right_answer"]
            if counts["exact_left_right_answer"]
            else 0.0
        ),
        "gates": {
            "gt_box_diagnostic_min_500": len(candidates) >= 500,
            "potential_train_min_1000": len(candidates) >= 1000,
            "scene_disjoint_split_status": "NOT_YET_FROZEN",
        },
    }
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
