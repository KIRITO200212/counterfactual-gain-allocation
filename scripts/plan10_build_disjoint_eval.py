#!/usr/bin/env python3
"""Build a deterministic scene-disjoint 12-row Plan10 behavior cohort."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


STRATA = ("axis", "LF_RB", "LB_RF")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path}: expected non-empty JSON objects")
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def relation_stratum(relation: str) -> str:
    if relation in {"left", "right"}:
        return "axis"
    if relation in {"left-front", "right-back"}:
        return "LF_RB"
    if relation in {"left-back", "right-front"}:
        return "LB_RF"
    raise ValueError(f"unsupported relation {relation!r}")


def select_disjoint_rows(
    candidates: Sequence[Mapping[str, Any]],
    *,
    excluded_scenes: set[str],
    per_stratum: int,
    seed: int,
) -> list[dict[str, Any]]:
    pools: dict[str, list[dict[str, Any]]] = {}
    for offset, stratum in enumerate(STRATA):
        pool = [
            dict(row)
            for row in candidates
            if row.get("operator_valid") is True
            and str(row["scene_id"]) not in excluded_scenes
            and relation_stratum(str(row["answer_relation"])) == stratum
        ]
        random.Random(seed + 10_003 * offset).shuffle(pool)
        # Keep the first deterministic row per scene within each stratum.
        unique: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in pool:
            scene = str(row["scene_id"])
            if scene not in seen:
                seen.add(scene)
                unique.append(row)
        if len(unique) < per_stratum:
            raise ValueError(f"{stratum} has only {len(unique)} eligible scenes")
        pools[stratum] = unique

    def search(index: int, used: frozenset[str]) -> list[dict[str, Any]] | None:
        if index == len(STRATA):
            return []
        stratum = STRATA[index]
        available = [row for row in pools[stratum] if str(row["scene_id"]) not in used]
        for chosen in itertools.combinations(available, per_stratum):
            scenes = frozenset(str(row["scene_id"]) for row in chosen)
            remainder = search(index + 1, used | scenes)
            if remainder is not None:
                return [dict(row) for row in chosen] + remainder
        return None

    selected = search(0, frozenset())
    if selected is None:
        raise ValueError("no cross-stratum scene-disjoint evaluation assignment exists")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--train-selector", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-stratum", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()
    audit_path = args.output.with_suffix(".audit.json")
    if args.output.exists() or audit_path.exists():
        raise SystemExit("refusing to overwrite evaluation manifest or audit")
    if args.per_stratum != 4:
        parser.error("Plan10 disjoint G=4 cohort freezes four rows per stratum")

    train_rows = read_jsonl(args.train_selector)
    train_scenes = {str(row["scene_id"]) for row in train_rows}
    train_parents = {str(row["pairaug_parent_sample_id"]) for row in train_rows}
    source_rows = read_jsonl(args.source_manifest)
    selected = select_disjoint_rows(
        source_rows,
        excluded_scenes=train_scenes,
        per_stratum=args.per_stratum,
        seed=args.seed,
    )
    selected_scenes = {str(row["scene_id"]) for row in selected}
    selected_ids = {str(row["sample_id"]) for row in selected}
    if train_scenes & selected_scenes or train_parents & selected_ids:
        raise RuntimeError("evaluation cohort overlaps selector scenes or parents")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    audit = {
        "schema_version": 1,
        "artifact": "plan10_r1a_scene_disjoint_behavior12",
        "status": "EXPLORATORY_SMALL_COHORT_NOT_CONFIRMATORY",
        "frozen_before_training_results_visible": True,
        "seed": args.seed,
        "selection_uses_model_outputs": False,
        "rows": len(selected),
        "unique_scenes": len(selected_scenes),
        "strata": dict(Counter(relation_stratum(str(row["answer_relation"])) for row in selected)),
        "train_scene_overlap": len(train_scenes & selected_scenes),
        "train_parent_overlap": len(train_parents & selected_ids),
        "source": {"path": str(args.source_manifest), "sha256": sha256_file(args.source_manifest)},
        "train_selector": {"path": str(args.train_selector), "sha256": sha256_file(args.train_selector)},
        "output": {"path": str(args.output), "sha256": sha256_file(args.output)},
        "ordered_sample_ids": [str(row["sample_id"]) for row in selected],
        "code_sha256": sha256_file(Path(__file__)),
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
