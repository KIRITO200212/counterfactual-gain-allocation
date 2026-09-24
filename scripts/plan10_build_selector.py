#!/usr/bin/env python3
"""Build the frozen 24-parent / 40-update Plan10 exploratory selector."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import stdev
from typing import Any, Mapping, Sequence


STRATA = ("axis", "LF_RB", "LB_RF")
TARGET_VISITS = {"axis": 14, "LF_RB": 13, "LB_RF": 13}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(value)
    if not rows:
        raise ValueError(f"{path}: empty JSONL")
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def candidate_record(
    row: Mapping[str, Any],
    *,
    required_prefixes: int,
    min_boundary_probability: float,
    min_credit_std: float,
) -> dict[str, Any]:
    prefixes = [prefix for prefix in row["prefixes"] if prefix["valid"]]
    mapped = str(row["mapped_letter"])
    probabilities = [
        float(prefix["option_probabilities"][mapped]) for prefix in prefixes
    ]
    credits = [float(prefix["credit_t2"]) for prefix in prefixes]
    suffix_count = int(row["suffix_mapped_count"])
    all_wrong = bool(row["all_wrong_full_rollouts"])
    fully_valid = len(prefixes) == required_prefixes
    max_probability = max(probabilities) if probabilities else 0.0
    credit_std = stdev(credits) if len(credits) >= 2 else 0.0
    reachable = suffix_count > 0 or max_probability >= min_boundary_probability
    return {
        "parent_sample_id": str(row["parent_sample_id"]),
        "swap_sample_id": str(row["sample_id"]),
        "scene_id": str(row["scene_id"]),
        "stratum": str(row["stratum"]),
        "all_wrong": all_wrong,
        "fully_valid": fully_valid,
        "reachable": reachable,
        "suffix_mapped_count": suffix_count,
        "max_boundary_mapped_probability": max_probability,
        "credit_t2_std": credit_std,
        "eligible": all_wrong and fully_valid and reachable and credit_std >= min_credit_std,
    }


def select_candidates(
    candidates: Sequence[Mapping[str, Any]], *, per_stratum: int
) -> list[dict[str, Any]]:
    ranked_by_stratum: dict[str, list[dict[str, Any]]] = {}
    for stratum in STRATA:
        ranked = sorted(
            (
                dict(candidate)
                for candidate in candidates
                if candidate["stratum"] == stratum and candidate["eligible"]
            ),
            key=lambda candidate: (
                -int(candidate["suffix_mapped_count"] > 0),
                -int(candidate["suffix_mapped_count"]),
                -float(candidate["max_boundary_mapped_probability"]),
                -float(candidate["credit_t2_std"]),
                str(candidate["parent_sample_id"]),
            ),
        )
        # A scene may contribute several directional rows.  Retain its best
        # ranked parent before the cross-stratum unique-scene assignment.
        scene_unique: list[dict[str, Any]] = []
        seen_scenes: set[str] = set()
        for candidate in ranked:
            scene = str(candidate["scene_id"])
            if scene in seen_scenes:
                continue
            seen_scenes.add(scene)
            scene_unique.append(candidate)
        if len(scene_unique) < per_stratum:
            raise ValueError(
                f"stratum {stratum} has only {len(scene_unique)} eligible unique-scene "
                f"parents; required {per_stratum}"
            )
        ranked_by_stratum[stratum] = scene_unique

    # Find the lexicographically first feasible combination in frozen rank
    # order.  This avoids a greedy early stratum consuming a scene required by
    # a later stratum while preserving the preregistered ranking priority.
    def search(index: int, used_scenes: frozenset[str]) -> list[dict[str, Any]] | None:
        if index == len(STRATA):
            return []
        stratum = STRATA[index]
        available = [
            candidate
            for candidate in ranked_by_stratum[stratum]
            if str(candidate["scene_id"]) not in used_scenes
        ]
        for chosen in itertools.combinations(available, per_stratum):
            scenes = {str(candidate["scene_id"]) for candidate in chosen}
            if len(scenes) != per_stratum:
                continue
            remainder = search(index + 1, used_scenes | frozenset(scenes))
            if remainder is not None:
                return [dict(candidate) for candidate in chosen] + remainder
        return None

    selected = search(0, frozenset())
    if selected is None:
        counts = {stratum: len(ranked_by_stratum[stratum]) for stratum in STRATA}
        raise ValueError(
            f"no cross-stratum unique-scene assignment exists for {per_stratum} "
            f"parents per stratum; eligible scene counts={counts}"
        )
    return selected


def build_schedule(
    selected: Sequence[Mapping[str, Any]],
    joint_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    by_parent: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in joint_rows:
        parent = str(row["pairaug_parent_sample_id"])
        branch = str(row["pairaug_branch"])
        if branch in by_parent[parent]:
            raise ValueError(f"duplicate {parent}/{branch} row in candidate manifest")
        by_parent[parent][branch] = row
    selected_by_stratum = {
        stratum: [str(row["parent_sample_id"]) for row in selected if row["stratum"] == stratum]
        for stratum in STRATA
    }
    full_cycle = [str(row["parent_sample_id"]) for row in selected]
    rng = random.Random(seed)
    rng.shuffle(full_cycle)
    repeat_cycle: list[str] = []
    for offset, stratum in enumerate(STRATA):
        values = list(selected_by_stratum[stratum])
        random.Random(seed + 10_003 * (offset + 1)).shuffle(values)
        repeat_count = TARGET_VISITS[stratum] - len(values)
        if repeat_count < 0 or repeat_count > len(values):
            raise ValueError(
                f"cannot schedule target {TARGET_VISITS[stratum]} visits from "
                f"{len(values)} unique {stratum} parents"
            )
        repeat_cycle.extend(values[:repeat_count])
    random.Random(seed + 1_000_003).shuffle(repeat_cycle)

    output: list[dict[str, Any]] = []
    for cycle, parents in enumerate((full_cycle, repeat_cycle)):
        cycle_rng = random.Random(seed + 1_000_003 * cycle)
        for pair_index, parent in enumerate(parents):
            branches = by_parent.get(parent, {})
            if set(branches) != {"source", "swap"}:
                raise ValueError(f"candidate manifest lacks one source/swap pair for {parent}")
            branch_order = ["source", "swap"]
            if cycle_rng.random() < 0.5:
                branch_order.reverse()
            for branch in branch_order:
                row = dict(branches[branch])
                row.update(
                    {
                        "schema_version": "Plan10PrefixBridgeSelector-v1",
                        "sample_id": f"{parent}::p10c{cycle}::{branch}",
                        "pairaug_cycle": cycle,
                        "pairaug_pair_index": pair_index,
                    }
                )
                output.append(row)
    for position, row in enumerate(output):
        row["cycle_position"] = position
    if len(output) != 80:
        raise RuntimeError(f"expected 80 rows / 40 updates, got {len(output)}")
    if len({str(row["sample_id"]) for row in output}) != len(output):
        raise RuntimeError("selector schedule contains duplicate sample ids")
    for position in range(0, len(output), 2):
        pair = output[position : position + 2]
        if (
            len({str(row["pairaug_parent_sample_id"]) for row in pair}) != 1
            or len({int(row["pairaug_cycle"]) for row in pair}) != 1
            or {str(row["pairaug_branch"]) for row in pair} != {"source", "swap"}
        ):
            raise RuntimeError(f"invalid adjacent pair at schedule rows {position}:{position + 2}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-report", action="append", type=Path, required=True)
    parser.add_argument(
        "--candidate-joint-manifest", action="append", type=Path, required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-stratum", type=int, default=8)
    parser.add_argument("--min-boundary-mapped-probability", type=float, default=0.01)
    parser.add_argument("--min-credit-std", type=float, default=1e-4)
    parser.add_argument("--feasibility-amendment", type=Path)
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()
    audit_path = args.output.with_suffix(".audit.json")
    if args.output.exists() or audit_path.exists():
        raise SystemExit("refusing to overwrite an existing selector or audit")
    if args.per_stratum not in (7, 8):
        parser.error("Plan10 R1/R1a permits only 8 or 7 parents per stratum")
    amendment_contract: dict[str, Any] | None = None
    if args.per_stratum == 7:
        if args.feasibility_amendment is None:
            parser.error("7-per-stratum selector requires --feasibility-amendment")
        amendment = json.loads(args.feasibility_amendment.read_text(encoding="utf-8"))
        if amendment.get("amendment") != "plan10_r1a_selector_feasibility":
            parser.error("unexpected feasibility amendment identity")
        amendment_contract = {
            "path": str(args.feasibility_amendment),
            "sha256": sha256_file(args.feasibility_amendment),
        }
    elif args.feasibility_amendment is not None:
        parser.error("feasibility amendment is only valid with --per-stratum 7")
    if not 0.0 < args.min_boundary_mapped_probability < 1.0:
        parser.error("minimum boundary probability must lie in (0,1)")
    if args.min_credit_std <= 0.0:
        parser.error("minimum credit std must be positive")

    calibration_rows: list[dict[str, Any]] = []
    report_contracts: list[dict[str, Any]] = []
    seen_samples: set[str] = set()
    required_prefixes: int | None = None
    for path in args.calibration_report:
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("diagnostic") != "plan10_prefix_value_calibration":
            raise ValueError(f"unexpected diagnostic in {path}")
        prefix_seeds = list(report["prefix_seeds"])
        if required_prefixes is None:
            required_prefixes = len(prefix_seeds)
        elif required_prefixes != len(prefix_seeds):
            raise ValueError("calibration shards use different prefix multiplicities")
        for row in report["rows"]:
            sample_id = str(row["sample_id"])
            if sample_id in seen_samples:
                raise ValueError(f"duplicate calibration sample {sample_id}")
            seen_samples.add(sample_id)
            calibration_rows.append(dict(row))
        report_contracts.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "shard_index": report.get("shard_index"),
                "num_shards": report.get("num_shards"),
                "prefix_seeds": prefix_seeds,
                "suffix_samples_per_prefix": report["suffix_samples_per_prefix"],
            }
        )
    if required_prefixes is None:
        raise ValueError("no calibration rows loaded")

    candidates = [
        candidate_record(
            row,
            required_prefixes=required_prefixes,
            min_boundary_probability=args.min_boundary_mapped_probability,
            min_credit_std=args.min_credit_std,
        )
        for row in calibration_rows
    ]
    selected = select_candidates(candidates, per_stratum=args.per_stratum)
    joint_rows: list[dict[str, Any]] = []
    joint_keys: dict[tuple[str, str], dict[str, Any]] = {}
    joint_contracts: list[dict[str, Any]] = []
    duplicate_joint_rows = 0
    identity_fields = (
        "pairaug_original_letter",
        "pairaug_mapped_letter",
        "joint_s_swap_0",
        "joint_source_logps_0",
        "joint_null_logps_0",
        "question",
        "options",
    )
    for path in args.candidate_joint_manifest:
        rows_from_path = read_jsonl(path)
        for row in rows_from_path:
            key = (str(row["pairaug_parent_sample_id"]), str(row["pairaug_branch"]))
            existing = joint_keys.get(key)
            if existing is not None:
                if any(existing[field] != row[field] for field in identity_fields):
                    raise ValueError(f"conflicting duplicate joint row {key} across manifests")
                duplicate_joint_rows += 1
                continue
            copied = dict(row)
            joint_keys[key] = copied
            joint_rows.append(copied)
        joint_contracts.append(
            {"path": str(path), "sha256": sha256_file(path), "rows": len(rows_from_path)}
        )
    schedule = build_schedule(selected, joint_rows, seed=args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        for row in schedule:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    selected_ids = {str(row["parent_sample_id"]) for row in selected}
    audit = {
        "schema_version": 1,
        "artifact": "plan10_prefix_bridge_selector",
        "status": "EXPLORATORY_NOT_CONFIRMATORY",
        "selection_rule_frozen_before_candidate_scan": {
            "all_wrong_g4": True,
            "all_prefix_boundaries_valid": True,
            "minimum_credit_t2_std": args.min_credit_std,
            "reachability": (
                "at least one fixed-prefix suffix mapped hit OR max normalized "
                f"A-D mapped probability >= {args.min_boundary_mapped_probability}"
            ),
            "ranking": (
                "suffix-hit presence/count, max boundary mapped probability, "
                "credit std, stable parent id"
            ),
        },
        "feasibility_amendment": amendment_contract,
        "calibration_reports": report_contracts,
        "candidate_joint_manifests": {
            "inputs": joint_contracts,
            "duplicate_rows_deduplicated": duplicate_joint_rows,
            "merged_rows": len(joint_rows),
        },
        "candidate_counts": {
            "total": len(candidates),
            "all_wrong": sum(bool(row["all_wrong"]) for row in candidates),
            "fully_valid": sum(bool(row["fully_valid"]) for row in candidates),
            "reachable": sum(bool(row["reachable"]) for row in candidates),
            "eligible": sum(bool(row["eligible"]) for row in candidates),
            "eligible_by_stratum": dict(
                Counter(str(row["stratum"]) for row in candidates if row["eligible"])
            ),
        },
        "selected_parents": selected,
        "selected_parent_count": len(selected_ids),
        "selected_scene_count": len({str(row["scene_id"]) for row in selected}),
        "selected_by_stratum": dict(Counter(str(row["stratum"]) for row in selected)),
        "schedule": {
            "seed": args.seed,
            "updates": len(schedule) // 2,
            "rows": len(schedule),
            "unique_parents": len(selected_ids),
            "parent_visits_by_stratum": dict(
                Counter(
                    next(
                        str(candidate["stratum"])
                        for candidate in selected
                        if candidate["parent_sample_id"] == row["pairaug_parent_sample_id"]
                    )
                    for row in schedule[::2]
                )
            ),
            "adjacent_pairs": True,
        },
        "output": {"path": str(args.output), "sha256": sha256_file(args.output)},
        "code_sha256": sha256_file(Path(__file__)),
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
