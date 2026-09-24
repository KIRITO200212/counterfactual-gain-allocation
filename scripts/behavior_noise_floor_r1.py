#!/usr/bin/env python3
"""G=16 rollout driver for Gate R1 (behavior noise floor), seed sets 2/3.

Implements the rollout workload of ``gates.gate_r1_behavior_noise_floor`` from
amendment ``asymmetric_directional_credit_revision_20260826``
(configs/phase1_prereg.yaml): base model, same 4090, same eval code, all 174
directional train rows, swap/null branches at G=16, one completion per fixed
seed.  Gate R1 holds NO route-fork authority; it only provides the comparison
scale for the behavior gate.

Code-path identity with LOSS_ALIGNED_PROXY_GATE_V3
--------------------------------------------------
The generation stream is *literally* the V3 code: this module imports
``apply_edit``, ``generate_rollouts``, ``shard_records`` and
``validate_resume_prefix`` from ``scripts/answer_proxy_gate_v3.py`` (a frozen,
hash-bound file; its sha256 is recorded in the run contract) and reuses the
same imported building blocks V3 uses (``official_prompt``,
``build_official_generation_config``, ``build_chat_prompt``,
``clean_stage3_mca_text``, ``stage3_format_valid``, the frozen pixel
operators).  The official CoT prompt, the official GenerationConfig
(temperature 1.0, top_p 1.0, top_k 50, repetition_penalty 1.0, do_sample,
max_new_tokens 1024) and one-completion-per-seed semantics are unchanged.

Deliberate deviations from V3 (documented, not silent):

- the factual branch is skipped (Gate R1 needs only swap/null rollouts);
- teacher-forced arms are NOT rescored: TF scores are seed-invariant and the
  TF side of the TF-rollout correlation is read from the frozen set-1
  artifact ``runs/answer_proxy_gate_v3/answer_proxy_v3.jsonl``
  (full_template_span) by ``analyze_behavior_noise_floor_r1.py``;
- the rollout seed list is parameterized (V3 hardcodes
  ``ROLLOUT_SEEDS = 1701..1716``).  ``generate_rollouts`` already takes the
  seed list as an argument, so only the caller changes.

Contract honesty: V3's contract writer would record the frozen 1701-1716 list,
so this script writes its OWN run contract via the same
``ensure_run_contract`` mechanism, recording the *actual* CLI-supplied seed
list (e.g. 1801-1816 for set2, 1901-1916 for set3).  The contract and the
per-row ``per_rollout[*].seed`` fields therefore always reflect the seeds that
were really generated.

Output hygiene is identical to V3: rows stream to ``*.jsonl.tmp`` and are
atomically renamed on completion; a pre-existing complete output is never
overwritten; a tmp file resumes only when its completed sample_ids are an
ordered prefix of the contract ``ordered_sample_ids``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from PIL import Image
from transformers import AutoProcessor, GenerationConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from analyze_answer_proxy_v3 import relation_family  # noqa: E402
from answer_proxy_gate import build_chat_prompt, option_letter, sha256_file  # noqa: E402
from answer_proxy_gate_v3 import (  # noqa: E402
    CODE_IDENTITY_PATHS,
    DEFAULT_IMAGES_ROOT,
    DEFAULT_MANIFEST,
    DEFAULT_MODEL,
    EXPECTED_SATURATION_GENERATION,
    NEWLY_GENERATED_BRANCHES,
    apply_edit,
    generate_rollouts,
    shard_records,
    validate_resume_prefix,
)
from frozen_operator_diagnostic import read_jsonl  # noqa: E402
from regcfpo.qwen_compat import (  # noqa: E402
    load_qwen_config_with_compat,
    validate_qwen_weight_tying,
)
from regcfpo.run_contract import ensure_run_contract  # noqa: E402
from reward_saturation_audit import (  # noqa: E402
    build_official_generation_config,
    hash_required_model_identity,
    official_prompt,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (  # noqa: E402
    Qwen2_5_VLForConditionalGeneration,
)

DIAGNOSTIC = "behavior_noise_floor_r1"
AMENDMENT = "asymmetric_directional_credit_revision_20260826"
DEFAULT_OUTPUT_ROOT = Path("runs/behavior_noise_floor_r1")
DEFAULT_NUM_SEEDS = 16
OUTPUT_NAME = "behavior_noise_floor_r1.jsonl"

# Seed sets frozen by the amendment: set1=1701-1716 (already produced by the
# V3 gate), set2=1801-1816, set3=1901-1916.
FROZEN_SEED_SETS = {
    "set2": list(range(1801, 1817)),
    "set3": list(range(1901, 1917)),
}


# ---------------------------------------------------------------------------
# pure helpers (unit-tested without a GPU)


def parse_seed_list(
    *,
    seeds: str | None = None,
    seed_start: int | None = None,
    num_seeds: int = DEFAULT_NUM_SEEDS,
) -> list[int]:
    """Parse the CLI seed specification into an explicit seed list.

    Exactly one of ``seeds`` / ``seed_start`` must be given.  ``seeds`` accepts
    either an inclusive range ("1801-1816") or a comma list
    ("1801,1802,...").  ``seed_start`` expands to ``num_seeds`` consecutive
    seeds.  The result must contain exactly ``num_seeds`` unique positive
    integers (G=16 is frozen for Gate R1).
    """

    if num_seeds < 1:
        raise ValueError("num_seeds must be positive")
    if (seeds is None) == (seed_start is None):
        raise ValueError("exactly one of seeds / seed_start must be provided")
    parsed: list[int]
    if seed_start is not None:
        if seed_start < 1:
            raise ValueError("seed_start must be a positive integer")
        parsed = list(range(seed_start, seed_start + num_seeds))
    elif "-" in seeds:
        parts = seeds.split("-")
        if len(parts) != 2 or not all(part.isdigit() for part in parts):
            raise ValueError(f"invalid seed range {seeds!r}; expected e.g. 1801-1816")
        start, end = int(parts[0]), int(parts[1])
        if end < start:
            raise ValueError(f"invalid seed range {seeds!r}: end < start")
        parsed = list(range(start, end + 1))
    else:
        try:
            parsed = [int(token) for token in seeds.split(",") if token.strip()]
        except ValueError as exc:
            raise ValueError(f"invalid seed list {seeds!r}") from exc
    if len(parsed) != num_seeds:
        raise ValueError(
            f"seed specification yields {len(parsed)} seeds; G={num_seeds} is frozen"
        )
    if len(set(parsed)) != len(parsed) or any(seed < 1 for seed in parsed):
        raise ValueError(f"seeds must be unique positive integers: {parsed!r}")
    return parsed


def run_contract_payload(
    *,
    seeds: Sequence[int],
    model: Path,
    model_identity: Mapping[str, str],
    manifest: Path,
    manifest_sha256: str,
    ordered_sample_ids: Sequence[str],
    shard_index: int,
    num_shards: int,
    limit: int | None,
    device: str,
    attention: str,
    min_pixels: int,
    max_pixels: int,
    code_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Gate R1 run contract; records the ACTUAL seed list (never V3's)."""

    return {
        "schema_version": 1,
        "diagnostic": DIAGNOSTIC,
        "amendment": AMENDMENT,
        "gate": "gate_r1_behavior_noise_floor",
        "gate_role": (
            "comparison scale for the behavior gate only; holds NO "
            "route-fork authority (forks are decided on 2026-09-01)"
        ),
        "relation_to_v3": (
            "extends the V3 G=16 workload (seed set 1701-1716) with "
            "additional fixed seed sets; identical prompt, generation "
            "config, operators and per-row schema; factual branch and "
            "teacher-forced arms are not regenerated (TF scores are "
            "seed-invariant and read from the frozen set-1 artifact)"
        ),
        "branches": list(NEWLY_GENERATED_BRANCHES),
        "arms_regenerated": [],
        "rollout": {
            "num_generations": len(seeds),
            "seeds": list(seeds),
            "one_completion_per_seed": True,
            "prompt": "official Stage-3 CoT template (reward_saturation_audit.official_prompt)",
            "generation_config": dict(EXPECTED_SATURATION_GENERATION),
        },
        "environment_isolation": "PYTHONNOUSERSITE=1",
        "model_path": str(model),
        "model_config_sha256": model_identity["config.json"],
        "model_identity_sha256": dict(model_identity),
        "manifest_path": str(manifest),
        "manifest_sha256": manifest_sha256,
        "ordered_sample_ids": list(ordered_sample_ids),
        "shard": {"index": shard_index, "num_shards": num_shards},
        "limit": limit,
        "device": device,
        "attention": attention,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "torch_version": torch.__version__,
        "code_sha256": dict(code_hashes),
    }


# ---------------------------------------------------------------------------
# GPU-bound generation (V3 code path)


def process_record_rollouts(
    *,
    model,
    processor,
    record: Mapping[str, Any],
    images_root: Path,
    letters: Sequence[str],
    generation_config: GenerationConfig,
    seeds: Sequence[int],
    device: str,
) -> dict[str, Any]:
    """Swap/null G=16 rollouts for one row; per-row schema mirrors V3."""

    image_path = images_root / record["image"]
    row_out: dict[str, Any] = {
        "sample_id": record["sample_id"],
        "scene_id": record["scene_id"],
        "relation_class": record["relation_class"],
        "relation_family": relation_family(
            record["answer_relation"], record["mapped_relation"]
        ),
        "answer_letter": record["answer_letter"],
        "mapped_answer_letter": record.get(
            "mapped_answer_letter", record["answer_letter"]
        ),
        "branches": {},
    }
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        for branch in NEWLY_GENERATED_BRANCHES:
            edited, status = apply_edit(branch, image, record)
            if edited is None:
                row_out["branches"][branch] = {
                    "accepted": False,
                    "reject_reason": status,
                }
                continue
            template_prompt = build_chat_prompt(
                processor, image_path, official_prompt(dict(record))
            )
            row_out["branches"][branch] = {
                "accepted": True,
                "rollout": generate_rollouts(
                    model=model,
                    processor=processor,
                    image=edited,
                    prompt=template_prompt,
                    letters=letters,
                    generation_config=generation_config,
                    seeds=seeds,
                    device=device,
                ),
            }
            if edited is not image:
                edited.close()
    return row_out


# ---------------------------------------------------------------------------
# main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--images-root", type=Path, default=DEFAULT_IMAGES_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    seed_group = parser.add_mutually_exclusive_group(required=True)
    seed_group.add_argument(
        "--seeds",
        help='inclusive range ("1801-1816") or comma list ("1801,1802,...")',
    )
    seed_group.add_argument("--seed-start", type=int, help="first of 16 consecutive seeds")
    parser.add_argument("--num-seeds", type=int, default=DEFAULT_NUM_SEEDS)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attention", choices=["sdpa", "eager"], default="sdpa")
    parser.add_argument("--min-pixels", type=int, default=12544)
    parser.add_argument("--max-pixels", type=int, default=100352)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA device requested but CUDA is unavailable")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("require 0 <= shard-index < num-shards")
    if args.max_new_tokens != EXPECTED_SATURATION_GENERATION["max_new_tokens"]:
        parser.error("max-new-tokens is frozen at 1024 (V3 generation identity)")
    try:
        seeds = parse_seed_list(
            seeds=args.seeds, seed_start=args.seed_start, num_seeds=args.num_seeds
        )
    except ValueError as exc:
        parser.error(str(exc))

    records = [dict(row) for row in read_jsonl(args.manifest)]
    if not records:
        parser.error("manifest must contain at least one record")
    selected = shard_records(records, args.shard_index, args.num_shards)
    if args.limit is not None:
        selected = selected[: args.limit]
    if not selected:
        parser.error("shard selection is empty")
    ordered_ids = [record["sample_id"] for record in selected]

    if args.num_shards == 1:
        output_path = args.output_dir / OUTPUT_NAME
        contract_path = args.output_dir / "run_config.json"
    else:
        tag = f"shard{args.shard_index:02d}of{args.num_shards:02d}"
        output_path = args.output_dir / f"behavior_noise_floor_r1.{tag}.jsonl"
        contract_path = args.output_dir / f"run_config.{tag}.json"
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    code_hashes = {
        path: sha256_file(PROJECT_ROOT / path) for path in CODE_IDENTITY_PATHS
    }
    code_hashes["scripts/behavior_noise_floor_r1.py"] = sha256_file(
        Path(__file__).resolve()
    )
    model_identity = hash_required_model_identity(args.model)
    ensure_run_contract(
        contract_path,
        run_contract_payload(
            seeds=seeds,
            model=args.model,
            model_identity=model_identity,
            manifest=args.manifest,
            manifest_sha256=sha256_file(args.manifest),
            ordered_sample_ids=ordered_ids,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            limit=args.limit,
            device=args.device,
            attention=args.attention,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            code_hashes=code_hashes,
        ),
        existing_artifacts=(output_path, tmp_path),
    )

    if output_path.exists():
        existing_ids = [row.get("sample_id") for row in read_jsonl(output_path)]
        if existing_ids == ordered_ids:
            raise SystemExit(
                f"complete output {output_path} already exists; refusing to overwrite"
            )
        raise SystemExit(
            f"existing output {output_path} does not match the contract "
            "ordered_sample_ids; refusing to overwrite"
        )

    completed_ids: list[str] = []
    selected_by_id = {record["sample_id"]: record for record in selected}
    if tmp_path.exists():
        for row in read_jsonl(tmp_path):
            sample_id = row.get("sample_id")
            if sample_id not in selected_by_id:
                raise SystemExit(
                    f"partial output {tmp_path} contains unknown sample_id {sample_id!r}"
                )
            expected = selected_by_id[sample_id]
            for field in ("scene_id", "answer_letter", "mapped_answer_letter"):
                if row.get(field) != expected.get(field, expected["answer_letter"]):
                    raise SystemExit(
                        f"partial output {tmp_path}: {field} mismatch for {sample_id!r}"
                    )
            per_seeds = [
                entry.get("seed")
                for branch in NEWLY_GENERATED_BRANCHES
                for entry in (
                    row.get("branches", {}).get(branch, {}).get("rollout", {})
                ).get("per_rollout", [])
            ]
            if per_seeds and sorted(set(per_seeds)) != sorted(seeds):
                raise SystemExit(
                    f"partial output {tmp_path}: per_rollout seeds do not match "
                    f"the contract seed list for {sample_id!r}"
                )
            completed_ids.append(sample_id)
        try:
            validate_resume_prefix(completed_ids, ordered_ids)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        print(f"resuming: {len(completed_ids)}/{len(ordered_ids)} rows complete")

    config_result = load_qwen_config_with_compat(args.model, local_files_only=True)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    processor.image_processor.min_pixels = args.min_pixels
    processor.image_processor.max_pixels = args.max_pixels
    processor.tokenizer.padding_side = "left"
    generation_config = build_official_generation_config(
        max_new_tokens=args.max_new_tokens,
        temperature=1.0,
        pad_token_id=processor.tokenizer.pad_token_id,
    )
    model_dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        config=config_result.config,
        local_files_only=True,
        torch_dtype=model_dtype,
        attn_implementation=args.attention,
        low_cpu_mem_usage=True,
        device_map={"": args.device},
    ).eval()
    print(
        "model_load_audit="
        + json.dumps(validate_qwen_weight_tying(model), sort_keys=True),
        flush=True,
    )

    started = time.perf_counter()
    done = set(completed_ids)
    mode = "a" if completed_ids else "w"
    with tmp_path.open(mode, encoding="utf-8") as sink:
        for index, record in enumerate(selected, start=1):
            if record["sample_id"] in done:
                continue
            row_out = process_record_rollouts(
                model=model,
                processor=processor,
                record=record,
                images_root=args.images_root,
                letters=[option_letter(i) for i in range(len(record["options"]))],
                generation_config=generation_config,
                seeds=seeds,
                device=args.device,
            )
            sink.write(json.dumps(row_out, ensure_ascii=False) + "\n")
            sink.flush()
            print(f"sample={record['sample_id']} {index}/{len(selected)}", flush=True)
    tmp_path.replace(output_path)
    print(
        json.dumps(
            {
                "rows": len(selected),
                "seeds": seeds,
                "wall_seconds": time.perf_counter() - started,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
