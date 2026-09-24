#!/usr/bin/env python3
"""Screening-348 behavior-gate G=16 evaluation driver (per leg, per eval group).

Implements the data-collection half of the screening_348 behavior gate frozen
in amendment ``a_plus_plus_screening_authorization_20260831``
(``frozen_confirmatory_spec``, configs/phase1_prereg.yaml): G=16 rollout
log-odds on all 174 directional train rows, swap/null branches, for one
screening leg (AReG or continued, final checkpoint) and one frozen evaluation
seed group.

Seed-group mapping (amendment
``behavior_eval_seed_group_mapping_20260901``): each frozen evaluation seed
anchor e in {1701, 1702, 1703} expands to the deterministic G=16 seed list
``[e + 1000*j for j in 0..15]``.  The three groups are pairwise disjoint.

Code-path identity with LOSS_ALIGNED_PROXY_GATE_V3 / Gate R1
------------------------------------------------------------
Identical to ``scripts/behavior_noise_floor_r1.py``: the generation stream is
literally the V3 code (``generate_rollouts``, ``apply_edit`` imported from the
frozen, hash-bound ``scripts/answer_proxy_gate_v3.py``; official CoT prompt,
official GenerationConfig temperature 1.0 / top_p 1.0 / top_k 50 /
repetition_penalty 1.0 / do_sample / max_new_tokens 1024, one completion per
fixed seed).  The per-row output schema mirrors the V3 / Gate R1 artifacts so
the Gate R1 readout helpers (``row_log_odds``, ``tf_delta_proxy``, ...)
consume these files unchanged.

Deliberate deviation from Gate R1 (documented, not silent): teacher-forced
arms (answer_only + full_template_span) ARE rescored here, because TF scores
are model-dependent and each screening leg needs its own TF side for the
TF-rollout correlation secondary endpoint.  The factual branch is skipped
(the behavior gate uses swap/null only; factual fidelity is evaluated by the
separate e3 harness).

Output hygiene is identical to V3 / Gate R1: rows stream to ``*.jsonl.tmp``
and are atomically renamed on completion; a pre-existing complete output is
never overwritten; a tmp file resumes only when its completed sample_ids are
an ordered prefix of the contract ``ordered_sample_ids``.
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
from transformers import AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from analyze_answer_proxy_v3 import relation_family  # noqa: E402
from answer_proxy_gate import option_letter, sha256_file  # noqa: E402
from answer_proxy_gate_v3 import (  # noqa: E402
    ARMS,
    CODE_IDENTITY_PATHS,
    EXPECTED_SATURATION_GENERATION,
    NEWLY_GENERATED_BRANCHES,
    apply_edit,
    generate_rollouts,
    score_branch_arms,
    validate_resume_prefix,
)
from behavior_noise_floor_r1 import DEFAULT_NUM_SEEDS  # noqa: E402
from frozen_operator_diagnostic import read_jsonl  # noqa: E402
from regcfpo.qwen_compat import (  # noqa: E402
    load_qwen_config_with_compat,
    validate_qwen_weight_tying,
)
from regcfpo.run_contract import ensure_run_contract  # noqa: E402
from reward_saturation_audit import (  # noqa: E402
    build_official_generation_config,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (  # noqa: E402
    Qwen2_5_VLForConditionalGeneration,
)

DIAGNOSTIC = "screening_behavior_eval"
AMENDMENTS = (
    "a_plus_plus_screening_authorization_20260831",
    "behavior_eval_seed_group_mapping_20260901",
)
DEFAULT_MANIFEST = Path("manifests/v3_final/v3_directional_train.jsonl")
DEFAULT_IMAGES_ROOT = Path("data/raw/spatialladder26k/images")
DEFAULT_OUTPUT_ROOT = Path("runs/screening_348_behavior")
OUTPUT_NAME = "screening_behavior_eval.jsonl"

# Frozen evaluation seed anchors (prereg frozen_confirmatory_spec).
FROZEN_EVAL_ANCHORS = (1701, 1702, 1703)
SEED_STRIDE = 1000


def screening_model_identity(model: Path) -> dict[str, Any]:
    """Model identity for trained-leg checkpoints.

    Trained ``final/`` directories saved by the trainer lack two files the
    frozen base-model identity requires (``chat_template.jinja`` — the trainer
    saves ``chat_template.json`` instead — and ``video_preprocessor_config.json``).
    ``hash_required_model_identity`` (frozen base-model audit) therefore
    refuses them.  Here we hash every frozen identity file that IS present,
    plus ``chat_template.json`` when present, and record the missing list
    explicitly (fail-open would be silent; this is fail-explicit).
    """

    from reward_saturation_audit import MODEL_IDENTITY_FILES

    present: dict[str, str] = {}
    missing: list[str] = []
    for name in MODEL_IDENTITY_FILES:
        path = model / name
        if path.is_file():
            present[name] = sha256_file(path)
        else:
            missing.append(name)
    chat_template_json = model / "chat_template.json"
    if chat_template_json.is_file():
        present["chat_template.json"] = sha256_file(chat_template_json)
    if not present:
        raise FileNotFoundError(f"model directory {model} has no identity files")
    return {"files": present, "missing": sorted(missing)}


# ---------------------------------------------------------------------------
# pure helpers (unit-tested without a GPU)


def expand_eval_anchor(anchor: int, num_seeds: int = DEFAULT_NUM_SEEDS) -> list[int]:
    """Frozen mapping: anchor e -> [e + 1000*j for j in 0..num_seeds-1]."""

    if anchor not in FROZEN_EVAL_ANCHORS:
        raise ValueError(
            f"eval anchor must be one of {FROZEN_EVAL_ANCHORS}; got {anchor!r}"
        )
    if num_seeds != DEFAULT_NUM_SEEDS:
        raise ValueError(f"G={DEFAULT_NUM_SEEDS} is frozen; got {num_seeds}")
    seeds = [anchor + SEED_STRIDE * j for j in range(num_seeds)]
    if len(set(seeds)) != num_seeds:
        raise ValueError("seed expansion produced duplicates")
    other = [a for a in FROZEN_EVAL_ANCHORS if a != anchor]
    for a in other:
        overlap = set(seeds) & {a + SEED_STRIDE * j for j in range(num_seeds)}
        if overlap:
            raise ValueError(f"seed groups overlap: {sorted(overlap)!r}")
    return seeds


def run_contract_payload(
    *,
    eval_anchor: int,
    seeds: Sequence[int],
    model: Path,
    model_identity: Mapping[str, str],
    manifest: Path,
    manifest_sha256: str,
    ordered_sample_ids: Sequence[str],
    limit: int | None,
    device: str,
    attention: str,
    min_pixels: int,
    max_pixels: int,
    code_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Screening behavior run contract; records the ACTUAL seed list."""

    return {
        "schema_version": 1,
        "diagnostic": DIAGNOSTIC,
        "amendments": list(AMENDMENTS),
        "gate": "screening_348_behavior_gate",
        "relation_to_v3": (
            "identical V3 G=16 generation stream (one completion per seed, "
            "official prompt/config/operators); teacher-forced arms "
            "regenerated per leg (model-dependent); factual branch skipped; "
            "seed groups follow the frozen anchor expansion "
            "[e + 1000*j for j in 0..15]"
        ),
        "branches": list(NEWLY_GENERATED_BRANCHES),
        "arms_regenerated": list(ARMS),
        "eval_seed_anchor": int(eval_anchor),
        "rollout": {
            "num_generations": len(seeds),
            "seeds": list(seeds),
            "one_completion_per_seed": True,
            "prompt": "official Stage-3 CoT template (reward_saturation_audit.official_prompt)",
            "generation_config": dict(EXPECTED_SATURATION_GENERATION),
        },
        "environment_isolation": "PYTHONNOUSERSITE=1",
        "model_path": str(model),
        "model_config_sha256": model_identity["files"]["config.json"],
        "model_identity_sha256": dict(model_identity["files"]),
        "model_identity_missing": list(model_identity["missing"]),
        "manifest_path": str(manifest),
        "manifest_sha256": manifest_sha256,
        "ordered_sample_ids": list(ordered_sample_ids),
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


def process_record_behavior(
    *,
    model,
    processor,
    record: Mapping[str, Any],
    images_root: Path,
    letters: Sequence[str],
    generation_config,
    seeds: Sequence[int],
    device: str,
) -> dict[str, Any]:
    """Swap/null TF arms + G=16 rollouts for one row; schema mirrors V3."""

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
            arms, template_prompt = score_branch_arms(
                model=model,
                processor=processor,
                record=record,
                image=edited,
                image_path=image_path,
                letters=letters,
                device=device,
            )
            row_out["branches"][branch] = {
                "accepted": True,
                "arms": arms,
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
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--images-root", type=Path, default=DEFAULT_IMAGES_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--eval-anchor",
        type=int,
        required=True,
        help=f"one of the frozen evaluation seed anchors {FROZEN_EVAL_ANCHORS}",
    )
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
    if args.max_new_tokens != EXPECTED_SATURATION_GENERATION["max_new_tokens"]:
        parser.error("max-new-tokens is frozen at 1024 (V3 generation identity)")
    try:
        seeds = expand_eval_anchor(args.eval_anchor)
    except ValueError as exc:
        parser.error(str(exc))

    records = [dict(row) for row in read_jsonl(args.manifest)]
    if not records:
        parser.error("manifest must contain at least one record")
    selected = records
    if args.limit is not None:
        selected = selected[: args.limit]
    ordered_ids = [record["sample_id"] for record in selected]

    output_path = args.output_dir / OUTPUT_NAME
    contract_path = args.output_dir / "run_config.json"
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    code_hashes = {
        path: sha256_file(PROJECT_ROOT / path) for path in CODE_IDENTITY_PATHS
    }
    code_hashes["scripts/screening_behavior_eval.py"] = sha256_file(
        Path(__file__).resolve()
    )
    model_identity = screening_model_identity(args.model)
    ensure_run_contract(
        contract_path,
        run_contract_payload(
            eval_anchor=args.eval_anchor,
            seeds=seeds,
            model=args.model,
            model_identity=model_identity,
            manifest=args.manifest,
            manifest_sha256=sha256_file(args.manifest),
            ordered_sample_ids=ordered_ids,
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
            print(f"complete output {output_path} already exists; skipping")
            return 0
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
            row_out = process_record_behavior(
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
    elapsed = time.perf_counter() - started
    print(f"wrote {output_path} ({len(selected)} rows, {elapsed:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
