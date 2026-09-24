#!/usr/bin/env python3
"""G=16 driver for LOSS_ALIGNED_PROXY_GATE_V3.

Implements the frozen rollout/scoring workload of amendment
``loss_aligned_proxy_amendment_20260822`` (configs/phase1_prereg.yaml;
authority results/phase1/phase1_plan5.md) on the frozen 174-row directional
train cohort:

- branches: factual / pixel_pair_slot_swap / canonical_resampling_return;
- teacher-forced arms: answer_only and full_template_span (full_option_text
  is retired), 4 option letters per branch, candidate suffix spans hard-
  asserted via ``candidate_suffix_mask``;
- swap and null branches generate G=16 rollouts each with the official CoT
  prompt and official GenerationConfig (temperature 1.0, top_p 1.0, top_k 50,
  repetition_penalty 1.0, do_sample, max_new_tokens 1024), one completion per
  fixed seed 1701-1716;
- the factual branch never regenerates: it reuses the dual-cohort saturation
  G=4 results only after per-row byte-identity verification of the official
  prompt and of the saturation generation identity; any mismatch is a hard
  failure (never an approximate reuse).

Output hygiene: rows stream to ``*.jsonl.tmp`` and are atomically renamed on
completion; a pre-existing complete output is never overwritten; a tmp file
resumes only when its completed sample_ids are an ordered prefix of the
contract ``ordered_sample_ids``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, GenerationConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
)

from regcfpo.operators.pixel_ops import (
    canonical_resampling_return,
    pixel_pair_slot_swap,
)
from regcfpo.qwen_compat import (
    load_qwen_config_with_compat,
    validate_qwen_weight_tying,
)
from regcfpo.run_contract import ensure_run_contract

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from analyze_answer_proxy_v3 import relation_family  # noqa: E402
from answer_proxy_gate import (  # noqa: E402
    build_chat_prompt,
    count_rollout_predictions,
    diagnostic_prompt,
    option_letter,
    sha256_file,
)
from frozen_operator_diagnostic import (  # noqa: E402
    prepare_canonical_candidate,
    read_jsonl,
    score_canonical_candidate,
)
from regcfpo.audit import (  # noqa: E402
    clean_stage3_mca_text,
    stage3_format_valid,
)
from reward_saturation_audit import (  # noqa: E402
    build_official_generation_config,
    hash_required_model_identity,
    official_prompt,
)

BRANCH_FACTUAL = "factual"
BRANCH_SWAP = "pixel_pair_slot_swap"
BRANCH_NULL = "canonical_resampling_return"
BRANCHES = (BRANCH_FACTUAL, BRANCH_SWAP, BRANCH_NULL)
NEWLY_GENERATED_BRANCHES = (BRANCH_SWAP, BRANCH_NULL)
ARMS = ("answer_only", "full_template_span")

ROLLOUT_SEEDS = list(range(1701, 1717))  # frozen G=16 seeds 1701..1716
SATURATION_SEEDS = [1701, 1702, 1703, 1704]
EXPECTED_SATURATION_GENERATION = {
    "temperature": 1.0,
    "max_new_tokens": 1024,
    "do_sample": True,
    "top_p": 1.0,
    "top_k": 50,
    "repetition_penalty": 1.0,
}

DEFAULT_MANIFEST = Path("manifests/v3_final/v3_directional_train.jsonl")
DEFAULT_IMAGES_ROOT = Path("data/raw/spatialladder26k/images")
DEFAULT_MODEL = Path("models/SpatialLadder-3B")
DEFAULT_OUTPUT_DIR = Path("runs/answer_proxy_gate_v3")
DEFAULT_SATURATION_DIR = Path("runs/saturation_dual_cohort/train_distribution")
DEFAULT_SATURATION_MANIFEST = Path(
    "manifests/v3_final/saturation_manifest_train_distribution.jsonl"
)

CODE_IDENTITY_PATHS = (
    "scripts/answer_proxy_gate_v3.py",
    "scripts/analyze_answer_proxy_v3.py",
    "scripts/answer_proxy_gate.py",
    "scripts/reward_saturation_audit.py",
    # the driver imports candidate-span helpers from this module as well
    "scripts/frozen_operator_diagnostic.py",
    "src/regcfpo/operators/pixel_ops.py",
    "src/regcfpo/qwen_adapter.py",
    "src/regcfpo/qwen_compat.py",
    "src/regcfpo/audit.py",
)


class SaturationReuseError(ValueError):
    """Raised when the saturation G=4 results cannot be reused byte-identically."""


# ---------------------------------------------------------------------------
# pure helpers (unit-tested without a GPU)


def shard_records(
    records: Sequence[dict[str, Any]], shard_index: int, num_shards: int
) -> list[dict[str, Any]]:
    """Deterministic manifest-order sharding: row i belongs to i % num_shards."""

    if not 0 <= shard_index < num_shards:
        raise ValueError(
            f"shard_index {shard_index} out of range for num_shards {num_shards}"
        )
    return [row for index, row in enumerate(records) if index % num_shards == shard_index]


def validate_resume_prefix(
    completed_ids: Sequence[str], ordered_ids: Sequence[str]
) -> None:
    """A tmp file resumes only as an ordered prefix of the contract ids."""

    if list(completed_ids) != list(ordered_ids[: len(completed_ids)]):
        raise ValueError(
            "partial tmp output is not an ordered prefix of the contract "
            f"ordered_sample_ids (completed={list(completed_ids)[:5]}..., "
            f"expected prefix={list(ordered_ids[: len(completed_ids)])[:5]}...); "
            "refusing to resume"
        )


def verify_saturation_prompt_identity(
    record: Mapping[str, Any], saturation_record: Mapping[str, Any]
) -> None:
    """Byte-identity of the official CoT prompt for one reused factual row."""

    expected_prompt = official_prompt(dict(record))
    reused_prompt = official_prompt(dict(saturation_record))
    if expected_prompt != reused_prompt:
        position = next(
            (
                index
                for index, pair in enumerate(zip(expected_prompt, reused_prompt))
                if pair[0] != pair[1]
            ),
            min(len(expected_prompt), len(reused_prompt)),
        )
        raise SaturationReuseError(
            f"prompt byte mismatch for sample_id {record.get('sample_id')!r} at "
            f"offset {position}: len(expected)={len(expected_prompt)} "
            f"len(reused)={len(reused_prompt)}; refusing approximate reuse"
        )
    for field in ("scene_id", "answer_letter", "image"):
        if record.get(field) != saturation_record.get(field):
            raise SaturationReuseError(
                f"{field} mismatch for sample_id {record.get('sample_id')!r}: "
                f"{record.get(field)!r} != {saturation_record.get(field)!r}"
            )


def validate_saturation_generation_identity(run_config: Mapping[str, Any]) -> None:
    """The reused saturation run must match the frozen generation identity."""

    mismatches: dict[str, Any] = {}
    if list(run_config.get("seeds") or []) != SATURATION_SEEDS:
        mismatches["seeds"] = {
            "expected": SATURATION_SEEDS,
            "observed": run_config.get("seeds"),
        }
    resolved = dict(run_config.get("generation_config_policy", {}).get("resolved", {}))
    for key, expected in EXPECTED_SATURATION_GENERATION.items():
        observed_top = run_config.get(key)
        observed_resolved = resolved.get(key)
        if observed_top is not None and observed_top != expected:
            mismatches[key] = {"expected": expected, "observed": observed_top}
        if observed_resolved is not None and observed_resolved != expected:
            mismatches[f"resolved.{key}"] = {
                "expected": expected,
                "observed": observed_resolved,
            }
    # top_p/top_k/repetition_penalty only live in the resolved config; their
    # absence means the identity cannot be verified, which fails closed.
    for key in ("top_p", "top_k", "repetition_penalty"):
        if key not in resolved:
            mismatches[f"resolved.{key}"] = {"expected": "present", "observed": None}
    if mismatches:
        raise SaturationReuseError(
            "saturation generation identity mismatch; refusing approximate reuse: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )


def load_saturation_reuse(
    saturation_dir: Path,
    saturation_manifest: Path,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Load and fully verify the reused factual G=4 rollouts (fail-closed)."""

    run_config_path = saturation_dir / "run_config.json"
    if not run_config_path.is_file():
        raise SaturationReuseError(f"missing saturation run config {run_config_path}")
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    validate_saturation_generation_identity(run_config)

    manifest_rows = {row["sample_id"]: row for row in read_jsonl(saturation_manifest)}
    seed_rows: dict[int, dict[str, dict[str, Any]]] = {}
    for seed in SATURATION_SEEDS:
        seed_path = saturation_dir / f"seed_{seed}.jsonl"
        if not seed_path.is_file():
            raise SaturationReuseError(f"missing saturation seed file {seed_path}")
        rows: dict[str, dict[str, Any]] = {}
        for row in read_jsonl(seed_path):
            if row.get("seed") != seed:
                raise SaturationReuseError(
                    f"{seed_path}: row for {row.get('sample_id')!r} carries "
                    f"seed={row.get('seed')!r}"
                )
            rows[row["sample_id"]] = row
        seed_rows[seed] = rows

    for record in records:
        sample_id = record["sample_id"]
        saturation_record = manifest_rows.get(sample_id)
        if saturation_record is None:
            raise SaturationReuseError(
                f"sample_id {sample_id!r} missing from {saturation_manifest}"
            )
        verify_saturation_prompt_identity(record, saturation_record)
        for seed in SATURATION_SEEDS:
            row = seed_rows[seed].get(sample_id)
            if row is None:
                raise SaturationReuseError(
                    f"sample_id {sample_id!r} missing from seed_{seed}.jsonl"
                )
            if row.get("answer_letter") != record["answer_letter"]:
                raise SaturationReuseError(
                    f"answer_letter mismatch for {sample_id!r} in seed_{seed}.jsonl"
                )
    return {
        "run_config": run_config,
        "manifest_rows": manifest_rows,
        "seed_rows": seed_rows,
    }


def collect_factual_rollout(
    record: Mapping[str, Any],
    letters: Sequence[str],
    seed_rows: Mapping[int, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Aggregate the verified reused G=4 factual rollouts for one row."""

    sample_id = record["sample_id"]
    per_rollout: list[dict[str, Any]] = []
    completions: list[str] = []
    for seed in SATURATION_SEEDS:
        row = seed_rows[seed][sample_id]
        completion = str(row["completion"])
        completions.append(completion)
        per_rollout.append(
            {
                "seed": seed,
                "parsed_answer": clean_stage3_mca_text(completion),
                "hit_max_new_tokens": bool(row["hit_max_new_tokens"]),
                "completion_token_count": int(row["completion_token_count"]),
                "prompt_token_count": int(row["prompt_token_count"]),
                "completion": completion,
            }
        )
    counts = count_rollout_predictions(completions, list(letters))
    num_generations = len(SATURATION_SEEDS)
    return {
        "counts": counts,
        "frequency": {
            letter: counts[letter] / num_generations for letter in letters
        },
        "answered_fraction": sum(counts.values()) / num_generations,
        "format_valid_fraction": float(
            np.mean([stage3_format_valid(c) for c in completions])
        ),
        "num_generations": num_generations,
        "completions": completions,
        "per_rollout": per_rollout,
    }


# ---------------------------------------------------------------------------
# GPU-bound scoring and generation


def apply_edit(
    branch: str, image: Image.Image, record: Mapping[str, Any]
) -> tuple[Image.Image | None, str]:
    """Apply the frozen pixel operators; rejected edits are never scored."""

    if branch == BRANCH_FACTUAL:
        return image, "accepted"
    box_a, box_b = record["gt_box_a"], record["gt_box_b"]
    if branch == BRANCH_SWAP:
        result = pixel_pair_slot_swap(image, box_a, box_b)
    else:
        result = canonical_resampling_return(image, box_a, box_b)
    if not result.accepted:
        return None, str(result.reject_reason)
    return result.image, "accepted"


def score_text_candidate(
    *,
    model,
    processor,
    prompt: str,
    candidate_text: str,
    image: Image.Image,
    device: str,
) -> dict[str, Any]:
    """Teacher-forced score; the candidate span is hard-asserted by
    ``candidate_suffix_mask`` inside ``prepare_canonical_candidate``."""

    candidate = prepare_canonical_candidate(
        processor=processor,
        prompt=prompt,
        candidate_name="proxy",
        candidate_text=candidate_text,
        image=image,
        device=device,
    )
    total = score_canonical_candidate(model, candidate)
    token_count = len(candidate.token_ids)
    return {
        "logprob": total,
        "token_count": token_count,
        "length_normalized": total / max(1, token_count),
        "token_ids": candidate.token_ids,
    }


def score_branch_arms(
    *,
    model,
    processor,
    record: Mapping[str, Any],
    image: Image.Image,
    image_path: Path,
    letters: Sequence[str],
    device: str,
) -> tuple[dict[str, Any], str]:
    diag_prompt = build_chat_prompt(processor, image_path, diagnostic_prompt(record))
    template_prompt = build_chat_prompt(processor, image_path, official_prompt(dict(record)))
    arms: dict[str, Any] = {}
    for letter in letters:
        arms[letter] = {
            "answer_only": score_text_candidate(
                model=model,
                processor=processor,
                prompt=diag_prompt,
                candidate_text=letter,
                image=image,
                device=device,
            ),
            "full_template_span": score_text_candidate(
                model=model,
                processor=processor,
                prompt=template_prompt,
                candidate_text=f"<answer> {letter} </answer>",
                image=image,
                device=device,
            ),
        }
    return arms, template_prompt


def generate_rollouts(
    *,
    model,
    processor,
    image: Image.Image,
    prompt: str,
    letters: Sequence[str],
    generation_config: GenerationConfig,
    seeds: Sequence[int],
    device: str,
) -> dict[str, Any]:
    """One completion per fixed seed under the official generation identity."""

    inputs = processor(
        text=[prompt], images=[image], padding=True, return_tensors="pt"
    ).to(device)
    prompt_length = inputs.input_ids.shape[1]
    prompt_token_count = int(inputs.attention_mask[0].sum().item())
    pad_token_id = processor.tokenizer.pad_token_id
    max_new_tokens = int(generation_config.max_new_tokens)
    completions: list[str] = []
    per_rollout: list[dict[str, Any]] = []
    for seed in seeds:
        torch.manual_seed(seed)
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(seed)
        with torch.inference_mode():
            generated = model.generate(**inputs, generation_config=generation_config)
        new_tokens = generated[0, prompt_length:]
        completion = processor.batch_decode(
            generated[:, prompt_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        token_count = int((new_tokens != pad_token_id).sum().item())
        completions.append(completion)
        per_rollout.append(
            {
                "seed": int(seed),
                "parsed_answer": clean_stage3_mca_text(completion),
                "hit_max_new_tokens": bool(token_count >= max_new_tokens),
                "completion_token_count": token_count,
                "prompt_token_count": prompt_token_count,
                "completion": completion,
            }
        )
    counts = count_rollout_predictions(completions, list(letters))
    num_generations = len(seeds)
    return {
        "counts": counts,
        "frequency": {
            letter: counts[letter] / num_generations for letter in letters
        },
        "answered_fraction": sum(counts.values()) / num_generations,
        "format_valid_fraction": float(
            np.mean([stage3_format_valid(c) for c in completions])
        ),
        "num_generations": num_generations,
        "completions": completions,
        "per_rollout": per_rollout,
    }


def process_record(
    *,
    model,
    processor,
    record: Mapping[str, Any],
    images_root: Path,
    letters: Sequence[str],
    generation_config: GenerationConfig,
    seed_rows: Mapping[int, Mapping[str, Mapping[str, Any]]],
    device: str,
) -> dict[str, Any]:
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
        for branch in BRANCHES:
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
            branch_out: dict[str, Any] = {"accepted": True, "arms": arms}
            if branch == BRANCH_FACTUAL:
                branch_out["rollout"] = collect_factual_rollout(
                    record, letters, seed_rows
                )
                branch_out["factual_rollout_source"] = (
                    "reused_saturation_dual_cohort_g4"
                )
            else:
                branch_out["rollout"] = generate_rollouts(
                    model=model,
                    processor=processor,
                    image=edited,
                    prompt=template_prompt,
                    letters=letters,
                    generation_config=generation_config,
                    seeds=ROLLOUT_SEEDS,
                    device=device,
                )
            row_out["branches"][branch] = branch_out
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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--saturation-dir", type=Path, default=DEFAULT_SATURATION_DIR)
    parser.add_argument(
        "--saturation-manifest", type=Path, default=DEFAULT_SATURATION_MANIFEST
    )
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
        parser.error("max-new-tokens is frozen at 1024 for the V3 gate")

    records = [dict(row) for row in read_jsonl(args.manifest)]
    if not records:
        parser.error("manifest must contain at least one record")
    selected = shard_records(records, args.shard_index, args.num_shards)
    if args.limit is not None:
        selected = selected[: args.limit]
    if not selected:
        parser.error("shard selection is empty")
    ordered_ids = [record["sample_id"] for record in selected]

    # Verify the factual reuse source byte-identically before any GPU work.
    reuse = load_saturation_reuse(args.saturation_dir, args.saturation_manifest, selected)

    if args.num_shards == 1:
        output_path = args.output_dir / "answer_proxy_v3.jsonl"
        contract_path = args.output_dir / "run_config.json"
    else:
        tag = f"shard{args.shard_index:02d}of{args.num_shards:02d}"
        output_path = args.output_dir / f"answer_proxy_v3.{tag}.jsonl"
        contract_path = args.output_dir / f"run_config.{tag}.json"
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    saturation_hashes = {
        "saturation_manifest_sha256": sha256_file(args.saturation_manifest),
        "saturation_run_config_sha256": sha256_file(
            args.saturation_dir / "run_config.json"
        ),
        "saturation_summary_sha256": sha256_file(
            Path(str(args.saturation_dir) + ".summary.json")
        ),
        "saturation_seed_sha256": {
            str(seed): sha256_file(args.saturation_dir / f"seed_{seed}.jsonl")
            for seed in SATURATION_SEEDS
        },
    }
    code_hashes = {
        path: sha256_file(PROJECT_ROOT / path) for path in CODE_IDENTITY_PATHS
    }
    model_identity = hash_required_model_identity(args.model)
    ensure_run_contract(
        contract_path,
        {
            "schema_version": 1,
            "diagnostic": "loss_aligned_proxy_gate_v3",
            "amendment": "loss_aligned_proxy_amendment_20260822",
            "arms": list(ARMS),
            "retired_arms": ["full_option_text"],
            "branches": list(BRANCHES),
            "newly_generated_branches": list(NEWLY_GENERATED_BRANCHES),
            "rollout": {
                "num_generations": len(ROLLOUT_SEEDS),
                "seeds": ROLLOUT_SEEDS,
                "one_completion_per_seed": True,
                "prompt": "official Stage-3 CoT template (reward_saturation_audit.official_prompt)",
                "generation_config": dict(EXPECTED_SATURATION_GENERATION),
            },
            "factual_branch": {
                "source": "reused_saturation_dual_cohort_g4",
                "seeds": SATURATION_SEEDS,
                "verification": "per-row prompt byte identity plus generation identity",
            },
            "thresholds": {
                "answered_fraction_min": 0.95,
                "factual_top1_agreement_min": 0.90,
                "swap_conditional_sign_min": 0.90,
                "null_conditional_sign_min": 0.95,
                "swap_scene_spearman_min": 0.80,
                "swap_scene_spearman_lcb95_min": 0.70,
            },
            "environment_isolation": "PYTHONNOUSERSITE=1",
            "model_path": str(args.model),
            "model_config_sha256": model_identity["config.json"],
            "model_identity_sha256": model_identity,
            "manifest_path": str(args.manifest),
            "manifest_sha256": sha256_file(args.manifest),
            "ordered_sample_ids": ordered_ids,
            "shard": {"index": args.shard_index, "num_shards": args.num_shards},
            "limit": args.limit,
            "device": args.device,
            "attention": args.attention,
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
            "torch_version": torch.__version__,
            "code_sha256": code_hashes,
            **saturation_hashes,
        },
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
            row_out = process_record(
                model=model,
                processor=processor,
                record=record,
                images_root=args.images_root,
                letters=[option_letter(i) for i in range(len(record["options"]))],
                generation_config=generation_config,
                seed_rows=reuse["seed_rows"],
                device=args.device,
            )
            sink.write(json.dumps(row_out, ensure_ascii=False) + "\n")
            sink.flush()
            print(f"sample={record['sample_id']} {index}/{len(selected)}", flush=True)
    tmp_path.replace(output_path)
    print(
        json.dumps(
            {"rows": len(selected), "wall_seconds": time.perf_counter() - started},
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
