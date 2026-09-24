#!/usr/bin/env python3
"""Run resumable G=4 factual rollouts for the reward-saturation audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image
import transformers
from transformers import (
    AutoProcessor,
    GenerationConfig,
    Qwen2_5_VLForConditionalGeneration,
)
import yaml

from regcfpo.audit import (
    OFFICIAL_GRPO_EPSILON,
    OFFICIAL_GRPO_STD_CORRECTION,
    OFFICIAL_REWARD_WEIGHTS,
    audit_reward_saturation,
    clean_stage3_mca_text,
    stage3_format_valid,
)
from regcfpo.qwen_compat import (
    load_qwen_config_with_compat,
    validate_qwen_weight_tying,
)
from regcfpo.run_contract import ensure_run_contract


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPATIALLADDER_COMMIT = "7a0d2ee85c28728835300310a349a53a15967f2e"
OFFICIAL_GENERATION_CONFIG_SOURCE = (
    "VLM-R1/src/open-r1-multimodal/src/open_r1/trainer/grpo_trainer.py"
)
OFFICIAL_GENERATION_CONFIG_LINES = (384, 389)
RETRY_POLICY = {
    "automatic_generation_retries": 0,
    "partial_seed_jsonl": "reject",
    "complete_seed_jsonl": "skip",
    "empty_seed_jsonl": "restart_from_seed",
}
MODEL_IDENTITY_FILES = (
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "chat_template.jinja",
    "special_tokens_map.json",
    "added_tokens.json",
    "merges.txt",
    "vocab.json",
    "model.safetensors.index.json",
    "video_preprocessor_config.json",
)


class SeedOutputError(ValueError):
    """Raised when an existing per-seed output is unsafe to resume."""


def build_official_generation_config(
    *,
    max_new_tokens: int,
    temperature: float,
    pad_token_id: int,
) -> GenerationConfig:
    """Mirror the explicit GenerationConfig constructed by the pinned trainer.

    Passing this object to ``generate`` is essential.  Omitting it would make
    Transformers inherit checkpoint-local generation settings, including the
    released checkpoint's non-official repetition penalty and cache policy.
    """

    return GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        pad_token_id=pad_token_id,
    )


def validate_frozen_seeds(seeds: list[int]) -> None:
    """Require four distinct integer rollout streams for the frozen G=4 audit."""

    if len(seeds) != 4:
        raise ValueError("the frozen reward audit requires exactly four seeds")
    if any(not _strict_integer(seed) for seed in seeds):
        raise ValueError("the frozen reward audit seeds must be integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("the frozen reward audit seeds must be unique")


def validate_preregistered_generation_policy(
    path: Path,
    *,
    seeds: list[int],
    generation_config: GenerationConfig,
) -> dict[str, Any]:
    """Load and enforce the frozen generation fields used by this audit."""

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read preregistration config {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in preregistration config {path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("reproducibility"), dict):
        raise ValueError("preregistration config must define a reproducibility mapping")
    reproducibility = payload["reproducibility"]
    expected = {
        "generation_seeds": seeds,
        "group_size": len(seeds),
        "max_completion_length": generation_config.max_new_tokens,
        "generation_temperature": generation_config.temperature,
        "generation_top_p": generation_config.top_p,
        "generation_top_k": generation_config.top_k,
        "generation_repetition_penalty": generation_config.repetition_penalty,
        "generation_use_cache": generation_config.use_cache,
    }
    mismatches = {
        key: {"preregistered": reproducibility.get(key), "requested": value}
        for key, value in expected.items()
        if reproducibility.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "requested generation policy does not match preregistration: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )
    if reproducibility.get("group_size") != 4:
        raise ValueError("preregistered reward audit group_size must be 4")
    upstream = payload.get("upstream")
    if not isinstance(upstream, dict) or upstream.get("commit") != SPATIALLADDER_COMMIT:
        raise ValueError(
            f"preregistration upstream.commit must equal {SPATIALLADDER_COMMIT}"
        )
    return expected


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
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
            yield value


def stable_subset(
    records: list[dict[str, Any]], limit: int | None, selection_seed: int
) -> list[dict[str, Any]]:
    ordered = sorted(
        records,
        key=lambda row: hashlib.sha256(
            f"{selection_seed}:{row['sample_id']}".encode()
        ).hexdigest(),
    )
    return ordered if limit is None else ordered[:limit]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_required_model_identity(model: Path) -> dict[str, str]:
    """Hash every frozen non-weight model/processor/tokenizer identity file."""

    missing = [name for name in MODEL_IDENTITY_FILES if not (model / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"model directory {model} is missing required identity files: {missing}"
        )
    return {name: sha256_file(model / name) for name in MODEL_IDENTITY_FILES}


def build_run_contract(
    *,
    model: Path,
    model_manifest: Path,
    manifest: Path,
    prereg_config: Path,
    records: list[dict[str, Any]],
    selection_seed: int,
    seeds: list[int],
    batch_size: int,
    generation_config: GenerationConfig,
    min_pixels: int,
    max_pixels: int,
    attention: str,
) -> dict[str, Any]:
    """Build the immutable provenance and generation contract for future runs."""

    validate_frozen_seeds(seeds)
    model_identity_sha256 = hash_required_model_identity(model)
    resolved_generation_config = generation_config.to_dict()
    preregistered_generation_policy = validate_preregistered_generation_policy(
        prereg_config,
        seeds=seeds,
        generation_config=generation_config,
    )
    return {
        "schema_version": 1,
        "audit": "official_stage3_reward_saturation",
        "prompt_policy": f"SpatialLadder Stage-3 at commit {SPATIALLADDER_COMMIT}",
        "accuracy_policy": "official clean_text MCA exact match",
        "format_policy": "official Stage-3 think/answer format predicate",
        "reward_weights": list(OFFICIAL_REWARD_WEIGHTS),
        "advantage_policy": {
            "reward": "accuracy_plus_format",
            "std_correction": OFFICIAL_GRPO_STD_CORRECTION,
            "epsilon": OFFICIAL_GRPO_EPSILON,
        },
        "model_path": str(model),
        "model_config_sha256": model_identity_sha256["config.json"],
        "model_identity_sha256": model_identity_sha256,
        "model_manifest_path": str(model_manifest),
        "model_manifest_sha256": sha256_file(model_manifest),
        "manifest_path": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "preregistration": {
            "path": str(prereg_config),
            "sha256": sha256_file(prereg_config),
            "validated_generation_policy": preregistered_generation_policy,
        },
        "driver_sha256": sha256_file(Path(__file__).resolve()),
        "audit_sha256": sha256_file(PROJECT_ROOT / "src" / "regcfpo" / "audit.py"),
        "qwen_compat_sha256": sha256_file(
            PROJECT_ROOT / "src" / "regcfpo" / "qwen_compat.py"
        ),
        "torch_version": str(torch.__version__),
        "transformers_version": transformers.__version__,
        "dtype": "bfloat16",
        "generation_config_policy": {
            "upstream_commit": SPATIALLADDER_COMMIT,
            "upstream_source": OFFICIAL_GENERATION_CONFIG_SOURCE,
            "upstream_lines_inclusive": list(OFFICIAL_GENERATION_CONFIG_LINES),
            "construction": "explicit GenerationConfig; never inherit checkpoint sampling policy",
            "resolved": resolved_generation_config,
        },
        "do_sample": generation_config.do_sample,
        "padding_side": "left",
        "retry_policy": dict(RETRY_POLICY),
        "selected_sample_ids": [record["sample_id"] for record in records],
        "selection_seed": selection_seed,
        "seeds": seeds,
        "batch_size": batch_size,
        "max_new_tokens": generation_config.max_new_tokens,
        "temperature": generation_config.temperature,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "attention": attention,
    }


def official_prompt(record: dict[str, Any]) -> str:
    question = record["question"]
    options = record["options"]
    question_with_options = question + "\nOptions:\n" + "\n".join(options)
    return (
        f"Question: {question_with_options} \n"
        "Please Think about this question as if you were a human pondering deeply. "
        "Engage in an internal dialogue using expressions such as 'let me think', 'wait', "
        "'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought "
        "expressions It's encouraged to include self-reflection or verification in the "
        "reasoning process. \n"
        "Please provide your detailed reasoning between the <think> </think> tags, and then "
        "answer the question with the option's letter from the given choices (e.g., A, B, "
        "etc.) within the <answer> </answer> tags."
    )


def _strict_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_binary_reward(path: Path, line_number: int, row: dict[str, Any], key: str) -> None:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SeedOutputError(f"{path}:{line_number}: {key} must be numeric binary 0/1")
    numeric = float(value)
    if not np.isfinite(numeric) or numeric not in (0.0, 1.0):
        raise SeedOutputError(f"{path}:{line_number}: {key} must be finite binary 0/1")


def load_completed(
    path: Path,
    *,
    records: list[dict[str, Any]],
    expected_seed: int,
) -> dict[str, dict[str, Any]]:
    """Validate a seed JSONL as empty or complete; reject partial adoption."""

    expected: dict[str, dict[str, Any]] = {}
    for record in records:
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise SeedOutputError("selected records must have non-empty string sample_id")
        if sample_id in expected:
            raise SeedOutputError(f"selected records contain duplicate sample_id {sample_id!r}")
        if "scene_id" not in record or "answer_letter" not in record:
            raise SeedOutputError(
                f"selected record {sample_id!r} must define scene_id and answer_letter"
            )
        expected[sample_id] = record
    if not expected:
        raise SeedOutputError("selected records must not be empty")
    if not path.exists():
        return {}
    completed: dict[str, dict[str, Any]] = {}
    for line_number, row in enumerate(read_jsonl(path), start=1):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise SeedOutputError(f"{path}:{line_number}: invalid sample_id")
        if sample_id not in expected:
            raise SeedOutputError(
                f"{path}:{line_number}: unknown sample_id {sample_id!r}"
            )
        if sample_id in completed:
            raise SeedOutputError(
                f"{path}:{line_number}: duplicate sample_id {sample_id!r}"
            )
        observed_seed = row.get("seed")
        if not _strict_integer(observed_seed) or observed_seed != expected_seed:
            raise SeedOutputError(
                f"{path}:{line_number}: wrong seed; expected {expected_seed}, "
                f"received {observed_seed!r}"
            )
        record = expected[sample_id]
        if row.get("scene_id") != record["scene_id"]:
            raise SeedOutputError(f"{path}:{line_number}: scene mismatch for {sample_id!r}")
        if row.get("answer_letter") != record["answer_letter"]:
            raise SeedOutputError(f"{path}:{line_number}: answer mismatch for {sample_id!r}")
        _validate_binary_reward(path, line_number, row, "accuracy_reward")
        _validate_binary_reward(path, line_number, row, "format_reward")
        completed[sample_id] = row
    if completed and len(completed) != len(expected):
        missing = [sample_id for sample_id in expected if sample_id not in completed]
        raise SeedOutputError(
            f"{path}: partial seed JSONL has {len(completed)}/{len(expected)} rows; "
            f"missing={missing[:5]}; refusing resume"
        )
    return completed


def nonempty_seed_artifacts(output_dir: Path) -> list[Path]:
    """Return seed files that cannot be adopted without an existing contract."""

    artifacts: list[Path] = []
    for path in output_dir.glob("seed_*.jsonl"):
        if not path.is_file():
            continue
        try:
            has_content = bool(path.read_text(encoding="utf-8").strip())
        except (OSError, UnicodeDecodeError):
            has_content = True
        if has_content:
            artifacts.append(path)
    return artifacts


def reject_uncontracted_seed_files(output_dir: Path, seeds: list[int]) -> None:
    """Reject per-seed files whose names are outside the frozen seed set."""

    expected_names = {f"seed_{seed}.jsonl" for seed in seeds}
    extras = sorted(
        path.name
        for path in output_dir.glob("seed_*.jsonl")
        if path.is_file() and path.name not in expected_names
    )
    if extras:
        raise SeedOutputError(f"uncontracted seed JSONL files present: {extras}")


def batches(values: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def generate_seed(
    *,
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    records: list[dict[str, Any]],
    images_root: Path,
    output_path: Path,
    seed: int,
    batch_size: int,
    generation_config: GenerationConfig,
    device: str,
) -> None:
    completed = load_completed(
        output_path,
        records=records,
        expected_seed=seed,
    )
    if completed:
        print(f"seed={seed}: already complete ({len(completed)} rows)")
        return

    # Empty/missing is the only resumable state.  Seeding remains once per
    # uninterrupted seed stream; a crash leaves a partial file that is rejected.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    max_new_tokens = generation_config.max_new_tokens
    if not isinstance(max_new_tokens, int) or max_new_tokens < 1:
        raise ValueError("generation_config.max_new_tokens must be a positive integer")
    if generation_config.pad_token_id != processor.tokenizer.pad_token_id:
        raise ValueError(
            "generation_config.pad_token_id must match processor.tokenizer.pad_token_id"
        )
    with output_path.open("w", encoding="utf-8") as sink:
        for batch_index, batch in enumerate(batches(records, batch_size), start=1):
            images: list[Image.Image] = []
            texts: list[str] = []
            try:
                for record in batch:
                    image_path = images_root / record["image"]
                    if not image_path.is_file():
                        raise FileNotFoundError(image_path)
                    image = Image.open(image_path).convert("RGB")
                    images.append(image)
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": str(image_path)},
                                {"type": "text", "text": official_prompt(record)},
                            ],
                        }
                    ]
                    texts.append(
                        processor.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                    )

                started = time.perf_counter()
                inputs = processor(
                    text=texts,
                    images=images,
                    padding=True,
                    return_tensors="pt",
                ).to(device)
                prompt_length = inputs.input_ids.shape[1]
                sliding_window = getattr(model.config, "sliding_window", None)
                if (
                    sliding_window is not None
                    and prompt_length + max_new_tokens > int(sliding_window)
                ):
                    raise RuntimeError(
                        "prompt plus completion exceeds the frozen 4.49 sliding-window "
                        f"boundary: {prompt_length}+{max_new_tokens}>{sliding_window}"
                    )
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        generation_config=generation_config,
                    )
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                completions = processor.batch_decode(
                    generated[:, prompt_length:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                generated_tokens = generated[:, prompt_length:]
                pad_token_id = processor.tokenizer.pad_token_id
                for row_index, (record, completion) in enumerate(
                    zip(batch, completions, strict=True)
                ):
                    predicted = clean_stage3_mca_text(completion)
                    completion_token_count = int(
                        (generated_tokens[row_index] != pad_token_id).sum().item()
                    )
                    result = {
                        "schema_version": 1,
                        "sample_id": record["sample_id"],
                        "scene_id": record["scene_id"],
                        "seed": seed,
                        "answer_letter": record["answer_letter"],
                        "predicted_answer": predicted,
                        "accuracy_reward": float(
                            predicted == record["answer_letter"].casefold()
                        ),
                        "format_reward": float(stage3_format_valid(completion)),
                        "completion": completion,
                        "prompt_token_count": int(inputs.attention_mask[row_index].sum().item()),
                        "completion_token_count": completion_token_count,
                        "hit_max_new_tokens": completion_token_count >= max_new_tokens,
                        "batch_seconds": elapsed,
                        "batch_size": len(batch),
                    }
                    sink.write(json.dumps(result, ensure_ascii=False) + "\n")
                    sink.flush()
            finally:
                for image in images:
                    image.close()
            print(
                f"seed={seed} batch={batch_index} completed="
                f"{min(batch_index * batch_size, len(records))}/{len(records)}"
            )


def write_summary(
    output_dir: Path,
    selected: list[dict[str, Any]],
    seeds: list[int],
    summary_path: Path,
    *,
    max_new_tokens: int,
    temperature: float,
    min_pixels: int,
    max_pixels: int,
    config_compatibility: dict[str, Any],
    model_load_audit: dict[str, Any],
    runtime_audit: dict[str, Any],
    run_contract: dict[str, Any],
) -> None:
    by_seed = {
        seed: load_completed(
            output_dir / f"seed_{seed}.jsonl",
            records=selected,
            expected_seed=seed,
        )
        for seed in seeds
    }
    accuracy_rows: list[list[float]] = []
    format_rows: list[list[float]] = []
    sample_ids: list[str] = []
    scene_ids: list[str] = []
    format_rewards: list[float] = []
    truncation_flags: list[bool] = []
    observed_batch_seconds: list[float] = []
    for record in selected:
        sample_id = record["sample_id"]
        accuracy_rows.append(
            [by_seed[seed][sample_id]["accuracy_reward"] for seed in seeds]
        )
        format_row = [by_seed[seed][sample_id]["format_reward"] for seed in seeds]
        format_rows.append(format_row)
        format_rewards.extend(format_row)
        truncation_flags.extend(
            bool(by_seed[seed][sample_id].get("hit_max_new_tokens", False))
            for seed in seeds
        )
        observed_batch_seconds.extend(
            float(by_seed[seed][sample_id]["batch_seconds"]) for seed in seeds
        )
        sample_ids.append(sample_id)
        scene_ids.append(record["scene_id"])
    if not accuracy_rows:
        raise RuntimeError("no sample has a complete set of seed rollouts")
    accuracy_matrix = np.asarray(accuracy_rows, dtype=np.float32)
    format_matrix = np.asarray(format_rows, dtype=np.float32)
    audit = audit_reward_saturation(
        accuracy_matrix,
        format_matrix,
        expected_group_size=len(seeds),
    )
    summary = {
        "schema_version": 1,
        "model_outputs_dir": str(output_dir),
        "seeds": seeds,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "config_compatibility": config_compatibility,
        "model_load_audit": model_load_audit,
        "runtime_audit": runtime_audit,
        "run_contract": run_contract,
        "completed_sample_ids": sample_ids,
        "completed_scene_count": len(set(scene_ids)),
        "mean_format_reward": float(np.mean(format_rewards, dtype=np.float32)),
        "truncation_fraction": float(np.mean(truncation_flags, dtype=np.float32)),
        "mean_observed_batch_seconds_per_row": float(
            np.mean(observed_batch_seconds, dtype=np.float32)
        ),
        "reward_audit": audit.to_dict(),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> int:
    process_started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument(
        "--model-manifest",
        type=Path,
        default=Path("manifests/model_manifest.json"),
    )
    parser.add_argument(
        "--prereg-config",
        type=Path,
        default=Path("configs/phase1_prereg.yaml"),
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--images-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--selection-seed", type=int, default=20270818)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1701, 1702, 1703, 1704])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--min-pixels", type=int, default=16 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=128 * 28 * 28)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attention", choices=["sdpa", "eager"], default="sdpa")
    args = parser.parse_args()
    try:
        validate_frozen_seeds(args.seeds)
    except ValueError as exc:
        parser.error(str(exc))
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if (
        args.batch_size < 1
        or args.max_new_tokens < 1
        or args.temperature <= 0
        or args.min_pixels < 1
        or args.max_pixels < args.min_pixels
    ):
        parser.error("batch size, token/pixel bounds, and temperature must be valid")

    records = stable_subset(list(read_jsonl(args.manifest)), args.limit, args.selection_seed)
    if not records:
        parser.error("manifest selection must contain at least one record")
    print(f"selected_records={len(records)}")
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    processor.image_processor.min_pixels = args.min_pixels
    processor.image_processor.max_pixels = args.max_pixels
    pad_token_id = processor.tokenizer.pad_token_id
    if not _strict_integer(pad_token_id):
        raise ValueError("processor.tokenizer.pad_token_id must be an integer")
    generation_config = build_official_generation_config(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        pad_token_id=pad_token_id,
    )
    reject_uncontracted_seed_files(args.output_dir, args.seeds)
    run_contract = ensure_run_contract(
        args.output_dir / "run_config.json",
        build_run_contract(
            model=args.model,
            model_manifest=args.model_manifest,
            manifest=args.manifest,
            prereg_config=args.prereg_config,
            records=records,
            selection_seed=args.selection_seed,
            seeds=args.seeds,
            batch_size=args.batch_size,
            generation_config=generation_config,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            attention=args.attention,
        ),
        existing_artifacts=nonempty_seed_artifacts(args.output_dir),
    )
    config_result = load_qwen_config_with_compat(args.model, local_files_only=True)
    config_compatibility = config_result.to_record()
    print(
        "config_compatibility="
        + json.dumps(config_compatibility, ensure_ascii=False, sort_keys=True),
        flush=True,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        config=config_result.config,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attention,
        low_cpu_mem_usage=True,
        device_map={"": args.device},
    ).eval()
    model_load_audit = validate_qwen_weight_tying(model)
    print(
        "model_load_audit="
        + json.dumps(model_load_audit, ensure_ascii=False, sort_keys=True),
        flush=True,
    )
    print(
        f"model_loaded device={model.device} dtype={next(model.parameters()).dtype} "
        f"attention={args.attention}"
    )
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    for seed in args.seeds:
        generate_seed(
            model=model,
            processor=processor,
            records=records,
            images_root=args.images_root,
            output_path=args.output_dir / f"seed_{seed}.jsonl",
            seed=seed,
            batch_size=args.batch_size,
            generation_config=generation_config,
            device=args.device,
        )
    runtime_audit = {
        "requested_device": args.device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "resolved_model_device": str(model.device),
        "device_name": torch.cuda.get_device_name(model.device)
        if args.device.startswith("cuda")
        else None,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated()
        if args.device.startswith("cuda")
        else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved()
        if args.device.startswith("cuda")
        else None,
        "wall_seconds": time.perf_counter() - process_started,
    }
    write_summary(
        args.output_dir,
        records,
        args.seeds,
        args.summary,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        config_compatibility=config_compatibility,
        model_load_audit=model_load_audit,
        runtime_audit=runtime_audit,
        run_contract=run_contract,
    )
    if args.device.startswith("cuda"):
        print(f"peak_allocated_bytes={runtime_audit['peak_allocated_bytes']}")
        print(f"peak_reserved_bytes={runtime_audit['peak_reserved_bytes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
