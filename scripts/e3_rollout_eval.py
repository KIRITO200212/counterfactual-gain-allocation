#!/usr/bin/env python3
"""E3 smoke rollout evaluation (comparative dev eval, not the frozen audit).

Single-seed G-rollouts-per-row factual/replay evaluation for the plan6 E3
50-step scientific smoke.  This script deliberately mirrors the frozen
reward-saturation audit so E3 comparisons use official semantics:

- generation: ``build_official_generation_config`` (explicit
  GenerationConfig, never the checkpoint-local sampling policy), left-padded
  batched generation, one ``torch.manual_seed`` per output file;
- prompts: the official Stage-3 templates.  MCA rows reuse
  ``reward_saturation_audit.official_prompt`` verbatim; NA (numeric answer)
  rows use the official stage3 numeric template from
  ``grpo_spld_stage3.py:get_question_prompt``;
- scoring: MCA exact letter match via ``clean_stage3_mca_text`` and the
  official numeric ``mean_relative_accuracy`` (bug-for-bug port of the
  stage3 ``na_reward``), plus ``stage3_format_valid``.

It is NOT the frozen preregistered saturation audit: the frozen driver
requires exactly four seeds and letter-only manifests, while E3 needs a
single-seed comparative eval on MCA+NA manifests (confirmation-distribution
dev, diagonal dev, replay sentinel).  Resume policy is fail-closed like the
frozen driver: a complete output file is skipped, a partial file aborts.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from regcfpo.audit import clean_stage3_mca_text, stage3_format_valid  # noqa: E402
from regcfpo.qwen_compat import (  # noqa: E402
    load_qwen_config_with_compat,
    validate_qwen_weight_tying,
)
from regcfpo.run_contract import ensure_run_contract  # noqa: E402
from reward_saturation_audit import (  # noqa: E402
    build_official_generation_config,
    official_prompt,
    read_jsonl,
    sha256_file,
    stable_subset,
)

# Official stage3 question-type routing (grpo_spld_stage3.py:278-289).
MCA_QUESTION_TYPES = ("relative distance", "relative direction", "appearance order")
NA_QUESTION_TYPES = ("object size", "room size", "object count", "absolute distance")

# Official stage3 numeric-answer post-prompt (grpo_spld_stage3.py:375-378).
NA_POST_PROMPT = (
    "Please provide your detailed reasoning between the <think> </think> tags, "
    "and then answer the question with a numerical value (e.g., 42 or 3.1) "
    "within the <answer> </answer> tags."
)
# Official stage3 shared pre-prompt (grpo_spld_stage3.py:362-368); the MCA
# branch itself is inherited verbatim via official_prompt().
STAGE3_PRE_PROMPT = (
    "Question: {question} \n"
    "Please Think about this question as if you were a human pondering deeply. "
    "Engage in an internal dialogue using expressions such as 'let me think', "
    "'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural "
    "language thought expressions It's encouraged to include self-reflection or "
    "verification in the reasoning process. \n"
)


def normalize_options(options: Any) -> list[str] | None:
    """Return the official options list, or None when absent/empty/"None"."""

    if options is None:
        return None
    if isinstance(options, str):
        if options.strip() in ("", "None"):
            return None
        raise ValueError(f"options must be a list or null, got string: {options!r}")
    if isinstance(options, list):
        cleaned = [str(item) for item in options]
        return cleaned or None
    raise ValueError(f"unsupported options type: {type(options).__name__}")


def route_accuracy_mode(record: dict[str, Any]) -> str:
    """Route one record to the official MCA or NA accuracy path."""

    answer_letter = record.get("answer_letter")
    if answer_letter is not None and str(answer_letter).strip() != "":
        return "mca"
    question_type = record.get("question_type")
    if question_type in NA_QUESTION_TYPES:
        return "na"
    if question_type in MCA_QUESTION_TYPES:
        return "mca"
    raise ValueError(
        "record %r cannot be routed: no answer_letter and unsupported "
        "question_type %r" % (record.get("sample_id"), question_type)
    )


def build_prompt(record: dict[str, Any], mode: str) -> str:
    """Build the official Stage-3 prompt for one record."""

    if mode == "mca":
        # Frozen-audit parity: official_prompt appends the options block and
        # the MCA post-prompt exactly as the frozen driver does.
        return official_prompt(record)
    if mode == "na":
        question = str(record["question"]).replace("<image>", "")
        options = normalize_options(record.get("options"))
        if options:
            question += "\nOptions:\n" + "\n".join(options)
        return STAGE3_PRE_PROMPT.format(question=question) + NA_POST_PROMPT
    raise ValueError(f"unknown accuracy mode: {mode!r}")


def answer_text(record: dict[str, Any], mode: str) -> str:
    """Return the reference answer text used by the scorer."""

    if mode == "mca":
        return str(record.get("answer_letter") or record.get("answer"))
    return str(record["answer"])


def _to_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def official_stage3_na_reward(completion: str, answer: str) -> float:
    """Bug-for-bug port of the official stage3 ``na_reward``.

    Mirrors ``mean_relative_accuracy(start=.5, end=.95, interval=.05)`` from
    grpo_spld_stage3.py, including the ``num_pts = ... + 2`` quirk: in binary
    floating point the expression evaluates to 10.999..., so ``int`` yields
    10 linspace points (thresholds 0.5, 0.45, ..., 0.05).  The division uses
    the raw target.  The official corpus has no zero targets and the official
    code would raise on one; here a zero target is scored as exact match
    (documented deviation).
    """

    pred = _to_float(clean_stage3_mca_text(completion))
    target = _to_float(clean_stage3_mca_text(answer))
    if pred is None or target is None:
        return 0.0
    if target == 0.0:
        return 1.0 if pred == 0.0 else 0.0
    num_pts = (0.95 - 0.5) / 0.05 + 2
    conf_intervs = np.linspace(0.5, 0.95, int(num_pts))
    rel_err = abs(pred - target) / target
    return float((rel_err <= 1 - conf_intervs).mean())


def score_completion(completion: str, mode: str, answer: str) -> dict[str, Any]:
    """Score one completion with the official accuracy + format rewards."""

    predicted = clean_stage3_mca_text(completion)
    if mode == "mca":
        accuracy = float(predicted == answer.casefold())
    elif mode == "na":
        accuracy = official_stage3_na_reward(completion, answer)
    else:
        raise ValueError(f"unknown accuracy mode: {mode!r}")
    return {
        "predicted_answer": predicted,
        "accuracy_reward": float(accuracy),
        "format_reward": float(stage3_format_valid(completion)),
    }


def validate_completed_file(
    path: Path, *, records: list[dict[str, Any]], seed: int, rollouts_per_row: int
) -> bool:
    """Fail-closed resume check: complete file -> skip, partial -> error."""

    if not path.is_file():
        return False
    rows = list(read_jsonl(path))
    if not rows:
        return False
    expected_ids = [record["sample_id"] for record in records]
    seen_ids = [row.get("sample_id") for row in rows]
    if seen_ids != expected_ids:
        raise ValueError(
            f"partial or mismatched output file {path}: "
            f"{len(seen_ids)} rows, expected {len(expected_ids)}; delete it to rerun"
        )
    for line_number, row in enumerate(rows, start=1):
        if row.get("seed") != seed:
            raise ValueError(f"{path}:{line_number}: seed mismatch")
        rewards = row.get("accuracy_rewards")
        if not isinstance(rewards, list) or len(rewards) != rollouts_per_row:
            raise ValueError(f"{path}:{line_number}: malformed accuracy_rewards")
    return True


def summarize_rows(
    rows: list[dict[str, Any]], records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Aggregate per-row rollout records into the eval summary body."""

    mean_accuracy = float(np.mean([row["mean_accuracy"] for row in rows]))
    mean_format = float(np.mean([row["mean_format"] for row in rows]))
    truncation = float(np.mean([float(row["hit_max_new_tokens"]) for row in rows]))
    summary: dict[str, Any] = {
        "rows": len(rows),
        "mean_accuracy": mean_accuracy,
        "mean_format_reward": mean_format,
        "truncation_fraction": truncation,
    }
    by_id = {record["sample_id"]: record for record in records}

    def stratified(key: str) -> dict[str, Any]:
        strata: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            record = by_id.get(row["sample_id"], {})
            label = record.get(key)
            if label is None:
                continue
            strata.setdefault(str(label), []).append(row)
        return {
            label: {
                "rows": len(group),
                "mean_accuracy": float(np.mean([r["mean_accuracy"] for r in group])),
                "mean_format_reward": float(np.mean([r["mean_format"] for r in group])),
            }
            for label, group in sorted(strata.items())
        }

    for key in ("cohort", "question_type", "replay_bucket"):
        values = stratified(key)
        if values:
            summary[f"by_{key}"] = values
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_mode.setdefault(row["accuracy_mode"], []).append(row)
    summary["by_accuracy_mode"] = {
        mode: {
            "rows": len(group),
            "mean_accuracy": float(np.mean([r["mean_accuracy"] for r in group])),
            "mean_format_reward": float(np.mean([r["mean_format"] for r in group])),
        }
        for mode, group in sorted(by_mode.items())
    }
    return summary


def generate_file(
    *,
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    records: list[dict[str, Any]],
    images_root: Path,
    output_path: Path,
    seed: int,
    rollouts_per_row: int,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    device: str,
) -> list[dict[str, Any]]:
    """Generate and score one deterministic rollout file (skipped if complete)."""

    if validate_completed_file(
        output_path, records=records, seed=seed, rollouts_per_row=rollouts_per_row
    ):
        print(f"seed={seed}: already complete ({len(records)} rows)")
        return list(read_jsonl(output_path))

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    pad_token_id = processor.tokenizer.pad_token_id
    generation_config = build_official_generation_config(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        pad_token_id=pad_token_id,
    )
    generation_config.num_return_sequences = rollouts_per_row

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written: list[dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8") as sink:
        batches = (
            records[start : start + batch_size]
            for start in range(0, len(records), batch_size)
        )
        for batch_index, batch in enumerate(batches, start=1):
            images: list[Image.Image] = []
            texts: list[str] = []
            try:
                modes = [route_accuracy_mode(record) for record in batch]
                for record, mode in zip(batch, modes, strict=True):
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
                                {"type": "text", "text": build_prompt(record, mode)},
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
                    text=texts, images=images, padding=True, return_tensors="pt"
                ).to(device)
                prompt_length = inputs.input_ids.shape[1]
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs, generation_config=generation_config
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
                for row_index, (record, mode) in enumerate(
                    zip(batch, modes, strict=True)
                ):
                    answer = answer_text(record, mode)
                    slice_ = slice(
                        row_index * rollouts_per_row, (row_index + 1) * rollouts_per_row
                    )
                    row_completions = completions[slice_]
                    scored = [
                        score_completion(completion, mode, answer)
                        for completion in row_completions
                    ]
                    token_counts = [
                        int((generated_tokens[i] != pad_token_id).sum().item())
                        for i in range(
                            row_index * rollouts_per_row,
                            (row_index + 1) * rollouts_per_row,
                        )
                    ]
                    result = {
                        "schema_version": 1,
                        "sample_id": record["sample_id"],
                        "scene_id": record["scene_id"],
                        "seed": seed,
                        "accuracy_mode": mode,
                        "answer": answer,
                        "predicted_answers": [s["predicted_answer"] for s in scored],
                        "accuracy_rewards": [s["accuracy_reward"] for s in scored],
                        "format_rewards": [s["format_reward"] for s in scored],
                        "mean_accuracy": float(
                            np.mean([s["accuracy_reward"] for s in scored])
                        ),
                        "mean_format": float(
                            np.mean([s["format_reward"] for s in scored])
                        ),
                        "completions": row_completions,
                        "prompt_token_count": int(
                            inputs.attention_mask[row_index].sum().item()
                        ),
                        "completion_token_counts": token_counts,
                        "hit_max_new_tokens": any(
                            count >= max_new_tokens for count in token_counts
                        ),
                        "batch_seconds": elapsed,
                        "batch_size": len(batch),
                    }
                    sink.write(json.dumps(result, ensure_ascii=False) + "\n")
                    sink.flush()
                    written.append(result)
            finally:
                for image in images:
                    image.close()
            print(
                f"seed={seed} batch={batch_index} "
                f"completed={min(batch_index * batch_size, len(records))}/{len(records)}",
                flush=True,
            )
    return written


def main() -> int:
    process_started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--images-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--selection-seed", type=int, default=20270818)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--rollouts-per-row", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--min-pixels", type=int, default=16 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=128 * 28 * 28)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attention", choices=["sdpa", "eager"], default="sdpa")
    args = parser.parse_args()
    if (
        args.batch_size < 1
        or args.rollouts_per_row < 1
        or args.max_new_tokens < 1
        or args.temperature <= 0
        or args.min_pixels < 1
        or args.max_pixels < args.min_pixels
        or (args.limit is not None and args.limit < 1)
    ):
        parser.error("invalid batch/token/pixel/temperature/limit bounds")

    records = stable_subset(list(read_jsonl(args.manifest)), args.limit, args.selection_seed)
    if not records:
        parser.error("manifest selection must contain at least one record")
    # Validate routing before touching the GPU.
    for record in records:
        route_accuracy_mode(record)
        normalize_options(record.get("options"))
    print(f"selected_records={len(records)}")

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    processor.image_processor.min_pixels = args.min_pixels
    processor.image_processor.max_pixels = args.max_pixels
    config_result = load_qwen_config_with_compat(args.model, local_files_only=True)
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
    model_load_audit = validate_qwen_weight_tying(model)

    run_contract = ensure_run_contract(
        args.output_dir / "run_config.json",
        {
            "schema_version": 1,
            "audit": "e3_rollout_eval_v1",
            "note": (
                "comparative E3 smoke eval; NOT the frozen preregistered "
                "reward-saturation audit"
            ),
            "model_path": str(args.model),
            "model_config_sha256": sha256_file(args.model / "config.json"),
            "manifest_path": str(args.manifest),
            "manifest_sha256": sha256_file(args.manifest),
            "selection_seed": args.selection_seed,
            "seed": args.seed,
            "rollouts_per_row": args.rollouts_per_row,
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
            "attention": args.attention,
            "device": args.device,
            "dtype": "bfloat16" if args.device.startswith("cuda") else "float32",
            "ordered_sample_ids": [record["sample_id"] for record in records],
            "code_sha256": {
                "scripts/e3_rollout_eval.py": sha256_file(
                    PROJECT_ROOT / "scripts/e3_rollout_eval.py"
                ),
                "scripts/reward_saturation_audit.py": sha256_file(
                    PROJECT_ROOT / "scripts/reward_saturation_audit.py"
                ),
                "src/regcfpo/audit.py": sha256_file(
                    PROJECT_ROOT / "src/regcfpo/audit.py"
                ),
            },
            "resume_policy": "complete file -> skip; partial file -> fail closed",
        },
        existing_artifacts=(args.summary,),
    )

    rows = generate_file(
        model=model,
        processor=processor,
        records=records,
        images_root=args.images_root,
        output_path=args.output_dir / f"seed_{args.seed}.jsonl",
        seed=args.seed,
        rollouts_per_row=args.rollouts_per_row,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=args.device,
    )
    summary = {
        "schema_version": 1,
        "audit": "e3_rollout_eval_v1",
        "model_path": str(args.model),
        "manifest_path": str(args.manifest),
        "seed": args.seed,
        "rollouts_per_row": args.rollouts_per_row,
        "model_load_audit": model_load_audit,
        "run_contract": run_contract,
        "runtime_seconds": time.perf_counter() - process_started,
        **summarize_rows(rows, records),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"summary written to {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
