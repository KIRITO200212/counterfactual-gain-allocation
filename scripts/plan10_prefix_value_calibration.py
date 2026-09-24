#!/usr/bin/env python3
"""Calibrate whether Plan10 prefix credit predicts reachable mapped answers.

For each swapped-image prompt this diagnostic first samples G full on-policy
completions.  It then freezes every valid reasoning prefix immediately before
the generated ``<answer>`` tag and independently resamples a short suffix.
The report compares the detached mapped-vs-original boundary credit with the
empirical mapped-answer frequency from those suffix samples.

This is a read-only theta0 diagnostic: it performs no optimization and writes
one atomic JSON report.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

import torch
from PIL import Image
from transformers import AutoProcessor
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from answer_proxy_gate import build_chat_prompt, option_letter, sha256_file  # noqa: E402
from frozen_operator_diagnostic import read_jsonl  # noqa: E402
from regcfpo.audit import clean_stage3_mca_text  # noqa: E402
from regcfpo.qwen_compat import (  # noqa: E402
    load_qwen_config_with_compat,
    validate_qwen_weight_tying,
)
from regcfpo.training.prefix_credit import (  # noqa: E402
    OPTION_LETTERS,
    build_answer_token_styles,
    locate_answer_candidate_tokens,
)
from reward_saturation_audit import (  # noqa: E402
    build_official_generation_config,
    official_prompt,
)
from screening_behavior_eval import screening_model_identity  # noqa: E402


DEFAULT_PREFIX_SEEDS = (1701, 2701, 3701, 4701)


def _set_seed(seed: int, device: str) -> None:
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        average_rank = 0.5 * ((start + 1) + stop)
        for position in range(start, stop):
            ranks[order[position]] = average_rank
        start = stop
    return ranks


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_mean = mean(xs)
    y_mean = mean(ys)
    x_centered = [value - x_mean for value in xs]
    y_centered = [value - y_mean for value in ys]
    denominator = math.sqrt(
        sum(value * value for value in x_centered)
        * sum(value * value for value in y_centered)
    )
    if denominator == 0.0:
        return None
    return sum(x * y for x, y in zip(x_centered, y_centered)) / denominator


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Tie-aware Spearman correlation, or ``None`` for a constant side."""

    if len(xs) != len(ys) or not xs:
        return None
    return _pearson(_average_ranks(xs), _average_ranks(ys))


def relation_stratum(relation: str) -> str:
    if relation in {"left", "right"}:
        return "axis"
    if relation in {"left-front", "right-back"}:
        return "LF_RB"
    if relation in {"left-back", "right-front"}:
        return "LB_RF"
    raise ValueError(f"unsupported relation for Plan10 selector: {relation!r}")


def _model_inputs_with_ids(base_inputs: Mapping[str, Any], input_ids: torch.Tensor) -> dict[str, Any]:
    result = dict(base_inputs)
    result["input_ids"] = input_ids
    result["attention_mask"] = torch.ones_like(input_ids)
    return result


def _option_index(letter: str) -> int:
    upper = str(letter).upper()
    if upper not in OPTION_LETTERS:
        raise ValueError(f"expected one of A-D, got {letter!r}")
    return OPTION_LETTERS.index(upper)


def _swap_rows(
    rows: Sequence[Mapping[str, Any]],
    limit: int | None,
    *,
    deduplicate_parents: bool = False,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in rows:
        row = dict(raw)
        if row.get("pairaug_branch") != "swap":
            continue
        parent = str(row.get("pairaug_parent_sample_id", row["sample_id"]))
        if parent in seen:
            if deduplicate_parents:
                continue
            raise ValueError(f"duplicate swapped parent {parent!r}")
        seen.add(parent)
        selected.append(row)
        if limit is not None and len(selected) >= limit:
            break
    if not selected:
        raise ValueError("manifest contains no pairaug_branch=swap rows")
    return selected


def _summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    prefixes = [
        prefix
        for row in rows
        for prefix in row["prefixes"]
        if prefix["valid"]
    ]
    credits_t1 = [float(prefix["credit_t1"]) for prefix in prefixes]
    credits_t2 = [float(prefix["credit_t2"]) for prefix in prefixes]
    suffix_rates = [float(prefix["suffix_mapped_rate"]) for prefix in prefixes]
    top_bottom: list[float] = []
    for row in rows:
        valid = [prefix for prefix in row["prefixes"] if prefix["valid"]]
        if len(valid) < 2:
            continue
        top = max(valid, key=lambda prefix: float(prefix["credit_t2"]))
        bottom = min(valid, key=lambda prefix: float(prefix["credit_t2"]))
        top_bottom.append(
            float(top["suffix_mapped_rate"]) - float(bottom["suffix_mapped_rate"])
        )
    per_stratum: dict[str, Any] = {}
    for stratum in ("axis", "LF_RB", "LB_RF"):
        selected = [row for row in rows if row["stratum"] == stratum]
        per_stratum[stratum] = {
            "groups": len(selected),
            "all_wrong_groups": sum(bool(row["all_wrong_full_rollouts"]) for row in selected),
            "suffix_reachable_groups": sum(bool(row["suffix_mapped_count"] > 0) for row in selected),
        }
    return {
        "groups": len(rows),
        "valid_prefixes": len(prefixes),
        "total_prefixes": sum(len(row["prefixes"]) for row in rows),
        "all_wrong_group_fraction": mean(
            [float(bool(row["all_wrong_full_rollouts"])) for row in rows]
        ),
        "suffix_reachable_group_fraction": mean(
            [float(row["suffix_mapped_count"] > 0) for row in rows]
        ),
        "mean_boundary_mapped_probability": (
            mean(float(prefix["option_probabilities"][prefix["mapped_letter"]]) for prefix in prefixes)
            if prefixes
            else None
        ),
        "spearman_credit_t1_vs_suffix_mapped_rate": spearman(credits_t1, suffix_rates),
        "spearman_credit_t2_vs_suffix_mapped_rate": spearman(credits_t2, suffix_rates),
        "temperature_rank_invariant": (
            _average_ranks(credits_t1) == _average_ranks(credits_t2) if prefixes else None
        ),
        "mean_top_minus_bottom_suffix_mapped_rate": mean(top_bottom) if top_bottom else None,
        "strata": per_stratum,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attention", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--prefix-seed", action="append", type=int, default=[])
    parser.add_argument("--suffix-samples", type=int, default=8)
    parser.add_argument("--suffix-seed-base", type=int, default=91001)
    parser.add_argument("--max-suffix-tokens", type=int, default=64)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--exclude-report",
        action="append",
        type=Path,
        default=[],
        help="exclude parent ids already present in an earlier calibration report",
    )
    parser.add_argument(
        "--deduplicate-parents",
        action="store_true",
        help="evaluate the first visit of each parent in a multi-cycle selector",
    )
    parser.add_argument("--min-pixels", type=int, default=12544)
    parser.add_argument("--max-pixels", type=int, default=100352)
    args = parser.parse_args()
    prefix_seeds = tuple(args.prefix_seed) if args.prefix_seed else DEFAULT_PREFIX_SEEDS
    if len(prefix_seeds) < 2 or len(prefix_seeds) != len(set(prefix_seeds)):
        parser.error("use at least two distinct --prefix-seed values")
    if args.suffix_samples < 2:
        parser.error("--suffix-samples must be at least 2")
    if args.max_suffix_tokens < 8:
        parser.error("--max-suffix-tokens must be at least 8")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.num_shards < 1:
        parser.error("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must lie in [0, num-shards)")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing report: {args.output}")

    manifest_rows = [dict(row) for row in read_jsonl(args.manifest)]
    all_rows_before_exclusion = _swap_rows(
        manifest_rows,
        args.limit,
        deduplicate_parents=args.deduplicate_parents,
    )
    excluded_parents: set[str] = set()
    exclude_contracts: list[dict[str, str]] = []
    for path in args.exclude_report:
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("diagnostic") != "plan10_prefix_value_calibration":
            raise ValueError(f"unexpected exclusion diagnostic in {path}")
        excluded_parents.update(
            str(row["parent_sample_id"]) for row in report["rows"]
        )
        exclude_contracts.append({"path": str(path), "sha256": sha256_file(path)})
    all_rows = [
        row
        for row in all_rows_before_exclusion
        if str(row.get("pairaug_parent_sample_id", row["sample_id"])) not in excluded_parents
    ]
    if not all_rows:
        parser.error("exclusion reports removed every swapped row")
    indexed_rows = list(enumerate(all_rows))[args.shard_index :: args.num_shards]
    rows = [row for _, row in indexed_rows]
    if not rows:
        parser.error("selected shard contains no swapped rows")
    config_result = load_qwen_config_with_compat(args.model, local_files_only=True)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    processor.image_processor.min_pixels = args.min_pixels
    processor.image_processor.max_pixels = args.max_pixels
    processor.tokenizer.padding_side = "left"
    styles = build_answer_token_styles(processor.tokenizer)
    full_generation = build_official_generation_config(
        max_new_tokens=1024,
        temperature=1.0,
        pad_token_id=processor.tokenizer.pad_token_id,
    )
    suffix_generation = build_official_generation_config(
        max_new_tokens=args.max_suffix_tokens,
        temperature=1.0,
        pad_token_id=processor.tokenizer.pad_token_id,
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
    print(
        "model_load_audit=" + json.dumps(validate_qwen_weight_tying(model), sort_keys=True),
        flush=True,
    )

    evaluated: list[dict[str, Any]] = []
    started = time.perf_counter()
    for local_row_index, (global_row_index, row) in enumerate(indexed_rows):
        image_path = args.images_root / str(row["image"])
        mapped_letter = str(row["mapped_answer_letter"]).upper()
        original_letter = str(row.get("pairaug_original_letter", row["answer_letter"])).upper()
        mapped_index = _option_index(mapped_letter)
        original_index = _option_index(original_letter)
        if mapped_index == original_index:
            raise ValueError(f"mapped and original coincide for {row['sample_id']}")
        prompt = build_chat_prompt(processor, image_path, official_prompt(dict(row)))
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
            base_inputs = processor(
                text=[prompt], images=[image], padding=True, return_tensors="pt"
            ).to(args.device)
        prompt_length = int(base_inputs["input_ids"].shape[1])
        prefix_rows: list[dict[str, Any]] = []
        for prefix_index, prefix_seed in enumerate(prefix_seeds):
            _set_seed(prefix_seed, args.device)
            with torch.inference_mode():
                generated = model.generate(**base_inputs, generation_config=full_generation)
            completion_ids = generated[:, prompt_length:]
            completion_mask = torch.ones_like(completion_ids)
            locations = locate_answer_candidate_tokens(completion_ids, completion_mask, styles)
            completion = processor.batch_decode(
                completion_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            parsed_full = clean_stage3_mca_text(completion).upper()
            valid = bool(locations.valid[0].item())
            prefix_out: dict[str, Any] = {
                "prefix_seed": int(prefix_seed),
                "valid": valid,
                "full_parsed_answer": parsed_full,
                "full_completion": completion,
            }
            if valid:
                option_position = int(locations.positions[0].item())
                style_index = int(locations.style_indices[0].item())
                style = styles[style_index]
                answer_start = option_position - len(style.prefix_ids)
                option_boundary_ids = generated[:, : prompt_length + option_position]
                boundary_inputs = _model_inputs_with_ids(base_inputs, option_boundary_ids)
                with torch.inference_mode():
                    logits = model(**boundary_inputs, use_cache=False).logits[0, -1].float()
                candidate_ids = torch.tensor(
                    style.candidate_token_ids, dtype=torch.long, device=logits.device
                )
                option_logits = logits[candidate_ids]
                option_probabilities = torch.softmax(option_logits, dim=0)
                margin = float((option_logits[mapped_index] - option_logits[original_index]).item())
                reasoning_ids = completion_ids[:, :answer_start]
                reasoning_prefix = processor.batch_decode(
                    reasoning_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
                fixed_prefix_ids = generated[:, : prompt_length + answer_start]
                suffix_inputs = _model_inputs_with_ids(base_inputs, fixed_prefix_ids)
                suffix_rollouts: list[dict[str, Any]] = []
                for suffix_index in range(args.suffix_samples):
                    suffix_seed = (
                        args.suffix_seed_base
                        + global_row_index * 10_000
                        + prefix_index * 100
                        + suffix_index
                    )
                    _set_seed(suffix_seed, args.device)
                    with torch.inference_mode():
                        suffix_generated = model.generate(
                            **suffix_inputs, generation_config=suffix_generation
                        )
                    suffix_ids = suffix_generated[:, fixed_prefix_ids.shape[1] :]
                    suffix_text = processor.batch_decode(
                        suffix_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )[0]
                    suffix_rollouts.append(
                        {
                            "seed": int(suffix_seed),
                            "parsed_answer": clean_stage3_mca_text(suffix_text).upper(),
                            "suffix": suffix_text,
                        }
                    )
                suffix_mapped_count = sum(
                    rollout["parsed_answer"] == mapped_letter for rollout in suffix_rollouts
                )
                prefix_out.update(
                    {
                        "style": style.name,
                        "reasoning_prefix": reasoning_prefix,
                        "mapped_letter": mapped_letter,
                        "original_letter": original_letter,
                        "margin_mapped_minus_original": margin,
                        "credit_t1": float(torch.sigmoid(torch.tensor(margin)).item()),
                        "credit_t2": float(torch.sigmoid(torch.tensor(margin / 2.0)).item()),
                        "option_probabilities": {
                            option_letter(index): float(option_probabilities[index].item())
                            for index in range(len(OPTION_LETTERS))
                        },
                        "suffix_mapped_count": suffix_mapped_count,
                        "suffix_mapped_rate": suffix_mapped_count / args.suffix_samples,
                        "suffix_rollouts": suffix_rollouts,
                    }
                )
            prefix_rows.append(prefix_out)
        valid_prefixes = [prefix for prefix in prefix_rows if prefix["valid"]]
        full_mapped_count = sum(
            prefix["full_parsed_answer"] == mapped_letter for prefix in prefix_rows
        )
        suffix_mapped_count = sum(
            int(prefix.get("suffix_mapped_count", 0)) for prefix in prefix_rows
        )
        evaluated.append(
            {
                "sample_id": row["sample_id"],
                "parent_sample_id": row.get("pairaug_parent_sample_id", row["sample_id"]),
                "scene_id": row["scene_id"],
                "answer_relation": row["answer_relation"],
                "stratum": relation_stratum(str(row["answer_relation"])),
                "mapped_letter": mapped_letter,
                "original_letter": original_letter,
                "all_wrong_full_rollouts": full_mapped_count == 0,
                "full_mapped_count": full_mapped_count,
                "valid_prefix_count": len(valid_prefixes),
                "suffix_mapped_count": suffix_mapped_count,
                "prefixes": prefix_rows,
            }
        )
        print(
            f"sample={row['sample_id']} {local_row_index + 1}/{len(rows)} "
            f"valid={len(valid_prefixes)}/{len(prefix_rows)} "
            f"full_mapped={full_mapped_count} suffix_mapped={suffix_mapped_count}",
            flush=True,
        )

    report = {
        "schema_version": 1,
        "diagnostic": "plan10_prefix_value_calibration",
        "status": "EXPLORATORY_NOT_CONFIRMATORY",
        "model": str(args.model),
        "model_identity": screening_model_identity(args.model),
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "ordered_swap_sample_ids": [str(row["sample_id"]) for row in rows],
        "deduplicate_parents": bool(args.deduplicate_parents),
        "candidate_swap_rows_before_exclusion": len(all_rows_before_exclusion),
        "candidate_swap_rows_after_exclusion_before_sharding": len(all_rows),
        "exclude_reports": exclude_contracts,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "prefix_seeds": list(prefix_seeds),
        "suffix_samples_per_prefix": args.suffix_samples,
        "suffix_seed_base": args.suffix_seed_base,
        "max_suffix_tokens": args.max_suffix_tokens,
        "credit": "sigmoid(mapped-minus-original boundary margin / T)",
        "temperature_note": "T=1 and T=2 are monotone and therefore rank-identical",
        "summary": _summarize(evaluated),
        "rows": evaluated,
        "elapsed_seconds": time.perf_counter() - started,
        "code_sha256": sha256_file(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(report["summary"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
