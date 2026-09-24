#!/usr/bin/env python3
"""E3 mechanism diagnostic with the frozen full-template training proxy.

Scores the pixel-edit mechanism graph (factual / pixel_pair_slot_swap /
canonical_resampling_return, plus a canonical no-op identity check) with the
full-template answer-span candidates that the trained ReG-CFPO objective
consumes (trainer ``CANDIDATE_TEXT_TEMPLATE = "<answer> {letter}
</answer>"`` under the official Stage-3 CoT prompt).  Measured on the
confirmed lineage (447 confirmation rows; holdout never touched), so the E3
mechanism comparison uses the same quantity the training optimized.

Output schema matches ``analyze_pixel_confirmation.py`` input expectations
(branches with accepted / reject_reason / s_orig / s_mapped /
directional_gap); the canonical no-op must reproduce factual scores within
1e-5 or the run aborts fail-closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import torch
from PIL import Image
from transformers import AutoProcessor
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
)

from regcfpo.operators.pixel_ops import (
    canonical_noop,
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

from frozen_operator_diagnostic import (  # noqa: E402
    prepare_canonical_candidate,
    read_jsonl,
    score_canonical_candidate,
)
from reward_saturation_audit import official_prompt  # noqa: E402

BRANCHES = (
    "factual",
    "canonical_noop",
    "pixel_pair_slot_swap",
    "canonical_resampling_return",
)
CANDIDATE_TEMPLATE = "<answer> {letter} </answer>"
NOOP_THRESHOLD = 1e-5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def apply_branch(branch: str, image: Image.Image, record: Mapping[str, Any]):
    if branch == "factual":
        return image, None
    box_a, box_b = record["gt_box_a"], record["gt_box_b"]
    if branch == "canonical_noop":
        result = canonical_noop(image)
    elif branch == "pixel_pair_slot_swap":
        result = pixel_pair_slot_swap(image, box_a, box_b)
    elif branch == "canonical_resampling_return":
        result = canonical_resampling_return(image, box_a, box_b)
    else:
        raise ValueError(f"unknown branch {branch!r}")
    if not result.accepted:
        return None, {"reject_reason": result.reject_reason, "metadata": result.metadata}
    return result.image, None


def score_branch(
    *,
    model,
    processor,
    prompt: str,
    image: Image.Image,
    record: Mapping[str, Any],
    device: str,
) -> dict[str, Any]:
    scores: dict[str, float] = {}
    for name, letter in (
        ("original", record["answer_letter"]),
        ("mapped", record["mapped_answer_letter"]),
    ):
        candidate = prepare_canonical_candidate(
            processor=processor,
            prompt=prompt,
            candidate_name=name,
            candidate_text=CANDIDATE_TEMPLATE.format(letter=letter),
            image=image,
            device=device,
        )
        token_count = len(candidate.token_ids)
        if token_count <= 0:
            raise RuntimeError(f"empty candidate token span for {name!r}")
        scores[name] = score_canonical_candidate(model, candidate) / token_count
        del candidate
    return {
        "accepted": True,
        "reject_reason": "accepted",
        "s_orig": scores["original"],
        "s_mapped": scores["mapped"],
        "directional_gap": scores["mapped"] - scores["original"],
    }


def run_sample(
    *,
    model,
    processor,
    record: Mapping[str, Any],
    images_root: Path,
    device: str,
) -> dict[str, Any]:
    image_path = images_root / record["image"]
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": official_prompt(record)},
            ],
        }
    ]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    started = time.perf_counter()
    branches: dict[str, Any] = {}
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        for branch in BRANCHES:
            edited, rejection = apply_branch(branch, image, record)
            if edited is None:
                branches[branch] = {"accepted": False, **rejection}
                continue
            branches[branch] = score_branch(
                model=model,
                processor=processor,
                prompt=prompt,
                image=edited,
                record=record,
                device=device,
            )
            if edited is not image:
                edited.close()
    factual = branches["factual"]
    noop = branches["canonical_noop"]
    if noop.get("accepted"):
        for key in ("s_orig", "s_mapped"):
            error = abs(noop[key] - factual[key])
            if error > NOOP_THRESHOLD:
                raise RuntimeError(
                    f"canonical no-op diverged from factual by {error} "
                    f"on {record['sample_id']}"
                )
    return {
        "sample_id": record["sample_id"],
        "scene_id": record["scene_id"],
        "image": record["image"],
        "branches": branches,
        "wall_seconds": time.perf_counter() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--images-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attention", choices=["sdpa", "eager"], default="sdpa")
    parser.add_argument("--min-pixels", type=int, default=12544)
    parser.add_argument("--max-pixels", type=int, default=100352)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")

    selected = [dict(row) for row in read_jsonl(args.manifest)]
    if not selected:
        parser.error("manifest is empty")

    tmp_path = args.output.with_suffix(args.output.suffix + ".tmp")
    if tmp_path.exists():
        raise SystemExit(f"partial output {tmp_path} exists; refusing to continue")
    if args.output.exists():
        print(f"output {args.output} complete; skipping")
        return 0

    ensure_run_contract(
        args.output.parent / f"{args.output.stem}.run_config.json",
        {
            "schema_version": 2,
            "audit": "e3_mechanism_diagnostic_full_template_mean_v2",
            "model_path": str(args.model),
            "model_config_sha256": sha256_file(args.model / "config.json"),
            "manifest_path": str(args.manifest),
            "manifest_sha256": sha256_file(args.manifest),
            "branches": list(BRANCHES),
            "candidate_template": CANDIDATE_TEMPLATE,
            "score_aggregation": "mean_per_candidate_token",
            "prompt_policy": "official_stage3_cot_template",
            "noop_threshold": NOOP_THRESHOLD,
            "device": args.device,
            "attention": args.attention,
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
            "ordered_sample_ids": [row["sample_id"] for row in selected],
            "code_sha256": {
                path: sha256_file(PROJECT_ROOT / path)
                for path in (
                    "scripts/e3_mechanism_diagnostic.py",
                    "scripts/frozen_operator_diagnostic.py",
                    "scripts/reward_saturation_audit.py",
                    "src/regcfpo/operators/pixel_ops.py",
                    "src/regcfpo/qwen_adapter.py",
                )
            },
            "holdout_rule": "final holdout is never part of any E3 manifest",
        },
        existing_artifacts=(),
    )

    config_result = load_qwen_config_with_compat(args.model, local_files_only=True)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    processor.image_processor.min_pixels = args.min_pixels
    processor.image_processor.max_pixels = args.max_pixels
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
    with tmp_path.open("w", encoding="utf-8") as sink:
        for index, record in enumerate(selected, start=1):
            row = run_sample(
                model=model,
                processor=processor,
                record=record,
                images_root=args.images_root,
                device=args.device,
            )
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            sink.flush()
            print(
                f"sample={row['sample_id']} {index}/{len(selected)}", flush=True
            )
    tmp_path.replace(args.output)
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
