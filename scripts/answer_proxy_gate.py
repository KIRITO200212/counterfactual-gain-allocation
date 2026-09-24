#!/usr/bin/env python3
"""Answer-proxy consistency gate on viewed lineage rows.

Implements ``answer_proxy_gate_v2`` from configs/phase1_prereg.yaml
(amendment analysis_integrity_and_diagonal_disambiguation_20260820, plan4
section 8).  Compares four scoring arms on factual/swap/null images of a
seeded stratified sample of ~200 viewed rows (no new scenes):

1. answer-only single letter (frozen diagnostic template);
2. full option text, length-normalized;
3. full-template teacher-forced answer span (official CoT prompt,
   candidate ``<answer> X </answer>``), length-normalized;
4. full CoT rollout answer frequency (G=4, official generation config) as
   the behavioral reference distribution.

Gate thresholds: factual top-1 agreement >= 90%; swap/null directional sign
agreement >= 85%; Spearman rho >= 0.8.  Diagonal rows additionally report
four-candidate agreement.  All scores are length-normalized where indicated;
rollout answers are parsed with the frozen Stage-3 parser.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping

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

from frozen_operator_diagnostic import (  # noqa: E402
    candidate_suffix_mask,
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
    official_prompt,
)

SAMPLE_AXIS = 100
SAMPLE_DIAGONAL = 100
ROLLOUT_G = 4
SEED = 20270821


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def diagnostic_prompt(record: Mapping[str, Any]) -> str:
    options = "\n".join(str(option) for option in record["options"])
    return (
        f"Question: {record['question']}\nOptions:\n{options}\n"
        "Answer with only the option letter."
    )


def option_letter(index: int) -> str:
    return chr(ord("A") + index)


def count_rollout_predictions(
    completions: list[str], letters: list[str]
) -> dict[str, int]:
    """Count parsed answer letters case-insensitively over valid options.

    The Stage-3 parser lowercases the extracted answer, so predictions must
    be matched against the option letters case-insensitively; anything that
    does not parse to one of the provided option letters is ignored.
    """

    counts = {letter: 0 for letter in letters}
    upper_letters = {letter.upper(): letter for letter in letters}
    for completion in completions:
        predicted = clean_stage3_mca_text(completion).upper()
        if predicted in upper_letters:
            counts[upper_letters[predicted]] += 1
    return counts


def build_chat_prompt(processor, image_path: Path, text: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": text},
            ],
        }
    ]
    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def score_text_candidate(
    *, model, processor, prompt: str, candidate_text: str, image: Image.Image, device: str
) -> dict[str, Any]:
    """Teacher-forced score of an arbitrary candidate text (sum + length-norm)."""

    base_inputs = processor(text=[prompt], images=[image], padding=False, return_tensors="pt")
    full_inputs = processor(
        text=[prompt + candidate_text], images=[image], padding=False, return_tensors="pt"
    )
    base_ids = base_inputs["input_ids"].to(device)
    full_ids = full_inputs["input_ids"].to(device)
    mask = candidate_suffix_mask(base_ids, full_ids)
    candidate = prepare_canonical_candidate(
        processor=processor,
        prompt=prompt,
        candidate_name="proxy",
        candidate_text=candidate_text,
        image=image,
        device=device,
    )
    total = score_canonical_candidate(model, candidate)
    del candidate
    return {
        "logprob": total,
        "token_count": int(mask.sum().item()),
        "length_normalized": total / max(1, int(mask.sum().item())),
    }


def apply_edit(branch: str, image: Image.Image, record: Mapping[str, Any]):
    if branch == "factual":
        return image
    box_a, box_b = record["gt_box_a"], record["gt_box_b"]
    if branch == "pixel_pair_slot_swap":
        result = pixel_pair_slot_swap(image, box_a, box_b)
    else:
        result = canonical_resampling_return(image, box_a, box_b)
    if not result.accepted:
        return None
    return result.image


def rollout_arm(
    *,
    model,
    processor,
    image: Image.Image,
    prompt: str,
    letters: list[str],
    generation_config: GenerationConfig,
    device: str,
) -> dict[str, Any]:
    inputs = processor(
        text=[prompt], images=[image], padding=True, return_tensors="pt"
    ).to(device)
    prompt_length = inputs.input_ids.shape[1]
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            generation_config=generation_config,
            num_return_sequences=ROLLOUT_G,
        )
    completions = processor.batch_decode(
        generated[:, prompt_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    counts = count_rollout_predictions(completions, letters)
    return {
        "counts": counts,
        "frequency": {letter: counts[letter] / ROLLOUT_G for letter in letters},
        "answered_fraction": sum(counts.values()) / ROLLOUT_G,
        "format_valid_fraction": float(
            np.mean([stage3_format_valid(c) for c in completions])
        ),
        "completions": completions,
    }


def select_rows(
    axis_pool: list[dict[str, Any]],
    diagonal_pool: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    axis = sorted(axis_pool, key=lambda r: r["sample_id"])
    diagonal = sorted(diagonal_pool, key=lambda r: r["sample_id"])
    rng.shuffle(axis)
    rng.shuffle(diagonal)
    selected = axis[:SAMPLE_AXIS] + diagonal[:SAMPLE_DIAGONAL]
    selected.sort(key=lambda r: r["sample_id"])
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--axis-run", required=True, type=Path)
    parser.add_argument("--axis-manifest", required=True, type=Path)
    parser.add_argument("--diagonal-run", required=True, type=Path)
    parser.add_argument("--diagonal-manifest", required=True, type=Path)
    parser.add_argument("--images-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attention", choices=["sdpa", "eager"], default="sdpa")
    parser.add_argument("--min-pixels", type=int, default=12544)
    parser.add_argument("--max-pixels", type=int, default=100352)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA device requested but CUDA is unavailable")

    axis_manifest = {r["sample_id"]: r for r in read_jsonl(args.axis_manifest)}
    diagonal_manifest = {r["sample_id"]: r for r in read_jsonl(args.diagonal_manifest)}
    axis_pool = [
        dict(manifest_row, relation_class="axis")
        for row in read_jsonl(args.axis_run)
        if row["branches"]["pixel_pair_slot_swap"]["accepted"]
        and row["branches"]["canonical_resampling_return"]["accepted"]
        for manifest_row in [axis_manifest[row["sample_id"]]]
    ]
    diagonal_pool = [
        dict(manifest_row, relation_class="diagonal")
        for row in read_jsonl(args.diagonal_run)
        if all(
            row["branches"][b].get("accepted")
            for b in ("factual", "pixel_pair_slot_swap", "canonical_resampling_return")
        )
        for manifest_row in [diagonal_manifest[row["sample_id"]]]
    ]
    selected = select_rows(axis_pool, diagonal_pool)

    code_hashes = {
        path: sha256_file(PROJECT_ROOT / path)
        for path in (
            "scripts/answer_proxy_gate.py",
            "scripts/frozen_operator_diagnostic.py",
            "scripts/reward_saturation_audit.py",
            "src/regcfpo/operators/pixel_ops.py",
            "src/regcfpo/audit.py",
        )
    }
    ensure_run_contract(
        args.output.parent / f"{args.output.stem}.run_config.json",
        {
            "schema_version": 1,
            "diagnostic": "answer_proxy_gate_v2",
            "arms": [
                "answer_only_single_letter",
                "full_option_text_length_normalized",
                "full_template_answer_span_length_normalized",
                "cot_rollout_frequency_G4",
            ],
            "branches": ["factual", "pixel_pair_slot_swap", "canonical_resampling_return"],
            "sample": {"axis": SAMPLE_AXIS, "diagonal": SAMPLE_DIAGONAL, "seed": SEED},
            "rollout_g": ROLLOUT_G,
            "thresholds": {
                "factual_top1_agreement_min": 0.90,
                "directional_sign_agreement_min": 0.85,
                "spearman_min": 0.80,
            },
            "fallback_on_failure": "auxiliary loss uses full-template answer-span scores",
            "environment_isolation": "PYTHONNOUSERSITE=1",
            "model_path": str(args.model),
            "model_config_sha256": sha256_file(args.model / "config.json"),
            "axis_manifest_sha256": sha256_file(args.axis_manifest),
            "diagonal_manifest_sha256": sha256_file(args.diagonal_manifest),
            "axis_run_sha256": sha256_file(args.axis_run),
            "diagonal_run_sha256": sha256_file(args.diagonal_run),
            "ordered_sample_ids": [r["sample_id"] for r in selected],
            "device": args.device,
            "attention": args.attention,
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
            "max_new_tokens": args.max_new_tokens,
            "torch_version": torch.__version__,
            "code_sha256": code_hashes,
        },
        existing_artifacts=(args.output, args.summary),
    )

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
    torch.manual_seed(SEED)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(SEED)

    tmp_path = args.output.with_suffix(args.output.suffix + ".tmp")
    if tmp_path.exists():
        raise SystemExit(f"partial output {tmp_path} exists; refusing to continue")
    started = time.perf_counter()
    with tmp_path.open("w", encoding="utf-8") as sink:
        for index, record in enumerate(selected, start=1):
            image_path = args.images_root / record["image"]
            letters = [option_letter(i) for i in range(len(record["options"]))]
            answer_letter = record["answer_letter"]
            mapped_letter = record.get("mapped_answer_letter", answer_letter)
            row_out: dict[str, Any] = {
                "sample_id": record["sample_id"],
                "scene_id": record["scene_id"],
                "relation_class": record["relation_class"],
                "answer_letter": answer_letter,
                "mapped_answer_letter": mapped_letter,
                "branches": {},
            }
            with Image.open(image_path) as source:
                image = source.convert("RGB")
                for branch in (
                    "factual",
                    "pixel_pair_slot_swap",
                    "canonical_resampling_return",
                ):
                    edited = apply_edit(branch, image, record)
                    if edited is None:
                        row_out["branches"][branch] = {"accepted": False}
                        continue
                    diag_prompt_text = diagnostic_prompt(record)
                    template_prompt_text = official_prompt(record)
                    diag_prompt = build_chat_prompt(processor, image_path, diag_prompt_text)
                    template_prompt = build_chat_prompt(
                        processor, image_path, template_prompt_text
                    )
                    arms: dict[str, Any] = {}
                    for letter in letters:
                        idx = ord(letter) - ord("A")
                        arms[letter] = {
                            "answer_only": score_text_candidate(
                                model=model,
                                processor=processor,
                                prompt=diag_prompt,
                                candidate_text=letter,
                                image=edited,
                                device=args.device,
                            ),
                            "full_option_text": score_text_candidate(
                                model=model,
                                processor=processor,
                                prompt=diag_prompt,
                                candidate_text=record["options"][idx],
                                image=edited,
                                device=args.device,
                            ),
                            "full_template_span": score_text_candidate(
                                model=model,
                                processor=processor,
                                prompt=template_prompt,
                                candidate_text=f"<answer> {letter} </answer>",
                                image=edited,
                                device=args.device,
                            ),
                        }
                    rollouts = rollout_arm(
                        model=model,
                        processor=processor,
                        image=edited,
                        prompt=template_prompt,
                        letters=letters,
                        generation_config=generation_config,
                        device=args.device,
                    )
                    row_out["branches"][branch] = {
                        "accepted": True,
                        "arms": arms,
                        "rollout": rollouts,
                    }
                    if edited is not image:
                        edited.close()
            sink.write(json.dumps(row_out, ensure_ascii=False) + "\n")
            sink.flush()
            print(f"sample={record['sample_id']} {index}/{len(selected)}", flush=True)
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
