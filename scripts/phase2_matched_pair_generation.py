#!/usr/bin/env python3
"""Phase 2: same-target generation endpoint on the frozen 107-scene subset.

Why this exists: the Case-A behavioral readout used the *factual* arm, but the
factual arm is a preservation measure -- keeping it flat is the design goal, so it
cannot show whether the edited branch improved in free generation.  This driver
measures the target endpoint directly.

It reuses the frozen Phase-1 generation path verbatim
(``screening_behavior_eval.process_record_behavior`` ->
``answer_proxy_gate_v3.apply_edit`` / ``generate_rollouts``: official Stage-3 CoT
prompt, official generation config, one completion per fixed seed, swap and
resampling-null branches, teacher-forced arms regenerated per leg).  The only
deliberate deviation is G=4 instead of G=16, recorded in the run contract.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoProcessor
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from frozen_operator_diagnostic import read_jsonl  # noqa: E402
from screening_behavior_eval import (  # noqa: E402
    NEWLY_GENERATED_BRANCHES,
    build_official_generation_config,
    process_record_behavior,
)

from regcfpo.qwen_compat import (  # noqa: E402
    load_qwen_config_with_compat,
    validate_qwen_weight_tying,
)
from regcfpo.run_contract import ensure_run_contract  # noqa: E402

DIAGNOSTIC = "phase2_same_target_generation_v1"
DEFAULT_SEEDS = (1701, 1702, 1703, 1704)
CODE_IDENTITY = (
    "scripts/screening_behavior_eval.py",
    "scripts/answer_proxy_gate_v3.py",
    "scripts/frozen_operator_diagnostic.py",
    "src/regcfpo/operators/pixel_ops.py",
    "src/regcfpo/qwen_adapter.py",
)


def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--manifest", type=Path,
                    default=PROJECT_ROOT / "manifests/generated/phase2_same_target107.jsonl")
    ap.add_argument("--images-root", type=Path,
                    default=PROJECT_ROOT / "data/raw/spatialladder26k/images")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--attention", choices=("sdpa", "eager"), default="sdpa")
    ap.add_argument("--min-pixels", type=int, default=12544)
    ap.add_argument("--max-pixels", type=int, default=100352)
    args = ap.parse_args()

    if len(set(args.seeds)) != len(args.seeds) or not args.seeds:
        raise SystemExit("seeds must be distinct and non-empty")
    records = [dict(r) for r in read_jsonl(args.manifest)]
    ordered_ids = [r["sample_id"] for r in records]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "rows.jsonl"
    tmp = args.output_dir / "rows.jsonl.tmp"
    contract = args.output_dir / "run_config.json"

    ensure_run_contract(contract, {
        "schema_version": 1,
        "diagnostic": DIAGNOSTIC,
        "purpose": "same-target (edit-branch) free-generation endpoint for the "
                   "re-derived matched pair",
        "deviation_from_frozen_screening_driver": (
            f"G={len(args.seeds)} instead of G=16; identical prompt, generation "
            "config, operators and per-seed streaming otherwise"),
        "branches": list(NEWLY_GENERATED_BRANCHES),
        "seeds": list(args.seeds),
        "model_path": str(args.model),
        "manifest_path": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "ordered_sample_ids": ordered_ids,
        "min_pixels": args.min_pixels, "max_pixels": args.max_pixels,
        "attention": args.attention, "device": args.device,
        "torch_version": torch.__version__,
        "code_sha256": {p: sha256_file(PROJECT_ROOT / p) for p in CODE_IDENTITY},
    }, existing_artifacts=(out, tmp))

    if out.exists():
        print(f"complete output {out} exists; skipping")
        return 0

    done = set()
    if tmp.exists():
        done = {r.get("sample_id") for r in read_jsonl(tmp)}
        print(f"resuming: {len(done)}/{len(ordered_ids)}")

    config_result = load_qwen_config_with_compat(args.model, local_files_only=True)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    processor.image_processor.min_pixels = args.min_pixels
    processor.image_processor.max_pixels = args.max_pixels
    processor.tokenizer.padding_side = "left"
    generation_config = build_official_generation_config(
        max_new_tokens=1024, temperature=1.0,
        pad_token_id=processor.tokenizer.pad_token_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, config=config_result.config, local_files_only=True,
        torch_dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
        attn_implementation=args.attention, low_cpu_mem_usage=True,
        device_map={"": args.device}).eval()
    print("model_load_audit=" + json.dumps(validate_qwen_weight_tying(model),
                                           sort_keys=True), flush=True)

    started = time.perf_counter()
    with tmp.open("a" if done else "w", encoding="utf-8") as sink:
        for i, record in enumerate(records, start=1):
            if record["sample_id"] in done:
                continue
            row = process_record_behavior(
                model=model, processor=processor, record=record,
                images_root=args.images_root,
                letters=[chr(ord("A") + k) for k in range(len(record["options"]))],
                generation_config=generation_config, seeds=args.seeds,
                device=args.device)
            sink.write(json.dumps(row) + "\n")
            sink.flush()
            if i % 10 == 0 or i == len(records):
                print(f"{i}/{len(records)} {record['sample_id']} "
                      f"elapsed={time.perf_counter() - started:.0f}s", flush=True)
    tmp.replace(out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
