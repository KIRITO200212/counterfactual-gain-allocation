#!/usr/bin/env python3
"""Print or execute exactly one published training configuration."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]

def command(study, arm, seed, model, run_dir):
    common = [sys.executable, str(ROOT / "scripts/train_stage4.py"),
        "--model-path", str(model), "--run-dir", str(run_dir),
        "--seed", str(seed), "--num-generations", "4",
        "--per-device-train-batch-size", "4", "--num-iterations", "1",
        "--learning-rate", "1e-6", "--max-completion-length", "1024",
        "--attn-implementation", "sdpa", "--min-pixels", "12544",
        "--max-pixels", "100352", "--gradient-checkpointing", "--no-deepspeed",
        "--logging-steps", "1", "--save-total-limit", "1"]
    if study == "reg":
        if arm not in ("reg", "continued", "local") or seed not in (1234, 20260910):
            raise ValueError("ReG study requires reg/continued/local and seed 1234/20260910")
        common += ["--data-file", "data/splits/reg_train174.jsonl",
            "--image-root", "data/raw/spatialladder26k/images",
            "--gradient-accumulation-steps", "1", "--lr-scheduler-type", "linear",
            "--warmup-steps", "0", "--beta", "0", "--max-grad-norm", "1",
            "--max-steps", "50", "--save-steps", "25"]
        if arm == "continued":
            return common + ["--objective", "continued_grpo"]
        return common + ["--objective", "regcfpo" if arm == "reg" else "local_nondirectional_cfpo",
            "--gamma", "0.0176" if arm == "reg" else "0.0104",
            "--lambda-null", "0.5" if arm == "reg" else "0.0",
            "--margin-dir", "1.0", "--margin-null", "1.0", "--weight-clip", "4.0",
            "--gate-credit-mode", "legacy_closed_form_normalized"]
    if study != "bridge" or arm not in ("continued", "prefix", "full") or seed != 2001:
        raise ValueError("Bridge study requires continued/prefix/full and seed 2001")
    common += ["--data-file", "data/splits/bridge_train21_schedule80.jsonl",
        "--image-root", ".", "--gradient-accumulation-steps", "2",
        "--lr-scheduler-type", "constant", "--beta", "0.01", "--max-grad-norm", "10",
        "--max-steps", "40", "--save-steps", "100000", "--no-training-checkpoints"]
    if arm == "continued":
        return common + ["--objective", "continued_grpo"]
    return common + ["--objective", "pairaug_refgain_trust",
        "--gamma", "0.005", "--gate-credit-mode", "monotonic_k_over_g_raw",
        "--joint-gain-delta", "1.0", "--joint-gain-weight", "0.0",
        "--joint-trust-weight", "0.1", "--joint-source-kl-slack", "0.02",
        "--joint-null-kl-slack", "0.02", "--joint-prefix-credit-weight", "1.0",
        "--joint-prefix-credit-temperature", "2.0",
        "--joint-prefix-credit-scope", "reasoning_all_wrong",
        "--joint-answer-bridge-temperature", "1.0",
        "--joint-answer-bridge-weight", "0.01" if arm == "full" else "0.0"]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", choices=["reg","bridge"], required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--model", type=Path, default=Path("models/SpatialLadder-3B"))
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--execute", action="store_true")
    a=p.parse_args()
    out=a.run_dir or Path("runs")/f"{a.study}_{a.arm}_s{a.seed}"
    cmd=command(a.study,a.arm,a.seed,a.model,out)
    print(json.dumps(cmd, indent=2))
    if a.execute:
        if (ROOT/out).exists():
            raise SystemExit("Use a fresh run directory; implicit resume is forbidden.")
        env=dict(os.environ,TRAINING_SMOKE_AUTHORIZED="true",PYTHONNOUSERSITE="1")
        subprocess.run(cmd,cwd=ROOT,env=env,check=True)
if __name__=="__main__":
    main()
