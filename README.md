# Where Do Counterfactual Gains Go?

Reproduction code for **Where Do Counterfactual Gains Go? Branch Allocation and Answer Supervision in Spatial VLMs** (ICASSP 2027 submission; no claim of acceptance).

Repository: https://github.com/KIRITO200212/counterfactual-gain-allocation

## Reproduction scope

1. ReG versus continued GRPO and original-only source–edit control, seeds 1234/20260910: 50 updates, 174 questions / 76 scenes.
2. Original matched source (S), crop exchange (X) and resample-return (N) pixel operators.
3. Independent answer-boundary study: continued GRPO, Prefix-only, Full; seed 2001; 40 updates on 21 questions/scenes.
4. CPU reconstruction of paired, scene-equal bootstrap estimates (10,000 resamples, seed 20270908), branch allocation, absolute candidate support, gross preference switches and local bridge outcomes.

The ReG gate uses legacy_closed_form_normalized: successful groups have approximately unit weight in the reported one-prompt microbatches. The bridge CE uses raw detached-credit softmax weights; reasoning advantages use group-standardized credits.

## Contents

- src/regcfpo/: original objectives, gate, prefix/token math, matched pixel edits, model adapter and training integration.
- scripts/train_stage4.py: original training entry. scripts/train_paper.py selects only the reported experiment settings.
- scripts/e3_mechanism_diagnostic.py: three-branch teacher-forced scoring.
- scripts/phase2_matched_pair_generation.py: matched X/N generation.
- scripts/plan10_prefix_value_calibration.py: own-prefix boundary and suffix evaluation.
- scripts/reproduce_statistics.py: numerical reproduction without GPU or photos.
- data/splits/: selected derived metadata; no images or complete dataset.
- data/numerical/: pseudonymous numeric observations; no free-form completions or annotator records.
- results/: full-precision figure/table values and expected analysis.
- provenance/source_files.json: hashes of byte-identical copied original sources.

Some preserved support modules have unused historical modes. Only the recipes here are claimed as the paper experiments.

## CPU quick start

Python 3.10 was used. Run from this directory:

    python -m venv .venv
    . .venv/bin/activate
    python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
    python -m pip install -r requirements-cpu.txt
    python -m pip install --no-deps -e .
    PYTHONPATH=src python -m pytest -q
    python scripts/reproduce_statistics.py
    python scripts/train_paper.py --study reg --arm reg --seed 1234

The last command prints a command only. Add --execute after acquiring dependencies, images and weights. CPU verification does not reproduce training or establish GPU equivalence.

## Acquire dependencies and data

Download separately under their upstream terms:

- Code: https://github.com/ZJU-REAL/SpatialLadder, commit 7a0d2ee85c28728835300310a349a53a15967f2e.
- Data: https://huggingface.co/datasets/hongxingli/SpatialLadder-26k, revision 41d36f87a67ff676d438417b5a09079add8de866.
- Model: https://huggingface.co/hongxingli/SpatialLadder-3B, revision 0819c3adf8827a2ea6c0348d49a23503ecb1f428.
- Base model family: Qwen2.5-VL-3B-Instruct.

    git clone https://github.com/ZJU-REAL/SpatialLadder.git vendor/SpatialLadder
    git -C vendor/SpatialLadder checkout 7a0d2ee85c28728835300310a349a53a15967f2e

Use huggingface_hub.snapshot_download with the pinned revisions. Put model files under models/SpatialLadder-3B, and unpack images into data/raw/spatialladder26k/images so scene.../...jpg paths resolve. Original hashes are in manifests/. Never store credentials in the repository.

The recorded upstream dataset/model cards identify Apache-2.0. Read actual upstream terms and underlying image-data obligations before any redistribution. Model weights, photos, full data archives and the upstream checkout are not included here.

The TF input has 447 pre-filtered rows; the original operator filter yields 291 accepted questions / 107 scenes. The bridge training schedule has 80 S/X rows (40 update pairs), including intentional repeated parents. Deduplicate parents for evaluation, not training.

Generate local bridge X images from separately acquired source images:

    PYTHONPATH=src python scripts/prepare_bridge_images.py

This uses the original operator and writes only ignored data/generated/ files.

## Training

The published runtime uses PyTorch 2.5.1+cu124, torchvision 0.20.1+cu124, transformers 4.49.0, SDPA, bfloat16, gradient checkpointing and one process. Training updates all model parameters including vision; no flash-attn or DeepSpeed execution is used.

Install CUDA wheels first, then requirements-training.txt. Install upstream using:

    python -m pip install --no-deps -e vendor/SpatialLadder/VLM-R1/src/open-r1-multimodal

The Qwen 4.53-to-4.49 configuration compatibility path is preserved.

Run six configurations via scripts/train_paper.py with --study reg, --arm reg/continued/local and --seed 1234/20260910. Run the bridge with --study bridge, --arm continued/prefix/full and --seed 2001. Add --execute to train; otherwise the exact command is printed. Fresh run directories are required.

Every arm starts from the initial model. Do not initialize the bridge from ReG. The wrapper delegates the original immutable run contracts and sets the historical smoke guard only for the published 40/50-update recipes.

## Evaluation

Set CKPT to a completed runs/.../final checkpoint. Commands:

    python scripts/e3_mechanism_diagnostic.py --model "$CKPT" --manifest data/splits/axis_tf447_before_operator_filter.jsonl --images-root data/raw/spatialladder26k/images --output results/new_tf.jsonl --device cuda:0 --attention sdpa --min-pixels 12544 --max-pixels 100352
    python scripts/phase2_matched_pair_generation.py --model "$CKPT" --manifest data/splits/aligned107.jsonl --images-root data/raw/spatialladder26k/images --output-dir results/new_generation --seeds 1701 1702 1703 1704 --device cuda:0 --attention sdpa --min-pixels 12544 --max-pixels 100352
    python scripts/plan10_prefix_value_calibration.py --model "$CKPT" --manifest data/splits/bridge_train21_schedule80.jsonl --images-root . --output results/new_bridge.json --prefix-seed 5301 --prefix-seed 6301 --prefix-seed 7301 --prefix-seed 8301 --suffix-samples 8 --suffix-seed-base 293001 --max-suffix-tokens 64 --deduplicate-parents

Check --help for instrument options. Boundary probabilities normalize over A–D, not the full vocabulary. Each arm generates its own prefixes. Scene-equal rates differ from pooled counts (suffixes: Full 43/664, Prefix-only 33/672, continued 33/664).

## Interpretation and limitations

Matched resampling controls processing history, not semantic validity. Full-cohort results concern protocol-defined edits; supported-axis results are narrower post hoc checks. Annotation forms and annotator identities are not shipped. The bridge is a separate local training-scene study with one optimization seed and does not establish a net cross-scene generation gain. Intervals condition on fitted checkpoints; training seeds are not pooled as test examples.

## License and citation

Original contributions are MIT licensed; upstream-derived portions retain Apache-2.0 and their notices. See LICENSE and THIRD_PARTY_NOTICES.md. CITATION.cff lists authors in manuscript order without inventing a DOI or acceptance. Exact source hashes and verification results are provided for inspection.
