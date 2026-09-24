# Third-party notices

## SpatialLadder / VLM-R1

Dependency: https://github.com/ZJU-REAL/SpatialLadder
Pinned commit: 7a0d2ee85c28728835300310a349a53a15967f2e.

VLM-R1/LICENSE and VLM-R1/src/open-r1-multimodal/LICENSE specify Apache-2.0. A verbatim copy is provided in LICENSES/Apache-2.0.txt. The upstream checkout is acquired separately.

Ported/adapted upstream behavior includes the official Stage-3 prompt, multiple-choice/format reward handling, numeric-answer reward (documented in src/regcfpo/training/rewards.py and scripts/e3_rollout_eval.py), GRPO grouping contracts and model integration. Existing provenance comments are retained. Upstream-derived portions remain Apache-2.0; the top-level MIT license applies only to original contributions and does not relicense these portions.

Modifications add ReG auxiliary objectives, matched pixel edits, prefix/bridge supervision, experiment contracts and analyses. Original scientific source copies are byte-identical to the research implementation; provenance/source_files.json records their identity.

## Data/model

Selected data/splits/ metadata derive from SpatialLadder-26k at the revision in manifests/data_manifest.json. The recorded upstream model/data cards state Apache-2.0. Original data archives, images and model weights are not redistributed. Underlying image sources may have separate obligations; obtain them under upstream terms.

## Runtime

PyTorch, torchvision, NumPy, Pillow, transformers, TRL, Accelerate, datasets and other dependencies are installed separately and retain their own licenses. Version pinning does not transfer their copyrights.
