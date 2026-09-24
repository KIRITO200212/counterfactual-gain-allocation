"""Stable fingerprints for local Hugging Face checkpoint weight shards."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_weight_files(model_path: Path) -> list[Path]:
    """Resolve the exact shard files named by a local checkpoint."""

    model_path = Path(model_path)
    index_candidates = (
        model_path / "model.safetensors.index.json",
        model_path / "pytorch_model.bin.index.json",
    )
    index = next((path for path in index_candidates if path.is_file()), None)
    if index is not None:
        payload = json.loads(index.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"checkpoint index has no non-empty weight_map: {index}")
        raw_names = list(weight_map.values())
        if not all(isinstance(name, str) and name for name in raw_names):
            raise ValueError(f"checkpoint index contains invalid shard names: {index}")
        names = sorted(set(raw_names))
        files = []
        for name in names:
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe checkpoint shard path {name!r} in {index}")
            files.append(model_path / relative)
    else:
        files = sorted(model_path.glob("*.safetensors"))
        if not files:
            files = sorted(model_path.glob("pytorch_model*.bin"))
    if not files:
        raise ValueError(f"no checkpoint weight files found under {model_path}")
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise ValueError(f"checkpoint weight files are missing: {missing}")
    return files


def checkpoint_weight_fingerprint(model_path: Path) -> dict[str, Any]:
    """Hash every weight shard and combine names, sizes, and shard hashes."""

    model_path = Path(model_path)
    combined = hashlib.sha256()
    records = []
    for path in checkpoint_weight_files(model_path):
        relative = path.relative_to(model_path).as_posix()
        size = path.stat().st_size
        shard_sha256 = _sha256_file(path)
        combined.update(relative.encode("utf-8"))
        combined.update(b"\0")
        combined.update(str(size).encode("ascii"))
        combined.update(b"\0")
        combined.update(shard_sha256.encode("ascii"))
        combined.update(b"\n")
        records.append(
            {"path": relative, "size_bytes": size, "sha256": shard_sha256}
        )
    return {
        "algorithm": "sha256(name\\0size\\0shard_sha256\\n)",
        "sha256": combined.hexdigest(),
        "files": records,
    }


__all__ = ["checkpoint_weight_files", "checkpoint_weight_fingerprint"]
