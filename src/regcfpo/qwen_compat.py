"""Narrow compatibility handling for Qwen2.5-VL checkpoint configs.

Transformers 4.53 writes a nested ``text_config`` for Qwen2.5-VL.  The
Transformers 4.49 class used by the pinned SpatialLadder environment predates
that composite schema: it accepts the unknown key through ``**kwargs`` and
leaves it as a plain ``dict``.  ``GenerationConfig.from_model_config`` later
calls ``to_dict()`` on it and fails.

This module repairs only that observed schema crossing.  It first uses normal
``AutoConfig`` dispatch.  A repair is considered only when a Qwen2.5-VL config
actually retained ``text_config`` as a raw dictionary.  Before dropping the
redundant nested object, the flattened and nested language architectures are
checked for conflicts.  ``tie_word_embeddings`` is promoted when absent from
the outer 4.53 config because it controls real weight sharing in 4.49.

The checkpoint's ``config.json`` is read but never modified.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Optional


QWEN_MODEL_TYPE = "qwen2_5_vl"
QWEN_TEXT_MODEL_TYPE = "qwen2_5_vl_text"

# These fields determine the decoder tensor shapes or RoPE interpretation and
# must be present in both the flattened 4.49 schema and nested 4.53 schema.
_REQUIRED_SHARED_FIELDS = (
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "hidden_act",
    "max_position_embeddings",
    "rms_norm_eps",
    "rope_theta",
    "rope_scaling",
)

# Compare these whenever both schemas explicitly provide them.  A missing
# optional value must not be replaced with an unrelated class default, except
# for the narrowly audited promotion list below.
_OPTIONAL_SHARED_FIELDS = (
    "attention_dropout",
    "initializer_range",
    "use_sliding_window",
    "max_window_layers",
    "bos_token_id",
    "eos_token_id",
    "pad_token_id",
    "attention_bias",
    "head_dim",
    "tie_word_embeddings",
    "use_cache",
)

# SpatialLadder's index has an embedding weight but no independent lm_head
# weight.  Therefore the nested value must be retained when a 4.53 config did
# not serialize it at the outer level.
_PROMOTE_IF_OUTER_MISSING = ("tie_word_embeddings",)


class QwenConfigCompatibilityError(ValueError):
    """Raised when a nested config cannot be proven safe to flatten."""


class QwenWeightTyingError(RuntimeError):
    """Raised when loaded weights violate the resolved Qwen config."""


@dataclass(frozen=True)
class QwenConfigLoadResult:
    """A loaded config plus a stable, JSON-serializable compatibility audit."""

    config: Any
    action: str
    config_path: str
    config_class: str
    checkpoint_transformers_version: Optional[str]
    verified_fields: tuple[str, ...] = ()
    promoted_fields: tuple[str, ...] = ()
    removed_fields: tuple[str, ...] = ()

    @property
    def compatibility_applied(self) -> bool:
        return self.action != "standard_auto_config"

    def to_record(self) -> dict[str, Any]:
        """Return the compatibility action for stdout and experiment records."""

        return {
            "action": self.action,
            "compatibility_applied": self.compatibility_applied,
            "config_path": self.config_path,
            "config_class": self.config_class,
            "checkpoint_transformers_version": self.checkpoint_transformers_version,
            "verified_fields": list(self.verified_fields),
            "promoted_fields": list(self.promoted_fields),
            "removed_fields": list(self.removed_fields),
        }


def _read_raw_config(model_path: Path) -> tuple[Path, dict[str, Any]]:
    config_path = model_path / "config.json" if model_path.is_dir() else model_path
    if config_path.name != "config.json" or not config_path.is_file():
        raise QwenConfigCompatibilityError(
            f"expected a local checkpoint config.json, got {config_path}"
        )
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise QwenConfigCompatibilityError(f"invalid JSON in {config_path}: {error}") from error
    if not isinstance(raw, dict):
        raise QwenConfigCompatibilityError(f"{config_path} must contain a JSON object")
    return config_path, raw


def _config_class_name(config: Any) -> str:
    cls = type(config)
    return f"{cls.__module__}.{cls.__qualname__}"


def _standard_result(config: Any, config_path: Path, raw: dict[str, Any]) -> QwenConfigLoadResult:
    return QwenConfigLoadResult(
        config=config,
        action="standard_auto_config",
        config_path=str(config_path),
        config_class=_config_class_name(config),
        checkpoint_transformers_version=raw.get("transformers_version"),
    )


def _validate_and_flatten_text_config(
    raw: dict[str, Any],
) -> tuple[dict[str, Any], tuple[str, ...], tuple[str, ...]]:
    nested = raw.get("text_config")
    if not isinstance(nested, dict):
        raise QwenConfigCompatibilityError("raw text_config must be a JSON object")
    nested_model_type = nested.get("model_type")
    if nested_model_type != QWEN_TEXT_MODEL_TYPE:
        raise QwenConfigCompatibilityError(
            "unexpected text_config.model_type: "
            f"expected {QWEN_TEXT_MODEL_TYPE!r}, got {nested_model_type!r}"
        )

    for field in _REQUIRED_SHARED_FIELDS:
        missing_locations = [
            location
            for location, values in (("outer", raw), ("text_config", nested))
            if field not in values
        ]
        if missing_locations:
            locations = ", ".join(missing_locations)
            raise QwenConfigCompatibilityError(
                f"cannot verify required architecture field {field!r}; missing from {locations}"
            )

    positive_integer_fields = (
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "max_position_embeddings",
    )
    for location, values in (("outer", raw), ("text_config", nested)):
        for field in positive_integer_fields:
            value = values[field]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise QwenConfigCompatibilityError(
                    f"{location}.{field} must be a positive integer, got {value!r}"
                )
        for field in ("use_sliding_window", "use_cache"):
            if field in values and not isinstance(values[field], bool):
                raise QwenConfigCompatibilityError(
                    f"{location}.{field} must be a boolean, got {values[field]!r}"
                )

    compared_fields: list[str] = []
    for field in _REQUIRED_SHARED_FIELDS + _OPTIONAL_SHARED_FIELDS:
        if field not in raw or field not in nested:
            continue
        if raw[field] != nested[field]:
            raise QwenConfigCompatibilityError(
                f"architecture field conflict for {field!r}: "
                f"outer={raw[field]!r}, text_config={nested[field]!r}"
            )
        compared_fields.append(field)

    rope_scaling = nested["rope_scaling"]
    if not isinstance(rope_scaling, dict):
        raise QwenConfigCompatibilityError("text_config.rope_scaling must be an object")
    mrope_section = rope_scaling.get("mrope_section")
    if (
        not isinstance(mrope_section, list)
        or len(mrope_section) != 3
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in mrope_section
        )
    ):
        raise QwenConfigCompatibilityError(
            "text_config.rope_scaling.mrope_section must contain three positive integers"
        )
    hidden_size = nested["hidden_size"]
    attention_heads = nested["num_attention_heads"]
    if hidden_size % attention_heads:
        raise QwenConfigCompatibilityError(
            "text_config.hidden_size must be divisible by num_attention_heads"
        )
    head_dim = hidden_size // attention_heads
    if 2 * sum(mrope_section) != head_dim:
        raise QwenConfigCompatibilityError(
            "text_config mrope_section is incompatible with attention head dimension: "
            f"2*sum({mrope_section!r}) != {head_dim}"
        )
    compared_fields.append("rope_scaling.mrope_section_dimension")

    layer_types = nested.get("layer_types")
    if layer_types is not None:
        expected_layer_types = ["full_attention"] * nested["num_hidden_layers"]
        if layer_types != expected_layer_types:
            raise QwenConfigCompatibilityError(
                "cannot flatten nontrivial text_config.layer_types into the 4.49 schema"
            )
        compared_fields.append("text_config.layer_types")

    # ``sliding_window`` is meaningful only when sliding-window attention is
    # enabled.  The released checkpoint has false/false with an inert legacy
    # outer value and nested null, so comparing those inactive values would be
    # a false conflict.  If either side enables it, both values must exist and
    # agree exactly.
    if bool(raw.get("use_sliding_window")) or bool(nested.get("use_sliding_window")):
        if "sliding_window" not in raw or "sliding_window" not in nested:
            raise QwenConfigCompatibilityError(
                "enabled sliding-window attention requires sliding_window in both schemas"
            )
        if raw["sliding_window"] != nested["sliding_window"]:
            raise QwenConfigCompatibilityError(
                "architecture field conflict for 'sliding_window': "
                f"outer={raw['sliding_window']!r}, "
                f"text_config={nested['sliding_window']!r}"
            )
        compared_fields.append("sliding_window")

    sanitized = copy.deepcopy(raw)
    promoted_fields: list[str] = []
    for field in _PROMOTE_IF_OUTER_MISSING:
        if field not in nested:
            raise QwenConfigCompatibilityError(
                f"text_config must declare {field!r} before it can be removed"
            )
        if field == "tie_word_embeddings" and not isinstance(nested[field], bool):
            raise QwenConfigCompatibilityError(
                "text_config.tie_word_embeddings must be a boolean"
            )
        if field not in sanitized:
            sanitized[field] = copy.deepcopy(nested[field])
            promoted_fields.append(field)

    sanitized.pop("text_config")
    return sanitized, tuple(compared_fields), tuple(promoted_fields)


def load_qwen_config_with_compat(
    model_path: str | Path,
    *,
    local_files_only: bool = True,
    auto_config_class: Any = None,
) -> QwenConfigLoadResult:
    """Load a local config, repairing only the audited 4.53→4.49 Qwen case.

    Normal configs are returned directly from ``AutoConfig.from_pretrained``.
    The compatibility path is entered only when all of the following hold:

    * raw and loaded model types are ``qwen2_5_vl``;
    * raw ``text_config`` is a dictionary; and
    * normal AutoConfig dispatch retained ``config.text_config`` as a dict.

    ``auto_config_class`` is injectable so unit tests do not need model weights
    or a particular Transformers release.
    """

    path = Path(model_path)
    config_path, raw = _read_raw_config(path)
    if auto_config_class is None:
        from transformers import AutoConfig

        auto_config_class = AutoConfig
    config = auto_config_class.from_pretrained(path, local_files_only=local_files_only)

    is_qwen = raw.get("model_type") == QWEN_MODEL_TYPE and getattr(
        config, "model_type", None
    ) == QWEN_MODEL_TYPE
    retained_raw_text_config = isinstance(getattr(config, "text_config", None), dict)
    if not (is_qwen and isinstance(raw.get("text_config"), dict) and retained_raw_text_config):
        return _standard_result(config, config_path, raw)

    sanitized, verified_fields, promoted_fields = _validate_and_flatten_text_config(raw)
    config_class = type(config)
    if not hasattr(config_class, "from_dict"):
        raise QwenConfigCompatibilityError(
            f"{_config_class_name(config)} cannot reconstruct a sanitized config"
        )
    repaired = config_class.from_dict(sanitized)
    if isinstance(getattr(repaired, "text_config", None), dict):
        raise QwenConfigCompatibilityError(
            "sanitized Qwen config still retained text_config as a raw dictionary"
        )
    if hasattr(config, "_name_or_path"):
        repaired._name_or_path = config._name_or_path
    for field in promoted_fields:
        if getattr(repaired, field, None) != sanitized[field]:
            raise QwenConfigCompatibilityError(
                f"reconstructed config did not preserve promoted field {field!r}"
            )

    return QwenConfigLoadResult(
        config=repaired,
        action="flattened_raw_text_config_for_transformers_4_49",
        config_path=str(config_path),
        config_class=_config_class_name(repaired),
        checkpoint_transformers_version=raw.get("transformers_version"),
        verified_fields=verified_fields,
        promoted_fields=promoted_fields,
        removed_fields=("text_config",),
    )


def validate_qwen_weight_tying(model: Any) -> dict[str, Any]:
    """Audit the loaded input/output embedding relationship.

    The check belongs on every real loading path rather than only in a one-off
    smoke test.  A checkpoint with tied embeddings may omit ``lm_head.weight``;
    silently loading it under ``tie_word_embeddings=False`` would otherwise
    leave a large randomly initialized output head.
    """

    if not hasattr(model, "config"):
        raise QwenWeightTyingError("loaded model has no config")
    config = model.config
    if hasattr(config, "get_text_config"):
        text_config = config.get_text_config(decoder=True)
    else:
        text_config = config
    if isinstance(text_config, dict):
        raise QwenWeightTyingError("loaded model still has a raw dictionary text_config")
    expected = getattr(text_config, "tie_word_embeddings", None)
    if not isinstance(expected, bool):
        raise QwenWeightTyingError("config.tie_word_embeddings must be a boolean")

    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    if input_embeddings is None or output_embeddings is None:
        raise QwenWeightTyingError("loaded model must expose input and output embeddings")
    input_weight = input_embeddings.weight
    output_weight = output_embeddings.weight
    tied = input_weight.data_ptr() == output_weight.data_ptr()
    if expected and not tied:
        raise QwenWeightTyingError(
            "config requires tied word embeddings, but lm_head and input embeddings "
            "have different storage"
        )

    unique_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    return {
        "tie_word_embeddings_expected": expected,
        "input_output_embeddings_tied": tied,
        "input_embedding_shape": list(input_weight.shape),
        "output_embedding_shape": list(output_weight.shape),
        "unique_parameter_count": unique_parameter_count,
    }


__all__ = [
    "QwenConfigCompatibilityError",
    "QwenConfigLoadResult",
    "QwenWeightTyingError",
    "load_qwen_config_with_compat",
    "validate_qwen_weight_tying",
]
