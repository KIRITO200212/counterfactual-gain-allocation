"""Data contracts for the RelationPair-v2 diagnostic set.

The module deliberately contains no dataset- or framework-specific code.  It
provides the small, strict contract shared by data extraction, offline audits,
and training-manifest construction.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "RelationPair-v2"
SUPPORTED_RELATIONS = frozenset({"left", "right"})
INVERSE_RELATION = {"left": "right", "right": "left"}


class RelationPairValidationError(ValueError):
    """Raised when a RelationPair-v2 record violates the frozen contract."""


class OptionRelationMappingError(RelationPairValidationError):
    """Raised when answer options do not define a unique relation mapping."""


class SceneSplitLeakageError(RelationPairValidationError):
    """Raised when one scene occurs in more than one dataset split."""

    def __init__(self, overlaps: Mapping[str, Sequence[str]]) -> None:
        normalized = {
            scene: tuple(sorted(split_names))
            for scene, split_names in sorted(overlaps.items())
        }
        self.overlaps = normalized
        details = ", ".join(
            f"{scene!r} in {list(split_names)!r}"
            for scene, split_names in normalized.items()
        )
        super().__init__(f"scene split leakage detected: {details}")


_OPTION_PREFIX_RE = re.compile(
    r"^\s*(?:\(?[A-Z]\)?\s*[.):\-]|[A-Z]\s+)\s*", re.IGNORECASE
)
_LEFT_RE = re.compile(r"(?<![a-z])left(?:(?:\s+|-)of)?(?![\w-])", re.IGNORECASE)
_RIGHT_RE = re.compile(r"(?<![a-z])right(?:(?:\s+|-)of)?(?![\w-])", re.IGNORECASE)
_DIAGONAL_RE = re.compile(
    r"\b(?:upper|lower|top|bottom|north|south)[\s-]+(?:left|right)\b"
    r"|\b(?:left|right)[\s-]+(?:upper|lower|top|bottom|north|south)\b",
    re.IGNORECASE,
)
_NEGATED_DIRECTION_RE = re.compile(
    r"\b(?:not|never|neither|without)\b(?:\W+\w+){0,3}\W+(?:left|right)\b",
    re.IGNORECASE,
)
_SPATIALLADDER_TRAJECTORY_RE = re.compile(r"^(scene\d+)_\d+$")


def spatialladder_base_scene_id(image: str) -> str:
    """Return the underlying 3D scene, not a trajectory/view directory.

    Released paths use directories such as ``scene0000_00`` and
    ``scene0000_02`` for distinct camera trajectories through the same
    environment.  Treating the complete directory as a split key leaks one
    environment across splits, so RelationPair manifests use ``scene0000``.
    """

    image_path = _require_nonempty_string("image", image)
    directory, separator, filename = image_path.partition("/")
    if not separator or not filename:
        raise RelationPairValidationError(
            "SpatialLadder image must have '<scene>_<trajectory>/<frame>' form"
        )
    match = _SPATIALLADDER_TRAJECTORY_RE.fullmatch(directory)
    if match is None:
        raise RelationPairValidationError(
            f"unrecognized SpatialLadder scene/trajectory directory {directory!r}"
        )
    return match.group(1)


def normalize_relation(relation: str) -> str:
    """Return a canonical left/right relation or raise a validation error."""

    if not isinstance(relation, str):
        raise RelationPairValidationError("relation must be a string")
    canonical = relation.strip().lower().replace("_", " ").replace("-", " ")
    canonical = " ".join(canonical.split())
    aliases = {
        "left": "left",
        "left of": "left",
        "to the left": "left",
        "to the left of": "left",
        "right": "right",
        "right of": "right",
        "to the right": "right",
        "to the right of": "right",
    }
    try:
        return aliases[canonical]
    except KeyError as exc:
        raise RelationPairValidationError(
            f"unsupported relation {relation!r}; expected left/right"
        ) from exc


def inverse_relation(relation: str) -> str:
    """Return the frozen inverse relation (left <-> right)."""

    return INVERSE_RELATION[normalize_relation(relation)]


def option_letter(index: int) -> str:
    """Map a zero-based option index to its single-letter label."""

    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < 26:
        raise OptionRelationMappingError("option index must be an integer in [0, 25]")
    return chr(ord("A") + index)


def normalize_option_letter(letter: str) -> str:
    """Canonicalize an answer label such as ``"(a)"`` to ``"A"``."""

    if not isinstance(letter, str):
        raise OptionRelationMappingError("answer letter must be a string")
    stripped = letter.strip().upper()
    match = re.fullmatch(r"\(?([A-Z])\)?[.):]?", stripped)
    if match is None:
        raise OptionRelationMappingError(f"invalid answer letter {letter!r}")
    return match.group(1)


def parse_option_relation(option: str) -> str | None:
    """Parse a left/right answer option.

    Options for unrelated relations are returned as ``None``.  An option that
    mentions both left and right is rejected instead of being guessed, because
    a guess could silently corrupt the counterfactual answer mapping.
    """

    if not isinstance(option, str) or not option.strip():
        raise OptionRelationMappingError("each answer option must be a non-empty string")
    text = _OPTION_PREFIX_RE.sub("", option.strip()).replace("_", " ")
    # A diagonal or explicitly negated direction is a valid unrelated option,
    # not evidence for the pure binary relation used by RelationPair-v2.
    if _DIAGONAL_RE.search(text) or _NEGATED_DIRECTION_RE.search(text):
        return None
    has_left = _LEFT_RE.search(text) is not None
    has_right = _RIGHT_RE.search(text) is not None
    if has_left and has_right:
        raise OptionRelationMappingError(
            f"ambiguous option mentions both left and right: {option!r}"
        )
    if has_left:
        return "left"
    if has_right:
        return "right"
    return None


def resolve_option_relations(options: Sequence[str]) -> dict[str, str]:
    """Return the recognized ``option letter -> relation`` mapping.

    The function rejects duplicate mappings such as two distinct options that
    both mean ``left``.  Unrelated answer choices are allowed.
    """

    if isinstance(options, (str, bytes)) or not isinstance(options, Sequence):
        raise OptionRelationMappingError("options must be a sequence of strings")
    if not 2 <= len(options) <= 26:
        raise OptionRelationMappingError("options must contain between 2 and 26 choices")

    by_letter: dict[str, str] = {}
    by_relation: dict[str, str] = {}
    for index, option in enumerate(options):
        relation = parse_option_relation(option)
        if relation is None:
            continue
        letter = option_letter(index)
        previous = by_relation.get(relation)
        if previous is not None:
            raise OptionRelationMappingError(
                f"relation {relation!r} is mapped by both options {previous} and {letter}"
            )
        by_relation[relation] = letter
        by_letter[letter] = relation
    return by_letter


def relation_to_option_letters(options: Sequence[str]) -> dict[str, str]:
    """Return the unique ``relation -> option letter`` mapping."""

    return {relation: letter for letter, relation in resolve_option_relations(options).items()}


def _require_nonempty_string(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RelationPairValidationError(f"{name} must be a non-empty string")
    return value.strip()


def _normalize_box(name: str, box: Sequence[float]) -> tuple[float, float, float, float]:
    if isinstance(box, (str, bytes)) or not isinstance(box, Sequence) or len(box) != 4:
        raise RelationPairValidationError(f"{name} must contain [x1, y1, x2, y2]")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in box):
        raise RelationPairValidationError(f"{name} coordinates must be real numbers")
    normalized = tuple(float(value) for value in box)
    if not all(math.isfinite(value) for value in normalized):
        raise RelationPairValidationError(f"{name} coordinates must be finite")
    x1, y1, x2, y2 = normalized
    if x2 <= x1 or y2 <= y1:
        raise RelationPairValidationError(f"{name} must have positive width and height")
    return normalized


@dataclass(frozen=True, slots=True)
class RelationPairV2:
    """Validated immutable representation of one RelationPair-v2 example."""

    sample_id: str
    scene_id: str
    image: str
    question: str
    options: tuple[str, ...]
    answer_relation: str
    mapped_relation: str
    answer_letter: str
    mapped_answer_letter: str
    entity_a: str
    entity_b: str
    gt_box_a: tuple[float, float, float, float]
    gt_box_b: tuple[float, float, float, float]
    operator_valid: bool
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in ("sample_id", "scene_id", "image", "question", "entity_a", "entity_b"):
            object.__setattr__(
                self, field_name, _require_nonempty_string(field_name, getattr(self, field_name))
            )

        image_directory = self.image.partition("/")[0]
        spatialladder_match = _SPATIALLADDER_TRAJECTORY_RE.fullmatch(image_directory)
        if spatialladder_match is not None and self.scene_id != spatialladder_match.group(1):
            raise RelationPairValidationError(
                "scene_id must identify the base SpatialLadder environment, not its "
                f"trajectory directory: expected {spatialladder_match.group(1)!r}"
            )

        if self.entity_a == self.entity_b:
            raise RelationPairValidationError("entity_a and entity_b must be distinct")
        if self.schema_version != SCHEMA_VERSION:
            raise RelationPairValidationError(
                f"schema_version must be exactly {SCHEMA_VERSION!r}"
            )
        if not isinstance(self.operator_valid, bool):
            raise RelationPairValidationError("operator_valid must be a boolean")

        if isinstance(self.options, (str, bytes)) or not isinstance(self.options, Sequence):
            raise OptionRelationMappingError("options must be a sequence of strings")
        normalized_options = tuple(self.options)
        # resolve_option_relations also validates every option and uniqueness.
        relation_letters = relation_to_option_letters(normalized_options)
        object.__setattr__(self, "options", normalized_options)

        answer_relation = normalize_relation(self.answer_relation)
        mapped_relation = normalize_relation(self.mapped_relation)
        if mapped_relation != inverse_relation(answer_relation):
            raise RelationPairValidationError(
                "mapped_relation must be the left/right inverse of answer_relation"
            )
        object.__setattr__(self, "answer_relation", answer_relation)
        object.__setattr__(self, "mapped_relation", mapped_relation)

        answer_letter = normalize_option_letter(self.answer_letter)
        mapped_answer_letter = normalize_option_letter(self.mapped_answer_letter)
        if answer_letter == mapped_answer_letter:
            raise OptionRelationMappingError(
                "answer_letter and mapped_answer_letter must be distinct"
            )
        object.__setattr__(self, "answer_letter", answer_letter)
        object.__setattr__(self, "mapped_answer_letter", mapped_answer_letter)

        for relation, declared_letter, field_name in (
            (answer_relation, answer_letter, "answer_letter"),
            (mapped_relation, mapped_answer_letter, "mapped_answer_letter"),
        ):
            resolved_letter = relation_letters.get(relation)
            if resolved_letter is None:
                raise OptionRelationMappingError(
                    f"options contain no uniquely parseable {relation!r} answer"
                )
            if resolved_letter != declared_letter:
                raise OptionRelationMappingError(
                    f"{field_name}={declared_letter!r} but {relation!r} resolves to "
                    f"option {resolved_letter!r}"
                )

        object.__setattr__(self, "gt_box_a", _normalize_box("gt_box_a", self.gt_box_a))
        object.__setattr__(self, "gt_box_b", _normalize_box("gt_box_b", self.gt_box_b))

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "RelationPairV2":
        """Validate and construct a record from a JSON-compatible mapping."""

        if not isinstance(record, Mapping):
            raise RelationPairValidationError("record must be a mapping")
        required = {
            "sample_id",
            "scene_id",
            "image",
            "question",
            "options",
            "answer_relation",
            "mapped_relation",
            "answer_letter",
            "mapped_answer_letter",
            "entity_a",
            "entity_b",
            "gt_box_a",
            "gt_box_b",
            "operator_valid",
        }
        allowed = required | {"schema_version"}
        missing = sorted(required - set(record))
        unknown = sorted(set(record) - allowed)
        if missing:
            raise RelationPairValidationError(f"missing required fields: {missing}")
        if unknown:
            raise RelationPairValidationError(f"unknown RelationPair-v2 fields: {unknown}")
        kwargs = dict(record)
        kwargs.setdefault("schema_version", SCHEMA_VERSION)
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible record with a stable field layout."""

        return {
            "schema_version": self.schema_version,
            "sample_id": self.sample_id,
            "scene_id": self.scene_id,
            "image": self.image,
            "question": self.question,
            "options": list(self.options),
            "answer_relation": self.answer_relation,
            "mapped_relation": self.mapped_relation,
            "answer_letter": self.answer_letter,
            "mapped_answer_letter": self.mapped_answer_letter,
            "entity_a": self.entity_a,
            "entity_b": self.entity_b,
            "gt_box_a": list(self.gt_box_a),
            "gt_box_b": list(self.gt_box_b),
            "operator_valid": self.operator_valid,
        }


RelationPairLike = RelationPairV2 | Mapping[str, Any]


def coerce_relation_pair(record: RelationPairLike) -> RelationPairV2:
    """Return ``record`` as a validated :class:`RelationPairV2`."""

    if isinstance(record, RelationPairV2):
        return record
    return RelationPairV2.from_dict(record)


def validate_relation_pairs(records: Iterable[RelationPairLike]) -> tuple[RelationPairV2, ...]:
    """Validate a collection and reject duplicate sample identifiers."""

    validated: list[RelationPairV2] = []
    seen_sample_ids: set[str] = set()
    for record in records:
        pair = coerce_relation_pair(record)
        if pair.sample_id in seen_sample_ids:
            raise RelationPairValidationError(
                f"duplicate sample_id {pair.sample_id!r} in relation-pair collection"
            )
        seen_sample_ids.add(pair.sample_id)
        validated.append(pair)
    return tuple(validated)


def scene_split_overlaps(
    split_records: Mapping[str, Iterable[RelationPairLike]],
) -> dict[str, tuple[str, ...]]:
    """Return scenes appearing in two or more named splits."""

    if not isinstance(split_records, Mapping) or not split_records:
        raise RelationPairValidationError("split_records must be a non-empty mapping")
    scene_splits: dict[str, set[str]] = {}
    for raw_split_name, records in split_records.items():
        split_name = _require_nonempty_string("split name", raw_split_name)
        pairs = validate_relation_pairs(records)
        for pair in pairs:
            scene_splits.setdefault(pair.scene_id, set()).add(split_name)
    return {
        scene: tuple(sorted(split_names))
        for scene, split_names in sorted(scene_splits.items())
        if len(split_names) > 1
    }


def validate_scene_disjoint(
    split_records: Mapping[str, Iterable[RelationPairLike]],
) -> None:
    """Enforce the preregistered zero scene-crossing split invariant."""

    overlaps = scene_split_overlaps(split_records)
    if overlaps:
        raise SceneSplitLeakageError(overlaps)


__all__ = [
    "INVERSE_RELATION",
    "SCHEMA_VERSION",
    "SUPPORTED_RELATIONS",
    "OptionRelationMappingError",
    "RelationPairV2",
    "RelationPairValidationError",
    "SceneSplitLeakageError",
    "coerce_relation_pair",
    "inverse_relation",
    "normalize_option_letter",
    "normalize_relation",
    "option_letter",
    "parse_option_relation",
    "relation_to_option_letters",
    "resolve_option_relations",
    "scene_split_overlaps",
    "spatialladder_base_scene_id",
    "validate_relation_pairs",
    "validate_scene_disjoint",
]
