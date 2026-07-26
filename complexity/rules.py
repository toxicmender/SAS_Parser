"""Rule sets: which SAS construct implies which complexity tier and which
translation parity, for a given target language. See complexity/README.md.

The catalogue itself is **data**, not code: it lives in JSON profiles under
``complexity/profiles/`` and is loaded here into :class:`RuleSet`. That is what
makes the analysis retargetable — ``sparksql.json`` rates constructs against
Spark SQL, ``pyspark.json`` extends it with the handful of ratings that differ
when a full Python host language is available, and an operator can point
``complexity.rules_path`` at a file of their own without touching this package.

Profile resolution, highest precedence first:

1. an explicit ``path`` argument to :func:`load_ruleset`;
2. an explicit ``target`` argument (a name under ``profiles/``);
3. ``complexity.rules_path`` in config.json;
4. ``complexity.target`` in config.json;
5. :data:`DEFAULT_TARGET`.

A profile may ``extends`` another by name; the child's entries are deep-merged
over the parent's, so a derived target states only its differences.

Tier assignment follows the project's brief:

- **LOW** — simple SQL and macro variables.
- **MEDIUM** — hashing, MERGE, SFTP, mail, and similar "works, but the
  semantics differ" constructs.
- **HIGH** — arrays, DO loops, and ``%MACRO`` definitions.

Every catalogue is an **allowlist**. A construct with no entry contributes no
signal at all, which floors a chunk at LOW/DIRECT. Silence means "nothing
notable found", never "unknown": an unrecognised function must not inflate a
score.

Logger name: ``complexity.rules``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import app_config

from .models import ComplexityTier, TShirtSize, TranslationParity

logger = logging.getLogger(__name__)

PROFILE_DIR = Path(__file__).parent / "profiles"
DEFAULT_TARGET = "sparksql"
_CONFIG_SECTION = "complexity"

# Fallback per-tier score weights, used when a profile omits ``weights``.
# Weight only ranks units within a tier — it can never change the tier itself,
# which is presence-based (see complexity.analyzer).
WEIGHT_LOW = 1.0
WEIGHT_MEDIUM = 2.5
WEIGHT_HIGH = 5.0

_DEFAULT_WEIGHTS: dict[ComplexityTier, float] = {
    ComplexityTier.LOW: WEIGHT_LOW,
    ComplexityTier.MEDIUM: WEIGHT_MEDIUM,
    ComplexityTier.HIGH: WEIGHT_HIGH,
}

# The construct namespaces a profile may define. Each maps to the metadata (or
# detector) dimension the analyzer looks up; an unknown key in a profile is an
# error rather than a silently ignored typo.
CONSTRUCT_KINDS: frozenset[str] = frozenset(
    {
        "proc",
        "component_object",
        "function",
        "call_routine",
        "global_statement",
        "kind",
        "detector",
        "cross_file",
    }
)

# Fallback per-parity score weights for the *complexity* sizing dimension.
# Parity is what makes a size target-dependent: two files with identical tiers
# are not equal work if one's constructs are MANUAL against the target and the
# other's are DIRECT.
_DEFAULT_PARITY_WEIGHTS: dict[TranslationParity, float] = {
    TranslationParity.DIRECT: 0.0,
    TranslationParity.SUPPORTED: 0.5,
    TranslationParity.PARTIAL: 1.5,
    TranslationParity.HARD: 3.0,
    TranslationParity.MANUAL: 5.0,
}

# Default sizing model. Every number here is calibrated *together* with the
# anchor below and is meaningless apart from it, which is why the whole block
# lives in one place and ships as profile data.
DEFAULT_ANCHOR_RAW = 18.0

_DEFAULT_SIZE_SCALE: dict[TShirtSize, float] = {
    TShirtSize.SMALL: 2.0,
    TShirtSize.MEDIUM: 3.0,
    TShirtSize.LARGE: 5.0,
    TShirtSize.EXTRA_LARGE: 8.0,
}

# Upper bounds in points, at the geometric midpoints of the Fibonacci rungs
# (2|3 -> 2.5, 3|5 -> 4.0, 5|8 -> 6.5). Anything above the last is EXTRA_LARGE.
_DEFAULT_SIZE_BANDS: dict[TShirtSize, float] = {
    TShirtSize.SMALL: 2.5,
    TShirtSize.MEDIUM: 4.0,
    TShirtSize.LARGE: 6.5,
}

# Effort-dimension volume weights.
_DEFAULT_VOLUME_WEIGHTS: dict[str, float] = {
    "chunk": 0.5,
    "line": 0.01,
    "step": 1.0,
    "io": 1.0,
    "param": 0.5,
}

# Uncertainty-dimension weights: what the analysis could not pin down.
_DEFAULT_UNCERTAINTY_WEIGHTS: dict[str, float] = {
    "unresolved_ref": 3.0,
    "unclosed_block": 4.0,
    "unknown_chunk": 2.0,
    "diagnostic": 1.5,
}


@dataclass(frozen=True)
class SizeModel:
    """How raw effort becomes a :class:`TShirtSize`. See complexity/README.md.

    Sizing is **relative estimation against a fixed anchor**, which is the
    method's own prescription — you size a story by comparing it to a reference
    story the team knows. The anchor is fixed profile data rather than
    something recomputed per run: a corpus-relative (percentile) scheme would
    re-rate the same file differently depending on which files it was analysed
    alongside, and would be undefined for a single-file run.

    Attributes
    ----------
    anchor_raw
        Raw score of the reference **Medium** file. Everything is measured
        against it: ``points = raw / anchor_raw * scale[MEDIUM]``.
    anchor_describes
        Prose description of that reference file, so the calibration can be
        argued with instead of merely obeyed.
    scale
        Nominal points per rung (Fibonacci by default).
    bands
        Inclusive upper bound in points for SMALL, MEDIUM, and LARGE; above the
        LARGE bound is EXTRA_LARGE.
    volume, uncertainty
        Per-unit weights for the effort and uncertainty dimensions.
    parity_weights
        Per-parity weights for the complexity dimension.
    min_size_by_kind
        Floors: a file containing this ``SasChunkKind`` is never smaller than
        the given size. Applied after banding.
    min_size_by_parity
        The same, keyed on the file's worst parity. Ships empty — the lever
        exists, but chunk kind proved the better discriminator.
    """

    anchor_raw: float = DEFAULT_ANCHOR_RAW
    anchor_describes: str = ""
    scale: dict[TShirtSize, float] = field(
        default_factory=lambda: dict(_DEFAULT_SIZE_SCALE)
    )
    bands: dict[TShirtSize, float] = field(
        default_factory=lambda: dict(_DEFAULT_SIZE_BANDS)
    )
    volume: dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_VOLUME_WEIGHTS)
    )
    uncertainty: dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_UNCERTAINTY_WEIGHTS)
    )
    parity_weights: dict[TranslationParity, float] = field(
        default_factory=lambda: dict(_DEFAULT_PARITY_WEIGHTS)
    )
    min_size_by_kind: dict[str, TShirtSize] = field(default_factory=dict)
    min_size_by_parity: dict[TranslationParity, TShirtSize] = field(
        default_factory=dict
    )

    def points_for(self, raw: float) -> float:
        """Convert a raw score to points on the anchored Fibonacci scale."""
        if self.anchor_raw <= 0:
            return 0.0
        medium = self.scale.get(TShirtSize.MEDIUM, 3.0)
        return round(raw / self.anchor_raw * medium, 2)

    def band_for(self, points: float) -> TShirtSize:
        """The size *points* falls in, before any floor is applied."""
        for size in (TShirtSize.SMALL, TShirtSize.MEDIUM, TShirtSize.LARGE):
            if points <= self.bands.get(size, _DEFAULT_SIZE_BANDS[size]):
                return size
        return TShirtSize.EXTRA_LARGE

    def volume_weight(self, key: str) -> float:
        """Effort weight for *key*, falling back to the built-in default."""
        return self.volume.get(key, _DEFAULT_VOLUME_WEIGHTS.get(key, 0.0))

    def uncertainty_weight(self, key: str) -> float:
        """Uncertainty weight for *key*, falling back to the built-in default."""
        return self.uncertainty.get(key, _DEFAULT_UNCERTAINTY_WEIGHTS.get(key, 0.0))

    def parity_weight(self, parity: TranslationParity) -> float:
        """Complexity-dimension weight for *parity*."""
        return self.parity_weights.get(parity, _DEFAULT_PARITY_WEIGHTS[parity])


class RuleSetError(ValueError):
    """A rule-set profile is missing, malformed, or self-contradictory.

    Always raised rather than degraded past: a profile that does not parse
    means the operator asked for a classification scheme that cannot be
    applied, which should stop the analysis instead of silently scoring
    everything against a different one.
    """


@dataclass(frozen=True, slots=True)
class SignalSpec:
    """The classification attached to one recognised construct.

    Carries no weight of its own: weight is a property of the *tier*, supplied
    by the owning :class:`RuleSet`, so a spec can never disagree with its tier
    about how much it is worth.
    """

    category: str
    tier: ComplexityTier
    parity: TranslationParity
    note: str = ""


@dataclass(frozen=True)
class RuleSet:
    """A complete construct catalogue for one target language.

    Attributes
    ----------
    target
        Profile name, e.g. ``"sparksql"``.
    display_name
        Human-readable target, e.g. ``"Spark SQL"`` — used in reports.
    description
        What this profile rates and on what basis.
    weights
        Per-tier score weights.
    constructs
        ``{construct_kind: {name: SignalSpec}}`` for the kinds in
        :data:`CONSTRUCT_KINDS`. Names are lowercased except under ``"kind"``,
        whose keys are ``SasChunkKind`` values.
    flags
        ``(metadata_attribute, signal_name, spec)`` triples for the boolean /
        list-valued metadata flags.
    sizes
        The T-shirt sizing model — anchor, scale, bands, and the per-dimension
        weights. Lives per-target because parity feeds the complexity
        dimension, so the same file is genuinely less work against a target
        with a procedural host language.
    """

    target: str
    display_name: str
    description: str
    weights: dict[ComplexityTier, float]
    constructs: dict[str, dict[str, SignalSpec]]
    flags: tuple[tuple[str, str, SignalSpec], ...]
    sizes: SizeModel = field(default_factory=SizeModel)

    def spec(self, construct_kind: str, name: str) -> SignalSpec | None:
        """The spec for *name* in *construct_kind*, or ``None`` if unlisted."""
        return self.constructs.get(construct_kind, {}).get(name.lower())

    def weight_for(self, tier: ComplexityTier) -> float:
        """Score weight for *tier*."""
        return self.weights.get(tier, _DEFAULT_WEIGHTS[tier])

    @property
    def construct_count(self) -> int:
        """Total catalogued constructs, flags included."""
        return sum(len(v) for v in self.constructs.values()) + len(self.flags)

    def __str__(self) -> str:
        return (
            f"RuleSet(target={self.target!r}, display_name={self.display_name!r}, "
            f"constructs={self.construct_count})"
        )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _enum(
    raw: Any,
    enum: type[ComplexityTier] | type[TranslationParity] | type[TShirtSize],
    where: str,
) -> Any:
    """Parse *raw* into *enum*, naming *where* it came from on failure."""
    if not isinstance(raw, str):
        raise RuleSetError(f"{where}: expected a string, got {type(raw).__name__}")
    try:
        return enum(raw.upper())
    except ValueError:
        valid = ", ".join(m.value for m in enum)
        raise RuleSetError(
            f"{where}: {raw!r} is not a valid {enum.__name__} (expected one of {valid})"
        ) from None


def _spec(entry: Any, where: str) -> SignalSpec:
    """Parse one ``{category, tier, parity, note}`` object."""
    if not isinstance(entry, dict):
        raise RuleSetError(f"{where}: expected an object, got {type(entry).__name__}")
    missing = [k for k in ("category", "tier", "parity") if k not in entry]
    if missing:
        raise RuleSetError(f"{where}: missing required key(s) {', '.join(missing)}")
    category = entry["category"]
    if not isinstance(category, str) or not category.strip():
        raise RuleSetError(f"{where}: 'category' must be a non-empty string")
    note = entry.get("note", "")
    if not isinstance(note, str):
        raise RuleSetError(f"{where}: 'note' must be a string")
    return SignalSpec(
        category=category,
        tier=_enum(entry["tier"], ComplexityTier, f"{where}.tier"),
        parity=_enum(entry["parity"], TranslationParity, f"{where}.parity"),
        note=note,
    )


def _number(raw: Any, where: str) -> float:
    """Parse *raw* as a number, rejecting bools (which are ints in Python)."""
    if not isinstance(raw, (int, float)) or isinstance(raw, bool):
        raise RuleSetError(f"{where}: expected a number, got {type(raw).__name__}")
    return float(raw)


def _weight_map(
    raw: Any, where: str, defaults: dict[str, float]
) -> dict[str, float]:
    """Parse a ``{key: number}`` block, rejecting keys with no known meaning.

    An unknown key is an error rather than a no-op: a misspelled weight that is
    silently dropped would leave the operator believing they had retuned
    something they had not.
    """
    out = dict(defaults)
    if raw is None:
        return out
    if not isinstance(raw, dict):
        raise RuleSetError(f"{where}: expected an object")
    for key, value in raw.items():
        if key not in defaults:
            raise RuleSetError(
                f"{where}.{key}: unknown key "
                f"(expected one of {', '.join(sorted(defaults))})"
            )
        out[key] = _number(value, f"{where}.{key}")
    return out


def _size_model(raw: Any, where: str) -> SizeModel:
    """Parse the optional ``sizes`` block into a :class:`SizeModel`."""
    if raw is None:
        return SizeModel()
    if not isinstance(raw, dict):
        raise RuleSetError(f"{where}: expected an object")

    anchor_raw = DEFAULT_ANCHOR_RAW
    describes = ""
    anchor = raw.get("anchor")
    if anchor is not None:
        if not isinstance(anchor, dict):
            raise RuleSetError(f"{where}.anchor: expected an object")
        if "raw" in anchor:
            anchor_raw = _number(anchor["raw"], f"{where}.anchor.raw")
            if anchor_raw <= 0:
                raise RuleSetError(
                    f"{where}.anchor.raw: must be greater than 0 "
                    f"(every size is measured against it)"
                )
        describes = str(anchor.get("describes", ""))

    scale = dict(_DEFAULT_SIZE_SCALE)
    for key, value in (raw.get("scale") or {}).items():
        size = _enum(key, TShirtSize, f"{where}.scale key")
        scale[size] = _number(value, f"{where}.scale.{key}")

    bands = dict(_DEFAULT_SIZE_BANDS)
    for key, value in (raw.get("bands") or {}).items():
        size = _enum(key, TShirtSize, f"{where}.bands key")
        if size is TShirtSize.EXTRA_LARGE:
            raise RuleSetError(
                f"{where}.bands.{key}: EXTRA_LARGE is the open-ended top band "
                f"and takes no upper bound"
            )
        bands[size] = _number(value, f"{where}.bands.{key}")
    ordered = [bands[s] for s in (TShirtSize.SMALL, TShirtSize.MEDIUM, TShirtSize.LARGE)]
    if ordered != sorted(ordered) or len(set(ordered)) != len(ordered):
        raise RuleSetError(
            f"{where}.bands: bounds must strictly increase "
            f"SMALL < MEDIUM < LARGE, got {ordered}"
        )

    parity_weights = dict(_DEFAULT_PARITY_WEIGHTS)
    for key, value in (raw.get("parity_weights") or {}).items():
        parity = _enum(key, TranslationParity, f"{where}.parity_weights key")
        parity_weights[parity] = _number(value, f"{where}.parity_weights.{key}")

    min_by_kind: dict[str, TShirtSize] = {}
    for key, value in (raw.get("min_size_by_kind") or {}).items():
        min_by_kind[key] = _enum(value, TShirtSize, f"{where}.min_size_by_kind.{key}")

    min_by_parity: dict[TranslationParity, TShirtSize] = {}
    for key, value in (raw.get("min_size_by_parity") or {}).items():
        parity = _enum(key, TranslationParity, f"{where}.min_size_by_parity key")
        min_by_parity[parity] = _enum(
            value, TShirtSize, f"{where}.min_size_by_parity.{key}"
        )

    return SizeModel(
        anchor_raw=anchor_raw,
        anchor_describes=describes,
        scale=scale,
        bands=bands,
        volume=_weight_map(
            raw.get("volume"), f"{where}.volume", _DEFAULT_VOLUME_WEIGHTS
        ),
        uncertainty=_weight_map(
            raw.get("uncertainty"), f"{where}.uncertainty", _DEFAULT_UNCERTAINTY_WEIGHTS
        ),
        parity_weights=parity_weights,
        min_size_by_kind=min_by_kind,
        min_size_by_parity=min_by_parity,
    )


def _expand_groups(doc: dict[str, Any], where: str) -> dict[str, dict[str, Any]]:
    """Expand the optional ``construct_groups`` shorthand.

    A group attaches one classification to many names at once::

        {"kind": "function", "names": ["md5", "sha256"],
         "category": "hashing", "tier": "MEDIUM", "parity": "SUPPORTED"}

    which keeps a profile readable when a whole function family shares a
    rating. Groups are applied *before* the explicit ``constructs`` map, so a
    single named entry can still override its group.
    """
    groups = doc.get("construct_groups", [])
    if not isinstance(groups, list):
        raise RuleSetError(f"{where}.construct_groups: expected a list")
    out: dict[str, dict[str, Any]] = {}
    for i, group in enumerate(groups):
        at = f"{where}.construct_groups[{i}]"
        if not isinstance(group, dict):
            raise RuleSetError(f"{at}: expected an object")
        kind = group.get("kind")
        if kind not in CONSTRUCT_KINDS:
            raise RuleSetError(
                f"{at}.kind: {kind!r} is not a construct kind "
                f"(expected one of {', '.join(sorted(CONSTRUCT_KINDS))})"
            )
        names = group.get("names")
        if not isinstance(names, list) or not names:
            raise RuleSetError(f"{at}.names: expected a non-empty list")
        body = {k: v for k, v in group.items() if k not in ("kind", "names")}
        for name in names:
            if not isinstance(name, str):
                raise RuleSetError(f"{at}.names: entries must be strings")
            out.setdefault(kind, {})[name.lower()] = body
    return out


def _merge_raw(
    base: dict[str, Any], child: dict[str, Any]
) -> dict[str, Any]:
    """Deep-merge a child profile's document over its parent's.

    ``constructs`` merges per construct-kind and per construct name, so a child
    overriding one function leaves the rest of the family intact. ``flags``
    merges by signal name. ``sizes`` merges one level down, so a child can
    restate just its anchor and inherit every weight. Scalars and ``weights``
    keys replace outright.
    """
    merged: dict[str, Any] = {**base, **{
        k: v for k, v in child.items()
        if k not in ("constructs", "flags", "weights", "construct_groups", "sizes")
    }}

    merged["weights"] = {**base.get("weights", {}), **child.get("weights", {})}

    # `sizes` sub-blocks merge individually: a child restating only
    # `anchor` must not silently discard the parent's band and weight tuning.
    base_s: dict[str, Any] = base.get("sizes", {}) or {}
    child_s: dict[str, Any] = child.get("sizes", {}) or {}
    if base_s or child_s:
        sizes: dict[str, Any] = {**base_s}
        for key, value in child_s.items():
            if isinstance(value, dict) and isinstance(sizes.get(key), dict):
                sizes[key] = {**sizes[key], **value}
            else:
                sizes[key] = value
        merged["sizes"] = sizes

    base_c: dict[str, Any] = base.get("constructs", {})
    child_c: dict[str, Any] = child.get("constructs", {})
    constructs: dict[str, Any] = {k: dict(v) for k, v in base_c.items()}
    for kind, entries in child_c.items():
        constructs.setdefault(kind, {}).update(entries)
    merged["constructs"] = constructs

    by_name: dict[str, Any] = {f["name"]: f for f in base.get("flags", []) if "name" in f}
    order: list[str] = list(by_name)
    for flag in child.get("flags", []):
        name = flag.get("name")
        if name is None:
            raise RuleSetError("flags: every entry needs a 'name'")
        if name in by_name:
            by_name[name] = {**by_name[name], **flag}
        else:
            by_name[name] = flag
            order.append(name)
    merged["flags"] = [by_name[n] for n in order]
    return merged


def _read_profile_doc(
    path: Path, _seen: tuple[str, ...] = ()
) -> dict[str, Any]:
    """Read a profile JSON document, resolving ``extends`` recursively."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuleSetError(f"cannot read rule-set profile '{path}': {exc}") from exc
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuleSetError(f"rule-set profile '{path}' is not valid JSON: {exc}") from exc
    if not isinstance(doc, dict):
        raise RuleSetError(f"rule-set profile '{path}': top level must be an object")

    # Group shorthand expands before inheritance, so a parent's groups and a
    # child's named entries compose the way the file reads.
    groups = _expand_groups(doc, path.name)
    if groups:
        constructs = {k: dict(v) for k, v in groups.items()}
        for kind, entries in doc.get("constructs", {}).items():
            constructs.setdefault(kind, {}).update(entries)
        doc = {**doc, "constructs": constructs}

    parent_name = doc.get("extends")
    if parent_name is None:
        return doc
    if not isinstance(parent_name, str):
        raise RuleSetError(f"rule-set profile '{path}': 'extends' must be a string")
    target = doc.get("target", path.stem)
    if parent_name in _seen or parent_name == target:
        chain = " -> ".join([*_seen, str(target), parent_name])
        raise RuleSetError(f"rule-set profile '{path}': circular extends chain {chain}")
    parent_path = PROFILE_DIR / f"{parent_name}.json"
    if not parent_path.is_file():
        raise RuleSetError(
            f"rule-set profile '{path}' extends unknown profile {parent_name!r} "
            f"(looked for {parent_path})"
        )
    parent_doc = _read_profile_doc(parent_path, (*_seen, str(target)))
    return _merge_raw(parent_doc, doc)


def _ruleset_from_doc(doc: dict[str, Any], where: str) -> RuleSet:
    """Validate a resolved profile document into a :class:`RuleSet`."""
    target = doc.get("target")
    if not isinstance(target, str) or not target.strip():
        raise RuleSetError(f"{where}: 'target' must be a non-empty string")

    weights = dict(_DEFAULT_WEIGHTS)
    raw_weights = doc.get("weights", {})
    if not isinstance(raw_weights, dict):
        raise RuleSetError(f"{where}.weights: expected an object")
    for key, value in raw_weights.items():
        tier = _enum(key, ComplexityTier, f"{where}.weights key")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise RuleSetError(f"{where}.weights.{key}: expected a number")
        weights[tier] = float(value)

    raw_constructs = doc.get("constructs", {})
    if not isinstance(raw_constructs, dict):
        raise RuleSetError(f"{where}.constructs: expected an object")
    unknown = set(raw_constructs) - CONSTRUCT_KINDS
    if unknown:
        raise RuleSetError(
            f"{where}.constructs: unknown construct kind(s) "
            f"{', '.join(sorted(unknown))} "
            f"(expected one of {', '.join(sorted(CONSTRUCT_KINDS))})"
        )
    constructs: dict[str, dict[str, SignalSpec]] = {}
    for kind, entries in raw_constructs.items():
        if not isinstance(entries, dict):
            raise RuleSetError(f"{where}.constructs.{kind}: expected an object")
        # Chunk-kind keys are SasChunkKind values (upper case); every other
        # namespace is looked up by a lowercased identifier.
        constructs[kind] = {
            (name if kind == "kind" else name.lower()): _spec(
                entry, f"{where}.constructs.{kind}.{name}"
            )
            for name, entry in entries.items()
        }

    raw_flags = doc.get("flags", [])
    if not isinstance(raw_flags, list):
        raise RuleSetError(f"{where}.flags: expected a list")
    flags: list[tuple[str, str, SignalSpec]] = []
    for i, entry in enumerate(raw_flags):
        at = f"{where}.flags[{i}]"
        if not isinstance(entry, dict):
            raise RuleSetError(f"{at}: expected an object")
        attr = entry.get("attr")
        name = entry.get("name")
        if not isinstance(attr, str) or not attr:
            raise RuleSetError(f"{at}.attr: expected a non-empty string")
        if not isinstance(name, str) or not name:
            raise RuleSetError(f"{at}.name: expected a non-empty string")
        flags.append((attr, name, _spec(entry, at)))

    return RuleSet(
        target=target,
        display_name=str(doc.get("display_name") or target),
        description=str(doc.get("description") or ""),
        weights=weights,
        constructs=constructs,
        flags=tuple(flags),
        sizes=_size_model(doc.get("sizes"), f"{where}.sizes"),
    )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

_CACHE: dict[str, RuleSet] = {}


def available_profiles() -> list[str]:
    """Names of the profiles bundled under ``complexity/profiles/``, sorted."""
    if not PROFILE_DIR.is_dir():
        return []
    return sorted(p.stem for p in PROFILE_DIR.glob("*.json"))


def profile_path(target: str) -> Path:
    """Filesystem path of the bundled profile named *target*."""
    return PROFILE_DIR / f"{target}.json"


def load_ruleset(
    target: str | None = None,
    *,
    path: str | Path | None = None,
    use_cache: bool = True,
) -> RuleSet:
    """Load a :class:`RuleSet`, resolving the profile per the module docstring.

    Parameters
    ----------
    target
        Profile name under ``complexity/profiles/`` (e.g. ``"pyspark"``).
    path
        An explicit profile file, taking precedence over *target*. Lets an
        operator supply a catalogue this package does not ship.
    use_cache
        Reuse a previously parsed rule set for the same source. ``False``
        re-reads from disk (tests that rewrite a profile use it).
    """
    if path is None:
        path_cfg = app_config.get_typed_value(_CONFIG_SECTION, "rules_path", str)
        if target is None and path_cfg:
            path = path_cfg

    if path is not None:
        resolved = Path(path)
        key = f"path:{resolved.resolve()}"
        if not resolved.is_file():
            raise RuleSetError(f"rule-set profile not found: {resolved}")
    else:
        if target is None:
            target = app_config.get_typed_value(
                _CONFIG_SECTION, "target", str, DEFAULT_TARGET
            )
        resolved = profile_path(str(target))
        key = f"target:{target}"
        if not resolved.is_file():
            known = ", ".join(available_profiles()) or "none"
            raise RuleSetError(
                f"unknown complexity target {target!r} "
                f"(no {resolved.name} in {PROFILE_DIR}); available: {known}"
            )

    if use_cache and key in _CACHE:
        return _CACHE[key]

    ruleset = _ruleset_from_doc(_read_profile_doc(resolved), resolved.name)
    logger.info(
        f"load_ruleset: target={ruleset.target}  display_name={ruleset.display_name!r}  "
        f"constructs={ruleset.construct_count}  source={resolved}"
    )
    if use_cache:
        _CACHE[key] = ruleset
    return ruleset


def clear_cache() -> None:
    """Drop parsed rule sets (mirrors ``app_config.clear_cache``; tests use it)."""
    _CACHE.clear()
