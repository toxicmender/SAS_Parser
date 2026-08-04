"""How each dimension is measured: the counting behind the analyzer's rules.

Split out of :mod:`complexity.analyzer`, which now holds the aggregation rules
alone — presence-based tier, worst-case parity, and the three-dimension size
blend. What actually goes *into* each of those is here: counting contained
DATA/PROC steps, deduplicating dataset names, spanning lines, turning a rule
match into a :class:`~complexity.models.ComplexitySignal`, merging signals from
different sources, and writing the one-line rationale a verdict carries.

The division is between *policy* and *measurement*: a change to what a
construct implies belongs in a profile, a change to how the rules combine
belongs in ``analyzer``, and a change to what is counted belongs here.

Logger name: ``complexity.scoring``.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable

import app_config
from chunker.models import SasChunk, SasChunkMetadata
from chunker.scanner import _sanitise

from .models import ChunkComplexity, ComplexitySignal, ComplexityTier, TranslationParity
from .rules import RuleSet, SignalSpec

logger = logging.getLogger(__name__)

_CONFIG_SECTION = "complexity"


# DATA/PROC step headers, counted for the effort dimension. A %MACRO is one
# chunk however many steps its body wraps — the chunker nests them rather than
# emitting them separately — so a wrapper around six steps has to be measured
# by scanning, not by counting chunks. Patterns mirror chunker/metadata.py.
_STEP_HEADER_RE = re.compile(r"\b(?:data|proc)\s+[A-Za-z_'\"&]", re.IGNORECASE)


def _contained_steps(text: str) -> int:
    """Number of DATA/PROC step headers in *text*.

    Runs on sanitised text, so a step named in a comment or a quoted string
    never inflates the count — the same guarantee ``detectors`` relies on.
    """
    return len(_STEP_HEADER_RE.findall(_sanitise(text)))


def _dedupe(names: Iterable[str]) -> list[str]:
    """*names* with case-insensitive duplicates dropped, first spelling kept.

    SAS is case-insensitive about dataset names, so ``WORK.P`` and ``work.p``
    are one table; listing both would make a file look like it touched more
    data than it did.
    """
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        key = name.lower()
        if key not in seen:
            seen.add(key)
            out.append(name)
    return out


def _chunk_inputs(meta: SasChunkMetadata) -> list[str]:
    """Datasets a chunk reads, its macro body's literal reads included.

    The body reads count because a macro definition is where the SAS lives even
    though the read happens at call time — the same union
    :mod:`complexity.crossfile` resolves against.
    """
    return _dedupe([*meta.input_datasets, *meta.body_literal_inputs])


def _chunk_outputs(meta: SasChunkMetadata) -> list[str]:
    """Datasets a chunk writes, its macro body's literal writes included."""
    return _dedupe([*meta.output_datasets, *meta.body_literal_outputs])


def _file_datasets(
    scored: list[ChunkComplexity],
) -> tuple[list[str], list[str], list[str]]:
    """One file's ``(inputs, outputs, intermediates)`` rolled up from its chunks.

    A dataset this file writes and then reads back is an **intermediate**, not
    an input: nothing outside the file has to provide it. That is the same rule
    :mod:`complexity.crossfile` applies when deciding whether a read is a
    cross-file import, so the datasets section and the coupling section of a
    report can never contradict each other.
    """
    reads = _dedupe(d for c in scored for d in c.input_datasets)
    writes = _dedupe(d for c in scored for d in c.output_datasets)
    written = {d.lower() for d in writes}
    return (
        [d for d in reads if d.lower() not in written],
        writes,
        [d for d in reads if d.lower() in written],
    )


def _line_span(chunks: list[SasChunk]) -> int:
    """Lines covered by *chunks*, counting any line only once.

    Overlapping chunks are real: the chunker emits a parent chunk alongside
    the children it was split into, so summing per-chunk spans would double
    every oversized region.
    """
    covered: set[int] = set()
    for chunk in chunks:
        covered.update(range(chunk.start_line, chunk.end_line + 1))
    return len(covered)


def _scaled_dimensions(
    dims: tuple[float, float, float] | None, factor: float
) -> tuple[float, float, float] | None:
    """The anchor's dimension split, rescaled by *factor* (``None`` stays None)."""
    if dims is None:
        return None
    return (dims[0] * factor, dims[1] * factor, dims[2] * factor)


def _resolve_weight(explicit: float | None, key: str, default: float) -> float:
    """Weight precedence: explicit argument > config.json > catalogue default."""
    if explicit is not None:
        return float(explicit)
    value = app_config.get_typed_value(_CONFIG_SECTION, key, (int, float), default)
    return float(value)


def _resolve_optional_number(
    explicit: float | None, key: str, default: float | None
) -> float | None:
    """The same precedence, for a setting whose default is "unset".

    Distinct from :func:`_resolve_weight` because ``None`` here is a real
    answer — it means "take the profile's scale" — rather than a missing one.
    """
    if explicit is not None:
        return float(explicit)
    value = app_config.get_typed_value(_CONFIG_SECTION, key, (int, float))
    return float(value) if value is not None else default


def _signal(
    name: str, spec: SignalSpec, evidence: str, source: str, weight: float
) -> ComplexitySignal:
    """Build a :class:`ComplexitySignal` from a catalogue *spec*.

    *evidence* (what was found here) and the spec's note (standing guidance)
    are kept in separate fields, so a detector's source snippet never shadows
    the catalogue's explanation of why the construct is rated as it is.
    """
    return ComplexitySignal(
        name=name,
        category=spec.category,
        tier=spec.tier,
        parity=spec.parity,
        weight=weight,
        evidence=evidence,
        note=spec.note,
        source=source,
    )


def _lookup_many(
    ruleset: RuleSet,
    construct_kind: str,
    names: Iterable[str],
    weights: dict[ComplexityTier, float],
) -> list[ComplexitySignal]:
    """Signals for every *name* that has a *catalogue* entry.

    A name with no entry contributes nothing: the catalogue is an allowlist of
    constructs whose cost is understood, so an unrecognised function must not
    inflate a chunk's score (see the module docstring in ``rules``).
    """
    out: list[ComplexitySignal] = []
    for name in names:
        spec = ruleset.spec(construct_kind, name)
        if spec is not None:
            out.append(
                _signal(
                    f"{construct_kind}:{name.lower()}",
                    spec,
                    "",
                    "metadata",
                    weights[spec.tier],
                )
            )
    return out


def _metadata_signals(
    ruleset: RuleSet,
    kind: str,
    meta: SasChunkMetadata,
    weights: dict[ComplexityTier, float],
) -> list[ComplexitySignal]:
    """Signals derivable from a chunk's kind and extracted metadata."""
    signals: list[ComplexitySignal] = []

    kind_spec = ruleset.constructs.get("kind", {}).get(kind)
    if kind_spec is not None:
        signals.append(
            _signal(
                f"kind:{kind}", kind_spec, "", "metadata", weights[kind_spec.tier]
            )
        )

    if meta.proc_name:
        signals += _lookup_many(ruleset, "proc", [meta.proc_name], weights)
    signals += _lookup_many(
        ruleset, "component_object", meta.component_objects, weights
    )
    signals += _lookup_many(ruleset, "function", meta.recognized_functions, weights)
    signals += _lookup_many(
        ruleset, "call_routine", meta.recognized_call_routines, weights
    )
    if meta.global_statement_keyword:
        signals += _lookup_many(
            ruleset, "global_statement", [meta.global_statement_keyword], weights
        )

    for attr, name, spec in ruleset.flags:
        if getattr(meta, attr, None):
            signals.append(
                _signal(name, spec, "", "metadata", weights[spec.tier])
            )

    return signals


def _merge_signals(
    signals: Iterable[ComplexitySignal],
) -> list[ComplexitySignal]:
    """Collapse repeats of the same construct into one signal.

    Repetition is verbosity, not extra work (see the module docstring), so each
    distinct signal name survives once — carrying the first occurrence's
    evidence, annotated with a count when it fired more than once. Order of
    first appearance is preserved so a result reads in scan order.
    """
    merged: dict[str, ComplexitySignal] = {}
    counts: dict[str, int] = {}
    for signal in signals:
        counts[signal.name] = counts.get(signal.name, 0) + 1
        if signal.name not in merged:
            merged[signal.name] = signal
    out: list[ComplexitySignal] = []
    for name, signal in merged.items():
        n = counts[name]
        if n > 1:
            signal = signal.model_copy(
                update={"evidence": f"{signal.evidence} (×{n})".strip()}
            )
        out.append(signal)
    return out


def _rationale(
    tier: ComplexityTier,
    difficulty: TranslationParity,
    signals: list[ComplexitySignal],
) -> str:
    """One-line explanation of a verdict, naming the signals that drove it."""
    if not signals:
        return (
            f"{tier}: no complexity signals detected — nothing beyond plain "
            f"statements was recognised."
        )
    drivers = [s.name for s in signals if s.tier is tier]
    hardest = [s.name for s in signals if s.parity is difficulty]
    parts = [
        f"{tier} tier driven by {', '.join(dict.fromkeys(drivers))}"
        if drivers
        else f"{tier} tier"
    ]
    if hardest:
        parts.append(
            f"Spark parity {difficulty} from {', '.join(dict.fromkeys(hardest))}"
        )
    return "; ".join(parts) + "."
