"""The complexity analyzer: chunks and batches in, scored verdicts out.
See complexity/README.md.

This module owns the *aggregation* rules only. Which construct means what is
data, and lives in :mod:`complexity.rules`; what counts as a construct beyond
the chunker's own metadata is in :mod:`complexity.detectors`. Nothing here
hard-codes a tier.

Two aggregation rules, applied everywhere:

- **Tier is presence-based** — a unit's tier is the highest tier among its
  signals, so one ``ARRAY`` in an otherwise trivial step still reads HIGH. This
  matches the brief ("High for arrays, do loops, %macro definitions") literally;
  a weighted-threshold scheme would let a lone array average away to MEDIUM.
- **Difficulty is worst-case** — a unit's Spark parity is the least
  translatable parity among its signals, for the same reason.

``score`` exists only to rank units *within* a tier and never feeds back into
the tier. It sums each distinct construct's weight once, so a step that uses
five different hard constructs outranks one that mentions the same construct
five times — repetition is verbosity, variety is work.

Logger name: ``complexity.analyzer``.
"""

from __future__ import annotations

import logging
from typing import Iterable

import app_config
from chunker.models import (
    SasBatch,
    SasBatchResult,
    SasChunk,
    SasChunkMetadata,
    SasChunkResult,
    SasCorpus,
)

from . import rules
from .detectors import detect_constructs
from .models import (
    BatchComplexity,
    ChunkComplexity,
    ComplexitySignal,
    ComplexityTier,
    CorpusComplexityReport,
    SparkParity,
    max_tier,
    tier_rank,
    worst_parity,
)
from .rules import SignalSpec

logger = logging.getLogger(__name__)

_CONFIG_SECTION = "complexity"


class ComplexityAnalyzer:
    """Scores :class:`~chunker.models.SasChunk` and
    :class:`~chunker.models.SasBatch` objects for translation complexity.

    Parameters
    ----------
    weight_low, weight_medium, weight_high : float | None
        Override the per-tier score weights. ``None`` (default) reads
        ``complexity.weight_*`` from config.json, falling back to the
        catalogue defaults (:data:`complexity.rules.WEIGHT_LOW` and friends).
        Weights only rank units within a tier — they can never change a tier.
    use_detectors : bool
        Run the supplementary regex scans (:mod:`complexity.detectors`) for
        ARRAY / DO / MERGE / FILENAME-method constructs. Default ``True``;
        turning it off restricts the analysis to what the chunker's own
        metadata already reports.
    """

    def __init__(
        self,
        *,
        weight_low: float | None = None,
        weight_medium: float | None = None,
        weight_high: float | None = None,
        use_detectors: bool = True,
    ) -> None:
        self._weights: dict[ComplexityTier, float] = {
            ComplexityTier.LOW: _resolve_weight(
                weight_low, "weight_low", rules.WEIGHT_LOW
            ),
            ComplexityTier.MEDIUM: _resolve_weight(
                weight_medium, "weight_medium", rules.WEIGHT_MEDIUM
            ),
            ComplexityTier.HIGH: _resolve_weight(
                weight_high, "weight_high", rules.WEIGHT_HIGH
            ),
        }
        self._use_detectors = use_detectors
        logger.info(
            f"ComplexityAnalyzer  weights={ {t.value: w for t, w in self._weights.items()} }  "
            f"detectors={'on' if use_detectors else 'off'}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_chunk(self, chunk: SasChunk) -> ChunkComplexity:
        """Score a single chunk."""
        signals = self._signals_for_chunk(chunk)
        tier, difficulty, score = self._aggregate(signals)
        result = ChunkComplexity(
            chunk_id=chunk.chunk_id,
            source_id=chunk.source_id,
            kind=chunk.kind.value,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            tier=tier,
            score=score,
            translation_difficulty=difficulty,
            signals=signals,
            rationale=_rationale(tier, difficulty, signals),
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"analyze_chunk: {result}")
        return result

    def analyze_batch(self, batch: SasBatch) -> BatchComplexity:
        """Score a batch by aggregating its member chunks.

        The batch's score is the **sum** of its members' — ten simple steps
        genuinely are more work than one — while its tier and difficulty are
        the worst any single member reaches.
        """
        members = [self.analyze_chunk(c) for c in batch.chunks]
        tier = max_tier([m.tier for m in members])
        difficulty = worst_parity([m.translation_difficulty for m in members])
        score = round(sum(m.score for m in members), 3)
        signals = _merge_signals(s for m in members for s in m.signals)
        result = BatchComplexity(
            batch_id=batch.batch_id,
            source_files=list(batch.source_files),
            members=members,
            tier=tier,
            score=score,
            translation_difficulty=difficulty,
            signals=signals,
            rationale=_rationale(tier, difficulty, signals),
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"analyze_batch: {result}")
        return result

    def analyze_items(
        self,
        items: Iterable[SasBatch | SasChunk],
        *,
        source_ids: list[str] | None = None,
    ) -> CorpusComplexityReport:
        """Score a mixed sequence of batches and standalone chunks.

        Accepts exactly what the pipeline iterates over
        (``SasBatchResult.all_ordered_items``), so a caller can score the same
        units the LLM is asked to translate.
        """
        batches: list[BatchComplexity] = []
        chunks: list[ChunkComplexity] = []
        seen_sources: list[str] = []
        for item in items:
            if isinstance(item, SasBatch):
                scored_batch = self.analyze_batch(item)
                batches.append(scored_batch)
                candidates = scored_batch.source_files
            else:
                scored_chunk = self.analyze_chunk(item)
                chunks.append(scored_chunk)
                candidates = [scored_chunk.source_id or "<inline>"]
            for sid in candidates:
                if sid not in seen_sources:
                    seen_sources.append(sid)

        report = CorpusComplexityReport(
            source_ids=source_ids if source_ids is not None else seen_sources,
            chunks=chunks,
            batches=batches,
        )
        logger.info(f"analyze_items: {report}")
        return report

    def analyze_result(self, result: SasChunkResult) -> CorpusComplexityReport:
        """Score every chunk of a single-file :class:`SasChunkResult`."""
        return self.analyze_items(
            result.chunks,
            source_ids=[result.source_id or "<inline>"],
        )

    def analyze_batch_result(
        self, batch_result: SasBatchResult
    ) -> CorpusComplexityReport:
        """Score every batch and singleton of a :class:`SasBatchResult`."""
        return self.analyze_items(
            batch_result.all_ordered_items,
            source_ids=list(batch_result.source_ids),
        )

    def analyze_corpus(self, corpus: SasCorpus) -> CorpusComplexityReport:
        """Score every chunk of a multi-file :class:`SasCorpus`, unbatched."""
        return self.analyze_items(corpus.all_chunks, source_ids=corpus.source_ids)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _signals_for_chunk(self, chunk: SasChunk) -> list[ComplexitySignal]:
        """Every signal a chunk raises, from metadata and (optionally) detectors."""
        raw: list[ComplexitySignal] = list(
            _metadata_signals(chunk.kind.value, chunk.metadata)
        )
        if self._use_detectors:
            for construct in detect_constructs(chunk.text):
                spec = rules.DETECTOR_RULES.get(construct.name)
                if spec is None:
                    # A detector fired for a construct with no catalogue entry:
                    # a wiring bug, not a property of the SAS source. Log it and
                    # skip rather than inventing a classification.
                    logger.warning(
                        f"_signals_for_chunk: detector '{construct.name}' has no "
                        f"entry in rules.DETECTOR_RULES; signal dropped "
                        f"(chunk={chunk.chunk_id})"
                    )
                    continue
                raw.append(
                    _signal(construct.name, spec, construct.evidence, "detector")
                )
        return _merge_signals(raw)

    def _aggregate(
        self, signals: list[ComplexitySignal]
    ) -> tuple[ComplexityTier, SparkParity, float]:
        """Fold signals into (tier, difficulty, score) by the module's two rules."""
        tier = max_tier([s.tier for s in signals])
        difficulty = worst_parity([s.parity for s in signals])
        score = round(sum(self._weights[s.tier] for s in signals), 3)
        return tier, difficulty, score


def _resolve_weight(explicit: float | None, key: str, default: float) -> float:
    """Weight precedence: explicit argument > config.json > catalogue default."""
    if explicit is not None:
        return float(explicit)
    value = app_config.get_typed_value(_CONFIG_SECTION, key, (int, float), default)
    return float(value)


def _signal(
    name: str, spec: SignalSpec, evidence: str, source: str
) -> ComplexitySignal:
    """Build a :class:`ComplexitySignal` from a catalogue *spec*."""
    return ComplexitySignal(
        name=name,
        category=spec.category,
        tier=spec.tier,
        parity=spec.parity,
        weight=spec.weight,
        evidence=evidence or spec.note,
        source=source,
    )


def _lookup_many(
    prefix: str,
    names: Iterable[str],
    catalogue: dict[str, SignalSpec],
) -> list[ComplexitySignal]:
    """Signals for every *name* that has a *catalogue* entry.

    A name with no entry contributes nothing: the catalogue is an allowlist of
    constructs whose cost is understood, so an unrecognised function must not
    inflate a chunk's score (see the module docstring in ``rules``).
    """
    out: list[ComplexitySignal] = []
    for name in names:
        spec = catalogue.get(name.lower())
        if spec is not None:
            out.append(
                _signal(f"{prefix}:{name.lower()}", spec, spec.note, "metadata")
            )
    return out


def _metadata_signals(
    kind: str, meta: SasChunkMetadata
) -> list[ComplexitySignal]:
    """Signals derivable from a chunk's kind and extracted metadata."""
    signals: list[ComplexitySignal] = []

    kind_spec = rules.KIND_RULES.get(kind)
    if kind_spec is not None:
        signals.append(_signal(f"kind:{kind}", kind_spec, kind_spec.note, "metadata"))

    if meta.proc_name:
        signals += _lookup_many("proc", [meta.proc_name], rules.PROC_RULES)
    signals += _lookup_many(
        "component_object", meta.component_objects, rules.COMPONENT_OBJECT_RULES
    )
    signals += _lookup_many(
        "function", meta.recognized_functions, rules.FUNCTION_RULES
    )
    signals += _lookup_many(
        "call_routine", meta.recognized_call_routines, rules.CALL_ROUTINE_RULES
    )
    if meta.global_statement_keyword:
        signals += _lookup_many(
            "global_statement",
            [meta.global_statement_keyword],
            rules.GLOBAL_STATEMENT_RULES,
        )

    for attr, name, spec in rules.FLAG_RULES:
        if getattr(meta, attr, None):
            signals.append(_signal(name, spec, spec.note, "metadata"))

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
    difficulty: SparkParity,
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


def sort_by_complexity(
    items: Iterable[ChunkComplexity | BatchComplexity],
) -> list[ChunkComplexity | BatchComplexity]:
    """*items* ordered hardest-first (tier, then Spark parity, then score)."""
    from .models import parity_rank

    return sorted(
        items,
        key=lambda i: (
            tier_rank(i.tier),
            parity_rank(i.translation_difficulty),
            i.score,
        ),
        reverse=True,
    )
