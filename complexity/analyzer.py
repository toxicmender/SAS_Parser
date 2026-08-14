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

T-shirt sizing is the third rule, and the only one that counts rather than
maxes. Both rules above are presence-based, which is right for "how hard is the
hardest thing here?" and useless for "how much work is this file?" — a
2000-line file of plain DATA steps raises no signal at all. So each **source
file** is scored on three declared dimensions:

- **effort** — chunks, lines, contained DATA/PROC steps, dataset I/O, params;
- **complexity** — each distinct signal's tier weight plus its parity weight;
- **uncertainty** — unresolved cross-file references, unclosed blocks,
  ``UNKNOWN_*`` chunks, and parser diagnostics.

Each is then **log-transformed and min-max rescaled** onto 0-1 against a window
declared in multiples of the profile's anchor (the raw score of a documented
reference *Medium* file), blended by declared weights, and the blend is min-max
rescaled in log space onto the Fibonacci points range — then banded and floored
by chunk kind. The rescale lives in :class:`~complexity.rules.SizeModel`; this
module only decides what goes into each dimension, and the counting behind
each of them lives in :mod:`complexity.scoring`.

Two properties are worth stating here because they are what the transform buys:
the three dimensions are counted in incomparable units, so rescaling them onto
one range makes their mix an explicit weight rather than an accident of
magnitude; and within a dimension the returns diminish, so the 200th step of a
file moves a size less than the 20th. Sizing against a fixed anchor rather than
the corpus percentile stays deliberate — the windows are anchor-relative, so a
file's size still never depends on which files it was analysed alongside. See
complexity/README.md.

Logger name: ``complexity.analyzer``.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import app_config
from chunker.models import (
    SasBatch,
    SasBatchResult,
    SasChunk,
    SasChunkKind,
    SasChunkResult,
    SasCorpus,
    SasDiagnostic,
)
from .crossfile import UNRESOLVED_CONSTRUCTS, CrossFileIndex
from .detectors import detect_constructs
from .graph import build_graph
from .models import (
    BatchComplexity,
    ChunkComplexity,
    ComplexitySignal,
    ComplexityTier,
    CorpusComplexityReport,
    FileComplexity,
    TShirtSize,
    TranslationParity,
    max_size,
    max_tier,
    tier_rank,
    worst_parity,
)
from .rules import RuleSet, SizeModel, load_ruleset
from .scoring import (
    _chunk_inputs,
    _chunk_outputs,
    _contained_steps,
    _file_datasets,
    _file_paths,
    _line_span,
    _merge_signals,
    _metadata_signals,
    _rationale,
    _resolve_optional_number,
    _resolve_weight,
    _scaled_dimensions,
    _signal,
)

logger = logging.getLogger(__name__)

_CONFIG_SECTION = "complexity"

# Chunk kinds excluded from the analysis outright. A COMMENT_BLOCK carries no
# construct and no data flow — it raises no signal in any profile — yet it
# still counts as chunk and line volume, so scoring it makes a file's size
# partly a measure of how well it was documented. Excluded before scoring
# rather than before rendering, so the numbers change too and not just the
# prose: a well-commented file is not more work than a bare one.
EXCLUDED_KINDS: frozenset[SasChunkKind] = frozenset({SasChunkKind.COMMENT_BLOCK})


def _analysable(chunks: Iterable[SasChunk]) -> list[SasChunk]:
    """*chunks* with the excluded kinds dropped."""
    return [c for c in chunks if c.kind not in EXCLUDED_KINDS]


class ComplexityAnalyzer:
    """Scores :class:`~chunker.models.SasChunk` and
    :class:`~chunker.models.SasBatch` objects for translation complexity.

    Parameters
    ----------
    target : str | None
        Rule-set profile to score against — ``"sparksql"``, ``"pyspark"``, or
        any name under ``complexity/profiles/``. ``None`` (default) reads
        ``complexity.target`` from config.json, then falls back to
        :data:`complexity.rules.DEFAULT_TARGET`.
    rules_path : str | Path | None
        An explicit profile file, taking precedence over *target*. Lets an
        operator supply a catalogue this package does not ship.
    ruleset : RuleSet | None
        A pre-built rule set, bypassing profile resolution entirely. Wins over
        both *target* and *rules_path*.
    weight_low, weight_medium, weight_high : float | None
        Override the per-tier score weights. ``None`` (default) reads
        ``complexity.weight_*`` from config.json, then the profile's own
        ``weights``. Weights only rank units within a tier — they can never
        change a tier.
    use_detectors : bool
        Run the supplementary regex scans (:mod:`complexity.detectors`) for
        ARRAY / DO / MERGE / FILENAME-method constructs. Default ``True``;
        turning it off restricts the analysis to what the chunker's own
        metadata already reports.
    use_cross_file : bool | None
        Resolve each chunk's outward references against the rest of the corpus
        (:mod:`complexity.crossfile`) and raise signals for the ones another
        file satisfies — or nothing does. ``None`` (default) reads
        ``complexity.use_cross_file`` from config.json, then defaults to on.
        Turning it off scores every file as if it were the only one, which is
        what you want when comparing a file against itself over time.
    size_anchor : float | None
        Raw score of the reference MEDIUM file. ``None`` (default) reads
        ``complexity.size_anchor`` from config.json, then the profile's own
        ``sizes.anchor.raw``. Lowering it makes every file rate larger.
    min_story_points, max_story_points : float | None
        Ends of the story-point scale a file's ``points`` is reported on.
        ``None`` (default) reads ``complexity.min_story_points`` /
        ``complexity.max_story_points`` from config.json, then the profile's
        ``sizes.story_points``, then its scale's end rungs (2 and 8). These
        re-denominate the numbers only — a team that estimates on 1-13 gets
        its own scale and the identical set of sizes.
    """

    def __init__(
        self,
        target: str | None = None,
        *,
        rules_path: "str | Path | None" = None,
        ruleset: RuleSet | None = None,
        weight_low: float | None = None,
        weight_medium: float | None = None,
        weight_high: float | None = None,
        use_detectors: bool = True,
        use_cross_file: bool | None = None,
        size_anchor: float | None = None,
        min_story_points: float | None = None,
        max_story_points: float | None = None,
    ) -> None:
        self._rules = ruleset or load_ruleset(target, path=rules_path)
        # Weight precedence: explicit argument > config.json > the profile's
        # own weights (which themselves default to the module constants).
        self._weights: dict[ComplexityTier, float] = {
            ComplexityTier.LOW: _resolve_weight(
                weight_low, "weight_low", self._rules.weight_for(ComplexityTier.LOW)
            ),
            ComplexityTier.MEDIUM: _resolve_weight(
                weight_medium,
                "weight_medium",
                self._rules.weight_for(ComplexityTier.MEDIUM),
            ),
            ComplexityTier.HIGH: _resolve_weight(
                weight_high, "weight_high", self._rules.weight_for(ComplexityTier.HIGH)
            ),
        }
        self._use_detectors = use_detectors
        self._use_cross_file = (
            use_cross_file
            if use_cross_file is not None
            else bool(
                app_config.get_typed_value(
                    _CONFIG_SECTION, "use_cross_file", bool, True
                )
            )
        )
        # Anchor precedence matches the weights above: explicit > config >
        # profile. Replacing only the anchor keeps the profile's bands and
        # dimension weights, which are calibrated relative to it.
        anchor = _resolve_weight(
            size_anchor, "size_anchor", self._rules.sizes.anchor_raw
        )
        points_low = _resolve_optional_number(
            min_story_points, "min_story_points", self._rules.sizes.min_story_points
        )
        points_high = _resolve_optional_number(
            max_story_points, "max_story_points", self._rules.sizes.max_story_points
        )
        sizes = self._rules.sizes
        if anchor != sizes.anchor_raw:
            sizes = replace(
                sizes,
                anchor_raw=anchor,
                # The anchor's dimension split is the same measurement as its
                # total, so it moves with it — leaving it behind would leave
                # the profile self-contradictory the moment anyone retunes.
                anchor_dimensions=_scaled_dimensions(
                    sizes.anchor_dimensions,
                    anchor / self._rules.sizes.anchor_raw
                    if self._rules.sizes.anchor_raw > 0
                    else 1.0,
                ),
            )
        if (points_low, points_high) != (
            sizes.min_story_points,
            sizes.max_story_points,
        ):
            sizes = replace(
                sizes,
                min_story_points=points_low,
                max_story_points=points_high,
            )
        self._sizes: SizeModel = sizes
        story_low, story_high = self._sizes.story_point_range
        logger.info(
            f"ComplexityAnalyzer  target={self._rules.target}  "
            f"constructs={self._rules.construct_count}  "
            f"weights={ {t.value: w for t, w in self._weights.items()} }  "
            f"detectors={'on' if use_detectors else 'off'}  "
            f"cross_file={'on' if self._use_cross_file else 'off'}  "
            f"size_anchor={self._sizes.anchor_raw}  "
            f"story_points={story_low}-{story_high}"
        )

    @property
    def ruleset(self) -> RuleSet:
        """The rule set this analyzer scores against."""
        return self._rules

    @property
    def target(self) -> str:
        """Name of the target-language profile in use."""
        return self._rules.target

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_chunk(
        self, chunk: SasChunk, *, index: CrossFileIndex | None = None
    ) -> ChunkComplexity:
        """Score a single chunk.

        *index* supplies the corpus context for cross-file signals. Called
        without one — the plain ``analyze_chunk(chunk)`` form — a chunk is
        scored on its own contents alone, because a lone chunk has no corpus
        to resolve references against and inventing one would be a guess.
        """
        signals = self._signals_for_chunk(chunk, index)
        tier, difficulty, score = self._aggregate(signals)
        result = ChunkComplexity(
            chunk_id=chunk.chunk_id,
            source_id=chunk.source_id,
            kind=chunk.kind.value,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            input_datasets=_chunk_inputs(chunk.metadata),
            output_datasets=_chunk_outputs(chunk.metadata),
            external_refs=chunk.metadata.external_refs,
            tier=tier,
            score=score,
            translation_difficulty=difficulty,
            signals=signals,
            rationale=_rationale(tier, difficulty, signals),
            target=self._rules.target,
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"analyze_chunk: {result}")
        return result

    def analyze_batch(
        self, batch: SasBatch, *, index: CrossFileIndex | None = None
    ) -> BatchComplexity:
        """Score a batch by aggregating its member chunks.

        The batch's score is the **sum** of its members' — ten simple steps
        genuinely are more work than one — while its tier and difficulty are
        the worst any single member reaches.
        """
        members = [
            self.analyze_chunk(c, index=index) for c in _analysable(batch.chunks)
        ]
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
            target=self._rules.target,
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"analyze_batch: {result}")
        return result

    def analyze_items(
        self,
        items: Iterable[SasBatch | SasChunk],
        *,
        source_ids: list[str] | None = None,
        diagnostics: Iterable[SasDiagnostic] | None = None,
    ) -> CorpusComplexityReport:
        """Score a mixed sequence of batches and standalone chunks.

        Accepts exactly what the pipeline iterates over
        (``SasBatchResult.all_ordered_items``), so a caller can score the same
        units the LLM is asked to translate.

        *diagnostics* feed the uncertainty dimension of the T-shirt size.
        They are optional because not every entry point has them:
        :class:`~chunker.models.SasBatchResult` carries no diagnostics at all,
        so :meth:`analyze_batch_result` cannot supply them. When they are
        absent, every file's ``uncertainty_complete`` is set False rather than
        quietly reporting a lower uncertainty than the file really has.
        """
        item_list = list(items)
        # The index needs every chunk in the corpus before any single one can
        # be resolved, so it is built up front from a flattened view.
        all_chunks = _analysable(
            c
            for item in item_list
            for c in (item.chunks if isinstance(item, SasBatch) else [item])
        )
        index = CrossFileIndex.build(all_chunks) if self._use_cross_file else None

        batches: list[BatchComplexity] = []
        chunks: list[ChunkComplexity] = []
        seen_sources: list[str] = []
        # Batch ids per file, offered as cut points for oversized files.
        batches_by_source: dict[str, list[str]] = {}
        for item in item_list:
            if isinstance(item, SasBatch):
                scored_batch = self.analyze_batch(item, index=index)
                batches.append(scored_batch)
                candidates = scored_batch.source_files
                for sid in candidates:
                    batches_by_source.setdefault(sid, []).append(item.batch_id)
            elif item.kind in EXCLUDED_KINDS:
                # Not scored, but its file still exists and must keep its place
                # in the corpus — a file of nothing but comments would
                # otherwise vanish from the report entirely.
                candidates = [item.source_id or "<inline>"]
            else:
                scored_chunk = self.analyze_chunk(item, index=index)
                chunks.append(scored_chunk)
                candidates = [scored_chunk.source_id or "<inline>"]
            for sid in candidates:
                if sid not in seen_sources:
                    seen_sources.append(sid)

        files = self._file_complexities(
            item_list,
            index=index,
            batches_by_source=batches_by_source,
            diagnostics=list(diagnostics) if diagnostics is not None else None,
        )

        report = CorpusComplexityReport(
            source_ids=source_ids if source_ids is not None else seen_sources,
            chunks=chunks,
            batches=batches,
            files=files,
            # Nodes come from the files that were actually rolled up, not from
            # the index alone, so a file with no resolvable reference still
            # appears in the graph as an isolated one rather than going missing.
            graph=(
                build_graph(index, nodes=[f.source_id for f in files])
                if index is not None
                else None
            ),
            target=self._rules.target,
            target_display=self._rules.display_name,
        )
        logger.info(f"analyze_items: {report}")
        return report

    def analyze_result(self, result: SasChunkResult) -> CorpusComplexityReport:
        """Score every chunk of a single-file :class:`SasChunkResult`."""
        return self.analyze_items(
            result.chunks,
            source_ids=[result.source_id or "<inline>"],
            diagnostics=result.diagnostics,
        )

    def analyze_batch_result(
        self, batch_result: SasBatchResult
    ) -> CorpusComplexityReport:
        """Score every batch and singleton of a :class:`SasBatchResult`.

        A batch result carries no diagnostics, so the uncertainty dimension
        runs on chunk-level evidence only and every file is marked
        ``uncertainty_complete=False``. Pass the originating results'
        diagnostics to :meth:`analyze_items` directly if you need the full
        picture.
        """
        return self.analyze_items(
            batch_result.all_ordered_items,
            source_ids=list(batch_result.source_ids),
        )

    def analyze_corpus(self, corpus: SasCorpus) -> CorpusComplexityReport:
        """Score every chunk of a multi-file :class:`SasCorpus`, unbatched."""
        return self.analyze_items(
            corpus.all_chunks,
            source_ids=corpus.source_ids,
            diagnostics=corpus.all_diagnostics,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _signals_for_chunk(
        self, chunk: SasChunk, index: CrossFileIndex | None = None
    ) -> list[ComplexitySignal]:
        """Every signal a chunk raises: metadata, detectors, and cross-file refs."""
        raw: list[ComplexitySignal] = list(
            _metadata_signals(
                self._rules, chunk.kind.value, chunk.metadata, self._weights
            )
        )
        if index is not None and self._use_cross_file:
            for ref in index.refs_for(chunk):
                spec = self._rules.spec("cross_file", ref.name)
                if spec is None:
                    # Same reasoning as the detector branch below: a resolver
                    # emitting a construct the catalogue does not list is a
                    # wiring bug, not a property of the SAS source.
                    logger.warning(
                        f"_signals_for_chunk: cross-file construct {ref.name!r} has "
                        f"no 'cross_file' entry in target {self._rules.target!r}; "
                        f"signal dropped (chunk={chunk.chunk_id})"
                    )
                    continue
                raw.append(
                    _signal(
                        f"cross_file:{ref.name}",
                        spec,
                        ref.evidence,
                        "cross_file",
                        self._weights[spec.tier],
                    )
                )
        if self._use_detectors:
            for construct in detect_constructs(chunk.text):
                spec = self._rules.spec("detector", construct.name)
                if spec is None:
                    # A detector fired for a construct with no catalogue entry:
                    # a wiring bug, not a property of the SAS source. Log it and
                    # skip rather than inventing a classification.
                    logger.warning(
                        f"_signals_for_chunk: detector '{construct.name}' has no "
                        f"'detector' entry in target {self._rules.target!r}; signal "
                        f"dropped (chunk={chunk.chunk_id})"
                    )
                    continue
                raw.append(
                    _signal(
                        construct.name,
                        spec,
                        construct.evidence,
                        "detector",
                        self._weights[spec.tier],
                    )
                )
        return _merge_signals(raw)

    def _file_complexities(
        self,
        items: list[SasBatch | SasChunk],
        *,
        index: CrossFileIndex | None,
        batches_by_source: dict[str, list[str]],
        diagnostics: list[SasDiagnostic] | None,
    ) -> list[FileComplexity]:
        """Roll every chunk up to its source file and T-shirt size the result.

        Files, not batches, are the unit: a batch may span several files while
        every chunk belongs to exactly one, so a file rollup built from chunks
        is unambiguous however the corpus was batched.
        """
        by_source: dict[str, list[SasChunk]] = {}
        # Excluded chunks per file, counted rather than dropped silently: a
        # `chunk_count` that disagrees with the chunker's, with nothing saying
        # why, reads as a bug in one of the two.
        comments_by_source: dict[str, int] = {}
        for item in items:
            for chunk in item.chunks if isinstance(item, SasBatch) else [item]:
                source_id = chunk.source_id or "<inline>"
                # setdefault before the kind test, so a file whose every chunk
                # is excluded still gets a (zero-chunk) rollup.
                bucket = by_source.setdefault(source_id, [])
                if chunk.kind in EXCLUDED_KINDS:
                    comments_by_source[source_id] = (
                        comments_by_source.get(source_id, 0) + 1
                    )
                    continue
                bucket.append(chunk)

        diags_by_source: dict[str, int] = {}
        if diagnostics is not None:
            for diag in diagnostics:
                key = diag.source_id or "<inline>"
                diags_by_source[key] = diags_by_source.get(key, 0) + 1

        files: list[FileComplexity] = []
        for source_id, source_chunks in by_source.items():
            scored = [self.analyze_chunk(c, index=index) for c in source_chunks]
            signals = _merge_signals(s for m in scored for s in m.signals)
            tier = max_tier([m.tier for m in scored])
            difficulty = worst_parity([m.translation_difficulty for m in scored])

            effort = self._effort_raw(source_chunks)
            complexity = self._complexity_raw(signals)
            uncertainty = self._uncertainty_raw(
                source_chunks, signals, diags_by_source.get(source_id, 0)
            )
            size, floored_by = self._size_for(
                effort, complexity, uncertainty, source_chunks
            )
            reads, writes, intermediates = _file_datasets(scored)
            external_refs = _file_paths(scored)

            files.append(
                FileComplexity(
                    source_id=source_id,
                    size=size,
                    # Read off the *final* size, floors included, so a file
                    # labelled Medium never reports Small's points — which is
                    # what an un-snapped position did whenever a floor bound.
                    points=self._sizes.points_for_size(size),
                    continuous_points=self._sizes.points_for(
                        effort, complexity, uncertainty
                    ),
                    effort_raw=round(effort, 3),
                    complexity_raw=round(complexity, 3),
                    uncertainty_raw=round(uncertainty, 3),
                    effort_norm=round(self._sizes.normalize("effort", effort), 3),
                    complexity_norm=round(
                        self._sizes.normalize("complexity", complexity), 3
                    ),
                    uncertainty_norm=round(
                        self._sizes.normalize("uncertainty", uncertainty), 3
                    ),
                    blend=round(
                        self._sizes.blend_for(effort, complexity, uncertainty), 3
                    ),
                    chunk_count=len(source_chunks),
                    line_count=_line_span(source_chunks),
                    comment_chunk_count=comments_by_source.get(source_id, 0),
                    input_datasets=reads,
                    output_datasets=writes,
                    intermediate_datasets=intermediates,
                    external_refs=external_refs,
                    chunks=scored,
                    cross_file=index.profile_for(source_id) if index else None,
                    suggested_split=(
                        list(dict.fromkeys(batches_by_source.get(source_id, [])))
                        if size.needs_breakdown
                        else []
                    ),
                    floored_by=floored_by,
                    uncertainty_complete=diagnostics is not None,
                    tier=tier,
                    score=round(sum(m.score for m in scored), 3),
                    translation_difficulty=difficulty,
                    signals=signals,
                    rationale=_rationale(tier, difficulty, signals),
                    target=self._rules.target,
                )
            )

        files.sort(key=lambda f: f.source_id)
        if logger.isEnabledFor(logging.DEBUG):
            for f in files:
                logger.debug(f"_file_complexities: {f}")
        return files

    def _effort_raw(self, chunks: list[SasChunk]) -> float:
        """Volume: how much there is of it, regardless of how hard it is."""
        sizes = self._sizes
        steps = sum(_contained_steps(c.text) for c in chunks)
        io = sum(
            len(c.metadata.input_datasets)
            + len(c.metadata.output_datasets)
            + len(c.metadata.produces_macrovars)
            for c in chunks
        )
        params = sum(len(c.metadata.macro_param_names) for c in chunks)
        return (
            len(chunks) * sizes.volume_weight("chunk")
            + _line_span(chunks) * sizes.volume_weight("line")
            + steps * sizes.volume_weight("step")
            + io * sizes.volume_weight("io")
            + params * sizes.volume_weight("param")
        )

    def _complexity_raw(self, signals: list[ComplexitySignal]) -> float:
        """Difficulty: tier weight plus parity weight, per distinct construct.

        Parity is included because it is what makes a size target-dependent —
        two files with identical tiers are not equal work if one's constructs
        have no equivalent in the target and the other's are one-for-one.
        """
        return sum(
            self._weights[s.tier] + self._sizes.parity_weight(s.parity)
            for s in signals
        )

    def _uncertainty_raw(
        self,
        chunks: list[SasChunk],
        signals: list[ComplexitySignal],
        diagnostic_count: int,
    ) -> float:
        """Risk: what the analysis could not pin down.

        Distinct from complexity, and reported separately, because the two
        need different responses — a file that is large here needs someone to
        go and find the missing pieces, not more translation hands.
        """
        sizes = self._sizes
        unresolved = sum(
            1
            for s in signals
            if s.source == "cross_file"
            and s.name.removeprefix("cross_file:") in UNRESOLVED_CONSTRUCTS
        )
        unclosed = sum(1 for c in chunks if c.metadata.has_unclosed_block)
        unknown = sum(1 for c in chunks if c.kind.value.startswith("UNKNOWN"))
        return (
            unresolved * sizes.uncertainty_weight("unresolved_ref")
            + unclosed * sizes.uncertainty_weight("unclosed_block")
            + unknown * sizes.uncertainty_weight("unknown_chunk")
            + diagnostic_count * sizes.uncertainty_weight("diagnostic")
        )

    def _size_for(
        self,
        effort: float,
        complexity: float,
        uncertainty: float,
        chunks: list[SasChunk],
    ) -> tuple[TShirtSize, str]:
        """Band the three dimensions into a size, then apply the kind floors.

        Returns the size and the chunk kind that forced a floor (empty when
        the banding stood on its own), so a size that the numbers alone do not
        explain still says why.
        """
        sizes = self._sizes
        banded = sizes.band_for(sizes.points_for(effort, complexity, uncertainty))
        floor = TShirtSize.SMALL
        floored_by = ""
        for chunk in chunks:
            candidate = sizes.min_size_by_kind.get(chunk.kind.value)
            if candidate is not None and max_size([floor, candidate]) is candidate:
                if candidate is not floor:
                    floor, floored_by = candidate, chunk.kind.value
        final = max_size([banded, floor])
        return final, floored_by if final is not banded else ""

    def _aggregate(
        self, signals: list[ComplexitySignal]
    ) -> tuple[ComplexityTier, TranslationParity, float]:
        """Fold signals into (tier, difficulty, score) by the module's two rules."""
        tier = max_tier([s.tier for s in signals])
        difficulty = worst_parity([s.parity for s in signals])
        score = round(sum(self._weights[s.tier] for s in signals), 3)
        return tier, difficulty, score


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
