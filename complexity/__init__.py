"""Translation-complexity analysis for SAS chunks and batches.

Scores each :class:`~chunker.models.SasChunk` / :class:`~chunker.models.SasBatch`
on two orthogonal axes:

- **Data complexity** (:class:`ComplexityTier`) — LOW for simple SQL and macro
  variables, MEDIUM for hashing / MERGE / SFTP / mail, HIGH for arrays, DO
  loops, and ``%MACRO`` definitions. Presence-based: a unit's tier is the
  highest tier among the constructs it contains.
- **SAS -> Spark feature parity** (:class:`SparkParity`) — from ``DIRECT`` (a
  literal equivalent exists) to ``MANUAL`` (a human must redesign it). A unit's
  ``translation_difficulty`` is the worst parity among its constructs.

Public API:

- :class:`ComplexityAnalyzer` — the entry point: ``analyze_chunk``,
  ``analyze_batch``, ``analyze_items``, ``analyze_result``,
  ``analyze_batch_result``, ``analyze_corpus``.
- :class:`ChunkComplexity`, :class:`BatchComplexity`,
  :class:`CorpusComplexityReport`, :class:`ComplexitySignal` — result models.
  ``CorpusComplexityReport.to_markdown()`` renders a summary table.
- :class:`ComplexityTier`, :class:`SparkParity` — the two scales, plus the
  :func:`max_tier` / :func:`worst_parity` aggregation helpers.
- :func:`detect_constructs` — the supplementary ARRAY / DO / MERGE / FILENAME
  scans, usable on their own.
- :func:`sort_by_complexity` — order scored units hardest-first.
- :mod:`complexity.rules` — the signal catalogue; edit it to retune tiers and
  parity ratings without touching the analyzer.

This package reads the chunker's output and is imported by nobody in the
pipeline: complexity analysis is standalone, so scoring a corpus never changes
what the LLM is asked to translate.

See complexity/README.md.

Logger name: ``complexity``.
"""

from .analyzer import ComplexityAnalyzer, sort_by_complexity
from .detectors import DetectedConstruct, detect_constructs
from .models import (
    BatchComplexity,
    ChunkComplexity,
    ComplexitySignal,
    ComplexityTier,
    CorpusComplexityReport,
    SparkParity,
    max_tier,
    parity_rank,
    tier_rank,
    worst_parity,
)
from .rules import SignalSpec

__all__ = [
    # analyzer
    "ComplexityAnalyzer",
    "sort_by_complexity",
    # detectors
    "DetectedConstruct",
    "detect_constructs",
    # result models
    "BatchComplexity",
    "ChunkComplexity",
    "ComplexitySignal",
    "CorpusComplexityReport",
    # scales + helpers
    "ComplexityTier",
    "SparkParity",
    "max_tier",
    "worst_parity",
    "tier_rank",
    "parity_rank",
    # catalogue
    "SignalSpec",
]
