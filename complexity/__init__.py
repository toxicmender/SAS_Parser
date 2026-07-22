"""Translation-complexity analysis for SAS chunks and batches.

Scores each :class:`~chunker.models.SasChunk` / :class:`~chunker.models.SasBatch`
on two orthogonal axes:

- **Data complexity** (:class:`ComplexityTier`) — LOW for simple SQL and macro
  variables, MEDIUM for hashing / MERGE / SFTP / mail, HIGH for arrays, DO
  loops, and ``%MACRO`` definitions. Presence-based: a unit's tier is the
  highest tier among the constructs it contains.
- **Feature parity with the target language** (:class:`TranslationParity`) —
  from ``DIRECT`` (a literal equivalent exists) to ``MANUAL`` (a human must
  redesign it). A unit's ``translation_difficulty`` is the worst parity among
  its constructs. Which construct earns which rating is per-target data, so the
  same analysis retargets from Spark SQL to PySpark by switching profile.

Public API:

- :class:`ComplexityAnalyzer` — the entry point: ``analyze_chunk``,
  ``analyze_batch``, ``analyze_items``, ``analyze_result``,
  ``analyze_batch_result``, ``analyze_corpus``.
- :class:`ChunkComplexity`, :class:`BatchComplexity`,
  :class:`CorpusComplexityReport`, :class:`ComplexitySignal` — result models.
  ``CorpusComplexityReport.to_markdown()`` renders a summary table.
- :class:`ComplexityTier`, :class:`TranslationParity` — the two scales, plus the
  :func:`max_tier` / :func:`worst_parity` aggregation helpers.
- :func:`detect_constructs` — the supplementary ARRAY / DO / MERGE / FILENAME
  scans, usable on their own.
- :func:`sort_by_complexity` — order scored units hardest-first.
- :class:`RuleSet` / :func:`load_ruleset` / :func:`available_profiles` — the
  target-language rule sets, loaded from JSON profiles under
  ``complexity/profiles/``. Retune tiers and parity ratings, or add a target,
  by editing or adding a profile — no code change needed.

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
    TranslationParity,
    max_tier,
    parity_rank,
    tier_rank,
    worst_parity,
)
from .rules import (
    DEFAULT_TARGET,
    RuleSet,
    RuleSetError,
    SignalSpec,
    available_profiles,
    load_ruleset,
)

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
    "TranslationParity",
    "max_tier",
    "worst_parity",
    "tier_rank",
    "parity_rank",
    # rule sets / targets
    "RuleSet",
    "RuleSetError",
    "SignalSpec",
    "available_profiles",
    "load_ruleset",
    "DEFAULT_TARGET",
]
