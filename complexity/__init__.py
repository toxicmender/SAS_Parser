"""Translation-complexity analysis for SAS chunks, batches, and files.

Scores each :class:`~chunker.models.SasChunk` / :class:`~chunker.models.SasBatch`
on two orthogonal axes, and each **source file** on a third:

- **Data complexity** (:class:`ComplexityTier`) — LOW for simple SQL and macro
  variables, MEDIUM for hashing / MERGE / SFTP / mail, HIGH for arrays, DO
  loops, and ``%MACRO`` definitions. Presence-based: a unit's tier is the
  highest tier among the constructs it contains.
- **Feature parity with the target language** (:class:`TranslationParity`) —
  from ``DIRECT`` (a literal equivalent exists) to ``MANUAL`` (a human must
  redesign it). A unit's ``translation_difficulty`` is the worst parity among
  its constructs. Which construct earns which rating is per-target data, so the
  same analysis retargets from Spark SQL to PySpark by switching profile.
- **Migration size** (:class:`TShirtSize`) — Small / Medium / Large / Extra
  Large per source file, with Fibonacci story points. Unlike the other two this
  is volume-aware, which is the whole point: a long file of trivial steps
  raises no signal at all and would otherwise read as trivial. Size combines
  effort, complexity, and uncertainty — each log-transformed and min-max
  rescaled against a window measured relative to a reference file documented in
  the profile — and accounts for cross-file coupling.

Public API:

- :class:`ComplexityAnalyzer` — the entry point: ``analyze_chunk``,
  ``analyze_batch``, ``analyze_items``, ``analyze_result``,
  ``analyze_batch_result``, ``analyze_corpus``.
- :class:`ChunkComplexity`, :class:`BatchComplexity`, :class:`FileComplexity`,
  :class:`CrossFileProfile`, :class:`CorpusComplexityReport`,
  :class:`ComplexitySignal` — result models.
  ``CorpusComplexityReport.to_markdown()`` renders a summary table.
- :func:`write_reports` / :func:`render_file_report` /
  :func:`render_overall_report` / :func:`chunk_texts` — the two-level
  deliverable: one corpus report plus one report per source SAS script, each
  printing the SAS source behind every verdict it states.
- :func:`build_evaluation_prompt` / :func:`evaluate_report` /
  :class:`ComplexityEvaluation` — the optional LLM second opinion, asking a
  model where the rules are wrong or incomplete. Off unless called, and
  duck-typed on the client, so this package still depends on nothing but
  ``chunker`` and ``app_config``.
- :class:`ComplexityTier`, :class:`TranslationParity`, :class:`TShirtSize` —
  the three scales, plus the :func:`max_tier` / :func:`worst_parity` /
  :func:`max_size` aggregation helpers.
- :func:`detect_constructs` — the supplementary ARRAY / DO / MERGE / FILENAME
  scans, usable on their own.
- :class:`CrossFileIndex` — resolves macro / dataset / macro-variable / libref
  references across a corpus into import, export, or unresolved.
- :func:`sort_by_complexity` — order scored units hardest-first.
- :class:`RuleSet` / :class:`SizeModel` / :func:`load_ruleset` /
  :func:`available_profiles` — the target-language rule sets, loaded from JSON
  profiles under ``complexity/profiles/``. Retune tiers, parity ratings, and
  the sizing calibration, or add a target, by editing or adding a profile — no
  code change needed.

This package reads the chunker's output and is imported by nobody in the
pipeline: complexity analysis is standalone, so scoring a corpus never changes
what the LLM is asked to translate.

See complexity/README.md.

Logger name: ``complexity``.
"""

from .analyzer import ComplexityAnalyzer, sort_by_complexity
from .crossfile import (
    CROSS_FILE_CONSTRUCTS,
    UNRESOLVED_CONSTRUCTS,
    CrossFileIndex,
    CrossFileRef,
)
from .detectors import DetectedConstruct, detect_constructs
from .llm_eval import (
    ComplexityEvaluation,
    EvaluationFinding,
    FileEvaluation,
    FileEvaluationResult,
    build_evaluation_prompt,
    evaluate_file,
    evaluate_report,
    evaluation_prompts,
)
from .models import (
    BatchComplexity,
    ChunkComplexity,
    ComplexitySignal,
    ComplexityTier,
    CorpusComplexityReport,
    CrossFileProfile,
    FileComplexity,
    TShirtSize,
    TranslationParity,
    max_size,
    max_tier,
    parity_rank,
    size_rank,
    tier_rank,
    worst_parity,
)
from .report import (
    ChunkTextIndex,
    WrittenReports,
    chunk_texts,
    file_report_paths,
    render_file_report,
    render_overall_report,
    source_stems,
    write_reports,
)
from .rules import (
    DEFAULT_TARGET,
    RuleSet,
    RuleSetError,
    SignalSpec,
    SizeModel,
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
    # cross-file resolution
    "CrossFileIndex",
    "CrossFileRef",
    "CROSS_FILE_CONSTRUCTS",
    "UNRESOLVED_CONSTRUCTS",
    # result models
    "BatchComplexity",
    "ChunkComplexity",
    "ComplexitySignal",
    "CorpusComplexityReport",
    "CrossFileProfile",
    "FileComplexity",
    # Markdown rendering — overall + per source file
    "ChunkTextIndex",
    "WrittenReports",
    "chunk_texts",
    "file_report_paths",
    "render_file_report",
    "render_overall_report",
    "source_stems",
    "write_reports",
    # optional LLM evaluation
    "ComplexityEvaluation",
    "EvaluationFinding",
    "FileEvaluation",
    "FileEvaluationResult",
    "build_evaluation_prompt",
    "evaluate_file",
    "evaluate_report",
    "evaluation_prompts",
    # scales + helpers
    "ComplexityTier",
    "TranslationParity",
    "TShirtSize",
    "max_tier",
    "worst_parity",
    "max_size",
    "tier_rank",
    "parity_rank",
    "size_rank",
    # rule sets / targets
    "RuleSet",
    "RuleSetError",
    "SignalSpec",
    "SizeModel",
    "available_profiles",
    "load_ruleset",
    "DEFAULT_TARGET",
]
