"""Standalone complexity scoring and reports. See ``complexity/README.md``."""

from .analyzer import ComplexityAnalyzer, sort_by_complexity
from .crossfile import (
    CROSS_FILE_CONSTRUCTS,
    UNRESOLVED_CONSTRUCTS,
    CrossFileIndex,
    CrossFileRef,
)
from .detectors import DetectedConstruct, detect_constructs
from .graph import MAX_GRAPH_NODES, build_graph, render_markdown, render_png
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
    DependencyEdge,
    DependencyGraph,
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
from .naming import display_name, display_names, resolve_name
from .pdf import PdfRenderError, render_markdown_pdf, render_pdf
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
    # corpus dependency graph
    "DependencyEdge",
    "DependencyGraph",
    "build_graph",
    "render_markdown",
    "render_png",
    "MAX_GRAPH_NODES",
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
    # how a source file is named in a report, as opposed to identified
    "display_name",
    "display_names",
    "resolve_name",
    # optional PDF of a written Markdown report
    "PdfRenderError",
    "render_pdf",
    "render_markdown_pdf",
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
