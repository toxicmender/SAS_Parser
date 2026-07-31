"""Score a directory of SAS files and write the complexity reports as Markdown.

    python -m complexity path/to/sas_dir
    python -m complexity path/to/sas_dir --out-dir reports/
    python -m complexity path/to/sas_dir --target pyspark --out report.md
    python -m complexity path/to/sas_dir --rules-path my_profile.json --top 25
    python -m complexity path/to/sas_dir --out-dir reports/ --llm-eval
    python -m complexity path/to/sas_dir --out-dir reports/ --pdf

Chunks every matching file, batches the whole corpus with
:class:`~chunker.batcher.MultiFileBatcher` so cross-file dataset/macro edges
resolve into shared batches, and scores the resulting batches and singletons.
Scoring the *batched* units — rather than raw chunks — reports on the same work
items the pipeline translates, so an estimate lines up with what a migration run
will actually do.

Two kinds of report come out. ``--out`` (or stdout) writes the corpus-wide one,
:meth:`~complexity.models.CorpusComplexityReport.to_markdown`. ``--out-dir``
writes that *plus* one report per source SAS script under ``files/``, each
printing the SAS source of every chunk it mentions, with the overall report
indexing them. ``--pdf`` renders the overall report a second time as a PDF, for
the reader who is being sent the estimate rather than the repository.

``--llm-eval`` adds an optional second opinion: each file's verdict and source
are sent to a model with a prompt asking where the rules are wrong or
incomplete (see :mod:`complexity.llm_eval`). It is the only thing here that
touches the network — without it the analysis is entirely offline.
``--prompt-only`` writes those prompts and calls nothing.

Logger name: ``complexity.__main__``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from chunker import MultiFileBatcher, SasCorpus, SasSemanticChunker

from .analyzer import ComplexityAnalyzer
from .models import CorpusComplexityReport
from .report import (
    ChunkTextIndex,
    chunk_texts,
    render_overall_report,
    source_stems,
    write_reports,
)
from .rules import available_profiles

logger = logging.getLogger(__name__)

_EVALUATION_NAME = "llm-evaluation.md"
_PROMPT_DIR = "prompts"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m complexity",
        description=__doc__.splitlines()[0] if __doc__ else None,
    )
    parser.add_argument(
        "sas_dir",
        type=Path,
        help="Directory containing .sas files (searched recursively).",
    )
    parser.add_argument(
        "--pattern",
        default="*.sas",
        help="Glob for SAS files within sas_dir (default: *.sas).",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="Rule-set profile naming the output language "
        f"(available: {', '.join(sorted(available_profiles()))}). Defaults to "
        "config.json complexity.target, then the built-in default.",
    )
    parser.add_argument(
        "--rules-path",
        type=Path,
        default=None,
        help="A profile JSON of your own; wins over --target.",
    )
    parser.add_argument(
        "--size-anchor",
        type=float,
        default=None,
        help="Raw score of the reference MEDIUM file. Lowering it rates every "
        "file LARGER (default: config.json complexity.size_anchor, then the "
        "profile's own calibration).",
    )
    parser.add_argument(
        "--min-story-points",
        type=float,
        default=None,
        help="Points reported for the smallest file (default: config.json "
        "complexity.min_story_points, then the profile's scale, then 2).",
    )
    parser.add_argument(
        "--max-story-points",
        type=float,
        default=None,
        help="Points reported for the largest file. With --min-story-points "
        "this re-denominates the scale (say 1 and 13); sizes do not move, "
        "only the numbers (default: config.json complexity.max_story_points, "
        "then the profile's scale, then 8).",
    )
    parser.add_argument(
        "--no-cross-file",
        action="store_true",
        help="Score each file as if it were alone, without resolving its "
        "macro/dataset/libref references against the rest of the corpus.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="How many hardest units to list in the report (default: 10).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write the overall Markdown report here instead of printing it.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Write the overall report AND one report per source SAS script "
        "under this directory (individual ones in files/). Each individual "
        "report prints the SAS source behind every verdict.",
    )
    parser.add_argument(
        "--no-source-text",
        action="store_true",
        help="Leave the SAS source out of the individual reports and of any "
        "LLM prompt, reporting the verdicts alone.",
    )
    parser.add_argument(
        "--max-chunk-lines",
        type=int,
        default=0,
        help="Truncate each chunk's printed source to this many lines "
        "(default: 0, print it whole).",
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Also render the overall Markdown report as a PDF beside it "
        "(complexity-report.pdf under --out-dir, or --out's name with a .pdf "
        "suffix). Needs one of those two, since stdout has nowhere to put a "
        "file. The Markdown is written either way.",
    )
    parser.add_argument(
        "--no-graph-image",
        action="store_true",
        help="Skip drawing dependency-graph.png with --out-dir. The overall "
        "report's dependency edge table is written either way; the image "
        "needs the optional 'graph' extra (matplotlib) and is silently "
        "skipped without it.",
    )
    parser.add_argument(
        "--llm-eval",
        action="store_true",
        help="Ask a model to evaluate each file against its static verdict — "
        "the one option here that calls out to the network.",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Model id for --llm-eval (default: config.json llm_client.model). "
        "Implies --llm-eval.",
    )
    parser.add_argument(
        "--eval-top",
        type=int,
        default=0,
        help="Evaluate only the N largest files, since every file is a paid "
        "call (default: 0, evaluate all of them).",
    )
    parser.add_argument(
        "--prompt-only",
        action="store_true",
        help="Write the evaluation prompts instead of sending them — nothing "
        "is called, so the prompt can be read and tuned for free.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    # Checked before the corpus is scored: a run that cannot deliver what was
    # asked for should say so in a second, not after the analysis.
    if args.pdf and args.out is None and args.out_dir is None:
        logger.error("--pdf needs --out or --out-dir; stdout has nowhere to "
                     "put a file")
        print("--pdf needs --out or --out-dir.", file=sys.stderr)
        return 1

    sas_files = sorted(args.sas_dir.rglob(args.pattern))
    if not sas_files:
        logger.error(f"no files matching {args.pattern!r} under {args.sas_dir}")
        return 1
    logger.info(f"scoring {len(sas_files)} SAS file(s) under {args.sas_dir}")

    chunker = SasSemanticChunker()
    corpus = SasCorpus(
        file_results=[chunker.chunk_file(str(path)) for path in sas_files]
    )
    batch_result = MultiFileBatcher().batch(corpus)

    analyzer = ComplexityAnalyzer(
        args.target,
        rules_path=args.rules_path,
        size_anchor=args.size_anchor,
        min_story_points=args.min_story_points,
        max_story_points=args.max_story_points,
        use_cross_file=False if args.no_cross_file else None,
    )
    # analyze_items directly rather than analyze_batch_result: a batch result
    # carries no diagnostics, and passing the corpus's own means the
    # uncertainty dimension scores on the full evidence instead of chunk-level
    # signals alone (see ComplexityAnalyzer.analyze_batch_result).
    report = analyzer.analyze_items(
        batch_result.all_ordered_items,
        source_ids=corpus.source_ids,
        diagnostics=corpus.all_diagnostics,
    )
    # Verdicts carry no source text by design, so the renderers are handed a
    # lookup built from the same items that were scored. The batched items,
    # not the corpus: MultiFileBatcher re-ids every chunk per file (f1-, f2-,
    # ...), so a lookup built from the raw corpus would miss every one.
    texts = chunk_texts(batch_result.all_ordered_items)
    include_source = not args.no_source_text

    pdf_ok = True
    if args.out_dir is not None:
        written = write_reports(
            report,
            args.out_dir,
            texts=texts,
            top=args.top,
            include_source=include_source,
            max_source_lines=args.max_chunk_lines,
            graph_image=not args.no_graph_image,
        )
        print(f"wrote overall complexity report: {written.overall}")
        print(
            f"wrote {len(written.files)} individual report(s): "
            f"{args.out_dir / 'files'}"
        )
        if written.graph is not None:
            print(f"wrote dependency graph: {written.graph}")
        if args.out is not None and args.out != written.overall:
            _write(args.out, written.overall.read_text(encoding="utf-8"))
            print(f"wrote complexity report: {args.out}")
        if args.pdf:
            # From the written overall report, not from `args.out`: the PDF is
            # rendered beside dependency-graph.png so the image it links lands
            # in the document rather than resolving to nothing.
            pdf_ok = _write_pdf(written.overall)
    else:
        markdown = render_overall_report(report, top=args.top)
        if args.out is not None:
            _write(args.out, markdown)
            print(f"wrote complexity report: {args.out}")
            if args.pdf:
                pdf_ok = _write_pdf(args.out)
        else:
            print(markdown)

    if args.llm_eval or args.llm_model or args.prompt_only:
        status = _run_evaluation(args, report, texts, include_source=include_source)
        return status or (0 if pdf_ok else 1)
    return 0 if pdf_ok else 1


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_pdf(markdown_path: Path) -> bool:
    """Render *markdown_path* to a PDF beside it. False if it could not be.

    Imported lazily and only on this path, like every other optional artefact
    here. A failure is reported rather than raised — the Markdown is already on
    disk and the rest of the run is still worth finishing — but it does decide
    the exit status: the PDF was asked for by name.
    """
    from .pdf import PdfRenderError, render_pdf

    try:
        destination = render_pdf(markdown_path)
    except PdfRenderError as exc:
        logger.error(f"--pdf: {exc}")
        print(f"could not render the PDF: {exc}", file=sys.stderr)
        return False
    print(f"wrote complexity report PDF: {destination}")
    return True


def _run_evaluation(
    args: argparse.Namespace,
    report: CorpusComplexityReport,
    texts: ChunkTextIndex,
    *,
    include_source: bool,
) -> int:
    """The optional LLM pass: build the prompts, and send them unless asked not to.

    Imported lazily and only on this path, so the offline analysis never pays
    for — or depends on — the LLM stack.
    """
    from .llm_eval import evaluate_report, evaluation_prompts

    files = sorted(
        report.files,
        key=lambda f: (f.points, f.continuous_points),
        reverse=True,
    )
    wanted = (
        [f.source_id for f in files[: args.eval_top]] if args.eval_top > 0 else None
    )

    if args.prompt_only:
        prompts = evaluation_prompts(
            report,
            texts=texts,
            include_source=include_source,
            max_source_lines=args.max_chunk_lines,
            sources=wanted,
        )
        if args.out_dir is None:
            for source_id, prompt in prompts.items():
                print(f"\n===== {source_id} =====\n")
                print(prompt)
            return 0
        directory = Path(args.out_dir) / _PROMPT_DIR
        directory.mkdir(parents=True, exist_ok=True)
        stems = source_stems(prompts)
        for source_id, prompt in prompts.items():
            dest = directory / f"{stems[source_id]}.md"
            dest.write_text(prompt, encoding="utf-8")
        print(f"wrote {len(prompts)} evaluation prompt(s): {directory}")
        return 0

    try:
        from llm_client import LLMClient, LLMClientConfig
    except ImportError as exc:
        logger.error(f"--llm-eval needs the llm_client package: {exc!r}")
        print(f"--llm-eval is unavailable: {exc}", file=sys.stderr)
        return 1

    config = (
        LLMClientConfig(model=args.llm_model)
        if args.llm_model
        else LLMClientConfig()
    )
    evaluation = evaluate_report(
        LLMClient(config),
        report,
        texts=texts,
        include_source=include_source,
        max_source_lines=args.max_chunk_lines,
        limit=args.eval_top,
        model=config.model,
    )
    markdown = evaluation.to_markdown()
    if args.out_dir is not None:
        dest = Path(args.out_dir) / _EVALUATION_NAME
        _write(dest, markdown)
        print(f"wrote LLM complexity evaluation: {dest}")
    else:
        print(markdown)
    # A model that answered nothing usable is a failed run, not a quiet one.
    if evaluation.files and len(evaluation.failures) == len(evaluation.files):
        logger.error("_run_evaluation: every reply was unparseable")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
