"""Score SAS files — local or SharePoint-hosted — and write complexity reports.

Local::

    python -m complexity path/to/sas_dir
    python -m complexity path/to/sas_dir --out-dir reports/
    python -m complexity path/to/sas_dir --target pyspark --out report.md
    python -m complexity path/to/sas_dir --rules-path my_profile.json --top 25
    python -m complexity path/to/sas_dir --out-dir reports/ --llm-eval
    python -m complexity path/to/sas_dir --out-dir reports/ --pdf

SharePoint::

    python -m complexity --sharepoint                       # every request row
    python -m complexity --sharepoint --app "MyApp" --pdf
    python -m complexity --sharepoint --item-id 42
    python -m complexity --sharepoint --app "MyApp" --no-upload --out-dir reports/

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

The SharePoint flow
-------------------
``--sharepoint`` reads the complexity request list
(:mod:`complexity.sharepoint`), takes each row's sources out of the document
library, and uploads the reports to
``{application}/complexity/{label}/{timestamp}``. The row supplies defaults:
``Output_Language`` becomes ``--target`` and ``Preferred_LLM`` becomes
``--llm-model`` and implies ``--llm-eval``. An explicit flag still wins over
the row.

Delivery is **stage then upload**: the reports are written to a temporary
directory with the same unmodified ``write_reports`` / ``render_pdf`` the local
flow uses, and the tree is uploaded from there. That is deliberate —
``render_pdf`` needs the Markdown and ``dependency-graph.png`` co-located to
resolve the image, and ``WrittenReports.paths`` already enumerates the upload
set. The report layer is not made storage-agnostic. With ``--out-dir`` *and*
``--sharepoint``, that directory is the staging area and the files are kept as
well as uploaded.

With more than one row selected, each is scored as **its own corpus** and
uploaded to its own folder: cross-file resolution across unrelated applications
would corrupt every verdict. One row failing does not abort the rest; the exit
status is non-zero if any row failed, and each row's own outcome is in the
``run-summary.md`` uploaded beside its reports.

Logger name: ``complexity.__main__``.
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
        nargs="?",
        default=None,
        help="Directory containing .sas files (searched recursively). Omit it "
        "and pass --sharepoint to score from the document library instead.",
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
        "--sharepoint",
        action="store_true",
        help="Score applications from the SharePoint complexity request list "
        "instead of a local directory. With no narrowing flag, every row in "
        "the list is processed.",
    )
    parser.add_argument(
        "--app",
        default=None,
        help="With --sharepoint: score one application, filtering the request "
        "rows on their Application column.",
    )
    parser.add_argument(
        "--item-id",
        default=None,
        help="With --sharepoint: score exactly one request row, by its ID.",
    )
    parser.add_argument(
        "--sharepoint-out",
        default=None,
        help="With --sharepoint: upload into this folder instead of "
        "{application}/complexity/{label}/{timestamp}.",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="With --sharepoint: analyse from the library but deliver locally "
        "and upload nothing. The dry run.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser.parse_args(argv)


def _argument_error(args: argparse.Namespace) -> str | None:
    """What is wrong with this combination of flags, or ``None``.

    Checked before anything is scored: a run that cannot deliver what was
    asked for should say so in a second, not after the analysis.
    """
    if args.sas_dir is None and not args.sharepoint:
        return (
            "give a directory to score, or --sharepoint to read the request "
            "list"
        )
    if args.sas_dir is not None and args.sharepoint:
        return (
            "--sharepoint reads its sources from the document library; do not "
            "also give a local directory"
        )
    if not args.sharepoint:
        for flag, value in (
            ("--app", args.app),
            ("--item-id", args.item_id),
            ("--sharepoint-out", args.sharepoint_out),
            ("--no-upload", args.no_upload),
        ):
            if value:
                return f"{flag} needs --sharepoint"
    if args.app and args.item_id:
        return "--app and --item-id both narrow the row set; give one"
    # --pdf needs somewhere to put a file. SharePoint mode always has one (it
    # stages into a temporary directory), so the guard only binds locally.
    if args.pdf and not args.sharepoint and args.out is None and args.out_dir is None:
        return "--pdf needs --out or --out-dir; stdout has nowhere to put a file"
    return None


@dataclass
class _Sources:
    """What was loaded, and what could not be.

    Attributes
    ----------
    file_results : list
        One :class:`~chunker.models.SasChunkResult` per source file.
    label : str
        Where they came from, for the log line.
    failures : list[str]
        Sources that could not be read or chunked. Recorded rather than
        raised: one unreadable file must not lose the estimate for the rest,
        and the ``run-summary.md`` reports them.
    """

    file_results: list[Any]
    label: str
    failures: list[str]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    problem = _argument_error(args)
    if problem is not None:
        logger.error(problem)
        print(problem, file=sys.stderr)
        return 1

    if args.sharepoint:
        return _run_sharepoint(args)
    return _run_local(args)


# ---------------------------------------------------------------------------
# Sources — the seam between where the SAS came from and what is done with it
# ---------------------------------------------------------------------------


def _load_local_sources(args: argparse.Namespace) -> _Sources | None:
    """Chunk every file matching ``--pattern`` under ``sas_dir``.

    ``None`` when the directory holds no matching file, which is a failed run
    rather than an empty report.
    """
    sas_files = sorted(args.sas_dir.rglob(args.pattern))
    if not sas_files:
        logger.error(f"no files matching {args.pattern!r} under {args.sas_dir}")
        return None
    logger.info(f"scoring {len(sas_files)} SAS file(s) under {args.sas_dir}")

    chunker = SasSemanticChunker()
    results: list[Any] = []
    failures: list[str] = []
    for path in sas_files:
        try:
            results.append(chunker.chunk_file(str(path)))
        except Exception as exc:
            logger.error(f"could not chunk {path}: {exc}")
            failures.append(str(path))
    if not results:
        return None
    return _Sources(results, str(args.sas_dir), failures)


def _load_sharepoint_sources(
    application: str, *, client: Any | None = None
) -> _Sources | None:
    """Download and chunk one application's scripts from the library.

    No temporary files: ``chunk_text`` takes the source text with an explicit
    ``source_id``, and the drive-relative path is exactly the id wanted — it
    is what the per-file reports are named after (via ``source_stems``) and
    what a reader needs to find the script again.
    """
    from . import sharepoint as sp

    paths = sp.source_files(application, client=client)
    if not paths:
        logger.error(f"no source files under the {application!r} scripts folder")
        return None
    logger.info(f"scoring {len(paths)} SAS file(s) for {application!r}")

    chunker = SasSemanticChunker()
    results: list[Any] = []
    failures: list[str] = []
    for path in paths:
        try:
            results.append(chunker.chunk_text(sp.load(path, client=client), source_id=path))
        except Exception as exc:
            logger.error(f"could not chunk {path}: {exc}")
            failures.append(path)
    if not results:
        return None
    return _Sources(results, application, failures)


# ---------------------------------------------------------------------------
# Analysis — shared by both modes, untouched by either
# ---------------------------------------------------------------------------


def _analyse(
    args: argparse.Namespace, sources: _Sources, *, target: str | None
) -> tuple[CorpusComplexityReport, ChunkTextIndex, ComplexityAnalyzer]:
    """Batch and score *sources*. The middle of the run, identical in both modes."""
    corpus = SasCorpus(file_results=sources.file_results)
    batch_result = MultiFileBatcher().batch(corpus)

    analyzer = ComplexityAnalyzer(
        target,
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
    return report, texts, analyzer


# ---------------------------------------------------------------------------
# Delivery
# ---------------------------------------------------------------------------


def _deliver(
    args: argparse.Namespace,
    report: CorpusComplexityReport,
    texts: ChunkTextIndex,
    *,
    out_dir: Path | None,
    model: str | None = None,
    run_llm_eval: bool | None = None,
) -> tuple[int, list[Path]]:
    """Write the reports, and return ``(exit status, written paths)``.

    *out_dir* is an explicit parameter rather than a read of ``args.out_dir``
    because the SharePoint flow stages into a temporary directory. The written
    paths come back so the caller can upload them; locally they are ignored.
    """
    include_source = not args.no_source_text
    written_paths: list[Path] = []

    pdf_ok = True
    if out_dir is not None:
        written = write_reports(
            report,
            out_dir,
            texts=texts,
            top=args.top,
            include_source=include_source,
            max_source_lines=args.max_chunk_lines,
            graph_image=not args.no_graph_image,
        )
        written_paths = list(written.paths)
        print(f"wrote overall complexity report: {written.overall}")
        print(
            f"wrote {len(written.files)} individual report(s): "
            f"{Path(out_dir) / 'files'}"
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
            if pdf_ok:
                written_paths.append(written.overall.with_suffix(".pdf"))
    else:
        markdown = render_overall_report(report, top=args.top)
        if args.out is not None:
            _write(args.out, markdown)
            print(f"wrote complexity report: {args.out}")
            if args.pdf:
                pdf_ok = _write_pdf(args.out)
        else:
            print(markdown)

    wanted = (
        run_llm_eval
        if run_llm_eval is not None
        else bool(args.llm_eval or args.llm_model or args.prompt_only)
    )
    status = 0
    if wanted:
        status = _run_evaluation(
            args,
            report,
            texts,
            out_dir=out_dir,
            include_source=include_source,
            model=model,
        )
        written_paths.extend(_evaluation_paths(args, out_dir))
    return (status or (0 if pdf_ok else 1)), written_paths


def _evaluation_paths(args: argparse.Namespace, out_dir: Path | None) -> list[Path]:
    """Whatever :func:`_run_evaluation` wrote under *out_dir*, for uploading."""
    if out_dir is None:
        return []
    root = Path(out_dir)
    found: list[Path] = []
    evaluation = root / _EVALUATION_NAME
    if evaluation.is_file():
        found.append(evaluation)
    prompts = root / _PROMPT_DIR
    if prompts.is_dir():
        found.extend(sorted(p for p in prompts.iterdir() if p.is_file()))
    return found


def _run_local(args: argparse.Namespace) -> int:
    sources = _load_local_sources(args)
    if sources is None:
        return 1
    report, texts, _ = _analyse(args, sources, target=args.target)
    status, _ = _deliver(args, report, texts, out_dir=args.out_dir)
    return status


# ---------------------------------------------------------------------------
# SharePoint mode
# ---------------------------------------------------------------------------


def _timestamp() -> str:
    """A path-safe UTC stamp naming one run's upload folder."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _segment(value: str) -> str:
    """*value* reduced to one safe path segment for a SharePoint folder."""
    import re

    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-")
    return cleaned or "run"


def _run_sharepoint(args: argparse.Namespace) -> int:
    """Score every selected request row, each as its own corpus.

    One row failing does not abort the rest — a request list is a queue, not a
    transaction — but the exit status is non-zero if any did.
    """
    from app_config.sharepoint import SharePointError, get_sharepoint_client

    from . import sharepoint as sp

    client = get_sharepoint_client()
    try:
        if args.item_id:
            rows = [sp.request(args.item_id, client=client)]
        else:
            rows = sp.requests(application=args.app, client=client)
    except SharePointError as exc:
        logger.error(f"could not read the complexity request list: {exc}")
        print(f"could not read the complexity request list: {exc}", file=sys.stderr)
        return 1

    if not rows:
        which = f" for application {args.app!r}" if args.app else ""
        logger.error(f"no complexity request rows{which}")
        print(f"no complexity request rows{which}.", file=sys.stderr)
        return 1

    logger.info(f"processing {len(rows)} complexity request row(s)")
    status = 0
    for row in rows:
        try:
            if _run_request(args, row, client=client) != 0:
                status = 1
        except Exception as exc:  # one row must not take the others down
            logger.exception(
                f"request {row.item_id} ({row.application}) failed: {exc}"
            )
            print(f"request {row.item_id} ({row.application}) failed: {exc}",
                  file=sys.stderr)
            status = 1
    return status


def _profile_for(output_language: str | None) -> str | None:
    """The rules profile a row's ``Output_Language`` names.

    The row carries a *language* in whatever spelling an operator picked
    ("SparkSQL", "Spark SQL", "spark_sql"), while ``--target`` and the
    ``profiles/`` directory carry a *profile* name ("sparksql").
    :mod:`target_language` owns the folding between the two, so routing the
    row through it is what makes a display-cased value find its rule set —
    on a case-sensitive filesystem as much as a forgiving one.

    Raises
    ------
    target_language.UnknownTargetLanguage
        The row names a language this repo cannot translate into. Reported
        per row by :func:`_run_sharepoint`, which is the right blast radius:
        one mistyped row must not stop the others.
    """
    if not output_language:
        return None
    from target_language import resolve_target_language

    return resolve_target_language(output_language).complexity_profile


def _run_request(args: argparse.Namespace, row: Any, *, client: Any) -> int:
    """Score one request row and upload its reports."""
    from . import sharepoint as sp

    started = time.monotonic()
    # Explicit flags beat the row, which beats config.json, which beats the
    # built-in default — the repo-wide precedence rule, extended by one step.
    # --target already names a profile; the row names a language.
    target = args.target or _profile_for(row.output_language)
    model = args.llm_model or row.preferred_llm
    run_llm_eval = bool(args.llm_eval or args.prompt_only or model)

    sources = _load_sharepoint_sources(row.application, client=client)
    if sources is None:
        return 1
    report, texts, analyzer = _analyse(args, sources, target=target)

    # The label records what produced the estimate: the model when the LLM
    # pass ran, else the rules profile that did — which still means something
    # for an entirely offline run.
    label = _segment(model if (run_llm_eval and model) else analyzer.ruleset.target)
    timestamp = _timestamp()

    with _staging(args.out_dir) as staging:
        status, written = _deliver(
            args,
            report,
            texts,
            out_dir=staging,
            model=model,
            run_llm_eval=run_llm_eval,
        )
        summary = sp.render_run_summary(
            row,
            target=analyzer.ruleset.target,
            model=model if run_llm_eval else None,
            label=label,
            timestamp=timestamp,
            files_scored=len(report.files),
            failures=sources.failures,
            seconds=time.monotonic() - started,
            exit_status=status,
        )
        summary_path = staging / sp.RUN_SUMMARY_NAME
        summary_path.write_text(summary, encoding="utf-8")

        if args.no_upload:
            logger.info(
                f"--no-upload: {len(written) + 1} file(s) left in {staging}"
            )
            return status
        uploaded = _upload(
            args, row, label, timestamp, [*written, summary_path], staging,
            client=client,
        )
        print(f"uploaded {len(uploaded)} file(s) for {row.application}")
    return status


def _staging(out_dir: Path | None) -> Any:
    """A context manager yielding the directory reports are written into.

    With ``--out-dir`` that directory is used and kept — a caller who asked
    for the files locally gets them as well as the upload. Without it, a
    temporary directory is used and removed. Staging rather than streaming is
    deliberate: ``render_pdf`` needs the Markdown and ``dependency-graph.png``
    co-located to resolve the image.
    """
    import contextlib

    if out_dir is not None:
        @contextlib.contextmanager
        def _kept():
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            yield Path(out_dir)

        return _kept()

    @contextlib.contextmanager
    def _temporary():
        with tempfile.TemporaryDirectory(prefix="sas_complexity_") as name:
            yield Path(name)

    return _temporary()


def _upload(
    args: argparse.Namespace,
    row: Any,
    label: str,
    timestamp: str,
    paths: list[Path],
    staging: Path,
    *,
    client: Any,
) -> list[str]:
    """Upload the staged tree, honouring ``--sharepoint-out``."""
    from app_config.sharepoint import SharePointConfig

    from . import sharepoint as sp

    if args.sharepoint_out:
        # An explicit destination replaces the whole convention, so it is used
        # as the base with no application/label/timestamp beneath it.
        from dataclasses import replace

        override = replace(
            SharePointConfig.from_env(),
            file_server_base_path=args.sharepoint_out.strip().strip("/"),
        )
        return sp.upload_reports(
            "", "", "", paths, staging_root=staging, client=client, config=override
        )
    return sp.upload_reports(
        row.application,
        label,
        timestamp,
        paths,
        staging_root=staging,
        client=client,
    )


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
    out_dir: Path | None,
    include_source: bool,
    model: str | None = None,
) -> int:
    """The optional LLM pass: build the prompts, and send them unless asked not to.

    Imported lazily and only on this path, so the offline analysis never pays
    for — or depends on — the LLM stack.

    *out_dir* is passed in rather than read off *args* so the SharePoint flow's
    staging directory collects ``prompts/`` and ``llm-evaluation.md`` too, and
    they get uploaded with everything else.
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
        if out_dir is None:
            for source_id, prompt in prompts.items():
                print(f"\n===== {source_id} =====\n")
                print(prompt)
            return 0
        directory = Path(out_dir) / _PROMPT_DIR
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

    # The "complexity" role overlay lets this run carry its own model and
    # timeout without disturbing the conversion pipeline's; an explicit
    # --llm-model (or a SharePoint row's Preferred_LLM, passed as *model*)
    # still wins over both.
    wanted_model = model or args.llm_model
    config = (
        LLMClientConfig.for_role("complexity", model=wanted_model)
        if wanted_model
        else LLMClientConfig.for_role("complexity")
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
    if out_dir is not None:
        dest = Path(out_dir) / _EVALUATION_NAME
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
