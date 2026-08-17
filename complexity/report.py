"""Markdown rendering: one overall report plus one report per source file.

:meth:`~complexity.models.CorpusComplexityReport.to_markdown` answers "how big
is this migration?" across a corpus. It cannot answer "what is in *this* file,
and why did it score that?" — for that a reader needs the file's own verdict
next to the SAS that produced it. So this module renders a **second** kind of
report, one per source file, and an index tying them to the overall one.

The individual reports print each chunk's **source text** alongside its verdict.
That text is passed in rather than read off the verdict: a
:class:`~complexity.models.ChunkComplexity` deliberately carries no ``text``
field, because a verdict model that embedded its own source would double the
size of every serialised report and duplicate what the chunker already holds.
:func:`chunk_texts` builds the lookup from whatever the analysis was run on.

The lookup is keyed on ``(source_id, chunk_id)`` and never on ``chunk_id``
alone: files are chunked independently, so two files' first chunks can share an
id (the same reason :mod:`complexity.crossfile` keys its reference table that
way). A pair with no entry renders a placeholder rather than silently dropping
the section — a missing snippet is a wiring problem worth seeing.

Rendering only. Nothing here scores anything, and nothing here calls an LLM;
:mod:`complexity.llm_eval` is the module that does.

Logger name: ``complexity.report``.
"""

from __future__ import annotations

import logging
import re
# collections.abc rather than typing: Iterable is used as an isinstance test
# below, not only as an annotation.
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from chunker.models import (
    SasBatch,
    SasBatchResult,
    SasChunk,
    SasChunkResult,
    SasCorpus,
)

from .models import (
    ChunkComplexity,
    ComplexitySignal,
    CorpusComplexityReport,
    FileComplexity,
    PathLocation,
    SasPathRef,
)
from .naming import resolve_name

if TYPE_CHECKING:  # the real type for a checker, no import at run time
    from data_hydration.models import HydrationPlan as _HydrationPlan  # noqa: F401

logger = logging.getLogger(__name__)

# The source id analyzer.py assigns to a chunk that came from a string rather
# than a file. Restated here (not imported) so this module stays renderer-only.
_INLINE = "<inline>"

# Default file names under an output directory. The overall report sits at the
# top; the per-file ones go in a subdirectory so a corpus of 200 files does not
# bury it.
OVERALL_REPORT_NAME = "complexity-report.md"
FILE_REPORT_DIR = "files"
# The dependency graph image, beside the overall report that links it.
GRAPH_IMAGE_NAME = "dependency-graph.png"

#: ``(source_id, chunk_id) -> SAS source text``.
ChunkTextIndex = Mapping[tuple[str, str], str]


class WrittenReports(NamedTuple):
    """What :func:`write_reports` put on disk."""

    overall: Path
    #: ``source_id -> path``, in the order the files were rendered.
    files: dict[str, Path]
    #: The dependency graph image, or None when none was drawn — no edges, no
    #: matplotlib, or too many files to be legible. Never a failure on its own.
    graph: Path | None = None

    @property
    def paths(self) -> list[Path]:
        """Every written path, overall report first."""
        image = [self.graph] if self.graph else []
        return [self.overall, *self.files.values(), *image]


# ---------------------------------------------------------------------------
# Source text lookup
# ---------------------------------------------------------------------------


def chunk_texts(source: object) -> dict[tuple[str, str], str]:
    """Build the ``(source_id, chunk_id) -> text`` lookup the renderers take.

    Accepts whatever the analysis was run on — a :class:`SasCorpus`, a
    :class:`SasChunkResult`, a :class:`SasBatchResult`, a single batch or
    chunk, or any iterable of those — so a caller never has to flatten the
    corpus itself just to render it.
    """
    texts: dict[tuple[str, str], str] = {}
    _collect_texts(source, texts)
    logger.debug(f"chunk_texts: indexed {len(texts)} chunk(s)")
    return texts


def _collect_texts(source: object, into: dict[tuple[str, str], str]) -> None:
    """Walk *source* depth-first, recording every chunk's text in *into*."""
    if isinstance(source, SasChunk):
        into[(source.source_id or _INLINE, source.chunk_id)] = source.text
    elif isinstance(source, SasBatch):
        _collect_texts(source.chunks, into)
    elif isinstance(source, SasChunkResult):
        _collect_texts(source.chunks, into)
    elif isinstance(source, SasBatchResult):
        _collect_texts(source.all_ordered_items, into)
    elif isinstance(source, SasCorpus):
        _collect_texts(source.file_results, into)
    elif isinstance(source, Iterable) and not isinstance(source, (str, bytes)):
        for item in source:
            _collect_texts(item, into)
    else:
        logger.warning(
            f"chunk_texts: ignoring {type(source).__name__}, which holds no chunks"
        )


# ---------------------------------------------------------------------------
# Small formatting helpers
# ---------------------------------------------------------------------------


def _fmt_list(values: Iterable[str], empty: str = "none") -> str:
    joined = ", ".join(values)
    return joined or empty


def _fence(text: str) -> str:
    """A fence long enough that *text* cannot close it early.

    SAS rarely contains a backtick run, but a report that silently breaks its
    own Markdown when it does is worse than three extra characters.
    """
    longest = max((len(m) for m in re.findall(r"`+", text)), default=0)
    return "`" * max(3, longest + 1)


def _source_block(text: str, *, max_lines: int = 0) -> list[str]:
    """*text* as a fenced ``sas`` block, truncated to *max_lines* (0 = all)."""
    body = text.rstrip("\n")
    lines = body.split("\n")
    truncated = 0
    if max_lines > 0 and len(lines) > max_lines:
        truncated = len(lines) - max_lines
        lines = lines[:max_lines]
        body = "\n".join(lines)
    fence = _fence(body)
    out = [f"{fence}sas", body, fence]
    if truncated:
        out += ["", f"_… {truncated} further line(s) not shown._"]
    return out


def _signal_rows(signals: list[ComplexitySignal]) -> list[str]:
    """A Markdown table of *signals* — the evidence behind a verdict."""
    rows = [
        "| Construct | Tier | Parity | Found by | Evidence and guidance |",
        "| --- | --- | --- | --- | --- |",
    ]
    for signal in signals:
        # Pipes inside a snippet would split the row into extra columns.
        detail = signal.detail.replace("|", "\\|").replace("\n", " ") or "—"
        rows.append(
            f"| `{signal.name}` | {signal.tier} | {signal.parity} "
            f"| {signal.source} | {detail} |"
        )
    return rows


def _dataset_lines(file: FileComplexity) -> list[str]:
    """The file's data interface: what it needs, what it makes, what stays inside.

    Split three ways because the three carry different instructions. Inputs
    must exist before this file runs; outputs are what downstream files wait
    on; intermediates are the file's own business and nobody has to provide
    them. A file that touches no dataset at all gets no section — an empty one
    would say nothing the absence does not.
    """
    if not (file.input_datasets or file.output_datasets or file.intermediate_datasets):
        return []
    lines = [
        "",
        "## Datasets",
        "",
        f"- Inputs (read here, written elsewhere): {_fmt_list(file.input_datasets)}",
        f"- Outputs (written here): {_fmt_list(file.output_datasets)}",
    ]
    if file.intermediate_datasets:
        lines.append(
            f"- Intermediates (written and read here): "
            f"{_fmt_list(file.intermediate_datasets)}"
        )
    return lines


#: How each location kind is introduced in the Paths section, and the order the
#: groups are printed in. Filesystem first because it is the common case and the
#: one with an obvious target-side answer; pipe and device last because they need
#: a decision rather than a mapping.
_PATH_GROUPS: tuple[tuple[PathLocation, str], ...] = (
    (PathLocation.FILESYSTEM, "Filesystem (needs a volume or external location)"),
    (PathLocation.REMOTE, "Remote services (needs network egress)"),
    (PathLocation.EMAIL, "Email destinations"),
    (PathLocation.PIPE, "Shell pipes (a command, not a location)"),
    (PathLocation.DEVICE, "Other devices"),
)


def _path_lines(file: FileComplexity) -> list[str]:
    """Everywhere outside the SAS libraries this file reaches.

    Grouped by kind because the kinds need different answers: a filesystem path
    wants a volume, an FTP reference wants egress and a credential, and a shell
    pipe wants somebody to decide what replaces it. A flat list would hide that.

    A file that reaches nothing outside gets no section, the same rule
    :func:`_dataset_lines` follows — an empty heading says nothing the absence
    does not.
    """
    if not file.external_refs:
        return []
    lines = ["", "## Paths", ""]
    for location, heading in _PATH_GROUPS:
        refs = [r for r in file.external_refs if r.location is location]
        if not refs:
            continue
        lines.append(f"- {heading}:")
        lines += [f"  - {_fmt_path_ref(r)}" for r in refs]
    return lines


def _fmt_path_ref(ref: SasPathRef) -> str:
    """One reference as a report line: the value as written, then its provenance.

    ``raw`` rather than ``path`` because this is for a human to recognise in
    their own source; the normalised form exists for matching, not for reading.
    """
    parts = [f"`{ref.raw}`", f"— {ref.statement}"]
    if ref.binds:
        parts.append(f"`{ref.binds}`")
    if ref.device:
        parts.append(f"via {ref.device}")
    if ref.engine:
        # An engine changes what the directory *is* — an `spde` library is
        # partitioned storage, not a folder of datasets — so it belongs next to
        # the path rather than being left for the reader to infer.
        parts.append(f"engine `{ref.engine}`")
    if ref.has_macro_ref:
        # The one thing a reader must not miss: this value is not what SAS
        # resolves at run time, so it cannot be mapped as written.
        parts.append("**(unresolved macro reference)**")
    return " ".join(parts)


def _hydration_lines(file: FileComplexity, plan: Any | None) -> list[str]:
    """What hydrating this file's sources would do, or nothing without a plan.

    *plan* is a :class:`data_hydration.models.HydrationPlan`, passed in by the
    caller rather than stored on :class:`FileComplexity`: a stored field would
    make :mod:`complexity.models` import :mod:`data_hydration` at module scope
    and turn an optional capability into a hard dependency of a shipped package.
    It is typed loosely here for the same reason — this module must not import
    that one.

    ``None`` (the default everywhere) renders nothing, so a report built without
    ``--hydration`` is byte-identical to one from before this existed. A file
    that hydrates nothing gets no heading either, the rule
    :func:`_path_lines` and :func:`_dataset_lines` already follow.
    """
    if plan is None:
        return []
    items = plan.by_source_id(file.source_id)
    if not items:
        return []
    lines = ["", "## Hydration", ""]
    for item in items:
        target = f"`{item.target_table}`"
        lines.append(f"- {item.source} -> {target}")
        detail = f"  - {item.strategy}"
        if item.partition:
            detail += f" — partition `{item.partition}`"
        lines.append(f"{detail}: {item.strategy_reason}")
        for note in item.notes:
            lines.append(f"  - {note}")
        for blocker in item.blockers:
            # Bolded for the same reason an unresolved macro is bolded above:
            # it is the line that decides whether this can run unattended.
            lines.append(f"  - **Needs operator input:** {blocker}")
    return lines


def _cross_file_lines(
    file: FileComplexity, names: Mapping[str, str] | None = None
) -> list[str]:
    """The file's coupling to the rest of the corpus, or nothing when absent.

    ``depends_on`` / ``depended_on_by`` hold peer ``source_id``s, so they are
    named through *names*; imports, exports, and unresolved references are
    construct names and print as they are.
    """
    profile = file.cross_file
    if profile is None:
        return []
    if not (
        profile.is_coupled
        or profile.imports
        or profile.exports
        or profile.unresolved
    ):
        return [
            "",
            "## Cross-file coupling",
            "",
            "Nothing crosses this file's boundary: it is translatable on its own.",
        ]
    lines = [
        "",
        "## Cross-file coupling",
        "",
        "- Depends on: "
        + _fmt_list(resolve_name(p, names) for p in profile.depends_on),
        "- Depended on by: "
        + _fmt_list(resolve_name(p, names) for p in profile.depended_on_by),
        f"- Imports: {_fmt_list(profile.imports)}",
        f"- Exports: {_fmt_list(profile.exports)}",
        f"- Unresolved: {_fmt_list(profile.unresolved)}",
    ]
    if profile.unresolved:
        claim = (
            f"searched against {profile.corpus_files} file(s) in scope"
            if profile.corpus_files > 1
            # With one file there was nothing to search, so "unresolved" is a
            # statement about scope rather than about the reference.
            else "only one file was in scope, so these are merely external"
        )
        lines += ["", f"Unresolved references — {claim}."]
    return lines


# ---------------------------------------------------------------------------
# Individual (per source file) report
# ---------------------------------------------------------------------------


def render_file_report(
    file: FileComplexity,
    *,
    texts: ChunkTextIndex | None = None,
    target_display: str = "",
    include_source: bool = True,
    max_source_lines: int = 0,
    overall_link: str | None = None,
    names: Mapping[str, str] | None = None,
    hydration: Any | None = None,
) -> str:
    """Render one source file's own complexity report as Markdown.

    Every chunk the report mentions is printed with its SAS source, so the
    verdict and the code that earned it can be read together. *texts* supplies
    that source (see :func:`chunk_texts`); without it — or with
    ``include_source=False`` — the report renders the verdicts alone.

    *max_source_lines* caps each printed chunk (0, the default, prints it
    whole). *overall_link* is a relative path back to the corpus report.

    *names* is the corpus-wide ``source_id -> display name`` mapping (see
    :mod:`complexity.naming`), used for this file's title and for the peers it
    is coupled to; without one each falls back to its own file name. The full
    path is still printed once, under the title, since this report is the one
    place a reader may want to know where the file actually lives.

    *hydration* is an optional ``data_hydration.HydrationPlan``; when given, the
    loads this file's sources imply are printed after its paths. See
    :func:`_hydration_lines`.
    """
    lines = [
        f"# Complexity report — {resolve_name(file.source_id, names)}",
        "",
    ]
    if resolve_name(file.source_id, names) != file.source_id:
        # The one place the full path is still printed: a reader looking at a
        # single file's verdict is the reader who may need to go open it.
        lines.append(f"- Path: `{file.source_id}`")
    lines += [
        f"- Target: **{target_display or file.target or 'unknown'}**",
        f"- Size: **{file.size.label}** — **{file.points:g}** story points "
        f"(continuous position {file.continuous_points:.2f})",
        f"- Tier: **{file.tier}** — parity **{file.translation_difficulty}**",
        f"- Chunks: {file.chunk_count} across {file.line_count} line(s)"
        + (
            f", plus {file.comment_chunk_count} comment block(s) excluded from "
            f"the analysis"
            if file.comment_chunk_count
            else ""
        ),
        f"- Effort: {file.effort_raw:.1f} ({file.effort_norm:.2f}) · "
        f"Complexity: {file.complexity_raw:.1f} ({file.complexity_norm:.2f}) · "
        f"Uncertainty: {file.uncertainty_raw:.1f} ({file.uncertainty_norm:.2f})",
        f"- Blend: {file.blend:.2f} · Raw total: {file.raw_total:.1f}",
    ]
    if file.floored_by:
        lines.append(
            f"- Size floored at {file.size.label} by a `{file.floored_by}` chunk — "
            f"the numbers alone rated it smaller."
        )
    if not file.uncertainty_complete:
        lines.append(
            "- Uncertainty is a **lower bound**: parser diagnostics were not "
            "available to this run."
        )
    if overall_link:
        lines.append(f"- Corpus report: [{overall_link}]({overall_link})")

    lines += ["", "## Verdict", "", file.rationale or "No signals fired."]

    if file.needs_breakdown:
        cuts = (
            ", ".join(f"`{c}`" for c in file.suggested_split)
            if file.suggested_split
            else "no batch boundaries available — split by step"
        )
        lines += [
            "",
            "## Break this file down first",
            "",
            "Extra Large is an instruction, not just a magnitude: this file "
            "should be split before anyone estimates or starts it.",
            "",
            f"Suggested cut points: {cuts}",
        ]

    # The file's own interface first, then how it couples to the corpus: the
    # second only makes sense once the reader knows what the file reads and
    # writes.
    lines += _dataset_lines(file)
    lines += _path_lines(file)
    # After the paths, because a hydration item is an answer to one of them: the
    # reader has just seen what the file reaches, and this says what becomes of it.
    lines += _hydration_lines(file, hydration)
    lines += _cross_file_lines(file, names)

    lines += ["", f"## Drivers ({len(file.signals)})", ""]
    if file.signals:
        lines += _signal_rows(file.signals)
    else:
        lines.append(
            "No construct in this file is in the catalogue, so nothing beyond "
            "plain statements was recognised. Its size is volume alone."
        )

    lines += ["", f"## Chunks ({len(file.chunks)})", ""]
    if not file.chunks:
        lines.append("_No chunks._")
    for chunk in sorted(file.chunks, key=lambda c: (c.start_line, c.chunk_id)):
        lines += _chunk_section(
            chunk,
            texts=texts,
            include_source=include_source,
            max_source_lines=max_source_lines,
        )

    return "\n".join(lines).rstrip() + "\n"


def _chunk_section(
    chunk: ChunkComplexity,
    *,
    texts: ChunkTextIndex | None,
    include_source: bool,
    max_source_lines: int,
) -> list[str]:
    """One chunk's verdict, its signals, and the SAS that produced them."""
    lines = [
        f"### `{chunk.chunk_id}` — {chunk.kind or 'unknown'} "
        f"(lines {chunk.start_line}–{chunk.end_line})",
        "",
        f"- Tier **{chunk.tier}** · parity **{chunk.translation_difficulty}** "
        f"· score {chunk.score:.2f}",
        f"- {chunk.rationale or 'No signals fired.'}",
    ]
    if chunk.input_datasets or chunk.output_datasets:
        # What makes the file's Datasets section auditable: a reader who
        # doubts a rollup can find the chunk that put each name in it.
        lines.append(
            f"- Reads: {_fmt_list(chunk.input_datasets)} · "
            f"Writes: {_fmt_list(chunk.output_datasets)}"
        )
    if chunk.external_refs:
        # The same audit trail for the Paths section above.
        lines.append(f"- Paths: {_fmt_list(r.raw for r in chunk.external_refs)}")
    if chunk.signals:
        # Labelled and set apart, so the verdict bullets above and the
        # evidence bullets below do not read as one undifferentiated list.
        lines += ["", "Signals:", "", *(f"- {signal}" for signal in chunk.signals)]
    if not include_source:
        return [*lines, ""]

    key = (chunk.source_id or _INLINE, chunk.chunk_id)
    text = (texts or {}).get(key)
    if text is None:
        logger.warning(
            f"_chunk_section: no source text for {key}; rendering a placeholder"
        )
        lines += ["", "_Source text unavailable for this chunk._", ""]
        return lines
    return [*lines, "", *_source_block(text, max_lines=max_source_lines), ""]


# ---------------------------------------------------------------------------
# Overall (corpus) report
# ---------------------------------------------------------------------------


def _hydration_summary(plan: Any | None) -> list[str]:
    """The corpus-wide ingestion surface, or nothing without a plan.

    The number an operator wants before signing off a migration: how many tables
    move, how much of it is partitioned, and how much still needs a human.
    """
    if plan is None or not plan.items:
        return []
    kinds = ", ".join(f"{kind} ({count})" for kind, count in plan.counts_by_kind().items())
    partitioned = sum(1 for item in plan.items if item.partition is not None)
    lines = [
        "",
        "## Hydration",
        "",
        f"- Target tables: **{len(plan.target_tables)}** "
        f"from {len(plan.items)} load item(s)",
        f"- Sources: {kinds}",
        f"- Partitioned items: {partitioned}",
        f"- Run date stamped into target names: `{plan.run_date}`",
    ]
    if plan.blocked_count:
        lines.append(
            f"- ⚠️ **{plan.blocked_count} item(s) need operator input** before "
            f"they can run — see the per-file reports."
        )
    else:
        lines.append("- Every item can run unattended.")
    return lines


def render_overall_report(
    report: CorpusComplexityReport,
    *,
    top: int = 10,
    file_links: Mapping[str, str] | None = None,
    graph_image: str | None = None,
    hydration: Any | None = None,
) -> str:
    """The corpus report, with an index of the individual reports appended.

    The body is :meth:`CorpusComplexityReport.to_markdown` unchanged — this
    adds the index, so the overall report and the per-file ones are navigable
    as one deliverable. Without *file_links* it is the corpus report verbatim.

    *graph_image* is passed straight through: a path to a rendered dependency
    graph, relative to where this Markdown will be written.

    *hydration* is an optional ``data_hydration.HydrationPlan``, summarised
    before the file index. ``None`` renders nothing at all.
    """
    body = report.to_markdown(top=top, graph_image=graph_image)
    summary = _hydration_summary(hydration)
    if summary:
        body = body.rstrip() + "\n" + "\n".join(summary) + "\n"
    if not file_links:
        return body

    names = report.names

    lines = [
        body,
        "",
        f"## Individual reports ({len(file_links)})",
        "",
        "One report per source SAS script, each printing the SAS behind every "
        "verdict.",
        "",
        "| File | Size | Points | Report |",
        "| --- | --- | ---: | --- |",
    ]
    # Points are deck entries, so files tie on them constantly; the continuous
    # position orders them within a rung.
    by_points = sorted(
        report.files,
        key=lambda f: (f.points, f.continuous_points),
        reverse=True,
    )
    for f in by_points:
        link = file_links.get(f.source_id)
        cell = f"[{link}]({link})" if link else "—"
        lines.append(
            f"| {resolve_name(f.source_id, names)} | {f.size.label} "
            f"| {f.points:.1f} | {cell} |"
        )
    # A link with no matching FileComplexity would otherwise vanish silently.
    scored = {f.source_id for f in report.files}
    for source_id, link in file_links.items():
        if source_id not in scored:
            lines.append(
                f"| {resolve_name(source_id, names)} | — | — | [{link}]({link}) |"
            )
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Writing the set
# ---------------------------------------------------------------------------


def _stem_for(source_id: str, taken: dict[str, str]) -> str:
    """A unique, filesystem-safe stem for *source_id*.

    Two scripts with the same basename in different directories collide on
    stem alone, so the later one gets a numeric suffix rather than quietly
    overwriting the earlier one.
    """
    if source_id in taken:
        return taken[source_id]
    stem = Path(source_id).stem or "report"
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "report"
    candidate = stem
    used = set(taken.values())
    suffix = 2
    while candidate in used:
        candidate = f"{stem}_{suffix}"
        suffix += 1
    taken[source_id] = candidate
    return candidate


def source_stems(source_ids: Iterable[str]) -> dict[str, str]:
    """``source_id -> unique, filesystem-safe stem``, in the order given.

    Every per-source artefact — a report, a prompt, an index entry — names its
    file through this, so the same script keeps the same stem across all of
    them and two scripts sharing a basename never collide.
    """
    taken: dict[str, str] = {}
    for source_id in source_ids:
        _stem_for(source_id, taken)
    return taken


def file_report_paths(
    report: CorpusComplexityReport, out_dir: Path | str
) -> dict[str, Path]:
    """``source_id -> destination path`` for the individual reports.

    Exposed so a caller can name the outputs (a prompt bundle, an index page)
    without writing them first.
    """
    directory = Path(out_dir) / FILE_REPORT_DIR
    stems = source_stems(f.source_id for f in report.files)
    return {
        source_id: directory / f"{stem}.md" for source_id, stem in stems.items()
    }


def write_reports(
    report: CorpusComplexityReport,
    out_dir: Path | str,
    *,
    texts: ChunkTextIndex | None = None,
    top: int = 10,
    include_source: bool = True,
    max_source_lines: int = 0,
    overall_name: str = OVERALL_REPORT_NAME,
    graph_image: bool = True,
    hydration: Any | None = None,
) -> WrittenReports:
    """Write the overall report and one report per source file under *out_dir*.

    The overall report lands at ``out_dir/complexity-report.md`` and links to
    each individual report under ``out_dir/files/``. Returns the paths.

    With *graph_image* (the default) the dependency graph is also drawn to
    ``out_dir/dependency-graph.png`` and linked from the overall report — when
    matplotlib is installed and there is a graph to draw. The report's edge
    table is written either way.

    *hydration* is an optional ``data_hydration.HydrationPlan``: summarised in
    the overall report and broken down per file. Omitting it (the default)
    leaves every report exactly as it was before hydration existed.
    """
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    destinations = file_report_paths(report, directory)
    # Once, from the whole corpus, and handed to every renderer below: a file
    # named `etl/load.sas` in the overall table must not be `load.sas` in its
    # own report or in the graph image.
    names = report.names

    written: dict[str, Path] = {}
    for file in report.files:
        dest = destinations[file.source_id]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(
            render_file_report(
                file,
                texts=texts,
                target_display=report.target_display,
                include_source=include_source,
                max_source_lines=max_source_lines,
                overall_link=f"../{overall_name}",
                names=names,
                hydration=hydration,
            ),
            encoding="utf-8",
        )
        logger.debug(f"write_reports: wrote {dest}")
        written[file.source_id] = dest

    links = {
        source_id: dest.relative_to(directory).as_posix()
        for source_id, dest in written.items()
    }

    drawn: Path | None = None
    if graph_image and report.graph is not None:
        # Imported here rather than at module scope so that a report run
        # without the optional matplotlib dependency never even reaches the
        # import — and so this module stays renderer-only for Markdown.
        from .graph import render_png

        drawn = render_png(
            report.graph, directory / GRAPH_IMAGE_NAME, names=names
        )

    overall = directory / overall_name
    overall.write_text(
        render_overall_report(
            report,
            top=top,
            file_links=links,
            graph_image=(
                drawn.relative_to(directory).as_posix() if drawn else None
            ),
            hydration=hydration,
        ),
        encoding="utf-8",
    )
    logger.info(
        f"write_reports: wrote {overall} and {len(written)} individual "
        f"report(s) under {directory / FILE_REPORT_DIR}"
    )
    return WrittenReports(overall=overall, files=written, graph=drawn)
