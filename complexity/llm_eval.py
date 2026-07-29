"""An optional second opinion: ask an LLM to evaluate what the rules cannot.

The static analysis in this package is deliberately rule-driven and
presence-based. That is its strength — it is reproducible, offline, and every
verdict names the catalogue entry behind it — and also its ceiling. It counts
constructs, not *intent*, so there are whole classes of migration work it
cannot see:

- business rules packed into one arithmetic-heavy DATA step, which scores zero;
- a ``DO`` loop that is really a join, or a ``RETAIN`` that is really a running
  total — both rated by construct, neither by what they compute;
- hard-coded paths, dates, and environment assumptions;
- dead steps whose output nothing consumes, which inflate every count.

So this module builds a prompt that hands an LLM the static verdict **and** the
SAS source behind it, and asks for exactly the judgements the rules cannot
make. The static analysis is the ground the prompt stands on, not something the
LLM is asked to reproduce: the numbers are given, and the question is where
they are wrong or incomplete.

Nothing here is required. ``python -m complexity`` stays offline unless
``--llm-eval`` is passed, and no other module in this package imports this one.

**Decoupling.** The *llm* argument is anything with a LangChain-style
``invoke(input) -> message`` — an :class:`llm_client.LLMClient`, a raw chat
model, or a fake in tests — and structured output is used only when the object
offers it. So this package gains no dependency on ``llm_client``, and none at
all on ``pipeline``: a complexity evaluation never touches the translation
path, and the severity vocabulary below (P0/P1/P2) is restated rather than
imported from :mod:`pipeline.response_models` for that reason.

Logger name: ``complexity.llm_eval``.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Iterable, Literal

from pydantic import BaseModel, Field

from .models import CorpusComplexityReport, FileComplexity
from .report import ChunkTextIndex, _fence, _fmt_list, _source_block

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# What we ask the LLM for
# ---------------------------------------------------------------------------


class EvaluationFinding(BaseModel):
    """One hazard the static analysis did not raise, or under-rated."""

    severity: Literal["P0", "P1", "P2"] = Field(
        description="P0 for a silent-error risk (the translation runs and "
        "produces wrong results), P1 for a loud failure, P2 for a "
        "maintainability or performance concern."
    )
    note: str = Field(
        description="What the hazard is and what a migrator must check. Name "
        "the chunk id and the SAS statement it comes from."
    )


class FileEvaluation(BaseModel):
    """An LLM's assessment of one source file's migration effort.

    Deliberately not a re-scoring: the model is asked whether the measured size
    is right and *why*, so a disagreement arrives as an argument a human can
    check rather than as a competing number with no provenance.
    """

    summary: str = Field(
        description="What this file does, in two or three sentences — its "
        "business purpose, not a restatement of its syntax."
    )
    size_verdict: Literal["agree", "larger", "smaller"] = Field(
        description="Whether the static T-shirt size is right, too small "
        "('larger'), or too big ('smaller')."
    )
    size_rationale: str = Field(
        description="Why. Point at specific code — a step that hides a lot of "
        "logic, or bulk that is really the same trivial step repeated."
    )
    findings: list[EvaluationFinding] = Field(
        default_factory=list,
        description="Hazards the construct-level rules missed or under-rated, "
        "worst first. Omit anything the static drivers table already states.",
    )
    manual_steps: list[str] = Field(
        default_factory=list,
        description="Work a human must do by hand rather than review — "
        "decisions the translation cannot make on its own.",
    )
    suggested_split: list[str] = Field(
        default_factory=list,
        description="If this file should be broken up, the chunk ids to cut "
        "between, each with a few words on why that seam. Empty otherwise.",
    )
    open_questions: list[str] = Field(
        default_factory=list,
        description="What you would need to see — another file, a data "
        "dictionary, a schedule — to raise your confidence.",
    )


class FileEvaluationResult(BaseModel):
    """The evaluation of one file, alongside the static verdict it argues with.

    ``evaluation`` is ``None`` when the model's reply could not be parsed into
    :class:`FileEvaluation`; the reply is then kept in ``prose`` and the reason
    in ``error``. A corpus run finishes either way — one unusable reply must
    not cost the other files their evaluations.
    """

    source_id: str
    static_size: str = ""
    static_points: float = 0.0
    evaluation: FileEvaluation | None = None
    prose: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        """Whether a structured evaluation came back."""
        return self.evaluation is not None

    def to_markdown(self) -> str:
        """This file's evaluation as a Markdown section."""
        head = (
            f"## {self.source_id}",
            "",
            f"Static verdict: **{self.static_size or 'unknown'}**, "
            f"{self.static_points:.1f} points.",
            "",
        )
        if self.evaluation is None:
            body = [
                f"_Structured evaluation unavailable: {self.error or 'unknown error'}._",
            ]
            if self.prose.strip():
                fence = _fence(self.prose)
                body += ["", "The model's reply, unparsed:", "", fence,
                         self.prose.strip(), fence]
            return "\n".join([*head, *body])

        ev = self.evaluation
        verdict = {
            "agree": "agrees with the static size",
            "larger": "rates this **larger** than the static size",
            "smaller": "rates this **smaller** than the static size",
        }[ev.size_verdict]
        lines = [
            *head,
            ev.summary.strip(),
            "",
            f"**Size** — {verdict}. {ev.size_rationale.strip()}",
        ]
        if ev.findings:
            lines += ["", "**Findings**", ""]
            for finding in ev.findings:
                marker = "⚠️ " if finding.severity == "P0" else ""
                lines.append(
                    f"- {marker}**{finding.severity}** — {finding.note.strip()}"
                )
        lines += _bullets("Manual steps", ev.manual_steps)
        lines += _bullets("Suggested split", ev.suggested_split)
        lines += _bullets("Open questions", ev.open_questions)
        return "\n".join(lines)

    def __str__(self) -> str:
        state = "ok" if self.ok else f"failed ({self.error})"
        return f"FileEvaluationResult({self.source_id}, {state})"


class ComplexityEvaluation(BaseModel):
    """Every file's LLM evaluation for one corpus."""

    target: str = ""
    target_display: str = ""
    model: str = ""
    files: list[FileEvaluationResult] = Field(default_factory=list)

    @property
    def failures(self) -> list[FileEvaluationResult]:
        """Files whose reply could not be parsed."""
        return [f for f in self.files if not f.ok]

    def for_source(self, source_id: str) -> FileEvaluationResult | None:
        """This corpus's evaluation of *source_id*, or ``None``."""
        for file in self.files:
            if file.source_id == source_id:
                return file
        return None

    def to_markdown(self) -> str:
        """The whole evaluation as one Markdown document."""
        lines = [
            "# LLM complexity evaluation",
            "",
            f"- Target: **{self.target_display or self.target or 'unknown'}**",
            f"- Model: {self.model or 'unknown'}",
            f"- Files evaluated: {len(self.files)}",
        ]
        if self.failures:
            lines.append(
                f"- Unparseable replies: {len(self.failures)} "
                f"({_fmt_list(f.source_id for f in self.failures)})"
            )
        lines += [
            "",
            "This is a second opinion on the static analysis, not a "
            "replacement for it: the model was given each file's measured "
            "verdict and its SAS source, and asked where the rules are wrong "
            "or incomplete. Treat a disagreement as an argument to check, not "
            "as a corrected number.",
            "",
        ]
        for file in self.files:
            lines += [file.to_markdown(), ""]
        return "\n".join(lines).rstrip() + "\n"

    def __str__(self) -> str:
        return (
            f"ComplexityEvaluation(model={self.model!r}, "
            f"files={len(self.files)}, failed={len(self.failures)})"
        )


def _bullets(heading: str, values: list[str]) -> list[str]:
    if not values:
        return []
    return ["", f"**{heading}**", "", *(f"- {v.strip()}" for v in values)]


# ---------------------------------------------------------------------------
# The prompt
# ---------------------------------------------------------------------------


_PROMPT_TEMPLATE = """\
You are a senior data engineer planning the migration of a SAS codebase to \
{target}. You are reviewing one source file.

A static analyser has already scored this file. Its verdict, its evidence, and \
the file's full SAS source are below. Your job is the part the analyser cannot \
do.

The analyser is rule-driven and presence-based: it recognises constructs from a \
fixed catalogue and sizes a file from how many there are, how hard each one is \
against the target, and how much it could not resolve. So it is reliable about \
syntax and blind to intent. It cannot see business rules packed into one \
arithmetic-heavy DATA step (which scores zero), a DO loop that is really a \
join, a RETAIN that is really a running total, hard-coded paths or dates, or a \
step whose output nothing consumes. Those are what you are being asked for.

Do not re-derive the numbers below and do not restate the drivers table. Argue \
with it.

# Static verdict — {source_id}

- Size: {size} ({points:.1f} story points), tier {tier}, parity {parity}
- Effort {effort_raw:.1f} ({effort_norm:.2f}), complexity {complexity_raw:.1f} \
({complexity_norm:.2f}), uncertainty {uncertainty_raw:.1f} \
({uncertainty_norm:.2f}); blend {blend:.2f}
- {chunk_count} chunk(s) across {line_count} line(s)
- Rationale: {rationale}
{extra_facts}
## Constructs the analyser recognised

{drivers}

## Cross-file coupling

{cross_file}

# SAS source

{source}

# What to report

1. **Purpose** — what this file is for, in business terms.
2. **Size** — is the measured size right? Say `larger` if the file hides more \
work than its constructs suggest, `smaller` if its bulk is the same trivial \
step repeated. Justify it against specific code.
3. **Findings** — hazards the construct rules missed or under-rated, worst \
first. P0 means the translation would run and produce wrong results; P1 means \
it would fail loudly; P2 is maintainability or performance. Name the chunk id \
and the statement.
4. **Manual steps** — what a human must decide rather than review.
5. **Split points** — if this file should be broken up, which chunk ids to cut \
between and why that seam.
6. **Open questions** — what you would need to see to be more confident.

Ground every claim in something you can point at in the source above. If the \
source does not support a judgement, say so instead of guessing — an honest \
"cannot tell from this file alone" is more useful here than a confident \
invention.
"""


def build_evaluation_prompt(
    file: FileComplexity,
    *,
    texts: ChunkTextIndex | None = None,
    target_display: str = "",
    include_source: bool = True,
    max_source_lines: int = 0,
) -> str:
    """The evaluation prompt for one source file.

    Pure and offline — building a prompt never calls anything. Exposed on its
    own so the prompt can be inspected, diffed, and tuned (``--prompt-only``)
    without spending a call, and so a caller can send it through its own
    client.
    """
    extra: list[str] = []
    if file.floored_by:
        extra.append(
            f"- Size was floored by a `{file.floored_by}` chunk; the numbers "
            f"alone rated it smaller."
        )
    if not file.uncertainty_complete:
        extra.append(
            "- Uncertainty is a lower bound: parser diagnostics were not "
            "available to this run."
        )
    if file.needs_breakdown:
        extra.append(
            "- The analyser rates this Extra Large, which means it should be "
            "split before it is estimated."
        )

    if file.signals:
        drivers = "\n".join(
            f"- `{s.name}` [{s.tier}/{s.parity}] — {s.detail or 'no detail'}"
            for s in file.signals
        )
    else:
        drivers = (
            "None. No construct in this file is in the catalogue, so its size "
            "comes from volume alone — read the source with that in mind."
        )

    profile = file.cross_file
    if profile is None:
        cross_file = "Not resolved — this file was scored in isolation."
    elif profile.imports or profile.exports or profile.unresolved:
        cross_file = "\n".join(
            [
                f"- Depends on: {_fmt_list(profile.depends_on)}",
                f"- Depended on by: {_fmt_list(profile.depended_on_by)}",
                f"- Imports: {_fmt_list(profile.imports)}",
                f"- Exports: {_fmt_list(profile.exports)}",
                f"- Unresolved: {_fmt_list(profile.unresolved)}",
            ]
        )
    else:
        cross_file = "Nothing crosses this file's boundary."

    source_lines: list[str] = []
    for chunk in sorted(file.chunks, key=lambda c: (c.start_line, c.chunk_id)):
        source_lines += [
            f"## `{chunk.chunk_id}` — {chunk.kind or 'unknown'} "
            f"(lines {chunk.start_line}–{chunk.end_line}), tier {chunk.tier}, "
            f"parity {chunk.translation_difficulty}",
            "",
        ]
        if include_source:
            key = (chunk.source_id or "<inline>", chunk.chunk_id)
            text = (texts or {}).get(key)
            if text is None:
                logger.warning(
                    f"build_evaluation_prompt: no source text for {key}"
                )
                source_lines += ["_Source unavailable._", ""]
            else:
                source_lines += [
                    *_source_block(text, max_lines=max_source_lines),
                    "",
                ]
    source = "\n".join(source_lines).strip() or "_No source available._"

    prompt = _PROMPT_TEMPLATE.format(
        target=target_display or file.target or "the target language",
        source_id=file.source_id,
        size=file.size.label,
        points=file.points,
        tier=file.tier,
        parity=file.translation_difficulty,
        effort_raw=file.effort_raw,
        effort_norm=file.effort_norm,
        complexity_raw=file.complexity_raw,
        complexity_norm=file.complexity_norm,
        uncertainty_raw=file.uncertainty_raw,
        uncertainty_norm=file.uncertainty_norm,
        blend=file.blend,
        chunk_count=file.chunk_count,
        line_count=file.line_count,
        rationale=file.rationale or "no signals fired",
        extra_facts=("\n".join(extra) + "\n") if extra else "",
        drivers=drivers,
        cross_file=cross_file,
        source=source,
    )
    logger.debug(
        f"build_evaluation_prompt: {file.source_id}  chars={len(prompt)}  "
        f"chunks={len(file.chunks)}"
    )
    return prompt


def evaluation_prompts(
    report: CorpusComplexityReport,
    *,
    texts: ChunkTextIndex | None = None,
    include_source: bool = True,
    max_source_lines: int = 0,
    sources: Iterable[str] | None = None,
) -> dict[str, str]:
    """``source_id -> prompt`` for every file in *report* (or just *sources*)."""
    wanted = set(sources) if sources is not None else None
    return {
        f.source_id: build_evaluation_prompt(
            f,
            texts=texts,
            target_display=report.target_display,
            include_source=include_source,
            max_source_lines=max_source_lines,
        )
        for f in report.files
        if wanted is None or f.source_id in wanted
    }


# ---------------------------------------------------------------------------
# Invocation
# ---------------------------------------------------------------------------


def evaluate_file(
    llm: Any,
    file: FileComplexity,
    *,
    texts: ChunkTextIndex | None = None,
    target_display: str = "",
    include_source: bool = True,
    max_source_lines: int = 0,
) -> FileEvaluationResult:
    """Ask *llm* to evaluate one file; never raises on a bad reply.

    A model that cannot honour the schema, or one whose reply will not parse,
    yields a result with ``evaluation=None`` carrying the prose and the reason
    — the same bargain the pipeline strikes with structured output, for the
    same reason: a usable prose answer beats an exception.
    """
    prompt = build_evaluation_prompt(
        file,
        texts=texts,
        target_display=target_display,
        include_source=include_source,
        max_source_lines=max_source_lines,
    )
    result = FileEvaluationResult(
        source_id=file.source_id,
        static_size=file.size.label,
        static_points=file.points,
    )
    try:
        evaluation, prose, error = _invoke(llm, prompt)
    except Exception as exc:  # the call itself failed — network, auth, budget
        logger.error(f"evaluate_file: {file.source_id} failed: {exc!r}")
        return result.model_copy(update={"error": repr(exc)})
    logger.info(
        f"evaluate_file: {file.source_id} -> "
        f"{'ok' if evaluation is not None else f'unparsed ({error})'}"
    )
    return result.model_copy(
        update={"evaluation": evaluation, "prose": prose, "error": error}
    )


def evaluate_report(
    llm: Any,
    report: CorpusComplexityReport,
    *,
    texts: ChunkTextIndex | None = None,
    include_source: bool = True,
    max_source_lines: int = 0,
    limit: int = 0,
    model: str = "",
) -> ComplexityEvaluation:
    """Evaluate every file in *report* — one call per file.

    Per file rather than per corpus: a whole corpus rarely fits one prompt, and
    a file is the unit the sizing is stated in, so a per-file answer lines up
    with the verdict it is arguing with.

    *limit* evaluates only the *n* largest files (0, the default, evaluates all
    of them), because every file is a paid call and the largest are where a
    second opinion is worth most.
    """
    files = sorted(report.files, key=lambda f: f.points, reverse=True)
    if limit > 0:
        files = files[:limit]
    logger.info(
        f"evaluate_report: evaluating {len(files)} of {len(report.files)} "
        f"file(s) against {report.target!r}"
    )
    results = [
        evaluate_file(
            llm,
            file,
            texts=texts,
            target_display=report.target_display,
            include_source=include_source,
            max_source_lines=max_source_lines,
        )
        for file in files
    ]
    evaluation = ComplexityEvaluation(
        target=report.target,
        target_display=report.target_display,
        model=model or _model_name(llm),
        files=results,
    )
    logger.info(f"evaluate_report: {evaluation}")
    return evaluation


def _model_name(llm: Any) -> str:
    """Best-effort model id, for the report header. Never raises."""
    config = getattr(llm, "config", None)
    for candidate in (
        getattr(config, "model", None),
        getattr(llm, "model_name", None),
        getattr(llm, "model", None),
    ):
        if isinstance(candidate, str) and candidate:
            return candidate
    return type(llm).__name__


def _invoke(llm: Any, prompt: str) -> tuple[FileEvaluation | None, str, str]:
    """Send *prompt*; return ``(evaluation, prose, error)``.

    Structured output is used when the object offers it and will accept the
    schema, so an :class:`llm_client.LLMClient` gets a typed answer while a
    plain chat model or a test fake still works through ``invoke``. Either way
    the reply text is kept, so a failure to parse never loses the answer.
    """
    envelope = None
    if hasattr(llm, "invoke_structured"):
        supports = getattr(llm, "supports_structured_output", None)
        if supports is None or supports(FileEvaluation):
            envelope = llm.invoke_structured(FileEvaluation, prompt)
    if envelope is not None:
        parsed = envelope.get("parsed") if isinstance(envelope, dict) else envelope
        if isinstance(parsed, FileEvaluation):
            return parsed, "", ""
        raw = envelope.get("raw") if isinstance(envelope, dict) else None
        prose = _text_of(raw)
        error = str(
            (envelope.get("parsing_error") if isinstance(envelope, dict) else None)
            or "the model returned no parsed object"
        )
        # The gateway declined the schema but the prose may still be an answer:
        # try to read it before giving up.
        recovered, parse_error = _parse_evaluation(prose)
        if recovered is not None:
            return recovered, prose, ""
        return None, prose, f"{error}; {parse_error}" if parse_error else error

    prose = _text_of(llm.invoke(prompt))
    recovered, parse_error = _parse_evaluation(prose)
    if recovered is not None:
        return recovered, "", ""
    return None, prose, parse_error or "no JSON object in the reply"


def _text_of(reply: Any) -> str:
    """The text of a LangChain message, a string, or anything else."""
    if reply is None:
        return ""
    return str(getattr(reply, "content", reply))


# A fenced JSON block, which is how a model asked for JSON usually answers.
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _parse_evaluation(text: str) -> tuple[FileEvaluation | None, str]:
    """Best-effort :class:`FileEvaluation` from prose; ``(None, why)`` on failure."""
    if not text.strip():
        return None, "empty reply"
    for candidate in _json_candidates(text):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        try:
            return FileEvaluation.model_validate(data), ""
        except Exception as exc:
            return None, f"reply did not match the schema: {exc!r}"
    return None, "no JSON object in the reply"


def _json_candidates(text: str) -> list[str]:
    """Substrings of *text* worth trying as JSON, most likely first."""
    candidates = [m.group(1) for m in _JSON_FENCE_RE.finditer(text)]
    stripped = text.strip()
    if stripped.startswith("{"):
        candidates.append(stripped)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    return candidates
