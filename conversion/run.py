"""Running one conversion request end to end.

The orchestration between :mod:`conversion.requests` (which rows there are),
:mod:`conversion.sources` (their scripts), :class:`~pipeline.engine.SasLLMPipeline`
(the translation) and :mod:`conversion.upload` (where the output goes) — with
:func:`~conversion.requests.update_request_status` closing the row at the end.

It lives here rather than in ``main.py`` for the same reason
:mod:`complexity.sharepoint` lives beside ``complexity/__main__.py``: the CLI
should be argument parsing and dispatch, and everything worth testing should be
reachable without it. :func:`run_request` takes a transport and a pipeline
factory, so the whole flow runs against a fake.

The shape of a run
------------------
1. Read the request row's scripts from ``{base}/{app}/scripts_original``.
2. Apply the XREF cross-reference — physical paths over the raw text
   (:func:`xref.pre.rewrite_source_text`), dataset names through the batchers'
   ``databricks_mapping``, which the caller passes to the pipeline.
3. Translate the whole application as **one corpus on one thread**, so
   cross-file dataset and macro edges resolve.
4. Upload one file per source to
   ``{base}/{app}/scripts_converted/{model}/{timestamp}``, effective prompts
   under its ``prompts/`` subdirectory, and any validation artefacts.
5. Write the row's ``Status``, whichever way the run went.

**No temporary directory.** ``run_texts`` takes the text and the drive-relative
path that names it, which is the source id the notebooks and run facts are keyed
by; staging the corpus to disk just to have paths to pass would name every
source after a directory that no longer exists.

Logger name: ``conversion.run``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import app_config
from app_config.sharepoint import SharePointConfig, SharePointError

# Mode names only — the xref functions themselves are imported lazily at each
# call site, so this module stays importable without the rewriters' stack.
from xref.apply import APPLY_BOTH, APPLY_POST, APPLY_PRE

from .paths import validation as validation_folder
from .requests import ConversionItem, ConversionRequest
from .sources import load, source_files
from .upload import (
    upload_converted_script,
    upload_prompt_file,
    upload_validation_file,
)

logger = logging.getLogger(__name__)

#: Written to the request row's ``Status`` as a run starts and ends. Plain
#: words rather than codes: an operator reads this column in a browser.
STATUS_RUNNING = "In Progress"
STATUS_DONE = "Completed"
STATUS_FAILED = "Failed"

#: The validation artefacts uploaded beside the converted scripts.
SUMMARY_NAME = "summary.json"
REPORT_NAME = "report.md"
REPORT_PDF_NAME = "report.pdf"


# Re-exported so `conversion.run.utc_stamp` keeps working; the one
# implementation is app_config's, because the complexity CLI names its run
# folders with the same stamp and the two must not drift.
utc_stamp = app_config.utc_stamp


@dataclass
class RunOutcome:
    """What one request produced.

    Attributes
    ----------
    request : ConversionRequest
        The row this run was driven by.
    status : int
        ``0`` on success, non-zero otherwise — the shape a CLI exit status
        wants, so a multi-row run sums them without translating.
    model : str
        The model the row selected (or the configured default).
    timestamp : str
        The run folder's stamp.
    uploaded : list[str]
        Drive-relative paths of everything written.
    error : str | None
        Why it failed, when it did.
    """

    request: ConversionRequest
    status: int = 0
    model: str = ""
    timestamp: str = ""
    uploaded: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == 0


def select_requests(
    rows: list[ConversionRequest],
    *,
    request_id: str | None = None,
    application: str | None = None,
) -> list[ConversionRequest]:
    """
    *rows* narrowed by ``--request-id`` or ``--app``.

    Both compare as stripped text (and the application case-insensitively),
    because a list column routinely carries a trailing space that an exact
    match would silently miss — see the note in :mod:`xref.sourcing` on the
    reference's exact comparison doing exactly that.
    """
    if request_id is not None:
        wanted = request_id.strip()
        return [row for row in rows if str(row.item_id or "").strip() == wanted]
    if application is not None:
        wanted = application.strip().casefold()
        return [
            row
            for row in rows
            if (row.application_name or "").strip().casefold() == wanted
        ]
    return list(rows)


def model_for(
    request: ConversionRequest,
    items: list[ConversionItem],
    default: str,
) -> str:
    """
    The model this request's scripts are converted with.

    The conversions list holds one row per *script*, each naming a model, while
    this flow runs a whole application through one pipeline on one thread. So
    the first conversion row that names a model wins and *default* (config.json
    ``llm_client.model``) applies when none does. Picking per script would mean
    a pipeline per script, which would forfeit the cross-file batching that
    makes the translation coherent.
    """
    for item in items:
        if str(item.app_request_id or "") == str(request.item_id or ""):
            if item.preferred_llm:
                return item.preferred_llm
    return default


def _set_status(
    item_id: Any, status: str, *, client: Any, config: SharePointConfig | None
) -> None:
    """Write the row's ``Status``, never raising.

    A status write that fails must not turn a completed conversion into a
    failed one — the scripts are already uploaded, which is the thing that
    mattered — so this reports and moves on.
    """
    from .requests import update_request_status

    try:
        update_request_status(item_id, status, client=client, config=config)
    except SharePointError as exc:
        logger.warning(
            f"_set_status: could not write Status={status!r} on row {item_id!r}: {exc}"
        )


def run_request(
    request: ConversionRequest,
    *,
    build_pipeline: Callable[[str, bool], Any],
    model: str,
    client: Any,
    config: SharePointConfig | None = None,
    xref_mappings: Any | None = None,
    xref_mode: str | None = None,
    upload: bool = True,
    timestamp: str | None = None,
) -> RunOutcome:
    """
    Convert one application and upload the result.

    Parameters
    ----------
    request : ConversionRequest
        The row to run. Its ``application_name`` names both the folder read
        from and the folder written to.
    build_pipeline : Callable[[str, bool], Any]
        ``(model, validate) -> SasLLMPipeline``. Injected rather than
        constructed here so this module needs neither the LLM stack nor the
        reference corpus to be importable, and so a test can pass a fake.
    model : str
        The model id, already resolved by :func:`model_for`.
    xref_mappings : XrefMappings | None
        The application's cross-reference rows. When and where they are applied
        is *xref_mode*'s business. ``None`` disables the substitution entirely
        (``--no-xref``), whatever the mode says.
    xref_mode : str | None
        ``"pre"``, ``"post"`` or ``"both"``; ``None`` reads
        :func:`xref.apply.configured_mode` (``XREF_APPLY`` > ``config.json``
        ``xref.apply`` > ``"pre"``). ``"pre"`` rewrites the SAS source before
        chunking, ``"post"`` the generated code afterwards, ``"both"`` does
        each and reports what only post reached. Under ``"pre"`` and
        ``"both"`` the dataset half rides on the pipeline instead — the caller
        passes ``mappings.dataset_mapping`` as ``databricks_mapping``.
    upload : bool
        ``False`` converts and reports but writes nothing back, and leaves the
        row's ``Status`` alone. The dry run.

    Returns
    -------
    RunOutcome
        Never raises for an expected failure: a request list is a queue, and
        one bad row must not take the others down. Programming errors still
        propagate.
    """
    stamp = timestamp or utc_stamp()
    application = request.application_name
    outcome = RunOutcome(request=request, model=model, timestamp=stamp)
    logger.info(
        f"run_request: application={application!r} model={model!r} "
        f"validate={request.is_validation_required} upload={upload} "
        f"timestamp={stamp}"
    )

    try:
        sources = _load_sources(application, client=client, config=config)
    except SharePointError as exc:
        return _failed(outcome, f"could not read the source scripts: {exc}")
    if not sources:
        return _failed(
            outcome,
            f"no source scripts under {application!r}/scripts_original",
        )

    # Resolved once, before anything is converted: the mode decides both
    # whether the source is rewritten now and whether the output is rewritten
    # later, and reading it twice could straddle a config change mid-run.
    mode = _resolve_xref_mode(xref_mode, xref_mappings)
    sas_paths: list[Any] = []
    if xref_mappings is not None and mode in (APPLY_PRE, APPLY_BOTH):
        # Taken before the rewrite, so these are the SAS-side values — see
        # _report_unmapped_paths on why that is the useful moment.
        sas_paths = _sas_paths(sources)
        sources = _apply_xref(sources, xref_mappings)

    if upload:
        _set_status(request.item_id, STATUS_RUNNING, client=client, config=config)

    validating = bool(request.is_validation_required)
    try:
        pipeline = build_pipeline(model, validating)
        # The request id is the thread id, so this run's conversation memory,
        # run facts and verdicts are all keyed to the row that asked for it.
        outputs = pipeline.run_texts(
            sources, thread_id=str(request.item_id or application)
        )
    except Exception as exc:  # the row fails; the queue continues
        logger.exception(f"run_request: {application!r} failed during translation")
        if upload:
            _set_status(request.item_id, STATUS_FAILED, client=client, config=config)
        return _failed(outcome, f"translation failed: {exc}")

    if xref_mappings is not None and mode in (APPLY_POST, APPLY_BOTH):
        _apply_xref_post(
            outputs,
            xref_mappings,
            output_language=request.output_language or pipeline.output_language,
            report_only_post=mode == APPLY_BOTH,
        )
    # After the post pass, so a path the post pass fixed is not reported as
    # having escaped. Under "post" there is no SAS-side inventory to check
    # against — nothing rewrote the source, so every path in the output is
    # expected to be a SAS one until post has had its turn.
    _report_unmapped_paths(outputs, sas_paths)

    log_item_summaries(outputs)
    log_token_usage(pipeline)
    if validating:
        log_validation_summary(outputs)

    if not upload:
        logger.info(f"run_request: --no-upload; {len(outputs)} item(s) not written")
        return outcome

    try:
        outcome.uploaded = _upload_outputs(
            request,
            outputs,
            pipeline=pipeline,
            model=model,
            timestamp=stamp,
            validating=validating,
            client=client,
            config=config,
        )
    except SharePointError as exc:
        _set_status(request.item_id, STATUS_FAILED, client=client, config=config)
        return _failed(outcome, f"could not upload the converted scripts: {exc}")

    _set_status(request.item_id, STATUS_DONE, client=client, config=config)
    logger.info(
        f"run_request: {application!r} done — {len(outcome.uploaded)} file(s) uploaded"
    )
    return outcome


def _failed(outcome: RunOutcome, message: str) -> RunOutcome:
    logger.error(f"run_request: {outcome.request.application_name!r}: {message}")
    outcome.status = 1
    outcome.error = message
    return outcome


def _load_sources(
    application: str, *, client: Any, config: SharePointConfig | None
) -> list[tuple[str, str]]:
    """``(drive_relative_path, text)`` for every source script, sorted.

    The path is the *source id*: it names the file in the reports, in the
    notebooks and in the run facts, so it must be the library path rather than
    anything local.
    """
    paths = source_files(application, client=client, config=config)
    sources: list[tuple[str, str]] = []
    for path in paths:
        try:
            sources.append((path, load(path, client=client)))
        except SharePointError as exc:
            # One unreadable file must not lose the rest of the application.
            logger.warning(f"_load_sources: skipping {path!r}: {exc}")
    logger.info(
        f"_load_sources: {len(sources)} of {len(paths)} script(s) loaded for "
        f"{application!r}"
    )
    return sources


def _apply_xref(sources: list[tuple[str, str]], mappings: Any) -> list[tuple[str, str]]:
    """The physical-path half of the XREF substitution, over raw source text.

    The dataset half rides on the pipeline's ``databricks_mapping`` — see
    :mod:`xref.apply` on why "pre" has two halves at two stages.
    """
    from xref.pre import rewrite_source_text

    rewritten: list[tuple[str, str]] = []
    total = 0
    for source_id, text in sources:
        text, stats = rewrite_source_text(text, mappings, source_id=source_id)
        total += len(stats.rewritten)
        rewritten.append((source_id, text))
    if total:
        logger.info(f"_apply_xref: remapped {total} physical path(s) across the corpus")
    return rewritten


def _resolve_xref_mode(explicit: str | None, mappings: Any) -> str:
    """When to apply the substitution, or :data:`APPLY_PRE` when it is moot.

    *explicit* wins; otherwise :func:`xref.apply.configured_mode` reads
    ``XREF_APPLY`` then ``config.json`` ``xref.apply``. With no mappings at all
    the answer cannot matter, so the config is not consulted — a deployment
    that does not cross-reference should not have to be configured for a mode
    it never uses.
    """
    if mappings is None:
        return APPLY_PRE
    if explicit:
        return explicit.strip().lower()
    from xref.apply import configured_mode

    return configured_mode()


def _sas_paths(sources: list[tuple[str, str]]) -> list[Any]:
    """Every filesystem location the SAS corpus names, before any rewriting.

    Read straight off the text with :func:`chunker.paths.extract_paths` — no
    chunking pass, since that function takes text — over the comments-blanked
    form it documents as its input. Called *before* :func:`_apply_xref`, so
    these are the SAS-side values; that is what makes them useful afterwards,
    when one of them turning up in the generated code means no mapping reached
    it (:func:`_report_unmapped_paths`).

    Filesystem only: a ``PIPE`` command or an email address surviving into the
    output is not a mapping failure.
    """
    from chunker.models import PathLocation
    from chunker.paths import extract_paths
    from chunker.scanner import _sanitise

    seen: set[str] = set()
    found: list[Any] = []
    for _, text in sources:
        for ref in extract_paths(_sanitise(text, blank_strings=False)):
            if ref.location is not PathLocation.FILESYSTEM or ref.raw in seen:
                continue
            seen.add(ref.raw)
            found.append(ref)
    return found


def _report_unmapped_paths(outputs: list[dict], sas_paths: list[Any]) -> list[str]:
    """Warn about SAS paths that reached the generated code unmapped.

    A SAS-side path in the output means one of two things, and neither is
    visible without saying so: no XREF row covered it, or the model wrote it
    back after ``pre`` had already rewritten its source. Either way the
    notebook now names a location that does not exist on the target.

    Reported, never fatal — the same rule the rest of this module follows: a
    request list is a queue, and a wrong path is a fixable review finding
    rather than a reason to fail a conversion that otherwise worked.
    """
    if not sas_paths:
        return []
    rendered = "\n".join(_generated_text(out) for out in outputs)
    survivors = sorted({ref.raw for ref in sas_paths if ref.raw in rendered})
    if survivors:
        logger.warning(
            f"_report_unmapped_paths: {len(survivors)} SAS path(s) reached the "
            f"generated code unmapped ({', '.join(survivors[:5])}"
            f"{', ...' if len(survivors) > 5 else ''}); no XREF path row "
            f"addresses them, so the notebook names a location that will not "
            f"exist on the target"
        )
    return survivors


def _generated_text(out: dict) -> str:
    """Everything one output holds as text, for a substring search.

    The document when there is one, the raw response otherwise — the same
    precedence :func:`pipeline.notebook.item_cells` uses.
    """
    document = out.get("document")
    if document:
        cells = document.get("cells") or []
        return "\n".join(str(cell.get("source", "")) for cell in cells)
    return str(out.get("response", ""))


def _apply_xref_post(
    outputs: list[dict], mappings: Any, *, output_language: str, report_only_post: bool
) -> int:
    """Rewrite each item's generated code in place. Returns the items changed.

    Operates on the ``document`` — the structured
    :class:`~pipeline.response_models.TranslationDocument` the notebooks are
    built from — so a rewrite reaches the deliverable. Rewriting only the
    Markdown ``response`` would leave the uploaded notebook untouched.

    Dispatch is **per cell**, on the cell's own ``language``, not on the run's
    target: a PySpark run routinely contains a ``sql`` cell, and handing that to
    the Python parser would no-op under :mod:`xref.rewrite`'s hard rule — a
    silent miss rather than a visible failure.
    """
    from xref.apply import apply_both, apply_post

    changed = 0
    for out in outputs:
        document = out.get("document")
        if not document:
            # No structured document: fall back to the raw Markdown, which is
            # what item_cells renders from in that case.
            response = str(out.get("response", ""))
            if not response.strip():
                continue
            rewritten = apply_post(response, output_language, mappings)
            if rewritten != response:
                out["response"] = rewritten
                changed += 1
            continue

        cells = document.get("cells") or []
        item_changed = False
        for cell in cells:
            if cell.get("kind") != "code":
                continue
            source = str(cell.get("source", ""))
            if not source.strip():
                continue
            language = _cell_language(cell, output_language)
            if language is None:
                continue
            if report_only_post:
                outcome = apply_both(source, language, mappings)
                rewritten = outcome.code
            else:
                rewritten = apply_post(source, language, mappings)
            if rewritten != source:
                cell["source"] = rewritten
                item_changed = True
        if item_changed:
            changed += 1
    if changed:
        logger.info(f"_apply_xref_post: rewrote generated code in {changed} item(s)")
    return changed


def _cell_language(cell: dict, output_language: str) -> str | None:
    """The target language *cell* should be parsed as, or ``None`` to skip it.

    A run has one target; its cells do not. A ``sql`` cell in a PySpark run and
    a ``python`` cell in a Spark SQL run each have to be read as what they
    declare, because parsing one as the other does not fail loudly — it no-ops
    under :mod:`xref.rewrite`'s hard rule, which is a silent miss.

    The tag is resolved through :func:`target_language.resolve_target_language`
    rather than a table kept here, so the spellings a cell may legitimately
    carry (``sql``, ``databrickssql``, ``python``, ``py``, ...) are the ones the
    rest of the run already accepts — invariant 11's "resolved once, and not
    re-derived downstream" applied to the cell rather than the run.

    An untagged cell is the run's target. A cell tagged with something no target
    claims — ``scala``, ``r``, a typo — returns ``None``: guessing at the run's
    target would parse it as a language it is not and report nothing, so it is
    logged and left exactly as written instead.
    """
    from target_language import UnknownTargetLanguage, resolve_target_language

    tag = str(cell.get("language") or "").strip()
    if not tag:
        return output_language
    try:
        return resolve_target_language(tag).display_name
    except UnknownTargetLanguage:
        logger.warning(
            f"_cell_language: cell declares language {tag!r}, which is not a "
            f"known target; leaving the cell as written rather than parsing it "
            f"as {output_language}"
        )
        return None


def _upload_outputs(
    request: ConversionRequest,
    outputs: list[dict],
    *,
    pipeline: Any,
    model: str,
    timestamp: str,
    validating: bool,
    client: Any,
    config: SharePointConfig | None,
) -> list[str]:
    """Write notebooks, effective prompts, and optional validation artifacts."""
    from pipeline.artifacts import prompt_artifacts_from_outputs
    from pipeline.notebook import notebook_to_json, notebooks_from_outputs

    application = request.application_name
    output_language = request.output_language or pipeline.output_language
    written: list[str] = []

    notebooks = notebooks_from_outputs(outputs, output_language=output_language)
    for name, notebook in notebooks.items():
        written.append(
            upload_converted_script(
                application,
                name,
                output_language,
                notebook_to_json(notebook),
                model,
                timestamp,
                client=client,
                config=config,
            )
        )

    for name, contents in prompt_artifacts_from_outputs(outputs).items():
        written.append(
            upload_prompt_file(
                application,
                name,
                contents,
                model,
                timestamp,
                client=client,
                config=config,
            )
        )

    if not validating:
        return written

    written.extend(
        _upload_validation(
            request,
            outputs,
            pipeline=pipeline,
            model=model,
            client=client,
            config=config,
        )
    )
    return written


def _upload_validation(
    request: ConversionRequest,
    outputs: list[dict],
    *,
    pipeline: Any,
    model: str,
    client: Any,
    config: SharePointConfig | None,
) -> list[str]:
    """Per-item verdicts, the aggregate summary, and the rendered report."""
    application = request.application_name
    written: list[str] = []

    for out in outputs:
        verdict = out.get("validation")
        if verdict is None:
            continue
        written.append(
            upload_validation_file(
                application,
                f"{out['item_id']}.json",
                json.dumps(verdict, indent=2),
                client=client,
                config=config,
            )
        )

    usage = pipeline.token_usage
    written.append(
        upload_validation_file(
            application,
            SUMMARY_NAME,
            json.dumps(validation_summary(outputs, token_usage=usage), indent=2),
            client=client,
            config=config,
        )
    )

    report = validation_report(
        outputs,
        model=model,
        instructions_fingerprint=pipeline.instructions_fingerprint,
        policy_fingerprint=pipeline.policy_fingerprint,
        token_usage=usage,
    )
    written.append(
        upload_validation_file(
            application,
            REPORT_NAME,
            report.to_markdown(),
            client=client,
            config=config,
        )
    )

    from validation import report_to_pdf

    written.append(
        upload_validation_file(
            application,
            REPORT_PDF_NAME,
            report_to_pdf(report),
            client=client,
            config=config,
        )
    )
    logger.info(
        f"_upload_validation: {len(written)} artefact(s) under "
        f"{validation_folder(application, config=config)!r}"
    )
    return written


# ---------------------------------------------------------------------------
# Run reporting — what a run produced, for the log and for the library
# ---------------------------------------------------------------------------


def item_header(out: dict) -> str:
    """The ``=== id files=[...] [verdict] ===`` banner for one item."""
    return f"=== {out['item_id']} files={out['source_files']}{format_verdict(out.get('validation'))} ==="


def format_verdict(validation: dict | None) -> str:
    """``  [PASS score=0.95]`` for one item's stored verdict, or ``""``."""
    if not validation:
        return ""
    status = "PASS" if validation["passed"] else "FAIL"
    return f"  [{status} score={validation['score']:.2f}]"


def log_item_summaries(outputs: list[dict]) -> None:
    """One INFO block listing every item's banner line."""
    import io

    from chunker._repl import print_iterable

    buf = io.StringIO()
    print_iterable(
        (item_header(out) for out in outputs),
        label=f"pipeline produced {len(outputs)} item response(s)",
        stream=buf,
    )
    logger.info(buf.getvalue().rstrip())


def log_validation_summary(outputs: list[dict]) -> None:
    """Aggregate the per-item inline verdicts and log pass/fail counts.

    The verdicts also live in the run's conversation memory
    (``pipeline.get_validation_facts(thread_id)``); this is the run-end read of
    what was scored inline.
    """
    verdicts = [o["validation"] for o in outputs if o.get("validation")]
    if not verdicts:
        logger.warning("inline validation on, but no verdicts were recorded")
        return
    passed = sum(1 for v in verdicts if v["passed"])
    mean_score = sum(v["score"] for v in verdicts) / len(verdicts)
    logger.info(
        f"inline validation: {passed}/{len(verdicts)} item(s) passed  "
        f"mean_score={mean_score:.3f}"
    )
    for out in outputs:
        verdict = out.get("validation")
        if verdict and not verdict["passed"]:
            failed = [
                m["metric"]
                for m in verdict["metrics"]
                if not m["passed"] and not m["skipped"]
            ]
            logger.warning(
                f"  FAIL {out['item_id']}  score={verdict['score']:.3f}  "
                f"metrics_below_threshold={failed}"
            )


def log_token_usage(pipeline: Any) -> None:
    """Log what the run cost in tokens.

    Silent when the gateway reported no usage — ``llm_client`` has already
    warned about that once, and repeating it would say nothing new.
    """
    usage = pipeline.token_usage
    if not usage.calls:
        return
    logger.info(f"token usage: {usage.summary()}")
    if usage.cache_read_tokens:
        cached = usage.cache_read_tokens / max(1, usage.input_tokens)
        logger.info(f"  prompt cache served {cached:.0%} of input tokens")


def validation_report(
    outputs: list[dict],
    *,
    model: str,
    instructions_fingerprint: str | None = None,
    policy_fingerprint: str | None = None,
    token_usage: Any = None,
) -> Any:
    """A ``ValidationReport`` reconstructed from the run's inline verdicts.

    Turns the per-item verdicts ``LiveValidator`` produced into the same
    aggregate an offline run yields, so the inline run renders to Markdown and
    PDF. *token_usage* is the pipeline's spend, which the verdicts cannot carry
    (they are per item); there is no judge figure, inline validation being the
    deterministic suite.
    """
    from validation import report_from_verdicts

    verdicts = [o["validation"] for o in outputs if o.get("validation")]
    return report_from_verdicts(
        verdicts,
        model=model,
        instructions_fingerprint=instructions_fingerprint,
        policy_fingerprint=policy_fingerprint,
        token_usage=token_usage,
    )


def validation_summary(outputs: list[dict], *, token_usage: Any = None) -> dict:
    """The aggregate pass/fail summary uploaded alongside the run."""
    verdicts = [o["validation"] for o in outputs if o.get("validation")]
    passed = sum(1 for v in verdicts if v["passed"])
    mean_score = sum(v["score"] for v in verdicts) / len(verdicts) if verdicts else None
    return {
        "items": len(verdicts),
        "passed": passed,
        "failed": len(verdicts) - passed,
        "mean_score": mean_score,
        # None when the gateway reported no usage, so a consumer can tell
        # "not reported" from "reported as zero".
        "token_usage": (
            token_usage.model_dump()
            if token_usage is not None and token_usage.calls
            else None
        ),
        "per_item": [
            {
                "item_id": o["item_id"],
                "passed": o["validation"]["passed"],
                "score": o["validation"]["score"],
            }
            for o in outputs
            if o.get("validation")
        ],
    }
