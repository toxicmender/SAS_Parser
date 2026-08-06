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
   ``{base}/{app}/scripts_converted/{model}/{timestamp}``, and any validation
   artefacts beside them.
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

from .paths import validation as validation_folder
from .requests import ConversionItem, ConversionRequest
from .sources import load, source_files
from .upload import upload_converted_script, upload_validation_file

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
            f"_set_status: could not write Status={status!r} on row "
            f"{item_id!r}: {exc}"
        )


def run_request(
    request: ConversionRequest,
    *,
    build_pipeline: Callable[[str, bool], Any],
    model: str,
    client: Any,
    config: SharePointConfig | None = None,
    xref_mappings: Any | None = None,
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
        Applied to the raw source text before chunking. The dataset half of
        the substitution rides on the pipeline instead — the caller passes
        ``mappings.dataset_mapping`` as ``databricks_mapping``.
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

    if xref_mappings is not None:
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
        f"run_request: {application!r} done — {len(outcome.uploaded)} file(s) "
        f"uploaded"
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


def _apply_xref(
    sources: list[tuple[str, str]], mappings: Any
) -> list[tuple[str, str]]:
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
    """Write the notebooks, and the validation artefacts when there are any."""
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
    mean_score = (
        sum(v["score"] for v in verdicts) / len(verdicts) if verdicts else None
    )
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
