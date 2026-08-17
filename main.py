"""Convert SAS to a Spark target — from SharePoint by default, or local files.

Two flows, chosen by whether a source directory is given:

**SharePoint (the default).** With no positional argument, the run is driven by
the SharePoint lists: every *pending* row of the requests list is converted, its
scripts read from ``{base}/{app}/scripts_original`` and the output written to
``{base}/{app}/scripts_converted/{model}/{timestamp}``. The row's ``Status`` is
written as the run starts and again when it ends. ``--request-id`` or ``--app``
narrows which rows are taken.

**Local (the fallback).** Give a directory of ``.sas`` files and everything
happens on disk: no lists, no library, no ``Status``. This is the offline path
for development and for a corpus that never went into SharePoint.

The choice is explicit either way — a positional path means local, its absence
means SharePoint. Nothing falls back silently, because converting the wrong
corpus because a config key was missing is worse than a clear error.

Usage
-----
::

    python main.py                              # every pending request row
    python main.py --request-id 42              # one row
    python main.py --app "MyApp"                # one application's rows
    python main.py --no-upload                  # dry run: convert, write nothing
    python main.py --check                      # preflight only: read, convert nothing
    python main.py path/to/sas --out-dir out/   # the local fallback
    python main.py path/to/sas --md report.md --pdf report.pdf

Debugging a SharePoint deployment
---------------------------------
``--check`` is the place to start: it resolves the configuration (saying which
environment variable or ``config.json`` key each value came from), mints a
Graph token and reports the permissions actually granted to the app
registration, then reads the library and each configured list — writing
nothing and paying no LLM. ``python -m app_config.sharepoint_check`` is the
same preflight standalone, and takes ``--offline`` to check the configuration
with no network at all.

``--log-file`` captures a run (secrets redacted), and ``--trace-http`` adds the
individual Graph requests. Plain ``--debug`` deliberately leaves the transport
libraries at INFO so the pipeline's own lines stay readable.

Installed as the ``sas-parser`` console script, so ``sas-parser --app MyApp``
is the same command.

Credentials
-----------
The LLM key resolves in this order, and only the first that applies is tried:

1. ``--vault-secret PATH`` — an explicit AppRole read of a named Vault secret
   (:meth:`llm_client.LLMClientConfig.from_vault_secret`).
2. **The AI Gateway chain**, the default whenever Vault is configured: an Entra
   ID service principal (from the Databricks secret scope) mints a JWT, that
   JWT logs in to Vault, and the secret it unlocks is the gateway token
   (:meth:`llm_client.LLMClientConfig.from_ai_gateway`). This is also where the
   ``ai-gateway-version`` header and the gateway's rate-limit pacing come from,
   which is why it is one call rather than a hand-assembled config.
3. ``OPENAI_API_KEY`` from the environment — the local-development path, taken
   when no Vault settings are present or ``--no-gateway-auth`` is passed. The
   gateway speaks the OpenAI protocol for every model it fronts, so that is the
   variable whatever ``--model`` names.

The chain is walked **once per invocation** and the resulting config is reused
for every row, so a ten-row run does not log in to Vault ten times.

SharePoint needs the ``sharepoint`` extra (``uv pip install -e ".[sharepoint]"``)
and a configured site: ``SHAREPOINT_SITE_ID`` (or ``SHAREPOINT_SITE_HOSTNAME``
plus ``SHAREPOINT_SITE_PATH``), ``SHAREPOINT_LIST_ID_SAS_REQUESTS``, and — for
the cross-reference — ``SHAREPOINT_LIST_ID_XREF``. All of these may live in
``config.json`` instead; see ``.env.example``.

Logger name: ``main``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import app_config
from app_config.logging_setup import configure_logging
from llm_client import LLMClientConfig
from target_language import (
    DEFAULT_OUTPUT_LANGUAGE,
    KNOWN_TARGETS,
    resolve_target_language,
)

logger = logging.getLogger("main")

_DEFAULT_MODEL = "gpt-5.4"


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="sas-parser",
        description=__doc__.splitlines()[0] if __doc__ else None,
    )
    parser.add_argument(
        "sas_dir",
        type=Path,
        nargs="?",
        default=None,
        help="Directory of .sas files to convert locally (searched "
        "recursively). Omit it to run from the SharePoint request list, which "
        "is the default.",
    )
    parser.add_argument(
        "--pattern",
        default="*.sas",
        help="Glob for SAS files within sas_dir (default: *.sas).",
    )

    sharepoint = parser.add_argument_group("SharePoint mode (the default)")
    sharepoint.add_argument(
        "--request-id",
        default=None,
        help="Convert exactly one request row, by its list item ID. Also the "
        "pipeline thread_id, so the run's memory and verdicts are keyed to it.",
    )
    sharepoint.add_argument(
        "--app",
        default=None,
        help="Convert every pending row for one application, filtering the "
        "request rows on their Application Name column.",
    )
    sharepoint.add_argument(
        "--all-rows",
        action="store_true",
        help="Include rows already marked done. By default only pending rows "
        "are converted, so a re-run does not redo completed applications.",
    )
    sharepoint.add_argument(
        "--no-upload",
        action="store_true",
        help="Read from the library and convert, but write nothing back and "
        "leave every row's Status alone. The dry run.",
    )
    sharepoint.add_argument(
        "--no-xref",
        action="store_true",
        help="Skip the XREF cross-reference substitution. By default the "
        "application's rows in the XREF list rewrite dataset names (and any "
        "path-marked rows, physical paths) before conversion.",
    )
    sharepoint.add_argument(
        "--xref-apply",
        choices=("pre", "post", "both"),
        default=None,
        help="When to apply the XREF substitution: 'pre' rewrites the SAS "
        "source before conversion, 'post' rewrites the generated code after, "
        "'both' does each and reports what only post reached. Overrides "
        "XREF_APPLY and config.json xref.apply (default: pre). Ignored under "
        "--no-xref, which turns the substitution off entirely.",
    )
    sharepoint.add_argument(
        "--check",
        action="store_true",
        help="Run the read-only SharePoint preflight and exit, converting "
        "nothing and paying no LLM: resolves the configuration (reporting "
        "which env var or config.json key each value came from), mints a Graph "
        "token and reports its granted permissions, then reads the library and "
        "each configured list. Writes nothing. Same as "
        "'python -m app_config.sharepoint_check'.",
    )

    local = parser.add_argument_group("Local mode (with a source directory)")
    local.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Write the translation as Jupyter notebooks under this directory "
        "— one .ipynb per SAS source file, plus _cross_file.ipynb for batches "
        "spanning several files — and one effective-prompt Markdown artifact "
        "per item under its prompts/ subdirectory.",
    )
    local.add_argument(
        "--md",
        type=Path,
        default=None,
        help="Write the inline-validation report as Markdown at this path.",
    )
    local.add_argument(
        "--pdf",
        type=Path,
        default=None,
        help="Render the inline-validation report to a PDF at this path.",
    )
    local.add_argument(
        "--model",
        default=None,
        help="Model id as the AI Gateway names it. Overrides config.json "
        f"llm_client.model (default when both are unset: {_DEFAULT_MODEL}). In "
        "SharePoint mode the conversions list decides and this is the fallback.",
    )
    local.add_argument(
        "--no-validate",
        action="store_true",
        help="Disable the inline LiveValidator, which scores each item as it "
        "is answered and stores the verdict in run memory. In SharePoint mode "
        "the request row decides.",
    )

    common = parser.add_argument_group("Both modes")
    common.add_argument(
        "--reference-dir",
        type=Path,
        default=Path("reference_docs"),
        help="Directory of reference PDFs for instruction chunking "
        "(default: ./reference_docs).",
    )
    common.add_argument(
        "--output-language",
        default=None,
        help="Target language to translate into: "
        f"{', '.join(t.display_name for t in KNOWN_TARGETS)} (spelling is "
        "folded, so 'sparksql' and 'Spark SQL' are one target). Drives the "
        "system prompt, the notebook kernel, and what validation checks the "
        "emitted code against. Omit to use config.json "
        f"pipeline.output_language, then {DEFAULT_OUTPUT_LANGUAGE}. In "
        "SharePoint mode the request row's Destination Language wins.",
    )
    common.add_argument(
        "--vault-secret",
        default=None,
        help="Vault KV path (relative to the mount) holding the LLM API key, "
        "read over AppRole (VAULT_ADDR / VAULT_ROLE_ID / VAULT_SECRET_ID). "
        "Overrides the default AI Gateway credential.",
    )
    common.add_argument(
        "--vault-key",
        default="api_key",
        help="Field within the --vault-secret secret to read (default: api_key).",
    )
    common.add_argument(
        "--no-gateway-auth",
        action="store_true",
        help="Skip the AI Gateway credential chain and use OPENAI_API_KEY from "
        "the environment instead.",
    )
    common.add_argument(
        "--delta-table",
        default=None,
        help="Persist conversation memory to this Delta table instead of the "
        "in-memory store (e.g. default.sas_parser_memory). Boots a Spark "
        "session against app_config.spark's master. Omit to keep everything in "
        "memory and start no JVM.",
    )
    common.add_argument(
        "--validation-retries",
        type=int,
        default=1,
        help="Re-generate an item that fails inline validation up to this many "
        "times, re-prompting with the failed metrics as feedback (default 1; "
        "pass 0 for observe-only: score and store, never act).",
    )
    common.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG logging for the whole pipeline. The HTTP transport "
        "libraries stay at INFO so the pipeline's own lines stay readable; add "
        "--trace-http for those.",
    )
    common.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Also write the log to this file, appending to it. The console "
        "output is unchanged. Secrets are redacted, but treat the file as "
        "sensitive.",
    )
    common.add_argument(
        "--trace-http",
        action="store_true",
        help="Log every HTTP request the Graph SDK and the LLM client make, "
        "with status codes and the SDK's own retries. Verbose by design — pair "
        "it with --log-file.",
    )
    return parser.parse_args(argv)


#: Flags that only mean something for one of the two modes, so naming one
#: against the other mode is a mistake worth reporting rather than ignoring.
_SHAREPOINT_ONLY = (
    "--request-id",
    "--app",
    "--all-rows",
    "--no-upload",
    "--no-xref",
    "--xref-apply",
    "--check",
)
_LOCAL_ONLY = ("--out-dir", "--md", "--pdf")


def _argument_error(args: argparse.Namespace) -> str | None:
    """What is wrong with this combination of flags, or ``None``.

    Checked before anything is read or converted: a run that cannot do what was
    asked should say so in a second, not after an LLM has been paid.
    """
    local_mode = args.sas_dir is not None
    if local_mode:
        for flag, value in (
            ("--request-id", args.request_id),
            ("--app", args.app),
            ("--all-rows", args.all_rows),
            ("--no-upload", args.no_upload),
            ("--no-xref", args.no_xref),
            ("--xref-apply", args.xref_apply),
            ("--check", args.check),
        ):
            if value:
                return (
                    f"{flag} drives the SharePoint request list; it does not "
                    f"apply to a local directory. Drop {flag}, or drop the "
                    f"directory to run from the list."
                )
        if not args.sas_dir.is_dir():
            return f"sas_dir is not a directory: {args.sas_dir}"
    else:
        for flag, value in (
            ("--out-dir", args.out_dir),
            ("--md", args.md),
            ("--pdf", args.pdf),
        ):
            if value:
                return (
                    f"{flag} writes locally; SharePoint mode delivers to the "
                    f"document library. Give a source directory to run "
                    f"locally, or drop {flag}."
                )
        if args.request_id and args.app:
            return "--request-id and --app both narrow the row set; give one"
        if args.check and (args.request_id or args.app or args.all_rows):
            return (
                "--check inspects the deployment rather than any particular "
                "row; drop the row filter."
            )
    if args.validation_retries < 0:
        return "--validation-retries must be >= 0"
    # --check converts nothing, so it needs neither the reference PDFs nor a
    # model. Requiring them would make the diagnostic unusable on exactly the
    # machine it exists to diagnose — a fresh checkout where reference_docs/ is
    # gitignored and absent.
    if not args.check and not args.reference_dir.is_dir():
        return f"reference_dir is not a directory: {args.reference_dir}"
    return None


# ---------------------------------------------------------------------------
# Environment and credentials
# ---------------------------------------------------------------------------


def _load_dotenv() -> None:
    """Load ``.env`` into the environment — see
    :func:`app_config.load_dotenv_file`."""
    app_config.load_dotenv_file()


def resolve_llm_config(args: argparse.Namespace, **overrides: Any) -> LLMClientConfig:
    """
    The run's LLM configuration, credential and all — see this module's
    docstring for the precedence.

    Built **once** and reused for every row: each of the first two paths is a
    round trip to Vault (and, for the gateway chain, to Entra ID before it), so
    resolving per row would repeat the whole login for no benefit. Per-row
    differences are applied with :meth:`~pydantic.BaseModel.model_copy`, which
    is why this returns a config rather than a bare key.

    Exits non-zero with a readable message rather than a traceback when a
    credential was asked for and could not be got: that is an operator's
    problem to fix, not a bug to debug.
    """
    from app_config import vault

    if args.vault_secret:
        try:
            config = LLMClientConfig.from_vault_secret(
                args.vault_secret, args.vault_key, **overrides
            )
        except vault.VaultError as exc:
            raise SystemExit(f"could not fetch the API key from Vault: {exc}") from exc
        logger.info(
            f"API key from Vault secret '{args.vault_secret}' "
            f"(field '{args.vault_key}') via AppRole"
        )
        return config

    if args.no_gateway_auth:
        logger.info("--no-gateway-auth: deferring to OPENAI_API_KEY")
        return LLMClientConfig(**overrides)

    if not vault.is_configured():
        logger.info(
            "no Vault configuration found (VAULT_ADDR and a login method); "
            "deferring to OPENAI_API_KEY"
        )
        return LLMClientConfig(**overrides)

    try:
        config = LLMClientConfig.from_ai_gateway(**overrides)
    except vault.VaultError as exc:
        raise SystemExit(
            f"could not fetch the AI Gateway credential from Vault: {exc}\n"
            f"Pass --no-gateway-auth to fall back to OPENAI_API_KEY, or "
            f"--vault-secret PATH to read a different secret over AppRole."
        ) from exc
    logger.info(
        f"API key from the AI Gateway secret in Vault via the Entra ID "
        f"service-principal chain (base_url={config.base_url or 'from config.json'})"
    )
    return config


# ---------------------------------------------------------------------------
# Pipeline construction — shared by both modes
# ---------------------------------------------------------------------------


def build_pipeline(
    args: argparse.Namespace,
    *,
    llm_config: LLMClientConfig,
    output_language: str | None,
    validate: bool,
) -> Any:
    """Load the reference corpus and wire up the pipeline.

    The target language is resolved here, before the validator is built, so
    both score and prompt against the same one — scoring a Spark SQL run's
    output as Python is exactly the mismatch that argument exists to prevent.
    """
    from pipeline import SasLLMPipeline
    from pipeline.setup import MemorySetup, PromptingSetup, ValidationSetup
    from prompt_builder import PromptBuilder
    from validation import LiveValidator

    logger.info(f"building instruction corpus from {args.reference_dir}")
    builder = PromptBuilder.from_reference_dir(str(args.reference_dir))

    if args.delta_table:
        from app_config.spark import describe_master, master_url

        logger.info(
            f"persisting conversation memory to Delta table "
            f"{args.delta_table} on {describe_master(master_url())}"
        )

    target = resolve_target_language(output_language)
    validator = LiveValidator(output_language=target.display_name) if validate else None
    return SasLLMPipeline(
        llm_config=llm_config,
        memory_setup=MemorySetup(delta_table=args.delta_table),
        output_language=target.display_name,
        prompting=PromptingSetup(prompt_builder=builder),
        validation=ValidationSetup(
            validator=validator, retries=args.validation_retries
        ),
    )


# ---------------------------------------------------------------------------
# SharePoint mode — the default
# ---------------------------------------------------------------------------


def _run_sharepoint(args: argparse.Namespace) -> int:
    """Convert every selected request row, each as its own corpus.

    One row failing does not abort the rest — a request list is a queue, not a
    transaction — but the exit status is non-zero if any did.
    """
    from app_config.sharepoint import SharePointError, get_sharepoint_client
    from conversion import requests as conv_requests, run as conv_run

    try:
        client = get_sharepoint_client()
        rows = (
            conv_requests.requests(client=client)
            if args.all_rows
            else conv_requests.pending_requests(client=client)
        )
    except SharePointError as exc:
        return _fail(f"could not read the SharePoint request list: {exc}")

    rows = conv_run.select_requests(
        rows, request_id=args.request_id, application=args.app
    )
    if not rows:
        which = (
            f" with ID={args.request_id!r}"
            if args.request_id
            else f" for application {args.app!r}"
            if args.app
            else " outstanding"
        )
        return _fail(
            f"no request row{which}"
            + ("" if args.all_rows else " (use --all-rows to include completed rows)")
        )

    try:
        items = conv_requests.conversion_items(client=client)
    except SharePointError as exc:
        logger.warning(
            f"could not read the conversions list ({exc}); every row falls back "
            f"to the configured model"
        )
        items = []

    default_model = args.model or app_config.llm_client_value("model", _DEFAULT_MODEL)
    # One credential resolution for the whole run — see resolve_llm_config.
    base_config = resolve_llm_config(args)

    logger.info(f"converting {len(rows)} request row(s)")
    status = 0
    for row in rows:
        model = conv_run.model_for(row, items, default_model)
        if not app_config.is_accessible_model(model):
            logger.error(
                f"request {row.item_id!r} selected model {model!r}, which is "
                f"not accessible to this deployment; skipping the row"
            )
            status = 1
            continue

        outcome = conv_run.run_request(
            row,
            build_pipeline=lambda m, v, _row=row: build_pipeline(
                args,
                llm_config=base_config.model_copy(update={"model": m}),
                output_language=_row.output_language or args.output_language,
                validate=v,
            ),
            model=model,
            client=client,
            xref_mappings=None if args.no_xref else _xref_for(row, client=client),
            xref_mode=args.xref_apply,
            upload=not args.no_upload,
        )
        if not outcome.ok:
            status = 1
            print(
                f"request {row.item_id} ({row.application_name}) failed: "
                f"{outcome.error}",
                file=sys.stderr,
            )
        else:
            print(
                f"converted {row.application_name}: "
                f"{len(outcome.uploaded)} file(s) uploaded"
                if outcome.uploaded
                else f"converted {row.application_name} (nothing uploaded)"
            )
    return status


def _xref_for(row: Any, *, client: Any) -> Any | None:
    """The application's XREF mappings, or ``None`` when there are none.

    A missing or unconfigured XREF list is not fatal: it means this deployment
    does not cross-reference, which is a normal state, so it is reported once
    and the conversion goes ahead with SAS-side names.
    """
    from app_config.sharepoint import SharePointError
    from xref import sourcing

    try:
        mappings = sourcing.mappings(row.application_name, client=client)
    except SharePointError as exc:
        logger.warning(
            f"no XREF mappings for {row.application_name!r} ({exc}); dataset "
            f"names are left as the SAS source has them"
        )
        return None
    return mappings or None


# ---------------------------------------------------------------------------
# Local mode — the fallback
# ---------------------------------------------------------------------------


def _discover_sas_files(sas_dir: Path, pattern: str) -> list[str]:
    """Recursively find .sas files under *sas_dir*, sorted.

    File order establishes the default execution sequence MultiFileBatcher uses
    to resolve cross-file producer/consumer tie-breaks, so a stable sort keeps
    batching reproducible across runs.
    """
    return [str(p) for p in sorted(sas_dir.rglob(pattern))]


def _run_local(args: argparse.Namespace) -> int:
    from conversion import run as conv_run

    sas_files = _discover_sas_files(args.sas_dir, args.pattern)
    if not sas_files:
        return _fail(f"no files matching {args.pattern!r} under {args.sas_dir}")
    logger.info(f"discovered {len(sas_files)} SAS file(s) under {args.sas_dir}")
    for path in sas_files:
        logger.info(f"  - {path}")

    # CLI flag > config.json llm_client.model > code default. Resolved here
    # because passing the argparse value unconditionally would count as an
    # "explicit argument" downstream and shadow the config.json entry.
    if args.model:
        model = args.model
        logger.info(f"model from --model: {model}")
    else:
        model = app_config.llm_client_value("model", _DEFAULT_MODEL)
        logger.info(f"model from config.json/default: {model}")

    llm_config = resolve_llm_config(args, model=model)
    validating = not args.no_validate
    pipeline = build_pipeline(
        args,
        llm_config=llm_config,
        output_language=args.output_language,
        validate=validating,
    )

    logger.info(f"running pipeline over {len(sas_files)} file(s) with model={model}")
    outputs = pipeline.run_files(sas_files)

    conv_run.log_item_summaries(outputs)
    conv_run.log_token_usage(pipeline)
    for out in outputs:
        print(f"\n{conv_run.item_header(out)}")
        print(out["response"])

    if args.out_dir:
        from pipeline.artifacts import write_prompts
        from pipeline.notebook import write_notebooks

        for path in write_notebooks(
            outputs, args.out_dir, output_language=pipeline.output_language
        ):
            print(f"wrote notebook: {path}")
        for path in write_prompts(outputs, args.out_dir):
            print(f"wrote prompt: {path}")

    if not validating:
        if args.md or args.pdf:
            logger.warning("--md/--pdf ignored: inline validation is off")
        return 0

    conv_run.log_validation_summary(outputs)
    if args.md or args.pdf:
        report = conv_run.validation_report(
            outputs,
            model=model,
            instructions_fingerprint=pipeline.instructions_fingerprint,
            policy_fingerprint=pipeline.policy_fingerprint,
            token_usage=pipeline.token_usage,
        )
        if args.md:
            args.md.parent.mkdir(parents=True, exist_ok=True)
            args.md.write_text(report.to_markdown(), encoding="utf-8")
            print(f"wrote inline-validation Markdown report: {args.md}")
        if args.pdf:
            from validation import report_to_pdf

            args.pdf.parent.mkdir(parents=True, exist_ok=True)
            args.pdf.write_bytes(report_to_pdf(report))
            print(f"wrote inline-validation PDF report: {args.pdf}")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _fail(message: str) -> int:
    logger.error(message)
    print(message, file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    configure_logging(
        debug=args.debug, log_file=args.log_file, trace_http=args.trace_http
    )
    # Before anything reads it: the Vault credentials, OPENAI_API_KEY, and the
    # SharePoint/Azure settings all live there. The real shell environment wins.
    _load_dotenv()

    problem = _argument_error(args)
    if problem is not None:
        return _fail(problem)

    if args.check:
        return _run_check(args)
    if args.sas_dir is not None:
        return _run_local(args)
    return _run_sharepoint(args)


def _run_check(args: argparse.Namespace) -> int:
    """Run the read-only SharePoint preflight and print its report."""
    from app_config import sharepoint_check

    results = sharepoint_check.run_checks()
    print(sharepoint_check.render(results, verbose=args.debug))
    return 1 if any(r.status == sharepoint_check.FAIL for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
