"""Demo: run the SAS -> target pipeline, from local files or from SharePoint.

Two modes, as argparse subcommands:

``local`` (the original flow)
    Runs the pipeline over a local directory of ``.sas`` files and optionally
    writes each response to a local ``--out-dir``. Model, validation on/off, and
    retries are CLI flags.

``sharepoint`` (Power Apps driven)
    Picks one conversion request from a SharePoint list — the row whose request
    id equals ``--request-id`` — pulls its input ``.sas`` scripts from the
    document library folder named after the request's ``application_name``, runs
    the pipeline, and writes the responses (and validation artifacts) back to
    SharePoint under ``<application_name>/output/<timestamp>/``. The Power Apps
    request id is used as the pipeline ``thread_id``, so the run's conversation
    memory, run facts, and validation verdicts are keyed to it. The model,
    validation on/off, application folder, and timestamp all come from the list
    row — not from CLI flags.

End-to-end wiring the pipeline is built for:

    reference_docs/*.pdf ── PromptBuilder.from_reference_dir ─┐
                                                              ├─> SasLLMPipeline
    <sas files> ──────────── discovered here ────────────────┘
                                                              │
                                    run_files() ── MultiFileBatcher ── LLM

`SasLLMPipeline.run_files` chunks every file, batches the whole corpus with
`MultiFileBatcher` (so cross-file dataset-flow / macro edges are resolved into
shared batches), and feeds every batch + singleton through the LLM on one
thread. Per-item reference guidance is retrieved from the `reference_docs`
corpus and injected ephemerally.

A `validation.LiveValidator` scores each batch the moment its response returns
(deterministic, offline metrics — no extra model call) and the verdict is stored
in that run's conversation memory. In ``local`` mode it is on unless
``--no-validate``; in ``sharepoint`` mode it follows the request's live-validation
flag. A failing item is re-generated once by default (with the failed metrics fed
back as a correction); tune with ``--validation-retries N`` (``0`` = observe-only).

The inline verdicts also aggregate into a PDF report (`validation.report_to_pdf`
over `validation.report_from_verdicts`): ``local`` mode writes it to ``--pdf``
when set, and ``sharepoint`` mode always uploads it as
``<application_name>/output/<timestamp>/validation/report.pdf`` beside the
per-item and summary JSON.

Usage
-----
    # needs the `anthropic` extra installed:
    #   uv pip install -e ".[anthropic]"
    # for `sharepoint`, also the SharePoint extra + Entra ID identity:
    #   uv pip install -e ".[sharepoint]"
    #
    # API key, either:
    #   - ANTHROPIC_API_KEY in the environment (default), or
    #   - fetched from Vault via AppRole app-based auth (--vault-secret);
    #     needs the `vault` extra and VAULT_ADDR / VAULT_ROLE_ID /
    #     VAULT_SECRET_ID set (see below).
    python demo_run.py local path/to/sas_dir
    python demo_run.py local path/to/sas_dir --model claude-sonnet-4-5 --debug
    python demo_run.py sharepoint --request-id REQ-1234
    python demo_run.py sharepoint --request-id REQ-1234 --vault-secret llm/anthropic

In ``local`` mode, without ``--model`` the model comes from config.json
(``llm_client.model``), falling back to the code default when that entry is null
or absent.

SharePoint (Power Apps)
-----------------------
``sharepoint`` mode reads the request list and the document library through
:mod:`app_config.sharepoint`, which delegates authentication to the Entra ID
service principal in :mod:`app_config.azure` (``AZURE_TENANT_ID`` /
``AZURE_CLIENT_ID`` / ``AZURE_CLIENT_SECRET``, app permission
``Sites.ReadWrite.All``). Point it at the site with ``SHAREPOINT_SITE_ID`` (or
``SHAREPOINT_SITE_HOSTNAME`` + ``SHAREPOINT_SITE_PATH``) and name the request
list with ``POWERAPPS_LIST_NAME`` (see :mod:`app_config.powerapps` for the
column-name settings). All of these may also live in ``config.json``.

Vault (app-based auth)
----------------------
Passing ``--vault-secret PATH`` retrieves the LLM API key from HashiCorp Vault
using **AppRole** auth — the app authenticates with a role_id / secret_id pair
(its own application identity) rather than a personal token. Set:

    VAULT_ADDR=https://vault.example:8200
    VAULT_ROLE_ID=...        # the app's role
    VAULT_SECRET_ID=...      # the app's secret

then ``--vault-secret llm/anthropic`` reads the ``api_key`` field of the KV
secret at that path (override the field with ``--vault-key``). Any ambient
``VAULT_TOKEN`` is ignored so the demo always runs under the AppRole identity.
Keep these out of source control — put them in ``.env`` (gitignored).

Run from the repo root so the default ``reference_docs`` path resolves.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import app_config
from chunker import SasLLMPipeline
from prompt_builder import PromptBuilder
from validation import LiveValidator

logger = logging.getLogger("demo_run")

_DEFAULT_MODEL = "claude-sonnet-4-5"


def _discover_sas_files(sas_dir: Path, pattern: str) -> list[str]:
    """Recursively find .sas files under *sas_dir*, sorted for deterministic order.

    File order establishes the default execution sequence MultiFileBatcher uses
    to resolve cross-file producer/consumer tie-breaks, so a stable sort keeps
    batching reproducible across runs.
    """
    paths = sorted(sas_dir.rglob(pattern))
    return [str(p) for p in paths]


def _load_dotenv() -> None:
    """Load ``.env`` (repo root, walking up from cwd) into the environment.

    Existing environment variables win (``override=False``), so a value already
    exported in the shell still takes precedence over the file. A missing
    ``.env`` is a no-op. python-dotenv is a declared dependency, but the import
    is guarded so the demo still runs if it is somehow absent.
    """
    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError:
        logger.debug("python-dotenv not installed; skipping .env load")
        return
    path = find_dotenv(usecwd=True)  # search from cwd (the repo root), walking up
    if path and load_dotenv(path):
        logger.debug(f"loaded environment from {path}")


def _fetch_api_key_from_vault(secret_path: str, secret_key: str) -> str:
    """Retrieve the LLM API key from Vault using AppRole (app-based) auth.

    The app logs in with a role_id / secret_id pair (``VAULT_ROLE_ID`` /
    ``VAULT_SECRET_ID``) — its own application identity — rather than a
    personal token; any ambient ``VAULT_TOKEN`` is dropped so this path always
    exercises AppRole. Connection settings (``VAULT_ADDR``, mount, ...) resolve
    via :meth:`app_config.vault.VaultConfig.from_env`. Configuration or lookup
    problems exit non-zero with a readable message rather than a traceback.
    """
    from app_config.vault import VaultClient, VaultConfig, VaultError

    base = VaultConfig.from_env()
    if not (base.role_id and base.secret_id):
        raise SystemExit(
            "--vault-secret needs AppRole credentials: set VAULT_ROLE_ID and "
            "VAULT_SECRET_ID (and VAULT_ADDR), or omit --vault-secret to use "
            "ANTHROPIC_API_KEY directly"
        )
    approle = replace(base, token=None)  # force app identity, ignore any token
    try:
        key = VaultClient(approle).get_secret(secret_path, secret_key)
    except VaultError as exc:
        raise SystemExit(f"could not fetch API key from Vault: {exc}") from exc
    if not isinstance(key, str) or not key:
        raise SystemExit(
            f"Vault secret '{secret_path}' field '{secret_key}' is empty or "
            "not a string"
        )
    return key


def _resolve_api_key(args: argparse.Namespace) -> str | None:
    """The LLM API key from Vault when ``--vault-secret`` is set, else ``None``.

    ``None`` lets the pipeline defer to ``ANTHROPIC_API_KEY`` / the provider env
    var. Shared by both subcommands.
    """
    if not args.vault_secret:
        return None
    key = _fetch_api_key_from_vault(args.vault_secret, args.vault_key)
    logger.info(
        f"API key from Vault secret '{args.vault_secret}' "
        f"(field '{args.vault_key}') via AppRole"
    )
    return key


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Flags shared by both subcommands (reference corpus, target language,
    Vault, validation retries, debug)."""
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path("reference_docs"),
        help="Directory of reference PDFs for instruction chunking "
        "(default: ./reference_docs).",
    )
    parser.add_argument(
        "--output-language",
        default="PySpark",
        help="Target language named in the system prompt (default: PySpark).",
    )
    parser.add_argument(
        "--vault-secret",
        default=None,
        help="Vault KV path (relative to the mount) holding the LLM API key, "
        "retrieved via AppRole app-based auth (VAULT_ADDR / VAULT_ROLE_ID / "
        "VAULT_SECRET_ID). When set, its value is used as the API key instead "
        "of ANTHROPIC_API_KEY.",
    )
    parser.add_argument(
        "--vault-key",
        default="api_key",
        help="Field within the --vault-secret secret to read (default: api_key).",
    )
    parser.add_argument(
        "--validation-retries",
        type=int,
        default=1,
        help="Re-generate an item that fails inline validation up to this "
        "many times, re-prompting with the failed metrics as feedback "
        "(default 1; pass 0 for observe-only: score and store, never act).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG logging for the whole pipeline.",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    # __doc__ is None under `python -OO`, which strips docstrings.
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else None
    )
    sub = parser.add_subparsers(dest="command", required=True)

    local = sub.add_parser(
        "local", help="Run over a local directory of .sas files (the original flow)."
    )
    local.add_argument(
        "sas_dir",
        type=Path,
        help="Directory containing local .sas files (searched recursively).",
    )
    local.add_argument(
        "--pattern",
        default="*.sas",
        help="Glob for SAS files within sas_dir (default: *.sas).",
    )
    local.add_argument(
        "--model",
        default=None,
        help="LangChain chat-model string. Overrides config.json "
        f"llm_client.model (default when both are unset: {_DEFAULT_MODEL}).",
    )
    local.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="If set, write each item's LLM response to a file under this dir.",
    )
    local.add_argument(
        "--no-validate",
        action="store_true",
        help="Disable the inline LiveValidator (on by default), which scores "
        "each item as it is answered and stores the verdict in run memory.",
    )
    local.add_argument(
        "--pdf",
        type=Path,
        default=None,
        help="Render the inline-validation verdicts to a PDF report at this "
        "path (only when validation is on).",
    )
    _add_common_args(local)

    sharepoint = sub.add_parser(
        "sharepoint",
        help="Run one Power Apps request read from a SharePoint list, with "
        "inputs and outputs in the document library.",
    )
    sharepoint.add_argument(
        "--request-id",
        required=True,
        help="The Power Apps request id — the list row to run. Also used as the "
        "pipeline thread_id.",
    )
    _add_common_args(sharepoint)

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Shared pipeline construction + reporting
# ---------------------------------------------------------------------------


def _build_pipeline(
    args: argparse.Namespace,
    *,
    model: str,
    api_key: str | None,
    validator: LiveValidator | None,
) -> SasLLMPipeline:
    """Load the reference corpus once and wire up the pipeline.

    In-memory message store (``delta_table=None``) — no Spark/JVM is booted.
    """
    logger.info(f"building instruction corpus from {args.reference_dir}")
    builder = PromptBuilder.from_reference_dir(str(args.reference_dir))
    return SasLLMPipeline(
        model=model,
        api_key=api_key,
        output_language=args.output_language,
        prompt_builder=builder,
        validator=validator,
        validation_retries=args.validation_retries,
    )


def _item_header(out: dict) -> str:
    """The ``=== id (kind) files=[...] [verdict] ===`` banner for one item."""
    verdict = _format_verdict(out.get("validation"))
    return (
        f"=== {out['item_id']} "
        f"({'batch' if out['is_batch'] else out['kind']}) "
        f"files={out['source_files']}{verdict} ==="
    )


def _format_verdict(validation: dict | None) -> str:
    """`  [PASS score=0.95]` for one item's stored verdict, or `""`."""
    if not validation:
        return ""
    status = "PASS" if validation["passed"] else "FAIL"
    return f"  [{status} score={validation['score']:.2f}]"


def _log_validation_summary(outputs: list[dict]) -> None:
    """Aggregate the per-item inline verdicts and log pass/fail counts.

    The verdicts also live in the run's conversation memory
    (``pipeline.get_validation_facts(thread_id)``); this is just the run-end
    read of what was scored inline.
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
        v = out.get("validation")
        if v and not v["passed"]:
            failed = [
                m["metric"]
                for m in v["metrics"]
                if not m["passed"] and not m["skipped"]
            ]
            logger.warning(
                f"  FAIL {out['item_id']}  score={v['score']:.3f}  "
                f"metrics_below_threshold={failed}"
            )


def _validation_report(
    outputs: list[dict], *, model: str, instructions_fingerprint: str | None = None
):
    """A ValidationReport reconstructed from the run's inline verdicts.

    Turns the per-item ``out["validation"]`` verdicts LiveValidator produced
    into the same aggregate report an offline run yields, so the inline run can
    be rendered to Markdown/PDF (:mod:`validation.pdf`).
    """
    from validation import report_from_verdicts

    verdicts = [o["validation"] for o in outputs if o.get("validation")]
    return report_from_verdicts(
        verdicts, model=model, instructions_fingerprint=instructions_fingerprint
    )


def _validation_summary(outputs: list[dict]) -> dict:
    """The aggregate pass/fail summary uploaded alongside the SharePoint run."""
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


# ---------------------------------------------------------------------------
# local mode
# ---------------------------------------------------------------------------


def _run_local(args: argparse.Namespace) -> int:
    if not args.sas_dir.is_dir():
        logger.error(f"sas_dir is not a directory: {args.sas_dir}")
        return 2
    if not args.reference_dir.is_dir():
        logger.error(f"reference_dir is not a directory: {args.reference_dir}")
        return 2

    sas_files = _discover_sas_files(args.sas_dir, args.pattern)
    if not sas_files:
        logger.error(f"no files matching {args.pattern!r} under {args.sas_dir}")
        return 1
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

    api_key = _resolve_api_key(args)
    validator = None if args.no_validate else LiveValidator()
    pipeline = _build_pipeline(
        args, model=model, api_key=api_key, validator=validator
    )

    logger.info(f"running pipeline over {len(sas_files)} file(s) with model={model}")
    outputs = pipeline.run_files(sas_files)

    logger.info(f"pipeline produced {len(outputs)} item response(s)")
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    for out in outputs:
        header = _item_header(out)
        print(f"\n{header}")
        print(out["response"])

        if args.out_dir:
            dest = args.out_dir / f"{out['item_id']}.txt"
            dest.write_text(f"{header}\n\n{out['response']}\n", encoding="utf-8")
            logger.debug(f"wrote {dest}")

    if validator is not None:
        _log_validation_summary(outputs)
        if args.pdf:
            from validation import report_to_pdf

            report = _validation_report(
                outputs,
                model=model,
                instructions_fingerprint=pipeline.instructions_fingerprint,
            )
            args.pdf.write_bytes(report_to_pdf(report))
            logger.info(f"wrote inline-validation PDF report: {args.pdf}")
            print(f"wrote inline-validation PDF report: {args.pdf}")
    elif args.pdf:
        logger.warning("--pdf ignored: inline validation is off (--no-validate)")

    return 0


# ---------------------------------------------------------------------------
# sharepoint mode
# ---------------------------------------------------------------------------


def _discover_sharepoint_sas(client, root: str) -> list[tuple[str, str]]:
    """Recursively find ``.sas`` files under the *root* library folder.

    Returns ``(library_path, relative_path)`` pairs sorted by relative path —
    mirroring the deterministic ordering :func:`_discover_sas_files` gives the
    local flow, so cross-file batching stays reproducible.
    """
    results: list[tuple[str, str]] = []

    def walk(sp_dir: str, rel_dir: str) -> None:
        for entry in client.list_directory(sp_dir):
            name = entry.get("name")
            if not name:
                continue
            child_sp = f"{sp_dir}/{name}"
            child_rel = f"{rel_dir}/{name}" if rel_dir else name
            if entry.get("is_folder"):
                walk(child_sp, child_rel)
            elif name.lower().endswith(".sas"):
                results.append((child_sp, child_rel))

    walk(root, "")
    results.sort(key=lambda pair: pair[1])
    return results


def _download_inputs(
    client, entries: list[tuple[str, str]], dest_dir: Path
) -> list[str]:
    """Download each ``(library_path, relative_path)`` into *dest_dir*.

    The relative structure is preserved so multi-folder inputs keep distinct
    source ids. Returns the local file paths in the same (sorted) order.
    """
    local_paths: list[str] = []
    for sp_path, rel_path in entries:
        content = client.read_file(sp_path)
        dest = dest_dir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        local_paths.append(str(dest))
        logger.debug(f"downloaded {sp_path!r} -> {dest}")
    return local_paths


def _already_exists(exc: Exception) -> bool:
    """True when a SharePoint create failed only because the folder exists."""
    text = str(exc).lower()
    return "alreadyexists" in text or "already exists" in text


def _ensure_directory(client, path: str) -> None:
    """Create every segment of *path* in the library, idempotently.

    Graph's simple upload does not create missing parent folders, so the output
    tree is materialised first. Each level is created in turn; an
    already-exists conflict is swallowed (the shared ``output`` folder recurs
    across runs), anything else re-raised.
    """
    from app_config.sharepoint import SharePointError

    segments = [s for s in path.strip("/").split("/") if s]
    current = ""
    for segment in segments:
        current = f"{current}/{segment}" if current else segment
        try:
            client.create_directory(current)
        except SharePointError as exc:
            if _already_exists(exc):
                logger.debug(f"directory already exists: {current!r}")
                continue
            raise


def _upload_outputs(
    client,
    req,
    outputs: list[dict],
    *,
    validating: bool,
    instructions_fingerprint: str | None = None,
) -> str:
    """Write responses (and validation artifacts) back to the library.

    Layout: ``<application_name>/output/<timestamp>/<item_id>.txt`` for each
    response, and — when validating — ``.../validation/<item_id>.json`` per item,
    a ``.../validation/summary.json`` aggregate, and a human-readable
    ``.../validation/report.pdf`` rendered from the inline verdicts. Returns the
    output folder.
    """
    out_dir = f"{req.application_name}/output/{req.timestamp}"
    validation_dir = f"{out_dir}/validation"
    _ensure_directory(client, out_dir)
    if validating:
        _ensure_directory(client, validation_dir)

    for out in outputs:
        header = _item_header(out)
        client.write_file(
            f"{out_dir}/{out['item_id']}.txt",
            f"{header}\n\n{out['response']}\n",
        )
        verdict = out.get("validation")
        if validating and verdict is not None:
            client.write_file(
                f"{validation_dir}/{out['item_id']}.json",
                json.dumps(verdict, indent=2),
            )

    if validating:
        from validation import report_to_pdf

        client.write_file(
            f"{validation_dir}/summary.json",
            json.dumps(_validation_summary(outputs), indent=2),
        )
        report = _validation_report(
            outputs,
            model=req.model,
            instructions_fingerprint=instructions_fingerprint,
        )
        client.write_file(f"{validation_dir}/report.pdf", report_to_pdf(report))
    return out_dir


def _run_sharepoint(args: argparse.Namespace) -> int:
    from app_config.powerapps import (
        PowerAppsConfig,
        PowerAppsError,
        parse_run_request,
        select_request,
    )
    from app_config.sharepoint import SharePointError, get_sharepoint_client

    if not args.reference_dir.is_dir():
        logger.error(f"reference_dir is not a directory: {args.reference_dir}")
        return 2

    pa_config = PowerAppsConfig.from_env()
    if not pa_config.list_name:
        logger.error(
            "no Power Apps list configured: set POWERAPPS_LIST_NAME (or "
            "powerapps.list_name in config.json)"
        )
        return 2

    client = get_sharepoint_client()

    # Resolve the request row -> normalised parameters.
    try:
        rows = client.read_list_items(pa_config.list_name)
        row = select_request(rows, args.request_id, pa_config)
        req = parse_run_request(row, pa_config)
    except (SharePointError, PowerAppsError) as exc:
        logger.error(f"could not resolve Power Apps request: {exc}")
        return 1
    logger.info(
        f"request {req.request_id!r}: application={req.application_name!r}  "
        f"model={req.model}  live_validation={req.live_validation}  "
        f"timestamp={req.timestamp}  (list item {req.item_id})"
    )

    # Discover + download the input .sas scripts from the library.
    try:
        entries = _discover_sharepoint_sas(client, req.application_name)
    except SharePointError as exc:
        logger.error(f"could not list SharePoint inputs: {exc}")
        return 1
    if not entries:
        logger.error(
            f"no .sas files under library folder {req.application_name!r}"
        )
        return 1
    logger.info(
        f"discovered {len(entries)} SAS file(s) under {req.application_name!r}"
    )
    for _, rel in entries:
        logger.info(f"  - {rel}")

    api_key = _resolve_api_key(args)
    validator = LiveValidator() if req.live_validation else None
    pipeline = _build_pipeline(
        args, model=req.model, api_key=api_key, validator=validator
    )

    tmp_dir = Path(tempfile.mkdtemp(prefix="sas_parser_sp_"))
    try:
        try:
            local_paths = _download_inputs(client, entries, tmp_dir)
        except SharePointError as exc:
            logger.error(f"could not download SharePoint inputs: {exc}")
            return 1

        # The Power Apps request id is the pipeline thread_id, so run facts and
        # validation verdicts land in memory keyed to this request.
        logger.info(
            f"running pipeline over {len(local_paths)} file(s) with "
            f"model={req.model}  thread_id={req.request_id!r}"
        )
        outputs = pipeline.run_files(local_paths, thread_id=req.request_id)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    logger.info(f"pipeline produced {len(outputs)} item response(s)")
    for out in outputs:
        print(f"\n{_item_header(out)}")
        print(out["response"])

    try:
        out_dir = _upload_outputs(
            client,
            req,
            outputs,
            validating=validator is not None,
            instructions_fingerprint=pipeline.instructions_fingerprint,
        )
    except SharePointError as exc:
        logger.error(f"could not upload results to SharePoint: {exc}")
        return 1
    logger.info(f"uploaded {len(outputs)} response(s) to {out_dir!r}")

    if validator is not None:
        _log_validation_summary(outputs)

    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    # Populate the environment from .env before anything reads it (the Vault
    # AppRole creds, ANTHROPIC_API_KEY, and the SharePoint/Azure settings live
    # there); real shell env wins.
    _load_dotenv()

    if args.command == "sharepoint":
        return _run_sharepoint(args)
    return _run_local(args)


if __name__ == "__main__":
    sys.exit(main())
