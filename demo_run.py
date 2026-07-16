"""Demo: run the SAS -> target pipeline over a directory of local .sas files.

End-to-end wiring the pipeline is built for:

    reference_docs/*.pdf ── PromptBuilder.from_reference_dir ─┐
                                                              ├─> SasLLMPipeline
    <sas_dir>/**/*.sas ───────── discovered here ────────────┘
                                                              │
                                    run_files() ── MultiFileBatcher ── LLM

`SasLLMPipeline.run_files` chunks every file, batches the whole corpus with
`MultiFileBatcher` (so cross-file dataset-flow / macro edges are resolved into
shared batches), and feeds every batch + singleton through the LLM on one
thread. Per-item reference guidance is retrieved from the `reference_docs`
corpus and injected ephemerally.

A `validation.LiveValidator` is attached by default, so each batch is scored
the moment its response returns (deterministic, offline metrics — no extra
model call) and the verdict is stored in that run's conversation memory
beside the item's run fact. The demo prints the per-item verdict and an
aggregate; pass ``--no-validate`` to turn it off. A failing item is
re-generated once by default (with the failed metrics fed back as a
correction); tune with ``--validation-retries N`` (``0`` = observe-only).

Usage
-----
    # needs the `anthropic` extra installed:
    #   uv pip install -e ".[anthropic]"
    #
    # API key, either:
    #   - ANTHROPIC_API_KEY in the environment (default), or
    #   - fetched from Vault via AppRole app-based auth (--vault-secret);
    #     needs the `vault` extra and VAULT_ADDR / VAULT_ROLE_ID /
    #     VAULT_SECRET_ID set (see below).
    python demo_run.py path/to/sas_dir
    python demo_run.py path/to/sas_dir --model claude-sonnet-4-5 --debug
    python demo_run.py path/to/sas_dir --vault-secret llm/anthropic

Without ``--model``, the model comes from config.json (``llm_client.model``),
falling back to the code default when that entry is null or absent.

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
import logging
import sys
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "sas_dir",
        type=Path,
        help="Directory containing local .sas files (searched recursively).",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path("reference_docs"),
        help="Directory of reference PDFs for instruction chunking "
        "(default: ./reference_docs).",
    )
    parser.add_argument(
        "--pattern",
        default="*.sas",
        help="Glob for SAS files within sas_dir (default: *.sas).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="LangChain chat-model string. Overrides config.json "
        f"llm_client.model (default when both are unset: {_DEFAULT_MODEL}).",
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
        "--out-dir",
        type=Path,
        default=None,
        help="If set, write each item's LLM response to a file under this dir.",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Disable the inline LiveValidator (on by default), which scores "
        "each item as it is answered and stores the verdict in run memory.",
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    # Populate the environment from .env before anything reads it (the Vault
    # AppRole creds and ANTHROPIC_API_KEY live there); real shell env wins.
    _load_dotenv()

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

    # Load + chunk + index the reference corpus once (cached on disk after the
    # first run). This is the "document chunking for instructions" half.
    logger.info(f"building instruction corpus from {args.reference_dir}")
    builder = PromptBuilder.from_reference_dir(str(args.reference_dir))

    # CLI flag > config.json llm_client.model > code default. Resolved here
    # because passing the argparse value unconditionally would count as an
    # "explicit argument" downstream and shadow the config.json entry.
    if args.model:
        model = args.model
        logger.info(f"model from --model: {model}")
    else:
        model = app_config.llm_client_value("model", _DEFAULT_MODEL)
        logger.info(f"model from config.json/default: {model}")

    # API key: from Vault via AppRole when --vault-secret is given, else None
    # (the pipeline then defers to ANTHROPIC_API_KEY / the provider env var).
    api_key = None
    if args.vault_secret:
        api_key = _fetch_api_key_from_vault(args.vault_secret, args.vault_key)
        logger.info(
            f"API key from Vault secret '{args.vault_secret}' "
            f"(field '{args.vault_key}') via AppRole"
        )

    # Inline validation (on unless --no-validate): the deterministic, offline
    # suite scores each item as its response returns, adding no model call, and
    # stores every verdict in the run's conversation memory.
    validator = None if args.no_validate else LiveValidator()

    # In-memory message store (delta_table=None) — no Spark/JVM is booted.
    pipeline = SasLLMPipeline(
        model=model,
        api_key=api_key,
        output_language=args.output_language,
        prompt_builder=builder,
        validator=validator,
        validation_retries=args.validation_retries,
    )

    # run_files chunks every file and batches the corpus via MultiFileBatcher,
    # then runs every batch/singleton through the LLM on one shared thread.
    logger.info(f"running pipeline over {len(sas_files)} file(s) with model={model}")
    outputs = pipeline.run_files(sas_files)

    logger.info(f"pipeline produced {len(outputs)} item response(s)")
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    for out in outputs:
        verdict = _format_verdict(out.get("validation"))
        header = (
            f"=== {out['item_id']} "
            f"({'batch' if out['is_batch'] else out['kind']}) "
            f"files={out['source_files']}{verdict} ==="
        )
        print(f"\n{header}")
        print(out["response"])

        if args.out_dir:
            dest = args.out_dir / f"{out['item_id']}.txt"
            dest.write_text(
                f"{header}\n\n{out['response']}\n", encoding="utf-8"
            )
            logger.debug(f"wrote {dest}")

    if validator is not None:
        _log_validation_summary(outputs)

    return 0


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


if __name__ == "__main__":
    sys.exit(main())
