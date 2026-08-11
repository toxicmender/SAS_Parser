"""Console and file logging for the command-line entry points.

The three entry points (``main``, ``python -m complexity``, ``python -m
validation``) all want the same thing: one line format, a DEBUG switch, and —
when a run is being investigated rather than just run — a log file and the HTTP
wire trace. That setup lived as a copied ``logging.basicConfig`` call in each
of them, which is why none of them grew a file handler.

Three deliberate choices:

**DEBUG does not mean the transport libraries go DEBUG.** ``basicConfig(level=
DEBUG)`` sets the *root* level, so every dependency inherits it, and the
``httpx``/``kiota``/``msal`` chain under a SharePoint run emits enough per
request to bury the first-party lines that say what the run is doing. They are
pinned at INFO unless :func:`configure_logging` is asked for *trace_http*.

**Secrets are redacted at the handler.** A wire trace is the one place bearer
tokens and client secrets reach the log, and a log that cannot be pasted into a
ticket is half a debugging tool. :class:`RedactingFilter` masks them on the way
out — see its docstring for what "redacted" is and is not worth.

**The file gets everything the console gets.** No separate level: a debugging
session that has to be re-run at a different verbosity to get a usable file is
a debugging session that lost its evidence.

Logger name: ``app_config.logging_setup``.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"

#: Libraries on the SharePoint/Graph and LLM transport paths that log per
#: request at DEBUG. Held at INFO unless the wire trace is asked for, so that
#: ``--debug`` stays readable. Names are logger *prefixes*: setting ``kiota``
#: covers ``kiota_http.middleware`` and friends, since a logger inherits the
#: level of its nearest configured ancestor.
TRANSPORT_LOGGERS: tuple[str, ...] = (
    "azure",
    "httpcore",
    "httpx",
    "kiota",
    "kiota_http",
    "msal",
    "msgraph",
    "msgraph_core",
    "openai",
    "urllib3",
)

# What a redacted value is replaced with. Distinctive enough to grep for, so a
# reader can tell "the log redacted this" from "the value was empty".
_MASK = "<redacted>"

#: Patterns whose *last* group is the secret. Ordered most specific first; each
#: is applied to the formatted message once.
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Authorization: Bearer eyJ... — the Graph and gateway request header.
    re.compile(r"(?i)\b(bearer\s+)([A-Za-z0-9\-._~+/]{8,}=*)"),
    # JSON token payloads as MSAL and the gateway return them.
    re.compile(
        r"(?i)([\"']?(?:access_token|refresh_token|id_token|client_secret"
        r"|api_key|secret_id|role_id)[\"']?\s*[:=]\s*[\"']?)([^\s\"',&}]{4,})"
    ),
    # The same names as URL query parameters.
    re.compile(
        r"(?i)([?&](?:access_token|client_secret|api_key|code|sig)=)([^&\s]{4,})"
    ),
)


class RedactingFilter(logging.Filter):
    """Masks bearer tokens and secret-shaped key/value pairs in log records.

    Attached to every handler :func:`configure_logging` installs, because the
    wire trace is exactly the output someone wants to share and exactly the
    output that carries an access token.

    This is a safety net for *our own* logs and the transport libraries', not a
    guarantee: it matches the shapes those libraries actually emit
    (:data:`_SECRET_PATTERNS`), so a secret logged in some other shape passes
    through. Treat a log file as sensitive regardless — the filter lowers the
    cost of a mistake, it does not license pasting logs anywhere.

    Filtering rewrites ``record.msg`` and drops ``record.args`` once the two
    are merged, which is safe here because these records are on their way to a
    handler and are not re-formatted afterwards.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # a broken format string is the caller's bug, not ours
            return True
        redacted = redact(message)
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True


def redact(text: str) -> str:
    """*text* with anything matching :data:`_SECRET_PATTERNS` masked.

    Exposed separately from :class:`RedactingFilter` so callers that print
    rather than log — :mod:`app_config.sharepoint_check` renders its report to
    stdout — can reuse one definition of what counts as a secret.
    """
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(lambda m: f"{m.group(1)}{_MASK}", text)
    return text


def configure_logging(
    *,
    debug: bool = False,
    log_file: str | Path | None = None,
    trace_http: bool = False,
) -> None:
    """Install the console (and optionally file) handlers for a CLI run.

    Parameters
    ----------
    debug : bool
        DEBUG rather than INFO for first-party loggers. The transport
        libraries stay at INFO regardless — see *trace_http*.
    log_file : str | Path | None
        Also write the log to this path, creating parent directories and
        appending to an existing file. Appending rather than truncating is
        deliberate: successive attempts during one debugging session belong in
        one file, in order.
    trace_http : bool
        Let the transport libraries in :data:`TRANSPORT_LOGGERS` log at DEBUG,
        which is what shows the individual Graph requests, their status codes,
        and the retries the SDK middleware performs on its own. Verbose by
        design; pair it with *log_file*.

    Replaces any existing handlers on the root logger, so calling this twice in
    one process (a test, an embedding caller) reconfigures rather than
    duplicating output.
    """
    level = logging.DEBUG if debug else logging.INFO
    formatter = logging.Formatter(LOG_FORMAT)
    redactor = RedactingFilter()

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.addFilter(redactor)
    handlers: list[logging.Handler] = [console]

    resolved: Path | None = None
    if log_file is not None:
        resolved = Path(log_file)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(resolved, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.addFilter(redactor)
        handlers.append(file_handler)

    logging.basicConfig(level=level, handlers=handlers, force=True)

    # Applied after basicConfig so the root level does not drag them along.
    transport_level = logging.DEBUG if trace_http else logging.INFO
    for name in TRANSPORT_LOGGERS:
        logging.getLogger(name).setLevel(transport_level)

    if resolved is not None:
        logger.info(f"logging to '{resolved}'")
    if trace_http:
        logger.info(
            "HTTP wire trace on: transport libraries "
            f"({', '.join(TRANSPORT_LOGGERS)}) log at DEBUG. Tokens are "
            "redacted, but treat the output as sensitive."
        )
