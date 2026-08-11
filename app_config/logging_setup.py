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

**A crash is part of the log.** Python's default hook prints an unhandled
traceback straight to stderr, which is the one place ``--log-file`` cannot
reach — so the captured file used to stop mid-run with nothing saying why,
exactly when it mattered most. :func:`configure_logging` routes unhandled
exceptions (on the main thread and on worker threads) through the handlers
instead, so the traceback lands in the file, redacted, and still reaches the
console once.

Logger name: ``app_config.logging_setup``.
"""

from __future__ import annotations

import logging
import re
import sys
import threading
from pathlib import Path
from types import TracebackType

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

    The traceback is masked too, and separately: an exception's text is not
    part of ``getMessage()``, so a token echoed in the ``str()`` of a Graph
    error would otherwise reach the file untouched on exactly the crash path
    the log is being captured for.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        self._redact_traceback(record)
        try:
            message = record.getMessage()
        except Exception:  # a broken format string is the caller's bug, not ours
            return True
        redacted = redact(message)
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True

    @staticmethod
    def _redact_traceback(record: logging.LogRecord) -> None:
        """Mask ``record.exc_text``, formatting it first if nothing has yet.

        Caching the redacted text on the record is what makes this hold for
        *every* handler: the formatter reuses ``exc_text`` once it is set, so
        the console and the file get the same masked traceback, and re-running
        the patterns over an already-masked string changes nothing.
        """
        if not record.exc_info:
            return
        if not record.exc_text:
            record.exc_text = logging.Formatter().formatException(record.exc_info)
        record.exc_text = redact(record.exc_text)


def redact(text: str) -> str:
    """*text* with anything matching :data:`_SECRET_PATTERNS` masked.

    Exposed separately from :class:`RedactingFilter` so callers that print
    rather than log — :mod:`app_config.sharepoint_check` renders its report to
    stdout — can reuse one definition of what counts as a secret.
    """
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(lambda m: f"{m.group(1)}{_MASK}", text)
    return text


#: What the crash line says before the traceback. Distinctive so that "did the
#: run finish?" is one grep against a captured file.
_CRASH_MESSAGE = "unhandled exception; the run did not finish"


def _log_crash(
    exc_type: type[BaseException],
    exc: BaseException,
    tb: TracebackType | None,
) -> None:
    """Emit an unhandled exception through the handlers rather than to stderr.

    ``KeyboardInterrupt`` is deferred to the default hook: a Ctrl-C is the
    operator ending the run, not a failure, and a CRITICAL traceback for it
    would be noise in the very file this exists to keep readable.
    """
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc, tb)
        return
    logger.critical(_CRASH_MESSAGE, exc_info=(exc_type, exc, tb))


def _install_crash_handlers() -> None:
    """Point :mod:`sys` and :mod:`threading` at :func:`_log_crash`.

    Assigned rather than chained, so configuring twice in one process leaves
    one hook rather than a stack of them that print the same traceback twice.
    The threading hook matters here specifically because the SharePoint client
    drives its async Graph calls from a worker thread — a failure there would
    otherwise reach stderr and never the file.
    """
    sys.excepthook = _log_crash

    def _thread_hook(args: threading.ExceptHookArgs) -> None:
        # SystemExit in a thread only ends that thread and is silent by
        # default; keeping it silent here means no behaviour changes except
        # that a real failure now reaches the file.
        if issubclass(args.exc_type, SystemExit):
            return
        _log_crash(args.exc_type, args.exc_value or args.exc_type(), args.exc_traceback)

    threading.excepthook = _thread_hook


def configure_logging(
    *,
    debug: bool = False,
    log_file: str | Path | None = None,
    trace_http: bool = False,
    capture_crashes: bool = True,
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
    capture_crashes : bool
        Route unhandled exceptions through the handlers, so a traceback is
        captured by *log_file* instead of going only to stderr. On by default
        because that is what a CLI wants; a caller embedding this in a larger
        process that owns its own :data:`sys.excepthook` passes ``False``.

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

    if capture_crashes:
        _install_crash_handlers()

    if resolved is not None:
        logger.info(f"logging to '{resolved}'")
    if trace_http:
        logger.info(
            "HTTP wire trace on: transport libraries "
            f"({', '.join(TRANSPORT_LOGGERS)}) log at DEBUG. Tokens are "
            "redacted, but treat the output as sensitive."
        )
