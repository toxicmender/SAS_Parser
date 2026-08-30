"""One redacted logging and HTTP-trace policy for every v2 entry point."""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from types import TracebackType
from typing import TextIO

from sas_migrate.config.models import ObservabilitySettings

from .redaction import redact_text

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
TRANSPORT_LOGGERS: tuple[str, ...] = (
    "azure",
    "databricks",
    "httpcore",
    "httpx",
    "hvac",
    "kiota",
    "kiota_http",
    "msal",
    "msgraph",
    "msgraph_core",
    "openai",
    "urllib3",
)

LOGGER = logging.getLogger(__name__)


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.exc_info:
            if record.exc_text is None:
                record.exc_text = logging.Formatter().formatException(record.exc_info)
            record.exc_text = redact_text(record.exc_text)
        try:
            message = record.getMessage()
        except (TypeError, ValueError):
            return True
        redacted = redact_text(message)
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True


def _log_crash(
    exc_type: type[BaseException],
    exc: BaseException,
    traceback: TracebackType | None,
) -> None:
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc, traceback)
        return
    LOGGER.critical(
        "unhandled exception; the run did not finish",
        exc_info=(exc_type, exc, traceback),
    )


def _install_crash_handlers() -> None:
    sys.excepthook = _log_crash

    def thread_hook(args: threading.ExceptHookArgs) -> None:
        if issubclass(args.exc_type, SystemExit):
            return
        _log_crash(
            args.exc_type,
            args.exc_value or args.exc_type(),
            args.exc_traceback,
        )

    threading.excepthook = thread_hook


def configure_observability(
    settings: ObservabilitySettings,
    *,
    stream: TextIO | None = None,
) -> tuple[logging.Handler, ...]:
    """Replace root handlers with the v2 redacted console/file policy."""

    formatter = logging.Formatter(LOG_FORMAT)
    console = logging.StreamHandler(stream)
    console.setFormatter(formatter)
    console.addFilter(RedactingFilter())
    handlers: list[logging.Handler] = [console]

    if settings.log_file is not None:
        path = Path(settings.log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.addFilter(RedactingFilter())
        handlers.append(file_handler)

    level = logging.DEBUG if settings.debug else logging.INFO
    logging.basicConfig(level=level, handlers=handlers, force=True)
    transport_level = logging.DEBUG if settings.trace_http else logging.INFO
    for name in TRANSPORT_LOGGERS:
        logging.getLogger(name).setLevel(transport_level)

    if settings.capture_crashes:
        _install_crash_handlers()
    if settings.trace_http:
        LOGGER.info(
            "HTTP tracing enabled for %s; output is redacted but remains sensitive",
            ", ".join(TRANSPORT_LOGGERS),
        )
    return tuple(handlers)


__all__ = [
    "LOG_FORMAT",
    "TRANSPORT_LOGGERS",
    "RedactingFilter",
    "configure_observability",
]
