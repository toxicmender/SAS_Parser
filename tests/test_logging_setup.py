"""
Tests for app_config.logging_setup — the shared CLI logging configuration.

Three behaviours carry the weight: --debug must not drag the HTTP transport
libraries to DEBUG (or the wire trace is always on and the first-party lines
are unreadable), anything secret-shaped must be masked before it reaches a
handler, because the log file exists to be shared, and an unhandled traceback
must reach the log file — a captured run that stops mid-way without saying why
is the failure mode --log-file exists to prevent.
"""

from __future__ import annotations

import logging
import pathlib
import sys
import threading
from types import TracebackType

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

from app_config.logging_setup import (
    TRANSPORT_LOGGERS,
    RedactingFilter,
    configure_logging,
    redact,
)


@pytest.fixture(autouse=True)
def _restore_logging():
    """Put the root handlers, transport levels and crash hooks back."""
    root = logging.getLogger()
    handlers = list(root.handlers)
    level = root.level
    levels = {name: logging.getLogger(name).level for name in TRANSPORT_LOGGERS}
    hooks = (sys.excepthook, threading.excepthook)
    yield
    for handler in list(root.handlers):
        if handler not in handlers:
            handler.close()
    root.handlers[:] = handlers
    root.setLevel(level)
    for name, saved in levels.items():
        logging.getLogger(name).setLevel(saved)
    sys.excepthook, threading.excepthook = hooks


# ---------------------------------------------------------------------------
# Levels
# ---------------------------------------------------------------------------


def test_default_is_info():
    configure_logging()
    assert logging.getLogger().level == logging.INFO


def test_debug_sets_the_root_to_debug():
    configure_logging(debug=True)
    assert logging.getLogger().level == logging.DEBUG


def test_debug_does_not_drag_the_transport_libraries_to_debug():
    configure_logging(debug=True)
    for name in TRANSPORT_LOGGERS:
        assert logging.getLogger(name).level == logging.INFO, name


def test_trace_http_puts_the_transport_libraries_at_debug():
    configure_logging(trace_http=True)
    for name in TRANSPORT_LOGGERS:
        assert logging.getLogger(name).level == logging.DEBUG, name


def test_trace_http_works_without_debug():
    """A wire trace at INFO must still emit: the record passes the library's
    own level and root's handlers have no level of their own."""
    configure_logging(trace_http=True)
    assert logging.getLogger().level == logging.INFO
    assert logging.getLogger("httpx").isEnabledFor(logging.DEBUG)


def test_configuring_twice_replaces_rather_than_duplicates():
    configure_logging()
    first = len(logging.getLogger().handlers)
    configure_logging()
    assert len(logging.getLogger().handlers) == first


# ---------------------------------------------------------------------------
# The log file
# ---------------------------------------------------------------------------


def test_log_file_receives_the_records(tmp_path):
    log = tmp_path / "run.log"
    configure_logging(log_file=log)
    logging.getLogger("test").info("hello from the run")

    assert "hello from the run" in log.read_text(encoding="utf-8")


def test_log_file_parents_are_created(tmp_path):
    log = tmp_path / "deep" / "nested" / "run.log"
    configure_logging(log_file=log)

    assert log.is_file()


def test_log_file_is_appended_not_truncated(tmp_path):
    log = tmp_path / "run.log"
    configure_logging(log_file=log)
    logging.getLogger("test").info("first attempt")
    configure_logging(log_file=log)
    logging.getLogger("test").info("second attempt")

    text = log.read_text(encoding="utf-8")
    assert "first attempt" in text
    assert "second attempt" in text


def test_console_still_gets_records_when_a_file_is_used(tmp_path, capsys):
    configure_logging(log_file=tmp_path / "run.log")
    logging.getLogger("test").info("on both")

    assert "on both" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Crash capture
# ---------------------------------------------------------------------------


def _raise(
    exc: BaseException,
) -> tuple[type[BaseException], BaseException, TracebackType | None]:
    """Real exc_info for *exc*, raised so it carries an actual traceback.

    Built from the exception itself rather than returning ``sys.exc_info()``,
    whose three members are each Optional and would need narrowing at every
    call site.
    """
    try:
        raise exc
    except type(exc):
        return type(exc), exc, exc.__traceback__


def test_an_unhandled_traceback_reaches_the_log_file(tmp_path):
    """The whole point: a captured run must say why it stopped."""
    log = tmp_path / "run.log"
    configure_logging(log_file=log)

    sys.excepthook(*_raise(RuntimeError("the thing exploded")))

    text = log.read_text(encoding="utf-8")
    assert "the thing exploded" in text
    assert "RuntimeError" in text
    assert "Traceback (most recent call last)" in text


def test_the_crash_is_logged_once_after_configuring_twice(tmp_path):
    """Hooks are assigned, not chained, so a second call must not double it."""
    log = tmp_path / "run.log"
    configure_logging(log_file=log)
    configure_logging(log_file=log)

    sys.excepthook(*_raise(RuntimeError("exploded once")))

    assert log.read_text(encoding="utf-8").count("exploded once") == 1


def test_a_keyboard_interrupt_is_left_to_the_default_hook(tmp_path, monkeypatch):
    """Ctrl-C is the operator stopping the run, not a failure to report."""
    log = tmp_path / "run.log"
    configure_logging(log_file=log)
    seen = []
    monkeypatch.setattr(sys, "__excepthook__", lambda *a: seen.append(a))

    sys.excepthook(*_raise(KeyboardInterrupt()))

    assert len(seen) == 1
    assert "KeyboardInterrupt" not in log.read_text(encoding="utf-8")


def test_capture_crashes_false_leaves_the_hook_alone():
    """An embedding caller that owns its own hook keeps it."""
    original = sys.excepthook

    configure_logging(capture_crashes=False)

    assert sys.excepthook is original


def test_a_worker_thread_crash_reaches_the_log_file(tmp_path):
    """The SharePoint client drives Graph from a worker thread; a failure
    there must not be the one that escapes the file."""
    log = tmp_path / "run.log"
    configure_logging(log_file=log)

    def boom():
        raise RuntimeError("worker thread exploded")

    thread = threading.Thread(target=boom)
    thread.start()
    thread.join()

    assert "worker thread exploded" in log.read_text(encoding="utf-8")


def test_a_secret_in_a_traceback_is_masked(tmp_path):
    """The crash path carries exception text, which getMessage() never sees."""
    log = tmp_path / "run.log"
    configure_logging(log_file=log)

    sys.excepthook(*_raise(RuntimeError("GET /me?access_token=abc123def456 failed")))

    text = log.read_text(encoding="utf-8")
    assert "abc123def456" not in text
    assert "<redacted>" in text


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, secret",
    [
        ("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9", "eyJhbGciOiJIUzI1NiJ9"),
        ('{"access_token": "abc123def456"}', "abc123def456"),
        ("client_secret=s3cr3t-value", "s3cr3t-value"),
        ('"api_key": "sk-abcdef123456"', "sk-abcdef123456"),
        ("https://host/cb?access_token=tok123456&x=1", "tok123456"),
    ],
)
def test_redact_masks_secret_shaped_values(text, secret):
    result = redact(text)
    assert secret not in result
    assert "<redacted>" in result


def test_redact_leaves_ordinary_text_alone():
    text = "converting MyApp: 12 file(s) from Apps/MyApp/scripts_original"
    assert redact(text) == text


def test_the_filter_masks_a_record_on_its_way_to_a_handler(tmp_path):
    log = tmp_path / "run.log"
    configure_logging(log_file=log)
    logging.getLogger("test").info("sending Bearer eyJhbGciOiJIUzI1NiJ9 upstream")

    text = log.read_text(encoding="utf-8")
    assert "eyJhbGciOiJIUzI1NiJ9" not in text
    assert "<redacted>" in text


def test_the_filter_merges_args_before_masking():
    """A %-style record is only secret-shaped once its args are merged."""
    record = logging.LogRecord(
        "test", logging.INFO, __file__, 1, "access_token=%s", ("abcdef123456",), None
    )
    RedactingFilter().filter(record)

    assert "abcdef123456" not in record.getMessage()


@pytest.mark.parametrize(
    "text",
    [
        "prompt tokens=1024, completion tokens=256",
        "packed 12 chunks within a token budget of 8192",
        "thread_id=42 token_usage=3110",
    ],
)
def test_redact_leaves_token_counts_alone(text):
    """A bare 'token' is not a secret here: the pipeline logs token *counts*
    constantly, and masking those would gut the logs it exists to protect."""
    assert redact(text) == text


def test_the_filter_passes_a_broken_format_string_through():
    """A caller's formatting bug must not become a swallowed log line."""
    record = logging.LogRecord(
        "test", logging.INFO, __file__, 1, "%d items", ("not a number",), None
    )
    assert RedactingFilter().filter(record) is True
