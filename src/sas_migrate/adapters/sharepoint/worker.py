"""A synchronous boundary around one long-lived asynchronous event loop."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Coroutine
from concurrent.futures import Future
from typing import Any, Self, TypeVar

T = TypeVar("T")


class WorkerClosedError(RuntimeError):
    """Raised when work is submitted after the worker has been closed."""


class SingleLoopWorker:
    """Run every coroutine on one event loop owned by one worker thread.

    The loop is persistent so async HTTP connection pools never migrate between
    loops. Calls remain safe when the caller is itself a notebook/event-loop
    thread, while submissions are serialized by the worker loop.
    """

    def __init__(self, *, name: str = "sharepoint") -> None:
        self._name = name
        self._ready = threading.Event()
        self._closed = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread = threading.Thread(
            target=self._serve,
            name=f"{name}-event-loop",
            daemon=True,
        )
        self._thread.start()
        self._ready.wait()

    @property
    def thread_id(self) -> int | None:
        return self._thread.ident

    def _serve(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    def run(self, coroutine: Coroutine[Any, Any, T]) -> T:
        """Block until *coroutine* completes on the owned loop."""

        loop = self._loop
        if self._closed or loop is None or loop.is_closed():
            coroutine.close()
            raise WorkerClosedError(f"{self._name} worker is closed")
        if threading.get_ident() == self._thread.ident:
            coroutine.close()
            raise RuntimeError(
                f"{self._name} worker cannot synchronously re-enter its own loop"
            )
        future: Future[T] = asyncio.run_coroutine_threadsafe(coroutine, loop)
        return future.result()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        loop = self._loop
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(loop.stop)
        if threading.get_ident() != self._thread.ident:
            self._thread.join()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


__all__ = ["SingleLoopWorker", "WorkerClosedError"]
