"""Deterministic section chunking and fingerprinted ingestion."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable

from sas_migrate.application.ports.knowledge import KnowledgeRepository
from sas_migrate.core.targets import TargetId
from sas_migrate.core.tokens import TokenCounter

from .models import (
    DocumentExtraction,
    DocumentSection,
    KnowledgeChunk,
    KnowledgeRole,
    KnowledgeSource,
)

_PARAGRAPHS = re.compile(r"\n\s*\n")


def _sha(value: str | bytes) -> str:
    data = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(data).hexdigest()


class InstructionChunker:
    def __init__(
        self,
        counter: TokenCounter,
        *,
        min_tokens: int = 175,
        max_tokens: int = 1_300,
        overlap_tokens: int = 90,
    ) -> None:
        if not 0 <= overlap_tokens < max_tokens:
            raise ValueError("overlap_tokens must be below max_tokens")
        if not 0 <= min_tokens <= max_tokens:
            raise ValueError("min_tokens must be between zero and max_tokens")
        self._counter = counter
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    @property
    def fingerprint(self) -> str:
        return _sha(
            json.dumps(
                {
                    "encoding": self._counter.encoding,
                    "estimator": self._counter.estimator,
                    "max_tokens": self.max_tokens,
                    "min_tokens": self.min_tokens,
                    "overlap_tokens": self.overlap_tokens,
                    "version": 2,
                },
                sort_keys=True,
            )
        )

    def chunk(
        self,
        sections: Iterable[DocumentSection],
        *,
        role: KnowledgeRole,
        target: TargetId | None = None,
    ) -> tuple[KnowledgeChunk, ...]:
        ordered = tuple(sections)
        chunks: list[KnowledgeChunk] = []
        buffer: list[DocumentSection] = []
        buffer_tokens = 0

        def parent(section: DocumentSection) -> str:
            return " > ".join(section.section_path.split(" > ")[:-1])

        def flush() -> None:
            nonlocal buffer, buffer_tokens
            if not buffer:
                return
            section_path = (
                buffer[0].section_path
                if len(buffer) == 1
                else parent(buffer[0]) or buffer[0].section_path
            )
            body = "\n\n".join(
                section.text.strip() for section in buffer if section.text.strip()
            )
            keys = tuple(
                dict.fromkeys(
                    key for section in buffer for key in section.construct_keys
                )
            )
            tags = frozenset(tag for section in buffer for tag in section.tags)
            windows = self._windows(body)
            for window in windows:
                ordinal = len(chunks)
                text = f"{section_path}\n\n{window}".strip()
                chunks.append(
                    KnowledgeChunk(
                        chunk_id=f"{buffer[0].source_id}::c{ordinal:04d}",
                        source_id=buffer[0].source_id,
                        role=role,
                        section_path=section_path,
                        text=text,
                        page_start=min(section.page_start for section in buffer),
                        page_end=max(section.page_end for section in buffer),
                        ordinal=ordinal,
                        token_count=self._counter.count_text(text),
                        construct_keys=keys,
                        tags=tags,
                        target=target,
                    )
                )
            buffer = []
            buffer_tokens = 0

        for section in ordered:
            if buffer and (
                parent(section) != parent(buffer[0]) or buffer_tokens >= self.min_tokens
            ):
                flush()
            buffer.append(section)
            buffer_tokens += self._counter.count_text(section.text)
        flush()
        return tuple(chunks)

    def _windows(self, text: str) -> tuple[str, ...]:
        if self._counter.count_text(text) <= self.max_tokens:
            return (text,)
        units: list[str] = []
        for paragraph in (value.strip() for value in _PARAGRAPHS.split(text)):
            if not paragraph:
                continue
            units.extend(self._split_unit(paragraph))
        windows: list[str] = []
        current: list[str] = []
        for unit in units:
            candidate = "\n\n".join((*current, unit))
            if current and self._counter.count_text(candidate) > self.max_tokens:
                windows.append("\n\n".join(current))
                current = self._overlap(current)
            current.append(unit)
        if current:
            windows.append("\n\n".join(current))
        return tuple(windows)

    def _split_unit(self, text: str) -> tuple[str, ...]:
        if self._counter.count_text(text) <= self.max_tokens:
            return (text,)
        words = text.split()
        if not words:
            return ()
        ratio = self._counter.count_text(text) / len(words)
        step = max(1, int(self.max_tokens / ratio))
        return tuple(
            " ".join(words[index : index + step])
            for index in range(0, len(words), step)
        )

    def _overlap(self, units: list[str]) -> list[str]:
        selected: list[str] = []
        for unit in reversed(units):
            candidate = "\n\n".join((unit, *selected))
            if self._counter.count_text(candidate) > self.overlap_tokens:
                break
            selected.insert(0, unit)
        return selected


class KnowledgeIngestionService:
    def __init__(
        self,
        repository: KnowledgeRepository,
        chunker: InstructionChunker,
    ) -> None:
        self._repository = repository
        self._chunker = chunker

    async def ingest(
        self,
        extraction: DocumentExtraction,
        *,
        role: KnowledgeRole,
        content: bytes,
        target: TargetId | None = None,
        metadata: dict[str, str] | None = None,
    ) -> tuple[KnowledgeChunk, ...]:
        content_sha = _sha(content)
        extraction_fingerprint = _sha(
            f"{content_sha}\0{self._chunker.fingerprint}\0{role.value}\0{target}"
        )
        if (
            await self._repository.source_fingerprint(extraction.source_id)
            == extraction_fingerprint
        ):
            return await self._repository.chunks(
                source_ids=frozenset({extraction.source_id})
            )
        chunks = self._chunker.chunk(
            extraction.sections,
            role=role,
            target=target,
        )
        source = KnowledgeSource(
            source_id=extraction.source_id,
            role=role,
            content_sha256=content_sha,
            extraction_fingerprint=extraction_fingerprint,
            target=target,
            metadata=metadata or {},
        )
        await self._repository.replace_source(source, chunks)
        return chunks


__all__ = ["InstructionChunker", "KnowledgeIngestionService"]
