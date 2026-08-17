"""Mandatory response-target validation port."""

from __future__ import annotations

from collections.abc import Collection
from typing import Protocol

from sas_migrate.core.responses import TranslationDocument
from sas_migrate.core.targets import ResolvedTarget
from sas_migrate.core.targets.validation import ResponseValidationResult


class ResponseValidator(Protocol):
    def validate(
        self,
        document: TranslationDocument,
        target: ResolvedTarget,
        *,
        known_chunk_ids: Collection[str],
    ) -> ResponseValidationResult: ...
