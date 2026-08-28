"""Operational local translation adapter composed entirely from v2 services."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from sas_migrate.application.conversion import (
    ConversionTranslationCommand,
    ConversionTranslationResult,
)
from sas_migrate.application.ports import ArtifactWrite, LLMPort
from sas_migrate.application.translation import (
    BudgetedResponseAttemptService,
    PromptAssembler,
    RunStateService,
    TokenAccountingService,
    TokenAuditPersistenceService,
    TokenBudgetEnforcer,
    TranslateCorpus,
    TranslateCorpusRequest,
    TranslationArtifactService,
    TranslationPromptBuilder,
    translation_items,
)
from sas_migrate.application.translation.artifacts import ArtifactLocator
from sas_migrate.core.runs import ItemStatus
from sas_migrate.core.sas import MultiFileBatcher, SasCorpus, SasSemanticChunker
from sas_migrate.core.targets import KNOWN_TARGETS
from sas_migrate.core.tokens import TokenBudgetPolicy, TokenEstimator

from .runtime import (
    DirectoryArtifactRepository,
    InMemoryAcceptedResponseRepository,
    InMemoryRunEventRepository,
    InMemoryTokenRecordRepository,
    SystemClock,
)


class LocalConversionTranslator:
    """Parse, batch, translate, validate, and persist one local request."""

    def __init__(
        self,
        *,
        output_dir: str | Path,
        llm_factory: Callable[[str], LLMPort],
        policy: TokenBudgetPolicy,
        max_attempts: int = 2,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self._artifacts = DirectoryArtifactRepository(output_dir)
        self._llm_factory = llm_factory
        self._policy = policy
        self._max_attempts = max_attempts

    async def translate(
        self,
        command: ConversionTranslationCommand,
    ) -> ConversionTranslationResult:
        run_id = f"conversion-{command.request.request_id}"
        if command.dry_run:
            return await self._plan(command, run_id)

        chunker = SasSemanticChunker()
        results = []
        for source in command.sources:
            try:
                text = source.content.decode("utf-8")
            except UnicodeDecodeError as exc:
                return ConversionTranslationResult(
                    ok=False,
                    error=f"source {source.name!r} is not valid UTF-8: {exc}",
                )
            results.append(chunker.chunk_text(text, source_id=source.name))
        batches = MultiFileBatcher().batch(SasCorpus(file_results=results))
        items = translation_items(batches)
        if not items:
            return ConversionTranslationResult(
                ok=False,
                error="SAS parsing produced no translatable items",
            )

        clock = SystemClock()
        events = InMemoryRunEventRepository()
        tokens = InMemoryTokenRecordRepository()
        memory = InMemoryAcceptedResponseRepository()
        counter = TokenEstimator(command.model)
        assembler = PromptAssembler(counter)
        service = TranslateCorpus(
            attempts=BudgetedResponseAttemptService(
                llm=self._llm_factory(command.model),
                budgets=TokenBudgetEnforcer(assembler),
                accounting=TokenAccountingService(counter),
                audit=TokenAuditPersistenceService(tokens, self._artifacts),
            ),
            prompts=TranslationPromptBuilder(assembler),
            artifacts=TranslationArtifactService(self._artifacts),
            runs=RunStateService(
                events=events,
                memory=memory,
                token_records=tokens,
                clock=clock,
            ),
            memory=memory,
            token_records=tokens,
        )
        outcome = await service.run(
            TranslateCorpusRequest(
                run_id=run_id,
                thread_id=f"local-{command.request.request_id}",
                target=command.target,
                items=items,
                policy=self._policy,
                max_attempts=self._max_attempts,
            )
        )
        summary = await self._artifacts.write(
            run_id,
            ArtifactWrite(
                artifact_id="run-summary.json",
                media_type="application/json",
                content=(outcome.model_dump_json(indent=2) + "\n").encode(),
                metadata={"kind": "conversion_run_summary"},
            ),
        )
        summary_locator = ArtifactLocator(
            artifact_id="run-summary.json",
            location=summary,
            kind="conversion_run_summary",
            media_type="application/json",
        )
        failures = tuple(item for item in outcome.items if item.status is ItemStatus.FAILED)
        return ConversionTranslationResult(
            ok=not failures,
            artifacts=(*outcome.artifacts, summary_locator),
            error=("; ".join(item.error or "translation failed" for item in failures) or None),
        )

    async def _plan(
        self,
        command: ConversionTranslationCommand,
        run_id: str,
    ) -> ConversionTranslationResult:
        artifact_id = "conversion-plan.json"
        content = json.dumps(
            {
                "schema_version": 2,
                "request_id": command.request.request_id,
                "application_name": command.request.application_name,
                "target": command.target.target.value,
                "sqlglot_dialect": next(
                    definition.sqlglot_dialect
                    for definition in KNOWN_TARGETS
                    if definition.target is command.target.target
                ),
                "model": command.model,
                "sources": [source.name for source in command.sources],
                "token_policy": self._policy.model_dump(mode="json"),
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n"
        location = await self._artifacts.write(
            run_id,
            ArtifactWrite(
                artifact_id=artifact_id,
                media_type="application/json",
                content=content.encode(),
                metadata={"kind": "conversion_plan"},
            ),
        )
        return ConversionTranslationResult(
            ok=True,
            artifacts=(
                ArtifactLocator(
                    artifact_id=artifact_id,
                    location=location,
                    kind="conversion_plan",
                    media_type="application/json",
                ),
            ),
        )


__all__ = ["LocalConversionTranslator"]
