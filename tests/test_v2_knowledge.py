"""Phase 6 knowledge ingestion and attributed retrieval contracts."""

from __future__ import annotations

import asyncio
import pathlib
import sys

import fitz
import pytest
from pydantic import ValidationError

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from sas_migrate.adapters.knowledge import (
    InMemoryKnowledgeRepository,
    PyMuPdfInstructionReader,
)
from sas_migrate.application.knowledge import (
    ConstructKey,
    DocumentExtraction,
    DocumentSection,
    InstructionChunker,
    KnowledgeIngestionService,
    KnowledgeRetriever,
    KnowledgeRole,
    RetrievalQuery,
    RetrievalTier,
    RuleScope,
    UserRule,
    UserRuleSet,
)
from sas_migrate.core.targets import TargetId
from sas_migrate.core.tokens import TokenCategory, TokenEstimator


def _counter() -> TokenEstimator:
    return TokenEstimator(
        encoding="characters",
        text_counter=len,
        estimator="test",
    )


def _section(
    path: str,
    text: str,
    *,
    source_id: str = "guide",
    page: int = 1,
    keys: tuple[ConstructKey, ...] = (),
) -> DocumentSection:
    return DocumentSection(
        source_id=source_id,
        section_path=path,
        text=text,
        page_start=page,
        page_end=page,
        construct_keys=keys,
    )


def _extraction(*sections: DocumentSection) -> DocumentExtraction:
    return DocumentExtraction(
        source_id=sections[0].source_id,
        sections=sections,
        strategy="test",
        page_count=max(section.page_end for section in sections),
    )


def test_chunker_merges_siblings_and_aggregates_construct_attribution() -> None:
    chunks = InstructionChunker(
        _counter(), min_tokens=100, max_tokens=1_000, overlap_tokens=10
    ).chunk(
        (
            _section(
                "Functions > INTNX",
                "date arithmetic",
                keys=(ConstructKey(kind="function", name="INTNX"),),
            ),
            _section(
                "Functions > INTCK",
                "date interval counts",
                page=2,
                keys=(ConstructKey(kind="function", name="INTCK"),),
            ),
        ),
        role=KnowledgeRole.SAS_REFERENCE,
    )
    assert len(chunks) == 1
    assert chunks[0].section_path == "Functions"
    assert {str(key) for key in chunks[0].construct_keys} == {
        "function:intnx",
        "function:intck",
    }
    assert (chunks[0].page_start, chunks[0].page_end) == (1, 2)


def test_chunker_splits_large_units_and_preserves_overlap_budget() -> None:
    text = "\n\n".join(("a" * 25, "b" * 25, "c" * 25, "d" * 25))
    chunks = InstructionChunker(
        _counter(), min_tokens=0, max_tokens=55, overlap_tokens=30
    ).chunk(
        (_section("Guide > Large", text),),
        role=KnowledgeRole.TARGET_GUIDE,
        target=TargetId.SPARK_SQL,
    )
    assert len(chunks) >= 2
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    assert all(chunk.target is TargetId.SPARK_SQL for chunk in chunks)


def test_ingestion_sha_cache_reuses_unchanged_chunks() -> None:
    repository = InMemoryKnowledgeRepository()
    service = KnowledgeIngestionService(
        repository,
        InstructionChunker(_counter(), min_tokens=0, max_tokens=1_000),
    )
    extraction = _extraction(_section("Guide", "content"))

    async def scenario() -> None:
        first = await service.ingest(
            extraction,
            role=KnowledgeRole.SAS_REFERENCE,
            content=b"pdf bytes",
        )
        second = await service.ingest(
            extraction,
            role=KnowledgeRole.SAS_REFERENCE,
            content=b"pdf bytes",
        )
        assert first == second

    asyncio.run(scenario())
    assert repository.write_count == 1


def test_pdf_reader_uses_toc_and_reports_empty_pages() -> None:
    document = fitz.open()
    first = document.new_page()
    first.insert_text((72, 72), "INTNX reference text")
    document.new_page()
    document.set_toc([[1, "Functions > INTNX", 1]])
    content = document.tobytes()
    document.close()
    extraction = PyMuPdfInstructionReader().read(content, source_id="manual.pdf")
    assert extraction.strategy == "toc"
    assert extraction.sections[0].section_path == "Functions > INTNX"
    assert extraction.sections[0].text == "INTNX reference text"
    assert extraction.diagnostics[0].code == "empty_page"


def test_user_rules_parse_and_enforce_scope_target_kind_and_metadata() -> None:
    rules = UserRuleSet.from_markdown(
        """## Output rules
Always include risks.

## [when:proc:sql] [target:spark_sql] SQL rule
Prefer explicit joins.

## [kind:data_step] [meta:uses_hash] Hash rule
Explain hash-object replacement.

## [topic:partitioning] Partition guidance
Discuss repartitioning.
"""
    )
    query = RetrievalQuery(
        text="partitioning",
        target=TargetId.SPARK_SQL,
        constructs=frozenset({"proc:sql"}),
        chunk_kinds=frozenset({"data_step"}),
        metadata_flags=frozenset({"uses_hash"}),
    )
    selected = rules.select(query)
    assert [rule.scope for rule in selected] == [
        RuleScope.ALWAYS,
        RuleScope.CONDITIONAL,
        RuleScope.CONDITIONAL,
        RuleScope.TOPIC,
    ]
    pyspark = query.model_copy(update={"target": TargetId.PYSPARK})
    assert "Prefer explicit joins." not in [rule.text for rule in rules.select(pyspark)]


def test_invalid_conditional_rule_is_rejected() -> None:
    with pytest.raises(ValidationError, match="requires a condition"):
        UserRule(
            rule_id="bad",
            text="bad",
            scope=RuleScope.CONDITIONAL,
        )


def test_retrieval_orders_user_hazard_construct_and_topical_results() -> None:
    repository = InMemoryKnowledgeRepository()
    ingestor = KnowledgeIngestionService(
        repository,
        InstructionChunker(_counter(), min_tokens=0, max_tokens=10_000),
    )
    intnx = ConstructKey(kind="function", name="intnx")
    symput = ConstructKey(kind="call_routine", name="symput")
    extraction = _extraction(
        _section("Functions > INTNX", "date interval arithmetic", keys=(intnx,)),
        _section("Calls > SYMPUT", "macro scope hazard", page=2, keys=(symput,)),
        _section("Spark > Partitioning", "dataframe repartition joins", page=3),
    )
    rules = UserRuleSet.from_markdown(
        "## Always\nReturn auditable output.\n\n"
        "## [when:function:intnx] Dates\nPreserve SAS date epochs."
    )

    async def scenario():
        await ingestor.ingest(
            extraction,
            role=KnowledgeRole.SAS_REFERENCE,
            content=b"corpus",
        )
        return await KnowledgeRetriever(repository, user_rules=rules).select(
            RetrievalQuery(
                text="dataframe repartition joins",
                target=TargetId.SPARK_SQL,
                constructs=frozenset({"function:intnx", "call_routine:symput"}),
                hazards=frozenset({"call_routine:symput"}),
                max_tokens=100_000,
            )
        )

    selection = asyncio.run(scenario())
    tiers = [result.tier for result in selection.results]
    assert tiers[:4] == [
        RetrievalTier.USER_ALWAYS,
        RetrievalTier.USER_WHEN,
        RetrievalTier.HAZARD,
        RetrievalTier.CONSTRUCT,
    ]
    assert RetrievalTier.TOPICAL in tiers


def test_retrieval_returns_distinct_attributed_prompt_categories() -> None:
    repository = InMemoryKnowledgeRepository()
    ingestor = KnowledgeIngestionService(
        repository,
        InstructionChunker(_counter(), min_tokens=0, max_tokens=10_000),
    )

    async def scenario():
        await ingestor.ingest(
            _extraction(_section("Guide > SQL", "databricks sql merge")),
            role=KnowledgeRole.TARGET_GUIDE,
            content=b"guide",
            target=TargetId.SPARK_SQL,
        )
        return await KnowledgeRetriever(
            repository,
            user_rules=UserRuleSet.from_markdown(
                "## Project\nUse catalog-qualified tables."
            ),
        ).select(
            RetrievalQuery(
                text="databricks sql merge",
                target=TargetId.SPARK_SQL,
                max_tokens=100_000,
            )
        )

    selection = asyncio.run(scenario())
    assert {component.category for component in selection.components} == {
        TokenCategory.PROJECT_INSTRUCTIONS,
        TokenCategory.REFERENCE_GUIDANCE,
    }
    assert all(component.source_id for component in selection.components)
