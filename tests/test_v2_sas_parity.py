"""Differential legacy/v2 parity; legacy models are test-only adapters."""

from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from chunker import SasCorpus as LegacyCorpus
from chunker import SasSemanticChunker as LegacyChunker
from chunker.batcher import MultiFileBatcher as LegacyBatcher
from chunker.batcher import coalesce_into_batches as legacy_coalesce
from sas_migrate.core.sas import (
    MultiFileBatcher,
    SasCorpus,
    SasSemanticChunker,
    coalesce_into_batches,
)
from sas_migrate.core.tokens import TokenBudgetPolicy

SOURCES = (
    "data work.out; set ext.in; if first.id then total=0; total+amount; run;",
    "proc sql; create table work.out as select * from ext.in where id in (select id from ext.keep); quit;",
    "%macro build(ds,out); data &out.; set &ds.; run; %mend; %build(work.in,work.out);",
    "filename feed ftp '/incoming/orders.csv'; data work.x; infile feed; input id amount; run;",
    "libname edw oracle path=EDW schema=finance user='&user.' pass='&pass.';",
    "data work.a work.b; set work.src; if flag then output work.a; else output work.b; run;",
    "options user=permlib; data x; set y; run;",
    "/* heading */\n%let cutoff=10;\ndata work.x; set work.y; if amount>&cutoff.; run;",
)


def _dump(model) -> dict:
    return model.model_dump(mode="json")


@pytest.mark.parametrize("source", SOURCES)
def test_single_source_chunking_matches_legacy(source: str) -> None:
    kwargs = {"min_words": 1, "max_words": 9_999, "timeout": None}
    legacy = LegacyChunker(**kwargs).chunk_text(source, source_id="parity.sas")
    modern = SasSemanticChunker(**kwargs).chunk_text(source, source_id="parity.sas")
    assert _dump(modern) == _dump(legacy)


def test_oversized_chunking_and_overlap_match_legacy() -> None:
    body = "\n".join(f"value_{index} = {index};" for index in range(100))
    source = f"data work.large;\n{body}\nrun;\n"
    kwargs = {"min_words": 1, "max_words": 24, "timeout": None}
    assert _dump(SasSemanticChunker(**kwargs).chunk_text(source)) == _dump(
        LegacyChunker(**kwargs).chunk_text(source)
    )


def test_cross_file_batching_matches_legacy() -> None:
    sources = (
        "data work.a; set ext.raw; run;\n%macro report(ds); proc print data=&ds.; run; %mend;",
        "%report(work.a);\ndata work.b; set work.a; run;\n",
        "proc means data=work.b; run;\n",
    )
    kwargs = {"min_words": 1, "max_words": 9_999, "timeout": None}
    legacy_results = [
        LegacyChunker(**kwargs).chunk_text(source, source_id=f"file-{index}.sas")
        for index, source in enumerate(sources)
    ]
    modern_results = [
        SasSemanticChunker(**kwargs).chunk_text(source, source_id=f"file-{index}.sas")
        for index, source in enumerate(sources)
    ]
    legacy = LegacyBatcher().batch(LegacyCorpus(file_results=legacy_results))
    modern = MultiFileBatcher().batch(SasCorpus(file_results=modern_results))
    assert _dump(modern) == _dump(legacy)


def test_policy_packing_matches_legacy_token_limit() -> None:
    source = "".join(f"data work.t{index}; x={index}; run;\n" for index in range(5))
    kwargs = {"min_words": 1, "max_words": 9_999, "timeout": None}
    legacy_result = LegacyChunker(**kwargs).chunk_text(source)
    modern_result = SasSemanticChunker(**kwargs).chunk_text(source)
    legacy_items = LegacyBatcher().batch(
        LegacyCorpus(file_results=[legacy_result])
    ).all_ordered_items
    modern_items = MultiFileBatcher().batch(
        SasCorpus(file_results=[modern_result])
    ).all_ordered_items
    costs = lambda _item: 35
    legacy_packed = legacy_coalesce(
        legacy_items,
        max_chunks=8,
        max_tokens=80,
        item_cost=costs,
    )
    policy = TokenBudgetPolicy(
        max_input_tokens=100,
        reserved_output_tokens=10,
        safety_margin_tokens=10,
    )
    modern_packed = coalesce_into_batches(
        modern_items,
        max_chunks=8,
        policy=policy,
        item_cost=costs,
    )
    assert [_dump(batch) for batch in modern_packed] == [
        _dump(batch) for batch in legacy_packed
    ]
