"""Phase 2 golden and property gates for the extracted SAS core."""

from __future__ import annotations

import inspect
import json
import pathlib
import random
import subprocess
import sys
from itertools import pairwise

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
ROOT = SRC.parent
FIXTURE = ROOT / "tests" / "fixtures" / "sas_core_golden.json"
sys.path.insert(0, str(SRC))

from sas_migrate.core.sas import (
    MultiFileBatcher,
    SasBatch,
    SasBatchResult,
    SasCorpus,
    SasSemanticChunker,
    coalesce_into_batches,
)
from sas_migrate.core.sas.dependencies import discover_dependency_edges
from sas_migrate.core.sas.dependencies.context import context_edges
from sas_migrate.core.sas.dependencies.datasets import dataset_edges
from sas_migrate.core.sas.dependencies.macros import macro_edges
from sas_migrate.core.sas.dependencies.models import DependencyEdgeFamily
from sas_migrate.core.sas.metadata.datasets import dataset_inputs, dataset_outputs
from sas_migrate.core.sas.metadata.external import engine_references, path_references
from sas_migrate.core.sas.metadata.macros import defined_macros, invoked_macros
from sas_migrate.core.tokens import TokenBudgetPolicy


def _chunk_summary(chunk) -> dict:
    metadata = chunk.metadata
    return {
        "chunk_id": chunk.chunk_id,
        "kind": chunk.kind.value,
        "start_char": chunk.start_char,
        "end_char": chunk.end_char,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "input_datasets": list(dataset_inputs(metadata)),
        "output_datasets": list(dataset_outputs(metadata)),
        "defines_macros": list(defined_macros(metadata)),
        "invokes_macros": list(invoked_macros(metadata)),
        "produces_macrovars": metadata.produces_macrovars,
        "consumes_macrovars": metadata.consumes_macrovars,
        "declared_macro_vars": metadata.declared_macro_vars,
    }


def _batch_summary(batch) -> dict:
    return {
        "batch_id": batch.batch_id,
        "chunk_ids": [chunk.chunk_id for chunk in batch.chunks],
        "reason": batch.reason,
        "source_files": batch.source_files,
        "input_datasets": batch.input_datasets,
        "output_datasets": batch.output_datasets,
        "is_cross_file": batch.is_cross_file,
        "is_global_context": batch.is_global_context,
    }


def _golden_run() -> tuple[dict, SasCorpus, SasBatchResult]:
    golden = json.loads(FIXTURE.read_text("utf-8"))
    chunker = SasSemanticChunker(min_words=1, max_words=9_999, timeout=None)
    results = [
        chunker.chunk_text(source, source_id=name)
        for name, source in golden["sources"].items()
    ]
    corpus = SasCorpus(file_results=results)
    return golden, corpus, MultiFileBatcher().batch(corpus)


def test_golden_chunks_offsets_metadata_and_source_text_match() -> None:
    golden, corpus, _ = _golden_run()
    for (source_id, source), result in zip(
        golden["sources"].items(), corpus.file_results, strict=True
    ):
        assert [_chunk_summary(chunk) for chunk in result.chunks] == golden["chunks"][source_id]
        for chunk in result.chunks:
            assert source[chunk.start_char : chunk.end_char] == chunk.text


def test_golden_edges_reasons_and_order_match() -> None:
    golden, corpus, batch_result = _golden_run()
    edges = discover_dependency_edges(corpus)
    assert [
        {
            "kind": edge.kind,
            "from": edge.from_chunk_id,
            "to": edge.to_chunk_id,
            "reason": edge.reason,
            "cross_file": edge.cross_file,
        }
        for edge in edges
    ] == golden["edges"]
    assert [_batch_summary(batch) for batch in batch_result.batches] == golden["batches"]
    assert [chunk.chunk_id for chunk in batch_result.singletons] == golden["singletons"]
    assert [
        item.batch_id if isinstance(item, SasBatch) else item.chunk_id
        for item in batch_result.all_ordered_items
    ] == golden["ordered_items"]
    assert dataset_edges(edges) == tuple(
        edge for edge in edges if edge.family is DependencyEdgeFamily.DATASET
    )
    assert macro_edges(edges) == tuple(
        edge
        for edge in edges
        if edge.family
        in {DependencyEdgeFamily.MACRO, DependencyEdgeFamily.MACRO_VARIABLE}
    )
    assert context_edges(edges) == tuple(
        edge for edge in edges if edge.family is DependencyEdgeFamily.CONTEXT
    )


def test_metadata_concern_views_project_the_extracted_metadata() -> None:
    source = (
        'filename input "/landing/source.csv";\n'
        "libname lake '/warehouse';\n"
        "libname edw oracle path=EDW schema=analytics;\n"
        "data work.output; infile input; run;\n"
    )
    result = SasSemanticChunker(
        min_words=1,
        max_words=9_999,
        timeout=None,
    ).chunk_text(source)
    path_refs = [
        reference
        for chunk in result.chunks
        for reference in path_references(chunk.metadata)
    ]
    engine_refs = [
        reference
        for chunk in result.chunks
        for reference in engine_references(chunk.metadata)
    ]
    assert [reference.path for reference in path_refs] == [
        "/landing/source.csv",
        "/warehouse",
    ]
    assert [reference.engine for reference in engine_refs] == ["oracle"]


def test_graph_discovery_is_deterministic_and_does_not_mutate_corpus() -> None:
    _, corpus, _ = _golden_run()
    before = corpus.model_dump_json()
    first = discover_dependency_edges(corpus)
    second = discover_dependency_edges(corpus)
    assert first == second
    assert corpus.model_dump_json() == before


def test_generated_source_is_preserved_by_top_level_chunks() -> None:
    rng = random.Random(20260818)
    chunker = SasSemanticChunker(min_words=1, max_words=9_999, timeout=None)
    for case in range(40):
        statements = [f"v{index} = {rng.randint(0, 10_000)};" for index in range(case + 1)]
        source = f"data work.case_{case};\n" + "\n".join(statements) + "\nrun;\n"
        result = chunker.chunk_text(source, source_id=f"case-{case}.sas")
        top_level = [chunk for chunk in result.chunks if chunk.parent_id is None]
        assert "".join(chunk.text for chunk in top_level) == source
        assert all(source[chunk.start_char : chunk.end_char] == chunk.text for chunk in result.chunks)


def test_oversized_children_overlap_and_cover_the_parent() -> None:
    body = "\n".join(f"value_{index} = {index};" for index in range(80))
    source = f"data work.large;\n{body}\nrun;\n"
    result = SasSemanticChunker(min_words=1, max_words=20, timeout=None).chunk_text(source)
    parent = next(chunk for chunk in result.chunks if chunk.parent_id is None)
    children = [chunk for chunk in result.chunks if chunk.parent_id == parent.chunk_id]
    assert len(children) > 1
    assert children[0].start_char == parent.start_char
    assert children[-1].end_char == parent.end_char
    assert all(right.start_char < left.end_char for left, right in pairwise(children))
    for index in range(parent.start_char, parent.end_char):
        assert any(child.start_char <= index < child.end_char for child in children)


def test_token_packing_uses_shared_policy_not_a_standalone_limit() -> None:
    source = "".join(f"data work.t{index}; x={index}; run;\n" for index in range(3))
    result = SasSemanticChunker(min_words=1, max_words=9_999, timeout=None).chunk_text(source)
    items = MultiFileBatcher().batch(SasCorpus(file_results=[result])).all_ordered_items
    policy = TokenBudgetPolicy(
        max_input_tokens=100,
        reserved_output_tokens=20,
        safety_margin_tokens=10,
    )
    packed = coalesce_into_batches(
        items,
        max_chunks=8,
        policy=policy,
        item_cost=lambda _item: 40,
    )
    assert [len(batch.chunks) for batch in packed] == [1, 1, 1]
    assert "max_tokens" not in inspect.signature(coalesce_into_batches).parameters


def test_sas_core_import_does_not_reach_outer_application_packages() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import sas_migrate.core.sas; "
                "forbidden=('app_config','llm_client','memory','prompt_builder',"
                "'sas_migrate.application','sas_migrate.adapters'); "
                "assert not any(n.startswith(forbidden) for n in sys.modules)"
            ),
        ],
        cwd=SRC,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
