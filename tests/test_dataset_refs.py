"""SasDatasetRef: the stored source of a chunk's dataset metadata.

``SasChunkMetadata.dataset_refs`` records every SAS dataset a chunk names and
what it does with it; ``input_datasets``, ``output_datasets``,
``dropped_datasets`` and the ``body_*`` lists are views of it. These tests pin
the views, the two ways to rewrite the references, and that metadata written
as lists — old JSON, or a caller's constructor — still loads.
"""

from __future__ import annotations

import pytest

from chunker import DatasetRole, SasChunkMetadata, SasDatasetRef
from chunker.metadata import _merge_meta

R, W, U, D = DatasetRole.READ, DatasetRole.WRITE, DatasetRole.UPDATE, DatasetRole.DROP


def _ref(name: str, role: DatasetRole, **kw) -> SasDatasetRef:
    return SasDatasetRef(name=name, role=role, **kw)


def _param(param: str, pos: int, role: DatasetRole) -> SasDatasetRef:
    return _ref(f"&{param}", role, in_macro_body=True, param=param, param_pos=pos)


def _meta(*refs: SasDatasetRef) -> SasChunkMetadata:
    return SasChunkMetadata(dataset_refs=refs)


# ── the views ────────────────────────────────────────────────────────────────


def test_views_split_references_by_role():
    meta = _meta(
        _ref("work.b", R),
        _ref("lib.master", U),
        _ref("work.a", W),
        _ref("work.tmp", D),
        _ref("work.b", R, via="merge"),  # the same dataset named twice
    )
    # UPDATE reads and writes; first-seen order, each name once.
    assert meta.input_datasets == ["work.b", "lib.master"]
    assert meta.output_datasets == ["lib.master", "work.a"]
    assert meta.dropped_datasets == ["work.tmp"]


def test_macro_body_references_stay_out_of_the_chunk_lists():
    meta = _meta(
        _ref("lib.in", R, in_macro_body=True),
        _ref("lib.out", W, in_macro_body=True),
        _param("ds", 0, R),
        _param("out", -1, W),
        _param("ds", 0, R),
    )
    assert (meta.input_datasets, meta.output_datasets) == ([], [])
    assert meta.body_literal_inputs == ["lib.in"]
    assert meta.body_literal_outputs == ["lib.out"]
    assert meta.body_param_inputs == [{"param": "ds", "pos": 0}]
    assert meta.body_param_outputs == [{"param": "out", "pos": -1}]


def test_a_chunk_reads_its_datasets_in_source_order():
    meta = _meta(_ref("work.z", W), _ref("work.a", W))
    assert meta.output_datasets == ["work.z", "work.a"]  # never sorted: _LAST_


def test_the_views_follow_the_references_they_come_from():
    # The views are cached per references tuple: a copy with new references,
    # an assignment, or a caller editing a view it was handed cannot leave a
    # stale answer behind.
    meta = _meta(_ref("work.a", R))
    meta.input_datasets.append("work.junk")
    assert meta.input_datasets == ["work.a"]
    copy = meta.add_dataset_refs([_ref("work.b", R)])
    assert (meta.input_datasets, copy.input_datasets) == (["work.a"], ["work.a", "work.b"])
    meta.dataset_refs = (_ref("work.c", R),)
    assert meta.input_datasets == ["work.c"]
    # And the cache is no part of the model: equality and dumps ignore it.
    assert meta == _meta(_ref("work.c", R))
    assert "_dataset_views_cache" not in meta.model_dump()
    assert "_dataset_views_cache" not in str(meta)


# ── lists as input ───────────────────────────────────────────────────────────


LISTS = {
    "input_datasets": ["work.a", "lib.b"],
    "output_datasets": ["work.c", "work.a"],
    "body_literal_inputs": ["lib.x"],
    "body_literal_outputs": ["work.y"],
    "body_param_inputs": [{"param": "ds", "pos": 0}],
    "body_param_outputs": [{"param": "out", "pos": -1}],
}


def test_lists_build_the_references_behind_them():
    meta = SasChunkMetadata(**LISTS)
    for view, value in LISTS.items():
        assert getattr(meta, view) == value, view
    assert _ref("work.a", R) in meta.dataset_refs
    assert _ref("work.a", W) in meta.dataset_refs


def test_old_json_without_references_loads():
    meta = SasChunkMetadata.model_validate({"step_name": "c", **LISTS})
    assert meta.step_name == "c"
    assert meta.output_datasets == ["work.c", "work.a"]


@pytest.mark.parametrize("mode", ["python", "json"])
def test_dumped_metadata_round_trips(mode):
    meta = SasChunkMetadata(
        dataset_refs=(_ref("work.a", U, via="modify"), _param("ds", 0, R)),
        invokes_macros=["m"],
    )
    dumped = meta.model_dump(mode=mode)
    # The views are serialised next to the references they come from...
    assert dumped["input_datasets"] == ["work.a"]
    # ...and give way to them when the dump is loaded back.
    assert SasChunkMetadata.model_validate(dumped) == meta
    assert SasChunkMetadata.model_validate_json(meta.model_dump_json()) == meta


def test_a_view_cannot_be_updated_in_place():
    meta = _meta(_ref("work.a", R))
    with pytest.raises(ValueError, match="views of dataset_refs"):
        meta.model_copy(update={"input_datasets": ["work.b"]})


# ── rewriting ────────────────────────────────────────────────────────────────


def test_map_dataset_names_renames_drops_and_spares_parameters():
    meta = _meta(_ref("&t", R), _ref("_last_", R), _ref("work.x", W), _param("ds", 0, R))

    def rename(ref: SasDatasetRef) -> str | None:
        return {"&t": "work.t", "_last_": None}.get(ref.name, ref.name)

    mapped = meta.map_dataset_names(rename)
    assert mapped.input_datasets == ["work.t"]
    assert mapped.output_datasets == ["work.x"]
    assert mapped.body_param_inputs == [{"param": "ds", "pos": 0}]
    assert meta.input_datasets == ["&t", "_last_"]  # the original is untouched


def test_map_dataset_names_merges_names_that_meet():
    meta = _meta(_ref("&a", R), _ref("&b", R))
    mapped = meta.map_dataset_names(lambda ref: "work.same")
    assert mapped.dataset_refs == (_ref("work.same", R),)


def test_unchanged_rewrites_return_the_same_metadata():
    meta = _meta(_ref("work.a", R))
    assert meta.map_dataset_names(lambda ref: ref.name) is meta
    assert meta.add_dataset_refs([_ref("work.a", R)]) is meta


def test_add_dataset_refs_appends_after_the_chunks_own():
    meta = _meta(_ref("work.a", W))
    added = meta.add_dataset_refs([_ref("work.b", W, via="macro_call"), _ref("work.a", W)])
    assert added.output_datasets == ["work.a", "work.b"]
    assert len(added.dataset_refs) == 2


# ── split regions ────────────────────────────────────────────────────────────


def test_a_split_child_keeps_the_regions_order():
    parent = _meta(_ref("work.zz", W), _ref("work.aa", W))
    child = _meta(_ref("work.aa", W), _ref("work.mid", W))
    merged = _merge_meta(parent, child)
    assert merged.output_datasets == ["work.zz", "work.aa", "work.mid"]
