"""
test_data_hydration_includes.py — where the scripts a corpus %INCLUDEs are.

An %INCLUDE names a script by its path on the SAS server, which means nothing
here; the script is looked for by file name, in the local corpus and in the
application's SharePoint scripts folder. Pinned:

* **The file name an %INCLUDE opens**, however it is spelled: a path, a member
  of a fileref's directory, a macro variable.
* **Where it was found**, ignoring case and at any depth, with "nowhere looked"
  kept apart from "looked and not found".
* **The CLIs**: ``python -m data_hydration --check-includes`` reports and
  stores it; a FILENAME only %INCLUDE reads is code, never a load.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

from chunker import SasCorpus, SasSemanticChunker, resolve_corpus_references
from data_hydration.config import HydrationConfig
from data_hydration.includes import (
    include_checks,
    include_file_name,
    is_include,
    local_index,
    match_includes,
    sharepoint_index,
)
from data_hydration.inventory import InventoryRow, inventory_rows, plan_from_inventory


def _rows(**files: str) -> list[InventoryRow]:
    chunker = SasSemanticChunker()
    corpus = SasCorpus(
        file_results=[chunker.chunk_text(t, source_id=f"{n}.sas") for n, t in files.items()]
    )
    return inventory_rows(resolve_corpus_references(corpus).file_results)


def _includes(src: str) -> list[InventoryRow]:
    return [r for r in _rows(job=src) if is_include(r)]


# ── the file name ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("src", "name"),
    [
        ("%include '/sas/prod/macros/Util.sas';", "Util.sas"),
        ("%include 'C:\\SAS\\Macros\\load.sas';", "load.sas"),
        ('%let root = /SAS;\n%include "&root/lib/x.sas";', "x.sas"),
        # Unresolved but for the directory: the file name is still known.
        ('%include "&nowhere/lib/y.sas";', "y.sas"),
        # A member of a fileref's directory, bound or not: SAS adds .sas.
        ("filename src '/code';\n%include src(util);", "util.sas"),
        ("%include src(util);", "util.sas"),
        ("%include src('setup.txt');", "setup.txt"),
        ("%let f = src(util);\n%include &f;", "util.sas"),
        # No file name to look for.
        ("%include &f;", None),
        ('%include "/lib/&name..sas";', None),
        # A whole-file fileref nothing binds: its name is the FILENAME's.
        ("%include setup;", None),
    ],
)
def test_the_file_name_an_include_opens(src, name):
    [row] = _includes(src + "\n")
    assert include_file_name(row) == name


def test_a_bound_whole_file_fileref_opens_its_file():
    [row] = _includes("filename setup '/code/setup.sas';\n%include setup;\n")
    assert include_file_name(row) == "setup.sas"


# ── the places ───────────────────────────────────────────────────────────────


def test_the_local_index_lists_every_file_by_lowercased_name(tmp_path):
    (tmp_path / "macros").mkdir()
    (tmp_path / "macros" / "Util.SAS").write_text("")
    (tmp_path / "util.sas").write_text("")
    (tmp_path / "job.sas").write_text("")
    assert local_index(tmp_path) == {
        "util.sas": ("macros/Util.SAS", "util.sas"),
        "job.sas": ("job.sas",),
    }


class _Library:
    """A document library's folders, as SharePointClient.list_directory
    sees them."""

    def __init__(self, tree: dict[str, list[str]]):
        self.tree = tree
        self.listed: list[str] = []

    def list_directory(self, path=""):
        self.listed.append(path)
        return [
            {"name": name.rstrip("/"), "is_folder": name.endswith("/")}
            for name in self.tree.get(path, [])
        ]


def test_the_sharepoint_index_walks_every_folder():
    library = _Library(
        {
            "App/scripts_original": ["job.sas", "macros/", "Readme.md"],
            "App/scripts_original/macros": ["Util.sas", "deep/"],
            "App/scripts_original/macros/deep": ["util.sas"],
        }
    )
    index = sharepoint_index("/App/scripts_original/", client=library)
    assert index["util.sas"] == (
        "App/scripts_original/macros/Util.sas",
        "App/scripts_original/macros/deep/util.sas",
    )
    assert set(index) == {"job.sas", "readme.md", "util.sas"}


def test_a_sharepoint_walk_stops_at_its_folder_limit(monkeypatch, caplog):
    import data_hydration.includes as includes

    monkeypatch.setattr(includes, "MAX_SHAREPOINT_FOLDERS", 2)
    library = _Library({"a": ["b/", "x.sas"], "a/b": ["c/", "y.sas"], "a/b/c": ["z.sas"]})
    assert set(sharepoint_index("a", client=library)) == {"x.sas", "y.sas"}
    assert "stopped after 2 folder(s)" in caplog.text


# ── matching ─────────────────────────────────────────────────────────────────


SRC = (
    "%include '/sas/prod/Util.sas';\n"
    "%include '/sas/prod/common.sas';\n"
    "%include '/elsewhere/missing.sas';\n"
    "%include &f;\n"
    "libname raw '/data/raw';\n"
)


def test_each_include_records_where_its_script_was_found():
    local = {"util.sas": ("macros/util.sas",)}
    sharepoint = {"common.sas": ("App/scripts_original/common.sas",), "util.sas": ("App/u/util.sas",)}
    rows = match_includes(_rows(job=SRC), local=local, sharepoint=sharepoint)
    found = {r.raw: (r.found_local, r.found_sharepoint) for r in rows if is_include(r)}
    assert found == {
        "/sas/prod/Util.sas": (("macros/util.sas",), ("App/u/util.sas",)),
        "/sas/prod/common.sas": ((), ("App/scripts_original/common.sas",)),
        "/elsewhere/missing.sas": ((), ()),
        # Searched, but there was no name to search for.
        "&f": ((), ()),
    }
    # Every other row is as it was.
    [libname] = [r for r in rows if r.statement == "libname"]
    assert (libname.found_local, libname.found_sharepoint) == (None, None)


def test_a_place_nobody_looked_in_stays_none():
    rows = match_includes(_rows(job=SRC), local={"util.sas": ("u.sas",)})
    assert {r.found_sharepoint for r in rows} == {None}


def test_the_checks_name_each_script_once_with_everyone_who_includes_it():
    rows = match_includes(
        _rows(a=SRC, b="%include '/other/place/UTIL.sas';\n"),
        local={"util.sas": ("macros/util.sas",)},
    )
    checks = {c.file_name or c.spelled: c for c in include_checks(rows)}
    assert list(checks) == ["Util.sas", "common.sas", "missing.sas", "&f"]
    util = checks["Util.sas"]
    assert util.included_by == (("a.sas", 1), ("b.sas", 1))
    assert (util.status, util.local, util.sharepoint) == ("found", ("macros/util.sas",), None)
    assert checks["missing.sas"].status == "missing"
    assert checks["&f"].status == "unnamed"
    # Unchecked: nobody looked.
    assert {c.status for c in include_checks(_rows(job=SRC))} == {"unchecked"}


def test_the_matches_survive_the_table_round_trip():
    from data_hydration.inventory import _values

    [row] = [
        r
        for r in match_includes(_rows(job=SRC), local={"util.sas": ("m/util.sas",)})
        if r.raw.endswith("Util.sas")
    ]
    stored = row.model_dump()
    stored["found_local"] = ["m/util.sas"]
    assert InventoryRow.model_validate(stored) == row
    values = dict(zip(InventoryRow.model_fields, _values(row)))
    assert (values["found_local"], values["found_sharepoint"]) == (["m/util.sas"], None)


# ── the plan ─────────────────────────────────────────────────────────────────


def test_a_fileref_only_include_reads_names_code_not_data():
    rows = _rows(
        setup="filename src '/code/macros';\nfilename feed '/data/feed';\n",
        job="%include src(util);\ndata a; infile feed(day1.csv); input x; run;\n",
    )
    plan = plan_from_inventory(rows, config=HydrationConfig(catalog="main"))
    # The macro directory is SAS to convert; the feed directory holds data.
    assert [i.source.locator for i in plan.items] == ["/data/feed"]


def test_a_fileref_also_read_by_infile_holds_data():
    rows = _rows(job="filename both '/data/mixed';\n%include both(a);\ndata x; infile both(b.csv); run;\n")
    plan = plan_from_inventory(rows, config=HydrationConfig(catalog="main"))
    assert [i.source.locator for i in plan.items] == ["/data/mixed"]


# ── the CLI ──────────────────────────────────────────────────────────────────


@pytest.fixture
def corpus_dir(tmp_path):
    (tmp_path / "macros").mkdir()
    (tmp_path / "macros" / "util.sas").write_text("%macro util; %mend;\n")
    (tmp_path / "job.sas").write_text(
        "%include '/sas/prod/util.sas';\n%include '/sas/prod/common.sas';\n"
    )
    return tmp_path


def _main(*argv: str) -> int:
    from data_hydration.__main__ import main

    return main(list(argv))


def test_the_cli_reports_where_each_include_was_found(corpus_dir, capsys):
    assert _main(str(corpus_dir), "--dry-run", "--check-includes") == 0
    out = capsys.readouterr().out
    assert "Included scripts — 2, 1 not found" in out
    assert "util.sas  ->  local: macros/util.sas" in out
    assert "common.sas  ->  ** missing" in out


def test_the_cli_looks_in_sharepoint_and_stores_the_answer(corpus_dir, monkeypatch, capsys):
    import data_hydration.includes as includes
    import data_hydration.inventory as inventory

    listed: list[str] = []

    def fake_index(folder, **_):
        listed.append(folder)
        return {"common.sas": (f"{folder}/common.sas",)}

    from app_config.sharepoint import SharePointConfig

    stored: dict[str, list[InventoryRow]] = {}
    monkeypatch.setattr(includes, "sharepoint_index", fake_index)
    monkeypatch.setattr(
        SharePointConfig, "from_env", classmethod(lambda cls: SharePointConfig(file_server_base_path="Apps"))
    )
    monkeypatch.setattr(
        inventory, "write_inventory", lambda rows, table, **_: stored.setdefault(table, rows)
    )
    argv = [str(corpus_dir), "--dry-run", "--sharepoint-app", "MyApp", "--inventory-table", "m.s.t"]
    assert _main(*argv) == 0
    assert listed == ["Apps/MyApp/scripts_original"]
    out = capsys.readouterr().out
    assert "common.sas  ->  SharePoint: Apps/MyApp/scripts_original/common.sas" in out
    [common] = [r for r in stored["m.s.t"] if r.raw.endswith("common.sas")]
    assert (common.found_local, common.found_sharepoint) == (
        (),
        ("Apps/MyApp/scripts_original/common.sas",),
    )


def test_a_sharepoint_folder_that_cannot_be_listed_fails_the_run_not_the_check(
    corpus_dir, monkeypatch, capsys
):
    import data_hydration.includes as includes

    def refuse(*_, **__):
        raise RuntimeError("Graph says no")

    monkeypatch.setattr(includes, "sharepoint_index", refuse)
    assert _main(str(corpus_dir), "--dry-run", "--sharepoint-app", "MyApp") == 1
    captured = capsys.readouterr()
    assert "util.sas  ->  local: macros/util.sas" in captured.out
    assert "could not list SharePoint folder" in captured.err


def test_without_the_flag_nothing_is_looked_for(corpus_dir, capsys):
    assert _main(str(corpus_dir), "--dry-run") == 0
    assert "Included scripts" not in capsys.readouterr().out
