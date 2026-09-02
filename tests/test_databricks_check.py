"""Tests for the Databricks runtime preflight (app_config.databricks_check).

The three hazards this module exists to catch are all *environment* shapes, so
every test here fabricates one: a child process on a cluster, a Spark Connect
session, a mismatched pyspark, a failed library install. None of them needs a
cluster, and none needs pyspark installed.
"""

from __future__ import annotations

import json
import pathlib
import sys
import types

import pytest

from app_config import databricks_check as dc
from app_config.sharepoint_check import FAIL, PASS, SKIP, WARN


# ---------------------------------------------------------------------------
# Stand-ins
# ---------------------------------------------------------------------------


class _Ctx:
    def __init__(self, master: str) -> None:
        self.master = master


class _Session:
    """A SparkSession stand-in. ``master=None`` models a Spark Connect client."""

    def __init__(self, version: str = "4.2.0", master: str | None = "spark://x:7077"):
        self.version = version
        self._master = master

    @property
    def sparkContext(self) -> _Ctx:
        if self._master is None:
            raise RuntimeError("sparkContext is not supported in Spark Connect")
        return _Ctx(self._master)


class _Shell:
    """An IPython shell stand-in. Its identity is all the detector reads."""


@pytest.fixture
def on_cluster(monkeypatch):
    """Make :func:`in_databricks_runtime` report a cluster."""
    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "19.0")
    monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)


@pytest.fixture
def off_cluster(monkeypatch):
    monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)


@pytest.fixture
def in_repl(monkeypatch):
    """Put an IPython shell in ``sys.modules`` — i.e. be the notebook's Python.

    IPython is not installed here and does not need to be: the detector only
    ever *looks* in ``sys.modules``, deliberately, so that finding out whether
    this is the notebook cannot import the modules whose import is the problem.
    That design choice is what makes every process shape fakeable in-process.
    """
    monkeypatch.setitem(
        sys.modules, "IPython", types.SimpleNamespace(get_ipython=lambda: _Shell())
    )


def _session(monkeypatch, session):
    monkeypatch.setattr(dc, "_active_session", lambda: session)


# ---------------------------------------------------------------------------
# runtime — hazard: a %sh/subprocess child is not the notebook
# ---------------------------------------------------------------------------


def test_runtime_skips_off_cluster(off_cluster, monkeypatch):
    _session(monkeypatch, None)
    result = dc.check_runtime()
    assert result.status == SKIP
    assert "not running on a Databricks cluster" in result.summary


def test_runtime_passes_in_the_notebook(on_cluster, in_repl, monkeypatch):
    _session(monkeypatch, _Session())
    result = dc.check_runtime()
    assert result.status == PASS
    assert "19.0" in result.summary
    assert result.detail["notebook REPL"].startswith("yes")
    assert result.detail["active SparkSession"] == "yes"


def test_runtime_passes_in_the_notebook_without_a_session(on_cluster, in_repl, monkeypatch):
    """A session is no longer required for a pass, either — it never meant this."""
    _session(monkeypatch, None)
    assert dc.check_runtime().status == PASS


def test_runtime_detects_a_child_process(on_cluster, monkeypatch):
    """The env var says cluster, but there is no REPL: a %sh child."""
    _session(monkeypatch, None)
    result = dc.check_runtime()
    assert result.status == FAIL
    assert "child process" in result.summary
    # The fix must name the actual cause, not restate the symptom.
    assert result.fix is not None
    assert "%sh" in result.fix and "DATABRICKS_TOKEN" in result.fix


def test_runtime_detects_a_child_process_that_has_a_session(on_cluster, monkeypatch):
    """The regression this stage was rewritten for.

    A child process acquires a SparkSession as a *side effect* of the failure
    being diagnosed: the SDK's ``runtime`` auth strategy imports ``dbruntime``
    on its way to raising, and that import attaches Py4J to the driver's JVM.
    The old session-based discriminator therefore reported a pass for every
    caller that ran after the first failed secret read. The REPL is the signal
    with no such lag.
    """
    _session(monkeypatch, _Session())
    result = dc.check_runtime()
    assert result.status == FAIL
    assert "child process" in result.summary
    assert result.detail["active SparkSession"] == "yes"
    assert result.detail["notebook REPL"].startswith("no")


def test_runtime_warns_for_a_child_process_carrying_a_pat(on_cluster, monkeypatch):
    """A child with its own credential is a supported setup, not a failure."""
    monkeypatch.setenv("DATABRICKS_TOKEN", "dapi-child")
    _session(monkeypatch, None)
    result = dc.check_runtime()
    assert result.status == WARN
    assert "DATABRICKS_TOKEN" in result.summary
    assert result.detail["workspace credential"] == "DATABRICKS_TOKEN is set"


def test_runtime_reports_the_repl_context_without_the_token(on_cluster, in_repl, monkeypatch):
    """The REPL context carries apiToken. It must never reach the report."""
    context = types.SimpleNamespace(
        notebookId="4321", clusterId="0101-x-abcd", apiToken="dapi-SECRET-VALUE"
    )
    monkeypatch.setitem(
        sys.modules,
        "dbruntime.databricks_repl_context",
        types.SimpleNamespace(get_context=lambda: context),
    )
    _session(monkeypatch, _Session())
    result = dc.check_runtime()
    assert result.status == PASS
    assert "notebookId=4321" in result.detail["REPL context"]
    assert "dapi-SECRET-VALUE" not in json.dumps(result.detail)
    assert "dapi-SECRET-VALUE" not in dc.to_json([result])


# ---------------------------------------------------------------------------
# session — hazard: the master we assume is not the master in force
# ---------------------------------------------------------------------------


def test_session_reports_the_real_master(off_cluster, monkeypatch):
    _session(monkeypatch, _Session(master="spark://cluster:7077"))
    result = dc.check_session()
    assert result.status == PASS
    assert result.detail["master"] == "spark://cluster:7077"


def test_session_warns_on_spark_connect(off_cluster, monkeypatch):
    _session(monkeypatch, _Session(master=None))
    result = dc.check_session()
    assert result.status == WARN
    assert "Spark Connect" in result.summary
    assert result.detail["access mode"].startswith("standard or serverless")


def test_session_fails_when_a_cluster_run_is_local(on_cluster, monkeypatch):
    """A local session on a cluster means the runtime's was never picked up."""
    _session(monkeypatch, _Session(master="local[*]"))
    result = dc.check_session()
    assert result.status == FAIL
    assert "local" in result.summary


def test_session_skips_without_one(off_cluster, monkeypatch):
    _session(monkeypatch, None)
    assert dc.check_session().status == SKIP


# ---------------------------------------------------------------------------
# pyspark — hazard: the `spark` extra shadows the runtime's client
# ---------------------------------------------------------------------------


def test_pyspark_matches_the_server(monkeypatch):
    _session(monkeypatch, _Session(version="4.2.0"))
    monkeypatch.setattr(dc, "_installed_version", lambda d: "4.2.0")
    assert dc.check_pyspark().status == PASS


def test_pyspark_tolerates_a_patch_difference(monkeypatch):
    """Databricks reports its own patch strings; only major.minor matters."""
    _session(monkeypatch, _Session(version="4.2.0"))
    monkeypatch.setattr(dc, "_installed_version", lambda d: "4.2.3")
    assert dc.check_pyspark().status == PASS


def test_pyspark_fails_when_the_extra_shadows_the_runtime(monkeypatch):
    _session(monkeypatch, _Session(version="4.2.0"))
    monkeypatch.setattr(dc, "_installed_version", lambda d: "4.1.2")
    result = dc.check_pyspark()
    assert result.status == FAIL
    assert "4.1.2" in result.summary and "4.2.0" in result.summary
    assert result.fix is not None and "spark" in result.fix


def test_pyspark_skips_when_absent(monkeypatch):
    monkeypatch.setattr(dc, "_installed_version", lambda d: None)
    assert dc.check_pyspark().status == SKIP


# ---------------------------------------------------------------------------
# packages — hazard: a failed library install the notebook attached through
# ---------------------------------------------------------------------------


def _pyproject(tmp_path, core: str, extras: str = "") -> "object":
    path = tmp_path / "pyproject.toml"
    path.write_text(
        f'[project]\nname = "x"\ndependencies = [{core}]\n'
        f"[project.optional-dependencies]\n{extras}",
        encoding="utf-8",
    )
    return path


def test_packages_passes_when_all_present(tmp_path, monkeypatch):
    path = _pyproject(tmp_path, '"alpha>=1.0", "beta"')
    monkeypatch.setattr(dc, "_installed_version", lambda d: "2.0")
    result = dc.check_packages(path)
    assert result.status == PASS
    assert result.detail["declared"] == 2


def test_packages_fails_on_a_missing_dependency(tmp_path, monkeypatch):
    path = _pyproject(tmp_path, '"alpha>=1.0", "beta"')
    monkeypatch.setattr(
        dc, "_installed_version", lambda d: None if d == "beta" else "2.0"
    )
    result = dc.check_packages(path)
    assert result.status == FAIL
    assert result.detail["missing"] == "beta"
    assert result.fix is not None and "Libraries tab" in result.fix


def test_packages_fails_below_the_declared_floor(tmp_path, monkeypatch):
    """The pydantic case: preinstalled, but older than pyproject requires."""
    pytest.importorskip("packaging")
    path = _pyproject(tmp_path, '"pydantic>=2.13.4"')
    monkeypatch.setattr(dc, "_installed_version", lambda d: "2.13.3")
    result = dc.check_packages(path)
    assert result.status == FAIL
    assert "2.13.3" in result.detail["below floor"]


def test_packages_reads_the_real_pyproject_by_default(monkeypatch):
    """find_pyproject walks up from the module, not from the working directory."""
    monkeypatch.chdir(sys.prefix)
    found = dc.find_pyproject()
    assert found is not None and found.name == "pyproject.toml"
    assert dc.check_packages().status == PASS


def test_packages_skips_without_a_checkout(monkeypatch):
    monkeypatch.setattr(dc, "find_pyproject", lambda: None)
    assert dc.check_packages().status == SKIP


# ---------------------------------------------------------------------------
# extras — hazard: delta-spark installed where Delta is built in
# ---------------------------------------------------------------------------


def test_extras_fails_on_delta_spark_on_a_cluster(on_cluster, tmp_path, monkeypatch):
    path = _pyproject(tmp_path, '"alpha"', 'spark = ["pyspark"]\n')
    monkeypatch.setattr(
        dc, "_installed_version", lambda d: "4.2.0" if d == "delta-spark" else None
    )
    result = dc.check_extras(path)
    assert result.status == FAIL
    assert "delta-spark" in result.summary


def test_extras_allows_delta_spark_off_a_cluster(off_cluster, tmp_path, monkeypatch):
    """The Docker stack installs it deliberately; only a cluster is the error."""
    path = _pyproject(tmp_path, '"alpha"')
    monkeypatch.setattr(dc, "_installed_version", lambda d: "4.2.0")
    assert dc.check_extras(path).status == PASS


def test_extras_reports_which_are_installed(off_cluster, tmp_path, monkeypatch):
    path = _pyproject(
        tmp_path, '"alpha"', 'sharepoint = ["msgraph-sdk", "msal"]\nvault = ["hvac"]\n'
    )
    monkeypatch.setattr(
        dc, "_installed_version", lambda d: "1.0" if d == "msal" else None
    )
    detail = dc.check_extras(path).detail
    assert detail["sharepoint"] == "partial (msal)"
    assert detail["vault"] == "not installed"


# ---------------------------------------------------------------------------
# The run and its reporting
# ---------------------------------------------------------------------------


def test_run_checks_is_offline_by_default(off_cluster, monkeypatch):
    _session(monkeypatch, None)
    names = [r.name for r in dc.run_checks()]
    assert names == [
        "runtime",
        "session",
        "pyspark",
        "packages",
        "extras",
        "libraries",
    ]
    assert "secrets" not in names


def test_run_checks_adds_the_secret_stage_on_request(off_cluster, monkeypatch):
    _session(monkeypatch, None)
    monkeypatch.setattr(
        dc, "check_secret_scope", lambda: dc.CheckResult("secrets", SKIP, "stubbed")
    )
    assert [r.name for r in dc.run_checks(check_secrets=True)][-1] == "secrets"


def test_main_exits_non_zero_when_a_stage_fails(on_cluster, monkeypatch, capsys):
    """A child process must fail the command, not merely mention it."""
    _session(monkeypatch, None)
    assert dc.main([]) == 1
    out = capsys.readouterr().out
    assert "Databricks runtime preflight" in out
    assert "[FAIL] runtime" in out


def test_main_report_is_ascii(on_cluster, monkeypatch, capsys):
    """The report is read over RDP as often as in a modern terminal."""
    _session(monkeypatch, _Session(master="local[*]"))
    dc.main(["--verbose"])
    out = capsys.readouterr().out
    assert out.isascii(), [c for c in out if not c.isascii()]


def test_main_json_is_machine_readable(off_cluster, monkeypatch, capsys):
    _session(monkeypatch, None)
    dc.main(["--json"])
    payload = json.loads(capsys.readouterr().out)
    assert {r["name"] for r in payload} == {
        "runtime",
        "session",
        "pyspark",
        "packages",
        "extras",
        "libraries",
    }


def test_active_session_survives_pyspark_being_absent(monkeypatch):
    """app_config must not require pyspark; the helper degrades to None."""
    monkeypatch.setitem(sys.modules, "pyspark", None)
    monkeypatch.setitem(sys.modules, "pyspark.sql", None)
    assert dc._active_session() is None


def test_active_session_survives_a_broken_pyspark(monkeypatch):
    broken = types.ModuleType("pyspark.sql")

    class _Boom:
        @staticmethod
        def getActiveSession():
            raise RuntimeError("py4j gateway is gone")

    broken.SparkSession = _Boom  # pyright: ignore[reportAttributeAccessIssue]
    monkeypatch.setitem(sys.modules, "pyspark.sql", broken)
    assert dc._active_session() is None


# ---------------------------------------------------------------------------
# The early warning main() emits, so a child process says so before it fails
# ---------------------------------------------------------------------------


def test_main_entry_point_warns_on_a_child_process(on_cluster, monkeypatch, caplog):
    import logging

    import main as cli

    _session(monkeypatch, None)
    with caplog.at_level(logging.WARNING):
        cli._warn_if_databricks_child_process()
    assert "child process" in caplog.text
    assert "%sh" in caplog.text


def test_main_entry_point_is_quiet_in_the_notebook(on_cluster, in_repl, monkeypatch, caplog):
    import logging

    import main as cli

    _session(monkeypatch, _Session())
    with caplog.at_level(logging.WARNING):
        cli._warn_if_databricks_child_process()
    assert caplog.text == ""


def test_main_entry_point_is_quiet_for_a_child_with_a_pat(on_cluster, monkeypatch, caplog):
    """It logs only on FAIL, and a PAT-carrying child is a WARN."""
    import logging

    import main as cli

    monkeypatch.setenv("DATABRICKS_TOKEN", "dapi-child")
    _session(monkeypatch, None)
    with caplog.at_level(logging.WARNING):
        cli._warn_if_databricks_child_process()
    assert caplog.text == ""


def test_main_entry_point_never_blocks_a_run(off_cluster, monkeypatch, caplog):
    """Advice must not be able to break the command it is advising about."""
    import logging

    import main as cli

    def boom() -> None:
        raise RuntimeError("preflight itself is broken")

    monkeypatch.setattr(dc, "check_runtime", boom)
    with caplog.at_level(logging.WARNING):
        cli._warn_if_databricks_child_process()
    assert caplog.text == ""


# ---------------------------------------------------------------------------
# libraries — hazard: the library set was resolved against another runtime
# ---------------------------------------------------------------------------


def _library_set(tmp_path, *runtimes: int):
    """A fake checkout carrying a tagged requirements file per runtime."""
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    directory = tmp_path / "databricks"
    directory.mkdir()
    for runtime in runtimes:
        (directory / f"requirements-dbr{runtime}.txt").write_text(
            f"# header\n# databricks-runtime: {runtime}\nhvac==2.4.0\n",
            encoding="utf-8",
        )
    return tmp_path / "pyproject.toml"


def test_libraries_skips_off_cluster(off_cluster, tmp_path):
    assert dc.check_libraries(_library_set(tmp_path, 19)).status == SKIP


def test_libraries_passes_when_a_set_matches(on_cluster, tmp_path):
    result = dc.check_libraries(_library_set(tmp_path, 18, 19))
    assert result.status == PASS
    assert "requirements-dbr19.txt" in result.summary


def test_libraries_warns_when_the_running_runtime_has_no_set(
    on_cluster, monkeypatch, tmp_path
):
    """The user's case: an 18 LTS cluster against a set resolved for DBR 19."""
    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "18.3")
    result = dc.check_libraries(_library_set(tmp_path, 19))
    assert result.status == WARN
    assert "DBR 18.3" in result.summary and "DBR 19" in result.summary
    assert result.fix is not None and "silently upgraded" in result.fix


def test_libraries_reads_the_runtime_from_the_file_not_the_filename(
    on_cluster, monkeypatch, tmp_path
):
    """The tag is the source of truth, so adding a set is adding files only."""
    pyproject = _library_set(tmp_path, 19)
    stray = tmp_path / "databricks" / "requirements-dbr21.txt"
    stray.write_text("# no tag here\nhvac==2.4.0\n", encoding="utf-8")

    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "21.0")

    result = dc.check_libraries(pyproject)
    assert result.status == WARN  # the untagged file does not count as a set


def test_libraries_skips_without_a_databricks_directory(on_cluster, tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    result = dc.check_libraries(tmp_path / "pyproject.toml")
    assert result.status == SKIP


def test_the_repo_ships_a_tagged_set_for_every_requirements_file():
    """Guards the real files: a set is a *pair*, and both carry the same tag."""
    directory = pathlib.Path(__file__).resolve().parents[1] / "databricks"
    requirements = sorted(directory.glob("requirements-dbr*.txt"))
    assert requirements, "no compute library sets found"
    for path in requirements:
        runtime = path.stem.removeprefix("requirements-dbr")
        constraints = directory / f"constraints-dbr{runtime}.txt"
        assert constraints.is_file(), f"{path.name} has no constraints sibling"
        for file in (path, constraints):
            match = dc._RUNTIME_TAG.search(file.read_text(encoding="utf-8"))
            assert match, f"{file.name} has no '# databricks-runtime:' tag"
            assert match.group(1) == runtime, f"{file.name} is tagged {match.group(1)}"


def test_the_library_sets_and_their_constraints_are_disjoint():
    """A package is either shipped by the runtime or installed over it, not both."""
    directory = pathlib.Path(__file__).resolve().parents[1] / "databricks"

    def _names(path):
        names = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.split("#")[0].split(";")[0].strip()
            if "==" in line:
                names.add(line.split("==")[0].strip().lower())
        return names

    for path in sorted(directory.glob("requirements-dbr*.txt")):
        runtime = path.stem.removeprefix("requirements-dbr")
        overlap = _names(path) & _names(directory / f"constraints-dbr{runtime}.txt")
        assert not overlap, f"DBR {runtime}: in both files: {sorted(overlap)}"


def test_session_skip_stops_pointing_at_a_passing_runtime_stage(
    on_cluster, in_repl, monkeypatch
):
    """The two used to be one branch, and decoupling them changed the message.

    "On a cluster with no session" meant "a child process" only while the
    runtime stage discriminated that way. In the notebook there is legitimately
    no session until something touches Spark, and sending a reader to a check
    that passes is the sort of dead end this rewrite exists to remove.
    """
    _session(monkeypatch, None)
    assert dc.check_runtime().status == PASS

    result = dc.check_session()

    assert result.status == SKIP
    assert "see the runtime stage" not in result.summary


def test_session_skip_still_points_at_the_runtime_stage_for_a_child(
    on_cluster, monkeypatch
):
    _session(monkeypatch, None)
    assert "see the runtime stage" in dc.check_session().summary


# ---------------------------------------------------------------------------
# The report names its own product
# ---------------------------------------------------------------------------


def test_render_names_databricks_not_sharepoint(off_cluster, monkeypatch):
    """The regression a cluster report surfaced.

    This module re-exported ``sharepoint_check.render``, whose defaults name
    SharePoint. ``main()`` overrode them at its call site, so the CLI was fine
    and the form the module docstring documents for notebooks -- ``from
    app_config.databricks_check import render`` -- printed "SharePoint
    preflight" and "PASSED: SharePoint is reachable and configured" over a
    Databricks report.
    """
    _session(monkeypatch, None)
    report = dc.render(dc.run_checks())

    assert report.splitlines()[0] == "Databricks runtime preflight"
    assert "SharePoint" not in report


def test_render_all_clear_line_names_the_runtime(off_cluster, monkeypatch):
    _session(monkeypatch, None)
    passing = [dc.CheckResult("runtime", PASS, "fine")]

    assert "the runtime environment" in dc.render(passing)
    assert "SharePoint is reachable" not in dc.render(passing)


def test_main_and_render_agree(on_cluster, in_repl, monkeypatch, capsys):
    """One source for the strings, so the CLI and the notebook cannot diverge."""
    _session(monkeypatch, _Session(master="spark://x:7077"))
    dc.main([])
    assert capsys.readouterr().out.splitlines()[0] == "Databricks runtime preflight"


# ---------------------------------------------------------------------------
# A check that cannot check must not pass
# ---------------------------------------------------------------------------


def _no_checkout(monkeypatch):
    """No pyproject.toml above the module: package folders copied somewhere."""
    monkeypatch.setattr(dc, "find_pyproject", lambda start=None: None)


def _no_distribution(monkeypatch):
    def _missing(_name):
        raise dc.importlib_metadata.PackageNotFoundError(_name)

    monkeypatch.setattr(dc.importlib_metadata, "metadata", _missing)


def _distribution(monkeypatch, text: str):
    from email import message_from_string

    monkeypatch.setattr(
        dc.importlib_metadata, "metadata", lambda _name: message_from_string(text)
    )


_WHEEL_METADATA = """\
Metadata-Version: 2.1
Name: sas-parser
Provides-Extra: vault
Provides-Extra: sharepoint
Requires-Dist: pydantic>=2.13.4
Requires-Dist: hvac>=2.0; extra == "vault"
Requires-Dist: msgraph-sdk>=1.5; extra == "sharepoint"
Requires-Dist: definitely-not-installed>=1.0; extra == "sharepoint"
"""


def test_extras_skips_rather_than_passing_when_nothing_declares_them(
    on_cluster, monkeypatch
):
    """The defect a cluster report surfaced: [ ok ] having opened nothing.

    A green check that did not check is worse than a red one, because it is
    believed. This is the same shape as the runtime stage's old discriminator.
    """
    _no_checkout(monkeypatch)
    _no_distribution(monkeypatch)

    result = dc.check_extras()

    assert result.status == SKIP
    assert result.status != PASS
    assert "cannot be checked" in result.summary
    assert result.detail["pyproject.toml"] == "not found above this module"
    assert result.fix is not None and "pyproject.toml" in result.fix


def test_packages_skips_on_the_same_condition(on_cluster, monkeypatch):
    _no_checkout(monkeypatch)
    _no_distribution(monkeypatch)

    result = dc.check_packages()

    assert result.status == SKIP
    assert "cannot be checked" in result.summary


def _installed(monkeypatch, *present: str):
    """Own which distributions exist, rather than inheriting the venv's.

    CI's test job installs neither `vault` nor `sharepoint` -- the in-memory
    paths need none of them -- so a test that read the real environment here
    would be asserting which extras this checkout happens to carry instead of
    whether the metadata fallback parses and resolves them.
    """
    # Above every declared floor, so "present" means present rather than
    # accidentally testing the version comparison as well.
    monkeypatch.setattr(
        dc, "_installed_version", lambda d: "99.0" if d in present else None
    )


def test_extras_falls_back_to_the_installed_distribution(on_cluster, monkeypatch):
    """A wheel carries the same requirement set in another spelling."""
    _no_checkout(monkeypatch)
    _distribution(monkeypatch, _WHEEL_METADATA)
    _installed(monkeypatch, "hvac", "msgraph-sdk")

    result = dc.check_extras()

    assert result.status == PASS
    assert "distribution" in result.detail["declared by"]
    # The extra == "..." markers were read back off Requires-Dist.
    assert result.detail["vault"] == "installed"
    # sharepoint declares two, one of which is absent.
    assert result.detail["sharepoint"] == "partial (msgraph-sdk)"


def test_packages_falls_back_to_the_installed_distribution(on_cluster, monkeypatch):
    _no_checkout(monkeypatch)
    _distribution(monkeypatch, _WHEEL_METADATA)
    _installed(monkeypatch, "pydantic")

    result = dc.check_packages()

    assert result.status == PASS
    assert result.detail["declared"] == 1  # only the unmarked Requires-Dist
    assert "distribution" in result.detail["declared by"]


def test_packages_fails_on_a_missing_core_dependency_from_metadata(
    on_cluster, monkeypatch
):
    """The fallback is a real check, not a formality: it can fail."""
    _no_checkout(monkeypatch)
    _distribution(monkeypatch, _WHEEL_METADATA)
    _installed(monkeypatch)  # nothing installed

    result = dc.check_packages()

    assert result.status == FAIL
    assert result.detail["missing"] == "pydantic"


def test_the_checkout_wins_over_the_distribution(on_cluster, monkeypatch, tmp_path):
    """Both present: the source tree is what the code was resolved against."""
    pyproject = _library_set(tmp_path, 19)
    pyproject.write_text(
        '[project]\ndependencies = ["hvac>=2.0"]\n', encoding="utf-8"
    )
    _distribution(monkeypatch, _WHEEL_METADATA)

    _, _, declared_by = dc.requirement_sources(pyproject)

    assert declared_by == str(pyproject)


def test_requirement_sources_is_none_when_neither_exists(monkeypatch):
    _no_checkout(monkeypatch)
    _no_distribution(monkeypatch)

    assert dc.requirement_sources() is None


# ---------------------------------------------------------------------------
# pyspark on a cluster, where its dist-info is not visible
# ---------------------------------------------------------------------------


def test_pyspark_passes_when_nothing_shadows_the_runtime(on_cluster, monkeypatch):
    """The stage was inert on the one platform it exists for.

    On DBR the runtime's pyspark has no dist-info in the notebook-scoped env,
    so the client lookup returns None -- and the stage reported "correct for an
    in-memory run" directly above a live Spark session. Absence of dist-info
    with a session running does not mean pyspark is absent; it means no pip
    install is shadowing the runtime's, which is the state this stage exists to
    confirm.
    """
    monkeypatch.setattr(dc, "_installed_version", lambda d: None)
    _session(monkeypatch, _Session(version="4.1.0"))

    result = dc.check_pyspark()

    assert result.status == PASS
    assert "shadowing" in result.summary and "4.1.0" in result.summary
    assert "in-memory" not in result.summary
    assert result.detail["server Spark"] == "4.1.0"


def test_pyspark_still_skips_for_a_real_in_memory_run(off_cluster, monkeypatch):
    """No pyspark and no session: the message was right for this case all along."""
    monkeypatch.setattr(dc, "_installed_version", lambda d: None)
    _session(monkeypatch, None)

    result = dc.check_pyspark()

    assert result.status == SKIP
    assert "correct for an in-memory run" in result.summary


# ---------------------------------------------------------------------------
# Where is this code actually running from
# ---------------------------------------------------------------------------


def test_runtime_reports_where_app_config_was_imported_from(on_cluster, monkeypatch):
    """`executable` names the interpreter, not the package.

    On a cluster those routinely come from different places, and which copy of
    app_config is running is the fact that explains what the packages, extras
    and libraries stages were able to read.
    """
    _session(monkeypatch, None)
    location = dc.check_runtime().detail["app_config"]

    assert location.endswith("app_config")
    assert pathlib.Path(location).is_dir()
