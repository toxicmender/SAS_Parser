"""The Docker stack's version pins must match what ``uv.lock`` resolves.

``docker/app.Dockerfile`` installs pyspark with ``uv sync --locked``, so the
*driver* version is whatever ``uv.lock`` pins. ``docker/spark.Dockerfile``
takes its version from the ``PYSPARK_VERSION`` build argument, so the
*cluster* version is whatever ``docker-compose.yml`` defaults that to. A
driver and a cluster on different Spark versions fail at handshake, and the
two are coupled by nothing but the comment above the compose entry — this
test is what makes that coupling real.

When it fails, a dependency bump moved ``uv.lock`` without moving compose (or
the reverse). Fix the pin, do not relax the test: the alternative is a stack
that builds cleanly and then hangs the first time a job reaches an executor.

The same reasoning applies to Python: both images take ``PYTHON_VERSION``,
and ``requires-python`` in ``pyproject.toml`` bounds what the lock is valid
for, so a base image below that floor is equally a build-time-clean,
runtime-broken combination.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = REPO_ROOT / "uv.lock"
COMPOSE_PATH = REPO_ROOT / "docker-compose.yml"
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
CI_DOCKERFILE_PATH = REPO_ROOT / "docker" / "ci.spark-delta.Dockerfile"
CI_DOCKERIGNORE_PATH = REPO_ROOT / "docker" / "ci.spark-delta.Dockerfile.dockerignore"
CI_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "ci.yml"

# Matches compose's shell-style substitution defaults: ${NAME:-default}. The
# file is read as text rather than parsed as YAML because the default lives
# *inside* the scalar — a YAML parse hands back the literal "${...}" string
# and this expression would still be needed to get at it.
_DEFAULT_RE = r"\$\{%s:-([^}]*)\}"


def _locked_version(package: str) -> str:
    """The version ``uv.lock`` resolves *package* to."""
    lock = tomllib.loads(LOCK_PATH.read_text(encoding="utf-8"))
    for entry in lock.get("package", []):
        if entry.get("name") == package:
            return entry["version"]
    pytest.fail(f"{package!r} is not in uv.lock")


def _compose_defaults(variable: str) -> list[str]:
    """Every ``${variable:-default}`` default in the compose file."""
    text = COMPOSE_PATH.read_text(encoding="utf-8")
    found = re.findall(_DEFAULT_RE % re.escape(variable), text)
    assert found, f"no ${{{variable}:-...}} default found in docker-compose.yml"
    return found


def test_pyspark_pin_matches_lock() -> None:
    """The cluster images build the same pyspark the app image installs."""
    locked = _locked_version("pyspark")
    for default in _compose_defaults("PYSPARK_VERSION"):
        assert default == locked, (
            f"docker-compose.yml defaults PYSPARK_VERSION to {default}, but "
            f"uv.lock resolves pyspark to {locked}. The app image installs "
            f"the locked version and the spark images build the compose one, "
            f"so they would not agree at handshake. Update the compose "
            f"default (every occurrence) to {locked}."
        )


def test_pyspark_pin_is_consistent_across_compose() -> None:
    """All three occurrences agree — the build arg and both image tags.

    The image tag is what a rebuild overwrites, so a tag left on the old
    version silently serves a stale image to the next `docker compose up`.
    """
    defaults = _compose_defaults("PYSPARK_VERSION")
    assert len(set(defaults)) == 1, (
        f"docker-compose.yml gives PYSPARK_VERSION more than one default: "
        f"{sorted(set(defaults))}"
    )


def test_dockerfile_extras_exist_in_pyproject() -> None:
    """Every ``--extra`` the images install is a real extra.

    Not a style check. ``uv sync`` **exits 2** on an extra that is not in
    ``optional-dependencies`` — it does not warn and carry on — so one stale
    flag stops the image building at all. ``--extra sql`` outlived sqlglot
    becoming a core dependency and did exactly that, silently, because nothing
    in the suite builds an image.
    """
    pyproject = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
    declared = set(pyproject["project"].get("optional-dependencies", {}))

    for dockerfile in sorted(Path(REPO_ROOT / "docker").glob("*.Dockerfile")):
        used = set(re.findall(r"--extra\s+([A-Za-z0-9_.-]+)", dockerfile.read_text()))
        unknown = sorted(used - declared)
        assert not unknown, (
            f"{dockerfile.name} installs undeclared extra(s) {unknown}; "
            f"pyproject.toml declares {sorted(declared)}. uv sync exits 2 on "
            f"these, so the image cannot build."
        )


def test_ci_extras_exist_in_pyproject() -> None:
    """The same check for the workflow, which shares the failure mode."""
    workflow = CI_WORKFLOW_PATH
    if not workflow.exists():  # pragma: no cover - defensive
        pytest.skip("no ci.yml")
    pyproject = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
    declared = set(pyproject["project"].get("optional-dependencies", {}))

    used = set(re.findall(r"--extra\s+([A-Za-z0-9_.-]+)", workflow.read_text()))
    unknown = sorted(used - declared)
    assert not unknown, (
        f"ci.yml installs undeclared extra(s) {unknown}; pyproject.toml "
        f"declares {sorted(declared)}."
    )


def test_spark_delta_ci_uses_a_locked_repository_image() -> None:
    """The mandatory live runtime job builds our pinned Docker definition."""
    workflow = CI_WORKFLOW_PATH.read_text(encoding="utf-8")
    dockerfile = CI_DOCKERFILE_PATH.read_text(encoding="utf-8")
    dockerignore = CI_DOCKERIGNORE_PATH.read_text(encoding="utf-8")

    assert "docker/build-push-action@v6" in workflow
    assert "file: docker/ci.spark-delta.Dockerfile" in workflow
    assert "REQUIRE_DELTA_TESTS=1" in workflow
    assert "tests/test_spark_delta_runtime.py" in workflow
    assert "uv sync --locked --no-install-project --extra dev --extra spark" in dockerfile
    assert "uv pip install --python /opt/venv --no-deps --editable ." in dockerfile
    assert re.search(r"ARG DELTA_SPARK_VERSION=\d+\.\d+\.\d+", dockerfile)
    assert not re.search(
        r"^COPY --from=ghcr\.io/astral-sh/uv:latest\b", dockerfile, re.MULTILINE
    )
    assert "**" in dockerignore.splitlines()
    assert "!src/**" in dockerignore
    assert "!tests/**" in dockerignore
    assert "!.git/**" not in dockerignore
    assert "!.claude/**" not in dockerignore


def test_compose_python_satisfies_requires_python() -> None:
    """The images' Python is not below ``pyproject.toml``'s floor."""
    pyproject = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
    requires = pyproject["project"]["requires-python"]
    floor = re.search(r">=\s*(\d+)\.(\d+)", requires)
    assert floor, f"cannot read a lower bound from requires-python {requires!r}"
    minimum = (int(floor.group(1)), int(floor.group(2)))

    for default in _compose_defaults("PYTHON_VERSION"):
        parts = default.split(".")
        version = (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)
        assert version >= minimum, (
            f"docker-compose.yml defaults PYTHON_VERSION to {default}, below "
            f"pyproject.toml's requires-python {requires!r}"
        )
