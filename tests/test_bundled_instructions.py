"""
Conformance tests for the instruction set shipped at
``prompt_builder/instructions/``.

Content, not machinery: the parsing rules themselves are covered by
``test_user_instructions.py``. What these assert is that the *bundled* files
stay wired to the pipeline that consumes them — every directive key reachable,
no documentation ingested as rules, and the unconditional tier small enough to
leave room for everything else.

Each check corresponds to a defect the set actually shipped with:

* ``instructions/README.md`` parsed as three always-on sections and was
  prompted with every item, for every target language;
* ``[when: macro_function:sysfunc]`` named a construct kind the pipeline never
  emits, so the rule could never fire;
* the unconditional tier reached 903 of the 1500-word default budget.

Fully offline: reads the shipped markdown, imports no LLM client, and touches
no reference PDF.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

from chunker.keywords import (
    _SAS_CALL_ROUTINES,
    _SAS_FUNCTIONS,
    SAS_DATA_STEP_STATEMENT_TOKENS,
    SAS_FUNCTION_CATEGORIES,
)
from chunker.models import SasChunkKind
from pipeline.prompting import _META_FLAG_ATTRS
from prompt_builder.user_instructions import (
    SCOPE_ALWAYS,
    SCOPE_TOPIC,
    UserInstructionSet,
    kinds_of,
    langs_of,
    metas_of,
    normalize_language,
    scope_of,
)

INSTRUCTIONS_DIR = pathlib.Path(__file__).resolve().parents[1] / (
    "prompt_builder/instructions"
)

# The construct-key kinds ``_constructs_for_item`` can emit for an item. A
# directive naming anything else is dead on arrival — it will never match.
EMITTABLE_KINDS = frozenset(
    {"proc", "function", "call_routine", "component_object",
     "global_statement", "category", "statement"}
)
# ``macro_statement`` is emitted, but only ever for these two names.
EMITTABLE_MACRO_STATEMENTS = frozenset({"goto", "abort"})

# Ceiling on rules injected into *every* item regardless of content. Guidance
# that applies to everything is guidance the budget pays for every time, so it
# stays small; anything larger belongs behind a scope. Counted in tokens, the
# currency the budget is spent in — this set runs ~1.7 tokens per word, so a
# word ceiling would have been reading ~40% low.
MAX_UNCONDITIONAL_TOKENS = 750

# Ceiling on `[topic]` sections. They are not construct-gated — they are ranked
# against the reference corpus — so in the *shipped* configuration, where the
# pipeline builds a corpus-less builder from these instructions alone, a topical
# section has nothing to lose to and is retrieved for every item. It is charged
# like an unconditional rule while looking scoped, which is exactly why it needs
# a ceiling of its own rather than being trusted to the ranker.
MAX_TOPICAL_TOKENS = 320


@pytest.fixture(scope="module")
def bundled() -> UserInstructionSet:
    return UserInstructionSet.from_dir(str(INSTRUCTIONS_DIR))


def _unconditional(chunk) -> bool:
    """Fires for every item: primary scope 'always' and no modifier gate."""
    return (
        scope_of(chunk) == SCOPE_ALWAYS
        and not kinds_of(chunk)
        and not metas_of(chunk)
    )


def _tokens(chunks) -> int:
    return sum(c.token_count for c in chunks)


def test_bundled_set_is_non_empty(bundled):
    assert len(bundled.chunks) > 20, "the shipped set failed to load"


def test_bundled_set_parses_without_diagnostics(bundled):
    """No unknown directives, malformed keys, or empty sections."""
    assert [
        (d.code, d.doc_id, d.message) for d in bundled.diagnostics
    ] == []


def test_no_documentation_is_ingested_as_instructions(bundled):
    """A README describes the rules; it is not one of them.

    ``from_dir`` skips ``README*.md`` and ``_``-prefixed filenames. Ingested,
    such a file parses as always-on sections and is prompted with every item.
    ``doc_id`` is the file's relative path, slugged, so it names the source.
    """
    offenders = sorted(
        {c.doc_id for c in bundled.chunks if "readme" in c.doc_id.lower()}
    )
    assert offenders == [], f"documentation ingested as instructions: {offenders}"


def test_category_map_names_exist_in_the_keyword_catalogues():
    """The other end of the ``[category:]`` contract.

    A category is only reachable if some recognised function maps to it, so a
    typo in the map silently creates a family nothing can ever match — the
    mirror image of a directive naming a category that does not exist.
    """
    known = _SAS_FUNCTIONS | _SAS_CALL_ROUTINES
    unknown = sorted(n for n in SAS_FUNCTION_CATEGORIES if n not in known)
    assert unknown == [], (
        f"SAS_FUNCTION_CATEGORIES names absent from _SAS_FUNCTIONS and "
        f"_SAS_CALL_ROUTINES, so nothing can ever match them: {unknown}"
    )


def test_every_category_used_by_an_instruction_has_members(bundled):
    """A `[category: x]` rule is dead unless some function maps to ``x``."""
    mapped = set(SAS_FUNCTION_CATEGORIES.values())
    used = {
        key.name
        for c in bundled.chunks
        for key in c.construct_keys or ()
        if key.kind == "category"
    }
    assert used <= mapped, f"categories with no members: {sorted(used - mapped)}"


def test_every_directive_key_is_reachable(bundled):
    """Every ``[when:]``/``[category:]`` key can actually be emitted.

    A key naming a construct kind the pipeline never produces, or a function
    absent from the chunker's catalogues, is a rule that silently never fires
    — the failure mode this whole module exists to prevent.
    """
    unreachable: list[str] = []
    for chunk in bundled.chunks:
        for key in chunk.construct_keys or ():
            if key.kind == "macro_statement":
                ok = key.name in EMITTABLE_MACRO_STATEMENTS
            elif key.kind not in EMITTABLE_KINDS:
                ok = False
            elif key.kind == "function":
                ok = key.name in _SAS_FUNCTIONS
            elif key.kind == "call_routine":
                ok = key.name in _SAS_CALL_ROUTINES
            elif key.kind == "category":
                ok = key.name in set(SAS_FUNCTION_CATEGORIES.values())
            elif key.kind == "statement":
                ok = key.name in SAS_DATA_STEP_STATEMENT_TOKENS
            else:
                ok = True
            if not ok:
                unreachable.append(f"{key} in '{chunk.section_path}'")
    assert unreachable == [], f"directive keys that can never match: {unreachable}"


def test_every_kind_directive_names_a_real_chunk_kind(bundled):
    valid = {k.value for k in SasChunkKind}
    bad = [
        f"{kind} in '{c.section_path}'"
        for c in bundled.chunks
        for kind in kinds_of(c)
        if kind not in valid
    ]
    assert bad == [], f"[kind: ...] tokens matching no SasChunkKind: {bad}"


def test_every_meta_directive_names_a_real_flag(bundled):
    valid = {token for token, _ in _META_FLAG_ATTRS}
    bad = [
        f"{meta} in '{c.section_path}'"
        for c in bundled.chunks
        for meta in metas_of(c)
        if meta not in valid
    ]
    assert bad == [], f"[meta: ...] tokens matching no pipeline flag: {bad}"


def test_language_scoped_sections_name_a_known_target(bundled):
    from target_language import KNOWN_TARGETS

    known = {t.key for t in KNOWN_TARGETS} | {
        normalize_language(a) for t in KNOWN_TARGETS for a in t.aliases
    }
    bad = [
        f"{lang} in '{c.section_path}'"
        for c in bundled.chunks
        for lang in langs_of(c)
        if lang not in known
    ]
    assert bad == [], f"[lang: ...] tokens matching no known target: {bad}"


def test_unconditional_tier_stays_within_budget(bundled):
    """Always-on rules are charged to every item; keep the tier small."""
    unconditional = [c for c in bundled.chunks if _unconditional(c)]
    total = _tokens(unconditional)
    breakdown = sorted(
        ((c.token_count, c.section_path) for c in unconditional),
        reverse=True,
    )
    assert total <= MAX_UNCONDITIONAL_TOKENS, (
        f"unconditional instructions total {total} tokens "
        f"(ceiling {MAX_UNCONDITIONAL_TOKENS}); scope one of these behind a "
        f"[when:]/[category:]/[kind:] directive: {breakdown}"
    )


def test_topical_tier_stays_within_budget(bundled):
    """`[topic]` is not a cheaper way to be always-on.

    A topical section is ranked, not construct-gated. The shipped pipeline
    builds a corpus-less builder from these instructions alone, so there is
    nothing for it to be out-ranked by and it lands on every item — charged
    like an unconditional rule while reading as a scoped one. Ceiling it
    separately, because the unconditional check cannot see it.
    """
    topical = [c for c in bundled.chunks if scope_of(c) == SCOPE_TOPIC]
    total = _tokens(topical)
    breakdown = sorted(
        ((c.token_count, c.section_path) for c in topical), reverse=True
    )
    assert total <= MAX_TOPICAL_TOKENS, (
        f"[topic] instructions total {total} tokens (ceiling "
        f"{MAX_TOPICAL_TOKENS}). With no reference corpus these are retrieved "
        f"for every item, so they cost what an always-on rule costs: {breakdown}"
    )


def test_guidance_charged_to_every_item_is_bounded(bundled):
    """The number that actually matters: what an item pays before it has
    matched a single construct of its own."""
    always_charged = [
        c
        for c in bundled.chunks
        if _unconditional(c) or scope_of(c) == SCOPE_TOPIC
    ]
    total = _tokens(always_charged)
    ceiling = MAX_UNCONDITIONAL_TOKENS + MAX_TOPICAL_TOKENS
    assert total <= ceiling, (
        f"{total} tokens reach every item regardless of its constructs "
        f"(ceiling {ceiling})"
    )


def test_sparksql_slice_is_scoped_to_sparksql(bundled):
    """Files under ``sparksql/`` must not leak into another target's run.

    ``from_dir`` derives the scope from the directory name, so this fails only
    if a section overrode it with an explicit, wrong ``[lang: ...]``.
    """
    unscoped = [
        c.section_path
        for c in bundled.chunks
        if c.doc_id.startswith("sparksql_") and "sparksql" not in langs_of(c)
    ]
    assert unscoped == [], f"sparksql sections not scoped to SparkSQL: {unscoped}"


def test_pyspark_slice_is_scoped_to_pyspark(bundled):
    """Files under pyspark/ must not leak into another target run."""
    unscoped = [
        c.section_path
        for c in bundled.chunks
        if c.doc_id.startswith("pyspark_") and "pyspark" not in langs_of(c)
    ]
    assert unscoped == [], f"pyspark sections not scoped to PySpark: {unscoped}"


def test_proc_cas_guidance_exists_for_both_targets(bundled):
    cas = [
        chunk
        for chunk in bundled.chunks
        if any(
            key.kind == "proc" and key.name == "cas"
            for key in chunk.construct_keys
        )
    ]
    assert {lang for chunk in cas for lang in langs_of(chunk)} == {
        "pyspark",
        "sparksql",
    }
    assert all("PROC_STEP" in kinds_of(chunk) for chunk in cas)
