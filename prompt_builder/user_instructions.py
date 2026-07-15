"""Operator-supplied instruction parsing: a plain ``str`` -> scoped chunks.

See prompt_builder/README.md.

An operator writes project rules as ordinary markdown; each ``## heading``
opens one instruction, and an optional directive in the heading sets its
scope:

* ``## Output format`` — **always** (the default): injected into every item.
* ``## [when: proc:sql, component_object:hash] SQL rules`` — **conditional**:
  injected only when the item's constructs intersect the listed keys. The
  ``kind:name`` syntax is exactly ``str(ConstructKey)``.
* ``## [topic] Partitioning guidance`` — **topical**: indexed for retrieval
  and surfaced by ranking, like a reference chunk.
* ``## [example: proc:sql] SQL join`` — **example** (few-shot): a worked
  SAS -> target pair, injected when the item's constructs intersect the
  listed keys and rendered in its own ``## Worked examples`` block. A bare
  ``[example]`` (no keys) is unconditional — shown to every item.

A heading may carry several leading bracket groups, combined as AND across
clauses. ``## [lang: sparksql, pyspark] ...`` restricts a section to the
named output language(s); it is a modifier, so it stacks with a primary
scope — ``## [when: proc:sql] [lang: sparksql] SQL rules`` fires only for a
SparkSQL run whose item uses PROC SQL. A section with no ``[lang: ...]`` is
language-agnostic. Matching is case/space/underscore-insensitive
(:func:`normalize_language`), and the run's ``output_language`` is applied
at selection time.

A heading-less string is a single always-on instruction; non-empty text
before the first heading becomes an always-on "General" instruction. Parsing
never raises on malformed input — it emits ``InstructionDiagnostic``s
(``UNKNOWN_DIRECTIVE``, ``INVALID_CONSTRUCT_KEY``, ``EMPTY_INSTRUCTION``) and
degrades toward *over*-inclusion (an unparseable scope becomes always-on),
because an operator rule silently vanishing is the worst failure mode here.

Instructions become ordinary :class:`InstructionChunk`s with
``role=DocRole.USER_INSTRUCTION`` and their scope recorded as a
``scope:<name>`` tag, so selector integration, budget filling, and the
LangChain ``Document`` round-trip all come for free. Page numbers are 0 —
user instructions have no pagination.

Logger name: ``prompt_builder.user_instructions``.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Iterable

import app_config
from pydantic import BaseModel, Field

from .models import ConstructKey, DocRole, InstructionChunk, InstructionDiagnostic

logger = logging.getLogger(__name__)

SCOPE_ALWAYS = "always"
SCOPE_WHEN = "when"
SCOPE_TOPIC = "topic"
SCOPE_EXAMPLE = "example"

# A heading carries zero or more leading ``[...]`` groups; each is one scope
# clause (``when:``/``topic``/``example``/``lang:``). Groups combine as AND
# across clauses: a single primary scope (when/topic/example, else always)
# coexists with orthogonal ``lang:`` filters — e.g.
# ``## [when: proc:sql] [lang: sparksql] SQL rules``.
_HEADING_RE = re.compile(r"^\s{0,3}##+\s*(?P<heading>.*\S)\s*$", re.MULTILINE)
_GROUP_RE = re.compile(r"^\[(?P<body>[^\]]*)\]")
_KEY_RE = re.compile(r"^[a-z_]\w*:\S+$")

_SCOPE_TAG_PREFIX = "scope:"
_LANG_TAG_PREFIX = "lang:"

_PREAMBLE_TITLE = "General"


def normalize_language(name: str) -> str:
    """Fold an output-language name to a comparison key.

    Case-, space-, hyphen-, and underscore-insensitive, so ``"SparkSQL"``,
    ``"Spark SQL"``, and ``"spark_sql"`` all match the same ``[lang: ...]``
    directive token.
    """
    return re.sub(r"[\s_-]+", "", name.lower())


def _scope_tag(scope: str) -> str:
    return f"{_SCOPE_TAG_PREFIX}{scope}"


def _lang_tag(language: str) -> str:
    return f"{_LANG_TAG_PREFIX}{language}"


def scope_of(chunk: InstructionChunk) -> str:
    """The parsed scope of a user-instruction chunk (``always`` fallback)."""
    for tag in chunk.tags:
        if tag.startswith(_SCOPE_TAG_PREFIX):
            return tag.removeprefix(_SCOPE_TAG_PREFIX)
    return SCOPE_ALWAYS


def langs_of(chunk: InstructionChunk) -> list[str]:
    """The normalized output languages a chunk is scoped to.

    Empty means language-agnostic — the chunk applies to every target. A
    non-empty list restricts the chunk to those languages (matched at
    selection time against the run's ``output_language``).
    """
    return [
        tag.removeprefix(_LANG_TAG_PREFIX)
        for tag in chunk.tags
        if tag.startswith(_LANG_TAG_PREFIX)
    ]


class UserInstructionSet(BaseModel):
    """
    Parsed operator instructions: ordered chunks, parse diagnostics, and a
    content fingerprint.

    Build via :meth:`from_text` or :meth:`from_file`; the plain constructor is
    for deserialisation. The fingerprint identifies the *source text* (not the
    parse), so eval run records can tell whether two runs used the same
    instructions — see ``validation/``.
    """

    chunks: list[InstructionChunk] = Field(default_factory=list)
    diagnostics: list[InstructionDiagnostic] = Field(default_factory=list)
    fingerprint: str = ""
    source: str | None = None  # file path when built via from_file

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        doc_id: str = "user",
        source: str | None = None,
        default_langs: Iterable[str] = (),
    ) -> "UserInstructionSet":
        """Parse *text* (markdown-ish, see module docstring) into a set.

        *default_langs* scopes every section without its own ``[lang: ...]``
        directive to those languages — used by :meth:`from_dir` so a file's
        location can name the target language without repeating it per
        section. Explicit ``[lang: ...]`` on a section overrides the default.
        """
        fingerprint = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        fallback_langs = [normalize_language(l) for l in default_langs if l]
        chunks: list[InstructionChunk] = []
        diagnostics: list[InstructionDiagnostic] = []

        for title, body in _split_sections(text):
            if not body.strip():
                diagnostics.append(
                    InstructionDiagnostic(
                        code="EMPTY_INSTRUCTION",
                        message=f"instruction '{title}' has no body; skipped",
                        doc_id=doc_id,
                    )
                )
                continue
            scope, clean_title, keys, langs = _parse_heading(
                title, doc_id, diagnostics
            )
            section_langs = langs or fallback_langs
            chunks.append(
                InstructionChunk(
                    chunk_id=f"{doc_id}::c{len(chunks):04d}",
                    doc_id=doc_id,
                    section_path=clean_title,
                    text=f"{clean_title}\n\n{body.strip()}",
                    page_start=0,
                    page_end=0,
                    role=DocRole.USER_INSTRUCTION,
                    construct_keys=keys,
                    tags=[_scope_tag(scope), *(_lang_tag(l) for l in section_langs)],
                )
            )

        logger.info(
            f"UserInstructionSet.from_text: {len(chunks)} instruction(s)  "
            f"always={sum(1 for c in chunks if scope_of(c) == SCOPE_ALWAYS)}  "
            f"when={sum(1 for c in chunks if scope_of(c) == SCOPE_WHEN)}  "
            f"topic={sum(1 for c in chunks if scope_of(c) == SCOPE_TOPIC)}  "
            f"example={sum(1 for c in chunks if scope_of(c) == SCOPE_EXAMPLE)}  "
            f"diagnostics={len(diagnostics)}  fingerprint={fingerprint}"
        )
        return cls(
            chunks=chunks,
            diagnostics=diagnostics,
            fingerprint=fingerprint,
            source=source,
        )

    @classmethod
    def from_file(cls, path: str, *, doc_id: str = "user") -> "UserInstructionSet":
        """Read *path* (UTF-8) and parse it via :meth:`from_text`."""
        text = Path(path).read_text(encoding="utf-8")
        return cls.from_text(text, doc_id=doc_id, source=str(path))

    @classmethod
    def from_dir(cls, directory: str) -> "UserInstructionSet":
        """Merge every ``*.md`` file under *directory* into one set.

        Files are read in sorted-relative-path order, so the merged
        fingerprint is deterministic. A file's **first path component**, when
        nested, names the target output language: sections in
        ``<dir>/sparksql/joins.md`` are scoped ``[lang: sparksql]`` unless
        they set their own ``[lang: ...]``. Files directly under *directory*,
        or under a subdirectory whose name starts with ``_`` (e.g.
        ``_common/``), are language-agnostic. Selection filters by the run's
        ``output_language``, so one directory can hold guidance for several
        targets side by side.

        A missing directory yields an empty set (mirrors :meth:`from_config`'s
        degradation): losing an instructions directory should not halt a run.
        """
        base = Path(directory)
        if not base.is_dir():
            logger.warning(
                f"UserInstructionSet.from_dir: '{directory}' is not a "
                f"directory; returning an empty instruction set"
            )
            return cls()

        chunks: list[InstructionChunk] = []
        diagnostics: list[InstructionDiagnostic] = []
        parts: list[str] = []
        for path in sorted(base.rglob("*.md"), key=lambda p: p.as_posix()):
            rel = path.relative_to(base)
            language = (
                rel.parts[0]
                if len(rel.parts) > 1 and not rel.parts[0].startswith("_")
                else None
            )
            # Per-file doc_id keeps chunk_ids unique across the merged set.
            doc_id = re.sub(r"\W+", "_", rel.with_suffix("").as_posix()).strip("_")
            text = path.read_text(encoding="utf-8")
            sub = cls.from_text(
                text,
                doc_id=doc_id or "user",
                source=str(path),
                default_langs=(language,) if language else (),
            )
            chunks.extend(sub.chunks)
            diagnostics.extend(sub.diagnostics)
            parts.append(f"{rel.as_posix()}\n{text}")

        fingerprint = hashlib.sha256(
            "\0".join(parts).encode("utf-8")
        ).hexdigest()[:16]
        logger.info(
            f"UserInstructionSet.from_dir: '{directory}' -> {len(chunks)} "
            f"instruction(s) from {len(parts)} file(s)  "
            f"diagnostics={len(diagnostics)}  fingerprint={fingerprint}"
        )
        return cls(
            chunks=chunks,
            diagnostics=diagnostics,
            fingerprint=fingerprint,
            source=str(base),
        )

    @classmethod
    def from_config(cls) -> "UserInstructionSet | None":
        """
        The standing instructions named by config.json, or ``None`` when
        unconfigured. ``user_instructions.dir`` (a directory of markdown
        files, merged via :meth:`from_dir`) takes precedence over
        ``user_instructions.path`` (a single file). A configured-but-missing
        path warns and returns ``None`` rather than raising — a deleted
        instructions source should not stop a run.
        """
        directory = app_config.get_value("user_instructions", "dir")
        if directory is not None:
            if not Path(directory).is_dir():
                logger.warning(
                    f"UserInstructionSet.from_config: configured instructions "
                    f"directory '{directory}' not found; continuing without "
                    f"user instructions"
                )
                return None
            logger.info(f"UserInstructionSet.from_config: loading dir '{directory}'")
            return cls.from_dir(str(directory))

        path = app_config.get_value("user_instructions", "path")
        if path is None:
            return None
        if not Path(path).is_file():
            logger.warning(
                f"UserInstructionSet.from_config: configured instructions "
                f"file '{path}' not found; continuing without user instructions"
            )
            return None
        logger.info(f"UserInstructionSet.from_config: loading '{path}'")
        return cls.from_file(str(path))

    # ------------------------------------------------------------------
    # Scope views
    # ------------------------------------------------------------------

    @property
    def always_chunks(self) -> list[InstructionChunk]:
        return [c for c in self.chunks if scope_of(c) == SCOPE_ALWAYS]

    @property
    def conditional_chunks(self) -> list[InstructionChunk]:
        return [c for c in self.chunks if scope_of(c) == SCOPE_WHEN]

    @property
    def topical_chunks(self) -> list[InstructionChunk]:
        return [c for c in self.chunks if scope_of(c) == SCOPE_TOPIC]

    @property
    def example_chunks(self) -> list[InstructionChunk]:
        return [c for c in self.chunks if scope_of(c) == SCOPE_EXAMPLE]

    def __len__(self) -> int:
        return len(self.chunks)

    def __str__(self) -> str:
        return (
            f"UserInstructionSet({len(self.chunks)} instruction(s), "
            f"fingerprint={self.fingerprint or '<empty>'})"
        )


# ---------------------------------------------------------------------------
# Parsing internals
# ---------------------------------------------------------------------------


def _split_sections(text: str) -> list[tuple[str, str]]:
    """
    ``(heading, body)`` pairs in document order. Heading-less text yields one
    pair titled with the preamble title; text before the first heading becomes
    its own pair.
    """
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [(_PREAMBLE_TITLE, text)] if text.strip() else []

    sections: list[tuple[str, str]] = []
    preamble = text[: matches[0].start()]
    if preamble.strip():
        sections.append((_PREAMBLE_TITLE, preamble))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((match.group("heading"), text[match.end() : end]))
    return sections


def _parse_heading(
    heading: str,
    doc_id: str,
    diagnostics: list[InstructionDiagnostic],
) -> tuple[str, str, list[ConstructKey], list[str]]:
    """``(scope, clean_title, construct_keys, langs)`` for one section heading.

    Consumes every leading ``[...]`` group. One primary scope (when/topic/
    example, else always) is combined with any orthogonal ``[lang: ...]``
    filters. When several primary-scope groups appear the last one wins.
    """
    text = heading.strip()
    groups: list[str] = []
    while True:
        match = _GROUP_RE.match(text)
        if match is None:
            break
        groups.append(match.group("body").strip())
        text = text[match.end() :].lstrip()
    title = text.strip()

    if not groups:
        return SCOPE_ALWAYS, title, [], []

    scope = SCOPE_ALWAYS
    keys: list[ConstructKey] = []
    langs: list[str] = []
    for body in groups:
        g_scope, g_keys, g_langs = _classify_group(
            body, heading, doc_id, diagnostics
        )
        if g_scope is not None:
            scope = g_scope
            if g_keys:
                keys = g_keys
        for language in g_langs:
            if language not in langs:
                langs.append(language)

    # A directive-only heading (no trailing title) falls back to the last
    # directive body as its label, matching the pre-stacking behaviour.
    if not title:
        title = groups[-1] or _PREAMBLE_TITLE
    return scope, title, keys, langs


def _classify_group(
    body: str,
    heading: str,
    doc_id: str,
    diagnostics: list[InstructionDiagnostic],
) -> tuple[str | None, list[ConstructKey], list[str]]:
    """One bracket group -> ``(primary_scope | None, keys, langs)``.

    ``lang:`` groups return ``None`` as the scope (a modifier, not a primary
    scope); every other recognised group returns its scope. Unknown groups
    degrade to always-on with a diagnostic.
    """
    lowered = body.lower()

    if lowered == SCOPE_TOPIC:
        return SCOPE_TOPIC, [], []

    if lowered == SCOPE_EXAMPLE:
        # Bare [example]: an unconditional few-shot example, shown to every item.
        return SCOPE_EXAMPLE, [], []

    if lowered.startswith("lang:"):
        return None, [], _parse_lang_tokens(body[5:])

    if lowered.startswith("example:"):
        keys = _parse_when_keys(body[8:], heading, doc_id, diagnostics)
        if keys:
            return SCOPE_EXAMPLE, keys, []
        # No usable key: keep it an example (not an always-on rule, which
        # would pollute the rules block) but drop the condition — shown to
        # every item rather than silently vanishing.
        diagnostics.append(
            InstructionDiagnostic(
                code="INVALID_CONSTRUCT_KEY",
                message=f"example-directive in heading '{heading.strip()}' "
                f"lists no valid construct keys; treating the example as "
                f"unconditional",
                doc_id=doc_id,
            )
        )
        return SCOPE_EXAMPLE, [], []

    if lowered.startswith("when:"):
        keys = _parse_when_keys(body[5:], heading, doc_id, diagnostics)
        if keys:
            return SCOPE_WHEN, keys, []
        # No usable key: over-include rather than silently drop the rule.
        diagnostics.append(
            InstructionDiagnostic(
                code="INVALID_CONSTRUCT_KEY",
                message=f"when-directive in heading '{heading.strip()}' lists "
                f"no valid construct keys; treating the instruction as always-on",
                doc_id=doc_id,
            )
        )
        return SCOPE_ALWAYS, [], []

    diagnostics.append(
        InstructionDiagnostic(
            code="UNKNOWN_DIRECTIVE",
            message=f"unknown directive '[{body}]' in heading "
            f"'{heading.strip()}'; treating the instruction as always-on",
            doc_id=doc_id,
        )
    )
    return SCOPE_ALWAYS, [], []


def _parse_lang_tokens(raw: str) -> list[str]:
    """Comma-separated ``[lang: ...]`` tokens, normalized and de-duped."""
    langs: list[str] = []
    for token in raw.split(","):
        language = normalize_language(token.strip())
        if language and language not in langs:
            langs.append(language)
    return langs


def _parse_when_keys(
    raw: str,
    heading: str,
    doc_id: str,
    diagnostics: list[InstructionDiagnostic],
) -> list[ConstructKey]:
    keys: list[ConstructKey] = []
    seen: set[ConstructKey] = set()
    for token in (t.strip().lower() for t in raw.split(",")):
        if not token:
            continue
        if not _KEY_RE.match(token):
            diagnostics.append(
                InstructionDiagnostic(
                    code="INVALID_CONSTRUCT_KEY",
                    message=f"'{token}' in heading '{heading.strip()}' is not "
                    f"a kind:name construct key; ignored",
                    doc_id=doc_id,
                )
            )
            continue
        kind, _, name = token.partition(":")
        key = ConstructKey(kind=kind, name=name)
        if key not in seen:
            seen.add(key)
            keys.append(key)
    return keys
