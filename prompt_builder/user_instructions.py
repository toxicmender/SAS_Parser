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

from pydantic import BaseModel, Field

from .models import ConstructKey, DocRole, InstructionChunk, InstructionDiagnostic

logger = logging.getLogger(__name__)

SCOPE_ALWAYS = "always"
SCOPE_WHEN = "when"
SCOPE_TOPIC = "topic"

_HEADING_RE = re.compile(r"^\s{0,3}##+\s*(?P<heading>.*\S)\s*$", re.MULTILINE)
_DIRECTIVE_RE = re.compile(r"^\[(?P<directive>[^\]]*)\]\s*(?P<title>.*)$")
_KEY_RE = re.compile(r"^[a-z_]\w*:\S+$")

_PREAMBLE_TITLE = "General"


def _scope_tag(scope: str) -> str:
    return f"scope:{scope}"


def scope_of(chunk: InstructionChunk) -> str:
    """The parsed scope of a user-instruction chunk (``always`` fallback)."""
    for tag in chunk.tags:
        if tag.startswith("scope:"):
            return tag.removeprefix("scope:")
    return SCOPE_ALWAYS


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
        cls, text: str, *, doc_id: str = "user", source: str | None = None
    ) -> "UserInstructionSet":
        """Parse *text* (markdown-ish, see module docstring) into a set."""
        fingerprint = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
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
            scope, clean_title, keys = _parse_heading(title, doc_id, diagnostics)
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
                    tags=[_scope_tag(scope)],
                )
            )

        logger.info(
            f"UserInstructionSet.from_text: {len(chunks)} instruction(s)  "
            f"always={sum(1 for c in chunks if scope_of(c) == SCOPE_ALWAYS)}  "
            f"when={sum(1 for c in chunks if scope_of(c) == SCOPE_WHEN)}  "
            f"topic={sum(1 for c in chunks if scope_of(c) == SCOPE_TOPIC)}  "
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
) -> tuple[str, str, list[ConstructKey]]:
    """``(scope, clean_title, construct_keys)`` for one section heading."""
    match = _DIRECTIVE_RE.match(heading.strip())
    if match is None:
        return SCOPE_ALWAYS, heading.strip(), []

    directive = match.group("directive").strip()
    title = match.group("title").strip() or directive
    lowered = directive.lower()

    if lowered == SCOPE_TOPIC:
        return SCOPE_TOPIC, title, []

    if lowered.startswith("when:"):
        keys = _parse_when_keys(directive[5:], heading, doc_id, diagnostics)
        if keys:
            return SCOPE_WHEN, title, keys
        # No usable key: over-include rather than silently drop the rule.
        diagnostics.append(
            InstructionDiagnostic(
                code="INVALID_CONSTRUCT_KEY",
                message=f"when-directive in heading '{heading.strip()}' lists "
                f"no valid construct keys; treating the instruction as always-on",
                doc_id=doc_id,
            )
        )
        return SCOPE_ALWAYS, title, []

    diagnostics.append(
        InstructionDiagnostic(
            code="UNKNOWN_DIRECTIVE",
            message=f"unknown directive '[{directive}]' in heading "
            f"'{heading.strip()}'; treating the instruction as always-on",
            doc_id=doc_id,
        )
    )
    return SCOPE_ALWAYS, title, []


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
