"""Macro-variable values and the reference expansion that resolves names.

A library or dataset in production SAS is routinely spelled through a macro
variable rather than written out::

    %let lname  = xwrk;
    %let suf    = batch_med;
    %let table1 = &suf;

    data &table1;
      set &lname..&table1;
    run;

While every dataset position in :mod:`chunker.metadata` required a bare
identifier, that step named no library the chunker could see, and the DATA
header's ``&table1`` was read as the identifier ``table1`` — inventing
``work.table1``, a dataset that does not exist. This module supplies the
missing half: the values a corpus assigns with ``%LET``, and the expansion that
turns ``&lname..&table1`` into ``xwrk.batch_med``.

Unresolved is reported, never guessed
-------------------------------------
A reference whose value is not in the table is kept **exactly as written** —
``&lib_out_spd..cia_hso_excl`` stays that, and reaches
``referenced_datasets`` / ``referenced_librefs`` like any other name. It is the
same posture :func:`chunker.metadata._clean_literal` takes for a dynamic
argument, applied to a name: a reference that only acquires a value at run time
is precisely what a migration needs told about, and dropping it would report the
step as touching no library at all.

Verbatim preservation is also why expansion substitutes over the whole string
rather than re-rendering it from tokens. ``&lname..&table1`` is "value of
lname", the dot SAS consumes to end that reference, then a literal dot, then
"value of table1"; re-emitting an *unresolved* reference from its parts would
have to re-escape that delimiter, and ``&lname.batch_med`` — which is what
naive re-rendering produces — means something else entirely.

Pure module: it imports nothing from the package, like :mod:`chunker.keywords`.
The pass that applies it to built chunks is
:func:`chunker.metadata.resolve_macro_var_refs`.

Logger name: ``chunker.macro_vars``.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping

logger = logging.getLogger(__name__)


#: A dataset-name token that may embed macro-variable references — ``&ds``,
#: ``&lib..&member``, ``lib.&member``, ``&ds.``. Every dataset position in
#: :mod:`chunker.metadata` is scanned with this rather than a bare identifier,
#: so a name that only exists once a macro variable resolves is still seen.
#: Published here so the statement grammar and the resolver cannot disagree
#: about what a name-carrying token looks like.
DS_REF_TOKEN = r"[A-Za-z_&][\w.&]*"

#: :data:`DS_REF_TOKEN` with a possessive quantifier, for the multi-dataset
#: ``SET`` / ``MERGE`` scans whose inner repetition must not backtrack (see
#: ``chunker.metadata._SET_RE``).
DS_REF_TOKEN_POSSESSIVE = r"[A-Za-z_&][\w.&]*+"

# One reference: its run of ampersands, its name, and the optional trailing dot
# SAS consumes as the reference's terminator. The ampersand run is captured
# rather than fixed at one because ``&&name`` is an *indirect* reference, which
# SAS resolves by rescanning — see :func:`_expand`.
_REF_RE = re.compile(r"(&+)([A-Za-z_]\w*)(\.?)")

# ``%let name = value`` up to the terminating semicolon. The name must be a
# plain identifier: ``%let &&outer&i = ...`` names a variable that is itself
# computed, so there is no key to store the value under. Deliberately does not
# require the ``;`` — a %LET truncated by an oversized split still assigns.
_LET_ASSIGN_RE = re.compile(r"%\s*let\s+([A-Za-z_]\w*)\s*=\s*([^;]*)", re.IGNORECASE)

# A %LET value worth keeping: one that could be a dataset or library name, or
# *part* of one. Any string at all can be a %LET value — an expression, a WHERE
# clause, a date literal, a path — and storing those on every chunk would bloat
# the serialised result to resolve names they can never name.
#
# Wider than DS_REF_TOKEN at the first character, which is the point: a bare
# number is not a name, but it is half of one in the idiom this exists to
# support — ``%let i = 1;`` feeding ``&&ds&i``, or ``%let yr = 2024;`` feeding
# ``&lib..sales&yr``.
_NAME_VALUE_RE = re.compile(r"[\w&][\w.&]*\Z")

#: Longest %LET value kept. A name-shaped value this long is not a name.
_MAX_VALUE_LEN = 128

# A value shaped like a dataset reference: ``libref.member``, where either half
# may still be an unresolved ``&ref`` and the separator may be the ``..`` that
# an expanded reference leaves behind. A one-level value is deliberately never
# a dataset — only the libref makes it unambiguous, the same rule
# ``chunker.batcher._TWO_LEVEL_DS_RE`` applies before rewriting a %LET value.
_DATASET_SHAPE_RE = re.compile(r"[A-Za-z_&][\w&]*\.\.?[A-Za-z_&][\w&.]*\Z")

#: Expansion passes before giving up. A chain of %LET indirections is a handful
#: deep in practice; the bound exists for the pathological source, since a
#: self-referential ``%let x = &x.y;`` would otherwise grow without end (the
#: expanded-name guard in :func:`resolve_refs` already stops that one, so this
#: is the second net, not the first).
_MAX_PASSES = 10


def strip_quotes(value: str) -> str:
    """*value* with one layer of matching surrounding quotes removed.

    SAS does not strip them — ``%let x = "a";`` stores ``"a"`` with the quotes —
    but a quoted value is still written to name the same dataset as an unquoted
    one, so name recognition looks through them. The same allowance
    :func:`chunker.batcher._map_let_values` makes before rewriting a %LET value.
    """
    value = value.strip()
    if len(value) >= 2 and value[0] in ("'", '"') and value[-1] == value[0]:
        return value[1:-1].strip()
    return value


def let_values(text: str) -> dict[str, str]:
    """Every ``%LET`` assignment in *text* that assigns a *name*, lowercased.

    *text* must be the comments-blanked, strings-intact form (``cf`` in
    :func:`chunker.metadata._metadata_for`): a %LET written inside a comment
    assigns nothing, and a quoted value has to survive to be unquoted here.

    Values that could not be a dataset or library name, or part of one, are
    dropped (see :data:`_NAME_VALUE_RE`), and a repeated name keeps the last
    assignment — what a sequential SAS session would hold by the chunk's end.
    Keys and values are both lowercased, consistent with the package's
    lowercase-everything policy: a value here only ever becomes part of a
    dataset or libref name.
    """
    values: dict[str, str] = {}
    for m in _LET_ASSIGN_RE.finditer(text):
        value = strip_quotes(m.group(2))
        if not value or len(value) > _MAX_VALUE_LEN:
            continue
        if not _NAME_VALUE_RE.match(value):
            continue
        values[m.group(1).lower()] = value.lower()
    if values and logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"let_values: {values}")
    return values


def _expand_once(
    text: str, table: Mapping[str, str], skip: frozenset[str], rescan: bool
) -> tuple[str, set[str]]:
    """One expansion pass over *text*; returns (new text, names expanded).

    A reference whose name is in *skip* or absent from *table* is returned
    verbatim — ampersands and trailing delimiter dot included — so an
    unresolved span comes out of the substitution byte-for-byte as it went in.
    """
    expanded: set[str] = set()

    def _sub(m: re.Match[str]) -> str:
        amps, name, dot = m.group(1), m.group(2).lower(), m.group(3)
        if len(amps) > 1:
            # Indirect: SAS's scanner consumes one ampersand per pass and
            # re-reads what is left, so ``&&a&i`` becomes ``&a1`` and only then
            # names a variable. Only done under `rescan` — see resolve_refs.
            # The name is deliberately *not* recorded as expanded: nothing was
            # looked up yet, and marking it would bar the pass that does.
            if not rescan:
                return m.group(0)
            return f"{amps[1:]}{name}{dot}"
        if name in skip or name not in table:
            return m.group(0)
        expanded.add(name)
        return table[name]

    return _REF_RE.sub(_sub, text), expanded


def _expand(text: str, table: Mapping[str, str], *, rescan: bool) -> str:
    """Expand *text* until nothing more resolves, or :data:`_MAX_PASSES`.

    A pass that changes nothing is the end of the work — which covers a value
    that expands to itself as well as a text with nothing left to look up.
    """
    out = text
    skip: frozenset[str] = frozenset()
    for _ in range(_MAX_PASSES):
        new, expanded = _expand_once(out, table, skip, rescan)
        if new == out or "&" not in new:
            return new
        out = new
        skip |= expanded
    logger.debug(f"_expand: {_MAX_PASSES} passes exhausted on {text!r} → {out!r}")
    return out


def resolve_refs(text: str, table: Mapping[str, str]) -> str:
    """*text* with every resolvable ``&name`` replaced by its value.

    Expansion repeats until nothing more resolves, so a chain
    (``&table1`` → ``&suf`` → ``batch_med``) resolves in full. A name already
    expanded on an earlier pass is not expanded again: that is what stops
    ``%let x = &x.y;`` — a legitimate SAS append idiom — from growing the
    string forever, and it leaves the self-reference written as it stands.

    The trailing dot of ``&name.`` is consumed with the reference, as SAS
    consumes it, so ``&lname..&table1`` yields ``xwrk.batch_med`` — one dot
    ends the reference, the other is the library separator.

    An **indirect** ``&&name&i`` is rescanned — the ubiquitous "loop over a
    numbered list" idiom, where ``&&ds&i`` has to become ``&ds1`` before it
    names anything — but all-or-nothing: unless the rescan resolves it
    completely, *text* comes back untouched. A half-rescanned ``&&a1`` is
    neither the source somebody can find in their own file nor a name anything
    can look up, so the reference is reported exactly as it was written.
    """
    if "&" not in text or not table:
        return text
    direct = _expand(text, table, rescan=False)
    if "&&" not in direct:
        return direct
    rescanned = _expand(direct, table, rescan=True)
    return rescanned if "&" not in rescanned else text


def has_macro_ref(name: str) -> bool:
    """True if *name* still holds an unresolved macro-variable reference."""
    return "&" in name


def is_dataset_shaped(value: str) -> bool:
    """True if *value* is written like a ``libref.member`` dataset reference.

    Either half may be an unresolved ``&ref``, so
    ``&lib_out_spd..cia_hso_excl`` qualifies and ``batch_med`` does not: a
    one-level value is just a string until something uses it in a dataset
    position, whereas a two-level one names a library on sight.
    """
    return bool(_DATASET_SHAPE_RE.match(value))
