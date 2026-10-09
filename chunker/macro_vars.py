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
from typing import NamedTuple

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


def name_value(raw: str) -> str | None:
    """*raw* as a value that can name a dataset, a library or part of one —
    lowercased, quotes stripped — or ``None`` when it cannot.

    The one test every source of macro-variable values applies, so ``%LET``,
    ``CALL SYMPUTX``, a ``%MACRO`` parameter's default and a call's argument
    cannot disagree about what counts as a name.
    """
    value = strip_quotes(raw)
    if not value or len(value) > _MAX_VALUE_LEN or not _NAME_VALUE_RE.match(value):
        return None
    return value.lower()


#: Longest %LET value kept as text for a path. A path is rarely longer.
_MAX_TEXT_LEN = 1024


def text_value(raw: str) -> str | None:
    """*raw* as a value pasted into a path — case, separators and quotes kept —
    or ``None`` when its value is not knowable here.

    The counterpart of :func:`name_value` for a reference inside a quoted path
    (``"&root/setup.sas"``), where SAS pastes the value as written:
    ``/SAS/Prod``, ``C:\\Projects``, or a quoted ``'/sas/x.sas'`` that
    ``%include &f;`` reads as its own quoted path. A value that calls a macro
    function (``%sysget(HOME)``, ``%sysfunc(getoption(work))``) is computed at
    run time, so it is unknown, as an empty one is.
    """
    value = raw.strip()
    if not value or len(value) > _MAX_TEXT_LEN or "%" in value:
        return None
    return value


def let_assignments(text: str) -> list[tuple[str, str]]:
    """Every ``%LET`` in *text*, in order: ``(name, value as written)``.

    Name lowercased, value stripped of surrounding whitespace and nothing
    else — the raw material :func:`let_values` and a called macro's
    global-assignment analysis (``chunker.metadata._MacroDef``) both read.
    """
    return [
        (m.group(1).lower(), m.group(2).strip()) for m in _LET_ASSIGN_RE.finditer(text)
    ]


def let_values(text: str) -> dict[str, str]:
    """The value each ``%LET`` in *text* leaves its variable holding, lowercased.

    *text* must be the comments-blanked, strings-intact form (``cf`` in
    :func:`chunker.metadata._metadata_for`): a %LET written inside a comment
    assigns nothing, and a quoted value has to survive to be unquoted here.

    A value that could not be a dataset or library name, or part of one (see
    :func:`name_value`), maps to ``""`` — *unknown from here on* — rather than
    being left out: ``%let sch = edw; ... %let sch = %scan(&list, 2);`` must not
    leave a later ``&sch`` resolving to the stale ``edw``. A repeated name keeps
    the last assignment — what a sequential SAS session would hold by the
    chunk's end.
    """
    values = {name: name_value(raw) or "" for name, raw in let_assignments(text)}
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
        # An empty value means *unknown* throughout this package — it is how a
        # layer hides a name it shadows — so it resolves nothing.
        value = None if name in skip else table.get(name)
        if not value:
            return m.group(0)
        expanded.add(name)
        return value

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


# ---------------------------------------------------------------------------
# Macro call arguments
# ---------------------------------------------------------------------------
# The grammar of a call's argument list, used by the batcher to resolve a
# macro body's parameterised datasets per call site and by
# chunker.metadata to resolve the database tables a called macro's
# pass-through SQL names. Pure, like the rest of this module.

# Locates the opening of a macro call's argument list: %macroname( . The
# balanced closing paren is found by _extract_call_arg_text below.
_CALL_OPEN_RE = re.compile(r"%\s*[A-Za-z_]\w*\s*\(")

# A keyword argument is name= at the start of the (stripped) argument, so a
# positional value like f(x=1) is not mistaken for keyword 'f(x'.
_KW_ARG_RE = re.compile(r"([A-Za-z_]\w*)\s*=(.*)$", re.DOTALL)

# One call in a run of them: %name, then its argument list when a "(" follows.
# Semicolons between calls are empty statements.
_CALL_RUN_RE = re.compile(r"[\s;]*(%\s*([A-Za-z_]\w*))\s*(\()?")
_PAREN_RE = re.compile(r"[()]")


class CallSpan(NamedTuple):
    """One macro call found by :func:`call_spans`."""

    name: str  # lowercased
    start: int  # the "%"
    end: int  # past its closing ")", or its name when no list follows
    closed: bool  # False: the argument list runs to the end of the text


def call_spans(mt: str) -> list[CallSpan]:
    """Each macro call *mt* opens with, back to back.

    A call needs no semicolon: it ends at the parenthesis closing its argument
    list — or at its name, when no list follows — so ``%pull(tbl=a)`` and
    ``%pull(tbl=b)`` on consecutive lines are two calls, and the next starts
    only where another ``%name`` follows; anything else ends the run.

    *mt* must be sanitised text (comments and string interiors blanked), so a
    parenthesis inside a quoted argument cannot end a call. A list left open
    runs to the end of the text and is not ``closed``. Offsets index *mt*,
    which is char-aligned with the text it was made from. The parentheses are
    found by regex, not character by character: the scanner runs this on every
    statement that opens with a call.
    """
    spans: list[CallSpan] = []
    pos = 0
    while m := _CALL_RUN_RE.match(mt, pos):
        end, closed = m.end(1), True
        if m.group(3) is not None:
            depth, end, closed = 0, len(mt), False
            for paren in _PAREN_RE.finditer(mt, m.start(3)):
                depth += 1 if paren.group() == "(" else -1
                if depth == 0:
                    end, closed = paren.end(), True
                    break
        spans.append(CallSpan(m.group(2).lower(), m.start(1), end, closed))
        pos = end
    return spans


def _balanced_text(text: str, start: int) -> str:
    """The text from *start*, just past an opening paren, to its balanced
    closing paren.

    Walks characters with a paren-depth counter, treating single- and
    double-quoted spans as opaque, so nested constructs like
    ``%clean(%str(a,b), out=f(x))`` yield the full argument text instead
    of stopping at the first ``)``.  Unbalanced (a truncated chunk), it is
    everything after the opening paren.
    """
    depth = 1
    quote: str | None = None
    for i in range(start, len(text)):
        ch = text[i]
        if quote:
            if ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return text[start:i]
    return text[start:]


def _extract_call_arg_text(call_text: str) -> str | None:
    """Return the text between the call's balanced outer parens, or None."""
    m = _CALL_OPEN_RE.search(call_text)
    if not m:
        return None
    return _balanced_text(call_text, m.end())


# A %MACRO statement's name, and the paren its parameter list opens with.
_MACRO_HEAD_RE = re.compile(r"%\s*macro\s+([A-Za-z_]\w*)\s*(\()?", re.IGNORECASE)


def macro_signature(text: str) -> list[tuple[str, str | None]]:
    """The ``(name, default)`` parameters of the first ``%MACRO`` statement
    in *text*, in signature order, names lowercased; ``default`` is ``None``
    for a positional parameter.

    The list is read with balanced parentheses and split on top-level commas,
    so a default holding either — ``list=%str(a,b)``, ``fmt=put(x, 8.)`` — is
    one parameter, as SAS reads it.
    """
    m = _MACRO_HEAD_RE.search(text)
    if m is None or m.group(2) is None:
        return []
    params: list[tuple[str, str | None]] = []
    for part in _split_call_args(_balanced_text(text, m.end())):
        name, eq, default = part.partition("=")
        params.append((name.strip().lower(), default.strip() if eq else None))
    return params


def _split_call_args(raw_args: str) -> list[str]:
    """Split an argument list on top-level commas (quote- and paren-aware)."""
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    quote: str | None = None
    for ch in raw_args:
        if quote:
            if ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append("".join(buf))
            buf = []
            continue
        buf.append(ch)
    parts.append("".join(buf))
    return [p.strip() for p in parts if p.strip()]


def _parse_call_args(
    call_text: str, *, as_written: bool = False
) -> tuple[list[str], dict[str, str]]:
    """
    Parse a MACRO_CALL chunk's raw text into (positional_args, keyword_args).

    Used by Fix B (parameterised macro output/input resolution): the
    definition's ``body_param_outputs``/``body_param_inputs`` reference a
    parameter by name and positional index, and this function recovers the
    actual values supplied at the call site so those references can be
    resolved to concrete dataset names.

    Quoting and trailing dots are stripped from each value so that
    ``work.orders``, ``'work.orders'``, and ``work.orders.`` all normalise
    to the same lowercase dataset key. *as_written* keeps each value as the
    call spells it, only trimmed — what a path the macro builds from it reads
    (``%setpaths(/SAS/Prod)``).
    """
    raw_args = _extract_call_arg_text(call_text)
    if raw_args is None:
        return [], {}

    clean = str.strip if as_written else _clean_arg_value
    positional: list[str] = []
    keyword: dict[str, str] = {}

    for part in _split_call_args(raw_args):
        kw = _KW_ARG_RE.match(part)
        if kw:
            keyword[kw.group(1).lower()] = clean(kw.group(2))
        else:
            positional.append(clean(part))

    return positional, keyword


def _clean_arg_value(value: str) -> str:
    """Strip quotes and a single trailing dot from a macro call argument."""
    v = value.strip()
    if v.startswith(("'", '"')) and v.endswith(("'", '"')):
        v = v[1:-1]
    return v.rstrip(".").lower()
