"""Where a physical path appears in SAS syntax — the single definition.

A SAS corpus is full of locations that mean nothing on the target: the library
``libname dataetl '/sasdata3/dataetl'`` points at, the flat file ``infile`` reads,
the source ``%include`` pulls in. Migrating the corpus means knowing what they
are, so this module recognises them and :mod:`chunker.metadata` stores what it
finds on every chunk.

One grammar, two consumers
--------------------------
This table used to live in :mod:`xref.pre`, which rewrites those same paths from
an XREF mapping. Two modules recognising "where a path belongs in SAS" would be
two definitions drifting apart — the failure Architecture.md invariant 12 names —
so the grammar lives here, with the chunker that owns SAS syntax, and
:mod:`xref.pre` imports it. A spec says where each quoted value of a match stands
(:meth:`PathSpec.value_spans`) — one, or several for a list such as ``%include
'/a.sas' '/b.sas';`` — and the substitution rewrites exactly those, leaving the
statement around them as written.

Not every quoted argument is a directory
----------------------------------------
``FILENAME`` takes a device keyword that redirects the identical syntax at an FTP
server, a mailbox, or a shell pipe. Those are real external dependencies — a job
that mails its output still has somewhere to send it after migration — so they
are classified (:class:`~chunker.models.PathLocation`) rather than dropped, and
an unrecognised device becomes ``DEVICE`` rather than being mistaken for a
filesystem path somebody then tries to mount.

Matching is by *statement*, never by a bare path-shaped string: a table name in a
comment or a string that merely looks like a path is not a path. The scans run on
the comments-blanked, strings-intact form of the text, because the value lives
inside the string literal that the fully sanitised form has already blanked.

Known limit: ``options sasautos=('/a' '/b')`` reports only the first entry;
reporting the first is better than reporting none.

A statement may also name a place through a fileref (``infile in;``,
``%include src(one);``). Those specs record the fileref, as
:attr:`~chunker.models.PathLocation.FILEREF`, and
:func:`chunker.metadata.resolve_filerefs` gives each the place the FILENAME in
force names.

Or it may concatenate places SAS treats as one — ``libname x ('/a' v9);``,
``filename f ('/q1.sas' '/q2.sas');``, ``infile '("/a.csv" "/b.csv")';`` — and
then each place is a reference with its position in the list
(:attr:`~chunker.models.SasPathRef.concat_index`). A library a LIBNAME names by
its libref is :attr:`~chunker.models.PathLocation.LIBREF`: no place of its own.

Not every LIBNAME names a place
-------------------------------
``libname edwprod oracle path=... user=... pass=...`` has no quoted directory, so
none of :data:`PATH_STATEMENTS` — every entry of which ends in a quoted value —
can see it, and the connection a migration has to reproduce went unrecorded.
:func:`extract_engine_refs` is the second scan, for that form. It is deliberately
*not* another :class:`PathSpec`: the ``head``/``q``/``path`` groups exist so
:mod:`xref.pre` can substitute a quoted value, and there is no quoted value here
to substitute. The two scans cannot both match one statement — one requires a
quoted argument, the other requires the second token to be a database engine.

Logger name: ``chunker.paths``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from .macro_vars import DS_REF_TOKEN, DS_REF_TOKEN_POSSESSIVE
from .models import PathLocation, SasEngineRef, SasPathRef

logger = logging.getLogger(__name__)

#: FILENAME device keywords that reach a remote service rather than a local
#: filesystem. Lowercased; compared against the device group verbatim.
_REMOTE_DEVICES = frozenset(
    {"ftp", "sftp", "url", "webdav", "hadoop", "s3", "azure", "zip"}
)

#: The device keyword addressing a mailbox.
_EMAIL_DEVICES = frozenset({"email", "emailsys"})

#: The device keyword handing the value to the shell. Not a location at all —
#: the "path" is a command line.
_PIPE_DEVICES = frozenset({"pipe"})

#: Device keywords that are filesystem access after all: these name *how* SAS
#: reads a local file, not somewhere else to read it from.
_FILESYSTEM_DEVICES = frozenset({"disk", "temp", "dummy", "clipbrd"})

#: The device types an INFILE or FILE statement may name after a quoted file
#: specification, or as DEVICE= (SAS 9.4 DATA Step Statements, INFILE
#: Statement: "device-type ... must immediately follow file-specification"):
#: ``infile 'ps -e' pipe;`` reads what the command prints.
_DEVICE_TYPES = (
    "activemq|catalog|clipboard|disk|dummy|email|ftp|gterm|hadoop|jms|pipe"
    "|plotter|printer|sftp|socket|tape|terminal|uprinter|url|webdav|zip"
)


def classify_location(device: str | None) -> PathLocation:
    """The :class:`~chunker.models.PathLocation` a device keyword implies.

    ``None`` — no device keyword, a bare quoted argument — is
    :attr:`~chunker.models.PathLocation.FILESYSTEM`. An unknown keyword is
    :attr:`~chunker.models.PathLocation.DEVICE`, deliberately: SAS gains device
    types, and a new one must surface as "something else" rather than quietly
    joining the paths a consumer will try to map to storage.
    """
    if device is None:
        return PathLocation.FILESYSTEM
    token = device.strip().lower()
    if token in _REMOTE_DEVICES:
        return PathLocation.REMOTE
    if token in _EMAIL_DEVICES:
        return PathLocation.EMAIL
    if token in _PIPE_DEVICES:
        return PathLocation.PIPE
    if token in _FILESYSTEM_DEVICES:
        return PathLocation.FILESYSTEM
    return PathLocation.DEVICE


def normalise_path(raw: str) -> str:
    """*raw* as a comparison key: stripped, lowercased, backslashes forward.

    The same normalisation :func:`chunker.statements._quoted_path` applies to
    quoted dataset references, minus its quote wrapper — that exists only to keep
    a path key out of the dataset *identifier* namespace, and nothing here shares
    a namespace with identifiers. Per-OS case sensitivity is ignored, consistent
    with the module's lowercase-everything policy.
    """
    return raw.strip().replace("\\", "/").lower()


@dataclass(frozen=True)
class PathSpec:
    """One statement form that names an external reference.

    Attributes
    ----------
    statement
        The value :attr:`~chunker.models.SasPathRef.statement` takes.
    keyword
        A lowercase literal every match must contain, used as a cheap substring
        gate before the regex runs — the convention the rest of
        :mod:`chunker.metadata` scans follow.
    pattern
        Compiled, with ``head`` / ``q`` / ``path`` groups (see the module
        docstring) plus optional ``binds`` and ``device`` groups.
    many
        The statement names several values: each quoted value in its
        ``paths`` group (``libname x ('/a' '/b');``), or, for a fileref spec,
        each member in ``members``.
    fileref
        The statement names a fileref, not a place (``infile in;``,
        ``file out(report.txt);``): its references are
        :attr:`~chunker.models.PathLocation.FILEREF`, ``binds`` the fileref,
        until :func:`chunker.metadata.resolve_filerefs` finds the FILENAME.
    at_statement
        A match counts only where a statement starts (:func:`_at_statement`).
        The fileref forms are bare words, so without it "file written" in a
        PUT string, a TITLE or a ``%PUT`` message would read as a FILE
        statement naming a fileref. Checked after the match rather than by a
        lookbehind, which would cost the pattern its literal prefix.
    concat
        The ``paths`` group is a concatenation (``libname x ('/a' '/b');``):
        each place it names carries its position
        (:attr:`~chunker.models.SasPathRef.concat_index`), and a bare word in
        a LIBNAME's list is a libref
        (:attr:`~chunker.models.PathLocation.LIBREF`). Implies *many*.
    sources
        The ``paths`` group is a ``%INCLUDE`` source list, whose sources mix
        as the statement likes: a quoted file (``'/a.sas'``, or a list in one
        string), a fileref, or members of the directory a fileref names
        (``src(one two)``). Each is a reference, in order; only a quoted
        file has a value to rewrite. The bare-word forms count only where a
        statement starts (as *at_statement*), so a quoted file in CALL
        EXECUTE text still counts. Implies *many*.
    """

    statement: str
    keyword: str
    pattern: re.Pattern[str]
    many: bool = False
    fileref: bool = False
    at_statement: bool = False
    concat: bool = False
    sources: bool = False

    def location_for(self, match: re.Match[str]) -> PathLocation:
        """Where this match points, from its device group if it has one."""
        if self.fileref:
            return PathLocation.FILEREF
        return classify_location(_group(match, "device"))

    def value_spans(self, match: re.Match[str]) -> list[tuple[int, int]]:
        """Where each quoted value of *match* stands in ``match.string``,
        quotes excluded: what :mod:`xref.pre` rewrites. A fileref has none."""
        if self.fileref:
            return []
        if not self.many:
            return [match.span("path")]
        start, end = match.span("paths")
        if self.sources:
            # A quoted source's values; a quoted member name in `src("a.txt")`
            # is no path.
            return [
                value.span("path")
                for source in _SOURCE_RE.finditer(match.string, start, end)
                if source.group("quoted") is not None
                for value in _VALUE_RE.finditer(match.string, *source.span("quoted"))
            ]
        return [m.span("path") for m in _VALUE_RE.finditer(match.string, start, end)]

    def refs_for(self, match: re.Match[str]) -> list[SasPathRef]:
        """Every :class:`~chunker.models.SasPathRef` *match* names.

        A quoted value that is empty is left out — ``infile '';`` names
        nothing, and an empty entry in an inventory is worse than no entry.
        """
        if self.fileref:
            return self._fileref_refs(match)
        if self.concat:
            return self._concat_refs(match)
        if self.sources:
            return self._source_refs(match)
        refs = [self._quoted_ref(match, start, end) for start, end in self.value_spans(match)]
        return [ref for ref in refs if ref is not None]

    def _source_refs(self, match: re.Match[str]) -> list[SasPathRef]:
        """One reference per file a ``%INCLUDE`` source list names, in order:
        ``/a.sas``, ``src(b)`` and ``setup`` for ``%include '/a.sas' src(b)
        setup;``. A fileref's is spelled as the statement spells it,
        lowercased, until :func:`chunker.metadata.resolve_filerefs` follows
        it."""
        start, end = match.span("paths")
        bare = _at_statement(match.string, match.start())
        refs: list[SasPathRef] = []
        for source in _SOURCE_RE.finditer(match.string, start, end):
            if source.group("quoted") is not None:
                for value in _VALUE_RE.finditer(match.string, *source.span("quoted")):
                    if (ref := self._quoted_ref(match, *value.span("path"))) is not None:
                        refs.append(ref)
                continue
            fref = source.group("fref")
            if fref is not None and bare:  # not internal lines or the keyboard
                refs += self._through_fileref(fref, source.group("members"))
        return refs

    def _statement_at(self, match: re.Match[str]) -> str:
        """The statement *match* is: a LIBNAME clause in a CREATE VIEW's USING
        list is ``using_libname`` (:func:`in_using_clause`)."""
        if self.statement == "libname" and in_using_clause(match.string, match.start()):
            return "using_libname"
        return self.statement

    def _quoted_ref(
        self, match: re.Match[str], start: int, end: int, index: int | None = None
    ) -> SasPathRef | None:
        """The reference for the quoted value at ``[start, end)`` of
        ``match.string``, or ``None`` when the value is empty."""
        raw = match.string[start:end].strip()
        if not raw:
            return None
        device = _group(match, "device")
        engine = _group(match, "engine")
        binds = _group(match, "binds")
        return SasPathRef(
            statement=self._statement_at(match),
            location=classify_location(device),
            path=normalise_path(raw),
            raw=raw,
            binds=binds.lower() if binds else None,
            device=device.lower() if device else None,
            engine=engine.lower() if engine else None,
            # SAS resolves macro references in a double-quoted value only:
            # in '/data/R&D/x.csv' the & is a character of the name.
            has_macro_ref="&" in raw and _governing_quote(match, start) != "'",
            concat_index=index,
        )

    def _concat_refs(self, match: re.Match[str]) -> list[SasPathRef]:
        """One reference per place a concatenation names, each with its
        position: ``/a`` (0), ``v9`` (1) and ``/b`` (2) for ``libname x ('/a'
        v9 '/b');``. A LIBNAME's engine applies to the directories it names,
        not to a library named by its libref (SAS 9.4 Global Statements,
        LIBNAME Statement: Rules for Library Concatenation)."""
        start, end = match.span("paths")
        binds = _group(match, "binds")
        refs: list[SasPathRef] = []
        for index, member in enumerate(_CONCAT_MEMBER_RE.finditer(match.string, start, end)):
            word = member.group("libref")
            if word is None:
                ref = self._quoted_ref(match, *member.span("path"), index)
                if ref is not None:
                    refs.append(ref)
            elif self.statement == "libname":
                refs.append(
                    SasPathRef(
                        statement=self._statement_at(match),
                        location=PathLocation.LIBREF,
                        path=word.lower(),
                        raw=word,
                        binds=binds.lower() if binds else None,
                        has_macro_ref="&" in word,
                        concat_index=index,
                    )
                )
        return refs

    def _fileref_refs(self, match: re.Match[str]) -> list[SasPathRef]:
        """``out(report.txt)`` for ``file out(report.txt);``, ``in`` for
        ``infile in;``."""
        return self._through_fileref(match.group("binds"), _group(match, "members"))

    def _through_fileref(self, fref: str, members: str | None) -> list[SasPathRef]:
        """The references made through the fileref *fref*: the file it names,
        or each of *members* of its directory (``src(one)`` and ``src(two)``
        for ``src(one two)``). Written as the statement spells them,
        lowercased, until the FILENAME's path replaces them."""
        spelled = [fref] if members is None else [f"{fref}({m})" for m in _MEMBER_RE.findall(members)]
        return [
            SasPathRef(
                statement=self.statement,
                location=PathLocation.FILEREF,
                path=raw.lower(),
                raw=raw,
                binds=fref.lower(),
                has_macro_ref="&" in raw,
            )
            for raw in spelled
        ]


def in_using_clause(text: str, pos: int) -> bool:
    """Whether the LIBNAME at *pos* of *text* is a clause in the USING list
    that ends a PROC SQL CREATE VIEW — ``using libname x '/p', libname y
    oracle ...;`` — after USING, or after the comma between two clauses.

    Its libref is "local to the view, and will not conflict with any
    identically named librefs in the SAS session" (SAS 9.4 SQL Procedure
    User's Guide, CREATE VIEW Statement): assigned while the view runs, never
    in the session. A LIBNAME statement stands after a ``;``, never a comma.
    """
    i = pos
    while i > 0 and text[i - 1].isspace():
        i -= 1
    if i > 0 and text[i - 1] == ",":
        return True
    if text[max(0, i - 5) : i].lower() != "using":
        return False
    return i == 5 or not (text[i - 6].isalnum() or text[i - 6] == "_")


def _at_statement(text: str, pos: int) -> bool:
    """Whether a statement starts at *pos* of *text*: at its start, after a
    ``;``, or after the THEN / ELSE (or ``%THEN`` / ``%ELSE``) that runs it,
    whitespace between."""
    i = pos
    while i > 0 and text[i - 1].isspace():
        i -= 1
    if i == 0 or text[i - 1] == ";":
        return True
    if text[max(0, i - 4) : i].lower() not in ("then", "else"):
        return False
    return i == 4 or not (text[i - 5].isalnum() or text[i - 5] == "_")


def _group(match: re.Match[str], name: str) -> str | None:
    """*match*'s *name* group, or ``None`` when the pattern has no such group.

    ``match.group(name)`` raises for a name the pattern never declared, and the
    specs deliberately declare only the groups their statement actually has —
    ``groupdict`` answers both questions with one lookup.
    """
    return match.groupdict().get(name)


def _governing_quote(match: re.Match[str], start: int) -> str:
    """The quote SAS reads the value at *start* between: its own, or, in a
    quoted list of files (``'("/a.sas" "/b.sas")'``), the list's — which
    decides whether a ``&`` in it is a macro reference."""
    if "paths" in match.re.groupindex:
        lo, hi = match.span("paths")
        for literal in _QUOTED_RE.finditer(match.string, lo, hi):
            if literal.start() < start < literal.end():
                return literal.group()[0]
    # A quoted command (`x 'cd &dir';`): its quotes govern the value in it.
    if "q" in match.re.groupindex and match.group("q") and match.start("q") < start:
        return match.group("q")
    return match.string[start - 1] if start else ""


# The quoted-value tail every spec ends with: an opening quote, the value, and
# the same quote again. A SAS statement's quoted literal does not span lines.
_VALUE = r"(?P<q>['\"])(?P<path>[^'\"\n]*)(?P=q)"
_VALUE_RE = re.compile(_VALUE)
# The device type an INFILE or FILE may name after its quoted value.
_DEVICE_AFTER = rf"(?:\s+(?:device\s*=\s*)?(?P<device>{_DEVICE_TYPES})\b)?"
# One quoted value, ungrouped, to repeat in a list.
_QUOTED = r"""(?:'[^'\n]*'|"[^"\n]*")"""
_QUOTED_RE = re.compile(_QUOTED)
# One place in a concatenation: a quoted value, or a LIBNAME's libref.
_CONCAT_MEMBER_RE = re.compile(rf"{_VALUE}|(?P<libref>{DS_REF_TOKEN})")
# One library in a LIBNAME concatenation, and the blank or comma between two.
# Possessive, so a list with no closing parenthesis fails at once instead of
# trying every way of splitting its words.
_LIBRARY = rf"(?:{_QUOTED}|{DS_REF_TOKEN_POSSESSIVE})"
_LIST_SEP = r"\s*+,?\s*+"
# A fileref, and the members of a directory it names: `src(one two)`.
_FREF = r"[A-Za-z_&][\w&]*"
_MEMBER_RE = re.compile(r"""'[^'\n]*'|"[^"\n]*"|[\w&.$]+""")
# Filerefs SAS reserves for its own destinations and in-stream data.
_RESERVED_FREFS = r"(?!(?:cards4?|datalines4?|print|log|_webout)\b)"
# %INCLUDE and its abbreviation %INC.
_INCLUDE = r"%\s*inc(?:lude)?\b"
# One %INCLUDE source: a quoted file, a fileref and the members of its
# directory, a fileref, internal lines (n, n-m, n:m) or the keyboard (*).
_SOURCE = rf"(?:{_QUOTED}|{_FREF}(?:\s*\([^)]*\))?|\d+(?:\s*[-:]\s*\d+)?|\*)"
_SOURCE_RE = re.compile(
    rf"(?P<quoted>{_QUOTED})|(?P<fref>{_FREF})(?:\s*\((?P<members>[^)]*)\))?"
    r"|\d+(?:\s*[-:]\s*\d+)?|\*"
)

#: Every statement form recognised, in scan order. Iterated by
#: :func:`extract_paths` and by :func:`xref.pre.rewrite_source_text`.
PATH_STATEMENTS: tuple[PathSpec, ...] = (
    # libname <libref> [<engine>] '<path>' — the engine group is *captured*
    # because an engine changes what the directory is (see SasPathRef.engine):
    # `libname x spde '/p'` is a partitioned SPD Engine library, not the plain
    # directory `libname x '/p'` names.
    #
    # The libref is matched with DS_REF_TOKEN, not a bare identifier, because
    # `libname &lib_out '/data/x';` is an ordinary way to write one — and while
    # this required an identifier, that statement matched nothing at all and its
    # directory went unrecorded. `binds` then holds the reference as written
    # until chunker.metadata.resolve_macro_var_refs gives it a value; the
    # `head`/`q`/`path` shape xref.pre substitutes on is unchanged, since a
    # wider libref only widens `head`.
    PathSpec(
        statement="libname",
        keyword="libname",
        pattern=re.compile(
            rf"(?P<head>\blibname\s+(?P<binds>{DS_REF_TOKEN})\s+"
            r"(?:(?P<engine>[A-Za-z_]\w*)\s+)?)" + _VALUE,
            re.IGNORECASE,
        ),
    ),
    # libname <libref> [<engine>] (<library> ...): a concatenation, one library
    # SAS searches in order. Each library is a quoted directory or a libref
    # already assigned, separated by blanks or commas: `libname profits
    # (income, costs, '/u/2011/capgain');` (SAS 9.4 Global Statements, LIBNAME
    # Statement, Form 4).
    PathSpec(
        statement="libname",
        keyword="libname",
        pattern=re.compile(
            rf"(?P<head>\blibname\s+(?P<binds>{DS_REF_TOKEN})"
            r"(?:\s+(?P<engine>[A-Za-z_]\w*))?\s*)"
            rf"\(\s*(?P<paths>{_LIBRARY}(?:{_LIST_SEP}{_LIBRARY})*+)\s*\)",
            re.IGNORECASE,
        ),
        many=True,
        concat=True,
    ),
    # filename <fileref> [<device>] '<path>'
    PathSpec(
        statement="filename",
        keyword="filename",
        pattern=re.compile(
            rf"(?P<head>\bfilename\s+(?P<binds>{DS_REF_TOKEN})\s+"
            r"(?:(?P<device>[A-Za-z_]\w*)\s+)?)" + _VALUE,
            re.IGNORECASE,
        ),
    ),
    # filename <fileref> ('<path>' ...): files read one after another, or
    # directories searched in order for a member (SAS 9.4 Companion for UNIX
    # Environments, Concatenating Filenames; Assigning a Fileref to Several
    # Directories).
    PathSpec(
        statement="filename",
        keyword="filename",
        pattern=re.compile(
            rf"(?P<head>\bfilename\s+(?P<binds>{DS_REF_TOKEN})"
            r"(?:\s+(?P<device>[A-Za-z_]\w*))?\s*)"
            rf"\(\s*(?P<paths>{_QUOTED}(?:{_LIST_SEP}{_QUOTED})*+)\s*\)",
            re.IGNORECASE,
        ),
        many=True,
        concat=True,
    ),
    # infile '<path>' [<device>] / file '<path>' [<device>] — the unquoted
    # form names a fileref a FILENAME already declared, so there is no path
    # here to record. A device type turns the value into what it addresses:
    # `infile 'gunzip -c /data/x.gz' pipe;` reads a command's output.
    PathSpec(
        statement="infile",
        keyword="infile",
        pattern=re.compile(r"(?P<head>\binfile\s+)" + _VALUE + _DEVICE_AFTER, re.IGNORECASE),
    ),
    # infile '("<path>" "<path>")': files read one after another, the list
    # inside one quoted string (Concatenating Filenames, as above).
    PathSpec(
        statement="infile",
        keyword="infile",
        pattern=re.compile(
            r"(?P<head>\binfile\s+)(?P<paths>"
            rf"""'\(\s*"[^"\n]*"(?:{_LIST_SEP}"[^"\n]*")*+\s*\)'"""
            rf"""|"\(\s*'[^'\n]*'(?:{_LIST_SEP}'[^'\n]*')*+\s*\)")""",
            re.IGNORECASE,
        ),
        many=True,
        concat=True,
    ),
    PathSpec(
        statement="file",
        keyword="file",
        pattern=re.compile(r"(?P<head>\bfile\s+)" + _VALUE + _DEVICE_AFTER, re.IGNORECASE),
    ),
    # %include <source> ...: quoted files ('/a.sas', or '("/a.sas" "/b.sas")'),
    # whole files filerefs name (src), and members of the directory one names
    # (src(one, "two.txt")), mixed as the statement likes — "you can mix the
    # types of included sources" (SAS 9.4 Global Statements, %INCLUDE
    # Statement) — one reference per file, in order. Internal lines (1-5) and
    # the keyboard (*) name no file. %INC is SAS's own abbreviation, and the
    # keyword gate is "inc" for both.
    PathSpec(
        statement="include",
        keyword="inc",
        pattern=re.compile(
            rf"(?P<head>{_INCLUDE})(?P<paths>(?:\s*+{_SOURCE})++)(?=\s*(?:[;/]|\Z))",
            re.IGNORECASE,
        ),
        many=True,
        sources=True,
    ),
    # infile in / file out [(member)]: the file a fileref names.
    PathSpec(
        statement="infile",
        keyword="infile",
        pattern=re.compile(
            rf"\binfile\s+{_RESERVED_FREFS}(?P<binds>{_FREF})\b"
            r"(?:\s*\((?P<members>[^)]*)\))?",
            re.IGNORECASE,
        ),
        fileref=True,
        at_statement=True,
    ),
    PathSpec(
        statement="file",
        keyword="file",
        pattern=re.compile(
            rf"\bfile\s+{_RESERVED_FREFS}(?P<binds>{_FREF})\b"
            r"(?:\s*\((?P<members>[^)]*)\))?",
            re.IGNORECASE,
        ),
        fileref=True,
        at_statement=True,
    ),
    # PROC IMPORT datafile='<path>' / PROC EXPORT outfile='<path>'. Keyed on the
    # option rather than the PROC: the option is what carries the path, and it
    # is unambiguous on its own.
    PathSpec(
        statement="proc_import",
        keyword="datafile",
        pattern=re.compile(r"(?P<head>\bdatafile\s*=\s*)" + _VALUE, re.IGNORECASE),
    ),
    PathSpec(
        statement="proc_export",
        keyword="outfile",
        pattern=re.compile(r"(?P<head>\boutfile\s*=\s*)" + _VALUE, re.IGNORECASE),
    ),
    # datafile=<fileref> / outfile=<fileref>: "the complete path and filename
    # or fileref" (SAS 9.4 Procedures Guide, IMPORT and EXPORT). A name
    # running on into a path (`datafile=sales.csv`) is no fileref.
    PathSpec(
        statement="proc_import",
        keyword="datafile",
        pattern=re.compile(
            rf"\bdatafile\s*=\s*(?P<binds>{_FREF})(?![\w.&\\/:])", re.IGNORECASE
        ),
        fileref=True,
    ),
    PathSpec(
        statement="proc_export",
        keyword="outfile",
        pattern=re.compile(
            rf"\boutfile\s*=\s*(?P<binds>{_FREF})(?![\w.&\\/:])", re.IGNORECASE
        ),
        fileref=True,
    ),
    # PROC CIMPORT infile='<path>' / infile=<fileref>: the transport file it
    # imports. Anchored on the PROC, since INFILE= alone is common.
    PathSpec(
        statement="proc_cimport",
        keyword="cimport",
        pattern=re.compile(
            r"(?P<head>\bcimport\b[^;]*?\binfile\s*=\s*)" + _VALUE, re.IGNORECASE
        ),
    ),
    PathSpec(
        statement="proc_cimport",
        keyword="cimport",
        pattern=re.compile(
            rf"\bcimport\b[^;]*?\binfile\s*=\s*(?P<binds>{_FREF})\b", re.IGNORECASE
        ),
        fileref=True,
    ),
    # PROC PYTHON infile='<path>' / infile=<fileref>: "a source file or
    # fileref that contains Python statements to run" (SAS 9.4 Procedures
    # Guide, PROC PYTHON Statement) — a script the job depends on, as an
    # %INCLUDE's is. Anchored on the PROC: PROC LUA's INFILE= names a script
    # on its LUAPATH, not a file.
    PathSpec(
        statement="proc_python",
        keyword="python",
        pattern=re.compile(
            r"(?P<head>\bproc\s+python\b[^;]*?\binfile\s*=\s*)" + _VALUE, re.IGNORECASE
        ),
    ),
    PathSpec(
        statement="proc_python",
        keyword="python",
        pattern=re.compile(
            rf"\bproc\s+python\b[^;]*?\binfile\s*=\s*(?P<binds>{_FREF})\b", re.IGNORECASE
        ),
        fileref=True,
    ),
    # PROC PRINTTO log='<path>' / print='<path>': where a job sends its log and
    # its output. Anchored on the PROC, since LOG= and PRINT= alone are too
    # common as words; one spec per option, so a statement naming both yields
    # both.
    PathSpec(
        statement="printto",
        keyword="printto",
        pattern=re.compile(
            r"(?P<head>\bprintto\b[^;]*?\blog\s*=\s*)" + _VALUE, re.IGNORECASE
        ),
    ),
    PathSpec(
        statement="printto",
        keyword="printto",
        pattern=re.compile(
            r"(?P<head>\bprintto\b[^;]*?\bprint\s*=\s*)" + _VALUE, re.IGNORECASE
        ),
    ),
    # ODS destinations: file='<path>' / path='<path>'. The \b keeps this off
    # ``outfile=``, whose 'file' has no word boundary before it.
    PathSpec(
        statement="ods",
        keyword="file=",
        pattern=re.compile(r"(?P<head>\bfile\s*=\s*)" + _VALUE, re.IGNORECASE),
    ),
    PathSpec(
        statement="ods",
        keyword="path=",
        pattern=re.compile(r"(?P<head>\bpath\s*=\s*)" + _VALUE, re.IGNORECASE),
    ),
    # `x setenv NAME /path;` (the SAS setenv command, quoted or not) and
    # `options set=NAME '/path';` define an environment variable. On UNIX one
    # whose name is upper-case and whose value is a full path serves as a
    # libref (a directory) or a fileref (a file) wherever no LIBNAME or
    # FILENAME assigns that name (SAS 9.4 Companion for UNIX Environments,
    # Using Environment Variables as Librefs / to Assign Filerefs). `binds` is
    # the name, lowercased as a libref is.
    PathSpec(
        statement="setenv",
        keyword="setenv",
        pattern=re.compile(
            r"(?P<head>(?i:\bx\s+['\"]?\s*setenv)\s+(?P<binds>[A-Z_][A-Z0-9_]*)\s+)"
            r"(?P<path>/[^\s;'\"]*)"
        ),
    ),
    PathSpec(
        statement="setenv",
        keyword="options",
        pattern=re.compile(
            r"(?P<head>(?i:\boptions\b[^;]*?\bset)\s*=\s*\(?\s*"
            r"(?P<binds>[A-Z_][A-Z0-9_]*)\s*=?\s*)(?P<q>['\"])(?P<path>/[^'\"\n]*)(?P=q)"
        ),
    ),
    # SAS's own cd command, which "changes the current directory of your SAS
    # session": relative file names are looked for there from then on. Only
    # as the one command of an X statement, CALL SYSTEM or %SYSEXEC — among
    # several (`x 'cd /a; ls';`) it runs in a child shell and changes nothing
    # (SAS 9.4 Companion for UNIX Environments, Executing Operating System
    # Commands from Your SAS Session).
    PathSpec(
        statement="cd",
        keyword="cd",
        pattern=re.compile(
            r"(?P<head>(?i:\bx\s+(?P<q>['\"]?)\s*cd)\s+)(?P<path>[^\s;'\"]+)"
            r"\s*(?P=q)\s*(?=;|\Z)"
        ),
        at_statement=True,
    ),
    PathSpec(
        statement="cd",
        keyword="system",
        pattern=re.compile(
            r"(?P<head>(?i:\bcall\s+system\s*\(\s*)(?P<q>['\"])\s*(?i:cd)\s+)"
            r"(?P<path>[^\s;'\"]+)\s*(?P=q)\s*\)"
        ),
        at_statement=True,
    ),
    PathSpec(
        statement="cd",
        keyword="sysexec",
        pattern=re.compile(
            r"(?P<head>(?i:%\s*sysexec\s+cd)\s+)(?P<path>[^\s;'\"]+)\s*(?=;|\Z)"
        ),
        at_statement=True,
    ),
    # options sasautos='<path>' — see the module docstring on the list form.
    PathSpec(
        statement="sasautos",
        keyword="sasautos",
        pattern=re.compile(
            r"(?P<head>\bsasautos\s*=\s*\(?\s*)" + _VALUE, re.IGNORECASE
        ),
    ),
)


def extract_paths(text: str) -> list[SasPathRef]:
    """Every external reference *text* names, in order of first appearance.

    *text* must be the **comments-blanked, strings-intact** form (``cf`` in
    :func:`chunker.metadata._metadata_for`): the values live inside string
    literals, which the fully sanitised form has blanked, and a path written
    inside a comment is not a path the job uses.

    Duplicates are dropped — a chunk naming one file twice has one dependency on
    it — keeping the first occurrence so the order stays the reading order.
    """
    if not text:
        return []
    lowered = text.lower()
    seen: set[SasPathRef] = set()
    refs: list[SasPathRef] = []
    for spec in PATH_STATEMENTS:
        if spec.keyword not in lowered:
            continue
        for match in spec.pattern.finditer(text):
            if spec.at_statement and not _at_statement(text, match.start()):
                continue
            for ref in spec.refs_for(match):
                if ref not in seen:
                    seen.add(ref)
                    refs.append(ref)
    if refs and logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"extract_paths: {len(refs)} external reference(s)")
    return refs


# ---------------------------------------------------------------------------
# Database-engine LIBNAMEs
# ---------------------------------------------------------------------------

#: LIBNAME engines that reach a *database*, not a directory. Lowercased.
#:
#: The engine names SAS/ACCESS 9.4 documents, each from its LIBNAME page
#: (``LIBNAME libref snow``, ``LIBNAME libref sapase``, …): the relational
#: interfaces, the previous names SAS still accepts (``sybase`` for SAP ASE,
#: ``sybaseiq`` for SAP IQ), and MongoDB and Salesforce from SAS/ACCESS for
#: Nonrelational Databases. Snowflake's engine is ``snow``, not ``snowflake``.
#: Not every one runs on every host — Snowflake, Spark, BigQuery and MongoDB
#: run on Linux but not AIX, OLE DB only on Windows — but a name is recognised
#: wherever it appears.
#:
#: ``spde`` is deliberately absent: the SPD Engine takes a quoted directory, so
#: it is a path LIBNAME that :data:`PATH_STATEMENTS` already matches, and it is
#: told apart from a plain one by :attr:`~chunker.models.SasPathRef.engine`.
#: An engine here takes no path at all — its location is the connection.
ENGINE_LIBNAMES: frozenset[str] = frozenset(
    {
        "aster",
        "bigquery",
        "db2",
        "greenplm",
        "hadoop",
        "hawq",
        "impala",
        "informix",
        "jdbc",
        "mongo",
        "mysql",
        "netezza",
        "odbc",
        "oledb",
        "oracle",
        "postgres",
        "redshift",
        "sapase",
        "saphana",
        "sapiq",
        "sforce",
        "snow",
        "spark",
        "sqlsvr",
        "sybase",
        "sybaseiq",
        "teradata",
        "vertica",
        "ybrick",
    }
)

# libname <libref> <engine> <options...>;
#
# The option tail spans newlines by design — a connection statement with a
# dozen options is routinely wrapped across lines, and the terminator is the
# semicolon, not the line end. A quoted value is one unit, since a connection
# string holds semicolons of its own: noprompt="DSN=dw;UID=u;". The repeat is
# possessive: a quote also matches [^;] (an unmatched one, as in O'Neil, must),
# and with no ';' to end on, giving characters back would try every way of
# splitting the quotes — time exponential in their number. In a CREATE VIEW's
# USING list a clause also ends at the comma before the next LIBNAME.
_ENGINE_LIBNAME_RE = re.compile(
    rf"\blibname\s+(?P<binds>{DS_REF_TOKEN})\s+(?P<engine>[A-Za-z_]\w*)\b"
    r"""(?P<opts>(?:"[^"\n]*"|'[^'\n]*'|(?!,\s*libname\b)[^;])*+)"""
    r"(?:;|(?=,\s*libname\b))",
    re.IGNORECASE,
)

# One ``key=value`` option. The value alternatives are ordered longest-form
# first: a quoted value may contain spaces, a parenthesised one may contain
# both, and only the bare form stops at whitespace.
_ENGINE_OPTION_RE = re.compile(
    r"(?P<key>[A-Za-z_]\w*)\s*=\s*"
    r"(?P<value>\"[^\"\n]*\"|'[^'\n]*'|\([^)]*\)|[^\s;]+)"
)


def _option_value(raw: str) -> str:
    """*raw* with one layer of surrounding quotes removed, else unchanged.

    Parentheses are kept: ``connection=(a b)`` means something different from
    ``connection=a b``, and no consumer of this can reconstruct the difference
    once the brackets are gone.
    """
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'":
        return raw[1:-1]
    return raw


def extract_engine_refs(text: str) -> list[SasEngineRef]:
    """Every database-engine ``LIBNAME`` *text* declares, in reading order.

    *text* must be the **comments-blanked, strings-intact** form, the same input
    :func:`extract_paths` takes and for the same reason: option values live
    inside string literals that the fully sanitised form has already blanked.

    A LIBNAME whose second token is not in :data:`ENGINE_LIBNAMES` is skipped —
    that covers path libraries, ``libname x clear;``, and ``libname _all_ list;``
    in one test, without this scan having to know what any of them are.

    Duplicates are dropped, keeping the first occurrence, exactly as
    :func:`extract_paths` does: a chunk declaring one connection twice has one
    dependency on it.
    """
    if not text or "libname" not in text.lower():
        return []
    seen: set[SasEngineRef] = set()
    refs: list[SasEngineRef] = []
    for match in _ENGINE_LIBNAME_RE.finditer(text):
        engine = match.group("engine").lower()
        if engine not in ENGINE_LIBNAMES:
            continue
        options = tuple(
            (m.group("key").lower(), _option_value(m.group("value")))
            for m in _ENGINE_OPTION_RE.finditer(match.group("opts"))
        )
        ref = SasEngineRef(
            engine=engine,
            binds=match.group("binds").lower(),
            options=options,
            # The whole statement, not just the values: an engine keyword can
            # itself arrive through a macro reference.
            has_macro_ref="&" in match.group(0),
            raw=" ".join(match.group(0).split()),
            statement=(
                "using_libname" if in_using_clause(text, match.start()) else "libname"
            ),
        )
        if ref not in seen:
            seen.add(ref)
            refs.append(ref)
    if refs and logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"extract_engine_refs: {len(refs)} engine LIBNAME(s)")
    return refs
