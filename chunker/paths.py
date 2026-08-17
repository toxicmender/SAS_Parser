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
:mod:`xref.pre` imports it. The match groups are named ``head`` / ``q`` / ``path``
because that is the shape its substitution helper is built on: rewriting is
``head + quote + new_path + quote``.

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

Known limit: ``options sasautos=('/a' '/b')`` reports only the first entry. The
concatenation form needs a scan the ``head``/``path`` substitution shape cannot
express, and reporting the first is better than reporting none.

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

    The same normalisation :func:`chunker.metadata._quoted_path` applies to
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
    """

    statement: str
    keyword: str
    pattern: re.Pattern[str]

    def location_for(self, match: re.Match[str]) -> PathLocation:
        """Where this match points, from its device group if it has one."""
        return classify_location(_group(match, "device"))

    def ref_for(self, match: re.Match[str]) -> SasPathRef | None:
        """*match* as a :class:`~chunker.models.SasPathRef`, or ``None``.

        ``None`` when the quoted value is empty — ``infile '';`` names nothing,
        and an empty entry in an inventory is worse than no entry.
        """
        raw = match.group("path").strip()
        if not raw:
            return None
        device = _group(match, "device")
        engine = _group(match, "engine")
        return SasPathRef(
            statement=self.statement,
            location=classify_location(device),
            path=normalise_path(raw),
            raw=raw,
            binds=(b.lower() if (b := _group(match, "binds")) else None),
            device=(device.lower() if device else None),
            engine=(engine.lower() if engine else None),
            has_macro_ref="&" in raw,
        )


def _group(match: re.Match[str], name: str) -> str | None:
    """*match*'s *name* group, or ``None`` when the pattern has no such group.

    ``match.group(name)`` raises for a name the pattern never declared, and the
    specs deliberately declare only the groups their statement actually has —
    ``groupdict`` answers both questions with one lookup.
    """
    return match.groupdict().get(name)


# The quoted-value tail every spec ends with: an opening quote, the value, and
# the same quote again. A SAS statement's quoted literal does not span lines.
_VALUE = r"(?P<q>['\"])(?P<path>[^'\"\n]*)(?P=q)"

#: Every statement form recognised, in scan order. Iterated by
#: :func:`extract_paths` and by :func:`xref.pre.rewrite_source_text`.
PATH_STATEMENTS: tuple[PathSpec, ...] = (
    # libname <libref> [<engine>] '<path>' — the engine group is *captured*
    # because an engine changes what the directory is (see SasPathRef.engine):
    # `libname x spde '/p'` is a partitioned SPD Engine library, not the plain
    # directory `libname x '/p'` names.
    PathSpec(
        statement="libname",
        keyword="libname",
        pattern=re.compile(
            r"(?P<head>\blibname\s+(?P<binds>[A-Za-z_]\w*)\s+"
            r"(?:(?P<engine>[A-Za-z_]\w*)\s+)?)" + _VALUE,
            re.IGNORECASE,
        ),
    ),
    # filename <fileref> [<device>] '<path>'
    PathSpec(
        statement="filename",
        keyword="filename",
        pattern=re.compile(
            r"(?P<head>\bfilename\s+(?P<binds>[A-Za-z_]\w*)\s+"
            r"(?:(?P<device>[A-Za-z_]\w*)\s+)?)" + _VALUE,
            re.IGNORECASE,
        ),
    ),
    # infile '<path>' / file '<path>' — the unquoted form names a fileref a
    # FILENAME already declared, so there is no path here to record.
    PathSpec(
        statement="infile",
        keyword="infile",
        pattern=re.compile(r"(?P<head>\binfile\s+)" + _VALUE, re.IGNORECASE),
    ),
    PathSpec(
        statement="file",
        keyword="file",
        pattern=re.compile(r"(?P<head>\bfile\s+)" + _VALUE, re.IGNORECASE),
    ),
    # %include '<path>'
    PathSpec(
        statement="include",
        keyword="include",
        pattern=re.compile(r"(?P<head>%\s*include\s+)" + _VALUE, re.IGNORECASE),
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
            ref = spec.ref_for(match)
            if ref is not None and ref not in seen:
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
#: ``spde`` is deliberately absent: the SPD Engine takes a quoted directory, so
#: it is a path LIBNAME that :data:`PATH_STATEMENTS` already matches, and it is
#: told apart from a plain one by :attr:`~chunker.models.SasPathRef.engine`.
#: An engine here takes no path at all — its location is the connection.
ENGINE_LIBNAMES: frozenset[str] = frozenset(
    {
        "oracle",
        "odbc",
        "teradata",
        "sqlsvr",
        "oledb",
        "db2",
        "postgres",
        "mysql",
        "netezza",
        "snowflake",
        "redshift",
        "saphana",
        "sybase",
        "informix",
        "hadoop",
        "spark",
        "bigquery",
    }
)

# libname <libref> <engine> <options...>;
#
# The option tail is ``[^;]*``, which spans newlines by design — a connection
# statement with a dozen options is routinely wrapped across lines, and the
# terminator is the semicolon, not the line end.
_ENGINE_LIBNAME_RE = re.compile(
    r"\blibname\s+(?P<binds>[A-Za-z_]\w*)\s+(?P<engine>[A-Za-z_]\w*)\b(?P<opts>[^;]*);",
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
        )
        if ref not in seen:
            seen.add(ref)
            refs.append(ref)
    if refs and logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"extract_engine_refs: {len(refs)} engine LIBNAME(s)")
    return refs
