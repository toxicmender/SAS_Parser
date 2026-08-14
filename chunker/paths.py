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

Logger name: ``chunker.paths``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from .models import PathLocation, SasPathRef

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
        return SasPathRef(
            statement=self.statement,
            location=classify_location(device),
            path=normalise_path(raw),
            raw=raw,
            binds=(b.lower() if (b := _group(match, "binds")) else None),
            device=(device.lower() if device else None),
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
    # libname <libref> [<engine>] '<path>'
    PathSpec(
        statement="libname",
        keyword="libname",
        pattern=re.compile(
            r"(?P<head>\blibname\s+(?P<binds>[A-Za-z_]\w*)\s+"
            r"(?:[A-Za-z_]\w*\s+)?)" + _VALUE,
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
