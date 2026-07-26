"""Cross-file reference resolution for complexity analysis.
See complexity/README.md.

A SAS script is rarely self-contained. It invokes macros defined in a shared
utility file, reads datasets another job wrote, and depends on librefs assigned
somewhere else entirely. None of that shows up when a file is scored on its own
contents, yet it is exactly what makes a migration hard: a file you cannot
translate without also having three others in front of you is more work than
its own text suggests.

This module resolves every outward reference a chunk makes against the rest of
the corpus and classifies it into one of three states:

- **internal** — satisfied within the same file. Not a cross-file reference at
  all, so it raises no signal.
- **import / export** — satisfied by *another file in the corpus*. A real
  dependency edge, in one direction or the other.
- **unresolved** — satisfied by no file in scope.

The third state is only assertable when the corpus holds more than one file.
With a single file there is nothing that could have been searched, so the same
reference is reported as **external** rather than unresolved, at a materially
lower rating. Claiming "this macro exists nowhere" on the strength of having
looked at one file would be an unfounded assertion, and it is the difference
between MEDIUM/PARTIAL and HIGH/MANUAL.

Everything here is derived from metadata the chunker already extracts —
``defines_macros`` / ``invokes_macros``, ``output_datasets`` /
``input_datasets``, ``produces_macrovars`` / ``consumes_macrovars``,
``defines_librefs``, and ``includes``. No new source scanning.

Two exclusion sets are reused from the chunker rather than re-listed, on the
same reasoning as :func:`chunker.scanner._sanitise` in
:mod:`complexity.detectors` — a second copy would drift:
``_STANDARD_AUTOCALL_MACROS`` (``%left``, ``%trim``, ... ship with SAS, so
calling one is never a missing dependency) and ``_DEFAULT_LIBREFS`` (``work``,
``sashelp``, ... are always assigned).

Logger name: ``complexity.crossfile``.
"""

from __future__ import annotations

import logging
from typing import Iterable, NamedTuple

from chunker.batcher import _DEFAULT_LIBREFS
from chunker.keywords import _STANDARD_AUTOCALL_MACROS
from chunker.models import SasChunk

from .models import CrossFileProfile

logger = logging.getLogger(__name__)

_INLINE = "<inline>"

# Construct names this module can emit. Every one needs a matching entry under
# a profile's "cross_file" namespace; a test asserts the correspondence, the
# same way one does for the detector names.
CROSS_FILE_CONSTRUCTS: frozenset[str] = frozenset(
    {
        "macro_import",
        "macro_export",
        "macro_unresolved",
        "macro_external",
        "dataset_import",
        "dataset_export",
        "dataset_unresolved",
        "macrovar_import",
        "macrovar_export",
        "libref_import",
        "libref_unresolved",
    }
)

# The subset that means "we could not find this at all". These feed the
# uncertainty dimension of the T-shirt size, because an unknown is risk rather
# than volume. A dataset read from nowhere is deliberately *not* here: an
# unmatched input dataset is usually a legitimate source table, not a gap.
UNRESOLVED_CONSTRUCTS: frozenset[str] = frozenset(
    {"macro_unresolved", "libref_unresolved"}
)


class CrossFileRef(NamedTuple):
    """One resolved outward reference.

    ``name`` is the catalogue key looked up under a profile's ``cross_file``
    namespace; ``evidence`` names what was referenced and where it was found;
    ``peer`` is the other file involved, or ``""`` when unresolved.
    """

    name: str
    evidence: str
    peer: str = ""


def _libref_of(dataset: str) -> str | None:
    """The libref qualifying *dataset*, or ``None`` if it has none.

    Mirrors the batcher's rule: a quoted physical path addresses a file
    directly rather than a library member, and an unqualified name has no
    libref to resolve.
    """
    if dataset.startswith("'") or dataset.startswith('"') or "." not in dataset:
        return None
    return dataset.split(".", 1)[0].lower()


def _owners(index: dict[str, set[str]], name: str, exclude: str) -> list[str]:
    """Files other than *exclude* that produce *name*, in sorted order."""
    return sorted(index.get(name.lower(), set()) - {exclude})


class CrossFileIndex:
    """Producer/consumer index over a whole corpus of chunks.

    Built once from every chunk in scope, then queried per chunk. All
    resolution happens in :meth:`build`, so :meth:`refs_for` and
    :meth:`profile_for` are pure lookups — re-analysing a chunk can never
    double-count a reference into a file profile.

    A single-file corpus still produces a usable index: imports and exports are
    vacuously empty, and unresolved references are reported as *external*
    instead (see the module docstring).
    """

    def __init__(self) -> None:
        self.corpus_files: int = 0
        # Keyed on (source_id, chunk_id), never chunk_id alone: the chunker
        # numbers chunks per file, so two files independently chunked both
        # start at "chunk-001" and would otherwise collide, silently serving
        # one file's references for another's.
        self._refs: dict[tuple[str, str], list[CrossFileRef]] = {}
        self._profiles: dict[str, CrossFileProfile] = {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def build(cls, chunks: Iterable[SasChunk]) -> "CrossFileIndex":
        """Index *chunks* and resolve every cross-file reference among them."""
        index = cls()
        chunk_list = list(chunks)

        sources: list[str] = []
        # Producers: name -> the files that create it.
        macros: dict[str, set[str]] = {}
        datasets: dict[str, set[str]] = {}
        macrovars: dict[str, set[str]] = {}
        librefs: dict[str, set[str]] = {}
        # Consumers: name -> the files that read it. Built in the same pass so
        # the export directions are a lookup rather than a rescan per name.
        macro_users: dict[str, set[str]] = {}
        dataset_users: dict[str, set[str]] = {}
        macrovar_users: dict[str, set[str]] = {}

        def _record(store: dict[str, set[str]], name: str, source: str) -> None:
            store.setdefault(name.lower(), set()).add(source)

        for chunk in chunk_list:
            source = chunk.source_id or _INLINE
            if source not in sources:
                sources.append(source)
            meta = chunk.metadata
            for macro in meta.defines_macros:
                _record(macros, macro, source)
            for dataset in meta.output_datasets:
                _record(datasets, dataset, source)
            # A macro definition's literal body outputs are produced by this
            # file too, once the macro is called (the batcher takes the same
            # view when computing a batch's outputs).
            for dataset in meta.body_literal_outputs:
                _record(datasets, dataset, source)
            for var in (*meta.produces_macrovars, *meta.declared_macro_vars):
                _record(macrovars, var, source)
            for libref in meta.defines_librefs:
                _record(librefs, libref, source)

            for macro in meta.invokes_macros:
                _record(macro_users, macro, source)
            for dataset in (*meta.input_datasets, *meta.body_literal_inputs):
                _record(dataset_users, dataset, source)
            for var in meta.consumes_macrovars:
                _record(macrovar_users, var, source)

        index.corpus_files = len(sources)
        multi = index.corpus_files > 1
        by_source: dict[str, list[CrossFileRef]] = {s: [] for s in sources}

        for chunk in chunk_list:
            source = chunk.source_id or _INLINE
            refs = index._resolve_chunk(
                chunk,
                source,
                producers=_Producers(macros, datasets, macrovars, librefs),
                consumers=_Consumers(macro_users, dataset_users, macrovar_users),
                multi=multi,
            )
            if refs:
                index._refs[(source, chunk.chunk_id)] = refs
                by_source[source].extend(refs)

        index._profiles = {
            source: _build_profile(source, by_source[source], index.corpus_files)
            for source in sources
        }

        if index._refs:
            logger.info(
                f"CrossFileIndex  files={index.corpus_files}  "
                f"chunks_with_refs={len(index._refs)}  "
                f"refs={sum(len(r) for r in index._refs.values())}"
            )
        return index

    def _resolve_chunk(
        self,
        chunk: SasChunk,
        source: str,
        *,
        producers: "_Producers",
        consumers: "_Consumers",
        multi: bool,
    ) -> list[CrossFileRef]:
        """Every cross-file reference *chunk* makes, in a stable order."""
        meta = chunk.metadata
        refs: list[CrossFileRef] = []
        macros = producers.macros
        datasets = producers.datasets
        macrovars = producers.macrovars
        librefs = producers.librefs

        # ── macros invoked here but defined elsewhere ────────────────────────
        for macro in meta.invokes_macros:
            key = macro.lower()
            if key in _STANDARD_AUTOCALL_MACROS:
                continue  # ships with SAS; never a missing dependency
            if source in macros.get(key, set()):
                continue  # defined in this same file
            peers = _owners(macros, key, source)
            if peers:
                refs.append(
                    CrossFileRef(
                        "macro_import",
                        f"%{macro} defined in {', '.join(peers)}",
                        peers[0],
                    )
                )
            elif multi:
                refs.append(
                    CrossFileRef(
                        "macro_unresolved",
                        f"%{macro} — no %MACRO definition in any of the "
                        f"{self.corpus_files} files analysed",
                    )
                )
            else:
                refs.append(
                    CrossFileRef(
                        "macro_external",
                        f"%{macro} defined outside this file "
                        f"(no other files were in scope to search)",
                    )
                )

        # ── macros defined here and used elsewhere ───────────────────────────
        for macro in meta.defines_macros:
            users = _owners(consumers.macros, macro, source)
            if users:
                refs.append(
                    CrossFileRef(
                        "macro_export",
                        f"%{macro} used by {', '.join(users)}",
                        users[0],
                    )
                )

        # ── datasets read here but written elsewhere ─────────────────────────
        for dataset in meta.input_datasets:
            key = dataset.lower()
            if source in datasets.get(key, set()):
                continue
            peers = _owners(datasets, key, source)
            if peers:
                refs.append(
                    CrossFileRef(
                        "dataset_import",
                        f"{dataset} written by {', '.join(peers)}",
                        peers[0],
                    )
                )
            elif multi:
                refs.append(
                    CrossFileRef(
                        "dataset_unresolved",
                        f"{dataset} not written by any analysed file — "
                        f"assumed to be an existing source table",
                    )
                )

        # ── datasets written here and read elsewhere ─────────────────────────
        for dataset in meta.output_datasets:
            users = _owners(consumers.datasets, dataset, source)
            if users:
                refs.append(
                    CrossFileRef(
                        "dataset_export",
                        f"{dataset} read by {', '.join(users)}",
                        users[0],
                    )
                )

        # ── macro variables consumed here but set elsewhere ──────────────────
        for var in meta.consumes_macrovars:
            key = var.lower()
            if source in macrovars.get(key, set()):
                continue
            peers = _owners(macrovars, key, source)
            if peers:
                refs.append(
                    CrossFileRef(
                        "macrovar_import",
                        f"&{var} set in {', '.join(peers)}",
                        peers[0],
                    )
                )

        # ── macro variables set here and read elsewhere ──────────────────────
        seen_exported_vars: set[str] = set()
        for var in (*meta.produces_macrovars, *meta.declared_macro_vars):
            if var.lower() in seen_exported_vars:
                continue
            seen_exported_vars.add(var.lower())
            users = _owners(consumers.macrovars, var, source)
            if users:
                refs.append(
                    CrossFileRef(
                        "macrovar_export",
                        f"&{var} read by {', '.join(users)}",
                        users[0],
                    )
                )

        # ── librefs used here but assigned elsewhere ─────────────────────────
        own_librefs = {lr.lower() for lr in meta.defines_librefs}
        seen_librefs: set[str] = set()
        for dataset in (
            *meta.input_datasets,
            *meta.output_datasets,
            *meta.body_literal_inputs,
            *meta.body_literal_outputs,
        ):
            libref = _libref_of(dataset)
            if (
                libref is None
                or libref in _DEFAULT_LIBREFS
                or libref in own_librefs
                or libref in seen_librefs
                # Assigned by a LIBNAME elsewhere in this same file. The
                # assignment is almost always its own chunk, so checking only
                # the current chunk's defines_librefs would report every
                # ordinary libref as unresolved.
                or source in librefs.get(libref, set())
            ):
                continue
            seen_librefs.add(libref)
            peers = _owners(librefs, libref, source)
            if peers:
                refs.append(
                    CrossFileRef(
                        "libref_import",
                        f"libref {libref} assigned in {', '.join(peers)}",
                        peers[0],
                    )
                )
            else:
                refs.append(
                    CrossFileRef(
                        "libref_unresolved",
                        f"libref {libref} has no LIBNAME in any analysed file — "
                        f"physical location unknown",
                    )
                )

        # %INCLUDE is deliberately not emitted here. It is the one cross-file
        # reference the chunker already surfaces as a metadata flag, and the
        # catalogue rates that flag MEDIUM/PARTIAL — exactly what this module
        # would assign. Emitting a second signal for the same construct would
        # double its contribution to the score without adding information.

        return refs

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def refs_for(self, chunk: SasChunk) -> list[CrossFileRef]:
        """Cross-file references made by *chunk*; empty when it makes none."""
        return self._refs.get(((chunk.source_id or _INLINE), chunk.chunk_id), [])

    def profile_for(self, source_id: str) -> CrossFileProfile | None:
        """The coupling profile for *source_id*, or ``None`` if not indexed."""
        return self._profiles.get(source_id)

    @property
    def profiles(self) -> dict[str, CrossFileProfile]:
        """Every file's coupling profile, keyed by source id."""
        return dict(self._profiles)

    def __str__(self) -> str:
        return (
            f"CrossFileIndex(files={self.corpus_files}, "
            f"chunks_with_refs={len(self._refs)})"
        )


class _Producers(NamedTuple):
    """Which files create each macro, dataset, macro variable, and libref."""

    macros: dict[str, set[str]]
    datasets: dict[str, set[str]]
    macrovars: dict[str, set[str]]
    librefs: dict[str, set[str]]


class _Consumers(NamedTuple):
    """Which files read each macro, dataset, and macro variable."""

    macros: dict[str, set[str]]
    datasets: dict[str, set[str]]
    macrovars: dict[str, set[str]]


def _build_profile(
    source: str, refs: list[CrossFileRef], corpus_files: int
) -> CrossFileProfile:
    """Fold one file's references into a :class:`CrossFileProfile`."""
    depends_on: list[str] = []
    depended_on_by: list[str] = []
    imports: list[str] = []
    exports: list[str] = []
    unresolved: list[str] = []

    for ref in refs:
        if ref.name in UNRESOLVED_CONSTRUCTS:
            if ref.evidence not in unresolved:
                unresolved.append(ref.evidence)
            continue
        if ref.name.endswith("_export"):
            if ref.evidence not in exports:
                exports.append(ref.evidence)
            if ref.peer and ref.peer not in depended_on_by:
                depended_on_by.append(ref.peer)
        elif ref.name.endswith("_import"):
            if ref.evidence not in imports:
                imports.append(ref.evidence)
            if ref.peer and ref.peer not in depends_on:
                depends_on.append(ref.peer)

    return CrossFileProfile(
        source_id=source,
        depends_on=sorted(depends_on),
        depended_on_by=sorted(depended_on_by),
        imports=imports,
        exports=exports,
        unresolved=unresolved,
        corpus_files=corpus_files,
    )
