"""When the XREF substitution happens: before conversion, after, or both.

``"pre"`` (the default)
    :func:`chunker.batcher.replace_dataset_names` over the batch result. The
    chunker has already worked out what is a dataset reference, so the
    substitution lands on known dataset names — in chunk metadata and in the
    ``%let`` values that carry one in source text — rather than on whatever a
    pattern match takes for a table.

    **"pre" has a second half, and it runs earlier.** Physical paths —
    ``LIBNAME`` / ``INFILE`` / ``%INCLUDE`` targets — are not dataset names,
    and :func:`chunker.batcher._map_ds` deliberately skips quoted literals, so
    nothing here reaches them. :func:`xref.pre.rewrite_source_text` does,
    over the **raw text before chunking**. A caller applying "pre" in full
    therefore does both:

    .. code-block:: python

        text, _ = xref.pre.rewrite_source_text(text, mappings)   # paths, pre-chunk
        pipeline = SasLLMPipeline(
            chunking=ChunkingSetup(databricks_mapping=mappings.dataset_mapping)
        )

    The second line is how :func:`apply_pre` reaches a real run: both batchers
    take ``databricks_mapping`` and apply it as a post-pass after grouping, so
    a pipeline constructed with it has already had this function's effect.
    :func:`apply_pre` is for a caller holding a :class:`SasBatchResult`
    directly.
``"post"``
    :mod:`xref.rewrite` over the generated code — **both halves too**: table
    names, and the physical paths that reached the output because no mapping
    row covered them when ``pre`` swept the source (or because the model wrote
    one of its own). Catches what the SAS-side extraction never saw, at the
    cost of having to parse model output.
``"both"``
    Runs each and reports the names only one of them reached
    (:func:`apply_both`). That difference is the evidence for choosing a
    permanent mode: names post found and pre did not are what escaped the
    SAS-side substitution.

:mod:`conversion.run` reads :func:`configured_mode` and honours all three; the
functions here are also usable directly by a caller driving the pieces itself.

Logger name: ``xref.apply``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import app_config

from .rewrite import (
    rewrite_python,
    rewrite_python_paths,
    rewrite_sql,
    rewrite_sql_paths,
)

if TYPE_CHECKING:  # avoid importing chunker at module scope
    from chunker.models import SasBatchResult

    from .sourcing import XrefMappings

logger = logging.getLogger(__name__)

APPLY_PRE = "pre"
APPLY_POST = "post"
APPLY_BOTH = "both"
APPLY_MODES = (APPLY_PRE, APPLY_POST, APPLY_BOTH)

DEFAULT_MODE = APPLY_PRE

def configured_mode() -> str:
    """
    When to apply the substitution: ``XREF_APPLY`` > ``config.json``
    ``xref.apply`` > ``"pre"``.

    An unrecognised value degrades to the default with a WARNING rather than
    raising — one bad config entry should not stop a conversion, and ``"pre"``
    is the mode that needs nothing installed.
    """
    import os

    configured = (
        os.environ.get("XREF_APPLY")
        or app_config.get_typed_value("xref", "apply", str, DEFAULT_MODE)
    )
    normalised = str(configured).strip().lower()
    if normalised not in APPLY_MODES:
        logger.warning(
            f"configured_mode: xref.apply {configured!r} is not one of "
            f"{'/'.join(APPLY_MODES)}; ignoring it ({DEFAULT_MODE!r} applies)"
        )
        return DEFAULT_MODE
    return normalised


def apply_pre(result: "SasBatchResult", mapping: dict[str, str]) -> "SasBatchResult":
    """
    *result* with SAS dataset names substituted, before conversion.

    Delegates entirely to :func:`chunker.batcher.replace_dataset_names`: the
    mapping's two key shapes (dotted = exact dataset, bare = libref prefix)
    are exactly what that function already classifies, so there is nothing to
    translate and ``chunker/batcher.py`` needs no change. Imported lazily so
    this package can be used without pulling the chunker in.
    """
    if not mapping:
        return result
    from chunker.batcher import replace_dataset_names

    return replace_dataset_names(result, mapping)


def apply_post(
    code: str,
    language: str,
    mappings: "XrefMappings",
    *,
    on_failure: str | None = None,
) -> str:
    """
    *code* with its table references **and** physical paths rewritten.

    Both halves, from the two slots ``pre`` reads at the other end:
    ``mappings.dataset_mapping`` for tables, ``mappings.by_path`` for paths. A
    ``post`` run that rewrote only tables would leave the paths ``pre`` did not
    reach exactly as the model wrote them, which is the gap this closes.

    *language* is resolved strictly before dispatch: Spark SQL uses sqlglot for
    tables, PySpark the ``ast`` module. Paths never need sqlglot — see
    :func:`~xref.rewrite.rewrite_sql_paths`.
    """
    if not mappings or not code.strip():
        return code
    from target_language import resolve_target_language

    datasets = mappings.dataset_mapping
    paths = mappings.by_path
    target = resolve_target_language(language)
    if target.sqlglot_dialect is not None:
        out = rewrite_sql(code, datasets, on_failure=on_failure)
        return rewrite_sql_paths(out, paths)
    out = rewrite_python(code, datasets, on_failure=on_failure)
    return rewrite_python_paths(out, paths, on_failure=on_failure)


@dataclass
class BothResult:
    """What ``"both"`` produced, and what the two modes disagreed about.

    Attributes
    ----------
    code : str
        The post-rewritten code. ``"both"`` applies pre *and* post, so this
        is the output either way.
    result : SasBatchResult | None
        The batch result after the pre pass, when one was given.
    pre_applied : bool
        Whether the pre pass had a batch result to work on.
    post_changed : bool
        Whether the post pass changed anything the pre pass had not already
        handled — the signal worth acting on.
    only_post : list[str]
        Source names — dataset *and* path — the post pass rewrote that the pre
        pass had left alone. A non-empty list means the mapping escaped the
        SAS-side extraction, which is precisely the evidence for keeping
        ``"post"``.
    """

    code: str
    result: "SasBatchResult | None" = None
    pre_applied: bool = False
    post_changed: bool = False
    only_post: list[str] = field(default_factory=list)


def apply_both(
    code: str,
    language: str,
    mappings: "XrefMappings",
    *,
    pre_code: str | None = None,
    on_failure: str | None = None,
) -> BothResult:
    """
    Run the post pass over *code* and report what it caught.

    *pre_code* is the same code as it stood after the pre pass, when the
    caller has it; the difference between the two is what only post reached.
    Without it, every post rewrite is reported, which over-reports rather than
    under-reports — the safe direction for a diagnostic.

    Both mapping namespaces are checked, since either can escape: a dataset
    name the metadata extraction missed, or a path with no XREF row when
    :func:`xref.pre.rewrite_source_text` swept the source.
    """
    baseline = pre_code if pre_code is not None else code
    rewritten = apply_post(code, language, mappings, on_failure=on_failure)
    checked = {**mappings.dataset_mapping, **mappings.by_path}
    only_post = [
        source
        for source, target in checked.items()
        if target in rewritten and target not in baseline
    ]
    if only_post:
        logger.warning(
            f"apply_both: {len(only_post)} mapping(s) were only reached by the "
            f"post pass ({', '.join(sorted(only_post)[:5])}"
            f"{', ...' if len(only_post) > 5 else ''}); those names escaped the "
            f"SAS-side substitution"
        )
    return BothResult(
        code=rewritten,
        pre_applied=pre_code is not None,
        post_changed=rewritten != code,
        only_post=sorted(only_post),
    )


def apply(
    mode: str | None,
    *,
    result: "SasBatchResult | None" = None,
    code: str | None = None,
    language: str | None = None,
    mappings: "XrefMappings",
    on_failure: str | None = None,
) -> Any:
    """
    Dispatch on *mode* (``None`` reads :func:`configured_mode`).

    Returns what the chosen mode produces: the rewritten
    :class:`~chunker.models.SasBatchResult` for ``"pre"``, the rewritten code
    for ``"post"``, and a :class:`BothResult` for ``"both"``. The caller knows
    which it asked for, so a union return beats three near-identical wrappers.

    Takes the whole :class:`~xref.sourcing.XrefMappings` rather than one flat
    dict: the two halves address different namespaces (dataset names, physical
    paths) and each mode needs a different subset of them.

    Raises
    ------
    ValueError
        The mode needs an input that was not given — ``"pre"`` without a
        *result*, or ``"post"`` / ``"both"`` without *code* and *language*.
    """
    resolved = (mode or configured_mode()).strip().lower()
    if resolved == APPLY_PRE:
        if result is None:
            raise ValueError("xref.apply('pre') needs a batch result")
        return apply_pre(result, mappings.dataset_mapping)
    if code is None or language is None:
        raise ValueError(f"xref.apply({resolved!r}) needs code and language")
    if resolved == APPLY_POST:
        return apply_post(code, language, mappings, on_failure=on_failure)
    if resolved == APPLY_BOTH:
        both = apply_both(code, language, mappings, on_failure=on_failure)
        if result is not None:
            both.result = apply_pre(result, mappings.dataset_mapping)
            both.pre_applied = True
        return both
    raise ValueError(
        f"unknown xref apply mode {mode!r}; expected one of "
        f"{'/'.join(APPLY_MODES)}"
    )
