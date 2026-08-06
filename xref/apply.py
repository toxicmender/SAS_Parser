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
        pipeline = SasLLMPipeline(databricks_mapping=mappings.dataset_mapping)

    The second line is how :func:`apply_pre` reaches a real run: both batchers
    take ``databricks_mapping`` and apply it as a post-pass after grouping, so
    a pipeline constructed with it has already had this function's effect.
    :func:`apply_pre` is for a caller holding a :class:`SasBatchResult`
    directly.
``"post"``
    :mod:`xref.rewrite` over the generated code. Catches a table the SAS-side
    extraction never saw, at the cost of having to parse model output.
``"both"``
    Runs each and reports the names only one of them reached
    (:func:`apply_both`). That difference is the evidence for choosing a
    permanent mode: names post found and pre did not are what escaped the
    metadata extraction.

Logger name: ``xref.apply``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import app_config

from .rewrite import rewrite_python, rewrite_sql

if TYPE_CHECKING:  # avoid importing chunker at module scope
    from chunker.models import SasBatchResult

logger = logging.getLogger(__name__)

APPLY_PRE = "pre"
APPLY_POST = "post"
APPLY_BOTH = "both"
APPLY_MODES = (APPLY_PRE, APPLY_POST, APPLY_BOTH)

DEFAULT_MODE = APPLY_PRE

# Target languages the post rewriter can parse, by their normalised key.
_SQL_LANGUAGES = frozenset({"sparksql", "sql", "spark_sql"})
_PYTHON_LANGUAGES = frozenset({"pyspark", "python", "py"})


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
    mapping: dict[str, str],
    *,
    on_failure: str | None = None,
) -> str:
    """
    *code* with its table references rewritten, after conversion.

    *language* picks the parser — Spark SQL through ``sqlglot``, PySpark
    through the ``ast`` module. A language neither one covers (Spark Scala,
    say) is returned unchanged with a WARNING: no parser means no safe
    rewrite, and an unsafe one is out of the question.
    """
    if not mapping or not code.strip():
        return code
    from target_language import normalize_language

    key = normalize_language(language)
    if key in _SQL_LANGUAGES:
        return rewrite_sql(code, mapping, on_failure=on_failure)
    if key in _PYTHON_LANGUAGES:
        return rewrite_python(code, mapping, on_failure=on_failure)
    logger.warning(
        f"apply_post: no XREF rewriter for target language {language!r}; the "
        f"generated code is unchanged"
    )
    return code


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
        Source names the post pass rewrote that the pre pass had left alone.
        A non-empty list means a dataset name escaped the SAS-side metadata
        extraction, which is precisely the evidence for keeping ``"post"``.
    """

    code: str
    result: "SasBatchResult | None" = None
    pre_applied: bool = False
    post_changed: bool = False
    only_post: list[str] = field(default_factory=list)


def apply_both(
    code: str,
    language: str,
    mapping: dict[str, str],
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
    """
    baseline = pre_code if pre_code is not None else code
    rewritten = apply_post(code, language, mapping, on_failure=on_failure)
    only_post = [
        source
        for source, target in mapping.items()
        if target in rewritten and target not in baseline
    ]
    if only_post:
        logger.warning(
            f"apply_both: {len(only_post)} mapping(s) were only reached by the "
            f"post pass ({', '.join(sorted(only_post)[:5])}"
            f"{', ...' if len(only_post) > 5 else ''}); those dataset names "
            f"escaped the SAS-side metadata extraction"
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
    mapping: dict[str, str],
    on_failure: str | None = None,
) -> Any:
    """
    Dispatch on *mode* (``None`` reads :func:`configured_mode`).

    Returns what the chosen mode produces: the rewritten
    :class:`~chunker.models.SasBatchResult` for ``"pre"``, the rewritten code
    for ``"post"``, and a :class:`BothResult` for ``"both"``. The caller knows
    which it asked for, so a union return beats three near-identical wrappers.

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
        return apply_pre(result, mapping)
    if code is None or language is None:
        raise ValueError(f"xref.apply({resolved!r}) needs code and language")
    if resolved == APPLY_POST:
        return apply_post(code, language, mapping, on_failure=on_failure)
    if resolved == APPLY_BOTH:
        both = apply_both(code, language, mapping, on_failure=on_failure)
        if result is not None:
            both.result = apply_pre(result, mapping)
            both.pre_applied = True
        return both
    raise ValueError(
        f"unknown xref apply mode {mode!r}; expected one of "
        f"{'/'.join(APPLY_MODES)}"
    )
