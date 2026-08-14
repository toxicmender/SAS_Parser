"""Which target a work item should actually be translated into.

A Spark SQL run asked for Spark SQL everywhere, including for the constructs
that have no Spark SQL answer — a ``%MACRO`` definition, ``CALL EXECUTE``,
``LAG``. The model produced something anyway, and the failure was silent:
``target_syntax`` parses the result happily because it *is* valid SQL, it is
just not equivalent SAS.

This module answers "can the run's target express this item?" from data the
package already owns and already reviews — the per-target parity ratings in
``complexity/profiles/``. A construct rating worse against the run's target than
against the fallback target is the evidence; nothing here is a hand-maintained
list of exceptions, because a second list would drift from the profiles that CI
and the reports are scored against (Architecture.md invariant 12).

The rule
--------
Fall back when any of the item's constructs is **not implementable** in the run's
target *and* the fallback rates it better. Both halves are load-bearing:

*Not implementable* is :attr:`~complexity.tiers.TranslationParity.HARD` or
:attr:`~complexity.tiers.TranslationParity.MANUAL` — "no direct equivalent;
needs restructuring into a different paradigm", and "no equivalent exists". A
``PARTIAL`` construct *can* be written in the target; it just needs care, which
is what the guidance and the risk notes are for. Without this half, ``MACRO_CALL``
(``PARTIAL`` against Spark SQL, ``SUPPORTED`` against PySpark) would move
essentially every real SAS item, and a Spark SQL run would quietly become a
PySpark run.

*Better* is :func:`~complexity.tiers.parity_rank` on the shared
``DIRECT < SUPPORTED < PARTIAL < HARD < MANUAL`` scale. Without this half,
``do_until`` — ``HARD`` against both — would move an item to trade one hard
problem for the identical hard problem in another language.

Against the shipped profiles that is ten constructs, in two coherent groups:

*The macro facility* — ``%MACRO`` definitions, macro control flow,
``CALL EXECUTE``, ``DOSUBL``, ``RESOLVE``, ``SYMGET``, ``CALL MODULE`` — plus
``PROC FCMP``. Pure SQL has no macro processor and no user-defined function
definition; PySpark has both.

*The DATA step's procedural core* — ``do_loop`` and ``link_return``, both
``HARD`` against Spark SQL and ``PARTIAL`` against PySpark. An iterative ``DO``
becomes vectorised columns or an explode, and ``LINK``/``RETURN`` becomes a
function; neither has a SQL form at all.

Note what is *not* in that list. ``do_until`` / ``do_while`` (unbounded row-wise
iteration), ``merge_no_by``, ``data_goto``, ``array`` and ``filename_pipe`` all
rate the same against both targets, so an item containing only those stays where
it is — the hard problem travels with it.

A construct in neither catalogue is
:attr:`~complexity.tiers.TranslationParity.DIRECT`, not "unknown" — the
catalogue is an allowlist and silence means "nothing notable found" (see
complexity/README.md). In practice both loaded rule sets carry full catalogues,
so this is a safety net rather than the usual path.

Where the constructs come from
------------------------------
The caller supplies them. :func:`pipeline.prompting._profile_constructs_for_item`
names what chunk metadata reports (``proc``, ``function``, ``call_routine``,
``component_object``, ``global_statement``), adds the member chunks' ``kind``,
and runs :func:`complexity.detectors.detect_constructs` for the ``detector``
family — the DATA step constructs metadata cannot report because they are found
by scanning source rather than by naming an identifier. That scan runs only when
there is somewhere to fall back to.

Logger name: ``complexity.fallback``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

from target_language import TargetLanguage

from .rules import RuleSet, load_ruleset
from .tiers import TranslationParity, parity_rank

logger = logging.getLogger(__name__)

#: Construct kinds this module compares. A caller naming a kind outside this set
#: (the pipeline emits ``statement``, ``macro_statement`` and ``category`` too)
#: is not an error — those simply have no catalogue to be rated against, and a
#: construct with no rating cannot be evidence that a target cannot express it.
COMPARED_KINDS: frozenset[str] = frozenset(
    {
        "proc",
        "function",
        "call_routine",
        "component_object",
        "global_statement",
        "kind",
        "detector",
    }
)

#: Parities meaning "the target cannot express this" — the first half of the
#: rule. HARD is "no direct equivalent; needs restructuring into a different
#: paradigm" and MANUAL is "no equivalent exists"; everything below them can be
#: written in the target, with care the guidance already supplies.
NOT_IMPLEMENTABLE: frozenset[TranslationParity] = frozenset(
    {TranslationParity.HARD, TranslationParity.MANUAL}
)


@dataclass(frozen=True)
class FallbackReason:
    """One construct that the run's target rates worse than the fallback.

    Carries both ratings rather than a rendered sentence: the log line, the
    notebook header and the prompt each want to say it differently, and a
    consumer that only wants to count them should not have to parse prose.
    """

    kind: str
    name: str
    run_parity: TranslationParity
    fallback_parity: TranslationParity

    def __str__(self) -> str:
        return (
            f"{self.kind}:{self.name} rates {self.run_parity} against the run's "
            f"target and {self.fallback_parity} against the fallback"
        )


@dataclass(frozen=True)
class TargetChoice:
    """What to translate one item into, and why.

    ``fell_back`` is not ``target != run_target``: a caller may disable the
    fallback, and then the target is the run's while the reasons still describe
    what would have moved it. Keeping both means a run with the fallback off can
    still report what it would have caught.
    """

    target: TargetLanguage
    fell_back: bool = False
    reasons: tuple[FallbackReason, ...] = ()

    def __bool__(self) -> bool:
        return self.fell_back


def _parity(rules: RuleSet, kind: str, name: str) -> TranslationParity:
    """*name*'s parity under *rules*, or ``DIRECT`` when it is not catalogued.

    ``DIRECT`` rather than a sentinel because that is what the catalogue's
    allowlist semantics already mean everywhere else in this package: silence is
    "nothing notable found", never "unknown".
    """
    spec = rules.constructs.get(kind, {}).get(name)
    return spec.parity if spec is not None else TranslationParity.DIRECT


def choose_target(
    constructs: Iterable[tuple[str, str]],
    *,
    run_target: TargetLanguage,
    fallback_to: TargetLanguage | None = None,
) -> TargetChoice:
    """The target *constructs* should be translated into.

    Parameters
    ----------
    constructs
        ``(kind, name)`` pairs, in the profiles' vocabulary — the pipeline's
        ``ConstructKey``s plus the item's ``SasChunkKind`` values under
        ``"kind"``. Kinds outside :data:`COMPARED_KINDS` are ignored.
    run_target
        What the run was asked for.
    fallback_to
        The target to move to. ``None`` disables the comparison entirely and
        returns *run_target*, which is what an operator switching the fallback
        off gets.

    Returns *run_target* unchanged whenever nothing is both unimplementable and
    better elsewhere, so a PySpark run — or a Spark SQL run of ordinary SQL —
    costs one dictionary lookup per construct and no behaviour change at all.
    """
    if fallback_to is None or fallback_to.key == run_target.key:
        return TargetChoice(target=run_target)

    run_rules = load_ruleset(run_target.complexity_profile)
    fallback_rules = load_ruleset(fallback_to.complexity_profile)

    reasons: list[FallbackReason] = []
    for kind, name in constructs:
        if kind not in COMPARED_KINDS:
            continue
        run_parity = _parity(run_rules, kind, name)
        if run_parity not in NOT_IMPLEMENTABLE:
            continue
        fallback_parity = _parity(fallback_rules, kind, name)
        if parity_rank(run_parity) > parity_rank(fallback_parity):
            reasons.append(
                FallbackReason(
                    kind=kind,
                    name=name,
                    run_parity=run_parity,
                    fallback_parity=fallback_parity,
                )
            )

    if not reasons:
        return TargetChoice(target=run_target)

    # Worst first: the reason a reader most needs is the one with the least
    # chance of translating, and the prompt and notebook header both show only
    # the first few.
    reasons.sort(key=lambda r: -parity_rank(r.run_parity))
    return TargetChoice(
        target=fallback_to, fell_back=True, reasons=tuple(reasons)
    )
