"""The verdict vocabulary: complexity tiers, translation parity, T-shirt sizes.

The three ordered enums every other model in this package is expressed in,
with the rank and rollup helpers that give them their order. Split out of
:mod:`complexity.models` because they are the vocabulary rather than the
models: a caller that only needs to compare two verdicts, or roll a list of
them up to one, needs this and nothing else.

Pure data — no logging, and no imports from the rest of this package.
"""

from __future__ import annotations

from enum import StrEnum


class ComplexityTier(StrEnum):
    """Data-complexity band of a chunk, batch, or corpus.

    Ordered LOW < MEDIUM < HIGH by :data:`_TIER_RANK`; a chunk's tier is the
    highest tier among the constructs it contains (presence-based, so a single
    ARRAY makes an otherwise-trivial chunk HIGH).
    """

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class TranslationParity(StrEnum):
    """How well a SAS construct maps onto **the target language**.

    The scale is language-neutral; which construct earns which rating is
    per-target data, supplied by a :class:`~complexity.rules.RuleSet` loaded
    from a JSON profile. The same construct can legitimately rate differently
    against Spark SQL and against PySpark — a ``%MACRO`` definition has no
    counterpart in pure SQL (``MANUAL``) but maps onto a parameterised Python
    function (``HARD``).

    Ordered DIRECT < SUPPORTED < PARTIAL < HARD < MANUAL by
    :data:`_PARITY_RANK`, from "translates one-for-one" to "no equivalent
    exists; a human must redesign it".

    DIRECT
        A literal equivalent exists (``PROC SQL`` select -> ``spark.sql``).
    SUPPORTED
        Idiomatic equivalent exists, mechanical rewrite (``PROC SORT`` ->
        ``ORDER BY`` / ``orderBy``).
    PARTIAL
        Equivalent exists but semantics differ enough to need care (a SAS
        match-merge is not a plain join — same-named columns overlay).
    HARD
        No direct equivalent; needs restructuring into a different paradigm
        (row-wise ``DO`` loops -> vectorised columns or a UDF).
    MANUAL
        Outside the target entirely; a human decision is required.
    """

    DIRECT = "DIRECT"
    SUPPORTED = "SUPPORTED"
    PARTIAL = "PARTIAL"
    HARD = "HARD"
    MANUAL = "MANUAL"


class TShirtSize(StrEnum):
    """Overall migration size of a **source file**, on the agile T-shirt scale.

    Distinct from :class:`ComplexityTier`, and deliberately so. A tier answers
    "how hard is the hardest thing in here?" and is presence-based: one ARRAY
    makes a file HIGH however short it is. A size answers "how much work is
    this file?", so it is **volume-aware** — without that it would be a
    relabelling of tier, and a 2000-line file of plain DATA steps (which raises
    no signal at all) would read SMALL, which is wrong.

    Four sizes, not six. XS/XXL are conventional but both ActiveCollab and
    Asana recommend starting at S/M/L/XL, and a size nobody can tell apart from
    its neighbour costs more than it explains.

    Sizes are a **combination** of three declared dimensions — effort,
    complexity, and uncertainty — following the guidance that a team must state
    what a size represents and then hold to it. The three are reported
    separately on :class:`FileComplexity` so a LARGE file can be read as bulky,
    hard, or unknown; those need different responses.

    Unlike a tier, a size is **target-dependent**: parity feeds the complexity
    dimension, so the same file is legitimately more work against Spark SQL
    than against PySpark.

    Ordered SMALL < MEDIUM < LARGE < EXTRA_LARGE by :data:`_SIZE_RANK`.

    EXTRA_LARGE carries an instruction, not just a magnitude: it is the
    published meaning of the top rung ("very large tasks needing breakdown"),
    so :attr:`needs_breakdown` is True there and only there, and
    :attr:`FileComplexity.suggested_split` names where to cut.
    """

    SMALL = "SMALL"
    MEDIUM = "MEDIUM"
    LARGE = "LARGE"
    EXTRA_LARGE = "EXTRA_LARGE"

    @property
    def label(self) -> str:
        """Human-readable name, e.g. ``"Extra Large"``."""
        return _SIZE_LABEL[self]

    @property
    def points(self) -> int:
        """Story points for this rung (Fibonacci: 2, 3, 5, 8).

        The bridge from a qualitative size to a summable quantity, and what a
        file actually reports: :attr:`FileComplexity.points` is its size's
        rung, so every estimate in a report is a planning-poker deck entry
        rather than a point somewhere between two of them.

        A profile's ``sizes.scale`` may restate these, and
        ``sizes.story_points`` (or ``complexity.min_story_points`` /
        ``max_story_points``) re-denominates them onto another range without
        moving a size — see :meth:`complexity.rules.SizeModel.rung_points`,
        which is the authority once a profile or config is in play. This table
        is the default the bands are calibrated against.
        """
        return _SIZE_POINTS[self]

    @property
    def needs_breakdown(self) -> bool:
        """Whether this size means "split this before working it"."""
        return self is TShirtSize.EXTRA_LARGE


# Rank tables backing the "max tier" / "worst parity" aggregation rules. Kept
# module-private and consulted through the helpers below so no call site
# open-codes an ordering that could drift from the enum.
_TIER_RANK: dict[ComplexityTier, int] = {
    ComplexityTier.LOW: 0,
    ComplexityTier.MEDIUM: 1,
    ComplexityTier.HIGH: 2,
}

_PARITY_RANK: dict[TranslationParity, int] = {
    TranslationParity.DIRECT: 0,
    TranslationParity.SUPPORTED: 1,
    TranslationParity.PARTIAL: 2,
    TranslationParity.HARD: 3,
    TranslationParity.MANUAL: 4,
}

_SIZE_RANK: dict[TShirtSize, int] = {
    TShirtSize.SMALL: 0,
    TShirtSize.MEDIUM: 1,
    TShirtSize.LARGE: 2,
    TShirtSize.EXTRA_LARGE: 3,
}

_SIZE_LABEL: dict[TShirtSize, str] = {
    TShirtSize.SMALL: "Small",
    TShirtSize.MEDIUM: "Medium",
    TShirtSize.LARGE: "Large",
    TShirtSize.EXTRA_LARGE: "Extra Large",
}

# Fibonacci rungs. The progression is geometric because estimation confidence
# is: the gap between a Small and a Medium is genuinely smaller than the gap
# between a Large and an Extra Large, and evenly-spaced rungs would imply a
# precision the method explicitly disclaims.
_SIZE_POINTS: dict[TShirtSize, int] = {
    TShirtSize.SMALL: 2,
    TShirtSize.MEDIUM: 3,
    TShirtSize.LARGE: 5,
    TShirtSize.EXTRA_LARGE: 8,
}


def tier_rank(tier: ComplexityTier) -> int:
    """Sort key for *tier* (LOW=0 < MEDIUM=1 < HIGH=2)."""
    return _TIER_RANK[tier]


def parity_rank(parity: TranslationParity) -> int:
    """Sort key for *parity* (DIRECT=0 < ... < MANUAL=4)."""
    return _PARITY_RANK[parity]


def max_tier(tiers: list[ComplexityTier]) -> ComplexityTier:
    """The highest tier in *tiers*; LOW for an empty list.

    An empty list means "no complexity signal fired at all" — a chunk with
    nothing recognisable in it is the simplest thing there is, not an
    unknown, so LOW is the correct floor rather than a separate NONE tier.
    """
    return max(tiers, key=tier_rank, default=ComplexityTier.LOW)


def worst_parity(parities: list[TranslationParity]) -> TranslationParity:
    """The least-translatable parity in *parities*; DIRECT for an empty list."""
    return max(parities, key=parity_rank, default=TranslationParity.DIRECT)


def size_rank(size: TShirtSize) -> int:
    """Sort key for *size* (SMALL=0 < MEDIUM=1 < LARGE=2 < EXTRA_LARGE=3)."""
    return _SIZE_RANK[size]


def max_size(sizes: list[TShirtSize]) -> TShirtSize:
    """The largest size in *sizes*; SMALL for an empty list.

    Used both to roll a corpus up to one headline size and to apply the
    per-chunk-kind size floors, which are a "no smaller than" rule.
    """
    return max(sizes, key=size_rank, default=TShirtSize.SMALL)
