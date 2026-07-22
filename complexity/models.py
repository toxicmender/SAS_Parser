"""Pydantic models for chunk complexity analysis. See complexity/README.md.

Pure data module — no logging, no imports from the rest of this package.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, computed_field


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


class ComplexitySignal(BaseModel):
    """One recognised construct and what it implies for translation.

    Signals are the atoms of the analysis: every tier, score, and difficulty
    on the models below is derived from a list of these, so a result always
    carries the evidence for its own verdict.

    Fields
    ------
    name
        Canonical construct identifier, e.g. ``"array"``, ``"proc:sql"``,
        ``"component_object:hash"``.
    category
        Coarse grouping for reporting, e.g. ``"array"``, ``"macro"``,
        ``"io"`` (see ``complexity.rules``).
    tier
        Data-complexity tier this construct alone implies.
    parity
        Spark feature-parity rating for this construct.
    weight
        Contribution to the numeric score, used to rank chunks *within* a
        tier. Never affects the tier itself.
    evidence
        What was actually found in this chunk — a source snippet for a
        detector signal, empty for a metadata signal (whose ``name`` already
        identifies it).
    note
        The catalogue's standing guidance for this construct: why it is rated
        the way it is, and what the translation trap is. Kept separate from
        *evidence* so a detector's snippet never shadows it — the guidance is
        usually the more useful half.
    source
        ``"metadata"`` when derived from :class:`~chunker.models.SasChunkMetadata`,
        ``"detector"`` when found by this package's own regex scans.
    """

    name: str
    category: str
    tier: ComplexityTier
    parity: TranslationParity
    weight: float = 1.0
    evidence: str = ""
    note: str = ""
    source: str = "metadata"

    @property
    def detail(self) -> str:
        """Evidence and note joined for display; either may be empty."""
        return " — ".join(p for p in (self.evidence, self.note) if p)

    def __str__(self) -> str:
        detail = f": {self.detail}" if self.detail else ""
        return f"{self.name} [{self.tier}/{self.parity}]{detail}"


class _ComplexityBase(BaseModel):
    """Shared shape of a scored unit (a chunk or a batch).

    Holds the verdict fields and the signal list they were computed from;
    subclasses add only their own identity fields.
    """

    tier: ComplexityTier = ComplexityTier.LOW
    score: float = 0.0
    translation_difficulty: TranslationParity = TranslationParity.DIRECT
    signals: list[ComplexitySignal] = Field(default_factory=list)
    rationale: str = ""
    # Which rule-set profile produced this verdict. A parity rating is only
    # meaningful against a named target, so every result carries its own.
    target: str = ""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def categories(self) -> list[str]:
        """Distinct signal categories present, sorted."""
        return sorted({s.category for s in self.signals})

    @property
    def high_signals(self) -> list[ComplexitySignal]:
        """The signals that forced a HIGH tier — the ones worth reviewing first."""
        return [s for s in self.signals if s.tier is ComplexityTier.HIGH]


class ChunkComplexity(_ComplexityBase):
    """Complexity verdict for a single :class:`~chunker.models.SasChunk`."""

    chunk_id: str
    source_id: str | None = None
    kind: str | None = None
    start_line: int = 0
    end_line: int = 0

    def __str__(self) -> str:
        return (
            f"ChunkComplexity {self.chunk_id} tier={self.tier} "
            f"score={self.score:.2f} difficulty={self.translation_difficulty} "
            f"signals={len(self.signals)}"
        )


class BatchComplexity(_ComplexityBase):
    """Complexity verdict for a :class:`~chunker.models.SasBatch`.

    Aggregated from its members: ``tier`` is the highest member tier,
    ``translation_difficulty`` the worst member parity, and ``score`` the sum
    of member scores — a batch of ten simple steps really is more work than
    one, so batch score is additive rather than averaged.
    """

    batch_id: str
    source_files: list[str] = Field(default_factory=list)
    members: list[ChunkComplexity] = Field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"BatchComplexity {self.batch_id} tier={self.tier} "
            f"score={self.score:.2f} difficulty={self.translation_difficulty} "
            f"members={len(self.members)}"
        )


class CorpusComplexityReport(BaseModel):
    """Complexity across every analysed item of a file, batch result, or corpus."""

    source_ids: list[str] = Field(default_factory=list)
    chunks: list[ChunkComplexity] = Field(default_factory=list)
    batches: list[BatchComplexity] = Field(default_factory=list)
    # The rule-set profile every unit below was scored against, and its
    # human-readable name — reports state the target so two reports scored
    # against different languages are never mistaken for comparable.
    target: str = ""
    target_display: str = ""

    @property
    def items(self) -> list[ChunkComplexity | BatchComplexity]:
        """Every scored unit — batches first, then standalone chunks."""
        return [*self.batches, *self.chunks]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def tier_counts(self) -> dict[str, int]:
        """How many scored units fall in each tier (all three keys always present)."""
        counts = {t.value: 0 for t in ComplexityTier}
        for item in self.items:
            counts[item.tier.value] += 1
        return counts

    @computed_field  # type: ignore[prop-decorator]
    @property
    def overall_tier(self) -> ComplexityTier:
        """Highest tier anywhere in the corpus."""
        return max_tier([item.tier for item in self.items])

    @computed_field  # type: ignore[prop-decorator]
    @property
    def overall_difficulty(self) -> TranslationParity:
        """Worst Spark parity anywhere in the corpus."""
        return worst_parity([item.translation_difficulty for item in self.items])

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_score(self) -> float:
        """Sum of every scored unit's score."""
        return round(sum(item.score for item in self.items), 3)

    def hardest(self, limit: int = 10) -> list[ChunkComplexity | BatchComplexity]:
        """The *limit* units most in need of attention.

        Ordered by tier, then Spark difficulty, then score — so a HIGH/MANUAL
        item always outranks a merely bulky LOW one.
        """
        return sorted(
            self.items,
            key=lambda i: (
                tier_rank(i.tier),
                parity_rank(i.translation_difficulty),
                i.score,
            ),
            reverse=True,
        )[:limit]

    def to_markdown(self, *, top: int = 10) -> str:
        """Render the report as a Markdown summary plus a hardest-items table."""
        counts = self.tier_counts
        lines = [
            "# SAS chunk complexity report",
            "",
            f"- Target: **{self.target_display or self.target or 'unknown'}**",
            f"- Sources: {', '.join(self.source_ids) or 'none'}",
            f"- Scored units: {len(self.items)} "
            f"({len(self.batches)} batch(es), {len(self.chunks)} chunk(s))",
            f"- Overall tier: **{self.overall_tier}**",
            f"- Overall Spark parity: **{self.overall_difficulty}**",
            f"- Total score: {self.total_score:.2f}",
            "",
            "## Tier breakdown",
            "",
            "| Tier | Units |",
            "| --- | ---: |",
        ]
        for tier in ComplexityTier:
            lines.append(f"| {tier.value} | {counts[tier.value]} |")

        hardest = self.hardest(top)
        if hardest:
            lines += [
                "",
                f"## Hardest {len(hardest)} unit(s)",
                "",
                "| Item | Tier | Spark parity | Score | Drivers |",
                "| --- | --- | --- | ---: | --- |",
            ]
            for item in hardest:
                item_id = (
                    item.batch_id
                    if isinstance(item, BatchComplexity)
                    else item.chunk_id
                )
                drivers = ", ".join(
                    dict.fromkeys(s.name for s in item.high_signals)
                ) or ", ".join(item.categories) or "—"
                lines.append(
                    f"| {item_id} | {item.tier} | {item.translation_difficulty} "
                    f"| {item.score:.2f} | {drivers} |"
                )
        return "\n".join(lines)

    def __str__(self) -> str:
        counts = self.tier_counts
        return (
            f"CorpusComplexityReport(units={len(self.items)}, "
            f"overall_tier={self.overall_tier}, "
            f"difficulty={self.overall_difficulty}, "
            f"low={counts['LOW']}, medium={counts['MEDIUM']}, high={counts['HIGH']})"
        )
