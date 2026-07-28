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
        """Nominal story points for this rung (Fibonacci: 2, 3, 5, 8).

        The bridge from a qualitative size to a summable quantity. A profile's
        ``sizes.scale`` may restate these; this is the default the bands are
        calibrated against, and :attr:`FileComplexity.points` carries the
        *computed* continuous value rather than the rung's nominal one — on
        whatever range ``sizes.story_points`` (or ``complexity.min_story_points``
        / ``max_story_points``) asks for, which re-denominates the numbers
        without moving a size.
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


class CrossFileProfile(BaseModel):
    """How one source file is coupled to the rest of the corpus.

    Built by :mod:`complexity.crossfile` from metadata the chunker already
    extracts. The three reference states are kept apart because they mean
    different things for a migration:

    - **imports/exports** — a real dependency on another file in the corpus.
      Translating this file in isolation would lose context.
    - **unresolved** — a reference satisfied by no file in scope. Only
      assertable when the corpus holds more than one file (see
      :attr:`corpus_files`); with a single file there is nothing to have
      searched, so the same reference is merely *external*.
    """

    source_id: str
    # Files whose outputs this file consumes, and which consume its outputs.
    depends_on: list[str] = Field(default_factory=list)
    depended_on_by: list[str] = Field(default_factory=list)
    # Construct names resolved elsewhere in the corpus, exported to it, and
    # satisfied nowhere — each as "<kind>:<name>" for display.
    imports: list[str] = Field(default_factory=list)
    exports: list[str] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)
    # How many files were in scope when this profile was built. 1 means
    # "unresolved" could not be established and was not claimed.
    corpus_files: int = 1

    @property
    def is_coupled(self) -> bool:
        """Whether this file has any cross-file reference in either direction."""
        return bool(self.depends_on or self.depended_on_by)

    def __str__(self) -> str:
        return (
            f"CrossFileProfile({self.source_id}, depends_on={len(self.depends_on)}, "
            f"depended_on_by={len(self.depended_on_by)}, "
            f"unresolved={len(self.unresolved)})"
        )


class FileComplexity(_ComplexityBase):
    """Overall migration verdict for one **source file**.

    The unit of T-shirt sizing. Files, not batches, because a batch may span
    several files (``SasBatch.source_files``) while every chunk carries exactly
    one ``source_id`` — so a file rollup built from chunks is unambiguous
    however the corpus was batched.

    Inherits ``tier`` / ``translation_difficulty`` (worst-case over its chunks,
    the same rules as everywhere else) and adds :attr:`size`, which is the
    volume-aware verdict those two cannot express.

    The three dimensions behind :attr:`size` are reported separately: a LARGE
    file that is large on ``uncertainty_raw`` needs its parse errors
    investigated, while one large on ``effort_raw`` just needs more hands. Each
    is carried twice — raw as counted, and ``*_norm`` after the log + min-max
    rescale that :attr:`points` is derived from — because the two answer
    different questions: the raw number is the measurement, the normalised one
    is how much of the scale that measurement actually claimed.
    """

    source_id: str
    size: TShirtSize = TShirtSize.SMALL
    # Computed position on the story-point scale — continuous, so it both ranks
    # files within a size and sums across a corpus into a backlog estimate.
    # Bounded by the scale's ends (the Fibonacci 2 and 8 unless a profile or
    # config says otherwise), because points are a min-max rescale. Changing
    # those ends re-denominates this number and never changes `size`.
    points: float = 0.0
    # The three declared dimensions, in raw (pre-normalisation) units.
    effort_raw: float = 0.0
    complexity_raw: float = 0.0
    uncertainty_raw: float = 0.0
    # The same three after the log + min-max rescale that produced `points`,
    # each on 0-1. Reported because the raw numbers alone no longer explain the
    # size: a dimension at 1.0 has saturated its window, and one at 0.0 is
    # below the level where it discriminates at all.
    effort_norm: float = 0.0
    complexity_norm: float = 0.0
    uncertainty_norm: float = 0.0
    # The weighted mean of the three normalised dimensions, on 0-1 — the single
    # number `points` is a log-space rescale of.
    blend: float = 0.0
    chunk_count: int = 0
    line_count: int = 0
    chunks: list[ChunkComplexity] = Field(default_factory=list)
    cross_file: CrossFileProfile | None = None
    # Batch ids inside this file, offered as cut points when it needs breaking
    # up. Empty when the analysis was not given batches to work from.
    suggested_split: list[str] = Field(default_factory=list)
    # The chunk kind that forced a size floor, when one applied — so a size
    # nobody can derive from the numbers still explains itself.
    floored_by: str = ""
    # False when the uncertainty dimension could not see parser diagnostics
    # (the batch-result entry points do not carry them). Recorded rather than
    # silently reporting a lower uncertainty than the file really has.
    uncertainty_complete: bool = True

    @computed_field  # type: ignore[prop-decorator]
    @property
    def raw_total(self) -> float:
        """Sum of the three dimensions, as counted — before any normalisation.

        Not what :attr:`points` scales: each dimension is rescaled separately
        (see :attr:`effort_norm`). This is the measurement a profile's anchor
        is stated in, which is what keeps ``anchor.raw`` re-measurable.
        """
        return round(self.effort_raw + self.complexity_raw + self.uncertainty_raw, 3)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def needs_breakdown(self) -> bool:
        """Whether this file should be split before translation."""
        return self.size.needs_breakdown

    def __str__(self) -> str:
        return (
            f"FileComplexity {self.source_id} size={self.size.label} "
            f"points={self.points:.1f} tier={self.tier} "
            f"difficulty={self.translation_difficulty} chunks={self.chunk_count}"
        )


class CorpusComplexityReport(BaseModel):
    """Complexity across every analysed item of a file, batch result, or corpus."""

    source_ids: list[str] = Field(default_factory=list)
    chunks: list[ChunkComplexity] = Field(default_factory=list)
    batches: list[BatchComplexity] = Field(default_factory=list)
    # Per-source-file rollups, carrying the T-shirt sizes.
    files: list[FileComplexity] = Field(default_factory=list)
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

    @computed_field  # type: ignore[prop-decorator]
    @property
    def size_counts(self) -> dict[str, int]:
        """How many files fall in each size (all four keys always present)."""
        counts = {s.value: 0 for s in TShirtSize}
        for f in self.files:
            counts[f.size.value] += 1
        return counts

    @computed_field  # type: ignore[prop-decorator]
    @property
    def overall_size(self) -> TShirtSize:
        """Largest file size in the corpus.

        Worst-case, like every other rollup here — a migration is not done
        until its hardest file is done, so an average would understate it.
        """
        return max_size([f.size for f in self.files])

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_points(self) -> float:
        """Summed story points across every file — the backlog estimate.

        The reason sizes carry a numeric scale at all: a qualitative label
        cannot be added up, and "how big is this migration?" is the question
        the sizing exists to answer.
        """
        return round(sum(f.points for f in self.files), 1)

    @property
    def files_needing_breakdown(self) -> list[FileComplexity]:
        """Files rated EXTRA_LARGE, largest first — split these before starting."""
        return sorted(
            (f for f in self.files if f.needs_breakdown),
            key=lambda f: f.points,
            reverse=True,
        )

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
        ]
        if self.files:
            lines += [
                f"- Overall size: **{self.overall_size.label}**",
                f"- Total story points: {self.total_points:.1f} "
                f"across {len(self.files)} file(s)",
            ]
        lines += [
            "",
            "## Tier breakdown",
            "",
            "| Tier | Units |",
            "| --- | ---: |",
        ]
        for tier in ComplexityTier:
            lines.append(f"| {tier.value} | {counts[tier.value]} |")

        if self.files:
            size_counts = self.size_counts
            lines += [
                "",
                "## File sizes",
                "",
                "| File | Size | Points | Tier | Parity | Effort | Cplx | Uncert "
                "| Blend |",
                "| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: |",
            ]
            # Raw dimensions with their normalised share alongside, since the
            # size follows from the second pair of numbers, not the first.
            for f in sorted(self.files, key=lambda x: x.points, reverse=True):
                lines.append(
                    f"| {f.source_id} | {f.size.label} | {f.points:.1f} "
                    f"| {f.tier} | {f.translation_difficulty} "
                    f"| {f.effort_raw:.1f} ({f.effort_norm:.2f}) "
                    f"| {f.complexity_raw:.1f} ({f.complexity_norm:.2f}) "
                    f"| {f.uncertainty_raw:.1f} ({f.uncertainty_norm:.2f}) "
                    f"| {f.blend:.2f} |"
                )
            lines += [
                "",
                "| Size | Files |",
                "| --- | ---: |",
            ]
            for size in TShirtSize:
                lines.append(f"| {size.label} | {size_counts[size.value]} |")

        breakdown = self.files_needing_breakdown
        if breakdown:
            lines += [
                "",
                f"## Files needing breakdown ({len(breakdown)})",
                "",
                "Extra Large is a recommendation, not just a magnitude: split "
                "these before translating them.",
                "",
            ]
            for f in breakdown:
                cuts = (
                    ", ".join(f.suggested_split)
                    if f.suggested_split
                    else "no batch boundaries available — split by step"
                )
                lines.append(
                    f"- **{f.source_id}** ({f.points:.1f} pts, "
                    f"{f.chunk_count} chunks, {f.line_count} lines) — "
                    f"suggested cut points: {cuts}"
                )

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
        sizing = (
            f", files={len(self.files)}, overall_size={self.overall_size}, "
            f"points={self.total_points:.1f}"
            if self.files
            else ""
        )
        return (
            f"CorpusComplexityReport(units={len(self.items)}, "
            f"overall_tier={self.overall_tier}, "
            f"difficulty={self.overall_difficulty}, "
            f"low={counts['LOW']}, medium={counts['MEDIUM']}, "
            f"high={counts['HIGH']}{sizing})"
        )
