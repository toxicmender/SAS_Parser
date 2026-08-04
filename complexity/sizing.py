"""T-shirt sizing: how raw effort becomes a size and a story-point estimate.

:class:`SizeModel` and the calibration constants it defaults from, split out of
:mod:`complexity.rules`. The division there is between the two things a profile
declares: *which construct means what* (the catalogue, in ``rules``) and *how
counted work becomes a size* (the scale, here). They ship in the same JSON file
and are read by the same loader, but nothing in the catalogue depends on the
scale or the other way round.

Every number in this module is calibrated **together with the anchor** and is
meaningless apart from it, which is why the whole block lives in one place —
see complexity/README.md for what the anchor is and what moving it does.

Logger name: ``complexity.sizing``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from .models import ComplexityTier, TranslationParity, TShirtSize

logger = logging.getLogger(__name__)

# Fallback per-tier score weights, used when a profile omits ``weights``.
# Weight only ranks units within a tier — it can never change the tier itself,
# which is presence-based (see complexity.analyzer).
WEIGHT_LOW = 1.0
WEIGHT_MEDIUM = 2.5
WEIGHT_HIGH = 5.0

_DEFAULT_WEIGHTS: dict[ComplexityTier, float] = {
    ComplexityTier.LOW: WEIGHT_LOW,
    ComplexityTier.MEDIUM: WEIGHT_MEDIUM,
    ComplexityTier.HIGH: WEIGHT_HIGH,
}

# The construct namespaces a profile may define. Each maps to the metadata (or
# detector) dimension the analyzer looks up; an unknown key in a profile is an
# error rather than a silently ignored typo.
CONSTRUCT_KINDS: frozenset[str] = frozenset(
    {
        "proc",
        "component_object",
        "function",
        "call_routine",
        "global_statement",
        "kind",
        "detector",
        "cross_file",
    }
)

# Fallback per-parity score weights for the *complexity* sizing dimension.
# Parity is what makes a size target-dependent: two files with identical tiers
# are not equal work if one's constructs are MANUAL against the target and the
# other's are DIRECT.
_DEFAULT_PARITY_WEIGHTS: dict[TranslationParity, float] = {
    TranslationParity.DIRECT: 0.0,
    TranslationParity.SUPPORTED: 0.5,
    TranslationParity.PARTIAL: 1.5,
    TranslationParity.HARD: 3.0,
    TranslationParity.MANUAL: 5.0,
}

# Default sizing model. Every number here is calibrated *together* with the
# anchor below and is meaningless apart from it, which is why the whole block
# lives in one place and ships as profile data.
DEFAULT_ANCHOR_RAW = 18.0

_DEFAULT_SIZE_SCALE: dict[TShirtSize, float] = {
    TShirtSize.SMALL: 2.0,
    TShirtSize.MEDIUM: 3.0,
    TShirtSize.LARGE: 5.0,
    TShirtSize.EXTRA_LARGE: 8.0,
}

# Upper bounds in points, at the geometric midpoints of the Fibonacci rungs
# (2|3 -> 2.5, 3|5 -> 4.0, 5|8 -> 6.5). Anything above the last is EXTRA_LARGE.
_DEFAULT_SIZE_BANDS: dict[TShirtSize, float] = {
    TShirtSize.SMALL: 2.5,
    TShirtSize.MEDIUM: 4.0,
    TShirtSize.LARGE: 6.5,
}

# Effort-dimension volume weights.
_DEFAULT_VOLUME_WEIGHTS: dict[str, float] = {
    "chunk": 0.5,
    "line": 0.01,
    "step": 1.0,
    "io": 1.0,
    "param": 0.5,
}

# Uncertainty-dimension weights: what the analysis could not pin down.
_DEFAULT_UNCERTAINTY_WEIGHTS: dict[str, float] = {
    "unresolved_ref": 3.0,
    "unclosed_block": 4.0,
    "unknown_chunk": 2.0,
    "diagnostic": 1.5,
}

# The three sizing dimensions, in report order.
DIMENSIONS: tuple[str, ...] = ("effort", "complexity", "uncertainty")

# Min-max window per dimension, as **multiples of the anchor**, applied after a
# log. Anchor-relative rather than absolute so the anchor stays the single knob
# that moves every verdict coherently (`complexity.size_anchor`): halve it and
# every window halves with it, so every file rates larger.
#
# `min` is the point below which a dimension stops discriminating — a file with
# less effort than 0.3 anchors is small however you count it — and `max` is
# where it saturates. Both floors and ceilings are the point of a min-max: the
# scale spends its resolution on the range real files actually occupy.
_DEFAULT_DIMENSION_BOUNDS: dict[str, tuple[float, float]] = {
    "effort": (0.35, 2.40),
    "complexity": (0.35, 1.40),
    "uncertainty": (0.00, 0.50),
}

# How far each normalised dimension can push the blend **on its own**. These
# deliberately sum to more than 1 and the blend is clamped there, so they are
# reaches rather than shares: a file that is nothing but volume still has to be
# able to reach the top of the scale, because an enormous file needs breaking
# up however plain its contents are. Effort reaches furthest for that reason;
# uncertainty reaches least, because it asks for an investigation rather than
# for translation hands — but being additive, it never *costs* a file anything
# to have no unknowns in it.
_DEFAULT_DIMENSION_WEIGHTS: dict[str, float] = {
    "effort": 0.88,
    "complexity": 0.50,
    "uncertainty": 0.20,
}

# Ends of the story-point scale, when a profile states neither `sizes.scale`
# ends nor a `sizes.story_points` range: the Fibonacci rungs the bands are
# calibrated on.
DEFAULT_MIN_STORY_POINTS = 2.0
DEFAULT_MAX_STORY_POINTS = 8.0

# The planning-poker deck. A reported estimate is always one of these, because
# that is what makes the progression Fibonacci rather than merely
# Fibonacci-inspired: the gaps widen with the number precisely so that nobody
# is asked to distinguish an 8 from a 9, and a scale that reports 6.77 has
# quietly reintroduced the precision the method exists to refuse.
FIBONACCI_POINTS: tuple[float, ...] = (
    1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0, 55.0, 89.0, 144.0, 233.0, 377.0,
)


def _nearest_fibonacci(value: float) -> float:
    """The deck entry closest to *value*, measured **geometrically**.

    Log distance, not linear, for the same reason the rungs are Fibonacci at
    all: on a geometric scale 4 is nearer to 5 than to 3, even though the
    arithmetic gaps are equal. Values outside the deck clamp to its ends.
    """
    if value <= FIBONACCI_POINTS[0]:
        return FIBONACCI_POINTS[0]
    if value >= FIBONACCI_POINTS[-1]:
        return FIBONACCI_POINTS[-1]
    return min(FIBONACCI_POINTS, key=lambda f: abs(math.log(f / value)))


def _log_fraction(value: float, low: float, high: float) -> float:
    """Where *value* sits between *low* and *high*, measured in log space.

    The inverse of the geometric rescale :meth:`SizeModel.points_for` applies,
    and the reason a band and a points value can be compared however the scale
    is denominated. Degenerate ranges fall back to a linear fraction.
    """
    if low <= 0 or high <= low or value <= 0:
        return (value - low) / (high - low) if high > low else 0.0
    return math.log(value / low) / math.log(high / low)


@dataclass(frozen=True)
class SizeModel:
    """How raw effort becomes a :class:`TShirtSize`. See complexity/README.md.

    Sizing is **relative estimation against a fixed anchor**, which is the
    method's own prescription — you size a story by comparing it to a reference
    story the team knows. The anchor is fixed profile data rather than
    something recomputed per run: a corpus-relative (percentile) scheme would
    re-rate the same file differently depending on which files it was analysed
    alongside, and would be undefined for a single-file run.

    Raw dimensions do not become points directly. Each is **log-transformed and
    min-max rescaled** onto 0-1 against its own anchor-relative window
    (:meth:`normalize`), the three are blended (:meth:`blend_for`), and the
    blend is min-max rescaled *in log space* onto the points scale
    (:meth:`points_for`). Two reasons:

    - the three dimensions are counted in incomparable units — effort runs to
      the hundreds on line and step counts while uncertainty rarely passes ten
      — so a plain sum silently weights them by magnitude rather than by
      intent. Rescaling to a common 0-1 makes the mix explicit and adjustable;
    - within a dimension the returns are diminishing: the 200th step of a file
      tells you far less than the 20th. A log says that, a raw count does not.

    Attributes
    ----------
    anchor_raw
        Raw score of the reference **Medium** file — the sum of its three
        dimensions. Every window below is a multiple of it, so it remains the
        single knob that moves every verdict at once: lower it and every file
        rates larger.
    anchor_dimensions
        That same reference file's ``(effort, complexity, uncertainty)`` split,
        when the profile states it. Because the dimensions are normalised
        separately, the anchor's *composition* now matters as much as its
        total, and a profile that declares it can be checked against the file
        it claims to describe.
    anchor_describes
        Prose description of that reference file, so the calibration can be
        argued with instead of merely obeyed.
    scale
        Nominal points per rung (Fibonacci by default). Together with *bands*
        it is the **calibration**: the two decide where each size begins, as a
        fraction of the scale's log span.
    min_story_points, max_story_points
        Ends of the reported story-point range. ``None`` (default) takes them
        from *scale*'s lowest and highest rungs. A file cannot score outside
        them, which is what makes the rescale a min-max rather than an
        open-ended ratio. Changing them re-denominates every ``points`` value
        without moving a single size boundary: the bands are read as fractions
        of the span, so a team that estimates on 1-13 gets its own numbers and
        the same verdicts.
    bands
        Inclusive upper bound in points for SMALL, MEDIUM, and LARGE, stated
        against *scale*; above the LARGE bound is EXTRA_LARGE.
    bounds
        ``{dimension: (min, max)}`` in multiples of the anchor — the min-max
        window each dimension is rescaled against, after the log.
    dimension_weights
        ``{dimension: weight}`` for blending the three normalised dimensions.
        Normalised to sum to 1 on use, so only their ratio matters.
    volume, uncertainty
        Per-unit weights for the effort and uncertainty dimensions.
    parity_weights
        Per-parity weights for the complexity dimension.
    min_size_by_kind
        Floors: a file containing this ``SasChunkKind`` is never smaller than
        the given size. Applied after banding.
    min_size_by_parity
        The same, keyed on the file's worst parity. Ships empty — the lever
        exists, but chunk kind proved the better discriminator.
    """

    anchor_raw: float = DEFAULT_ANCHOR_RAW
    anchor_describes: str = ""
    anchor_dimensions: tuple[float, float, float] | None = None
    min_story_points: float | None = None
    max_story_points: float | None = None
    scale: dict[TShirtSize, float] = field(
        default_factory=lambda: dict(_DEFAULT_SIZE_SCALE)
    )
    bands: dict[TShirtSize, float] = field(
        default_factory=lambda: dict(_DEFAULT_SIZE_BANDS)
    )
    bounds: dict[str, tuple[float, float]] = field(
        default_factory=lambda: dict(_DEFAULT_DIMENSION_BOUNDS)
    )
    dimension_weights: dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_DIMENSION_WEIGHTS)
    )
    volume: dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_VOLUME_WEIGHTS)
    )
    uncertainty: dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_UNCERTAINTY_WEIGHTS)
    )
    parity_weights: dict[TranslationParity, float] = field(
        default_factory=lambda: dict(_DEFAULT_PARITY_WEIGHTS)
    )
    min_size_by_kind: dict[str, TShirtSize] = field(default_factory=dict)
    min_size_by_parity: dict[TranslationParity, TShirtSize] = field(
        default_factory=dict
    )

    def window_for(self, dimension: str) -> tuple[float, float]:
        """The ``(min, max)`` window for *dimension*, in raw units.

        Stored anchor-relative and returned absolute, so every caller reads the
        window the current anchor implies rather than re-deriving it.
        """
        lo, hi = self.bounds.get(
            dimension, _DEFAULT_DIMENSION_BOUNDS.get(dimension, (0.0, 1.0))
        )
        return lo * self.anchor_raw, hi * self.anchor_raw

    def normalize(self, dimension: str, raw: float) -> float:
        """Rescale a raw dimension onto 0-1: log first, then min-max.

        ``log1p`` rather than ``log`` because every dimension legitimately
        reaches 0 — a file with nothing unresolved in it has no uncertainty at
        all — and ``log(0)`` is not a number a size can be built on. Values
        outside the window clamp: below the floor a dimension has stopped
        discriminating, above the ceiling it has saturated.
        """
        lo, hi = self.window_for(dimension)
        if hi <= lo:
            return 0.0
        span = math.log1p(hi) - math.log1p(lo)
        if span <= 0:
            return 0.0
        scaled = (math.log1p(max(raw, 0.0)) - math.log1p(lo)) / span
        return min(1.0, max(0.0, scaled))

    def blend_for(
        self, effort: float, complexity: float, uncertainty: float
    ) -> float:
        """The three normalised dimensions as one 0-1 value.

        A weighted **sum**, clamped at 1, rather than a weighted mean. The
        weights therefore say how far each dimension reaches on its own, and
        they sum past 1 on purpose: a mean would cap a file that is nothing but
        volume at the effort weight, so no amount of bulk could ever ask to be
        broken down, and every file with nothing unresolved in it would forfeit
        the uncertainty share for having no problems.
        """
        raws = {
            "effort": effort,
            "complexity": complexity,
            "uncertainty": uncertainty,
        }
        total = sum(
            self.normalize(d, raws[d])
            * max(self.dimension_weights.get(d, _DEFAULT_DIMENSION_WEIGHTS[d]), 0.0)
            for d in DIMENSIONS
        )
        return min(1.0, total)

    @property
    def story_point_range(self) -> tuple[float, float]:
        """The ``(min, max)`` story points a file can be reported as.

        Falls back to the scale's own end rungs, so a profile that says nothing
        about story points still reports the Fibonacci 2-8 the bands were
        calibrated on.
        """
        low = self.min_story_points
        high = self.max_story_points
        return (
            self.scale.get(TShirtSize.SMALL, DEFAULT_MIN_STORY_POINTS)
            if low is None
            else low,
            self.scale.get(TShirtSize.EXTRA_LARGE, DEFAULT_MAX_STORY_POINTS)
            if high is None
            else high,
        )

    def rung_points(self) -> dict[TShirtSize, float]:
        """The story points each size is **reported** as — one deck entry each.

        This is what a file's ``points`` is, and it is always a Fibonacci
        number: an estimate of 2, 3, 5, or 8, never the 6.77 a continuous
        rescale would land on. Estimation is deliberately coarse — the rungs
        widen so that nobody has to defend an 8 against a 9 — so reporting the
        un-snapped position would claim a precision the method disclaims, and
        would let two files a team cannot tell apart carry different numbers.

        On the profile's own scale these are the profile's rungs verbatim
        (Fibonacci 2/3/5/8 by default), so a stated calibration is never
        silently overridden. Re-denominating onto another range
        (``min_story_points`` / ``max_story_points``) keeps **both ends
        exactly** as asked for and re-derives the two interior rungs at their
        geometric positions, snapped back onto the deck: 1-13 reports
        1/2/5/13. Where a range is too narrow for four distinct deck entries
        the un-snapped value stands for that rung, because a scale whose rungs
        collide has stopped being a scale — Fibonacci is worth having, but not
        at the cost of two sizes reporting the same number.
        """
        scale_low = self.scale.get(TShirtSize.SMALL, DEFAULT_MIN_STORY_POINTS)
        scale_high = self.scale.get(
            TShirtSize.EXTRA_LARGE, DEFAULT_MAX_STORY_POINTS
        )
        low, high = self.story_point_range
        if (low, high) == (scale_low, scale_high):
            return dict(self.scale)

        rungs = {TShirtSize.SMALL: low, TShirtSize.EXTRA_LARGE: high}
        previous = low
        for size in (TShirtSize.MEDIUM, TShirtSize.LARGE):
            nominal = self.scale.get(size, _DEFAULT_SIZE_SCALE[size])
            position = _log_fraction(nominal, scale_low, scale_high)
            exact = (
                low * (high / low) ** position
                if low > 0 and high > low
                else low + position * (high - low)
            )
            snapped = _nearest_fibonacci(exact)
            if not previous < snapped < high:
                logger.debug(
                    f"rung_points: {size.value} snaps to {snapped} on the "
                    f"{low}-{high} scale, which would collide; keeping "
                    f"{exact:.2f}"
                )
                snapped = exact
            rungs[size] = round(snapped, 2)
            previous = rungs[size]
        return rungs

    def points_for_size(self, size: TShirtSize) -> float:
        """The story points *size* is reported as (see :meth:`rung_points`)."""
        return self.rung_points()[size]

    def points_for(
        self, effort: float, complexity: float = 0.0, uncertainty: float = 0.0
    ) -> float:
        """The continuous position on the scale, which the banding reads.

        Not what a file reports — :meth:`points_for_size` is, and it snaps to
        the deck. This is the un-rounded number behind that verdict: it is what
        :meth:`band_for` bands, what ranks two files *within* one size, and
        what a profile's anchor is re-measured against.

        The blend is min-max rescaled **in log space** — geometrically between
        the ends of :attr:`story_point_range` — for the same reason the default
        rungs are Fibonacci: the gap between Small and Medium is genuinely
        smaller than the gap between Large and Extra Large, so equal steps of
        evidence should be equal *ratios* of points, not equal differences. A
        blend of 0 is exactly the minimum, a blend of 1 exactly the maximum,
        and the reference file lands on MEDIUM by calibration.
        """
        low, high = self.story_point_range
        blend = self.blend_for(effort, complexity, uncertainty)
        if low <= 0 or high <= low:
            # A profile may flatten or invert the range; fall back to a linear
            # rescale rather than raising on a log of a non-positive number.
            return round(low + blend * (high - low), 2)
        return round(low * (high / low) ** blend, 2)

    def band_blends(self) -> dict[TShirtSize, float]:
        """Where each band's upper bound sits, as a fraction of the log span.

        Read off ``scale`` and ``bands`` — the calibration — rather than off
        the reported range, which is why re-denominating story points cannot
        move a size boundary.
        """
        low = self.scale.get(TShirtSize.SMALL, DEFAULT_MIN_STORY_POINTS)
        high = self.scale.get(TShirtSize.EXTRA_LARGE, DEFAULT_MAX_STORY_POINTS)
        return {
            size: _log_fraction(
                self.bands.get(size, _DEFAULT_SIZE_BANDS[size]), low, high
            )
            for size in (TShirtSize.SMALL, TShirtSize.MEDIUM, TShirtSize.LARGE)
        }

    def band_for(self, points: float) -> TShirtSize:
        """The size *points* falls in, before any floor is applied.

        Compares in blend space, so a profile reporting on 1-13 bands at the
        same evidence as one reporting on 2-8.
        """
        low, high = self.story_point_range
        position = _log_fraction(points, low, high)
        bounds = self.band_blends()
        for size in (TShirtSize.SMALL, TShirtSize.MEDIUM, TShirtSize.LARGE):
            if position <= bounds[size]:
                return size
        return TShirtSize.EXTRA_LARGE

    def volume_weight(self, key: str) -> float:
        """Effort weight for *key*, falling back to the built-in default."""
        return self.volume.get(key, _DEFAULT_VOLUME_WEIGHTS.get(key, 0.0))

    def uncertainty_weight(self, key: str) -> float:
        """Uncertainty weight for *key*, falling back to the built-in default."""
        return self.uncertainty.get(key, _DEFAULT_UNCERTAINTY_WEIGHTS.get(key, 0.0))

    def parity_weight(self, parity: TranslationParity) -> float:
        """Complexity-dimension weight for *parity*."""
        return self.parity_weights.get(parity, _DEFAULT_PARITY_WEIGHTS[parity])
