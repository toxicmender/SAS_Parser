"""PromptBuilder facade: reference PDFs -> a formatted guidance block per item.

See prompt_builder/README.md.

Ties the package together: load + chunk + index the reference corpus once at
construction, then :meth:`build` a Markdown guidance block for one pipeline
item's ``(query, constructs)``. Returns ``None`` when nothing is relevant, so
the caller can omit the block entirely.

The metadata -> ``(query, constructs)`` mapping deliberately lives in the
pipeline, not here, so ``prompt_builder`` imports nothing from ``chunker``.

Logger name: ``prompt_builder.builder``.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterable

import app_config

from .catalog import CorpusLoader, DocumentSpec, default_catalog
from .models import (
    ConstructKey,
    DocRole,
    InstructionChunk,
    SelectedInstruction,
    SelectionTier,
)
from .selector import InstructionSelector
from .user_instructions import UserInstructionSet

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Focus hints (directional stimulus): a compact per-item hint block naming the
# constructs, hazards, and retrieved topics the response should address —
# instance-specific keywords rendered above the reference guidance, so the
# model treats them as salient rather than inferring salience from pages of
# verbatim manual text.
# ---------------------------------------------------------------------------

# Human-readable label per construct kind ({name} is upper-cased on format,
# except component objects, which the guides write in lowercase prose).
_KIND_LABELS = {
    "function": "{name} function",
    "call_routine": "CALL {name} routine",
    "macro_function": "%{name} macro function",
    "macro_statement": "%{name} macro statement",
    "global_statement": "{name} statement",
    "proc": "PROC {name}",
    "component_object": "{lower} object",
    "format": "{name} format",
    "informat": "{name} informat",
    "option": "{name} option",
    "system_option": "{name} system option",
}


def _describe_construct(key: ConstructKey) -> str:
    template = _KIND_LABELS.get(key.kind, "{name} ({kind})")
    return template.format(
        name=key.name.upper(), lower=key.name.lower(), kind=key.kind
    )


# One-line caution per known hazard construct — the stimulus phrase reminding
# the model *why* the construct is dangerous, not just that it is present.
_HAZARD_CAUTIONS: dict[ConstructKey, str] = {
    ConstructKey(kind="call_routine", name="symput"): (
        "run-time macro-variable write; scope/timing differs from %LET"
    ),
    ConstructKey(kind="call_routine", name="symputx"): (
        "run-time macro-variable write; scope/timing differs from %LET"
    ),
    ConstructKey(kind="call_routine", name="symget"): (
        "run-time macro-variable read; value depends on execution order"
    ),
    ConstructKey(kind="call_routine", name="execute"): (
        "generates code at run time; macro timing is easy to get wrong"
    ),
    ConstructKey(kind="macro_statement", name="goto"): (
        "computed jump; control flow resolved at macro execution"
    ),
    ConstructKey(kind="macro_statement", name="abort"): (
        "halts the job/step; needs explicit error handling in the target"
    ),
    ConstructKey(kind="macro_function", name="sysfunc"): (
        "calls a DATA-step function at macro time"
    ),
}
_DEFAULT_CAUTION = "silent-error risk"

# Caps keeping the hint block a compact stimulus, not another guidance dump.
_MAX_HINT_CONSTRUCTS = 8
_MAX_HINT_TOPICS = 4


# ---------------------------------------------------------------------------
# Reasoning directives (conditional chain-of-thought): one imperative
# reasoning instruction per hazard construct, emitted only when the item
# carries that hazard — so the extra reasoning tokens are spent exactly on
# the items whose failure modes need them, not on every trivial chunk.
# ---------------------------------------------------------------------------

_HAZARD_DIRECTIVES: dict[ConstructKey, str] = {
    ConstructKey(kind="call_routine", name="symput"): (
        "Trace step by step when each macro variable is written versus "
        "read — a CALL SYMPUT value is not available until the step "
        "boundary — and verify the translation preserves that ordering."
    ),
    ConstructKey(kind="call_routine", name="symputx"): (
        "Trace step by step when each macro variable is written versus "
        "read — a CALL SYMPUTX value is not available until the step "
        "boundary — and verify the translation preserves that ordering."
    ),
    ConstructKey(kind="call_routine", name="symget"): (
        "Trace step by step which value each CALL SYMGET read observes "
        "given the surrounding execution order, and verify the translation "
        "reads the same value."
    ),
    ConstructKey(kind="call_routine", name="execute"): (
        "Walk through exactly what code CALL EXECUTE generates and when it "
        "runs (after the current step, with macro references resolved at "
        "generation time), and confirm the translation preserves that "
        "deferred execution."
    ),
    ConstructKey(kind="macro_statement", name="goto"): (
        "Enumerate every possible %GOTO target and reason through each "
        "resulting control-flow path before translating; make each path "
        "explicit in the target code."
    ),
    ConstructKey(kind="macro_statement", name="abort"): (
        "Decide explicitly how %ABORT's job/step-halt semantics map to "
        "error handling in the target before translating, and state the "
        "chosen behaviour."
    ),
    ConstructKey(kind="macro_function", name="sysfunc"): (
        "Reason through when each %SYSFUNC call evaluates (at macro "
        "resolution, before the step runs) and verify the translation "
        "computes the value at an equivalent point."
    ),
}


class PromptBuilder:
    """
    Build a per-item instruction block from a reference corpus.

    Construct from an in-memory chunk list, from :class:`DocumentSpec`s, or from
    a reference directory (the last two load + cache via :class:`CorpusLoader`).

    Parameters
    ----------
    chunks : Iterable[InstructionChunk]
        The instruction corpus to retrieve over.
    user_instructions : str | UserInstructionSet | None
        Operator-supplied project rules. A plain string is parsed via
        :meth:`UserInstructionSet.from_text` (see ``user_instructions.py``
        for the heading/directive syntax). Selected user chunks render in
        their own ``## Project instructions`` block above the reference
        guidance and take priority over every reference tier.
        ``[example: ...]``-scoped sections (few-shot worked pairs) instead
        render in a ``## {examples_heading}`` block placed last, adjacent
        to the item they demonstrate for.
    top_k : int | None
        Maximum topical (ranked) chunks per item. ``None`` (default) reads
        ``prompt_builder.top_k`` from config.json, falling back to 6 (see
        the ``app_config`` package).
    max_instruction_words : int | None
        Word budget for the whole guidance block. ``None`` reads
        ``prompt_builder.max_instruction_words``, falling back to 1500.
        Keep this >= the instruction chunker's ``max_words`` so any single
        reference section always fits.
    pinned_sections : Iterable[str]
        Section-path substrings always injected first.
    embeddings, embedding_cache_path, rrf_k, reranker :
        Forwarded to :class:`InstructionSelector` (dense retrieval is off unless
        ``embeddings`` is given).
    focus_hints : bool | None
        Render a compact ``## {hints_heading}`` block (directional stimulus:
        the item's hazards, matched constructs, and retrieved topics as
        explicit keywords) above the reference guidance. ``None`` (default)
        reads ``prompt_builder.focus_hints`` from config.json, falling back
        to ``True``.
    reasoning_directives : bool | None
        Render a ``## {directives_heading}`` block with one step-by-step
        reasoning instruction per hazard construct the item carries
        (conditional chain-of-thought: reasoning tokens are spent only on
        hazard items). ``None`` (default) reads
        ``prompt_builder.reasoning_directives`` from config.json, falling
        back to ``True``.
    heading : str
        Markdown H2 heading for the reference-guidance block.
    project_heading : str
        Markdown H2 heading for the user-instruction block.
    hints_heading : str
        Markdown H2 heading for the focus-hints block.
    directives_heading : str
        Markdown H2 heading for the reasoning-directives block.
    examples_heading : str
        Markdown H2 heading for the few-shot worked-examples block.
    """

    def __init__(
        self,
        chunks: Iterable[InstructionChunk],
        *,
        user_instructions: "str | UserInstructionSet | None" = None,
        user_max_words: int | None = None,
        top_k: int | None = None,
        max_instruction_words: int | None = None,
        pinned_sections: Iterable[str] = (),
        embeddings: Any | None = None,
        embedding_cache_path: str | None = None,
        rrf_k: int = 60,
        reranker: Callable[[str, list[str]], list[float]] | None = None,
        focus_hints: bool | None = None,
        reasoning_directives: bool | None = None,
        heading: str = "Relevant migration guidance",
        project_heading: str = "Project instructions",
        hints_heading: str = "Focus hints",
        directives_heading: str = "Reasoning directives",
        examples_heading: str = "Worked examples",
    ) -> None:
        self.top_k = app_config.resolve(top_k, "prompt_builder", "top_k", 6)
        self.max_instruction_words = app_config.resolve(
            max_instruction_words, "prompt_builder", "max_instruction_words", 1500
        )
        # None default keeps user chunks limited only by the overall budget.
        self.user_max_words = app_config.resolve(
            user_max_words, "user_instructions", "max_words", None
        )
        self.focus_hints = app_config.resolve(
            focus_hints, "prompt_builder", "focus_hints", True
        )
        self.reasoning_directives = app_config.resolve(
            reasoning_directives, "prompt_builder", "reasoning_directives", True
        )
        self.heading = heading
        self.project_heading = project_heading
        self.hints_heading = hints_heading
        self.directives_heading = directives_heading
        self.examples_heading = examples_heading
        if isinstance(user_instructions, str):
            user_instructions = UserInstructionSet.from_text(user_instructions)
        self.user_instructions = user_instructions
        # Retained so with_user_instructions can rebuild an equivalent
        # selector over the same reference corpus.
        self._pinned_sections = list(pinned_sections)
        self._embeddings = embeddings
        self._embedding_cache_path = embedding_cache_path
        self._rrf_k = rrf_k
        self._reranker = reranker
        self._selector = InstructionSelector(
            chunks,
            user_instructions=user_instructions,
            user_max_words=self.user_max_words,
            embeddings=embeddings,
            embedding_cache_path=embedding_cache_path,
            rrf_k=rrf_k,
            reranker=reranker,
            pinned_sections=pinned_sections,
        )

    def with_user_instructions(
        self, user_instructions: "str | UserInstructionSet | None"
    ) -> "PromptBuilder":
        """
        A new builder over the same reference corpus and settings, with
        *user_instructions* replacing any current set. The selector index is
        rebuilt once; the original builder is untouched.
        """
        return PromptBuilder(
            self._selector.reference_chunks,
            user_instructions=user_instructions,
            user_max_words=self.user_max_words,
            top_k=self.top_k,
            max_instruction_words=self.max_instruction_words,
            pinned_sections=self._pinned_sections,
            embeddings=self._embeddings,
            embedding_cache_path=self._embedding_cache_path,
            rrf_k=self._rrf_k,
            reranker=self._reranker,
            focus_hints=self.focus_hints,
            reasoning_directives=self.reasoning_directives,
            heading=self.heading,
            project_heading=self.project_heading,
            hints_heading=self.hints_heading,
            directives_heading=self.directives_heading,
            examples_heading=self.examples_heading,
        )

    @classmethod
    def from_specs(
        cls,
        specs: list[DocumentSpec],
        *,
        loader: CorpusLoader | None = None,
        cache_dir: str | None = None,
        pinned_sections: Iterable[str] = (),
        **kwargs: Any,
    ) -> "PromptBuilder":
        """Load + chunk *specs* (with the on-disk cache), then build."""
        if loader is None:
            loader = CorpusLoader(cache_dir=cache_dir) if cache_dir else CorpusLoader()
        chunks = loader.load(specs)
        # Spec-declared pins plus any passed explicitly.
        pins = list(pinned_sections)
        for spec in specs:
            pins.extend(spec.pinned_sections)
        builder = cls(chunks, pinned_sections=pins, **kwargs)
        # A budget below the chunker's window size silently drops whole
        # construct hits — the known misconfiguration; warn loudly.
        if builder.max_instruction_words < loader.chunker.max_words:
            logger.warning(
                f"from_specs: max_instruction_words="
                f"{builder.max_instruction_words} is below the chunker's "
                f"max_words={loader.chunker.max_words}; single reference "
                f"sections may not fit the budget and will be dropped whole"
            )
        return builder

    @classmethod
    def from_reference_dir(
        cls,
        reference_dir: str = "reference_docs",
        *,
        include_unknown: bool = False,
        loader: CorpusLoader | None = None,
        cache_dir: str | None = None,
        **kwargs: Any,
    ) -> "PromptBuilder":
        """
        Build from the default catalog of PDFs present under *reference_dir*.
        ``include_unknown=True`` also indexes PDFs the catalog doesn't
        recognise, with a generic auto-strategy spec.
        """
        return cls.from_specs(
            default_catalog(reference_dir, include_unknown=include_unknown),
            loader=loader,
            cache_dir=cache_dir,
            **kwargs,
        )

    def build(
        self, query: str, constructs: Iterable[ConstructKey] = ()
    ) -> str | None:
        """
        The Markdown block(s) for one item — a ``## Project instructions``
        block for selected user rules, then a compact ``## {hints_heading}``
        stimulus block, then a ``## {directives_heading}`` block of per-hazard
        reasoning instructions, then a ``## {heading}`` block for reference
        guidance, then a ``## {examples_heading}`` block of few-shot worked
        examples (last, adjacent to the item it demonstrates for), each
        omitted when empty — or ``None`` when nothing at all is relevant (so
        the caller injects no block).
        """
        constructs = list(constructs)
        picks = self._selector.select_detailed(
            query,
            constructs,
            max_words=self.max_instruction_words,
            top_k=self.top_k,
        )
        if not picks:
            logger.debug("build: no relevant instruction chunks; no block")
            return None
        examples = [
            p.chunk for p in picks if p.tier is SelectionTier.USER_EXAMPLE
        ]
        user = [
            p.chunk
            for p in picks
            if p.chunk.role is DocRole.USER_INSTRUCTION
            and p.tier is not SelectionTier.USER_EXAMPLE
        ]
        reference = [
            p for p in picks if p.chunk.role is not DocRole.USER_INSTRUCTION
        ]
        logger.debug(
            f"build: {len(user)} user + {len(examples)} example + "
            f"{len(reference)} reference chunk(s) injected"
        )
        blocks: list[str] = []
        if user:
            blocks.append(self._format_user(user))
        if self.focus_hints:
            hints = self._format_hints(picks, constructs)
            if hints:
                blocks.append(hints)
        if self.reasoning_directives:
            directives = self._format_directives(constructs)
            if directives:
                blocks.append(directives)
        if reference:
            blocks.append(self._format_reference(reference))
        if examples:
            blocks.append(self._format_examples(examples))
        return "\n\n".join(blocks)

    def _format_directives(self, constructs: list[ConstructKey]) -> str | None:
        """
        The conditional chain-of-thought block for one item: one imperative
        reasoning instruction per hazard construct the item carries, or
        ``None`` when the item has no hazard with a directive. Keyed on the
        *item's* constructs (like the hazard hint line), not on the
        selection, so the directive survives even when no reference section
        matched.
        """
        directives: list[str] = []
        for key in constructs:
            directive = _HAZARD_DIRECTIVES.get(key)
            if directive and directive not in directives:
                directives.append(directive)
        if not directives:
            return None
        return "\n".join(
            [
                f"## {self.directives_heading}",
                "",
                "Before writing the translation, in your Analysis:",
                *(f"- {d}" for d in directives),
            ]
        )

    def _format_hints(
        self,
        picks: list[SelectedInstruction],
        constructs: list[ConstructKey],
    ) -> str | None:
        """
        The directional-stimulus block for one item, or ``None`` when there
        is nothing to hint at. Hazards come from the *item's* constructs (a
        hazard deserves the caution even when no reference section matched);
        construct and topic lines come from the selection, so they are
        stop-list-filtered and budget-bounded already.
        """
        hazards: list[ConstructKey] = []
        for key in constructs:
            if key in self._selector.hazard_constructs and key not in hazards:
                hazards.append(key)
        matched: list[ConstructKey] = []
        topics: list[str] = []
        for pick in picks:
            if (
                pick.tier is SelectionTier.CONSTRUCT
                and pick.construct_key is not None
                and pick.construct_key not in matched
            ):
                matched.append(pick.construct_key)
            elif pick.tier is SelectionTier.TOPICAL:
                title = pick.chunk.section_path.rsplit(">", 1)[-1].strip()
                if title and title not in topics:
                    topics.append(title)

        lines: list[str] = []
        if hazards:
            cautions = ", ".join(
                f"{_describe_construct(k)} — "
                f"{_HAZARD_CAUTIONS.get(k, _DEFAULT_CAUTION)}"
                for k in hazards
            )
            lines.append(f"- ⚠️ Hazards to address explicitly: {cautions}")
        if matched:
            names = ", ".join(
                _describe_construct(k) for k in matched[:_MAX_HINT_CONSTRUCTS]
            )
            lines.append(f"- Constructs to map: {names}")
        if topics:
            lines.append(
                f"- Related reference topics: {'; '.join(topics[:_MAX_HINT_TOPICS])}"
            )
        if not lines:
            return None
        return "\n".join([f"## {self.hints_heading}", "", *lines])

    @staticmethod
    def _body_of(chunk: InstructionChunk) -> str:
        # Strip the retrieval-only title/breadcrumb prefix; the header line
        # rendered above the body already states the location.
        parts = chunk.text.split("\n\n", 1)
        return (parts[1] if len(parts) > 1 else chunk.text).strip()

    def _format_user(self, picks: list[InstructionChunk]) -> str:
        lines = [f"## {self.project_heading}", ""]
        for chunk in picks:
            # Operator rules cite no document or pages — just their heading.
            lines.append(f"### {chunk.section_path}")
            lines.append(self._body_of(chunk))
            lines.append("")
        return "\n".join(lines).rstrip()

    def _format_examples(self, picks: list[InstructionChunk]) -> str:
        lines = [
            f"## {self.examples_heading}",
            "",
            "Follow the structure and conventions these worked examples "
            "demonstrate:",
            "",
        ]
        for chunk in picks:
            # Like operator rules: the operator's own heading, no citations.
            lines.append(f"### {chunk.section_path}")
            lines.append(self._body_of(chunk))
            lines.append("")
        return "\n".join(lines).rstrip()

    @staticmethod
    def _selection_reason(pick: SelectedInstruction) -> str:
        """
        Why a reference chunk was injected, for its header — so the model can
        weigh an authoritative construct/hazard match above a merely related
        topical hit.
        """
        if pick.construct_key is not None and pick.tier in (
            SelectionTier.HAZARD,
            SelectionTier.CONSTRUCT,
        ):
            return f"{pick.tier.value}: {pick.construct_key.name}"
        return pick.tier.value  # pinned | topical

    def _format_reference(self, picks: list[SelectedInstruction]) -> str:
        lines = [f"## {self.heading}", ""]
        for pick in picks:
            chunk = pick.chunk
            pages = (
                f"p. {chunk.page_start}"
                if chunk.page_start == chunk.page_end
                else f"pp. {chunk.page_start}-{chunk.page_end}"
            )
            lines.append(
                f"### [{chunk.doc_id} · {chunk.section_path} · {pages} · "
                f"{self._selection_reason(pick)}]"
            )
            lines.append(self._body_of(chunk))
            lines.append("")
        return "\n".join(lines).rstrip()
