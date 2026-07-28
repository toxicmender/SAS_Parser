"""The structured shape the pipeline asks the LLM for, per work item.

One :class:`TranslationDocument` is the typed equivalent of the four Markdown
sections the unstructured system prompt asks for (Analysis / Mapping /
Translation / Risks). The pipeline requests it through LangChain's structured
output (``with_structured_output``); :mod:`pipeline.notebook` turns it into a
notebook without having to guess which fenced block was translated code and
which was an illustrative SAS snippet.

:meth:`TranslationDocument.to_markdown` is the load-bearing method. Whatever
the LLM returns, the pipeline persists *rendered Markdown* as the turn's AI
message and reports it as the item's ``response``, so conversation memory, the
resume path, and every ``validation`` metric keep seeing exactly the shape they
see today. In particular ``## Translation`` emits each code cell as a fenced
block tagged with its language, which is what
``validation.metrics.PythonSyntaxMetric`` scans for.

Pydantic only — no langchain import — so ``import chunker`` stays cheap
(``SasLLMPipeline`` is the lazily-imported one; see ``chunker/__init__.py``).

Pure data module — no logging.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Fence info strings we consider "already Python" when rendering, mirroring
# validation.metrics._PYTHON_INFO_STRINGS so a rendered document scores the
# same as a hand-written Markdown one.
_PYTHON_LANGUAGES = {"python", "python3", "py", "pyspark"}


class MappingEntry(BaseModel):
    """One SAS construct and what it becomes in the target language."""

    # Not plain `construct`: pydantic warns that it shadows a BaseModel
    # attribute, and the prefix reads better in the schema the model sees.
    sas_construct: str = Field(
        description="The SAS construct being mapped, e.g. 'PROC SORT NODUPKEY'."
    )
    equivalent: str = Field(
        description="Its equivalent in the target language, e.g. "
        "'dropDuplicates() after orderBy()'."
    )
    difference: str = Field(
        default="",
        description="Any semantic difference a reader must know about — date "
        "epoch offsets, MERGE vs join defaults, PDV vs DAG execution, macro "
        "expansion timing. Empty when the mapping is exact.",
    )


class RiskNote(BaseModel):
    """A hazard in the translation, worst first."""

    severity: Literal["P0", "P1", "P2"] = Field(
        description="P0 for a silent-error risk (wrong results, no failure), "
        "P1 for a loud failure, P2 for a stylistic or performance concern."
    )
    note: str = Field(description="What the risk is and what to check.")


class TranslationCell(BaseModel):
    """One cell of the translation, in execution order."""

    kind: Literal["code", "markdown"] = Field(
        description="'code' for runnable target-language code, 'markdown' for "
        "a prose note that belongs between two code cells."
    )
    source: str = Field(
        description="The cell body: target-language code, or Markdown prose. "
        "Do NOT wrap code in a Markdown fence — the fence is added when needed."
    )
    language: str | None = Field(
        default=None,
        description="Language of a code cell ('python' or 'sql'). Null for "
        "markdown cells.",
    )
    comment: str = Field(
        default="",
        description="Optional one-line heading for this cell, rendered above "
        "it. Use it to say what the cell does, e.g. 'Load the source table'.",
    )
    chunk_id: str | None = Field(
        default=None,
        description="The id of the batch member this cell implements, exactly "
        "as listed under '## Batch members' (e.g. 'f1-chunk-0001'). Set it on "
        "every cell when the batch has several members — it routes the cell "
        "into its source file's notebook; a cell serving several members "
        "carries the id of the one whose step it completes. May be null for "
        "a single-member batch.",
    )


class TranslationDocument(BaseModel):
    """One work item's translation, as the pipeline asks for it.

    Rendered to Markdown for memory/validation
    (:meth:`to_markdown`) and to notebook cells for the deliverable
    (:func:`pipeline.notebook.document_to_cells`).
    """

    analysis: str = Field(
        description="What the SAS construct(s) do and the reasoning that bears "
        "on correctness: execution order, PDV vs DAG semantics, macro "
        "expansion timing, and any hazard flagged in the item's context."
    )
    mapping: list[MappingEntry] = Field(
        default_factory=list,
        description="Per-construct mapping into the target language.",
    )
    cells: list[TranslationCell] = Field(
        default_factory=list,
        description="The translation itself, split into the cells a notebook "
        "should run in order. Preserve execution order across member chunks "
        "and files, and make cross-chunk dependencies explicit.",
    )
    risks: list[RiskNote] = Field(
        default_factory=list,
        description="Every risk worth flagging, worst first. Say so explicitly "
        "rather than guessing when a translation is ambiguous or unsafe.",
    )

    @property
    def code_cells(self) -> list[TranslationCell]:
        return [c for c in self.cells if c.kind == "code"]

    def fence_info(self, cell: TranslationCell) -> str:
        """The info string for *cell*'s fence in the rendered Markdown."""
        language = (cell.language or "").strip().lower()
        return language if language else "python"

    def to_markdown(self) -> str:
        """Render to the same four-section Markdown an unstructured run emits.

        This is what gets persisted as the AI turn and scored by the validation
        metrics, so the section headings and the fenced code blocks are part of
        the contract — see the module docstring.
        """
        lines: list[str] = ["## Analysis", "", self.analysis.strip(), ""]

        lines += ["## Mapping", ""]
        if self.mapping:
            for entry in self.mapping:
                text = f"- **{entry.sas_construct}** → {entry.equivalent}"
                if entry.difference.strip():
                    text += f" — {entry.difference.strip()}"
                lines.append(text)
        else:
            lines.append("_No construct mapping reported._")
        lines.append("")

        lines += ["## Translation", ""]
        if self.cells:
            for cell in self.cells:
                if cell.comment.strip():
                    lines += [f"### {cell.comment.strip()}", ""]
                if cell.kind == "code":
                    lines += [
                        f"```{self.fence_info(cell)}",
                        cell.source.strip("\n"),
                        "```",
                        "",
                    ]
                else:
                    lines += [cell.source.strip(), ""]
        else:
            lines += ["_No translation produced._", ""]

        lines += ["## Risks", ""]
        if self.risks:
            for risk in self.risks:
                marker = "⚠️ " if risk.severity == "P0" else ""
                lines.append(f"- {marker}**{risk.severity}** — {risk.note.strip()}")
        else:
            lines.append("_No risks flagged._")

        return "\n".join(lines).rstrip() + "\n"
