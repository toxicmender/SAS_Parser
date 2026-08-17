"""Render the effective LLM request for each pipeline output.

The pipeline keeps conversational history intentionally compact, but operators
still need an auditable copy of what the model actually received.  This module
turns the captured role/content message list into one Markdown file per item.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROMPTS_FOLDER = "prompts"


def _safe_name(value: object, used: set[str]) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "item")).strip("._")
    stem = stem or "item"
    candidate = stem
    suffix = 2
    while candidate.casefold() in used:
        candidate = f"{stem}_{suffix}"
        suffix += 1
    used.add(candidate.casefold())
    return f"{candidate}.md"


def _fenced(value: str, language: str = "text") -> str:
    longest = max((len(match) for match in re.findall(r"`+", value)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{language}\n{value}\n{fence}"


def _render_content(content: Any) -> str:
    if isinstance(content, str):
        return _fenced(content)
    return _fenced(
        json.dumps(content, ensure_ascii=False, indent=2, sort_keys=True), "json"
    )


def render_prompt_artifact(output: dict[str, Any]) -> str:
    """Render one output's captured prompt as auditable Markdown.

    Resumed outputs deliberately carry no ``prompt_messages`` because the
    current process did not make their historical LLM call.  Their artifact is
    labelled as unavailable and includes only the newly derived item request,
    never misrepresenting it as the request that produced the stored answer.
    """
    item_id = str(output.get("item_id") or "item")
    sources = [str(source) for source in output.get("source_files") or []]
    target = str(output.get("target_language") or "unknown")
    metadata = [
        f"- Item: `{item_id}`",
        f"- Target: `{target}`",
        f"- Sources: {', '.join(f'`{source}`' for source in sources) or '(none)'}",
    ]
    messages = output.get("prompt_messages")
    if not isinstance(messages, list):
        derived = str(output.get("prompt") or "")
        return "\n".join(
            [
                f"# Prompt unavailable: {item_id}",
                "",
                *metadata,
                "",
                "This item was resumed without a new model call, so the exact "
                "historical message list is unavailable in this process. The "
                "text below is a re-derived item request, not the prompt that "
                "produced the recovered answer.",
                "",
                "## Re-derived item request",
                "",
                _render_content(derived),
                "",
            ]
        )

    sections = [
        f"# Effective prompt: {item_id}",
        "",
        *metadata,
        "",
        "The messages below are the fully formatted request captured at the "
        "LLM client boundary for the accepted attempt.",
    ]
    for index, message in enumerate(messages, start=1):
        role = str(message.get("role") or "message").replace("_", " ").title()
        sections.extend(
            [
                "",
                f"## {index}. {role}",
                "",
                _render_content(message.get("content", "")),
            ]
        )
    sections.append("")
    return "\n".join(sections)


def prompt_artifacts_from_outputs(
    outputs: list[dict[str, Any]],
) -> dict[str, str]:
    """Return ``{safe_filename.md: rendered_prompt}`` in pipeline order."""
    used: set[str] = set()
    return {
        _safe_name(output.get("item_id"), used): render_prompt_artifact(output)
        for output in outputs
    }


def write_prompts(outputs: list[dict[str, Any]], out_dir: Path | str) -> list[Path]:
    """Write prompt Markdown under ``<out_dir>/prompts`` and return paths.

    If a resumed output has no exact message list and an artifact already
    exists, the exact artifact from the earlier run wins over the unavailable
    placeholder generated now.
    """
    directory = Path(out_dir) / PROMPTS_FOLDER
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    artifacts = prompt_artifacts_from_outputs(outputs)
    for output, (name, contents) in zip(outputs, artifacts.items(), strict=True):
        dest = directory / name
        if output.get("prompt_messages") is None and dest.exists():
            logger.info(f"write_prompts: preserving existing resumed prompt {dest}")
        else:
            dest.write_text(contents, encoding="utf-8")
            logger.debug(f"write_prompts: wrote {dest}")
        written.append(dest)
    logger.info(
        f"write_prompts: wrote {len(written)} prompt artifact(s) to {directory}"
    )
    return written
