"""Rendering a Markdown report as a PDF — the one implementation.

Both report surfaces need this: :mod:`complexity` renders its estimate to a
file beside the Markdown, and :mod:`validation` renders its verdicts to bytes
for upload. They had a renderer each, running the same markdown-it → PyMuPDF
``Story`` paging loop against two stylesheets that had already drifted, and the
weaker of the two was the one an operator actually received — no code folding,
no image resolution, and no cap on the paging loop.

So this is a leaf package, not part of either: it imports ``markdown_it`` and
``pymupdf`` and nothing from this repo, exactly like ``target_language``.
Callers pick their output shape — :func:`render_pdf` (file in, file out),
:func:`render_markdown_pdf` (text in, file out), or
:func:`render_markdown_bytes` (text in, bytes out) — and share everything else.

The Markdown is the primary artefact and stays the primary artefact — it diffs,
it renders in every viewer, and it is what the report layers produce. A PDF is
for the other audience: the report goes to someone who does not have the
repository checked out, and "here is a link to a .md file" is not an answer for
them. So this converts, and never replaces.

Two dependencies, both of them already core to this project rather than an
extra: ``markdown-it-py`` parses (it is the same CommonMark implementation the
reports are written against) and PyMuPDF's ``Story`` lays the resulting HTML
out onto pages. Nothing here shells out to a converter, and nothing here needs
a browser engine installed.

Unlike the dependency-graph image — a supplement that degrades to ``None`` when
matplotlib is absent — a PDF is only ever produced because a caller explicitly
asked for one. So failures here **raise** :class:`PdfRenderError` instead of
being logged and swallowed: silently not producing the one thing that was asked
for is the worse outcome.

Two things the layout engine needs help with:

- **Code blocks do not wrap.** ``Story`` clips a long ``<pre>`` line at the
  frame edge rather than folding it, and the individual reports print whole SAS
  statements, which are routinely wider than a page. So fenced code is
  soft-wrapped to :data:`CODE_WIDTH` columns *before* it becomes HTML.
- **Images resolve against an archive, not the filesystem.** The report links
  ``dependency-graph.png`` relative to itself, so the Markdown's own directory
  is handed to ``Story`` as a ``pymupdf.Archive`` and the image lands in the
  PDF instead of turning into a broken-link gap.

Logger name: ``reporting.pdf``.
"""

from __future__ import annotations

import html
import io
import logging
import textwrap
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Columns a fenced code line is folded at. Sized for the A4 page below at the
#: stylesheet's monospace size; a wider page would fit more, and folding early
#: only costs a wrapped line, so this does not need to track the CSS exactly.
CODE_WIDTH = 96

#: Page geometry, in points. A4 portrait with a 50pt (~18mm) margin all round.
PAGE_SIZE = "a4"
PAGE_MARGIN = 50.0

# A hard stop on the paging loop. `Story.place` returning "more" forever would
# otherwise fill a disk: it is not supposed to happen, but a runaway is a much
# worse failure than a truncated PDF with a warning against it.
_MAX_PAGES = 2000

#: The stylesheet, kept deliberately plain. `Story` implements a subset of CSS,
#: so this leans on what that subset does well — sizes, weights, borders,
#: backgrounds — and asks for no layout it cannot honour.
STYLESHEET = """
body { font-family: sans-serif; font-size: 9.5px; line-height: 1.4; }
h1 { font-size: 19px; margin-top: 10px; margin-bottom: 6px; }
h2 { font-size: 14px; margin-top: 12px; margin-bottom: 5px; }
h3 { font-size: 11.5px; margin-top: 10px; margin-bottom: 4px; }
h4, h5, h6 { font-size: 10px; margin-top: 8px; margin-bottom: 3px; }
p { margin-top: 0px; margin-bottom: 6px; }
ul, ol { margin-top: 0px; margin-bottom: 6px; }
li { margin-bottom: 2px; }
a { color: #1a4f8a; }
code { font-family: monospace; font-size: 8.5px; }
pre { font-family: monospace; font-size: 7.5px; background-color: #f4f5f7;
      margin-top: 4px; margin-bottom: 8px; }
table { width: 100%; margin-top: 4px; margin-bottom: 8px; }
th { font-size: 8.5px; text-align: left; background-color: #eceff4;
     border: 1px solid #b8c0cc; padding: 3px; }
td { font-size: 8.5px; text-align: left; border: 1px solid #d4dae2;
     padding: 3px; }
blockquote { margin-left: 10px; color: #444444; }
hr { margin-top: 6px; margin-bottom: 6px; }
"""
# Deliberately no `img` rule, and no width attribute on the tag either.
# `Story` fits an image into the space left on the page, keeping its aspect
# ratio, and that constraint wins: an `img { width: ... }` or a `width=` was
# measured making the dependency graph *smaller*, never larger. So the picture
# is as big as the page has room for, which is the best available answer — and
# it is a supplement in the PDF exactly as it is in the Markdown, with the edge
# table right beneath it carrying the same edges.


class PdfRenderError(RuntimeError):
    """A PDF was asked for and could not be produced."""


# ---------------------------------------------------------------------------
# Markdown -> HTML
# ---------------------------------------------------------------------------


def wrap_code(text: str, width: int = CODE_WIDTH) -> str:
    """Fold every line of *text* to *width* columns, preserving its indentation.

    Continuation lines keep the original line's leading whitespace so a folded
    SAS statement still reads as one statement rather than as a new one, and
    words are broken mid-token when a single token is itself too wide (a long
    quoted path, say) — losing the tail off the page edge is not an option.
    """
    if width <= 0:
        return text
    folded: list[str] = []
    for line in text.expandtabs(4).split("\n"):
        if len(line) <= width:
            folded.append(line)
            continue
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        # A deeply indented line must not indent its continuations off the
        # page; past a quarter of the width the fold just starts at the left.
        if len(indent) > width // 4:
            indent = " " * (width // 4)
        pieces = textwrap.wrap(
            line,
            width=width,
            initial_indent="",
            subsequent_indent=indent,
            break_long_words=True,
            break_on_hyphens=False,
            replace_whitespace=False,
            drop_whitespace=False,
        )
        folded.extend(pieces or [line])
    return "\n".join(folded)


def markdown_to_html(markdown: str, *, code_width: int = CODE_WIDTH) -> str:
    """*markdown* as an HTML fragment for the layout engine.

    Tables are enabled (the reports are mostly tables) and raw HTML is not: the
    individual reports embed SAS source, and a stray ``<`` in a source file
    must render as a ``<``, not open an element.

    Images are passed through untouched — see :data:`STYLESHEET` for why they
    are also left alone there.
    """
    from markdown_it import MarkdownIt
    from markdown_it.renderer import RendererHTML

    class _FoldingRenderer(RendererHTML):
        """The stock HTML renderer, with code blocks folded on the way out.

        A renderer subclass rather than an entry poked into ``renderer.rules``:
        the two code token types are the only thing being changed, and an
        override says so in a way that keeps the base class's own signatures.
        Nested so it can close over *code_width* without a second parameter.
        """

        @staticmethod
        def _code(text: str) -> str:
            folded = html.escape(wrap_code(text, code_width))
            return f"<pre><code>{folded}</code></pre>\n"

        def fence(self, tokens, idx, options, env) -> str:
            return self._code(tokens[idx].content)

        def code_block(self, tokens, idx, options, env) -> str:
            return self._code(tokens[idx].content)

    parser = MarkdownIt(
        "commonmark", {"html": False, "linkify": False},
        renderer_cls=_FoldingRenderer,
    )
    parser.enable("table")
    parser.enable("strikethrough")
    return parser.render(markdown)


# ---------------------------------------------------------------------------
# HTML -> PDF
# ---------------------------------------------------------------------------


def render_pdf(
    source: Path | str,
    destination: Path | str | None = None,
    *,
    code_width: int = CODE_WIDTH,
    page_size: str = PAGE_SIZE,
    margin: float = PAGE_MARGIN,
) -> Path:
    """Render the Markdown file *source* to a PDF and return where it landed.

    *destination* defaults to *source* with a ``.pdf`` suffix — the PDF sits
    beside the Markdown it was made from, which is also where the report's
    ``dependency-graph.png`` is, so images resolve.

    Raises :class:`PdfRenderError` if the Markdown cannot be read or the PDF
    cannot be written; see this module's docstring for why this one does not
    degrade quietly.
    """
    origin = Path(source)
    try:
        markdown = origin.read_text(encoding="utf-8")
    except OSError as exc:
        raise PdfRenderError(f"could not read {origin}: {exc}") from exc
    return render_markdown_pdf(
        markdown,
        destination or origin.with_suffix(".pdf"),
        base_dir=origin.parent,
        code_width=code_width,
        page_size=page_size,
        margin=margin,
    )


def _lay_out(
    markdown: str,
    target: Any,
    label: str,
    *,
    base_dir: Path | str | None,
    code_width: int,
    page_size: str,
    margin: float,
    css: str | None,
) -> int:
    """Render *markdown* into *target* (a path or a file object) and return the
    page count.

    The one paging loop. It is shared rather than duplicated because the loop
    has a cap in it (:data:`_MAX_PAGES`) — a ``Story.place`` that keeps
    answering "more" would otherwise fill a disk, and a second copy of the loop
    is a second chance to omit the guard. *label* names the destination in the
    log and error messages, since a file object has no useful name.
    """
    try:
        import pymupdf
    except ImportError as exc:  # pragma: no cover - pymupdf is a core dependency
        raise PdfRenderError(f"PDF rendering needs pymupdf: {exc}") from exc

    body = markdown_to_html(markdown, code_width=code_width)
    try:
        mediabox = pymupdf.paper_rect(page_size)
        frame = mediabox + (margin, margin, -margin, -margin)
        archive = pymupdf.Archive(str(base_dir)) if base_dir else None
        story = pymupdf.Story(
            html=body, user_css=STYLESHEET + (css or ""), archive=archive
        )

        writer = pymupdf.DocumentWriter(target)
        pages = 0
        more = True
        while more:
            device = writer.begin_page(mediabox)
            more, _ = story.place(frame)
            story.draw(device)
            writer.end_page()
            pages += 1
            if pages >= _MAX_PAGES:
                logger.warning(
                    f"_lay_out: stopped at {_MAX_PAGES} pages; {label} is truncated"
                )
                break
        writer.close()
    except PdfRenderError:
        raise
    except Exception as exc:
        raise PdfRenderError(f"could not write {label}: {exc}") from exc
    return pages


def render_markdown_bytes(
    markdown: str,
    *,
    base_dir: Path | str | None = None,
    code_width: int = CODE_WIDTH,
    page_size: str = PAGE_SIZE,
    margin: float = PAGE_MARGIN,
    css: str | None = None,
) -> bytes:
    """Render *markdown* to PDF bytes, for a caller with nowhere to put a file.

    What :mod:`validation` wants: its report is uploaded to SharePoint, so it
    never touches the filesystem. Everything else — the stylesheet, the code
    folding, the page cap — is identical to the file-writing path.

    *css* is appended after the built-in stylesheet (later rules win), for a
    caller that wants to restyle the page.
    """
    buffer = io.BytesIO()
    pages = _lay_out(
        markdown,
        buffer,
        "<in-memory PDF>",
        base_dir=base_dir,
        code_width=code_width,
        page_size=page_size,
        margin=margin,
        css=css,
    )
    data = buffer.getvalue()
    logger.info(f"render_markdown_bytes: rendered {pages} page(s), {len(data)} bytes")
    return data


def render_markdown_pdf(
    markdown: str,
    destination: Path | str,
    *,
    base_dir: Path | str | None = None,
    code_width: int = CODE_WIDTH,
    page_size: str = PAGE_SIZE,
    margin: float = PAGE_MARGIN,
    css: str | None = None,
) -> Path:
    """Render *markdown* text to a PDF at *destination*, and return that path.

    *base_dir* is the directory relative image links resolve against — the
    directory the Markdown was (or would be) written to. Without one, images
    are simply absent from the PDF; the text is unaffected.
    """
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    pages = _lay_out(
        markdown,
        str(target),
        str(target),
        base_dir=base_dir,
        code_width=code_width,
        page_size=page_size,
        margin=margin,
        css=css,
    )
    logger.info(f"render_markdown_pdf: wrote {target} ({pages} page(s))")
    return target
