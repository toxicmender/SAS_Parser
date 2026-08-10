# reporting

Shared Markdown-to-PDF rendering utilities for deliverable reports. It is a
leaf package, depending only on its rendering libraries, so both `complexity`
and `validation` can use the same paging and stylesheet implementation without
an import cycle or renderer drift.

`render_markdown_pdf()` writes a PDF, `render_markdown_bytes()` returns PDF
bytes for upload paths, and `markdown_to_html()` / `wrap_code()` expose the
intermediate formatting helpers. `PdfRenderError` identifies rendering
failures.

Logger names follow `reporting.*`.
