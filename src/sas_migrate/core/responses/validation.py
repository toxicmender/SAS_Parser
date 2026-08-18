"""Mandatory target and syntax validation for normalized responses."""

from __future__ import annotations

import ast
from collections.abc import Collection

from ..targets.models import ResolvedTarget
from ..targets.registry import SPARK_SQL
from ..targets.validation import (
    ResponseValidationResult,
    TargetIssueCode,
    TargetValidationIssue,
)
from .models import TranslationCell, TranslationCellKind, TranslationDocument
from .normalization import normalize_raw_response


def _syntax_error(language: str, source: str) -> str | None:
    if language == "python":
        try:
            ast.parse(source)
        except SyntaxError as exc:
            return f"{exc.msg} (line {exc.lineno})"
        return None

    dialect = SPARK_SQL.sqlglot_dialect
    if dialect is None:  # guarded by TargetDefinition, retained for type narrowing
        raise RuntimeError("Spark SQL has no registered SQLGlot dialect")

    import sqlglot
    from sqlglot.errors import ParseError

    try:
        statements = sqlglot.parse(source, read=dialect)
    except ParseError as exc:
        return f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
    if not statements or any(statement is None for statement in statements):
        return "SQL parser produced no complete statement"
    return None


def _effective_language(cell: TranslationCell, target: ResolvedTarget) -> str:
    return cell.language or target.canonical_language


class ResponseTargetValidator:
    """Always-on publication validator shared by structured and raw responses."""

    def validate(
        self,
        document: TranslationDocument,
        target: ResolvedTarget,
        *,
        known_chunk_ids: Collection[str],
        normalization_issues: Collection[TargetValidationIssue] = (),
    ) -> ResponseValidationResult:
        issues = list(normalization_issues)
        known = set(known_chunk_ids)
        code_cells = document.code_cells

        if document.target is not target.target:
            issues.append(
                TargetValidationIssue(
                    code=TargetIssueCode.TARGET_MISMATCH,
                    message=(
                        f"document reported {document.target.value!r}; "
                        f"resolved target is {target.target.value!r}"
                    ),
                )
            )

        non_empty = [cell for cell in code_cells if cell.source.strip()]
        if not non_empty:
            issues.append(
                TargetValidationIssue(
                    code=TargetIssueCode.EMPTY_CODE,
                    message="response contains no non-empty target code cell",
                )
            )

        effective_languages: set[str] = set()
        for index, cell in enumerate(document.cells):
            if cell.kind is not TranslationCellKind.CODE:
                continue
            language = _effective_language(cell, target)
            effective_languages.add(language)
            if cell.language is not None and cell.language != target.canonical_language:
                issues.append(
                    TargetValidationIssue(
                        code=TargetIssueCode.FOREIGN_LANGUAGE,
                        message=(
                            f"code cell declares {cell.language!r}; "
                            f"{target.display_name} requires "
                            f"{target.canonical_language!r}"
                        ),
                        cell_index=index,
                        chunk_id=cell.chunk_id,
                    )
                )

            if cell.chunk_id is not None and cell.chunk_id not in known:
                issues.append(
                    TargetValidationIssue(
                        code=TargetIssueCode.UNKNOWN_CHUNK,
                        message=f"code cell names unknown chunk {cell.chunk_id!r}",
                        cell_index=index,
                        chunk_id=cell.chunk_id,
                    )
                )
            elif len(known) > 1 and cell.chunk_id is None:
                issues.append(
                    TargetValidationIssue(
                        code=TargetIssueCode.UNKNOWN_CHUNK,
                        message="multi-member response code cell has no chunk attribution",
                        cell_index=index,
                    )
                )

            if cell.source.strip():
                error = _syntax_error(language, cell.source)
                if error is not None:
                    issues.append(
                        TargetValidationIssue(
                            code=TargetIssueCode.SYNTAX_ERROR,
                            message=f"{language} syntax check failed: {error}",
                            cell_index=index,
                            chunk_id=cell.chunk_id,
                        )
                    )

        if len(effective_languages) > 1:
            issues.append(
                TargetValidationIssue(
                    code=TargetIssueCode.MIXED_TARGETS,
                    message=(
                        "response mixes target code languages: "
                        f"{', '.join(sorted(effective_languages))}"
                    ),
                )
            )

        round_trip = normalize_raw_response(
            document.to_markdown(default_fence=target.fence),
            target,
        ).document
        original_code = tuple(
            (cell.source.strip("\r\n"), _effective_language(cell, target))
            for cell in code_cells
        )
        round_trip_code = tuple(
            (cell.source.strip("\r\n"), _effective_language(cell, target))
            for cell in round_trip.code_cells
        )
        if round_trip.target is not document.target or round_trip_code != original_code:
            issues.append(
                TargetValidationIssue(
                    code=TargetIssueCode.ROUND_TRIP_MISMATCH,
                    message="canonical Markdown changed response target identity or code",
                )
            )

        return ResponseValidationResult(
            valid=not issues,
            resolved_target=target.target,
            reported_target=document.target,
            issues=tuple(issues),
        )


__all__ = ["ResponseTargetValidator"]
