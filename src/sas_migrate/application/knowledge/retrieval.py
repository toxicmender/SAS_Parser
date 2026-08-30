"""Deterministic construct and topical instruction retrieval."""

from __future__ import annotations

import math
import re
from collections import Counter

from sas_migrate.application.ports.knowledge import KnowledgeRanker, KnowledgeRepository

from .models import (
    KnowledgeChunk,
    KnowledgeRole,
    KnowledgeSelection,
    RetrievalQuery,
    RetrievalTier,
    RetrievedKnowledge,
    RuleScope,
    UserRule,
)
from .rules import UserRuleSet

_WORD = re.compile(r"[a-zA-Z0-9_]+")


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(match.group(0).casefold() for match in _WORD.finditer(text))


class KnowledgeRetriever:
    def __init__(
        self,
        repository: KnowledgeRepository,
        *,
        user_rules: UserRuleSet | None = None,
        topical_ranker: KnowledgeRanker | None = None,
    ) -> None:
        self._repository = repository
        self._user_rules = user_rules or UserRuleSet()
        self._topical_ranker = topical_ranker

    async def select(self, query: RetrievalQuery) -> KnowledgeSelection:
        corpus = tuple(
            chunk
            for chunk in await self._repository.chunks()
            if chunk.target is None or chunk.target is query.target
        )
        candidates: list[RetrievedKnowledge] = []
        candidates.extend(self._rule_results(self._user_rules.select(query), query))

        pinned = tuple(value.casefold() for value in query.pinned_sections)
        for chunk in corpus:
            if pinned and any(
                value in chunk.section_path.casefold() for value in pinned
            ):
                candidates.append(
                    RetrievedKnowledge(
                        chunk=chunk,
                        tier=RetrievalTier.PINNED,
                        score=1.0,
                        reasons=("pinned section",),
                    )
                )

        for chunk in corpus:
            chunk_constructs = {str(key) for key in chunk.construct_keys}
            matches = query.constructs.intersection(chunk_constructs)
            if not matches:
                continue
            match = min(matches)
            hazard = match in query.hazards
            candidates.append(
                RetrievedKnowledge(
                    chunk=chunk,
                    tier=RetrievalTier.HAZARD if hazard else RetrievalTier.CONSTRUCT,
                    score=1.0,
                    matched_construct=match,
                    reasons=(("hazard construct" if hazard else "construct match"),),
                )
            )

        candidates.extend(self._topical(corpus, query))
        order = {tier: index for index, tier in enumerate(RetrievalTier)}
        candidates.sort(
            key=lambda result: (
                order[result.tier],
                -result.score,
                result.chunk.ordinal,
                result.chunk.chunk_id,
            )
        )
        selected: list[RetrievedKnowledge] = []
        seen: set[str] = set()
        total_tokens = 0
        for result in candidates:
            if result.chunk.chunk_id in seen:
                continue
            if query.max_results and len(selected) >= query.max_results:
                break
            if total_tokens + result.chunk.token_count > query.max_tokens:
                continue
            selected.append(result)
            seen.add(result.chunk.chunk_id)
            total_tokens += result.chunk.token_count
        return KnowledgeSelection(
            query=query,
            results=tuple(selected),
            components=tuple(result.to_prompt_component() for result in selected),
            selected_tokens=total_tokens,
        )

    def _topical(
        self,
        corpus: tuple[KnowledgeChunk, ...],
        query: RetrievalQuery,
    ) -> tuple[RetrievedKnowledge, ...]:
        if self._topical_ranker is not None:
            limit = max(4 * query.max_results, query.max_results)
            rankings = self._topical_ranker.rank(
                " ".join((query.text, *sorted(query.topics))),
                corpus,
                limit=limit or None,
            )
            chunks = {chunk.chunk_id: chunk for chunk in corpus}
            return tuple(
                RetrievedKnowledge(
                    chunk=chunks[ranking.chunk_id],
                    tier=(
                        RetrievalTier.USER_TOPIC
                        if chunks[ranking.chunk_id].role
                        is KnowledgeRole.USER_INSTRUCTION
                        else RetrievalTier.TOPICAL
                    ),
                    score=ranking.score,
                    reasons=tuple(
                        f"topical {signal.value} match" for signal in ranking.signals
                    ),
                )
                for ranking in rankings
                if ranking.chunk_id in chunks
            )
        query_tokens = _tokens(" ".join((query.text, *query.topics)))
        if not query_tokens or not corpus:
            return ()
        documents = tuple(
            _tokens(f"{chunk.section_path} {chunk.text}") for chunk in corpus
        )
        document_frequency = Counter(
            token for document in documents for token in set(document)
        )
        query_counts = Counter(query_tokens)
        results: list[RetrievedKnowledge] = []
        for chunk, document in zip(corpus, documents, strict=True):
            counts = Counter(document)
            score = 0.0
            for token, query_count in query_counts.items():
                if token not in counts:
                    continue
                inverse = (
                    math.log((len(corpus) + 1) / (document_frequency[token] + 0.5)) + 1
                )
                score += query_count * inverse * counts[token] / (counts[token] + 1.2)
            if score <= 0:
                continue
            results.append(
                RetrievedKnowledge(
                    chunk=chunk,
                    tier=(
                        RetrievalTier.USER_TOPIC
                        if chunk.role is KnowledgeRole.USER_INSTRUCTION
                        else RetrievalTier.TOPICAL
                    ),
                    score=score,
                    reasons=("topical lexical match",),
                )
            )
        return tuple(results)

    def _rule_results(
        self,
        rules: tuple[UserRule, ...],
        query: RetrievalQuery,
    ) -> tuple[RetrievedKnowledge, ...]:
        results: list[RetrievedKnowledge] = []
        for ordinal, rule in enumerate(rules):
            tier = (
                RetrievalTier.USER_ALWAYS
                if rule.scope is RuleScope.ALWAYS
                else RetrievalTier.USER_WHEN
                if rule.scope is RuleScope.CONDITIONAL
                else RetrievalTier.USER_TOPIC
            )
            chunk = KnowledgeChunk(
                chunk_id=rule.rule_id,
                source_id="user-rules",
                role=KnowledgeRole.USER_INSTRUCTION,
                section_path=f"Project rule {ordinal + 1}",
                text=rule.text,
                page_start=1,
                page_end=1,
                ordinal=ordinal,
                token_count=max(1, len(_tokens(rule.text))),
                target=rule.target or query.target,
            )
            results.append(
                RetrievedKnowledge(
                    chunk=chunk,
                    tier=tier,
                    score=1.0,
                    reasons=(f"scoped user rule: {rule.scope.value}",),
                )
            )
        return tuple(results)


__all__ = ["KnowledgeRetriever"]
