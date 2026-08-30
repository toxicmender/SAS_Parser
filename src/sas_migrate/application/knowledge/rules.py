"""Scoped project-instruction parsing and selection."""

from __future__ import annotations

import hashlib
import re

from sas_migrate.core.targets import TargetId

from .models import RetrievalQuery, RuleScope, UserRule

_HEADING = re.compile(r"^##+\s+(?P<title>.+?)\s*$", re.MULTILINE)
_TAG = re.compile(r"\[(?P<key>[a-z_-]+)(?::(?P<value>[^\]]+))?\]", re.IGNORECASE)


def _values(raw: str) -> frozenset[str]:
    return frozenset(
        value.strip().casefold().replace("-", "_").replace(" ", "_")
        for value in raw.split(",")
        if value.strip()
    )


class UserRuleSet:
    def __init__(self, rules: tuple[UserRule, ...] = ()) -> None:
        self.rules = rules

    @classmethod
    def from_markdown(cls, text: str, *, source_id: str = "user") -> UserRuleSet:
        matches = list(_HEADING.finditer(text))
        if not matches:
            if not text.strip():
                return cls()
            digest = hashlib.sha256(f"{source_id}\0{text}".encode()).hexdigest()[:16]
            return cls(
                (
                    UserRule(
                        rule_id=f"{source_id}::{digest}",
                        text=text.strip(),
                    ),
                )
            )
        rules: list[UserRule] = []
        for index, match in enumerate(matches):
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            if not body:
                continue
            title = match.group("title")
            tags = {
                tag.group("key").casefold().replace("-", "_"): (
                    tag.group("value") or ""
                )
                for tag in _TAG.finditer(title)
            }
            constructs = _values(tags.get("when", ""))
            kinds = _values(tags.get("kind", ""))
            flags = _values(tags.get("meta", ""))
            topics = _values(tags.get("topic", ""))
            if constructs or kinds or flags:
                scope = RuleScope.CONDITIONAL
            elif "topic" in tags:
                scope = RuleScope.TOPIC
                topics = topics or _values(re.sub(_TAG, "", title))
            else:
                scope = RuleScope.ALWAYS
            target_value = next(iter(_values(tags.get("target", ""))), None)
            target = TargetId(target_value) if target_value else None
            digest = hashlib.sha256(
                f"{source_id}\0{index}\0{title}\0{body}".encode()
            ).hexdigest()[:16]
            rules.append(
                UserRule(
                    rule_id=f"{source_id}::{digest}",
                    text=body,
                    scope=scope,
                    constructs=constructs,
                    chunk_kinds=kinds,
                    metadata_flags=flags,
                    topics=topics,
                    target=target,
                    priority=-index,
                )
            )
        return cls(tuple(rules))

    def select(self, query: RetrievalQuery) -> tuple[UserRule, ...]:
        selected: list[UserRule] = []
        query_terms = _values(" ".join((query.text, *query.topics)))
        for rule in self.rules:
            if not rule.enabled or (
                rule.target is not None and rule.target is not query.target
            ):
                continue
            if rule.scope is RuleScope.CONDITIONAL:
                if rule.constructs and not rule.constructs.intersection(
                    query.constructs
                ):
                    continue
                if rule.chunk_kinds and not rule.chunk_kinds.intersection(
                    query.chunk_kinds
                ):
                    continue
                if rule.metadata_flags and not rule.metadata_flags.intersection(
                    query.metadata_flags
                ):
                    continue
            elif rule.scope is RuleScope.TOPIC and not rule.topics.intersection(
                query_terms
            ):
                continue
            selected.append(rule)
        return tuple(sorted(selected, key=lambda rule: (-rule.priority, rule.rule_id)))


__all__ = ["UserRuleSet"]
