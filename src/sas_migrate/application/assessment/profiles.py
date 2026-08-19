"""Profile inheritance and validation independent of storage."""

from __future__ import annotations

from typing import Any

from sas_migrate.application.ports.assessment import AssessmentProfileRepository

from .models import AssessmentProfile, ComplexityTier, ConstructRule


class AssessmentProfileError(ValueError):
    pass


def _merge(base: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    excluded = {"constructs", "flags", "weights", "construct_groups", "sizes"}
    merged = {**base, **{key: value for key, value in child.items() if key not in excluded}}
    merged["weights"] = {**base.get("weights", {}), **child.get("weights", {})}
    base_sizes = base.get("sizes", {}) or {}
    child_sizes = child.get("sizes", {}) or {}
    sizes = dict(base_sizes)
    for key, value in child_sizes.items():
        current = sizes.get(key)
        sizes[key] = {**current, **value} if isinstance(current, dict) and isinstance(value, dict) else value
    merged["sizes"] = sizes
    constructs = {key: dict(value) for key, value in base.get("constructs", {}).items()}
    for kind, entries in child.get("constructs", {}).items():
        constructs.setdefault(kind, {}).update(entries)
    merged["constructs"] = constructs
    by_name = {flag["name"]: flag for flag in base.get("flags", []) if "name" in flag}
    order = list(by_name)
    for flag in child.get("flags", []):
        name = flag.get("name")
        if not name:
            raise AssessmentProfileError("every profile flag requires a name")
        if name not in by_name:
            order.append(name)
        by_name[name] = {**by_name.get(name, {}), **flag}
    merged["flags"] = [by_name[name] for name in order]
    return merged


def _resolve(
    name: str,
    repository: AssessmentProfileRepository,
    seen: tuple[str, ...],
) -> dict[str, Any]:
    if name in seen:
        raise AssessmentProfileError(f"circular profile inheritance: {' -> '.join((*seen, name))}")
    try:
        document = repository.load(name)
    except (KeyError, OSError, ValueError) as exc:
        raise AssessmentProfileError(f"cannot load assessment profile {name!r}: {exc}") from exc
    if not isinstance(document, dict):
        raise AssessmentProfileError(f"assessment profile {name!r} must be an object")
    parent = document.get("extends")
    if parent is None:
        return dict(document)
    if not isinstance(parent, str):
        raise AssessmentProfileError(f"assessment profile {name!r} has a non-string extends")
    return _merge(_resolve(parent, repository, (*seen, name)), document)


def load_profile(name: str, repository: AssessmentProfileRepository) -> AssessmentProfile:
    document = _resolve(name, repository, ())
    constructs: dict[str, dict[str, ConstructRule]] = {}
    try:
        for kind, entries in document.get("constructs", {}).items():
            constructs[kind] = {
                entry_name.casefold(): ConstructRule.model_validate(rule)
                for entry_name, rule in entries.items()
            }
        weights = {
            ComplexityTier(key): float(value)
            for key, value in document.get("weights", {}).items()
        }
        for tier, default in (
            (ComplexityTier.LOW, 1.0),
            (ComplexityTier.MEDIUM, 2.5),
            (ComplexityTier.HIGH, 5.0),
        ):
            weights.setdefault(tier, default)
        return AssessmentProfile(
            name=name,
            target=str(document.get("target", name)),
            display_name=str(document.get("display_name", name)),
            description=str(document.get("description", "")),
            extends=document.get("extends"),
            weights=weights,
            sizes=document.get("sizes", {}),
            constructs=constructs,
            flags=tuple(document.get("flags", ())),
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise AssessmentProfileError(f"invalid assessment profile {name!r}: {exc}") from exc


__all__ = ["AssessmentProfileError", "load_profile"]
