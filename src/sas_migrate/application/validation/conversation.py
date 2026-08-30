"""Thread/transcript reconstruction into the one validation run contract."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from pydantic import Field

from sas_migrate.core.models import ContractModel
from sas_migrate.core.targets import TargetId

from .models import EvaluationRun, ValidationUnit


class ConversationTurn(ContractModel):
    turn_id: str = Field(min_length=1)
    prompt: str
    response: str
    source: str = ""
    retrieval_context: tuple[str, ...] = ()


def run_from_transcript(
    run_id: str,
    target: TargetId,
    turns: Iterable[ConversationTurn | Mapping[str, object]],
    **memory: object,
) -> EvaluationRun:
    parsed = tuple(
        turn if isinstance(turn, ConversationTurn) else ConversationTurn.model_validate(turn)
        for turn in turns
    )
    return EvaluationRun.model_validate(
        {
            "run_id": run_id,
            "target": target,
            "units": tuple(
                ValidationUnit(
                    unit_id=turn.turn_id,
                    prompt=turn.prompt,
                    response=turn.response,
                    source=turn.source,
                    retrieval_context=turn.retrieval_context,
                )
                for turn in parsed
            ),
            **memory,
        }
    )


__all__ = ["ConversationTurn", "run_from_transcript"]
