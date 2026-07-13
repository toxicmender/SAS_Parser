"""Optional LLM-as-judge metric. See validation/README.md.

Kept out of :func:`validation.metrics.default_metrics` on purpose: it needs a
judge chat model (API key / network), and the default suite must stay
runnable offline. Wire it in explicitly::

    from llm_client import LLMClient, LLMClientConfig
    from validation import LLMJudgeMetric, ValidationRunner, default_metrics

    judge = LLMJudgeMetric(llm=LLMClient(LLMClientConfig(model="claude-haiku-4-5-20251001")))
    runner = ValidationRunner(pipeline, metrics=[*default_metrics(), judge])

Any object with a LangChain-style ``invoke(input) -> message`` works as the
judge — an :class:`llm_client.LLMClient` (which adds the retry / token-budget
layers), a raw chat model, or a fake in tests.

Logger name: ``validation.judge``.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from chunker.models import SasBatch, SasChunk

from .metrics import ValidationMetric
from .models import CaseRun, MetricResult

logger = logging.getLogger(__name__)

_SCORE_RE = re.compile(r"SCORE:\s*([1-5])")

_JUDGE_TEMPLATE = """\
You are grading a machine translation of SAS code into {output_language}.

Original SAS source:
```sas
{sas_source}
```

Candidate translation:
{translation}

Grade the translation on functional equivalence: does it read the same
inputs, apply the same transformations, and produce the same outputs as the
SAS source? Ignore stylistic differences.

Rubric:
5 = functionally equivalent, nothing missing
4 = equivalent except trivial details (formatting, labels)
3 = core logic right, a secondary behaviour missing or wrong
2 = major logic missing or wrong
1 = unrelated to the source, or no usable code

Reply with exactly one line: SCORE: <1-5>
"""


def _item_source(item: SasBatch | SasChunk) -> str:
    if isinstance(item, SasBatch):
        return "\n".join(c.text for c in item.chunks)
    return item.text


class LLMJudgeMetric(ValidationMetric):
    """
    Grade each item's translation 1–5 with a judge model, normalised to
    [0, 1] (``(n - 1) / 4``) and averaged over the case's items. An
    unparseable judge reply scores that item 0 rather than raising — a run
    should finish even when the judge misbehaves.
    """

    name = "llm_judge"
    default_threshold = 0.6

    def __init__(
        self,
        llm: Any,
        *,
        threshold: float | None = None,
        output_language: str = "PySpark",
    ) -> None:
        super().__init__(threshold)
        self._llm = llm
        self._output_language = output_language

    def evaluate(self, run: CaseRun) -> MetricResult:
        if not run.items:
            return self._result(0.0, "chunker/batcher produced no items")
        scores: list[float] = []
        unparseable = 0
        for item, response in zip(run.items, run.responses):
            prompt = _JUDGE_TEMPLATE.format(
                output_language=self._output_language,
                sas_source=_item_source(item),
                translation=response,
            )
            reply = self._llm.invoke(prompt)
            reply_text = str(getattr(reply, "content", reply))
            match = _SCORE_RE.search(reply_text)
            if match is None:
                logger.warning(
                    f"llm_judge: unparseable judge reply for case "
                    f"'{run.case.case_id}': {reply_text[:80]!r}"
                )
                unparseable += 1
                scores.append(0.0)
                continue
            scores.append((int(match.group(1)) - 1) / 4)
        details = f"mean of {len(scores)} judged item(s)"
        if unparseable:
            details += f"; {unparseable} unparseable repl(y/ies) scored 0"
        return self._result(sum(scores) / len(scores), details)
