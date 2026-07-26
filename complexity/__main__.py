"""Score a directory of SAS files and write the complexity report as Markdown.

    python -m complexity path/to/sas_dir
    python -m complexity path/to/sas_dir --target pyspark --out report.md
    python -m complexity path/to/sas_dir --rules-path my_profile.json --top 25

Chunks every matching file, batches the whole corpus with
:class:`~chunker.batcher.MultiFileBatcher` so cross-file dataset/macro edges
resolve into shared batches, and scores the resulting batches and singletons.
Scoring the *batched* units — rather than raw chunks — reports on the same work
items the pipeline translates, so an estimate lines up with what a migration run
will actually do.

The report is :meth:`~complexity.models.CorpusComplexityReport.to_markdown`;
without ``--out`` it goes to stdout. Nothing here calls an LLM: the analysis is
entirely offline and never touches the network.

Logger name: ``complexity.__main__``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from chunker import MultiFileBatcher, SasCorpus, SasSemanticChunker

from .analyzer import ComplexityAnalyzer
from .rules import available_profiles

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m complexity",
        description=__doc__.splitlines()[0] if __doc__ else None,
    )
    parser.add_argument(
        "sas_dir",
        type=Path,
        help="Directory containing .sas files (searched recursively).",
    )
    parser.add_argument(
        "--pattern",
        default="*.sas",
        help="Glob for SAS files within sas_dir (default: *.sas).",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="Rule-set profile naming the output language "
        f"(available: {', '.join(sorted(available_profiles()))}). Defaults to "
        "config.json complexity.target, then the built-in default.",
    )
    parser.add_argument(
        "--rules-path",
        type=Path,
        default=None,
        help="A profile JSON of your own; wins over --target.",
    )
    parser.add_argument(
        "--size-anchor",
        type=float,
        default=None,
        help="Raw score of the reference MEDIUM file. Lowering it rates every "
        "file LARGER (default: config.json complexity.size_anchor, then the "
        "profile's own calibration).",
    )
    parser.add_argument(
        "--no-cross-file",
        action="store_true",
        help="Score each file as if it were alone, without resolving its "
        "macro/dataset/libref references against the rest of the corpus.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="How many hardest units to list in the report (default: 10).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write the Markdown report here instead of printing it.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    sas_files = sorted(args.sas_dir.rglob(args.pattern))
    if not sas_files:
        logger.error(f"no files matching {args.pattern!r} under {args.sas_dir}")
        return 1
    logger.info(f"scoring {len(sas_files)} SAS file(s) under {args.sas_dir}")

    chunker = SasSemanticChunker()
    corpus = SasCorpus(
        file_results=[chunker.chunk_file(str(path)) for path in sas_files]
    )
    batch_result = MultiFileBatcher().batch(corpus)

    analyzer = ComplexityAnalyzer(
        args.target,
        rules_path=args.rules_path,
        size_anchor=args.size_anchor,
        use_cross_file=False if args.no_cross_file else None,
    )
    # analyze_items directly rather than analyze_batch_result: a batch result
    # carries no diagnostics, and passing the corpus's own means the
    # uncertainty dimension scores on the full evidence instead of chunk-level
    # signals alone (see ComplexityAnalyzer.analyze_batch_result).
    report = analyzer.analyze_items(
        batch_result.all_ordered_items,
        source_ids=corpus.source_ids,
        diagnostics=corpus.all_diagnostics,
    )
    markdown = report.to_markdown(top=args.top)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(markdown, encoding="utf-8")
        logger.info(f"wrote complexity report: {args.out}")
        print(f"wrote complexity report: {args.out}")
    else:
        print(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
