"""
test_complexity.py — unit tests for the complexity analysis package
(zero LLM, zero disk I/O).

Run:  python -m pytest tests/test_complexity.py -v
"""

import importlib.util
import json
import math
import pathlib
import shutil
import sys
import tempfile
import unittest
import unittest.mock
from dataclasses import replace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from chunker import MultiFileBatcher, SasChunkBatcher, SasSemanticChunker
from chunker.models import SasCorpus
from complexity import (
    CROSS_FILE_CONSTRUCTS,
    ComplexityAnalyzer,
    ComplexityTier,
    CrossFileIndex,
    DependencyGraph,
    PdfRenderError,
    TranslationParity,
    TShirtSize,
    build_evaluation_prompt,
    chunk_texts,
    detect_constructs,
    display_name,
    display_names,
    evaluate_report,
    evaluation_prompts,
    max_size,
    max_tier,
    render_file_report,
    render_overall_report,
    render_pdf,
    render_png,
    resolve_name,
    sort_by_complexity,
    worst_parity,
    write_reports,
)
from complexity.pdf import markdown_to_html, wrap_code
from complexity import rules, sizing


def _corpus(**files: str) -> SasCorpus:
    """Chunk each ``name=source`` pair into a corpus, keyed by ``<name>.sas``."""
    chunker = SasSemanticChunker()
    return SasCorpus(
        file_results=[
            chunker.chunk_text(src, source_id=f"{name}.sas")
            for name, src in files.items()
        ]
    )


def _file(report, source_id: str):
    """The FileComplexity for *source_id*, failing loudly if absent."""
    for f in report.files:
        if f.source_id == source_id:
            return f
    raise AssertionError(
        f"no FileComplexity for {source_id!r}; have {[f.source_id for f in report.files]}"
    )


def _cross_names(scored) -> set[str]:
    """Cross-file construct names on a scored unit, without the prefix."""
    return {
        s.name.removeprefix("cross_file:")
        for s in scored.signals
        if s.source == "cross_file"
    }


def _analyze(source: str, **kwargs):
    """Chunk *source* and return the CorpusComplexityReport for its chunks."""
    result = SasSemanticChunker().chunk_text(source, source_id="t.sas")
    return ComplexityAnalyzer(**kwargs).analyze_result(result)


def _only(source: str, **kwargs):
    """Analyze a source expected to produce exactly one chunk."""
    report = _analyze(source, **kwargs)
    if len(report.chunks) != 1:
        raise AssertionError(
            f"expected 1 chunk, got {len(report.chunks)}: "
            f"{[c.chunk_id for c in report.chunks]}"
        )
    return report.chunks[0]


def _names(scored) -> set[str]:
    return {s.name for s in scored.signals}


class TestLowTier(unittest.TestCase):
    """Simple SQL and macro variables are LOW."""

    def test_macro_variable_is_low(self):
        scored = _only("%let cutoff = 100;\n")
        self.assertEqual(scored.tier, ComplexityTier.LOW)
        self.assertEqual(scored.translation_difficulty, TranslationParity.DIRECT)
        self.assertIn("global_statement:let", _names(scored))

    def test_simple_proc_sql_is_low(self):
        scored = _only(
            "proc sql;\n  create table work.out as select * from work.in;\nquit;\n"
        )
        self.assertEqual(scored.tier, ComplexityTier.LOW)
        self.assertEqual(scored.translation_difficulty, TranslationParity.DIRECT)
        self.assertIn("proc:sql", _names(scored))

    def test_plain_data_step_is_low(self):
        scored = _only("data work.out;\n  set work.in;\n  x = y + 1;\nrun;\n")
        self.assertEqual(scored.tier, ComplexityTier.LOW)

    def test_macro_variable_reference_is_low(self):
        scored = _only("proc sql;\n  select * from work.a where x > &cut;\nquit;\n")
        self.assertEqual(scored.tier, ComplexityTier.LOW)
        self.assertIn("macro-var-reference", _names(scored))


class TestMediumTier(unittest.TestCase):
    """Hashing, MERGE, SFTP, and mail are MEDIUM."""

    def test_match_merge_with_by_is_medium(self):
        scored = _only("data work.out;\n  merge work.a work.b;\n  by id;\nrun;\n")
        self.assertEqual(scored.tier, ComplexityTier.MEDIUM)
        self.assertEqual(scored.translation_difficulty, TranslationParity.PARTIAL)
        self.assertIn("merge", _names(scored))
        self.assertNotIn("merge_no_by", _names(scored))

    def test_hash_object_is_medium(self):
        scored = _only(
            "data work.out;\n  set work.in;\n"
            '  declare hash h(dataset: "work.lookup");\n'
            "  h.definekey('id');\n  h.definedone();\nrun;\n"
        )
        self.assertEqual(scored.tier, ComplexityTier.MEDIUM)
        self.assertIn("component_object:hash", _names(scored))

    def test_hashing_function_is_medium(self):
        scored = _only("data work.out;\n  set work.in;\n  k = md5(name);\nrun;\n")
        self.assertEqual(scored.tier, ComplexityTier.MEDIUM)
        self.assertIn("function:md5", _names(scored))

    def test_filename_sftp_is_medium(self):
        scored = _only("filename xfer sftp 'o.csv' host='h.example.com';\n")
        self.assertEqual(scored.tier, ComplexityTier.MEDIUM)
        self.assertIn("filename_sftp", _names(scored))

    def test_filename_email_is_medium(self):
        scored = _only('filename m email "ops@example.com" subject="done";\n')
        self.assertEqual(scored.tier, ComplexityTier.MEDIUM)
        self.assertIn("filename_email", _names(scored))

    def test_call_symput_is_medium(self):
        scored = _only(
            "data _null_;\n  set work.in;\n  call symput('n', put(x, 8.));\nrun;\n"
        )
        self.assertEqual(scored.tier, ComplexityTier.MEDIUM)
        self.assertIn("call_routine:symput", _names(scored))

    def test_proc_transpose_is_medium(self):
        scored = _only("proc transpose data=work.a out=work.b;\nrun;\n")
        self.assertEqual(scored.tier, ComplexityTier.MEDIUM)
        self.assertIn("proc:transpose", _names(scored))


class TestHighTier(unittest.TestCase):
    """Arrays, DO loops, and %MACRO definitions are HIGH."""

    def test_array_is_high(self):
        scored = _only(
            "data work.out;\n  set work.in;\n  array s{12} s1-s12;\n  s{1} = 0;\nrun;\n"
        )
        self.assertEqual(scored.tier, ComplexityTier.HIGH)
        self.assertEqual(scored.translation_difficulty, TranslationParity.HARD)
        self.assertIn("array", _names(scored))

    def test_iterative_do_loop_is_high(self):
        scored = _only(
            "data work.out;\n  set work.in;\n  do i = 1 to 10;\n    t + i;\n  end;\nrun;\n"
        )
        self.assertEqual(scored.tier, ComplexityTier.HIGH)
        self.assertIn("do_loop", _names(scored))

    def test_do_while_is_high(self):
        scored = _only(
            "data work.out;\n  set work.in;\n  do while (x < 10);\n    x + 1;\n  end;\nrun;\n"
        )
        self.assertEqual(scored.tier, ComplexityTier.HIGH)
        self.assertIn("do_while", _names(scored))

    def test_do_until_is_high(self):
        scored = _only(
            "data work.out;\n  set work.in;\n  do until (x >= 10);\n    x + 1;\n  end;\nrun;\n"
        )
        self.assertEqual(scored.tier, ComplexityTier.HIGH)
        self.assertIn("do_until", _names(scored))

    def test_macro_definition_is_high_and_manual(self):
        scored = _only(
            "%macro build(ds=);\n  data out;\n    set &ds;\n  run;\n%mend build;\n"
        )
        self.assertEqual(scored.tier, ComplexityTier.HIGH)
        self.assertEqual(scored.translation_difficulty, TranslationParity.MANUAL)
        self.assertIn("kind:MACRO_DEFINITION", _names(scored))

    def test_one_to_one_merge_without_by_is_high(self):
        """Essentials Ch. 21: a BY-less MERGE pairs rows by position, with no
        key variable — there is no Spark join that reproduces it."""
        scored = _only("data work.out;\n  merge work.a work.b;\nrun;\n")
        self.assertEqual(scored.tier, ComplexityTier.HIGH)
        self.assertEqual(scored.translation_difficulty, TranslationParity.HARD)
        self.assertIn("merge_no_by", _names(scored))
        self.assertNotIn("merge", _names(scored))

    def test_call_execute_is_high(self):
        scored = _only(
            "data _null_;\n  set work.in;\n  call execute('%report');\nrun;\n"
        )
        self.assertEqual(scored.tier, ComplexityTier.HIGH)
        self.assertIn("call_routine:execute", _names(scored))


class TestAggregationRules(unittest.TestCase):
    """Tier is the max present; difficulty is the worst present."""

    def test_mixed_chunk_takes_max_tier_and_worst_parity(self):
        # A MERGE (MEDIUM/PARTIAL) plus an ARRAY (HIGH/HARD) in one step.
        scored = _only(
            "data work.out;\n  merge work.a work.b;\n  by id;\n"
            "  array s{3} s1-s3;\nrun;\n"
        )
        self.assertEqual(scored.tier, ComplexityTier.HIGH)
        self.assertEqual(scored.translation_difficulty, TranslationParity.HARD)
        # Both signals survive — the MEDIUM one is not discarded.
        self.assertIn("merge", _names(scored))
        self.assertIn("array", _names(scored))

    def test_single_high_construct_outweighs_many_low_ones(self):
        scored = _only(
            "data work.out;\n  set work.in;\n"
            "  a = 1; b = 2; c = 3; d = 4; e = 5;\n"
            "  array s{2} s1-s2;\nrun;\n"
        )
        self.assertEqual(scored.tier, ComplexityTier.HIGH)

    def test_helpers_default_to_floor_on_empty(self):
        self.assertEqual(max_tier([]), ComplexityTier.LOW)
        self.assertEqual(worst_parity([]), TranslationParity.DIRECT)

    def test_helpers_pick_extremes(self):
        self.assertEqual(
            max_tier([ComplexityTier.LOW, ComplexityTier.HIGH, ComplexityTier.MEDIUM]),
            ComplexityTier.HIGH,
        )
        self.assertEqual(
            worst_parity([TranslationParity.DIRECT, TranslationParity.MANUAL, TranslationParity.PARTIAL]),
            TranslationParity.MANUAL,
        )

    def test_repeated_construct_counted_once_in_score(self):
        one = _only("data work.out;\n  set work.in;\n  array a{2} a1-a2;\nrun;\n")
        many = _only(
            "data work.out;\n  set work.in;\n"
            "  array a{2} a1-a2;\n  array b{2} b1-b2;\n  array c{2} c1-c2;\nrun;\n"
        )
        # Same construct type, so the same score — repetition is verbosity.
        self.assertEqual(one.score, many.score)
        self.assertEqual(len([s for s in many.signals if s.name == "array"]), 1)
        # ...but the evidence records that it fired more than once.
        array_signal = next(s for s in many.signals if s.name == "array")
        self.assertIn("×3", array_signal.evidence)

    def test_unrecognised_constructs_contribute_nothing(self):
        # `zzz(...)` is not a SAS function and must not inflate anything.
        scored = _only("data work.out;\n  set work.in;\n  x = zzz(y);\nrun;\n")
        self.assertEqual(scored.tier, ComplexityTier.LOW)
        self.assertEqual(scored.score, 0.0)
        self.assertEqual(scored.signals, [])
        self.assertIn("no complexity signals", scored.rationale)


class TestDetectors(unittest.TestCase):
    """The supplementary scans, in isolation."""

    def _found(self, source: str) -> set[str]:
        return {c.name for c in detect_constructs(source)}

    def test_detects_core_constructs(self):
        found = self._found(
            "data x;\n  merge a b;\n  by id;\n  array s{3} s1-s3;\n  retain t 0;\n"
            "  do i = 1 to 3;\n    t + s{i};\n  end;\n"
            "  if first.id then flag = 1;\nrun;\n"
        )
        self.assertEqual(
            {"merge", "array", "retain", "do_loop", "by_group_first_last"},
            found & {"merge", "array", "retain", "do_loop", "by_group_first_last"},
        )

    def test_merge_split_keys_off_the_by_statement(self):
        with_by = self._found("data x;\n  merge a b;\n  by id;\nrun;\n")
        without_by = self._found("data x;\n  merge a b;\nrun;\n")
        self.assertIn("merge", with_by)
        self.assertNotIn("merge_no_by", with_by)
        self.assertIn("merge_no_by", without_by)
        self.assertNotIn("merge", without_by)

    def test_by_before_merge_does_not_count(self):
        """A BY belonging to an earlier SET does not make a later MERGE a
        match-merge; only a BY *after* the MERGE does."""
        found = self._found("data x;\n  set a;\n  by id;\n  merge b c;\nrun;\n")
        self.assertIn("merge_no_by", found)

    def test_macro_do_is_not_a_data_step_do_loop(self):
        found = self._found("%macro m;\n  %do i = 1 %to 10;\n  %end;\n%mend;\n")
        self.assertNotIn("do_loop", found)

    def test_macro_do_while_is_not_a_data_step_do_while(self):
        found = self._found("%macro m;\n  %do %while (&i < 10);\n  %end;\n%mend;\n")
        self.assertNotIn("do_while", found)

    def test_macro_goto_is_not_a_data_step_goto(self):
        self.assertNotIn("data_goto", self._found("%macro m;\n%goto done;\n%mend;\n"))

    def test_constructs_in_comments_do_not_fire(self):
        self.assertEqual(
            set(),
            self._found("/* array s{3} s1-s3; merge a b; do i = 1 to 3; */\n"),
        )

    def test_constructs_in_string_literals_do_not_fire(self):
        found = self._found("data x;\n  msg = 'array s{3}; merge a b;';\nrun;\n")
        self.assertNotIn("array", found)
        self.assertNotIn("merge", found)

    def test_plain_do_block_is_not_a_loop(self):
        # `if ... then do; ... end;` is a block, not an iteration.
        found = self._found("data x;\n  if a then do;\n    b = 1;\n  end;\nrun;\n")
        self.assertNotIn("do_loop", found)
        self.assertNotIn("do_while", found)
        self.assertNotIn("do_until", found)

    def test_filename_access_methods(self):
        self.assertIn("filename_sftp", self._found("filename f sftp 'a';"))
        self.assertIn("filename_email", self._found("filename f email 'a';"))
        self.assertIn("filename_url", self._found("filename f url 'a';"))
        self.assertIn("filename_pipe", self._found("filename f pipe 'ls';"))

    def test_plain_filename_has_no_access_method_signal(self):
        found = self._found("filename f '/tmp/out.txt';")
        self.assertFalse({n for n in found if n.startswith("filename_")})

    def test_filename_word_does_not_trigger_file_output(self):
        self.assertNotIn("file_output", self._found("filename f sftp 'a';"))

    def test_every_detector_name_has_a_catalogue_entry(self):
        """A detector with no rules entry would be silently dropped."""
        source = (
            "data x;\n  merge a b;\n  by k;\n  merge e f;\n"
            "  modify c;\n  update d;\n  array s{2} s1-s2;\n"
            "  retain t;\n  do i = 1 to 2;\n  end;\n  do while (a);\n  end;\n"
            "  do until (b);\n  end;\n  if first.id;\n  infile 'r.txt';\n"
            "  file print;\n  link sub;\n  goto top;\nrun;\n"
            "filename a sftp 'x'; filename b email 'y'; filename c url 'z';\n"
            "filename d pipe 'p'; filename e ftp 'q'; filename g socket 'h';\n"
        )
        for profile in rules.available_profiles():
            ruleset = rules.load_ruleset(profile)
            for construct in detect_constructs(source):
                self.assertIsNotNone(
                    ruleset.spec("detector", construct.name),
                    f"detector '{construct.name}' has no 'detector' entry in "
                    f"profile {profile!r}",
                )


class TestBatchAggregation(unittest.TestCase):
    SOURCE = (
        "data work.base;\n  set work.raw;\nrun;\n\n"
        "data work.final;\n  set work.base;\n  array s{3} s1-s3;\n"
        "  do i = 1 to 3;\n    t + s{i};\n  end;\nrun;\n"
    )

    def test_batch_takes_worst_member_tier_and_sums_scores(self):
        result = SasSemanticChunker().chunk_text(self.SOURCE, source_id="t.sas")
        batch_result = SasChunkBatcher().batch(result)
        report = ComplexityAnalyzer().analyze_batch_result(batch_result)

        self.assertTrue(report.batches, "expected the two steps to batch together")
        batch = report.batches[0]
        self.assertEqual(batch.tier, ComplexityTier.HIGH)
        self.assertEqual(batch.translation_difficulty, TranslationParity.HARD)
        self.assertEqual(len(batch.members), 2)
        self.assertAlmostEqual(batch.score, sum(m.score for m in batch.members))
        self.assertIn("t.sas", batch.source_files)

    def test_batch_signals_union_members(self):
        result = SasSemanticChunker().chunk_text(self.SOURCE, source_id="t.sas")
        report = ComplexityAnalyzer().analyze_batch_result(
            SasChunkBatcher().batch(result)
        )
        names = {s.name for s in report.batches[0].signals}
        self.assertIn("array", names)
        self.assertIn("do_loop", names)


class TestReport(unittest.TestCase):
    SOURCE = (
        "%let cut = 5;\n\n"
        "data work.m;\n  merge work.a work.b;\n  by id;\nrun;\n\n"
        "%macro build(ds=);\n  data o;\n    set &ds;\n  run;\n%mend build;\n"
    )

    def setUp(self):
        self.report = _analyze(self.SOURCE)

    def test_tier_counts_cover_all_tiers(self):
        counts = self.report.tier_counts
        self.assertEqual({"LOW", "MEDIUM", "HIGH"}, set(counts))
        self.assertEqual(len(self.report.items), sum(counts.values()))

    def test_overall_tier_and_difficulty_are_worst_case(self):
        self.assertEqual(self.report.overall_tier, ComplexityTier.HIGH)
        self.assertEqual(self.report.overall_difficulty, TranslationParity.MANUAL)

    def test_total_score_sums_items(self):
        self.assertAlmostEqual(
            self.report.total_score,
            round(sum(i.score for i in self.report.items), 3),
        )

    def test_hardest_orders_by_tier_then_parity(self):
        hardest = self.report.hardest(3)
        self.assertEqual(hardest[0].tier, ComplexityTier.HIGH)
        tiers = [i.tier for i in hardest]
        self.assertEqual(tiers, sorted(tiers, key=_tier_key, reverse=True))

    def test_hardest_respects_limit(self):
        self.assertEqual(len(self.report.hardest(2)), 2)

    def test_to_markdown_renders_summary_and_table(self):
        md = self.report.to_markdown(top=3)
        self.assertIn("# SAS chunk complexity report", md)
        self.assertIn("Overall tier", md)
        self.assertIn("| Tier | Units |", md)
        self.assertIn("| Item | Tier | Spark parity | Score | Drivers |", md)
        self.assertIn("t.sas", md)

    def test_sort_by_complexity_matches_hardest(self):
        self.assertEqual(
            [id(i) for i in sort_by_complexity(self.report.items)],
            [id(i) for i in self.report.hardest(len(self.report.items))],
        )


def _tier_key(tier):
    from complexity import tier_rank

    return tier_rank(tier)


class TestAnalyzerOptions(unittest.TestCase):
    ARRAY_STEP = "data work.out;\n  set work.in;\n  array s{3} s1-s3;\nrun;\n"

    def test_detectors_can_be_disabled(self):
        with_detectors = _only(self.ARRAY_STEP)
        without = _only(self.ARRAY_STEP, use_detectors=False)
        self.assertEqual(with_detectors.tier, ComplexityTier.HIGH)
        self.assertEqual(without.tier, ComplexityTier.LOW)
        self.assertNotIn("array", _names(without))

    def test_weight_overrides_change_score_not_tier(self):
        default = _only(self.ARRAY_STEP)
        heavy = _only(self.ARRAY_STEP, weight_high=100.0)
        self.assertEqual(default.tier, heavy.tier)
        self.assertGreater(heavy.score, default.score)

    def test_signal_carries_evidence_and_source(self):
        scored = _only(self.ARRAY_STEP)
        array_signal = next(s for s in scored.signals if s.name == "array")
        self.assertEqual(array_signal.source, "detector")
        self.assertTrue(array_signal.evidence)

        proc = _only("proc sort data=work.a;\nrun;\n")
        sort_signal = next(s for s in proc.signals if s.name == "proc:sort")
        self.assertEqual(sort_signal.source, "metadata")

    def test_detector_evidence_does_not_shadow_the_catalogue_note(self):
        """The standing guidance is usually more useful than the snippet, so
        both must survive on the signal."""
        scored = _only(self.ARRAY_STEP)
        array_signal = next(s for s in scored.signals if s.name == "array")
        self.assertIn("array s", array_signal.evidence.lower())
        self.assertIn("not a Spark ArrayType", array_signal.note)
        self.assertIn(array_signal.evidence, array_signal.detail)
        self.assertIn(array_signal.note, array_signal.detail)

    def test_metadata_signal_carries_note_without_evidence(self):
        proc = _only("proc sort data=work.a;\nrun;\n")
        sort_signal = next(s for s in proc.signals if s.name == "proc:sort")
        self.assertEqual(sort_signal.evidence, "")
        self.assertTrue(sort_signal.note)
        self.assertEqual(sort_signal.detail, sort_signal.note)

    def test_high_signals_expose_the_drivers(self):
        scored = _only(
            "data work.out;\n  merge work.a work.b;\n  by id;\n"
            "  array s{2} s1-s2;\nrun;\n"
        )
        self.assertEqual({s.name for s in scored.high_signals}, {"array"})

    def test_categories_are_sorted_and_distinct(self):
        scored = _only(self.ARRAY_STEP)
        self.assertEqual(scored.categories, sorted(set(scored.categories)))


class TestTShirtSize(unittest.TestCase):
    """Banding, the Fibonacci scale, the anchor, and the kind floors."""

    def test_scale_is_fibonacci_and_ordered(self):
        self.assertEqual(
            [s.points for s in TShirtSize], [2, 3, 5, 8]
        )
        self.assertEqual(
            [s.label for s in TShirtSize],
            ["Small", "Medium", "Large", "Extra Large"],
        )

    def test_only_extra_large_needs_breakdown(self):
        """XL is an instruction, not just a magnitude."""
        for size in TShirtSize:
            self.assertEqual(
                size.needs_breakdown, size is TShirtSize.EXTRA_LARGE, size
            )

    def test_max_size_is_worst_case_and_floors_at_small(self):
        self.assertEqual(max_size([]), TShirtSize.SMALL)
        self.assertEqual(
            max_size([TShirtSize.SMALL, TShirtSize.LARGE, TShirtSize.MEDIUM]),
            TShirtSize.LARGE,
        )

    def test_anchor_dimensions_band_to_medium(self):
        """The anchor is the reference Medium file, by definition.

        Scored from its *dimension split*, not its total: each dimension is
        rescaled against its own window, so the same 87.5 spent entirely on
        effort would not land in the same place.
        """
        for name in rules.available_profiles():
            sizes = rules.load_ruleset(name).sizes
            self.assertIsNotNone(
                sizes.anchor_dimensions, f"{name} states no anchor split"
            )
            points = sizes.points_for(*sizes.anchor_dimensions)
            self.assertEqual(sizes.band_for(points), TShirtSize.MEDIUM, name)
            self.assertAlmostEqual(points, 3.0, delta=0.1, msg=name)

    def test_banding_is_monotonic_in_raw_score(self):
        sizes = rules.load_ruleset("sparksql").sizes
        seen = [
            rules.load_ruleset("sparksql").sizes.band_for(sizes.points_for(raw))
            for raw in (1, 5, 10, 20, 30, 50, 90)
        ]
        ranks = [__import__("complexity").size_rank(s) for s in seen]
        self.assertEqual(ranks, sorted(ranks), seen)

    def test_lowering_the_anchor_makes_files_larger(self):
        """Sizes are relative to the anchor, so the anchor is the master knob."""
        source = "data a; set b; run;\nproc sort data=a; by id; run;\n"
        big = _analyze(source, size_anchor=1.0).files[0]
        small = _analyze(source, size_anchor=200.0).files[0]
        self.assertGreater(big.points, small.points)
        self.assertEqual(small.size, TShirtSize.SMALL)
        self.assertEqual(big.size, TShirtSize.EXTRA_LARGE)

    def test_macro_definition_is_never_small(self):
        """A tiny %MACRO is never Small, however little it contains."""
        f = _analyze("%macro noop;\n%mend noop;\n").files[0]
        self.assertGreaterEqual(
            __import__("complexity").size_rank(f.size),
            __import__("complexity").size_rank(TShirtSize.MEDIUM),
        )

    def test_kind_floor_binds_and_names_itself(self):
        """A short %MACRO bands below Medium on volume, so the floor acts."""
        f = _analyze("%macro noop;\n%mend noop;\n").files[0]
        self.assertEqual(f.size, TShirtSize.MEDIUM)
        self.assertEqual(f.floored_by, "MACRO_DEFINITION")
        # The floor is what did it, not the banding: the un-snapped position
        # is still below the Small/Medium boundary.
        self.assertLess(f.continuous_points, 2.5)
        # The reported estimate follows the *floored* size, so the label and
        # the number agree. Reporting the banding's 2.x under a Medium label
        # is exactly what a rung-valued estimate exists to prevent.
        self.assertEqual(f.points, 3.0)

    def test_config_only_file_stays_small(self):
        f = _analyze("%let a = 1;\n%let b = 2;\n%let c = 3;\n").files[0]
        self.assertEqual(f.size, TShirtSize.SMALL)

    def test_floored_by_is_empty_when_banding_stands_alone(self):
        """The floor is only reported when it actually changed the answer."""
        source = "%macro noop;\n%mend noop;\n" + "".join(
            f"data out{i}; set in{i}; run;\n" for i in range(40)
        )
        f = _analyze(source).files[0]
        # Bands above the MACRO_DEFINITION floor on its own volume.
        self.assertEqual(f.size, TShirtSize.LARGE)
        self.assertEqual(f.floored_by, "")

    def test_volume_alone_can_drive_a_large_size(self):
        """A long file of trivial steps raises no signal, but is still work.

        This is the case a tier cannot express: every step here is LOW/DIRECT,
        so a presence-based scale reads the file as trivial.
        """
        source = "".join(
            f"data out{i}; set in{i}; run;\n" for i in range(40)
        )
        f = _analyze(source).files[0]
        self.assertEqual(f.tier, ComplexityTier.LOW)
        self.assertGreater(f.effort_raw, 20)
        self.assertIn(f.size, (TShirtSize.LARGE, TShirtSize.EXTRA_LARGE))

    def test_volume_alone_can_reach_extra_large(self):
        """Bulk on its own must be able to ask to be broken down.

        The effort weight reaches past the Extra Large boundary by itself, so
        a file of nothing but trivial steps still rates XL once there are
        enough of them — an enormous file needs splitting however plain its
        contents are. This is why the dimension weights are reaches summed and
        clamped, not shares averaged.
        """
        f = _analyze(
            "".join(f"data out{i}; set in{i}; run;\n" for i in range(80))
        ).files[0]
        self.assertEqual(f.tier, ComplexityTier.LOW)
        self.assertEqual(f.complexity_raw, 0.0)
        self.assertEqual(f.size, TShirtSize.EXTRA_LARGE)

    def test_volume_saturates_at_the_top_of_its_window(self):
        """Past the ceiling, more of the same stops moving the number."""
        big = _analyze(
            "".join(f"data out{i}; set in{i}; run;\n" for i in range(200))
        ).files[0]
        bigger = _analyze(
            "".join(f"data out{i}; set in{i}; run;\n" for i in range(400))
        ).files[0]
        self.assertEqual(big.effort_norm, 1.0)
        self.assertGreater(bigger.effort_raw, big.effort_raw)
        self.assertEqual(bigger.points, big.points)


def _reference_file_source() -> str:
    """The file each profile's ``anchor.describes`` names, written out.

    ~200 lines: a %MACRO wrapping two DATA steps, nine further DATA steps, a
    match-merge, a PROC SORT, a PROC SUMMARY, two LIBNAMEs used throughout.
    """
    lines = ['libname edw "/mnt/edw";', 'libname mart "/mnt/mart";', ""]
    lines += ["%macro prep(ds=, out=);", "  data &out._stg;", "    set &ds;"]
    lines += [f"    length c{i} $8;" for i in range(10)]
    lines += ["  run;", "  data &out;", "    set &out._stg;"]
    lines += [f'    x{i} = c{i} || "_t";' for i in range(10)]
    lines += ["  run;", "%mend prep;", "", "%prep(ds=edw.raw, out=work.p);", ""]
    for step in range(9):
        lines += [f"data work.s{step};", "  set work.p;"]
        lines += [f"  y{step}_{j} = x{j} * {j + 1};" for j in range(12)]
        lines += ["run;", ""]
    lines += ["data work.matched;", "  merge work.s0 work.s1;", "  by id;"]
    lines += [f"  m{j} = y0_{j} + y1_{j};" for j in range(10)]
    lines += ["run;", ""]
    lines += ["proc sort data=work.matched;", "  by id;", "run;", ""]
    lines += [
        "proc summary data=work.matched;",
        "  var m0 m1 m2;",
        "  output out=mart.agg mean=;",
        "run;",
    ]
    return "\n".join(lines)


class TestDimensionRescale(unittest.TestCase):
    """The log + min-max rescale that turns raw dimensions into points."""

    def setUp(self):
        self.sizes = rules.load_ruleset("sparksql").sizes

    def test_window_is_anchor_relative(self):
        """Bounds are multiples of the anchor, so the anchor moves them all."""
        lo, hi = self.sizes.window_for("effort")
        halved = replace(self.sizes, anchor_raw=self.sizes.anchor_raw / 2)
        self.assertEqual(halved.window_for("effort"), (lo / 2, hi / 2))

    def test_min_max_clamps_at_both_ends(self):
        lo, hi = self.sizes.window_for("effort")
        self.assertEqual(self.sizes.normalize("effort", 0.0), 0.0)
        self.assertEqual(self.sizes.normalize("effort", lo), 0.0)
        self.assertEqual(self.sizes.normalize("effort", lo / 2), 0.0)
        self.assertEqual(self.sizes.normalize("effort", hi), 1.0)
        self.assertEqual(self.sizes.normalize("effort", hi * 100), 1.0)

    def test_normalisation_is_monotonic(self):
        lo, hi = self.sizes.window_for("effort")
        seen = [
            self.sizes.normalize("effort", raw)
            for raw in (lo, lo * 1.5, lo * 2, lo * 3, hi)
        ]
        self.assertEqual(seen, sorted(seen))

    def test_the_log_makes_returns_diminish(self):
        """Equal *increments* of raw effort buy less the further up you are.

        The point of the log: the 200th step of a file tells you less than the
        20th. Under a plain rescale these two gaps would be identical.
        """
        n = lambda raw: self.sizes.normalize("effort", raw)  # noqa: E731
        low_gap = n(80) - n(50)
        high_gap = n(140) - n(110)
        self.assertGreater(low_gap, high_gap)

    def test_equal_ratios_are_equal_steps(self):
        """...and the flip side: a doubling is a doubling wherever it happens."""
        n = lambda raw: self.sizes.normalize("effort", raw)  # noqa: E731
        self.assertAlmostEqual(n(80) - n(40), n(160) - n(80), places=2)

    def test_points_span_the_scale_ends(self):
        """A min-max rescale bottoms out at SMALL and tops out at EXTRA_LARGE."""
        self.assertEqual(self.sizes.points_for(0.0, 0.0, 0.0), 2.0)
        self.assertEqual(self.sizes.points_for(1e6, 1e6, 1e6), 8.0)

    def test_weights_are_reaches_not_shares(self):
        """Each dimension pushes on its own; the blend clamps rather than dilutes.

        A weighted mean would cap a volume-only file at the effort weight and
        would dock every file that has no unknowns in it.
        """
        weights = self.sizes.dimension_weights
        self.assertGreater(sum(weights.values()), 1.0)
        self.assertGreaterEqual(
            weights["effort"], self.sizes.band_blends()[TShirtSize.LARGE]
        )
        self.assertEqual(self.sizes.points_for(1e6, 1e6, 1e6), 8.0)

    def test_uncertainty_adds_rather_than_taking_a_share(self):
        clean = self.sizes.points_for(50.0, 37.5, 0.0)
        troubled = self.sizes.points_for(50.0, 37.5, 20.0)
        self.assertGreater(troubled, clean)

    def test_points_rescale_geometrically(self):
        """Log space, so the rungs stay Fibonacci-shaped rather than even."""
        half = self.sizes.points_for(*self._raw_for_blend(0.5))
        self.assertAlmostEqual(half, 4.0, places=1)  # sqrt(2 * 8), not (2 + 8)/2

    def _raw_for_blend(self, blend: float) -> tuple[float, float, float]:
        """Raw dimensions whose blend is *blend*, via effort alone."""
        share = blend / self.sizes.dimension_weights["effort"]
        lo, hi = self.sizes.window_for("effort")
        return (
            math.expm1(
                math.log1p(lo) + share * (math.log1p(hi) - math.log1p(lo))
            ),
            0.0,
            0.0,
        )

    def test_dimension_weights_decide_the_mix(self):
        """A dimension weighted to zero cannot move a size."""
        ignored = replace(
            self.sizes,
            dimension_weights={"effort": 1.0, "complexity": 0.0, "uncertainty": 0.0},
        )
        self.assertEqual(
            ignored.points_for(50.0, 0.0, 0.0), ignored.points_for(50.0, 500.0, 0.0)
        )
        self.assertGreater(
            self.sizes.points_for(50.0, 500.0, 0.0),
            self.sizes.points_for(50.0, 0.0, 0.0),
        )

    def test_file_reports_both_raw_and_normalised_dimensions(self):
        f = _analyze(
            "".join(f"data out{i}; set in{i}; run;\n" for i in range(200))
        ).files[0]
        self.assertGreater(f.effort_raw, 100)
        self.assertEqual(f.effort_norm, 1.0)
        self.assertEqual(f.complexity_norm, 0.0)
        self.assertAlmostEqual(
            f.blend,
            round(
                rules.load_ruleset("sparksql").sizes.blend_for(
                    f.effort_raw, f.complexity_raw, f.uncertainty_raw
                ),
                3,
            ),
        )

    def test_markdown_shows_the_normalised_share(self):
        md = _analyze("data a; set b; run;\n").to_markdown()
        self.assertIn("| Blend |", md)


class TestFibonacciStoryPoints(unittest.TestCase):
    """A reported estimate is always a planning-poker deck entry."""

    SOURCES = (
        "%let a = 1;\n",
        "%macro noop;\n%mend noop;\n",
        "".join(f"data out{i}; set in{i}; run;\n" for i in range(12)),
        _reference_file_source(),
        "".join(f"data out{i}; set in{i}; run;\n" for i in range(50)),
        "".join(f"data out{i}; set in{i}; run;\n" for i in range(80)),
    )

    def test_every_file_reports_a_fibonacci_number(self):
        for source in self.SOURCES:
            f = _analyze(source).files[0]
            self.assertIn(f.points, (2.0, 3.0, 5.0, 8.0), source[:30])

    def test_points_are_the_size_rung(self):
        """The number and the label always agree, floors included."""
        expected = {
            TShirtSize.SMALL: 2.0,
            TShirtSize.MEDIUM: 3.0,
            TShirtSize.LARGE: 5.0,
            TShirtSize.EXTRA_LARGE: 8.0,
        }
        seen = set()
        for source in self.SOURCES:
            f = _analyze(source).files[0]
            self.assertEqual(f.points, expected[f.size], source[:30])
            seen.add(f.size)
        # The corpus above spans the scale, so this is not vacuous.
        self.assertGreaterEqual(len(seen), 3, seen)

    def test_the_continuous_position_is_kept_alongside(self):
        """It still ranks two files inside one rung, which points cannot."""
        small = _analyze(self.SOURCES[0]).files[0]
        busier = _analyze(self.SOURCES[2]).files[0]
        self.assertEqual(small.points, busier.points)
        self.assertLess(small.continuous_points, busier.continuous_points)

    def test_total_points_sums_deck_entries(self):
        report = ComplexityAnalyzer().analyze_corpus(
            _corpus(a=self.SOURCES[0], b=self.SOURCES[3], c=self.SOURCES[5])
        )
        self.assertEqual(
            report.total_points, round(sum(f.points for f in report.files), 1)
        )
        for f in report.files:
            self.assertIn(f.points, sizing.FIBONACCI_POINTS)

    def test_nearest_deck_entry_is_geometric(self):
        """4 sits nearer 5 than 3 on a scale whose steps are ratios."""
        self.assertEqual(sizing._nearest_fibonacci(4.0), 5.0)
        self.assertEqual(sizing._nearest_fibonacci(3.4), 3.0)
        self.assertEqual(sizing._nearest_fibonacci(0.1), 1.0)
        self.assertEqual(sizing._nearest_fibonacci(10_000.0), 377.0)
        for entry in sizing.FIBONACCI_POINTS:
            self.assertEqual(sizing._nearest_fibonacci(entry), entry)

    def test_default_rungs_are_the_profiles_own(self):
        sizes = rules.load_ruleset("sparksql").sizes
        self.assertEqual(
            [sizes.points_for_size(s) for s in TShirtSize], [2.0, 3.0, 5.0, 8.0]
        )

    def test_a_redenominated_scale_keeps_its_ends_and_snaps_the_middle(self):
        sizes = replace(
            rules.load_ruleset("sparksql").sizes,
            min_story_points=1.0,
            max_story_points=13.0,
        )
        self.assertEqual(
            [sizes.points_for_size(s) for s in TShirtSize], [1.0, 2.0, 5.0, 13.0]
        )

    def test_rungs_stay_strictly_increasing_on_a_narrow_scale(self):
        """Too narrow for four deck entries: monotonicity wins over Fibonacci."""
        sizes = replace(
            rules.load_ruleset("sparksql").sizes,
            min_story_points=1.0,
            max_story_points=2.0,
        )
        rungs = [sizes.points_for_size(s) for s in TShirtSize]
        self.assertEqual(rungs, sorted(rungs))
        self.assertEqual(len(set(rungs)), 4, rungs)
        self.assertEqual((rungs[0], rungs[-1]), (1.0, 2.0))


class TestStoryPointRange(unittest.TestCase):
    """`min_story_points` / `max_story_points` re-denominate, never re-rate."""

    SOURCES = (
        "%let a = 1;\n",
        "".join(f"data out{i}; set in{i}; run;\n" for i in range(12)),
        _reference_file_source(),
        "".join(f"data out{i}; set in{i}; run;\n" for i in range(50)),
        "".join(f"data out{i}; set in{i}; run;\n" for i in range(80)),
    )

    def test_defaults_to_the_profiles_scale(self):
        sizes = rules.load_ruleset("sparksql").sizes
        self.assertEqual(sizes.story_point_range, (2.0, 8.0))

    def test_range_moves_the_reported_points(self):
        for source in self.SOURCES:
            wide = _analyze(source, min_story_points=1, max_story_points=13).files[0]
            self.assertGreaterEqual(wide.points, 1.0)
            self.assertLessEqual(wide.points, 13.0)

    def test_range_does_not_move_a_single_size(self):
        """The bands are fractions of the span, so the verdicts are identical."""
        for source in self.SOURCES:
            default = _analyze(source).files[0]
            wide = _analyze(source, min_story_points=1, max_story_points=13).files[0]
            self.assertEqual(wide.size, default.size, source[:30])
            self.assertEqual(wide.blend, default.blend, source[:30])

    def test_either_end_can_be_set_alone(self):
        f = _analyze(self.SOURCES[2], max_story_points=100).files[0]
        self.assertEqual(rules.load_ruleset("sparksql").sizes.story_point_range[0], 2.0)
        self.assertGreater(f.points, 3.0)
        self.assertEqual(f.size, TShirtSize.MEDIUM)

    def test_range_is_read_from_the_profile(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        doc = json.loads(
            rules.profile_path("sparksql").read_text(encoding="utf-8")
        )
        doc["sizes"]["story_points"] = {"min": 1, "max": 13}
        path = pathlib.Path(tmp) / "wide.json"
        path.write_text(json.dumps(doc), encoding="utf-8")
        sizes = rules.load_ruleset(path=str(path), use_cache=False).sizes
        self.assertEqual(sizes.story_point_range, (1.0, 13.0))
        self.assertEqual(sizes.points_for(0.0, 0.0, 0.0), 1.0)
        self.assertEqual(sizes.points_for(1e6, 1e6, 1e6), 13.0)
        self.assertEqual(
            sizes.band_for(sizes.points_for(*sizes.anchor_dimensions)),
            TShirtSize.MEDIUM,
        )

    def test_a_zero_minimum_is_rejected(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        doc = json.loads(
            rules.profile_path("sparksql").read_text(encoding="utf-8")
        )
        doc["sizes"]["story_points"] = {"min": 0, "max": 8}
        path = pathlib.Path(tmp) / "zero.json"
        path.write_text(json.dumps(doc), encoding="utf-8")
        with self.assertRaises(rules.RuleSetError):
            rules.load_ruleset(path=str(path), use_cache=False)


def _extra_large_source() -> str:
    """A file that is bulky *and* hard — the only thing that rates Extra Large.

    Volume alone saturates the effort window and tops out at Large, so an XL
    fixture has to spend the complexity dimension too: a macro carrying arrays,
    every DO form, LAG/DIF, run-time macro resolution and procedural jumps,
    then forty-five merge/retain steps behind it.
    """
    lines = [
        "%macro run_all(ds=, out=);",
        "  %if &ds ne %then %do;",
        "    data &out;",
        "      set &ds;",
        "      array s{5} s1-s5;",
        "      do i = 1 to 5;",
        "        s{i} = lag(s{i});",
        "      end;",
        "      do while (i < 3); i + 1; end;",
        "      do until (i > 9); i + 1; end;",
        '      p = dif(x); q = symget("v"); r = resolve("&v");',
        '      call execute("data _null_; run;");',
        '      call symput("k", x);',
        "      link fix;",
        "      return;",
        "      fix: y = 1;",
        "    run;",
        "  %end;",
        "%mend run_all;",
        'filename pipe_in pipe "ls -l";',
        'filename mailer email "a@b.c";',
        "proc fcmp outlib=work.f.g; run;",
        "proc iml; quit;",
    ]
    for i in range(45):
        lines += [
            f"data work.g{i};",
            f"  merge work.a{i} work.b{i};",
            "  retain acc;",
            f"  h = md5(put(x{i}, 8.));",
            "  if first.id then acc = 0;",
            "run;",
        ]
    return "\n".join(lines)


class TestAnchorCalibration(unittest.TestCase):
    """The anchor must stay the measured score of the file it describes.

    Without this, ``anchor.describes`` decays into a story about a number that
    no longer follows from it — and since every size is relative to the anchor,
    a drifted anchor silently re-rates the whole corpus.
    """

    def test_reference_file_measures_its_profile_anchor(self):
        result = SasSemanticChunker().chunk_text(
            _reference_file_source(), source_id="reference.sas"
        )
        for name in rules.available_profiles():
            scored = ComplexityAnalyzer(target=name).analyze_result(result).files[0]
            expected = rules.load_ruleset(name).sizes.anchor_raw
            self.assertAlmostEqual(
                scored.raw_total,
                expected,
                delta=1.0,
                msg=(
                    f"{name}: reference file measures {scored.raw_total}, but the "
                    f"profile anchors at {expected}. Re-measure and update the "
                    f"profile, or fix anchor.describes to match reality."
                ),
            )

    def test_reference_file_measures_its_anchor_dimensions(self):
        """The split, not just the total — each dimension has its own window."""
        result = SasSemanticChunker().chunk_text(
            _reference_file_source(), source_id="reference.sas"
        )
        for name in rules.available_profiles():
            scored = ComplexityAnalyzer(target=name).analyze_result(result).files[0]
            declared = rules.load_ruleset(name).sizes.anchor_dimensions
            measured = (
                scored.effort_raw,
                scored.complexity_raw,
                scored.uncertainty_raw,
            )
            for dimension, want, got in zip(rules.DIMENSIONS, declared, measured):
                self.assertAlmostEqual(
                    got,
                    want,
                    delta=1.0,
                    msg=(
                        f"{name}: reference file measures {dimension}={got}, but "
                        f"anchor.dimensions declares {want}"
                    ),
                )

    def test_reference_file_is_medium_against_every_target(self):
        """It is the definition of Medium, so it must read Medium everywhere."""
        result = SasSemanticChunker().chunk_text(
            _reference_file_source(), source_id="reference.sas"
        )
        for name in rules.available_profiles():
            scored = ComplexityAnalyzer(target=name).analyze_result(result).files[0]
            self.assertEqual(scored.size, TShirtSize.MEDIUM, name)

    def test_every_anchor_documents_itself(self):
        for name in rules.available_profiles():
            self.assertTrue(
                rules.load_ruleset(name).sizes.anchor_describes.strip(),
                f"{name} anchors on an undocumented number",
            )


class TestSizeDimensions(unittest.TestCase):
    """Effort, complexity, and uncertainty move a size independently."""

    def test_three_dimensions_sum_to_raw_total(self):
        f = _analyze("data a; set b; array x{3} p1-p3; run;\n").files[0]
        self.assertAlmostEqual(
            f.raw_total,
            round(f.effort_raw + f.complexity_raw + f.uncertainty_raw, 3),
        )

    def test_complexity_dimension_responds_to_parity_not_just_tier(self):
        """Same tier, different target: PySpark rates the macro more kindly."""
        source = "%macro build(ds=);\n  data o; set &ds; run;\n%mend build;\n"
        sql = _analyze(source, target="sparksql").files[0]
        py = _analyze(source, target="pyspark").files[0]
        self.assertEqual(sql.tier, py.tier)
        self.assertGreater(sql.complexity_raw, py.complexity_raw)

    def test_uncertainty_counts_unclosed_blocks(self):
        f = _analyze("%macro broken;\n  data x; set y; run;\n").files[0]
        self.assertGreater(f.uncertainty_raw, 0)

    def test_unresolved_libref_feeds_uncertainty(self):
        f = _analyze("data out; set ghost.tbl; run;\n").files[0]
        self.assertIn("libref_unresolved", _cross_names(f))
        self.assertGreater(f.uncertainty_raw, 0)

    def test_unmatched_dataset_is_not_uncertainty(self):
        """An unwritten input is a source table, not a missing dependency."""
        corpus = _corpus(
            one="data a; set external_src; run;\n",
            two="data b; set c; run;\n",
        )
        report = ComplexityAnalyzer().analyze_corpus(corpus)
        f = _file(report, "one.sas")
        self.assertIn("dataset_unresolved", _cross_names(f))
        self.assertEqual(f.uncertainty_raw, 0.0)

    def test_batch_result_path_flags_incomplete_uncertainty(self):
        """SasBatchResult carries no diagnostics; that is recorded, not hidden."""
        result = SasSemanticChunker().chunk_text(
            "data a; set b; run;\n", source_id="t.sas"
        )
        batched = ComplexityAnalyzer().analyze_batch_result(
            SasChunkBatcher().batch(result)
        )
        direct = ComplexityAnalyzer().analyze_result(result)
        self.assertFalse(batched.files[0].uncertainty_complete)
        self.assertTrue(direct.files[0].uncertainty_complete)


class TestCrossFile(unittest.TestCase):
    """Resolving references against the rest of the corpus."""

    LIB = "%macro build(ds=);\n  data out; set &ds; run;\n%mend build;\n%let env = prod;\n"
    JOB = "%build(ds=work.raw);\ndata final; set out; where e = \"&env\"; run;\n"

    def setUp(self):
        self.report = ComplexityAnalyzer().analyze_corpus(
            _corpus(lib=self.LIB, job=self.JOB)
        )

    def test_consumer_imports_and_producer_exports(self):
        job = _cross_names(_file(self.report, "job.sas"))
        lib = _cross_names(_file(self.report, "lib.sas"))
        self.assertIn("macro_import", job)
        self.assertIn("macrovar_import", job)
        self.assertIn("macro_export", lib)
        self.assertIn("macrovar_export", lib)

    def test_dependency_direction_is_recorded(self):
        job = _file(self.report, "job.sas").cross_file
        lib = _file(self.report, "lib.sas").cross_file
        assert job is not None and lib is not None
        self.assertEqual(job.depends_on, ["lib.sas"])
        self.assertEqual(lib.depended_on_by, ["job.sas"])
        self.assertTrue(job.is_coupled)

    def test_export_does_not_raise_the_producer_tier(self):
        """Being depended on is scheduling effort, not translation difficulty."""
        lib = _file(self.report, "lib.sas")
        exports = [
            s
            for s in lib.signals
            if s.name.startswith("cross_file:") and s.name.endswith("_export")
        ]
        self.assertTrue(exports)
        for signal in exports:
            self.assertEqual(signal.tier, ComplexityTier.LOW)
            self.assertEqual(signal.parity, TranslationParity.DIRECT)

    def test_same_file_reference_raises_nothing(self):
        """A macro defined and called in one file is not a cross-file ref."""
        report = ComplexityAnalyzer().analyze_result(
            SasSemanticChunker().chunk_text(
                self.LIB + "%build(ds=work.raw);\n", source_id="solo.sas"
            )
        )
        self.assertNotIn("macro_import", _cross_names(report.files[0]))
        self.assertNotIn("macro_unresolved", _cross_names(report.files[0]))

    def test_libref_assigned_in_another_chunk_of_the_same_file(self):
        """The LIBNAME is its own chunk, so file scope is what matters."""
        report = _analyze('libname mylib "/d";\ndata mylib.o; set mylib.i; run;\n')
        self.assertNotIn("libref_unresolved", _cross_names(report.files[0]))

    def test_missing_macro_is_unresolved_only_with_a_corpus_to_search(self):
        """HIGH/MANUAL requires having actually looked somewhere."""
        multi = ComplexityAnalyzer().analyze_corpus(
            _corpus(one="%ghost(x=1);\n", two="data z; set y; run;\n")
        )
        names = _cross_names(_file(multi, "one.sas"))
        self.assertIn("macro_unresolved", names)
        self.assertNotIn("macro_external", names)

        solo = _analyze("%ghost(x=1);\n")
        solo_names = _cross_names(solo.files[0])
        self.assertIn("macro_external", solo_names)
        self.assertNotIn("macro_unresolved", solo_names)

    def test_unresolved_macro_is_manual_but_external_is_not(self):
        multi = ComplexityAnalyzer().analyze_corpus(
            _corpus(one="%ghost(x=1);\n", two="data z; set y; run;\n")
        )
        self.assertEqual(
            _file(multi, "one.sas").translation_difficulty, TranslationParity.MANUAL
        )
        self.assertEqual(
            _analyze("%ghost(x=1);\n").files[0].translation_difficulty,
            TranslationParity.PARTIAL,
        )

    def test_standard_autocall_macros_are_never_flagged(self):
        """%left and friends ship with SAS; calling one is not a dependency."""
        report = ComplexityAnalyzer().analyze_corpus(
            _corpus(one="%let x = %left(  5 );\n", two="data z; set y; run;\n")
        )
        names = _cross_names(_file(report, "one.sas"))
        self.assertNotIn("macro_unresolved", names)
        self.assertNotIn("macro_external", names)

    def test_default_librefs_are_never_flagged(self):
        report = _analyze("data work.a; set sashelp.class; run;\n")
        self.assertNotIn("libref_unresolved", _cross_names(report.files[0]))

    def test_use_cross_file_off_drops_every_cross_file_signal(self):
        report = ComplexityAnalyzer(use_cross_file=False).analyze_corpus(
            _corpus(lib=self.LIB, job=self.JOB)
        )
        self.assertEqual(_cross_names(_file(report, "job.sas")), set())
        self.assertIsNone(_file(report, "job.sas").cross_file)

    def test_lone_chunk_analysis_raises_no_cross_file_signal(self):
        """analyze_chunk with no index has no corpus to resolve against."""
        chunk = SasSemanticChunker().chunk_text(
            "%ghost(x=1);\n", source_id="t.sas"
        ).chunks[0]
        self.assertEqual(_cross_names(ComplexityAnalyzer().analyze_chunk(chunk)), set())

    def test_include_is_left_to_the_existing_metadata_signals(self):
        """crossfile.py adds no %INCLUDE signal of its own.

        The chunker already surfaces %INCLUDE through both the INCLUDE chunk
        kind and the ``includes`` flag, and the catalogue rates both
        MEDIUM/PARTIAL — exactly what a cross-file entry would assign. Adding a
        third would inflate the score without adding information.
        """
        report = _analyze('%include "other.sas";\n')
        self.assertEqual(_cross_names(report.files[0]), set())
        self.assertNotIn("include", CROSS_FILE_CONSTRUCTS)

    def test_chunk_ids_do_not_collide_across_files(self):
        """Both files start at chunk-001; refs must not leak between them."""
        index = CrossFileIndex.build(_corpus(lib=self.LIB, job=self.JOB).all_chunks)
        lib = index.profile_for("lib.sas")
        assert lib is not None
        # lib.sas defines everything it uses; only job.sas depends outward.
        self.assertEqual(lib.depends_on, [])
        self.assertEqual(lib.depended_on_by, ["job.sas"])


class TestFileComplexity(unittest.TestCase):
    """The file rollup, and how it renders."""

    def test_batch_spanning_two_files_still_yields_two_file_rollups(self):
        corpus = _corpus(
            a="data shared.mid; set shared.raw; run;\n",
            b="data shared.out; set shared.mid; run;\n",
        )
        batched = MultiFileBatcher().batch(corpus)
        report = ComplexityAnalyzer().analyze_batch_result(batched)
        self.assertEqual(
            sorted(f.source_id for f in report.files), ["a.sas", "b.sas"]
        )

    def test_total_points_sums_files(self):
        report = ComplexityAnalyzer().analyze_corpus(
            _corpus(a="data a; set b; run;\n", b="%macro m;\n%mend m;\n")
        )
        self.assertAlmostEqual(
            report.total_points, round(sum(f.points for f in report.files), 1)
        )

    def test_overall_size_is_the_largest_file(self):
        report = ComplexityAnalyzer().analyze_corpus(
            _corpus(a="%let x = 1;\n", b="%macro m;\n%mend m;\n")
        )
        self.assertEqual(report.overall_size, TShirtSize.MEDIUM)

    def test_size_counts_cover_every_size(self):
        report = _analyze("data a; set b; run;\n")
        self.assertEqual({s.value for s in TShirtSize}, set(report.size_counts))
        self.assertEqual(len(report.files), sum(report.size_counts.values()))

    def test_line_span_counts_overlapping_chunks_once(self):
        """A split region emits a parent plus children covering the same lines."""
        source = "".join(f"data o{i}; set i{i}; run;\n" for i in range(30))
        f = _analyze(source).files[0]
        self.assertLessEqual(f.line_count, source.count("\n") + 1)

    def test_markdown_renders_sizes_and_breakdown(self):
        report = _analyze(
            "".join(f"data o{i}; set i{i}; run;\n" for i in range(60))
        )
        md = report.to_markdown()
        self.assertIn("## File sizes", md)
        self.assertIn("Overall size", md)
        self.assertIn("Total story points", md)
        if report.files_needing_breakdown:
            self.assertIn("Files needing breakdown", md)

    def test_extra_large_file_suggests_batch_cut_points(self):
        result = SasSemanticChunker().chunk_text(
            _extra_large_source(), source_id="big.sas"
        )
        report = ComplexityAnalyzer().analyze_batch_result(
            SasChunkBatcher().batch(result)
        )
        f = report.files[0]
        self.assertEqual(f.size, TShirtSize.EXTRA_LARGE)
        self.assertTrue(f.needs_breakdown)
        self.assertTrue(f.suggested_split, "XL file should offer cut points")

    def test_non_extra_large_files_suggest_no_split(self):
        f = _analyze("data a; set b; run;\n").files[0]
        self.assertFalse(f.needs_breakdown)
        self.assertEqual(f.suggested_split, [])


class TestCatalogueIntegrity(unittest.TestCase):
    """Every shipped profile must parse and be internally consistent."""

    def test_bundled_profiles_are_discoverable(self):
        profiles = rules.available_profiles()
        self.assertIn("sparksql", profiles)
        self.assertIn("pyspark", profiles)

    def test_every_bundled_profile_loads(self):
        for name in rules.available_profiles():
            ruleset = rules.load_ruleset(name)
            self.assertEqual(ruleset.target, name)
            self.assertTrue(ruleset.display_name)
            self.assertGreater(ruleset.construct_count, 0)

    def test_every_spec_has_a_category(self):
        for name in rules.available_profiles():
            ruleset = rules.load_ruleset(name)
            for kind, catalogue in ruleset.constructs.items():
                for key, spec in catalogue.items():
                    self.assertTrue(
                        spec.category, f"{name}:{kind}[{key}] has no category"
                    )
            for attr, signal_name, spec in ruleset.flags:
                self.assertTrue(attr)
                self.assertTrue(spec.category, f"{name}:flag[{signal_name}]")

    def test_every_profile_covers_all_three_tier_weights(self):
        for name in rules.available_profiles():
            ruleset = rules.load_ruleset(name)
            for tier in ComplexityTier:
                self.assertIsInstance(ruleset.weight_for(tier), float)

    def test_brief_constructs_land_in_their_stated_tiers(self):
        """The tier assignments the project brief names, asserted directly.

        Tiers describe the SAS side, so they must hold for *every* target —
        only parity may move between profiles.
        """
        for name in rules.available_profiles():
            rs = rules.load_ruleset(name)
            self.assertEqual(rs.spec("proc", "sql").tier, ComplexityTier.LOW, name)
            self.assertEqual(
                rs.spec("global_statement", "let").tier, ComplexityTier.LOW, name
            )
            self.assertEqual(
                rs.spec("component_object", "hash").tier, ComplexityTier.MEDIUM, name
            )
            for key in ("merge", "filename_sftp", "filename_email"):
                self.assertEqual(
                    rs.spec("detector", key).tier,
                    ComplexityTier.MEDIUM,
                    f"{name}:{key}",
                )
            for key in ("array", "do_loop", "do_while", "do_until", "merge_no_by"):
                self.assertEqual(
                    rs.spec("detector", key).tier, ComplexityTier.HIGH, f"{name}:{key}"
                )
            self.assertEqual(
                rs.constructs["kind"]["MACRO_DEFINITION"].tier,
                ComplexityTier.HIGH,
                name,
            )

    def test_every_cross_file_construct_has_a_catalogue_entry(self):
        """Mirrors the detector-coverage rule: a resolver that can emit a name
        the catalogue does not list would silently drop the signal."""
        for name in rules.available_profiles():
            catalogue = set(rules.load_ruleset(name).constructs.get("cross_file", {}))
            missing = sorted(CROSS_FILE_CONSTRUCTS - catalogue)
            self.assertEqual(missing, [], f"{name} is missing {missing}")

    def test_no_catalogue_entry_without_a_resolver_to_emit_it(self):
        for name in rules.available_profiles():
            catalogue = set(rules.load_ruleset(name).constructs.get("cross_file", {}))
            orphans = sorted(catalogue - CROSS_FILE_CONSTRUCTS)
            self.assertEqual(orphans, [], f"{name} lists unreachable {orphans}")

    def test_cross_file_tiers_are_target_independent(self):
        """Tiers describe the SAS side, so only parity may move (see below)."""
        base = rules.load_ruleset("sparksql").constructs["cross_file"]
        for name in rules.available_profiles():
            other = rules.load_ruleset(name).constructs["cross_file"]
            for key, spec in base.items():
                self.assertEqual(other[key].tier, spec.tier, f"{name}:{key}")

    def test_unresolved_macro_outranks_external_everywhere(self):
        """Proving absence is worse than merely not having looked."""
        for name in rules.available_profiles():
            cf = rules.load_ruleset(name).constructs["cross_file"]
            self.assertEqual(cf["macro_unresolved"].tier, ComplexityTier.HIGH, name)
            self.assertEqual(
                cf["macro_unresolved"].parity, TranslationParity.MANUAL, name
            )
            self.assertLess(
                rules.load_ruleset(name).sizes.parity_weight(cf["macro_external"].parity),
                rules.load_ruleset(name).sizes.parity_weight(
                    cf["macro_unresolved"].parity
                ),
                name,
            )

    def test_every_profile_has_a_usable_size_model(self):
        for name in rules.available_profiles():
            sizes = rules.load_ruleset(name).sizes
            self.assertGreater(sizes.anchor_raw, 0, name)
            self.assertTrue(sizes.anchor_describes, f"{name} anchor is undocumented")
            bounds = [
                sizes.bands[s]
                for s in (TShirtSize.SMALL, TShirtSize.MEDIUM, TShirtSize.LARGE)
            ]
            self.assertEqual(bounds, sorted(bounds), name)

    def test_hashing_functions_are_supported_in_spark_sql(self):
        """Spark SQL ships md5/sha1/sha2/crc32/xxhash64, so the SAS hashing
        functions are a mechanical rewrite, not a semantic mismatch. The hash
        *object* is a lookup table and stays PARTIAL."""
        rs = rules.load_ruleset("sparksql")
        self.assertEqual(rs.spec("function", "md5").parity, TranslationParity.SUPPORTED)
        self.assertEqual(
            rs.spec("function", "sha256").parity, TranslationParity.SUPPORTED
        )
        self.assertEqual(
            rs.spec("component_object", "hash").parity, TranslationParity.PARTIAL
        )


class TestRetargeting(unittest.TestCase):
    """The same analysis, remapped to another output language."""

    MACRO = "%macro build(ds=);\n  data o;\n    set &ds;\n  run;\n%mend build;\n"
    DO_STEP = (
        "data work.out;\n  set work.in;\n  do i = 1 to 10;\n    t + i;\n  end;\nrun;\n"
    )

    def test_default_target_is_spark_sql(self):
        self.assertEqual(ComplexityAnalyzer().target, rules.DEFAULT_TARGET)
        self.assertEqual(ComplexityAnalyzer().target, "sparksql")

    def test_macro_definition_is_manual_for_sql_but_hard_for_pyspark(self):
        """Pure SQL has no procedural host language; PySpark does, so a %MACRO
        maps onto a parameterised Python function."""
        sql = _only(self.MACRO, target="sparksql")
        py = _only(self.MACRO, target="pyspark")
        self.assertEqual(sql.translation_difficulty, TranslationParity.MANUAL)
        self.assertEqual(py.translation_difficulty, TranslationParity.HARD)
        # The SAS-side tier is a property of the source, not the target.
        self.assertEqual(sql.tier, py.tier)

    def test_do_loop_parity_moves_but_tier_does_not(self):
        sql = _only(self.DO_STEP, target="sparksql")
        py = _only(self.DO_STEP, target="pyspark")
        self.assertEqual(sql.tier, ComplexityTier.HIGH)
        self.assertEqual(py.tier, ComplexityTier.HIGH)
        self.assertEqual(sql.translation_difficulty, TranslationParity.HARD)
        self.assertEqual(py.translation_difficulty, TranslationParity.PARTIAL)

    def test_derived_profile_inherits_everything_it_does_not_restate(self):
        sql = rules.load_ruleset("sparksql")
        py = rules.load_ruleset("pyspark")
        # pyspark.json never mentions PROC SORT; it inherits the rating.
        self.assertEqual(
            py.spec("proc", "sort").parity, sql.spec("proc", "sort").parity
        )
        self.assertEqual(py.spec("detector", "array").tier, ComplexityTier.HIGH)
        # It restates array's note, so that one differs.
        self.assertNotEqual(
            py.spec("detector", "array").note, sql.spec("detector", "array").note
        )
        # Inheritance must not drop constructs.
        self.assertGreaterEqual(py.construct_count, sql.construct_count)

    def test_results_and_report_record_their_target(self):
        report = _analyze(self.MACRO, target="pyspark")
        self.assertEqual(report.target, "pyspark")
        self.assertEqual(report.target_display, "PySpark")
        self.assertTrue(all(c.target == "pyspark" for c in report.chunks))
        self.assertIn("PySpark", report.to_markdown())

    def test_explicit_ruleset_wins_over_target(self):
        sql = rules.load_ruleset("sparksql")
        analyzer = ComplexityAnalyzer(target="pyspark", ruleset=sql)
        self.assertEqual(analyzer.target, "sparksql")


class TestRuleSetLoading(unittest.TestCase):
    """JSON profile loading, inheritance, and validation."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _write(self, doc) -> str:
        path = pathlib.Path(self.tmp) / "custom.json"
        path.write_text(json.dumps(doc), encoding="utf-8")
        return str(path)

    def _minimal(self, **over):
        doc = {
            "target": "custom",
            "display_name": "Custom Target",
            "weights": {"LOW": 2.0, "MEDIUM": 4.0, "HIGH": 8.0},
            "constructs": {
                "detector": {
                    "array": {
                        "category": "array",
                        "tier": "MEDIUM",
                        "parity": "SUPPORTED",
                        "note": "arrays are easy here",
                    }
                }
            },
        }
        doc.update(over)
        return doc

    def test_custom_profile_file_overrides_the_catalogue(self):
        path = self._write(self._minimal())
        scored = _only(
            "data o;\n  set a;\n  array s{3} s1-s3;\nrun;\n",
            rules_path=path,
        )
        # This profile rates ARRAY as MEDIUM/SUPPORTED, not HIGH/HARD.
        self.assertEqual(scored.tier, ComplexityTier.MEDIUM)
        self.assertEqual(scored.translation_difficulty, TranslationParity.SUPPORTED)
        self.assertEqual(scored.target, "custom")

    def test_profile_weights_are_used_for_scoring(self):
        path = self._write(self._minimal())
        scored = _only(
            "data o;\n  set a;\n  array s{3} s1-s3;\nrun;\n", rules_path=path
        )
        self.assertEqual(scored.score, 4.0)  # the profile's MEDIUM weight

    def test_construct_groups_expand(self):
        doc = self._minimal(
            construct_groups=[
                {
                    "kind": "function",
                    "names": ["md5", "sha256"],
                    "category": "hashing",
                    "tier": "LOW",
                    "parity": "DIRECT",
                }
            ]
        )
        ruleset = rules.load_ruleset(path=self._write(doc), use_cache=False)
        self.assertEqual(ruleset.spec("function", "md5").tier, ComplexityTier.LOW)
        self.assertEqual(ruleset.spec("function", "sha256").tier, ComplexityTier.LOW)

    def test_unknown_target_raises_with_available_names(self):
        with self.assertRaises(rules.RuleSetError) as ctx:
            rules.load_ruleset("klingon")
        self.assertIn("klingon", str(ctx.exception))
        self.assertIn("sparksql", str(ctx.exception))

    def test_missing_profile_file_raises(self):
        with self.assertRaises(rules.RuleSetError):
            rules.load_ruleset(path=str(pathlib.Path(self.tmp) / "nope.json"))

    def test_malformed_json_raises(self):
        path = pathlib.Path(self.tmp) / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        with self.assertRaises(rules.RuleSetError):
            rules.load_ruleset(path=str(path), use_cache=False)

    def test_invalid_tier_names_the_offending_key(self):
        doc = self._minimal()
        doc["constructs"]["detector"]["array"]["tier"] = "EXTREME"
        with self.assertRaises(rules.RuleSetError) as ctx:
            rules.load_ruleset(path=self._write(doc), use_cache=False)
        message = str(ctx.exception)
        self.assertIn("EXTREME", message)
        self.assertIn("array", message)

    def test_invalid_parity_is_rejected(self):
        doc = self._minimal()
        doc["constructs"]["detector"]["array"]["parity"] = "TRIVIAL"
        with self.assertRaises(rules.RuleSetError):
            rules.load_ruleset(path=self._write(doc), use_cache=False)

    def test_unknown_construct_kind_is_rejected(self):
        doc = self._minimal()
        doc["constructs"]["procz"] = {}
        with self.assertRaises(rules.RuleSetError) as ctx:
            rules.load_ruleset(path=self._write(doc), use_cache=False)
        self.assertIn("procz", str(ctx.exception))

    def test_missing_required_key_is_rejected(self):
        doc = self._minimal()
        del doc["constructs"]["detector"]["array"]["parity"]
        with self.assertRaises(rules.RuleSetError) as ctx:
            rules.load_ruleset(path=self._write(doc), use_cache=False)
        self.assertIn("parity", str(ctx.exception))

    def test_extends_unknown_profile_is_rejected(self):
        doc = self._minimal(extends="does-not-exist")
        with self.assertRaises(rules.RuleSetError) as ctx:
            rules.load_ruleset(path=self._write(doc), use_cache=False)
        self.assertIn("does-not-exist", str(ctx.exception))

    def test_self_extends_is_rejected_rather_than_looping(self):
        doc = self._minimal(target="sparksql", extends="sparksql")
        with self.assertRaises(rules.RuleSetError) as ctx:
            rules.load_ruleset(path=self._write(doc), use_cache=False)
        self.assertIn("circular", str(ctx.exception))

    def _sizes(self, sizes: dict):
        return rules.load_ruleset(
            path=self._write(self._minimal(sizes=sizes)), use_cache=False
        ).sizes

    def test_bounds_are_read_from_the_profile(self):
        sizes = self._sizes(
            {"anchor": {"raw": 10.0}, "bounds": {"effort": {"min": 1.0, "max": 4.0}}}
        )
        self.assertEqual(sizes.window_for("effort"), (10.0, 40.0))

    def test_inverted_bounds_are_rejected(self):
        with self.assertRaises(rules.RuleSetError) as ctx:
            self._sizes({"bounds": {"effort": {"min": 2.0, "max": 1.0}}})
        self.assertIn("effort", str(ctx.exception))

    def test_unknown_bounded_dimension_is_rejected(self):
        with self.assertRaises(rules.RuleSetError) as ctx:
            self._sizes({"bounds": {"efrot": {"min": 0.0, "max": 1.0}}})
        self.assertIn("efrot", str(ctx.exception))

    def test_negative_dimension_weight_is_rejected(self):
        with self.assertRaises(rules.RuleSetError) as ctx:
            self._sizes({"dimension_weights": {"uncertainty": -1.0}})
        self.assertIn("uncertainty", str(ctx.exception))

    def test_all_zero_dimension_weights_are_rejected(self):
        with self.assertRaises(rules.RuleSetError):
            self._sizes(
                {
                    "dimension_weights": {
                        "effort": 0.0,
                        "complexity": 0.0,
                        "uncertainty": 0.0,
                    }
                }
            )

    def test_anchor_dimensions_must_sum_to_the_anchor(self):
        with self.assertRaises(rules.RuleSetError) as ctx:
            self._sizes(
                {
                    "anchor": {
                        "raw": 10.0,
                        "dimensions": {
                            "effort": 5.0,
                            "complexity": 2.0,
                            "uncertainty": 0.0,
                        },
                    }
                }
            )
        self.assertIn("anchor.raw", str(ctx.exception))

    def test_partial_anchor_dimensions_are_rejected(self):
        with self.assertRaises(rules.RuleSetError) as ctx:
            self._sizes({"anchor": {"raw": 10.0, "dimensions": {"effort": 10.0}}})
        self.assertIn("complexity", str(ctx.exception))


class TestComplexityCLI(unittest.TestCase):
    """`python -m complexity <dir>` writes the report as Markdown."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        (self.tmp / "load.sas").write_text(
            "%macro load;\n  data work.a;\n    set raw.a;\n  run;\n%mend load;\n"
            "%load;\n",
            encoding="utf-8",
        )
        (self.tmp / "report.sas").write_text(
            "proc sql;\n  create table work.b as select * from work.a;\nquit;\n",
            encoding="utf-8",
        )

    def _run(self, *args):
        from complexity.__main__ import main

        return main([str(self.tmp), *args])

    def test_writes_a_markdown_report(self):
        out = self.tmp / "report.md"
        self.assertEqual(self._run("--out", str(out)), 0)
        text = out.read_text(encoding="utf-8")
        self.assertIn("# SAS chunk complexity report", text)
        self.assertIn("## Tier breakdown", text)
        self.assertIn("## File sizes", text)
        # Both files were scored.
        self.assertIn("load.sas", text)
        self.assertIn("report.sas", text)

    def test_target_selects_the_profile(self):
        out = self.tmp / "pyspark.md"
        self.assertEqual(self._run("--target", "pyspark", "--out", str(out)), 0)
        self.assertIn("PySpark", out.read_text(encoding="utf-8"))

    def test_creates_a_missing_output_directory(self):
        out = self.tmp / "nested" / "deep" / "report.md"
        self.assertEqual(self._run("--out", str(out)), 0)
        self.assertTrue(out.is_file())

    def test_no_matching_files_is_an_error_exit(self):
        self.assertEqual(self._run("--pattern", "*.nope"), 1)

    def test_report_prints_when_no_out_is_given(self):
        import contextlib
        import io

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            self.assertEqual(self._run(), 0)
        self.assertIn("# SAS chunk complexity report", buffer.getvalue())


# ---------------------------------------------------------------------------
# Individual reports, source text, and the optional LLM evaluation
# ---------------------------------------------------------------------------

_LOAD_SAS = (
    "%macro load(lib=raw);\n"
    "  data work.a;\n"
    "    set &lib..a;\n"
    "    array x{3} v1-v3;\n"
    "    do i = 1 to 3;\n"
    "      x{i} = x{i} * 2;\n"
    "    end;\n"
    "  run;\n"
    "%mend load;\n"
    "%load;\n"
)

_REPORT_SAS = "proc sql;\n  create table work.b as select * from work.a;\nquit;\n"


def _evaluated(evaluation, source_id: str):
    """The FileEvaluationResult for *source_id*, failing loudly if absent."""
    result = evaluation.for_source(source_id)
    if result is None:
        raise AssertionError(
            f"no evaluation for {source_id!r}; have "
            f"{[f.source_id for f in evaluation.files]}"
        )
    return result


def _assessment(evaluation, source_id: str):
    """The parsed FileEvaluation for *source_id*, failing loudly if absent."""
    result = _evaluated(evaluation, source_id)
    if result.evaluation is None:
        raise AssertionError(
            f"{source_id!r} was not parsed: {result.error} / {result.prose[:80]!r}"
        )
    return result.evaluation


def _batched(**files: str):
    """Chunk, batch, and score *files*; return ``(report, texts)``.

    Batched rather than raw, because :class:`MultiFileBatcher` re-ids every
    chunk per file — a text lookup built from the unbatched corpus would miss
    every one, which is exactly the wiring this pair has to keep straight.
    """
    corpus = _corpus(**files)
    batch_result = MultiFileBatcher().batch(corpus)
    report = ComplexityAnalyzer().analyze_items(
        batch_result.all_ordered_items,
        source_ids=corpus.source_ids,
        diagnostics=corpus.all_diagnostics,
    )
    return report, chunk_texts(batch_result.all_ordered_items)


class TestChunkTexts(unittest.TestCase):
    """The (source_id, chunk_id) -> text lookup the renderers take."""

    def test_indexes_a_corpus_by_source_and_chunk_id(self):
        corpus = _corpus(load=_LOAD_SAS)
        texts = chunk_texts(corpus)
        self.assertTrue(texts)
        for (source_id, chunk_id), text in texts.items():
            self.assertEqual(source_id, "load.sas")
            self.assertTrue(chunk_id)
            self.assertIn("%macro", "".join(texts.values()))
            self.assertTrue(text.strip())

    def test_accepts_a_batch_result_and_a_chunk_result(self):
        corpus = _corpus(load=_LOAD_SAS)
        from_result = chunk_texts(corpus.file_results[0])
        from_batches = chunk_texts(MultiFileBatcher().batch(corpus))
        self.assertTrue(from_result)
        self.assertTrue(from_batches)
        # The batcher re-ids chunks, so the two disagree on the keys and agree
        # on the texts. That is the whole reason the key carries source_id.
        self.assertEqual(
            sorted(from_result.values()), sorted(from_batches.values())
        )

    def test_same_chunk_id_in_two_files_does_not_collide(self):
        report, texts = _batched(load=_LOAD_SAS, other=_LOAD_SAS)
        sources = {source_id for source_id, _ in texts}
        self.assertEqual(sources, {"load.sas", "other.sas"})
        # Two entries per file at least, none lost to a shared chunk id.
        self.assertEqual(len(texts), sum(f.chunk_count for f in report.files))


class TestFileReport(unittest.TestCase):
    """Per-source-file reports print the SAS behind every verdict."""

    def setUp(self):
        self.report, self.texts = _batched(load=_LOAD_SAS, report=_REPORT_SAS)
        self.file = _file(self.report, "load.sas")

    def _render(self, **kwargs):
        return render_file_report(
            self.file,
            texts=self.texts,
            target_display=self.report.target_display,
            **kwargs,
        )

    def test_prints_the_chunk_source_for_every_chunk_it_mentions(self):
        text = self._render()
        for chunk in self.file.chunks:
            self.assertIn(f"`{chunk.chunk_id}`", text)
        self.assertIn("```sas", text)
        self.assertIn("array x{3} v1-v3;", text)
        self.assertIn("%mend load;", text)

    def test_states_the_size_tier_and_dimensions(self):
        text = self._render()
        self.assertIn(f"# Complexity report — {self.file.source_id}", text)
        self.assertIn(self.file.size.label, text)
        self.assertIn("## Drivers", text)
        self.assertIn("## Chunks", text)
        self.assertIn("array", text)

    def test_no_source_text_keeps_the_verdicts_and_drops_the_sas(self):
        text = self._render(include_source=False)
        self.assertNotIn("```sas", text)
        self.assertIn("## Chunks", text)
        self.assertIn(self.file.chunks[0].chunk_id, text)

    def test_max_source_lines_truncates_and_says_so(self):
        text = self._render(max_source_lines=2)
        self.assertIn("%macro load(lib=raw);", text)
        self.assertNotIn("%mend load;", text)
        self.assertIn("further line(s) not shown", text)

    def test_a_missing_snippet_renders_a_placeholder(self):
        text = render_file_report(self.file, texts={})
        self.assertIn("Source text unavailable", text)
        # The verdict still renders in full.
        self.assertIn("## Drivers", text)

    def test_a_floored_size_says_which_kind_floored_it(self):
        self.assertEqual(self.file.floored_by, "MACRO_DEFINITION")
        self.assertIn("floored", self._render())


class TestOverallReport(unittest.TestCase):
    """The corpus report, plus its index of the individual ones."""

    def setUp(self):
        self.report, self.texts = _batched(load=_LOAD_SAS, report=_REPORT_SAS)

    def test_without_links_it_is_the_corpus_report_verbatim(self):
        self.assertEqual(
            render_overall_report(self.report, top=5),
            self.report.to_markdown(top=5),
        )

    def test_links_are_appended_as_an_index(self):
        text = render_overall_report(
            self.report, file_links={"load.sas": "files/load.md"}
        )
        self.assertIn("## Individual reports", text)
        self.assertIn("[files/load.md](files/load.md)", text)
        # Every scored file appears, linked or not.
        self.assertIn("report.sas", text)


class TestWriteReports(unittest.TestCase):
    """`write_reports` puts the overall report and the per-file ones on disk."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_writes_one_report_per_source_file_plus_the_overall_one(self):
        report, texts = _batched(load=_LOAD_SAS, report=_REPORT_SAS)
        written = write_reports(report, self.tmp, texts=texts)

        self.assertTrue(written.overall.is_file())
        self.assertEqual(written.overall.name, "complexity-report.md")
        self.assertEqual(set(written.files), {"load.sas", "report.sas"})
        for path in written.files.values():
            self.assertTrue(path.is_file())
            self.assertEqual(path.parent.name, "files")

        overall = written.overall.read_text(encoding="utf-8")
        self.assertIn("## Individual reports", overall)
        self.assertIn("files/load.md", overall)

        individual = written.files["load.sas"].read_text(encoding="utf-8")
        self.assertIn("array x{3} v1-v3;", individual)
        self.assertIn("../complexity-report.md", individual)

    def test_two_files_sharing_a_basename_get_distinct_reports(self):
        corpus = SasCorpus(
            file_results=[
                SasSemanticChunker().chunk_text(_LOAD_SAS, source_id=sid)
                for sid in ("a/job.sas", "b/job.sas")
            ]
        )
        batch_result = MultiFileBatcher().batch(corpus)
        report = ComplexityAnalyzer().analyze_items(
            batch_result.all_ordered_items, source_ids=corpus.source_ids
        )
        written = write_reports(
            report, self.tmp, texts=chunk_texts(batch_result.all_ordered_items)
        )
        stems = {path.stem for path in written.files.values()}
        self.assertEqual(len(stems), 2, f"reports collided: {stems}")


class TestEvaluationPrompt(unittest.TestCase):
    """The LLM prompt is built offline and carries verdict plus source."""

    def setUp(self):
        self.report, self.texts = _batched(load=_LOAD_SAS, report=_REPORT_SAS)
        self.file = _file(self.report, "load.sas")

    def test_carries_the_static_verdict_and_the_sas_source(self):
        prompt = build_evaluation_prompt(
            self.file, texts=self.texts, target_display=self.report.target_display
        )
        self.assertIn("Spark SQL", prompt)
        self.assertIn("load.sas", prompt)
        self.assertIn(self.file.size.label, prompt)
        self.assertIn("array x{3} v1-v3;", prompt)
        self.assertIn("```sas", prompt)
        # It asks for what the rules cannot answer, not for the rules again.
        self.assertIn("Argue with it", prompt)
        self.assertIn("Open questions", prompt)

    def test_include_source_false_drops_the_sas(self):
        prompt = build_evaluation_prompt(
            self.file, texts=self.texts, include_source=False
        )
        self.assertNotIn("```sas", prompt)
        self.assertIn(self.file.size.label, prompt)

    def test_one_prompt_per_file_and_sources_narrows_it(self):
        every = evaluation_prompts(self.report, texts=self.texts)
        self.assertEqual(set(every), {"load.sas", "report.sas"})
        one = evaluation_prompts(
            self.report, texts=self.texts, sources=["load.sas"]
        )
        self.assertEqual(set(one), {"load.sas"})


class _StructuredFake:
    """A client offering structured output, like ``llm_client.LLMClient``."""

    payload = {
        "summary": "Loads raw.a into work.a, doubling three measures.",
        "size_verdict": "larger",
        "size_rationale": "The ARRAY/DO pair is a wide-to-long restructure.",
        "findings": [{"severity": "P0", "note": "f1-chunk-0001: array aliases columns."}],
        "manual_steps": ["Decide the target schema."],
        "suggested_split": [],
        "open_questions": ["Where does raw.a come from?"],
    }

    def __init__(self):
        self.config = type("_C", (), {"model": "fake-structured"})()
        self.prompts = []

    def supports_structured_output(self, schema):
        return True

    def invoke_structured(self, schema, prompt):
        self.prompts.append(prompt)
        return {
            "raw": None,
            "parsed": schema.model_validate(self.payload),
            "parsing_error": None,
        }


class _ProseFake:
    """A plain chat model: no structured output, JSON in a fenced block."""

    def __init__(self, reply=None):
        self.model_name = "fake-prose"
        self.reply = reply or (
            "Sure:\n```json\n"
            + json.dumps(_StructuredFake.payload)
            + "\n```\nHope that helps."
        )
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return self.reply


class TestLLMEvaluation(unittest.TestCase):
    """Evaluation degrades rather than raising, whatever the client does."""

    def setUp(self):
        self.report, self.texts = _batched(load=_LOAD_SAS, report=_REPORT_SAS)

    def test_structured_client_yields_a_typed_evaluation_per_file(self):
        llm = _StructuredFake()
        evaluation = evaluate_report(llm, self.report, texts=self.texts)

        self.assertEqual(len(evaluation.files), len(self.report.files))
        self.assertEqual(evaluation.failures, [])
        self.assertEqual(evaluation.model, "fake-structured")
        self.assertEqual(_assessment(evaluation, "load.sas").size_verdict, "larger")
        self.assertEqual(
            _evaluated(evaluation, "load.sas").static_size,
            _file(self.report, "load.sas").size.label,
        )
        # One call per file, each carrying that file's source.
        self.assertEqual(len(llm.prompts), len(self.report.files))
        self.assertIn("array x{3} v1-v3;", llm.prompts[0] + llm.prompts[1])

        markdown = evaluation.to_markdown()
        self.assertIn("# LLM complexity evaluation", markdown)
        self.assertIn("P0", markdown)
        self.assertIn("Where does raw.a come from?", markdown)

    def test_json_in_a_prose_reply_is_recovered(self):
        evaluation = evaluate_report(_ProseFake(), self.report, texts=self.texts)
        self.assertEqual(evaluation.failures, [])
        self.assertEqual(_assessment(evaluation, "load.sas").size_verdict, "larger")

    def test_an_unusable_reply_is_kept_as_prose_rather_than_raising(self):
        evaluation = evaluate_report(
            _ProseFake(reply="I could not do that."), self.report, texts=self.texts
        )
        self.assertEqual(len(evaluation.failures), len(self.report.files))
        result = _evaluated(evaluation, "load.sas")
        self.assertFalse(result.ok)
        self.assertIn("I could not do that.", result.prose)
        self.assertTrue(result.error)
        # The document still renders, carrying the unparsed reply.
        self.assertIn("I could not do that.", evaluation.to_markdown())

    def test_a_raising_client_fails_only_that_file(self):
        class _Boom:
            calls = 0

            def invoke(self, prompt):
                _Boom.calls += 1
                if _Boom.calls == 1:
                    raise RuntimeError("gateway down")
                return "```json\n" + json.dumps(_StructuredFake.payload) + "\n```"

        evaluation = evaluate_report(_Boom(), self.report, texts=self.texts)
        self.assertEqual(len(evaluation.failures), 1)
        self.assertIn("gateway down", evaluation.failures[0].error)
        self.assertEqual(len(evaluation.files), len(self.report.files))

    def test_limit_evaluates_only_the_largest_files(self):
        llm = _StructuredFake()
        evaluation = evaluate_report(llm, self.report, texts=self.texts, limit=1)
        self.assertEqual(len(evaluation.files), 1)
        largest = max(self.report.files, key=lambda f: f.points)
        self.assertEqual(evaluation.files[0].source_id, largest.source_id)


class TestComplexityReportCLI(unittest.TestCase):
    """`--out-dir`, `--no-source-text`, and `--prompt-only`."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        (self.tmp / "load.sas").write_text(_LOAD_SAS, encoding="utf-8")
        (self.tmp / "report.sas").write_text(_REPORT_SAS, encoding="utf-8")
        self.out = self.tmp / "out"

    def _run(self, *args):
        from complexity.__main__ import main

        return main([str(self.tmp), *args])

    def test_out_dir_writes_the_overall_report_and_one_per_file(self):
        self.assertEqual(self._run("--out-dir", str(self.out)), 0)
        overall = self.out / "complexity-report.md"
        self.assertTrue(overall.is_file())
        self.assertIn("## Individual reports", overall.read_text(encoding="utf-8"))

        individual = sorted((self.out / "files").glob("*.md"))
        self.assertEqual([p.stem for p in individual], ["load", "report"])
        load = (self.out / "files" / "load.md").read_text(encoding="utf-8")
        self.assertIn("array x{3} v1-v3;", load)
        self.assertIn("```sas", load)

    def test_no_source_text_omits_the_sas(self):
        self.assertEqual(
            self._run("--out-dir", str(self.out), "--no-source-text"), 0
        )
        load = (self.out / "files" / "load.md").read_text(encoding="utf-8")
        self.assertNotIn("```sas", load)
        self.assertIn("## Drivers", load)

    def test_prompt_only_writes_prompts_and_calls_nothing(self):
        self.assertEqual(
            self._run("--out-dir", str(self.out), "--llm-eval", "--prompt-only"), 0
        )
        prompts = sorted((self.out / "prompts").glob("*.md"))
        self.assertEqual([p.stem for p in prompts], ["load", "report"])
        text = (self.out / "prompts" / "load.md").read_text(encoding="utf-8")
        self.assertIn("Static verdict", text)
        self.assertIn("array x{3} v1-v3;", text)
        # No evaluation was produced, because nothing was called.
        self.assertFalse((self.out / "llm-evaluation.md").exists())

    def test_eval_top_limits_the_prompts(self):
        self.assertEqual(
            self._run(
                "--out-dir",
                str(self.out),
                "--prompt-only",
                "--eval-top",
                "1",
            ),
            0,
        )
        self.assertEqual(len(list((self.out / "prompts").glob("*.md"))), 1)


class TestCommentBlocks(unittest.TestCase):
    """COMMENT_BLOCK chunks are excluded from the analysis, and counted."""

    BARE = "data work.a;\n  set work.b;\nrun;\n"
    COMMENTED = (
        "/* A header comment.\n"
        "   Several lines of documentation that are not migration work.\n"
        "   More of the same, at length. */\n"
        "data work.a;\n"
        "  set work.b;\n"
        "run;\n"
    )

    def test_the_chunker_really_does_emit_one(self):
        """Guards the premise: without a COMMENT_BLOCK the rest proves nothing."""
        result = SasSemanticChunker().chunk_text(self.COMMENTED, source_id="t.sas")
        kinds = [c.kind.value for c in result.chunks]
        self.assertIn("COMMENT_BLOCK", kinds)

    def test_no_comment_chunk_reaches_the_verdicts(self):
        report = _analyze(self.COMMENTED)
        self.assertNotIn(
            "COMMENT_BLOCK", {c.kind for c in _file(report, "t.sas").chunks}
        )
        self.assertNotIn("COMMENT_BLOCK", {c.kind for c in report.chunks})

    def test_counts_exclude_it_and_report_it(self):
        file = _file(_analyze(self.COMMENTED), "t.sas")
        self.assertEqual(file.chunk_count, 1)
        self.assertEqual(file.comment_chunk_count, 1)
        # The three comment lines are not in the span either.
        self.assertEqual(file.line_count, 3)

    def test_documentation_does_not_make_a_file_bigger(self):
        """The point of the exclusion: same code, same size, comments or not."""
        bare = _file(_analyze(self.BARE), "t.sas")
        commented = _file(_analyze(self.COMMENTED), "t.sas")
        self.assertEqual(bare.effort_raw, commented.effort_raw)
        self.assertEqual(bare.raw_total, commented.raw_total)
        self.assertEqual(bare.points, commented.points)

    def test_a_comment_only_file_still_gets_a_rollup(self):
        """It must not vanish from the corpus just because nothing scored."""
        report = _analyze("/* nothing but a note. */\n")
        file = _file(report, "t.sas")
        self.assertEqual(file.chunk_count, 0)
        self.assertEqual(file.comment_chunk_count, 1)
        self.assertEqual(file.chunks, [])
        self.assertEqual(file.size, TShirtSize.SMALL)

    def test_batch_members_exclude_it_too(self):
        """Filtering only the file rollup would leave batch scores inflated.

        ``include_comment_chunks`` is what puts a COMMENT_BLOCK *inside* a
        batch rather than beside it as a singleton — the path that reaches
        ``analyze_batch`` rather than ``analyze_items``.
        """
        corpus = _corpus(
            a="/* Header comment\n   over several lines. */\n"
            "data work.a;\n  set work.b;\nrun;\n"
            "data work.c;\n  set work.a;\nrun;\n"
        )
        batch_result = MultiFileBatcher(include_comment_chunks=True).batch(corpus)
        batched_kinds = {
            c.kind.value
            for item in batch_result.all_ordered_items
            for c in getattr(item, "chunks", [])
        }
        self.assertIn("COMMENT_BLOCK", batched_kinds)  # the premise

        report = ComplexityAnalyzer().analyze_items(
            batch_result.all_ordered_items, source_ids=corpus.source_ids
        )
        self.assertTrue(report.batches)
        for batch in report.batches:
            self.assertNotIn("COMMENT_BLOCK", {m.kind for m in batch.members})

    def test_the_report_says_how_many_were_excluded(self):
        text = render_file_report(_file(_analyze(self.COMMENTED), "t.sas"))
        self.assertIn("1 comment block(s) excluded", text)

    def test_a_file_without_comments_says_nothing_about_them(self):
        text = render_file_report(_file(_analyze(self.BARE), "t.sas"))
        self.assertNotIn("comment block(s) excluded", text)


class TestDatasets(unittest.TestCase):
    """The file's data interface: inputs, outputs, and internal intermediates."""

    # Reads edw.raw, writes work.stg, reads it back, writes mart.out.
    PIPELINE = (
        "data work.stg;\n"
        "  set edw.raw;\n"
        "run;\n"
        "data mart.out;\n"
        "  set work.stg;\n"
        "run;\n"
    )

    def setUp(self):
        self.file = _file(_analyze(self.PIPELINE), "t.sas")

    def test_a_dataset_written_then_read_here_is_an_intermediate(self):
        self.assertIn("work.stg", self.file.intermediate_datasets)
        # And so is not something anyone outside has to provide.
        self.assertNotIn("work.stg", self.file.input_datasets)

    def test_inputs_are_what_the_file_does_not_write_itself(self):
        self.assertEqual(self.file.input_datasets, ["edw.raw"])

    def test_outputs_are_everything_written(self):
        self.assertEqual(
            sorted(self.file.output_datasets), ["mart.out", "work.stg"]
        )

    def test_chunks_carry_their_own_reads_and_writes(self):
        first = min(self.file.chunks, key=lambda c: c.start_line)
        self.assertEqual(first.input_datasets, ["edw.raw"])
        self.assertEqual(first.output_datasets, ["work.stg"])

    def test_case_differences_are_one_dataset(self):
        file = _file(
            _analyze("data work.a;\n  set EDW.Raw;\nrun;\ndata b; set edw.raw; run;\n"),
            "t.sas",
        )
        self.assertEqual(len(file.input_datasets), 1)

    def test_the_report_renders_the_three_way_split(self):
        text = render_file_report(self.file)
        self.assertIn("## Datasets", text)
        self.assertIn("Inputs (read here, written elsewhere): edw.raw", text)
        self.assertIn("Intermediates (written and read here): work.stg", text)

    def test_a_file_touching_no_dataset_gets_no_section(self):
        text = render_file_report(_file(_analyze("%let x = 1;\n"), "t.sas"))
        self.assertNotIn("## Datasets", text)

    def test_the_rollup_agrees_with_cross_file_coupling(self):
        """An imported dataset must not also be claimed as locally produced."""
        report = ComplexityAnalyzer().analyze_corpus(
            _corpus(
                load="data raw.customers;\n  set edw.extract;\nrun;\n",
                use="data mart.agg;\n  set raw.customers;\nrun;\n",
            )
        )
        use = _file(report, "use.sas")
        self.assertIn("raw.customers", use.input_datasets)
        assert use.cross_file is not None
        self.assertEqual(use.cross_file.depends_on, ["load.sas"])


def _graph_corpus() -> SasCorpus:
    """load -> transform -> {report_a, report_b}: a fan-out with one root.

    The fan-out is the point: two files reading one dataset is the case a
    single-peer edge record silently loses.
    """
    return _corpus(
        load="data raw.customers;\n  set edw.extract;\nrun;\n",
        transform="data mart.agg;\n  set raw.customers;\nrun;\n",
        report_a="proc print data=mart.agg;\nrun;\n",
        report_b="proc means data=mart.agg;\nrun;\n",
    )


class TestDependencyGraph(unittest.TestCase):
    """Corpus-level dependency structure, built from the resolved references."""

    def setUp(self):
        self.report = ComplexityAnalyzer().analyze_corpus(_graph_corpus())
        self.graph = self.report.graph
        assert self.graph is not None

    def _edge(self, upstream: str, downstream: str):
        for edge in self.graph.edges:
            if (edge.upstream, edge.downstream) == (upstream, downstream):
                return edge
        raise AssertionError(
            f"no {upstream} -> {downstream} edge; have "
            f"{[(e.upstream, e.downstream) for e in self.graph.edges]}"
        )

    def test_every_dependant_gets_an_edge(self):
        """The lossy case: both readers of mart.agg, not just the first."""
        self._edge("transform.sas", "report_a.sas")
        self._edge("transform.sas", "report_b.sas")

    def test_every_dependant_is_recorded_on_the_producer_profile(self):
        transform = _file(self.report, "transform.sas").cross_file
        assert transform is not None
        self.assertEqual(
            transform.depended_on_by, ["report_a.sas", "report_b.sas"]
        )

    def test_an_edge_names_what_caused_it(self):
        self.assertEqual(self._edge("load.sas", "transform.sas").datasets,
                         ["raw.customers"])
        self.assertIn("dataset raw.customers",
                      self._edge("load.sas", "transform.sas").label)

    def test_import_and_export_fold_into_one_edge(self):
        """Both files report the same dependency; it is one edge, not two."""
        pairs = [(e.upstream, e.downstream) for e in self.graph.edges]
        self.assertEqual(len(pairs), len(set(pairs)))

    def test_nodes_include_every_analysed_file(self):
        self.assertEqual(
            self.graph.nodes,
            ["load.sas", "report_a.sas", "report_b.sas", "transform.sas"],
        )

    def test_layers_are_migration_waves(self):
        self.assertEqual(
            self.graph.layers,
            [["load.sas"], ["transform.sas"], ["report_a.sas", "report_b.sas"]],
        )

    def test_roots_and_leaves(self):
        self.assertEqual(self.graph.roots, ["load.sas"])
        self.assertEqual(self.graph.leaves, ["report_a.sas", "report_b.sas"])

    def test_a_clean_corpus_is_acyclic(self):
        self.assertTrue(self.graph.is_acyclic)
        self.assertEqual(self.graph.cycles, [])

    def test_macro_dependencies_are_edges_too(self):
        report = ComplexityAnalyzer().analyze_corpus(
            _corpus(
                lib="%macro build(ds=);\n  data out; set &ds; run;\n%mend build;\n",
                job="%build(ds=work.raw);\n",
            )
        )
        graph = report.graph
        assert graph is not None
        edge = next(e for e in graph.edges if e.upstream == "lib.sas")
        self.assertEqual(edge.downstream, "job.sas")
        self.assertEqual(edge.macros, ["build"])
        self.assertIn("macro build", edge.label)

    def test_a_cycle_is_reported_and_not_silently_ordered(self):
        """Two jobs each reading what the other writes cannot be a DAG."""
        report = ComplexityAnalyzer().analyze_corpus(
            _corpus(
                a="data lib.one;\n  set lib.two;\nrun;\n",
                b="data lib.two;\n  set lib.one;\nrun;\n",
            )
        )
        graph = report.graph
        assert graph is not None
        self.assertFalse(graph.is_acyclic)
        self.assertEqual(graph.cycles, [["a.sas", "b.sas"]])
        # Neither is given a wave it cannot be given; both land in the
        # trailing unordered layer.
        self.assertEqual(graph.layers, [["a.sas", "b.sas"]])

    def test_no_cross_file_analysis_means_no_graph(self):
        report = ComplexityAnalyzer(use_cross_file=False).analyze_corpus(
            _graph_corpus()
        )
        self.assertIsNone(report.graph)

    def test_a_file_depends_on_itself_is_not_an_edge(self):
        report = _analyze("data work.a;\n  set work.b;\nrun;\ndata c; set work.a; run;\n")
        graph = report.graph
        assert graph is not None
        self.assertEqual(graph.edges, [])


class TestDependencyGraphRendering(unittest.TestCase):
    """The Markdown section, and the optional image beside it."""

    def setUp(self):
        self.report = ComplexityAnalyzer().analyze_corpus(_graph_corpus())
        self.graph = self.report.graph
        assert self.graph is not None

    def test_the_overall_report_carries_the_edge_table(self):
        text = self.report.to_markdown()
        self.assertIn("## Dependency graph", text)
        self.assertIn("| Upstream (migrate first) | Downstream | Via |", text)
        self.assertIn("| load.sas | transform.sas | dataset raw.customers |", text)

    def test_the_overall_report_carries_the_migration_order(self):
        text = self.report.to_markdown()
        self.assertIn("### Migration order", text)
        self.assertIn("**Wave 1**: load.sas", text)
        self.assertIn("**Wave 3**: report_a.sas, report_b.sas", text)

    def test_an_image_is_linked_only_when_one_was_drawn(self):
        self.assertNotIn("![Dependency graph]", self.report.to_markdown())
        self.assertIn(
            "![Dependency graph](dependency-graph.png)",
            self.report.to_markdown(graph_image="dependency-graph.png"),
        )

    def test_an_uncoupled_corpus_gets_no_section(self):
        report = ComplexityAnalyzer().analyze_corpus(
            _corpus(a="%let x = 1;\n", b="%let y = 2;\n")
        )
        self.assertNotIn("## Dependency graph", report.to_markdown())

    def test_a_cycle_is_called_out_in_the_prose(self):
        report = ComplexityAnalyzer().analyze_corpus(
            _corpus(
                a="data lib.one;\n  set lib.two;\nrun;\n",
                b="data lib.two;\n  set lib.one;\nrun;\n",
            )
        )
        text = report.to_markdown()
        self.assertIn("not acyclic", text)
        self.assertIn("### Cycles (1)", text)
        self.assertIn("**Unordered** (in a cycle): a.sas, b.sas", text)

    def test_without_links_the_overall_report_is_still_verbatim(self):
        """The graph belongs to `to_markdown`, so this equality must hold."""
        self.assertEqual(
            render_overall_report(self.report, top=5),
            self.report.to_markdown(top=5),
        )


class TestDependencyGraphImage(unittest.TestCase):
    """`render_png` — optional, and never fatal when it cannot run."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        report = ComplexityAnalyzer().analyze_corpus(_graph_corpus())
        assert report.graph is not None
        self.graph = report.graph

    @unittest.skipUnless(
        importlib.util.find_spec("matplotlib"), "matplotlib is not installed"
    )
    def test_it_writes_a_png(self):
        dest = render_png(self.graph, self.tmp / "graph.png")
        self.assertIsNotNone(dest)
        assert dest is not None
        self.assertTrue(dest.is_file())
        self.assertGreater(dest.stat().st_size, 0)
        # A real PNG, not an empty file with the right name.
        self.assertEqual(dest.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_a_graph_with_no_edges_draws_nothing(self):
        empty = DependencyGraph(nodes=["a.sas"], edges=[])
        self.assertIsNone(render_png(empty, self.tmp / "none.png"))
        self.assertFalse((self.tmp / "none.png").exists())

    def test_too_many_files_falls_back_to_the_table(self):
        self.assertIsNone(
            render_png(self.graph, self.tmp / "big.png", max_nodes=2)
        )

    def test_a_missing_matplotlib_is_a_log_line_not_a_failure(self):
        with unittest.mock.patch.dict(sys.modules, {"matplotlib": None}):
            self.assertIsNone(render_png(self.graph, self.tmp / "absent.png"))
        self.assertFalse((self.tmp / "absent.png").exists())


class TestWriteReportsGraph(unittest.TestCase):
    """The image lands beside the overall report and is linked from it."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        corpus = _graph_corpus()
        batch_result = MultiFileBatcher().batch(corpus)
        self.report = ComplexityAnalyzer().analyze_items(
            batch_result.all_ordered_items,
            source_ids=corpus.source_ids,
            diagnostics=corpus.all_diagnostics,
        )
        self.texts = chunk_texts(batch_result.all_ordered_items)

    @unittest.skipUnless(
        importlib.util.find_spec("matplotlib"), "matplotlib is not installed"
    )
    def test_the_image_is_written_and_linked(self):
        written = write_reports(self.report, self.tmp, texts=self.texts)
        self.assertIsNotNone(written.graph)
        assert written.graph is not None
        self.assertEqual(written.graph.name, "dependency-graph.png")
        self.assertIn(written.graph, written.paths)
        overall = written.overall.read_text(encoding="utf-8")
        self.assertIn("![Dependency graph](dependency-graph.png)", overall)

    def test_it_can_be_switched_off(self):
        written = write_reports(
            self.report, self.tmp, texts=self.texts, graph_image=False
        )
        self.assertIsNone(written.graph)
        self.assertFalse((self.tmp / "dependency-graph.png").exists())
        # The edges are still reported, which is the point of the table.
        self.assertIn(
            "## Dependency graph", written.overall.read_text(encoding="utf-8")
        )


# ---------------------------------------------------------------------------
# Naming: reports print file names, models keep paths
# ---------------------------------------------------------------------------


def _pathed_corpus() -> SasCorpus:
    """Two `load.sas` scripts in different directories, plus an unambiguous one.

    The collision is the point: absolute source ids are what the CLI actually
    hands the chunker, and two scripts sharing a basename are common in a real
    SAS estate.
    """
    chunker = SasSemanticChunker()
    sources = {
        "/corp/sas/etl/load.sas": "data raw.customers;\n  set edw.extract;\nrun;\n",
        "/corp/sas/adhoc/load.sas": "data work.scratch;\n  set raw.customers;\nrun;\n",
        "/corp/sas/report.sas": "proc print data=work.scratch;\nrun;\n",
    }
    return SasCorpus(
        file_results=[
            chunker.chunk_text(src, source_id=sid) for sid, src in sources.items()
        ]
    )


class TestDisplayNames(unittest.TestCase):
    """`display_names` — the shortest tail of each path that stays unique."""

    def test_a_unique_basename_is_the_whole_name(self):
        self.assertEqual(
            display_names(["/a/b/load.sas", "/a/b/report.sas"]),
            {"/a/b/load.sas": "load.sas", "/a/b/report.sas": "report.sas"},
        )

    def test_a_collision_widens_only_the_files_that_collide(self):
        names = display_names(
            ["/s/etl/load.sas", "/s/adhoc/load.sas", "/s/deep/nest/report.sas"]
        )
        self.assertEqual(names["/s/etl/load.sas"], "etl/load.sas")
        self.assertEqual(names["/s/adhoc/load.sas"], "adhoc/load.sas")
        # The uncollided file keeps its short name rather than being widened
        # along with them.
        self.assertEqual(names["/s/deep/nest/report.sas"], "report.sas")

    def test_windows_separators_split_too(self):
        names = display_names(["D:\\corp\\etl\\load.sas", "D:\\corp\\qa\\load.sas"])
        self.assertEqual(
            sorted(names.values()), ["etl/load.sas", "qa/load.sas"]
        )

    def test_it_widens_as_far_as_it_must(self):
        names = display_names(["/a/x/one/f.sas", "/b/y/one/f.sas"])
        self.assertEqual(names["/a/x/one/f.sas"], "x/one/f.sas")
        self.assertEqual(names["/b/y/one/f.sas"], "y/one/f.sas")

    def test_duplicates_collapse_rather_than_colliding(self):
        self.assertEqual(display_names(["a.sas", "a.sas"]), {"a.sas": "a.sas"})

    def test_display_name_is_the_context_free_basename(self):
        self.assertEqual(display_name("/a/b/load.sas"), "load.sas")
        self.assertEqual(display_name("load.sas"), "load.sas")
        self.assertEqual(display_name(""), "")

    def test_resolve_name_falls_back_to_the_basename(self):
        self.assertEqual(resolve_name("/a/b/c.sas", None), "c.sas")
        self.assertEqual(resolve_name("/a/b/c.sas", {}), "c.sas")
        self.assertEqual(resolve_name("/a/b/c.sas", {"/a/b/c.sas": "b/c.sas"}),
                         "b/c.sas")


class TestReportsPrintNames(unittest.TestCase):
    """Every rendered report names files; only the model keeps the path."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        corpus = _pathed_corpus()
        batch_result = MultiFileBatcher().batch(corpus)
        self.report = ComplexityAnalyzer().analyze_items(
            batch_result.all_ordered_items,
            source_ids=corpus.source_ids,
            diagnostics=corpus.all_diagnostics,
        )
        self.texts = chunk_texts(batch_result.all_ordered_items)

    def test_the_model_still_holds_the_full_path(self):
        self.assertEqual(
            sorted(f.source_id for f in self.report.files),
            ["/corp/sas/adhoc/load.sas", "/corp/sas/etl/load.sas",
             "/corp/sas/report.sas"],
        )

    def test_the_corpus_report_prints_no_directory_prefix(self):
        text = self.report.to_markdown()
        self.assertNotIn("/corp/sas", text)
        self.assertIn("etl/load.sas", text)
        self.assertIn("adhoc/load.sas", text)
        self.assertIn("report.sas", text)

    def test_the_dependency_section_names_both_ends_of_an_edge(self):
        text = self.report.to_markdown()
        self.assertIn("| etl/load.sas | adhoc/load.sas |", text)
        self.assertIn("**Wave 1**: etl/load.sas", text)

    def test_cross_file_evidence_names_its_peers(self):
        scratch = _file(self.report, "/corp/sas/adhoc/load.sas")
        assert scratch.cross_file is not None
        joined = " ".join(scratch.cross_file.imports)
        self.assertIn("etl/load.sas", joined)
        self.assertNotIn("/corp/sas", joined)

    def test_an_individual_report_names_the_file_and_prints_its_path_once(self):
        markdown = render_file_report(
            _file(self.report, "/corp/sas/adhoc/load.sas"),
            texts=self.texts,
            names=self.report.names,
        )
        self.assertTrue(
            markdown.startswith("# Complexity report — adhoc/load.sas")
        )
        # Once, in the Path bullet — this is the report of the file a reader
        # might need to go open.
        self.assertEqual(markdown.count("/corp/sas/adhoc/load.sas"), 1)
        self.assertIn("- Path: `/corp/sas/adhoc/load.sas`", markdown)
        self.assertIn("- Depends on: etl/load.sas", markdown)

    def test_the_index_table_names_files(self):
        written = write_reports(self.report, self.tmp, texts=self.texts)
        overall = written.overall.read_text(encoding="utf-8")
        self.assertNotIn("/corp/sas", overall)
        self.assertIn("| etl/load.sas |", overall)

    def test_the_evaluation_prompt_names_the_file(self):
        prompts = evaluation_prompts(self.report, texts=self.texts)
        prompt = prompts["/corp/sas/adhoc/load.sas"]
        self.assertIn("# Static verdict — adhoc/load.sas", prompt)
        self.assertNotIn("/corp/sas/adhoc", prompt)


# ---------------------------------------------------------------------------
# PDF rendering
# ---------------------------------------------------------------------------


class TestMarkdownToHtml(unittest.TestCase):
    """The HTML handed to the layout engine."""

    def test_tables_are_rendered(self):
        html = markdown_to_html("| a | b |\n| --- | --- |\n| 1 | 2 |\n")
        self.assertIn("<table>", html)
        self.assertIn("<td>1</td>", html)

    def test_raw_html_in_the_source_is_escaped_not_executed(self):
        html = markdown_to_html("```sas\nif a<b and c>d then x=1;\n```\n")
        self.assertIn("&lt;b and c&gt;", html)

    def test_long_code_lines_are_folded(self):
        line = "data x; set y; where " + " and ".join(f"col{i} = {i}" for i in range(40))
        html = markdown_to_html(f"```sas\n{line}\n```\n", code_width=60)
        body = html.split("<code>")[1].split("</code>")[0]
        self.assertTrue(all(len(part) <= 60 for part in body.split("\n")))
        # Folded, not truncated: every token survives.
        self.assertIn("col39", body)


class TestWrapCode(unittest.TestCase):
    """Soft-wrapping, since the layout engine clips instead of folding."""

    def test_short_lines_are_untouched(self):
        self.assertEqual(wrap_code("run;\nquit;", 40), "run;\nquit;")

    def test_continuations_keep_the_original_indentation(self):
        folded = wrap_code("    set " + "a" * 30 + " " + "b" * 30 + ";", 40)
        lines = folded.split("\n")
        self.assertGreater(len(lines), 1)
        self.assertTrue(lines[1].startswith("    "))

    def test_a_single_over_long_token_is_broken(self):
        folded = wrap_code("x" * 100, 20)
        self.assertTrue(all(len(line) <= 20 for line in folded.split("\n")))
        self.assertEqual(folded.replace("\n", ""), "x" * 100)


class TestRenderPdf(unittest.TestCase):
    """`render_pdf` — the Markdown, converted, never replaced."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_it_writes_a_real_pdf_beside_the_markdown(self):
        source = self.tmp / "complexity-report.md"
        source.write_text(
            "# SAS chunk complexity report\n\n"
            "| File | Size |\n| --- | --- |\n| etl/load.sas | Small |\n\n"
            "```sas\ndata work.a; set raw.b; run;\n```\n",
            encoding="utf-8",
        )
        written = render_pdf(source)
        self.assertEqual(written, self.tmp / "complexity-report.pdf")
        self.assertEqual(written.read_bytes()[:5], b"%PDF-")
        # The Markdown is untouched: this converts, it does not replace.
        self.assertTrue(source.is_file())

    def test_the_text_survives_the_conversion(self):
        import pymupdf

        source = self.tmp / "r.md"
        source.write_text("# Heading\n\nA sentence about etl/load.sas.\n", "utf-8")
        with pymupdf.open(render_pdf(source)) as doc:
            text = doc[0].get_text()
        self.assertIn("Heading", text)
        self.assertIn("etl/load.sas", text)

    def test_a_named_destination_is_honoured(self):
        source = self.tmp / "r.md"
        source.write_text("# Heading\n", encoding="utf-8")
        dest = self.tmp / "nested" / "elsewhere.pdf"
        self.assertEqual(render_pdf(source, dest), dest)
        self.assertTrue(dest.is_file())

    def test_a_missing_markdown_file_raises(self):
        with self.assertRaises(PdfRenderError):
            render_pdf(self.tmp / "absent.md")

    @unittest.skipUnless(
        importlib.util.find_spec("matplotlib"), "matplotlib is not installed"
    )
    def test_the_dependency_graph_image_lands_in_the_pdf(self):
        import pymupdf

        corpus = _graph_corpus()
        batch_result = MultiFileBatcher().batch(corpus)
        report = ComplexityAnalyzer().analyze_items(
            batch_result.all_ordered_items,
            source_ids=corpus.source_ids,
            diagnostics=corpus.all_diagnostics,
        )
        written = write_reports(
            report, self.tmp, texts=chunk_texts(batch_result.all_ordered_items)
        )
        self.assertIsNotNone(written.graph)
        with pymupdf.open(render_pdf(written.overall)) as doc:
            images = sum(len(page.get_images()) for page in doc)
        self.assertEqual(images, 1)


class TestPdfCLI(unittest.TestCase):
    """`--pdf` — the flag, and what it refuses to do without a destination."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        (self.tmp / "load.sas").write_text(
            "data work.a;\n  set raw.a;\nrun;\n", encoding="utf-8"
        )

    def _run(self, *args):
        from complexity.__main__ import main

        return main([str(self.tmp), *args])

    def test_out_dir_gets_a_pdf_beside_the_overall_report(self):
        out = self.tmp / "reports"
        self.assertEqual(self._run("--out-dir", str(out), "--pdf"), 0)
        self.assertTrue((out / "complexity-report.md").is_file())
        self.assertEqual(
            (out / "complexity-report.pdf").read_bytes()[:5], b"%PDF-"
        )

    def test_out_gets_a_pdf_of_the_same_name(self):
        out = self.tmp / "estimate.md"
        self.assertEqual(self._run("--out", str(out), "--pdf"), 0)
        self.assertEqual(
            (self.tmp / "estimate.pdf").read_bytes()[:5], b"%PDF-"
        )

    def test_without_a_destination_it_is_an_error_exit(self):
        self.assertEqual(self._run("--pdf"), 1)
        self.assertEqual(list(self.tmp.glob("*.pdf")), [])

    def test_no_pdf_without_the_flag(self):
        out = self.tmp / "reports"
        self.assertEqual(self._run("--out-dir", str(out)), 0)
        self.assertEqual(list(out.glob("*.pdf")), [])


class TestPathsSection(unittest.TestCase):
    """The file's external references, beside its dataset interface.

    Reported, never scored: like the datasets section this says what a migration
    has to provision, which is a different question from how hard the code is.
    """

    SOURCE = (
        "libname dataetl '/sasdata3/dataetl';\n"
        "filename feed ftp 'rates.dat';\n"
        "filename notify email 'ops@example.com';\n"
        "\n"
        "data dataetl.summary;\n"
        "  infile '/data/in/cust.csv';\n"
        "  set dataetl.raw;\n"
        "run;\n"
    )

    def setUp(self):
        self.file = _file(_analyze(self.SOURCE), "t.sas")
        self.text = render_file_report(self.file, texts={})

    def test_every_kind_is_reported(self):
        paths = {r.path for r in self.file.external_refs}
        self.assertEqual(
            paths,
            {"/sasdata3/dataetl", "rates.dat", "ops@example.com", "/data/in/cust.csv"},
        )

    def test_refs_are_grouped_by_location(self):
        self.assertIn("## Paths", self.text)
        self.assertIn("Filesystem (needs a volume or external location)", self.text)
        self.assertIn("Remote services (needs network egress)", self.text)
        self.assertIn("Email destinations", self.text)
        # Provenance travels with the value: the libref is what a reader needs
        # to find the statement again.
        self.assertIn("`/sasdata3/dataetl` — libname `dataetl`", self.text)
        self.assertIn("via ftp", self.text)

    def test_a_file_touching_nothing_outside_gets_no_section(self):
        # An empty heading says nothing the absence does not — the same rule
        # the Datasets section follows.
        plain = _file(_analyze("data work.a;\n  set work.b;\nrun;\n"), "t.sas")
        self.assertEqual(plain.external_refs, [])
        self.assertNotIn("## Paths", render_file_report(plain, texts={}))

    def test_the_rollup_reconciles_against_its_chunks(self):
        # What makes the section auditable: every path in the file list came
        # from some chunk, and that chunk prints it too.
        from_chunks = {r.path for c in self.file.chunks for r in c.external_refs}
        self.assertEqual({r.path for r in self.file.external_refs}, from_chunks)
        self.assertIn("- Paths:", self.text)

    def test_an_unresolved_macro_reference_is_flagged(self):
        scored = _file(_analyze('libname raw "&root/in";\n'), "t.sas")
        self.assertIn(
            "**(unresolved macro reference)**", render_file_report(scored, texts={})
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
