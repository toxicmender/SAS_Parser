"""
test_complexity.py — unit tests for the complexity analysis package
(zero LLM, zero disk I/O).

Run:  python -m pytest tests/test_complexity.py -v
"""

import json
import pathlib
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from chunker import MultiFileBatcher, SasChunkBatcher, SasSemanticChunker
from chunker.models import SasCorpus
from complexity import (
    CROSS_FILE_CONSTRUCTS,
    ComplexityAnalyzer,
    ComplexityTier,
    CrossFileIndex,
    TranslationParity,
    TShirtSize,
    detect_constructs,
    max_size,
    max_tier,
    sort_by_complexity,
    worst_parity,
)
from complexity import rules


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

    def test_anchor_raw_bands_to_medium(self):
        """The anchor is the reference Medium file, by definition."""
        for name in rules.available_profiles():
            sizes = rules.load_ruleset(name).sizes
            points = sizes.points_for(sizes.anchor_raw)
            self.assertEqual(sizes.band_for(points), TShirtSize.MEDIUM, name)
            self.assertAlmostEqual(points, 3.0, places=2, msg=name)

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
        big = _analyze(source, size_anchor=2.0).files[0]
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
        # The floor is what did it, not the banding.
        self.assertLess(f.points, 2.5)

    def test_config_only_file_stays_small(self):
        f = _analyze("%let a = 1;\n%let b = 2;\n%let c = 3;\n").files[0]
        self.assertEqual(f.size, TShirtSize.SMALL)

    def test_floored_by_is_empty_when_banding_stands_alone(self):
        """The floor is only reported when it actually changed the answer."""
        f = _analyze("%macro noop;\n%mend noop;\n", size_anchor=1.0).files[0]
        self.assertEqual(f.size, TShirtSize.EXTRA_LARGE)
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
        source = "".join(
            f"data step{i}; set step{i - 1}; run;\n" for i in range(1, 60)
        )
        result = SasSemanticChunker().chunk_text(source, source_id="big.sas")
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
