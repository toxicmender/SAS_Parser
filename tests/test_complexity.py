"""
test_complexity.py — unit tests for the complexity analysis package
(zero LLM, zero disk I/O).

Run:  python -m pytest tests/test_complexity.py -v
"""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from chunker import SasChunkBatcher, SasSemanticChunker
from complexity import (
    ComplexityAnalyzer,
    ComplexityTier,
    SparkParity,
    detect_constructs,
    max_tier,
    sort_by_complexity,
    worst_parity,
)
from complexity import rules


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
        self.assertEqual(scored.translation_difficulty, SparkParity.DIRECT)
        self.assertIn("global_statement:let", _names(scored))

    def test_simple_proc_sql_is_low(self):
        scored = _only(
            "proc sql;\n  create table work.out as select * from work.in;\nquit;\n"
        )
        self.assertEqual(scored.tier, ComplexityTier.LOW)
        self.assertEqual(scored.translation_difficulty, SparkParity.DIRECT)
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
        self.assertEqual(scored.translation_difficulty, SparkParity.PARTIAL)
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
        self.assertEqual(scored.translation_difficulty, SparkParity.HARD)
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
        self.assertEqual(scored.translation_difficulty, SparkParity.MANUAL)
        self.assertIn("kind:MACRO_DEFINITION", _names(scored))

    def test_one_to_one_merge_without_by_is_high(self):
        """Essentials Ch. 21: a BY-less MERGE pairs rows by position, with no
        key variable — there is no Spark join that reproduces it."""
        scored = _only("data work.out;\n  merge work.a work.b;\nrun;\n")
        self.assertEqual(scored.tier, ComplexityTier.HIGH)
        self.assertEqual(scored.translation_difficulty, SparkParity.HARD)
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
        self.assertEqual(scored.translation_difficulty, SparkParity.HARD)
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
        self.assertEqual(worst_parity([]), SparkParity.DIRECT)

    def test_helpers_pick_extremes(self):
        self.assertEqual(
            max_tier([ComplexityTier.LOW, ComplexityTier.HIGH, ComplexityTier.MEDIUM]),
            ComplexityTier.HIGH,
        )
        self.assertEqual(
            worst_parity([SparkParity.DIRECT, SparkParity.MANUAL, SparkParity.PARTIAL]),
            SparkParity.MANUAL,
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
        for construct in detect_constructs(source):
            self.assertIn(
                construct.name,
                rules.DETECTOR_RULES,
                f"detector '{construct.name}' has no DETECTOR_RULES entry",
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
        self.assertEqual(batch.translation_difficulty, SparkParity.HARD)
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
        self.assertEqual(self.report.overall_difficulty, SparkParity.MANUAL)

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


class TestCatalogueIntegrity(unittest.TestCase):
    def test_every_spec_weight_matches_its_tier(self):
        expected = {
            ComplexityTier.LOW: rules.WEIGHT_LOW,
            ComplexityTier.MEDIUM: rules.WEIGHT_MEDIUM,
            ComplexityTier.HIGH: rules.WEIGHT_HIGH,
        }
        for catalogue_name, catalogue in rules.ALL_RULES.items():
            for key, spec in catalogue.items():
                self.assertEqual(
                    spec.weight,
                    expected[spec.tier],
                    f"{catalogue_name}[{key}] weight does not match its tier",
                )

    def test_every_spec_has_a_category(self):
        for catalogue_name, catalogue in rules.ALL_RULES.items():
            for key, spec in catalogue.items():
                self.assertTrue(
                    spec.category, f"{catalogue_name}[{key}] has no category"
                )

    def test_brief_constructs_land_in_their_stated_tiers(self):
        """The tier assignments the project brief names, asserted directly."""
        self.assertEqual(
            rules.PROC_RULES["sql"].tier, ComplexityTier.LOW
        )
        self.assertEqual(
            rules.GLOBAL_STATEMENT_RULES["let"].tier, ComplexityTier.LOW
        )
        for key in ("hash",):
            self.assertEqual(
                rules.COMPONENT_OBJECT_RULES[key].tier, ComplexityTier.MEDIUM
            )
        for key in ("merge", "filename_sftp", "filename_email"):
            self.assertEqual(
                rules.DETECTOR_RULES[key].tier, ComplexityTier.MEDIUM
            )
        for key in ("array", "do_loop", "do_while", "do_until"):
            self.assertEqual(rules.DETECTOR_RULES[key].tier, ComplexityTier.HIGH)
        self.assertEqual(
            rules.KIND_RULES["MACRO_DEFINITION"].tier, ComplexityTier.HIGH
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
