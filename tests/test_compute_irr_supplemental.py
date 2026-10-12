"""Regression tests for scripts/compute_irr_supplemental.py.

Run from the repository root:
  python -I -m unittest discover -s tests -v
"""
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "compute_irr_supplemental.py"

_spec = importlib.util.spec_from_file_location("compute_irr_supplemental", SCRIPT)
irr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(irr)


def panel_from_columns(field, columns):
    """Build {slot: [record]} from {slot: [value per frame]} for one field."""
    return {
        slot: [{"frame_num": i + 1, field: v, "overall": "Keep"} for i, v in enumerate(values)]
        for slot, values in columns.items()
    }


class HandCalculatedAC1(unittest.TestCase):
    """Known-case values worked out by hand, with K = 3 and three raters.

    Raw 1-3 case, field "clarity":
      item 1 ratings (1, 1, 1): all three pairs agree, so item agreement = 1.
      item 2 ratings (1, 2, 3): no pair agrees, so item agreement = 0.
      Pa = (1 + 0) / 2 = 0.5.
      Six ratings in total: value 1 appears 4 times, 2 once, 3 once.
      pi = (4/6, 1/6, 1/6) and sum pi(1 - pi) = 8/36 + 5/36 + 5/36 = 1/2.
      Pe = (1/2) / (K - 1) = 1/4.
      AC1 = (1/2 - 1/4) / (1 - 1/4) = 1/3.
    A chance term that wrongly adds a zero category 0 (K = 4) gives 0.4, so this
    test would fail on the earlier bug.
    """

    def test_raw_three_point_known_case(self):
        panel = panel_from_columns("clarity", {"r1": [1, 1], "r2": [1, 2], "r4": [1, 3]})
        ac1, n = irr.gwet_ac1(panel, "clarity")
        self.assertEqual(n, 2)
        self.assertAlmostEqual(ac1, 1 / 3, places=12)

    def test_collapsed_five_point_known_case(self):
        # Collapsed values: [5,4,4] -> [2,2,2]; [1,3,5] -> [0,1,2].
        # Counts: 0 x1, 1 x1, 2 x4, so Pe = (5/36 + 5/36 + 8/36) / 2 = 1/4 and AC1 = 1/3 on K = 3.
        panel = panel_from_columns("scope_rating", {"r1": [5, 1], "r2": [4, 3], "r4": [4, 5]})
        ac1, n = irr.gwet_ac1(panel, "scope_rating")
        self.assertEqual(n, 2)
        self.assertAlmostEqual(ac1, 1 / 3, places=12)


class MissingDataHandling(unittest.TestCase):
    """A null rating is never imputed; it removes that item from complete-case AC1 and pairwise."""

    def setUp(self):
        # Item 2 has r2 missing, so it drops out of clarity; items 1 and 3 remain.
        self.panel = panel_from_columns("clarity", {"r1": [1, 2, 3], "r2": [1, None, 3], "r4": [1, 2, 3]})

    def test_complete_case_items(self):
        self.assertEqual(irr.complete_items(self.panel, "clarity"), [0, 2])

    def test_ac1_uses_only_complete_items(self):
        # Pa = 1. Counts: 1 x3, 3 x3, 2 x0, so pi = (1/2, 0, 1/2) and Pe = (1/2) / 2 = 1/4.
        # AC1 = (1 - 1/4) / (1 - 1/4) = 1.
        ac1, n = irr.gwet_ac1(self.panel, "clarity")
        self.assertEqual(n, 2)
        self.assertAlmostEqual(ac1, 1.0, places=12)

    def test_pairwise_denominator_is_complete_items(self):
        pairwise, mean_pct = irr.pairwise_agreement(self.panel, "clarity")
        for pair in pairwise.values():
            self.assertEqual(pair["pairs"], 2)
        self.assertAlmostEqual(mean_pct, 1.0)

    def test_missing_ratings_are_listed(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            for slot, name in irr.SLOT_FILES.items():
                records = [{"frame_num": i + 1, **{f: 3 for f in irr.FIELDS}, "overall": "Keep"} for i in range(2)]
                if slot == "r2":
                    records[1]["clarity"] = None
                (tmp_dir / name).write_text(json.dumps(records), encoding="utf-8")
            _, missing = irr.load_panel(tmp_dir, include_frame27=False)
        self.assertEqual(missing, [{"slot": "r2", "frame_num": 2, "field": "clarity"}])


class RealDataMatchesCorrectedK3(unittest.TestCase):
    """Real panel values from reviewer_docs, to six decimals, against the corrected fixed-K3 results."""

    N27 = {
        "clarity": (0.893875, 0.901235),
        "tribal_knowledge": (0.242300, 0.493827),
        "frame_type_rating": (0.573924, 0.654321),
        "post_role_rating": (0.793893, 0.814815),
        "scope_rating": (0.864322, 0.876543),
        "applicability_rating": (0.659306, 0.703704),
        "main_point_quality": (0.975004, 0.975309),
        "epistemic_stance_rating": (0.766333, 0.790123),
    }
    N26 = {
        "clarity": 0.889499,
        "tribal_knowledge": 0.270849,
        "frame_type_rating": 0.590137,
        "post_role_rating": 0.785124,
        "scope_rating": 0.858619,
        "applicability_rating": 0.677625,
        "main_point_quality": 0.974030,
        "epistemic_stance_rating": 0.756384,
    }

    @classmethod
    def setUpClass(cls):
        cls.panel26, cls.missing26 = irr.load_panel(REPO_ROOT / "reviewer_docs", include_frame27=False)
        cls.panel27, cls.missing27 = irr.load_panel(REPO_ROOT / "reviewer_docs", include_frame27=True)
        cls.m27 = irr.compute(cls.panel27)
        cls.m26 = irr.compute(cls.panel26)

    def test_n27_ac1_and_pairwise(self):
        for field, (ac1_expected, pairwise_expected) in self.N27.items():
            with self.subTest(field=field):
                self.assertAlmostEqual(self.m27["fields"][field]["ac1"], ac1_expected, places=6)
                self.assertAlmostEqual(self.m27["fields"][field]["mean_pairwise_pct"], pairwise_expected, places=6)

    def test_n26_ac1(self):
        for field, expected in self.N26.items():
            with self.subTest(field=field):
                self.assertAlmostEqual(self.m26["fields"][field]["ac1"], expected, places=6)

    def test_n27_item_counts_and_missing_location(self):
        self.assertEqual(self.m27["n_frames"], 27)
        self.assertEqual(self.m27["fields"]["clarity"]["items_complete_case"], 27)
        self.assertEqual(self.m27["fields"]["extraction_accuracy"]["items_complete_case"], 26)
        self.assertEqual(self.missing27, [{"slot": "r2", "frame_num": 17, "field": "extraction_accuracy"}])

    def test_n27_frame27_is_the_last_record_for_each_slot(self):
        for slot in irr.SLOT_FILES:
            self.assertEqual(self.panel27[slot][-1]["frame_num"], 27)


class CommandLine(unittest.TestCase):
    def test_cli_exits_zero_and_writes_aggregate(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "irr_metrics.json"
            proc = subprocess.run(
                [sys.executable, "-I", str(SCRIPT), "--output", str(out)],
                capture_output=True, text=True, cwd=tmp,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("slot=r2 frame=17 field=extraction_accuracy", proc.stdout)
            data = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(sorted(data), ["llm_vs_human_n26", "n26", "n27"])
        self.assertEqual(data["n27"]["missing_ratings"][0]["frame_num"], 17)


class LlmVsHuman(unittest.TestCase):
    """Alignment by frame_num and the missing-value denominator."""

    def _panel(self):
        return {slot: [{"frame_num": n, "clarity": v, "overall": "Keep"}
                       for n, v in zip((1, 2, 3), vals)]
                for slot, vals in {"r1": [3, 2, 1], "r2": [3, 2, 2], "r4": [3, 3, 3]}.items()}

    def test_hand_computed_deltas_and_alignment_by_frame_num(self):
        # LLM records deliberately out of order: alignment must use frame_num, not position.
        llm = [{"frame_num": 3, "clarity": 1, "overall": "Revise"},
               {"frame_num": 1, "clarity": 3, "overall": "Keep"},
               {"frame_num": 2, "clarity": 2, "overall": "Keep"}]
        out = irr.llm_vs_human(self._panel(), llm)
        f = out["fields"]["clarity"]
        # LLM mean (3+2+1)/3 = 2; pooled human = (3+3+3+2+2+3+1+2+3)/9 = 22/9.
        self.assertEqual(f["frames_used"], 3)
        self.assertEqual(f["pooled_human_ratings"], 9)
        self.assertAlmostEqual(f["llm_mean"], 2.0)
        self.assertAlmostEqual(f["pooled_human_mean"], 22 / 9)
        self.assertAlmostEqual(f["llm_minus_human"], 2 - 22 / 9)
        self.assertEqual(out["overall"]["llm_keep"], 2)

    def test_missing_value_shrinks_denominator(self):
        llm = [{"frame_num": 1, "clarity": 3, "overall": "Keep"},
               {"frame_num": 2, "clarity": None, "overall": "Keep"},
               {"frame_num": 3, "clarity": 1, "overall": "Keep"}]
        f = irr.llm_vs_human(self._panel(), llm)["fields"]["clarity"]
        self.assertEqual(f["frames_used"], 2)
        self.assertEqual(f["pooled_human_ratings"], 6)
        self.assertEqual(f["frames_dropped_missing"], [2])
        self.assertAlmostEqual(f["llm_mean"], 2.0)
        self.assertAlmostEqual(f["pooled_human_mean"], (3 + 3 + 3 + 1 + 2 + 3) / 6)

    def test_alignment_rejects_missing_or_extra_frames(self):
        base = [{"frame_num": n, "clarity": 1, "overall": "Keep"} for n in (1, 2, 3)]
        with self.assertRaises(ValueError):
            irr.llm_vs_human(self._panel(), base[:2])
        with self.assertRaises(ValueError):
            irr.llm_vs_human(self._panel(), base + [{"frame_num": 27, "clarity": 1, "overall": "Keep"}])
        with self.assertRaises(ValueError):
            irr.llm_vs_human(self._panel(), base + [base[0]])

    def test_original_panel_reported_values(self):
        panel, _ = irr.load_panel(irr.DEFAULT_INPUT_DIR, False)
        out = irr.llm_vs_human(panel, irr.load_json(irr.DEFAULT_INPUT_DIR / irr.LLM_FILE))
        self.assertEqual(out["n_frames"], 26)
        self.assertEqual(out["overall"]["llm_keep"], 12)
        self.assertEqual(out["overall"]["human_majority_keep"], 24)
        self.assertAlmostEqual(out["fields"]["scope_rating"]["llm_minus_human"], -0.858974, places=6)
        self.assertAlmostEqual(out["fields"]["clarity"]["llm_minus_human"], 0.012821, places=6)
        self.assertAlmostEqual(out["fields"]["post_role_rating"]["llm_minus_human"], 0.384615, places=6)


if __name__ == "__main__":
    unittest.main()
