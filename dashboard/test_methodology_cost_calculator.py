"""Unit tests for the Methodology cost calculator's estimate() function.
Run from the dashboard folder: python -m unittest -v test_methodology_cost_calculator
"""
import unittest

import methodology_cost_calculator as calc

ALL_PRODUCTS = list(calc.PRODUCTS)


def _item(result, key):
    return next(i for i in result["items"] if i["key"] == key)


def _show(name, result):
    print(f"\n  [{name}]")
    for i in result["items"]:
        cost = calc.format_usd_range(i["low"], i["high"]) if i["status"] == "ok" else calc.UNKNOWN_LABEL
        print(f"    {i['label']}: {cost}")
    costed = any(i["status"] == "ok" for i in result["items"])
    total = calc.format_usd_range(result["total_low"], result["total_high"]) if costed else "nothing costed yet"
    print(f"    {result['total_label']}: {total}")
    g = result["gpu"]
    if g:
        gpus = g["gpus_low"] if g["gpus_low"] == g["gpus_high"] else f"{g['gpus_low']} to {g['gpus_high']}"
        print(f"    GPUs at once: {gpus} | finish: "
              f"{calc.format_duration_range(g['finish_minutes_low'], g['finish_minutes_high'])}")
    print(f"    warnings: {len(result['warnings'])}")


def _error_message(value):
    try:
        calc.estimate(value, None, "budget", ALL_PRODUCTS)
    except ValueError as exc:
        return str(exc)
    return None


class EstimateTests(unittest.TestCase):
    def test_10_min_budget_single_gpu(self):
        r = calc.estimate(10, None, "budget", ["cv_models"])
        _show("10 min, budget, single GPU, CV models", r)
        cv = _item(r, "cv_models")
        # 600 s x 13 to 20 = 2.17 to 3.33 h, x $0.40/h, no overhead on one GPU.
        self.assertAlmostEqual(cv["low"], 600 * 13 / 3600 * 0.40)
        self.assertAlmostEqual(cv["high"], 600 * 20 / 3600 * 0.40)
        self.assertEqual((r["gpu"]["gpus_low"], r["gpu"]["gpus_high"]), (1, 1))
        self.assertAlmostEqual(r["gpu"]["finish_minutes_low"], 130)
        self.assertAlmostEqual(r["gpu"]["finish_minutes_high"], 200)
        self.assertEqual(r["warnings"], [])

    def test_45_min_in_10_min_a100_matches_table_3_with_overhead(self):
        r = calc.estimate(45, 10, "a100", ["cv_models"])
        _show("45 min in 10 min, A100, CV models", r)
        cv = _item(r, "cv_models")
        self.assertEqual(calc.format_usd_range(cv["low"], cv["high"]), "$18 to $28")  # Table 3, Datacenter A100
        self.assertEqual((r["gpu"]["gpus_low"], r["gpu"]["gpus_high"]), (59, 90))   # "about 60 to 90 GPUs"
        self.assertLessEqual(r["gpu"]["finish_minutes_high"], 10 + 1e-6)
        self.assertEqual(r["warnings"], [calc.CHUNKING_WARNING])

    def test_90_min_in_20_min_serverless_matches_table_3_with_overhead(self):
        r = calc.estimate(90, 20, "serverless_a100", ["cv_models"])
        _show("90 min in 20 min, serverless A100, CV models", r)
        cv = _item(r, "cv_models")
        self.assertEqual(calc.format_usd_range(cv["low"], cv["high"]), "$61 to $94")  # Table 3, Serverless A100
        self.assertEqual((r["gpu"]["gpus_low"], r["gpu"]["gpus_high"]), (59, 90))
        self.assertEqual(r["warnings"], [calc.CHUNKING_WARNING])

    def test_table_3_all_tiers_with_overhead(self):
        expected = {
            ("budget", 45, 10): "$5 to $8", ("budget", 90, 20): "$10 to $15",
            ("mid", 45, 10): "$10 to $15", ("mid", 90, 20): "$20 to $30",
            ("a100", 45, 10): "$18 to $28", ("a100", 90, 20): "$37 to $56",
            ("serverless_a100", 45, 10): "$30 to $47", ("serverless_a100", 90, 20): "$61 to $94",
        }
        for (tier, minutes, target), text in expected.items():
            cv = _item(calc.estimate(minutes, target, tier, ["cv_models"]), "cv_models")
            # The table shows whole dollars; the calculator keeps cents under $10.
            self.assertEqual(f"${cv['low']:.0f} to ${cv['high']:.0f}", text, (tier, minutes, target))

    def test_each_product_alone(self):
        for key in ALL_PRODUCTS:
            r = calc.estimate(90, None, "budget", [key])
            _show(f"90 min, budget, single GPU, only {key}", r)
            self.assertEqual([i["key"] for i in r["items"]], [key])
            item = r["items"][0]
            if key == "corner_kicks":
                self.assertEqual(item["status"], "unknown")
                self.assertEqual((r["total_low"], r["total_high"]), (0, 0))
            else:
                self.assertGreater(item["low"], 0)
                self.assertGreaterEqual(item["high"], item["low"])
                self.assertAlmostEqual(r["total_low"], item["low"])
                self.assertAlmostEqual(r["total_high"], item["high"])
            # Only the CV models need a GPU.
            self.assertEqual(r["gpu"] is not None, key == "cv_models")

    def test_all_products_together(self):
        r = calc.estimate(90, None, "budget", ALL_PRODUCTS)
        _show("90 min, budget, single GPU, all products", r)
        self.assertEqual([i["key"] for i in r["items"]], ALL_PRODUCTS)
        known = [i for i in r["items"] if i["status"] == "ok"]
        self.assertEqual(len(known), 3)
        self.assertAlmostEqual(r["total_low"], sum(i["low"] for i in known))
        self.assertAlmostEqual(r["total_high"], sum(i["high"] for i in known))
        # Full match on the budget tier: 19.5 to 30 h x $0.40 = $7.80 to $12 for the CV models.
        cv = _item(r, "cv_models")
        self.assertAlmostEqual(cv["low"], 7.80)
        self.assertAlmostEqual(cv["high"], 12.00)

    def test_unknown_cost_product_is_excluded_from_total(self):
        with_unknown = calc.estimate(45, None, "mid", ["cv_models", "corner_kicks"])
        without = calc.estimate(45, None, "mid", ["cv_models"])
        _show("45 min, mid, CV models + corner kicks", with_unknown)
        corner = _item(with_unknown, "corner_kicks")
        self.assertEqual(corner["status"], "unknown")
        self.assertIsNone(corner["low"])
        self.assertIsNone(corner["high"])
        self.assertTrue(with_unknown["has_unknown"])
        self.assertEqual(with_unknown["total_label"], "Total, excluding unknown items")
        self.assertAlmostEqual(with_unknown["total_low"], without["total_low"])
        self.assertAlmostEqual(with_unknown["total_high"], without["total_high"])
        self.assertEqual(without["total_label"], "Total")

    def test_unrecognised_product_or_tier_is_rejected(self):
        with self.assertRaises(ValueError):
            calc.estimate(45, None, "mid", ["scouting_report"])
        with self.assertRaises(ValueError):
            calc.estimate(45, None, "h100", ["cv_models"])

    def test_zero_or_negative_length_is_rejected(self):
        for bad in (0, -5, -0.1, float("nan"), float("inf"), None, "90", True):
            with self.assertRaises(ValueError, msg=repr(bad)):
                calc.estimate(bad, None, "budget", ALL_PRODUCTS)
        print("\n  [zero / negative / non-numeric length] rejected with ValueError:", _error_message(0))

    def test_zero_or_negative_turnaround_is_rejected(self):
        for bad in (0, -10, float("nan")):
            with self.assertRaises(ValueError, msg=repr(bad)):
                calc.estimate(90, bad, "budget", ["cv_models"])

    def test_turnaround_slower_than_one_gpu_needs_no_chunking(self):
        r = calc.estimate(10, 600, "budget", ["cv_models"])
        single = calc.estimate(10, None, "budget", ["cv_models"])
        self.assertEqual((r["gpu"]["gpus_low"], r["gpu"]["gpus_high"]), (1, 1))
        self.assertEqual(r["warnings"], [])
        self.assertAlmostEqual(r["total_high"], single["total_high"])  # no overhead added

    def test_turnaround_faster_than_tables_warns(self):
        r = calc.estimate(90, 10, "a100", ["cv_models"])
        _show("90 min in 10 min, A100, CV models", r)
        self.assertGreater(r["gpu"]["gpus_high"], calc.TABLE_MAX_GPUS)
        self.assertEqual(r["warnings"], [calc.CHUNKING_WARNING, calc.FASTER_THAN_TABLES_WARNING])

    def test_gemini_products_ignore_tier_and_turnaround(self):
        a = calc.estimate(90, None, "budget", ["match_report", "training_plan"])
        b = calc.estimate(90, 20, "serverless_a100", ["match_report", "training_plan"])
        self.assertAlmostEqual(a["total_low"], b["total_low"])
        self.assertAlmostEqual(a["total_high"], b["total_high"])
        self.assertIsNone(b["gpu"])

    def test_wording(self):
        r = calc.estimate(90, 10, "a100", ALL_PRODUCTS)
        text = " ".join([i["how"] for i in r["items"]] + r["warnings"] + [calc.ESTIMATE_LABEL, calc.UNKNOWN_LABEL]).lower()
        for banned in ("accurate", "100%"):
            self.assertNotIn(banned, text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
