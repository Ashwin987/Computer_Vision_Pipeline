"""Unit tests for the Methodology cost calculator's estimate() function.
Run from the dashboard folder: python -m unittest -v test_methodology_cost_calculator
"""
import os
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
        # 600 s x 13 to 20 = 2.17 to 3.33 h, x $0.40/h, plus 5 min of startup on the one GPU.
        startup = calc.STARTUP_MINUTES_PER_GPU / 60 * 0.40
        self.assertAlmostEqual(cv["low"], 600 * 13 / 3600 * 0.40 + startup)
        self.assertAlmostEqual(cv["high"], 600 * 20 / 3600 * 0.40 + startup)
        self.assertEqual((r["gpu"]["gpus_low"], r["gpu"]["gpus_high"]), (1, 1))
        self.assertAlmostEqual(r["gpu"]["finish_minutes_low"], 130)
        self.assertAlmostEqual(r["gpu"]["finish_minutes_high"], 200)
        self.assertEqual(r["warnings"], [])

    def test_45_min_in_10_min_a100_matches_table_3_with_overhead(self):
        r = calc.estimate(45, 10, "a100", ["cv_models"])
        _show("45 min in 10 min, A100, CV models", r)
        cv = _item(r, "cv_models")
        self.assertEqual(calc.format_usd_range(cv["low"], cv["high"]), "$22 to $34")  # Table 3, Datacenter A100
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

    def test_table_3_all_tiers_match_the_calculator(self):
        from methodology_gpu_costs import _table3_rows, TABLE3_TIER_NAMES
        rows = {name: row for name, row in zip(TABLE3_TIER_NAMES.values(), _table3_rows())}
        for tier, name in TABLE3_TIER_NAMES.items():
            half = calc.estimate(45, 10, tier, ["cv_models"])
            match = calc.estimate(90, 20, tier, ["cv_models"])
            self.assertEqual(rows[name][3], calc.format_usd_range(half["total_low"], half["total_high"]), tier)
            self.assertEqual(rows[name][5], calc.format_usd_range(match["total_low"], match["total_high"]), tier)

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
        startup = calc.STARTUP_MINUTES_PER_GPU / 60 * 0.40
        self.assertAlmostEqual(cv["low"], 7.80 + startup)
        self.assertAlmostEqual(cv["high"], 12.00 + startup)

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


class TestTargetTurnaroundAndCalculatorApp(unittest.TestCase):
    def _estimate(self, video, target, tier="budget", products=ALL_PRODUCTS):
        return calc.estimate(video, target, tier, products)

    def test_45_min_in_10_min_finishes_exactly_at_target(self):
        g = self._estimate(45, 10)["gpu"]
        self.assertEqual(calc.format_duration_range(g["finish_minutes_low"], g["finish_minutes_high"]), "10 min")
        self.assertEqual((g["gpus_low"], g["gpus_high"]), (59, 90))

    def test_90_min_in_20_min_finishes_exactly_at_target(self):
        g = self._estimate(90, 20)["gpu"]
        self.assertEqual(calc.format_duration_range(g["finish_minutes_low"], g["finish_minutes_high"]), "20 min")
        self.assertEqual((g["gpus_low"], g["gpus_high"]), (59, 90))

    def test_90_min_in_10_min_budget_shows_cost_and_gpus(self):
        r = self._estimate(90, 10)
        self.assertEqual(calc.format_usd_range(r["total_low"], r["total_high"]), "$12 to $19")
        self.assertEqual((r["gpu"]["gpus_low"], r["gpu"]["gpus_high"]), (117, 180))

    def test_90_min_in_50_min_budget_costs_about_10_to_16(self):
        r = self._estimate(90, 50)
        self.assertEqual(calc.format_usd_range(r["total_low"], r["total_high"]), "$9.05 to $14")

    def test_target_longer_than_single_gpu_time_uses_one_gpu(self):
        r = self._estimate(90, 5000)
        self.assertEqual((r["gpu"]["gpus_low"], r["gpu"]["gpus_high"]), (1, 1))
        self.assertEqual(r["warnings"], [])
        self.assertEqual(calc.format_duration_range(r["gpu"]["finish_minutes_low"], r["gpu"]["finish_minutes_high"]), "19.5 to 30 h")

    def test_custom_targets_are_costed_for_any_whole_minute(self):
        for target in (1, 7, 13, 25, 37, 240):
            r = self._estimate(90, target)
            self.assertGreater(r["total_high"], 0, target)
            self.assertGreaterEqual(r["gpu"]["gpus_high"], r["gpu"]["gpus_low"], target)

    def test_240_minute_target_on_90_minute_video(self):
        r = self._estimate(90, 240)
        self.assertEqual((r["gpu"]["gpus_low"], r["gpu"]["gpus_high"]), (5, 8))
        self.assertEqual(r["warnings"], [calc.CHUNKING_WARNING])

    def test_non_whole_or_non_positive_targets_are_rejected(self):
        for bad in (0, -5, 2.5, 0.5, float("nan")):
            with self.assertRaises(ValueError, msg=repr(bad)):
                calc.estimate(90, bad, "budget", ALL_PRODUCTS)

    def test_app_shows_cost_for_custom_targets_and_tiers(self):
        for tier in calc.GPU_TIERS:
            at = _run_calculator(90, 7, tier)
            self.assertFalse(at.error, tier)
            self.assertIn("$", _total_text(at), tier)
            self.assertEqual(_metric(at, "Expected finish time (CV models)"), "7 min", tier)

    def test_app_rejects_zero_and_negative_targets_without_cost(self):
        for bad in (0, -5):
            at = _run_calculator(90, bad, "budget")
            self.assertTrue(at.error, bad)
            self.assertIn("whole number", at.error[0].value)
            self.assertEqual(_total_text(at), "", bad)

    def test_app_recalculates_when_target_changes(self):
        at = _run_calculator(90, 10, "budget")
        self.assertEqual(_metric(at, "Expected finish time (CV models)"), "10 min")
        at.number_input(key="mcc_target_minutes").set_value(25).run()
        self.assertEqual(_metric(at, "Expected finish time (CV models)"), "25 min")
        self.assertIn("$", _total_text(at))

    def test_app_shows_gpu_assumption_caption(self):
        at = _run_calculator(90, 10, "budget")
        captions = " ".join(c.value for c in at.caption)
        self.assertIn("Assumes about 117 to 180 GPUs running in parallel", captions)


def _run_calculator(video, target, tier, products=None):
    from streamlit.testing.v1 import AppTest
    here = os.path.dirname(os.path.abspath(__file__))
    code = (
        "import sys\n"
        f"sys.path.insert(0, {here!r})\n"
        "import methodology_cost_calculator as mcc\n"
        "mcc.render_cost_calculator()\n"
    )
    at = AppTest.from_string(code, default_timeout=60).run()
    at.number_input(key=calc._MINUTES_KEY).set_value(float(video)).run()
    at.radio(key="mcc_speed").set_value(calc._TARGET).run()
    at.number_input(key="mcc_target_minutes").set_value(float(target)).run()
    at.selectbox(key="mcc_tier").set_value(tier).run()
    for key in calc.PRODUCTS:
        at.checkbox(key=f"mcc_product_{key}").set_value(products is None or key in products).run()
    return at


def _total_text(at):
    return {m.label: m.value for m in at.metric}.get("Estimated total cost", "")


def _metric(at, label):
    return {m.label: m.value for m in at.metric}.get(label)


class TestStartupModelAndLayout(unittest.TestCase):
    TARGETS = (10, 20, 30, 50, 60, 120)

    def _total(self, video, target, tier="budget", products=None):
        return calc.estimate(video, target, tier, products or ALL_PRODUCTS)

    def test_shorter_target_costs_more_whenever_gpu_count_differs(self):
        for video in (45, 90):
            results = [(t, self._total(video, t)) for t in self.TARGETS]
            for (t_short, r_short), (t_long, r_long) in zip(results, results[1:]):
                if r_short["gpu"]["gpus_high"] != r_long["gpu"]["gpus_high"]:
                    self.assertGreater(r_short["total_high"], r_long["total_high"], (video, t_short, t_long))

    def test_target_longer_than_single_gpu_time_costs_the_no_rush_price(self):
        for video in (45, 90):
            self.assertAlmostEqual(self._total(video, 5000)["total_high"], self._total(video, None)["total_high"])
            self.assertAlmostEqual(self._total(video, 5000)["total_low"], self._total(video, None)["total_low"])

    def test_speed_factor_halves_compute_cost_at_two(self):
        from unittest import mock
        base = calc._cv_models(90, None, "budget")[3]["compute_cost_high"]
        with mock.patch.dict(calc.GPU_TIERS["budget"], {"speed_factor": 2.0}):
            fast = calc._cv_models(90, None, "budget")[3]["compute_cost_high"]
        self.assertAlmostEqual(fast, base / 2)

    def test_price_grid_cells_equal_the_calculator(self):
        lengths = [10, 45, 90]
        grid = calc.price_grid(lengths, self.TARGETS + (None,), "mid", ALL_PRODUCTS)
        for length, cells in grid:
            for target, cell in zip(self.TARGETS + (None,), cells):
                expected = calc._total_text(calc.estimate(length, target, "mid", ALL_PRODUCTS))
                self.assertEqual(cell, expected, (length, target))

    def test_cost_card_is_first_and_before_the_breakdown_table(self):
        for kwargs in (dict(video=90, target=20), dict(video=45, target=10), dict(video=90, target=None)):
            at = _run_calculator(kwargs["video"], kwargs["target"] or 20, "budget")
            if kwargs["target"] is None:
                at.radio(key="mcc_speed").set_value(calc._NO_RUSH).run()
            order = _element_types(at)
            self.assertIn("metric", order)
            first_table = order.index("table")
            self.assertLess(order.index("metric"), first_table, kwargs)
            self.assertIn("$", _total_text(at), kwargs)

    def test_only_corner_kicks_shows_nothing_costed_not_blank(self):
        at = _run_calculator(90, 20, "budget", products=["corner_kicks"])
        self.assertEqual(_total_text(at), "nothing costed yet")
        self.assertIn("Corner kicks: not costed yet (unknown), not included in the total.",
                      " ".join(m.value for m in at.markdown))

    def test_match_report_only_shows_a_cost_card(self):
        at = _run_calculator(90, 20, "budget", products=["match_report"])
        self.assertIn("$", _total_text(at))
        self.assertEqual(_metric(at, "GPUs needed at once"), "none needed")

    def test_no_rush_shows_a_cost_card(self):
        at = _run_calculator(90, 20, "budget")
        at.radio(key="mcc_speed").set_value(calc._NO_RUSH).run()
        self.assertIn("$", _total_text(at))
        self.assertEqual(_metric(at, "GPUs needed at once"), "1")


def _element_types(at):
    out = []

    def walk(node):
        for child in getattr(node, "children", {}).values():
            out.append(getattr(child, "type", ""))
            walk(child)

    walk(at.main)
    return out
