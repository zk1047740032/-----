import unittest

import numpy as np

from fiber1100_algorithm import evaluate_shape_record, evaluate_waveform
from waveform_core.leveling import LevelingSession
from waveform_core.timing_pulses import prepare_timed_waveform

from test_feedback import samples


def capture(seed, dip=0., *, width=150):
    return prepare_timed_waveform(*samples(seed, dip, width=width, period=4000))


class ReferenceAlignmentTests(unittest.TestCase):
    def test_custom_endpoints_and_independent_axis_tolerances(self):
        time_s, voltage_v = samples(1, -.12, period=4000)
        windows = {"y0": (.045, .11), "y1": (.14, .23), "y2": (.39, .54)}
        limits = {"y0": 2., "y1": 3., "y2": 5.}
        report = evaluate_waveform(time_s, voltage_v, red_windows=windows, confirmed_phase=(.16, .755),
                                   alignment_tolerance_pct=limits, raw_limit_pct=20.)
        self.assertEqual(report["window"]["alignment_windows"],
                         windows | {"y1": (.16, .23), "reference": (.635, .755)})
        self.assertEqual(report["shape"]["candidate"], {"axis": "y2", "step": 10})
        self.assertFalse(report["passed"])

        record = prepare_timed_waveform(time_s, voltage_v)
        session = LevelingSession()
        session.start(record, report)
        with self.assertRaisesRegex(ValueError, "锁定"):
            evaluate_shape_record(record, window=session.window, red_windows={"y2": (.4, .54)})
        updated = evaluate_shape_record(record, window=session.window, raw_limit_pct=20.,
                                        alignment_tolerance_pct=limits | {"y2": 15.})
        self.assertTrue(updated["passed"])
        self.assertEqual(updated["shape"]["pulse_regions"], report["shape"]["pulse_regions"])
        self.assertEqual(updated["shape"]["regions"]["y0"]["reference_band_pct"], [-2., 2.])
        self.assertEqual(updated["shape"]["regions"]["y2"]["reference_band_pct"], [-15., 15.])
        self.assertEqual(session.reassess(record, updated)["status"], "flat")

    def test_invalid_red_endpoints_and_tolerance_are_rejected(self):
        record = capture(1)
        with self.assertRaisesRegex(ValueError, "y1 红区"):
            evaluate_shape_record(record, red_windows={"y1": (.25, .15)})
        with self.assertRaisesRegex(ValueError, "允许偏差"):
            evaluate_shape_record(record, alignment_tolerance_pct={"y1": float("nan")})

    def test_y0_stays_on_the_left_shoulder_when_hold_start_moves(self):
        record = capture(1, -.12)
        initial = evaluate_shape_record(record, alignment_tolerance_pct=7.)
        for hold_start in (.065, .116, .14):
            with self.subTest(hold_start=hold_start):
                hold = (hold_start, .755)
                report = evaluate_shape_record(record, alignment_tolerance_pct=7., confirmed_phase=hold)
                window = report["window"]
                self.assertEqual(window["phase"], hold)
                self.assertEqual(window["alignment_windows"]["y0"], (.04, .125))
                self.assertEqual(report["shape"]["regions"]["y0"], initial["shape"]["regions"]["y0"])
                self.assertIsNotNone(report["shape"]["score_pct"])

    def test_thresholds_reclassify_same_numbers_without_another_capture(self):
        record = capture(1, -.12)
        initial = evaluate_shape_record(record, alignment_tolerance_pct=3.)
        self.assertEqual(initial["shape"]["candidate"], {"axis": "y2", "step": 10})
        self.assertTrue(initial["flatness"]["raw_passed"])
        self.assertFalse(initial["passed"])
        session = LevelingSession()
        session.start(record, initial)

        raw_pct = initial["flatness"]["raw_max_pct"]
        aligned = evaluate_shape_record(record, window=session.window, alignment_tolerance_pct=10.,
                                        raw_limit_pct=raw_pct)
        state = session.reassess(record, aligned)
        self.assertEqual(state["status"], "aligned")
        self.assertIsNone(state["next_action"])
        self.assertEqual(aligned["shape"]["directions"], {axis: "hold" for axis in ("y0", "y1", "y2")})
        self.assertFalse(aligned["flatness"]["raw_passed"])
        self.assertEqual(aligned["flatness"]["raw_max_pct"], raw_pct)
        self.assertEqual(aligned["shape"]["pulse_regions"], initial["shape"]["pulse_regions"])
        self.assertEqual(len(state["history"]), 1)
        self.assertEqual(state["history"][0]["thresholds"], initial["thresholds"])
        self.assertEqual(state["threshold_changes"][0]["thresholds"], aligned["thresholds"])

        passed = evaluate_shape_record(record, window=session.window, alignment_tolerance_pct=10.,
                                        raw_limit_pct=np.nextafter(raw_pct, np.inf))
        self.assertTrue(passed["passed"])
        self.assertEqual(session.reassess(record, passed)["status"], "flat")

    def test_lock_ramp_and_reference_geometry_for_each_new_capture(self):
        before = capture(3, -.22)
        options = dict(alignment_tolerance_pct=3., raw_limit_pct=15.)
        first = evaluate_shape_record(before, **options)
        session = LevelingSession()
        target = session.start(before, first)
        self.assertEqual(target["next_action"], {"kind": "trial", "axis": "y2", "step": 10})
        y0 = session.window["alignment_windows"]["y0"]
        blue = session.window["alignment_windows"]["reference"]
        self.assertEqual(y0, (.04, .125))
        self.assertEqual(blue[1], session.window["phase"][1])
        self.assertLess(blue[1], .9)
        after = capture(4, -.12)
        new = evaluate_shape_record(after, window=session.window, **options)
        self.assertEqual(new["window"], first["window"])
        self.assertLess(new["shape"]["score_pct"], first["shape"]["score_pct"])
        self.assertEqual(session.observe(after, new)["status"], "repeat")
        repeated = capture(5, -.121)
        repeat_report = evaluate_shape_record(repeated, window=session.window, **options)
        state = session.observe(repeated, repeat_report, applied=False)
        self.assertEqual(state["offsets"], [0, 0, 10])
        self.assertEqual(state["responses"][0]["verdict"], "better")
        self.assertIn("红蓝偏差改善", state["message"])

    def test_amplitude_change_preserves_relative_blue_measurements(self):
        time_s, voltage_v = samples(6, -.12)
        base = evaluate_shape_record(prepare_timed_waveform(time_s, voltage_v), alignment_tolerance_pct=3.)
        moved = evaluate_shape_record(prepare_timed_waveform(time_s, 1.25*voltage_v + .13),
                                      window=base["window"], alignment_tolerance_pct=3.)
        self.assertAlmostEqual(moved["shape"]["reference_level_v"], 1.25*base["shape"]["reference_level_v"])
        self.assertAlmostEqual(moved["shape"]["score_pct"], base["shape"]["score_pct"])
        self.assertAlmostEqual(moved["flatness"]["raw_max_pct"], base["flatness"]["raw_max_pct"])
