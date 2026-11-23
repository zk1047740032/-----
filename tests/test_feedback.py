import tempfile
import unittest
from pathlib import Path

import numpy as np

from fiber1100_algorithm import evaluate_shape_record, evaluate_waveform, run_leveling
from waveform_core.leveling import LevelingSession
from waveform_core.timing_pulses import prepare_timed_waveform


def samples(seed, dip=0.0, *, width=150, period=1000, count=5, dt=.2,
            deep_index=None, short_index=None, shoulder_spike=False, distorted_index=None):
    time = np.arange(-200., (count-1)*period + 2*width, dt)
    volts = np.random.default_rng(seed).normal(0, .0002, len(time))
    for i in range(count):
        relative_time = time - i*period
        duration = .66*width if i == short_index else width
        pulse = np.clip(np.minimum(relative_time, duration-relative_time)/7, 0, 1)
        phase = relative_time/duration
        distortion = dip*np.exp(-.5*((phase-.51)/.087)**2)
        if i == distorted_index:
            distortion -= .23*np.exp(-.5*((phase-.51)/.087)**2)
        if i == deep_index:
            distortion[(phase>.39) & (phase<.56)] -= .71
        if shoulder_spike:
            distortion[(phase>=.90) & (phase<.902)] -= .15
            distortion[(phase>=.88) & (phase<.882)] += .032
        volts += pulse*(1+distortion)
    return time*1e-9, volts


def prepare(seed, dip=0.0, **settings):
    return prepare_timed_waveform(*samples(seed, dip, **settings))


class WaveformFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.session = LevelingSession()
        self.start = prepare(0, -.22)

    def start_session(self):
        state = self.session.start(self.start, evaluate_shape_record(self.start))
        self.assertEqual(state["next_action"], {"kind": "trial", "axis": "y2", "step": 10})
        return state

    def observe(self, record, *, applied=True):
        report = evaluate_shape_record(record, window=self.session.window)
        return self.session.observe(record, report, applied=applied)

    def test_samples_alone_yield_hold_and_shape_directed_trials(self):
        time, voltage = samples(0, -.22)
        original = evaluate_waveform(time, voltage)
        shifted = evaluate_waveform(time, .87*voltage+.12)
        self.assertIsNone(original["window"]["reference_operating_point"])
        self.assertEqual(original["shape"]["candidate"], {"axis": "y2", "step": 10})
        self.assertEqual(shifted["shape"]["candidate"], original["shape"]["candidate"])
        self.assertAlmostEqual(shifted["flatness"]["raw_max_pct"], original["flatness"]["raw_max_pct"])
        high = evaluate_shape_record(prepare(0, .22))
        self.assertEqual(high["shape"]["candidate"], {"axis": "y2", "step": -10})
        flat = evaluate_waveform(*samples(1))
        self.assertEqual(flat["shape"]["status"], "flat")
        self.assertEqual(flat["shape"]["directions"], {f"y{i}": "hold" for i in range(3)})
        self.assertTrue(flat["flatness"]["raw_passed"])

    def test_one_defective_pulse_is_not_hidden_by_representative_selection(self):
        record = prepare(2, short_index=2)
        report = evaluate_shape_record(record)
        self.assertEqual(len(record.pulses), 5)
        self.assertNotEqual(report["shape"]["representative_pulse"], 3)
        self.assertEqual(report["flatness"]["pulse_coverage_check"]["status"], "early_loss_while_peers_hold")
        self.assertIsNone(self.session.start(record, report)["next_action"])
        deep = evaluate_waveform(*samples(3, -.12, deep_index=3))
        self.assertEqual(deep["flatness"]["pulses"][3]["geometry_recovered"], True)
        self.assertEqual(deep["shape"]["status"], "uncertain")
        self.assertGreater(deep["flatness"]["raw_max_pct"], 70)

    def test_each_region_has_both_directions(self):
        time = np.arange(-100., 600., .2)
        shape = np.clip(np.minimum(time, 150.-time)/7, 0, 1)
        for idx, (lo, hi) in enumerate(((.064, .15), (.18, .32), (.44, .60))):
            phase = time/150.
            zone = np.clip(np.minimum(phase-lo+.025, hi-phase+.025)/.025, 0, 1)
            for sign in (-1, 1):
                with self.subTest(axis=idx, sign=sign):
                    report = evaluate_waveform(time*1e-9, .12+shape*(1+sign*.21*zone))
                    expect = {"axis": f"y{idx}", "step": -10*sign}
                    self.assertEqual(report["shape"]["candidate"], expect)

    def test_median_gain_cannot_hide_a_worsening_pulse(self):
        self.start_session()
        self.assertEqual(self.observe(prepare(8, -.12, distorted_index=2))["status"], "repeat")
        state = self.observe(prepare(9, -.12, distorted_index=2), applied=False)
        self.assertEqual(state["next_action"]["kind"], "rollback")
        response = state["responses"][0]
        self.assertGreater(response["gain_range_pp"][0], response["noise_pp"])
        self.assertLess(response["worst_rms"]["gain_range_pp"][1], -response["worst_rms"]["noise_pp"])
        self.assertEqual(response["verdict"], "pulse_regression")
        self.assertEqual(state["best"]["offsets"], [0, 0, 0])

    def test_unclear_single_pulse_split_requires_more_waveform_evidence(self):
        with self.assertRaisesRegex(ValueError, "外沿有歧义"):
            prepare(2, count=1, deep_index=0)

    def test_fresh_capture_guard_preserves_acknowledged_step(self):
        self.start_session()
        self.session.mark_applied()
        with self.assertRaisesRegex(ValueError, "已用于本轮"):
            self.observe(self.start)
        time, voltage = samples(0, -.22)
        with self.assertRaisesRegex(ValueError, "已用于本轮"):
            self.observe(prepare_timed_waveform(time+.0001, voltage))
        for different, message in ((prepare(9, -.14, dt=.4), "采样间隔"),
                                   (prepare(9, -.14, period=1200), "重复频率"),
                                   (prepare(9, -.14, width=220), "半高宽")):
            with self.assertRaisesRegex(ValueError, message):
                self.observe(different)
        self.assertEqual(self.session.snapshot()["offsets"], [0, 0, 0])
        self.assertEqual(self.session.snapshot()["pending_offsets"], [0, 0, 10])
        self.assertEqual(self.session.snapshot()["next_action"]["kind"], "capture")
        self.assertEqual(self.observe(prepare(1, -.128))["offsets"], [0, 0, 10])
        self.assertEqual(self.session.snapshot()["trials"], 1)

    def test_repeats_and_measured_response_change_next_step(self):
        reference = self.start_session()["window"]
        self.assertEqual(self.observe(prepare(1, -.128))["status"], "repeat")
        state = self.observe(prepare(2, -.131), applied=False)
        self.assertEqual(state["next_action"], {"kind": "trial", "axis": "y2", "step": 10})
        self.assertEqual(state["window"], reference)
        self.assertIn("本轮已测区域响应", state["message"])
        evidence, = state["responses"]
        self.assertGreater(evidence["gain_range_pp"][0], evidence["noise_pp"])
        self.assertGreater(evidence["region_delta_pct"]["y2"], 0)
        self.assertLess(evidence["region_delta_pct"]["y0"], 0)
        state = self.observe(prepare(3, -.018))
        self.assertEqual(state["status"], "flat")
        self.assertIsNone(state["next_action"])
        self.assertEqual(state["best"]["offsets"], [0, 0, 20])
        self.assertTrue(state["best"]["raw_passed"])

    def test_worse_step_rolls_back_and_learns_opposite_direction(self):
        self.start_session()
        self.assertEqual(self.observe(prepare(5, -.32))["status"], "repeat")
        state = self.observe(prepare(6, -.32), applied=False)
        self.assertEqual(state["next_action"], {"kind": "rollback", "axis": "y2", "step": -10})
        self.assertEqual(state["best"]["offsets"], [0, 0, 0])
        state = self.observe(prepare(7, -.22))
        self.assertEqual(state["offsets"], [0, 0, 0])
        self.assertEqual(state["next_action"], {"kind": "trial", "axis": "y2", "step": -10})

    def test_unresolved_small_change_is_undone_after_repeat_budget(self):
        self.start_session()
        self.assertEqual(self.observe(prepare(11, -.219))["status"], "repeat")
        self.assertEqual(self.observe(prepare(12, -.221), applied=False)["status"], "repeat")
        state = self.observe(prepare(13, -.22), applied=False)
        self.assertEqual(state["next_action"], {"kind": "rollback", "axis": "y2", "step": -10})
        self.assertEqual(state["responses"][0]["verdict"], "repeat")

    def test_shape_target_cannot_hide_losing_the_raw_limit(self):
        before = prepare(20, -.105)
        initial = evaluate_shape_record(before)
        self.assertTrue(initial["flatness"]["raw_passed"])
        self.assertEqual(self.session.start(before, initial)["status"], "trial")
        after = prepare(22, shoulder_spike=True)
        report = evaluate_shape_record(after, window=self.session.window)
        self.assertEqual(report["shape"]["status"], "flat")
        self.assertFalse(report["flatness"]["raw_passed"])
        state = self.session.observe(after, report)
        self.assertEqual(state["next_action"]["kind"], "rollback")
        self.assertTrue(state["best"]["raw_passed"])

    def test_automatic_runner_collects_after_every_confirmed_change(self):
        with tempfile.TemporaryDirectory() as folder:
            offsets = {f"y{i}": 0 for i in range(3)}
            events = []
            dips = iter((-.22, -.128, -.131, -.018))

            def capture():
                time, voltage = samples(len(events), next(dips))
                path = Path(folder) / f"{len(events)}.npz"
                np.savez(path, time_s=time, voltage_v=voltage)
                events.append(("capture", offsets["y2"]))
                return path

            def apply_step(axis, delta):
                offsets[axis] += delta
                events.append((axis, delta))
                return True

            state = run_leveling(capture, apply_step, initial_y=[100, 100, 100], bounds=[(0, 200)]*3)
        self.assertEqual(state["status"], "flat")
        self.assertEqual(events, [("capture", 0), ("y2", 10), ("capture", 10),
                                  ("capture", 10), ("y2", 10), ("capture", 20)])
        self.assertEqual(state["best"]["parameters"], [100, 100, 120])

    def test_failed_command_is_never_repeated_and_bounds_are_not_clipped(self):
        with tempfile.TemporaryDirectory() as folder:
            time, voltage = samples(0, -.22)
            path = Path(folder) / "baseline.npz"
            np.savez(path, time_s=time, voltage_v=voltage)
            events = []

            def rejected(axis, delta):
                events.append((axis, delta))
                return False

            state = run_leveling(lambda: path, rejected, initial_y=[100]*3, bounds=[(0, 200)]*3)
            self.assertEqual(events, [("y2", 10)])
            self.assertEqual(state["status"], "interrupted")
            self.assertEqual(state["interrupted_action"]["axis"], "y2")
            self.assertIsNone(state["next_action"])
            limits = [(0, 200), (0, 200), (0, 105)]
            state = run_leveling(lambda: path, rejected, initial_y=[100]*3, bounds=limits)
            self.assertEqual(state["status"], "exhausted")
            self.assertEqual(events, [("y2", 10)])
            self.assertEqual(state["limit_blocks"][0]["requested"], 110)

    def test_capture_interruption_does_not_issue_the_pending_command_again(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "baseline.npz"
            time, voltage = samples(0, -.22)
            np.savez(path, time_s=time, voltage_v=voltage)
            calls = []

            def command(axis, delta):
                calls.append((axis, delta))
                return True

            # The oscilloscope adapter incorrectly returns the old record twice.
            state = run_leveling(lambda: path, command, initial_y=[100]*3, bounds=[(0, 200)]*3)
        self.assertEqual(calls, [("y2", 10)])
        self.assertEqual(state["status"], "interrupted")
        self.assertEqual(state["pending_offsets"], [0, 0, 10])
        self.assertEqual(state["offsets"], [0, 0, 0])
        self.assertIsNone(state["next_action"])


if __name__ == "__main__":
    unittest.main()
