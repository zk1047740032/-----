import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest
from unittest.mock import patch

import numpy as np

from waveform_core.timing_pulses import prepare_timed_file
from waveform_simulation import (
    DEFAULT_FREQUENCY,
    DEFAULT_WIDTH,
    SimulationSettings,
    run_simulation,
    scan_axis,
    simulate_capture,
)


def settings_for(destination, **overrides):
    values = dict(frequencies=scan_axis(*DEFAULT_FREQUENCY, "重频"),
                  widths=scan_axis(*DEFAULT_WIDTH, "脉宽"), destination=Path(destination),
                  scope_ip="192.168.1.10", board_port="COM42",
                  alignment_tolerance_pct=7.0, raw_limit_pct=20.0)
    return SimulationSettings(**(values | overrides))


def summary_for(outcome):
    return json.loads((outcome.directory / "run.json").read_text(encoding="utf-8"))


class SimulationTests(unittest.TestCase):
    def test_axis_step_and_invalid_configuration(self):
        self.assertEqual(list(scan_axis(80, 125, 20, "重频")), [80, 100, 120])
        for args in ((0, 100, 20), (100, 80, 20), (80, 100, 0)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                scan_axis(*args, "重频")
        for fields in (dict(scope_ip="invalid"), dict(raw_limit_pct=0),
                       dict(widths=range(5000, 5001)), dict(frequencies=range(1, 2))):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                settings_for(".", **fields)

    def test_single_evaluates_and_saves_loadable_capture(self):
        with TemporaryDirectory() as temporary:
            settings = settings_for(temporary, frequencies=range(100, 101), widths=range(300, 301))
            results, paths = [], []
            outcome = run_simulation(settings, Event(), paths.append, results.append, interval_s=0)
            self.assertEqual(outcome.status, "completed")
            self.assertEqual(paths, [outcome.directory])
            self.assertEqual(outcome.saved, 1)
            measurement, = results
            summary = summary_for(outcome)
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["thresholds"], settings.thresholds)
            self.assertEqual(summary["scope_ip"], settings.scope_ip)
            self.assertEqual(summary["board_port"], settings.board_port)
            self.assertEqual(summary["measurements"][0]["capture"], measurement.capture_path.name)
            record = prepare_timed_file(measurement.capture_path, point=(100, 300))
            self.assertEqual(len(record.pulses), 3)
            self.assertEqual(measurement.pulse["number"], measurement.report["shape"]["representative_pulse"])
            with np.load(measurement.capture_path, allow_pickle=False) as data:
                saved_report = json.loads(data["evaluation_json"].item())
            self.assertEqual(saved_report["passed"], measurement.report["passed"])
            self.assertEqual(saved_report["flatness"]["raw_max_pct"],
                             measurement.report["flatness"]["raw_max_pct"])

    def test_scan_steps_through_width_before_frequency_and_stops(self):
        with TemporaryDirectory() as temporary:
            settings = settings_for(temporary)
            cancelled = Event()
            points = []

            def record_point(result):
                points.append((result.frequency_khz, result.width_ns))
                if len(points) == 9:
                    cancelled.set()

            outcome = run_simulation(settings, cancelled, lambda _: None, record_point, interval_s=0)
            self.assertEqual(settings.count, 56)
            self.assertEqual(points, [(80, width) for width in range(100, 801, 100)] + [(100, 100)])
            self.assertEqual((outcome.status, outcome.saved), ("stopped", 9))
            self.assertEqual(summary_for(outcome)["status"], "stopped")
            self.assertEqual(len(list(outcome.directory.glob("*.npz"))), 9)
            restarted = settings_for(temporary, frequencies=range(80, 81), widths=range(100, 101))
            second = run_simulation(restarted, Event(), lambda _: None, lambda _: None, interval_s=0)
            self.assertEqual((second.status, second.saved), ("completed", 1))
            self.assertNotEqual(second.directory, outcome.directory)

    def test_save_error_keeps_previous_measurements(self):
        with TemporaryDirectory() as temporary:
            def simulate(frequency, width, seed):
                if width == 200:
                    # A directory at this path makes the second capture replace fail.
                    (directory / "0002_80kHz_200ns.npz").mkdir()
                return simulate_capture(frequency, width, seed)

            def started(path):
                nonlocal directory
                directory = path

            directory = None
            results = []
            with patch("waveform_simulation.simulate_capture", side_effect=simulate):
                outcome = run_simulation(settings_for(temporary), Event(), started,
                                         results.append, interval_s=0)
            self.assertEqual((outcome.status, outcome.saved, outcome.condition), ("failed", 1, (80, 200)))
            self.assertEqual(len(results), 1)
            self.assertTrue(results[0].capture_path.is_file())
            summary = summary_for(outcome)
            self.assertEqual(summary["failed_condition"], [80, 200])
            self.assertEqual(len(summary["measurements"]), 1)


if __name__ == "__main__":
    unittest.main()
