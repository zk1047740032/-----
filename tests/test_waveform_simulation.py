from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest
from unittest.mock import patch

import numpy as np

from waveform_core.timing_pulses import prepare_timed_waveform
from waveform_simulation import (
    DEFAULT_FREQUENCY,
    DEFAULT_WIDTH,
    FileEvaluationSettings,
    SimulationSettings,
    run_evaluation,
    run_simulation,
    scan_axis,
    simulate_capture,
)


def settings_for(**overrides):
    values = dict(frequencies=scan_axis(*DEFAULT_FREQUENCY, "重频"),
                  widths=scan_axis(*DEFAULT_WIDTH, "脉宽"),
                  alignment_tolerance_pct=7.0, raw_limit_pct=20.0)
    return SimulationSettings(**(values | overrides))


class SimulationTests(unittest.TestCase):
    def test_simulation_preserves_recorded_amplitude_width_and_noise(self):
        time_s, voltage_v = simulate_capture(80, 100, seed=42)
        record = prepare_timed_waveform(time_s, voltage_v, point=(80, 100))
        self.assertEqual(len(record.pulses), 3)
        self.assertGreater(np.std(voltage_v[time_s < -70e-9]), 0.01)
        for pulse in record.pulses:
            self.assertAlmostEqual(pulse["amplitude_v"], 1.6, delta=0.06)
            self.assertAlmostEqual(pulse["width50_ns"], 95.0, delta=1.5)

    def test_simulation_adapts_unmeasured_condition(self):
        time_s, voltage_v = simulate_capture(200, 325, seed=42)
        record = prepare_timed_waveform(time_s, voltage_v, point=(200, 325))
        self.assertEqual(len(record.pulses), 3)
        cadence = record.measurement_context["observed_cadence"]
        self.assertAlmostEqual(cadence["observed_frequency_khz"], 200, delta=0.1)
        for pulse in record.pulses:
            self.assertAlmostEqual(pulse["width50_ns"], 322.0, delta=2.0)

    def test_axis_step_and_invalid_configuration(self):
        self.assertEqual(list(scan_axis(80, 125, 20, "重频")), [80, 100, 120])
        for args in ((0, 100, 20), (100, 80, 20), (80, 100, 0)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                scan_axis(*args, "重频")
        for fields in (dict(alignment_tolerance_pct=-1), dict(raw_limit_pct=0),
                       dict(frequencies=range(1, 2), widths=range(100, 101))):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                settings_for(**fields)
        wide_pulse = range(5000, 5001)
        with self.assertRaisesRegex(ValueError, "周期"):
            settings_for(frequencies=range(200, 201), widths=wide_pulse)
        valid = settings_for(frequencies=range(150, 151), widths=wide_pulse)
        self.assertEqual(valid.widths, wide_pulse)

    def test_single_evaluation_publishes_measurement_without_writing(self):
        settings = settings_for(frequencies=range(100, 101), widths=range(300, 301))
        results = []
        with patch("numpy.savez_compressed") as save_capture:
            outcome = run_simulation(settings, Event(), results.append, interval_s=0)
        save_capture.assert_not_called()
        self.assertEqual((outcome.status, outcome.evaluated), ("completed", 1))
        measurement, = results
        self.assertEqual((measurement.frequency_khz, measurement.width_ns), (100, 300))
        self.assertEqual(measurement.report["thresholds"], settings.thresholds)
        self.assertEqual(len(measurement.report["flatness"]["pulses"]), 3)
        self.assertEqual(measurement.pulse["number"], measurement.report["shape"]["representative_pulse"])

    def test_scan_steps_through_width_before_frequency_and_stops(self):
        settings = settings_for()
        first_frequency, next_frequency = settings.frequencies[:2]
        first_width = settings.widths[0]
        expected = [(first_frequency, width) for width in settings.widths]
        expected.append((next_frequency, first_width))
        stop_after = len(expected)
        cancelled = Event()
        points = []

        def record_point(result):
            points.append((result.frequency_khz, result.width_ns))
            if len(points) == stop_after:
                cancelled.set()

        outcome = run_simulation(settings, cancelled, record_point, interval_s=0)
        self.assertEqual(settings.count, len(settings.frequencies) * len(settings.widths))
        self.assertEqual(points, expected)
        self.assertEqual((outcome.status, outcome.evaluated), ("stopped", stop_after))
        restarted = settings_for(frequencies=range(first_frequency, first_frequency + 1),
                                 widths=range(first_width, first_width + 1))
        second = run_simulation(restarted, Event(), lambda _: None, interval_s=0)
        self.assertEqual((second.status, second.evaluated), ("completed", 1))

    def test_evaluation_error_keeps_previous_measurements(self):
        settings = settings_for(frequencies=range(50, 51), widths=range(100, 201, 50))
        first_capture = simulate_capture(50, 100, seed=42)
        results = []
        with patch("waveform_simulation.simulate_capture",
                   side_effect=[first_capture, ValueError("模拟采集失败")]):
            outcome = run_simulation(settings, Event(), results.append, interval_s=0)
        self.assertEqual((outcome.status, outcome.evaluated, outcome.condition), ("failed", 1, (50, 150)))
        self.assertEqual(outcome.error, "模拟采集失败")
        measurement, = results
        self.assertEqual((measurement.frequency_khz, measurement.width_ns), (50, 100))

    def test_file_evaluation_uses_file_samples_and_current_thresholds(self):
        time_s, voltage_v = simulate_capture(190, 150, 12)
        thresholds = dict(alignment_tolerance_pct=3.0, raw_limit_pct=12.5)
        with TemporaryDirectory() as temporary:
            destination = Path(temporary)
            csv = destination / "波形.csv"
            npz = destination / "波形.npz"
            np.savetxt(csv, np.column_stack((time_s, voltage_v)), delimiter=",",
                       header="time_s,voltage_v", comments="")
            np.savez_compressed(npz, time_s=time_s, voltage_v=voltage_v)
            for source in (csv, npz):
                with self.subTest(source=source.name):
                    original = source.read_bytes()
                    results = []
                    settings = FileEvaluationSettings(source, thresholds)
                    outcome = run_evaluation(settings, Event(), results.append)
                    self.assertEqual((outcome.status, outcome.evaluated), ("completed", 1))
                    measurement, = results
                    self.assertEqual(measurement.source_file, source)
                    self.assertIsNone(measurement.frequency_khz)
                    self.assertEqual(measurement.report["thresholds"], thresholds)
                    self.assertEqual(len(measurement.report["flatness"]["pulses"]), 3)
                    self.assertEqual(source.read_bytes(), original)
                    self.assertEqual(set(destination.iterdir()), {csv, npz})

    def test_malformed_file_has_readable_error(self):
        with TemporaryDirectory() as temporary:
            destination = Path(temporary)
            source = destination / "invalid.csv"
            source.write_text("wrong,header\n1,2\n", encoding="utf-8")
            settings = FileEvaluationSettings(source,
                                              dict(alignment_tolerance_pct=7.0, raw_limit_pct=20.0))
            results = []
            outcome = run_evaluation(settings, Event(), results.append)
            self.assertEqual((outcome.status, outcome.evaluated), ("failed", 0))
            self.assertIn("CSV表头", outcome.error)
            self.assertFalse(results)
            self.assertEqual(list(destination.iterdir()), [source])


if __name__ == "__main__":
    unittest.main()
