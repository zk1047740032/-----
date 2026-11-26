from datetime import datetime, timezone
from io import BytesIO
from threading import Event
import unittest
from unittest.mock import Mock

import numpy as np

from fiber1100_algorithm import evaluate_waveform
from scope_capture import ScopeConnection, ScopeFrame, read_waveform
from waveform_simulation import ScopeEvaluationSettings, run_evaluation, simulate_capture


PREAMBLE = (b':WFMOUTPRE:BYT_NR 2;BIT_NR 16;ENCDG BINARY;BN_FMT RI;BYT_OR MSB;'
            b'WFID "Ch2, DC coupling, 20 points, Sample mode";NR_PT 20;PT_FMT Y;XUNIT "s";'
            b'XINCR 2E-9;XZERO -8E-9;PT_OFF 0;YUNIT "V";YMULT 0.002;YOFF 10;YZERO 0.13;:CURVE ')
DISPLAY_QUERY = b"HORIZONTAL:SCALE?;:CH2:SCALE?;POSITION?;OFFSET?\n"


class ScopeCaptureTests(unittest.TestCase):
    def test_signed_binary_frame_uses_its_own_scaling_and_consumes_terminator(self):
        raw = np.array([0x0D0A, -1, -32768, 10, 13] * 4, dtype=">i2")
        response = BytesIO(PREAMBLE + b"#240" + raw.tobytes() + b"\r\nnext response\n")
        frame = read_waveform(response)
        np.testing.assert_allclose(frame.time_s, -8e-9 + np.arange(20) * 2e-9)
        np.testing.assert_allclose(frame.voltage_v, (raw.astype(float) - 10) * 0.002 + 0.13)
        self.assertFalse(frame.time_s.flags.writeable)
        self.assertFalse(frame.voltage_v.flags.writeable)
        self.assertEqual(response.readline(), b"next response\n")

    def test_horizontal_scale_is_read_for_each_frame_independently_of_sample_spacing(self):
        capture = b":HORIZONTAL:RECORDLENGTH 20\n" + PREAMBLE + b"#240" + bytes(40) + b"\n"
        shifted = capture.replace(b"XINCR 2E-9;XZERO -8E-9", b"XINCR 4E-9;XZERO -16E-9")
        stream = BytesIO(capture + b":HORIZONTAL:SCALE 4.0000E-8;:CH2:SCALE 0.5;:CH2:POSITION 0;:CH2:OFFSET 0\n"
                         + capture + b"8.0000E-8;0.5;0;0\n" + shifted + b"8.0000E-8;0.5;0;0\n"
                         + capture + b"0;0.5;0;0\n")
        socket = Mock()
        scope = ScopeConnection(socket, stream)
        first = scope.read_frame()
        second = scope.read_frame()
        third = scope.read_frame()
        self.assertEqual((first.horizontal_scale_s, second.horizontal_scale_s), (40e-9, 80e-9))
        self.assertIs(first.time_s, second.time_s)
        np.testing.assert_allclose(third.time_s, -16e-9 + np.arange(20) * 4e-9)
        self.assertEqual(second.sample_interval_s, 2e-9)
        self.assertIsNone(first.pulse_view_ns)
        with self.assertRaisesRegex(ValueError, "水平标度无效"):
            scope.read_frame()
        commands = [call.args[0] for call in socket.sendall.call_args_list]
        self.assertEqual(commands.count(DISPLAY_QUERY), 4)

    def test_vertical_display_range_follows_channel_settings(self):
        capture = b"20\n" + PREAMBLE + b"#240" + bytes(40) + b"\n"
        stream = BytesIO(capture + b"40e-9;0.5;-2;0.125\n" + capture + b"40e-9;0.25;1.5;-0.125\n"
                         + capture + b"40e-9;0.25;1.5;0.125\n" + capture + b"40e-9;0.25;NaN;0.125\n")
        scope = ScopeConnection(Mock(), stream)
        first, second, third = (scope.read_frame() for _ in range(3))

        self.assertEqual(first.vertical_scale_v, 0.5)
        self.assertEqual(first.voltage_range_v, (-0.875, 3.125))
        self.assertEqual(second.vertical_scale_v, 0.25)
        self.assertEqual(second.voltage_range_v, (-1.5, 0.5))
        self.assertEqual(third.voltage_range_v, (-1.25, 0.75))
        np.testing.assert_array_equal(first.voltage_v, third.voltage_v)
        with self.assertRaisesRegex(ValueError, "垂直标度、位置或偏置无效"):
            scope.read_frame()

    def test_long_record_keeps_all_samples_and_focuses_the_pulse_nearest_the_trigger(self):
        points = 90_001
        time_ns = -36_000 + np.arange(points) * 0.8
        preamble = (PREAMBLE.replace(b"NR_PT 20", f"NR_PT {points}".encode())
                    .replace(b"XINCR 2E-9", b"XINCR 8E-10").replace(b"XZERO -8E-9", b"XZERO -36E-6"))
        cases = (
            (150, 40, 400, 20, 20_000),
            (800, 40, 1120, 20, 20_000),
            (150, 5000, 600, 20, 20_000),
            (150, 40, 400, 20_000, 60_000),  # The trigger neighborhood contains no pulse.
        )
        for width_ns, scale_ns, view_ns, rise_ns, period_ns in cases:
            with self.subTest(width_ns=width_ns, scale_ns=scale_ns, rise_ns=rise_ns):
                high = (time_ns - rise_ns) % period_ns < width_ns
                payload = np.where(high, 200, 0).astype(">i2").tobytes()
                capture = (f"{points}\n".encode() + preamble + f"#6{len(payload)}".encode()
                           + payload + f"\n{scale_ns}e-9;0.5;0;0\n".encode())
                frame = ScopeConnection(Mock(), BytesIO(capture)).read_frame()

                self.assertEqual(frame.time_s.size, points)
                np.testing.assert_allclose(frame.time_s * 1e9, time_ns, atol=1e-8)
                lo, hi = frame.pulse_view_ns
                self.assertLess(lo, rise_ns)
                self.assertGreater(hi, rise_ns + width_ns)
                self.assertGreater(lo, rise_ns - period_ns + width_ns)
                self.assertLess(hi, rise_ns + period_ns)
                self.assertAlmostEqual(hi - lo, view_ns, delta=2)

    def test_incomplete_or_unsupported_frame_is_rejected(self):
        payload = bytes(40)
        bad_responses = (
            PREAMBLE + b"#239" + payload + b"\n",
            PREAMBLE + b"#240" + payload[:35],
            PREAMBLE + b"#240" + payload,
            PREAMBLE.replace(b"PT_FMT Y", b"PT_FMT ENV") + b"#240" + payload + b"\n",
            PREAMBLE.replace(b"XINCR 2E-9", b"XINCR NaN") + b"#240" + payload + b"\n",
        )
        for response in bad_responses:
            with self.subTest(response=response), self.assertRaises(ValueError):
                read_waveform(BytesIO(response))

    def test_single_evaluation_uses_the_entire_scope_frame(self):
        time_s, voltage_v = simulate_capture(100, 150, seed=42)
        trigger_offset_ns = 37.
        time_s += trigger_offset_ns * 1e-9
        time_s.flags.writeable = voltage_v.flags.writeable = False
        frame = ScopeFrame(time_s, voltage_v, 0.2e-9, datetime(2026, 11, 25, tzinfo=timezone.utc))
        thresholds = dict(alignment_tolerance_pct=4.0, raw_limit_pct=13.0)
        settings = ScopeEvaluationSettings(frame, thresholds)
        measurements = []
        outcome = run_evaluation(settings, Event(), measurements.append)
        self.assertEqual((outcome.status, outcome.evaluated), ("completed", 1), outcome.error)
        measurement, = measurements
        expected = evaluate_waveform(time_s, voltage_v, **thresholds)
        self.assertEqual(measurement.report["thresholds"], thresholds)
        self.assertEqual(len(measurement.report["flatness"]["pulses"]), 3)
        self.assertEqual(measurement.report["passed"], expected["passed"])
        self.assertAlmostEqual(measurement.report["shape"]["score_pct"], expected["shape"]["score_pct"])
        self.assertAlmostEqual(measurement.report["flatness"]["raw_max_pct"], expected["flatness"]["raw_max_pct"])
        rise50_ns, width50_ns = np.asarray(measurement.scope_pulse_geometry).T
        np.testing.assert_allclose(rise50_ns, trigger_offset_ns + np.arange(3) * 10_000, atol=.2, rtol=0)
        np.testing.assert_allclose(width50_ns, 150, atol=8, rtol=0)
