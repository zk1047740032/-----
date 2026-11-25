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


class ScopeCaptureTests(unittest.TestCase):
    def test_signed_binary_frame_uses_its_own_scaling_and_consumes_terminator(self):
        raw = np.array([0x0D0A, -1, -32768, 10, 13] * 4, dtype=">i2")
        response = BytesIO(PREAMBLE + b"#240" + raw.tobytes() + b"\r\nnext response\n")
        frame = read_waveform(response)
        np.testing.assert_allclose(frame.time_s, -8e-9 + np.arange(20) * 2e-9)
        np.testing.assert_allclose(frame.voltage_v, (raw.astype(float) - 10) * 0.002 + 0.13)
        self.assertFalse(frame.voltage_v.flags.writeable)
        self.assertEqual(response.readline(), b"next response\n")

    def test_horizontal_scale_is_read_for_each_frame_independently_of_sample_spacing(self):
        capture = b":HORIZONTAL:RECORDLENGTH 20\n" + PREAMBLE + b"#240" + bytes(40) + b"\n"
        stream = BytesIO(capture + b":HORIZONTAL:SCALE 4.0000E-8\n"
                         + capture + b"8.0000E-8\n" + capture + b"0\n")
        socket = Mock()
        scope = ScopeConnection(socket, stream)
        first = scope.read_frame()
        second = scope.read_frame()
        self.assertEqual((first.horizontal_scale_s, second.horizontal_scale_s), (40e-9, 80e-9))
        np.testing.assert_array_equal(first.time_s, second.time_s)
        self.assertEqual(second.sample_interval_s, 2e-9)
        with self.assertRaisesRegex(ValueError, "水平标度无效"):
            scope.read_frame()
        commands = [call.args[0] for call in socket.sendall.call_args_list]
        self.assertEqual(commands.count(b"HORIZONTAL:SCALE?\n"), 3)

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
