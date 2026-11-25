import unittest
from unittest.mock import call, patch

from pulse_board_port import PulseBoardPort, PulseBoardState


# Synthetic register values for testing frame layout and checksums.
STATE = PulseBoardState(
    b"\x01\x23", b"\x04\x56", 1, b"\x89\xAB", b"\r\n", b"\xFF\x00", b"\x12\x34",
)
STATUS_REPLY = bytes.fromhex("57 00 00 32 0D 01 23 04 56 01 89 AB 0D 0A FF 00 12 34 4E B2 0D 0A")


class PulseBoardPortTests(unittest.TestCase):
    def setUp(self):
        self.serial_factory = self.enterContext(patch("pulse_board_port.Serial"))
        self.serial = self.serial_factory.return_value
        self.serial.write.side_effect = len
        self.serial.read.return_value = STATUS_REPLY
        self.board = PulseBoardPort("COM42")
        self.addCleanup(self.board.close)

    def set_timing(self):
        return self.board.set_timing_raw(STATE.frequency, STATE.pulse_width, trigger=1, save_flag=0xA5)

    def set_shape(self, index_d=STATE.index_d):
        return self.board.set_shape_raw(STATE.y0, STATE.y1, STATE.y2, index_d)

    def test_open_configures_115200_8n1_without_sending_a_command(self):
        self.serial_factory.assert_called_once_with(
            port=None, baudrate=115200, bytesize=8, parity="N", stopbits=1,
            timeout=1.0, write_timeout=1.0, xonxoff=False, rtscts=False, dsrdtr=False,
        )
        self.assertEqual(self.serial.port, "COM42")
        self.assertFalse(self.serial.dtr)
        self.assertFalse(self.serial.rts)
        self.assertEqual(self.serial.mock_calls, [call.open()])

    def test_settings_send_documented_frames_and_return_raw_status(self):
        timing_request = bytes.fromhex("57 00 00 22 06 01 23 04 56 01 A5 4C F0 0D 0A")
        shape_request = bytes.fromhex("57 00 00 42 08 89 AB 0D 0A FF 00 12 34 DA B6 0D 0A")
        self.assertEqual(self.set_timing(), STATE)
        self.assertEqual(self.set_shape(), STATE)
        self.assertEqual(self.serial.mock_calls, [
            call.open(),
            call.reset_input_buffer(), call.write(timing_request), call.read(22),
            call.reset_input_buffer(), call.write(shape_request), call.read(22),
        ])

    def test_bad_frame_is_rejected(self):
        for offset in (0, 1, 2, 3, 4, -4, -3, -2, -1):
            with self.subTest(offset=offset):
                reply = bytearray(STATUS_REPLY)
                reply[offset] ^= 1
                self.serial.read.return_value = bytes(reply)
                with self.assertRaisesRegex(OSError, "校验错误"):
                    self.set_timing()

    def test_valid_reply_with_different_readback_is_rejected(self):
        with self.assertRaisesRegex(OSError, "回读与下发值不一致"):
            self.set_shape(index_d=b"\x12\x35")
        self.serial.write.assert_called_once()

    def test_partial_reply_times_out_without_reissuing_the_setting(self):
        self.serial.read.return_value = STATUS_REPLY[:8]
        with self.assertRaisesRegex(TimeoutError, "8/22"):
            self.set_timing()
        self.serial.write.assert_called_once()

    def test_short_write_never_reads_an_acknowledgement(self):
        self.serial.write.side_effect = None
        self.serial.write.return_value = 3
        with self.assertRaisesRegex(OSError, "未完整写入"):
            self.set_shape()
        self.serial.read.assert_not_called()

    def test_invalid_register_length_is_rejected_before_writing(self):
        with self.assertRaisesRegex(ValueError, "双字节寄存器"):
            self.set_shape(index_d=b"\x12")
        self.serial.write.assert_not_called()
