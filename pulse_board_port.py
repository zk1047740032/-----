"""Serial transport for the pulse board's two documented write commands."""

from dataclasses import dataclass
import re
from struct import Struct
from threading import Lock

from serial import EIGHTBITS, PARITY_NONE, STOPBITS_ONE, Serial


BOARD_BAUDRATE = 115_200
SHAPE_INDEX_D = 120
SHAPE_REGISTER_MAX = 0xFFFF
FRAME_PREFIX = b"\x57\x00\x00"
FRAME_END = b"\r\n"
STATUS_COMMAND = 0x32
STATUS_FIELDS = Struct("2s2sB2s2s2s2s")


@dataclass(frozen=True)
class PulseBoardState:
    """Values returned by the board, with two-byte fields kept in wire order."""

    frequency: bytes
    pulse_width: bytes
    trigger: int
    y0: bytes
    y1: bytes
    y2: bytes
    index_d: bytes


def _frame(command: int, parameters: bytes) -> bytes:
    body = FRAME_PREFIX[1:] + bytes((command, len(parameters))) + parameters
    xor = 0
    for byte in body:
        xor ^= byte
    return FRAME_PREFIX[:1] + body + bytes((sum(body) & 0xFF, xor)) + FRAME_END


def _register_bytes(*registers: bytes) -> bytes:
    if any(not isinstance(value, bytes) or len(value) != 2 for value in registers):
        raise ValueError("双字节寄存器必须提供两个按协议顺序排列的原始字节。")
    return b"".join(registers)


class PulseBoardPort:
    def __init__(self, port: str):
        if not re.fullmatch(r"COM[1-9][0-9]*", port, re.IGNORECASE):
            raise ValueError("请选择有效的 COM 串口。")
        self.port = port
        self._lock = Lock()
        self._serial = Serial(
            port=None, baudrate=BOARD_BAUDRATE, bytesize=EIGHTBITS, parity=PARITY_NONE,
            stopbits=STOPBITS_ONE, timeout=1.0, write_timeout=1.0,
            xonxoff=False, rtscts=False, dsrdtr=False,
        )
        self._serial.dtr = False
        self._serial.rts = False
        self._serial.port = port
        self._serial.open()

    def close(self) -> None:
        with self._lock:
            self._serial.close()

    def set_timing_raw(self, frequency: bytes, pulse_width: bytes, *,
                       trigger: int, save_flag: int) -> PulseBoardState:
        if trigger not in (0, 1):
            raise ValueError("调制方式须为 0（内调制）或 1（外调制）。")
        timing = _register_bytes(frequency, pulse_width) + bytes((trigger,))
        return self._set_parameters(0x22, timing + bytes((save_flag,)), slice(0, 5), timing)

    def set_shape_raw(self, y0: bytes, y1: bytes, y2: bytes, index_d: bytes) -> PulseBoardState:
        shape = _register_bytes(y0, y1, y2, index_d)
        return self._set_parameters(0x42, shape, slice(5, 13), shape)

    def set_shape(self, y0: int, y1: int, y2: int) -> PulseBoardState:
        values = (y0, y1, y2, SHAPE_INDEX_D)
        if any(not isinstance(value, int) or not 0 <= value <= SHAPE_REGISTER_MAX for value in values):
            raise ValueError(f"y0、y1、y2 须为 0–{SHAPE_REGISTER_MAX} 的整数。")
        # The 1100 protocol example uses high-byte-first values. Verify this on the pulse board.
        return self.set_shape_raw(*(value.to_bytes(2, "big") for value in values))

    def _set_parameters(self, command: int, parameters: bytes, readback: slice,
                        expected: bytes) -> PulseBoardState:
        request = _frame(command, parameters)
        reply_size = STATUS_FIELDS.size + 9
        with self._lock:
            self._serial.reset_input_buffer()
            if self._serial.write(request) != len(request):
                raise OSError(f"脉冲板 0x{command:02X} 指令未完整写入，设置结果未知。")
            reply = self._serial.read(reply_size)
        if len(reply) != reply_size:
            raise TimeoutError(f"脉冲板 0x{command:02X} 应答超时（收到 {len(reply)}/{reply_size} 字节），设置结果未知。")
        status = reply[5:-4]
        if reply != _frame(STATUS_COMMAND, status):
            raise OSError(f"脉冲板 0x{command:02X} 应答帧格式或校验错误，设置结果未知。")
        actual = status[readback]
        if actual != expected:
            raise OSError(f"脉冲板 0x{command:02X} 回读与下发值不一致"
                          f"（发送 {expected.hex(' ')}，回读 {actual.hex(' ')}）。")
        return PulseBoardState(*STATUS_FIELDS.unpack(status))
