"""Read synchronized Tektronix waveform frames over the SCPI socket server."""

from dataclasses import dataclass, replace
from datetime import datetime
from functools import lru_cache
from math import isfinite

import numpy as np

from waveform_core.pulse_metrics import locate_pulses


SCOPE_PORT = 4000
SCOPE_CHANNEL = "CH2"
SCOPE_TIMEOUT_S = 3.0
MAX_SCOPE_POINTS = 1_000_000
MAX_TEXT_BYTES = 512
MAX_PREAMBLE_BYTES = 4096
FOCUS_SAMPLES_PER_SIDE = 16_384


@dataclass(frozen=True)
class ScopeFrame:
    time_s: np.ndarray
    voltage_v: np.ndarray
    sample_interval_s: float
    received_at: datetime
    horizontal_scale_s: float | None = None
    pulse_view_ns: tuple[float, float] | None = None


@lru_cache(maxsize=2)
def _sample_times(points, interval, origin):
    time_s = np.arange(points, dtype=float)
    time_s *= interval
    time_s += origin
    if not isfinite(time_s[-1]):
        raise ValueError("WAVFRM? 的时间换算结果无效。")
    time_s.flags.writeable = False
    return time_s


def _single_pulse_view_ns(frame, scale_s):
    t = frame.time_s
    if t.size > 4 * FOCUS_SAMPLES_PER_SIDE and t[0] <= 0 <= t[-1]:
        trigger_index = int(t.searchsorted(0.))
        start = max(0, trigger_index - FOCUS_SAMPLES_PER_SIDE)
        stop = min(t.size, trigger_index + FOCUS_SAMPLES_PER_SIDE)
        view = _pulse_view_ns(frame, scale_s, start, stop)
        if view is not None:
            return view
    return _pulse_view_ns(frame, scale_s, 0, t.size)


def _pulse_view_ns(frame, scale_s, first, stop):
    t = frame.time_s[first:stop]
    voltage = frame.voltage_v[first:stop]
    pulses, diagnostics = locate_pulses(t, voltage, frame.sample_interval_s)
    if not pulses:
        return None

    def distance_to_trigger(index):
        start, stop = pulses[index]
        return max(t[start], -t[stop], 0)

    index = min(range(len(pulses)), key=distance_to_trigger)
    start, stop = pulses[index]
    rise, fall = t[start], t[stop]
    width = fall - rise
    # Use ten scope divisions if they give a view that includes the whole pulse
    # and its baseline without making the pulse too small to inspect.
    span = min(4 * width, max(1.4 * width, 10 * scale_s))
    left, right = frame.time_s[0], frame.time_s[-1]
    if index > 0:
        left = (t[pulses[index - 1][1]] + rise) / 2
    if index + 1 < len(pulses):
        right = (fall + t[pulses[index + 1][0]]) / 2
    span = min(span, right - left)
    lo = max(left, min((rise + fall - span) / 2, right - span))
    # A cropped edge may hide a partial pulse or a neighbor that limits the view.
    threshold = diagnostics["baseline_v"] + .5 * diagnostics["bootstrap_amplitude_v"]
    if first and (voltage[0] >= threshold or (index == 0 and lo < (t[0] + rise) / 2)):
        return None
    if stop < frame.time_s.size and (voltage[-1] >= threshold
                                     or (index == len(pulses) - 1 and lo + span > (fall + t[-1]) / 2)):
        return None
    return float(lo * 1e9), float((lo + span) * 1e9)


def read_waveform(stream):
    preamble = bytearray()
    while len(preamble) < MAX_PREAMBLE_BYTES:
        char = stream.read(1)
        if char == b"#":
            break
        if char in (b"", b"\r", b"\n"):
            raise ValueError("WAVFRM? 未返回完整的波形前导和二进制块。")
        preamble.extend(char)
    else:
        raise ValueError("WAVFRM? 波形前导过长。")

    header, _, curve = preamble.decode("ascii").rpartition(";")
    if curve.strip().upper().lstrip(":") != "CURVE":
        raise ValueError("WAVFRM? 应答缺少 CURVE 数据段。")
    fields = {}
    for item in header.split(";"):
        name, _, value = item.strip().partition(" ")
        fields[name.rsplit(":", 1)[-1].upper()] = value.strip().strip('"\'')

    try:
        bytes_per_point = int(fields["BYT_NR"])
        format_fields = tuple(fields[key].upper() for key in ("BN_FMT", "BYT_OR", "PT_FMT", "XUNIT", "YUNIT"))
        points = int(fields["NR_PT"])
        point_offset = int(fields.get("PT_OFF", "0"))
        interval, origin, gain, offset, zero = (
            float(fields[key]) for key in ("XINCR", "XZERO", "YMULT", "YOFF", "YZERO"))
    except (KeyError, ValueError) as error:
        raise ValueError("WAVFRM? 缺少有效的波形点数、单位或换算参数。") from error
    if bytes_per_point != 2 or format_fields != ("RI", "MSB", "Y", "S", "V") or point_offset != 0:
        raise ValueError("需要以秒和伏为单位的 CH2 普通波形，请检查通道和采集模式。")
    if not 20 <= points <= MAX_SCOPE_POINTS:
        raise ValueError(f"波形点数超出实时读取范围（20 至 {MAX_SCOPE_POINTS:,} 点）。")
    if interval <= 0 or gain == 0 or not all(map(isfinite, (interval, origin, gain, offset, zero))):
        raise ValueError("WAVFRM? 的时间或电压换算参数无效。")

    width_byte = stream.read(1)
    if not width_byte.isdigit() or width_byte == b"0":
        raise ValueError("WAVFRM? 二进制块长度字段无效。")
    width = int(width_byte)
    digits = stream.read(width)
    size = points * 2
    if len(digits) != width or not digits.isdigit() or int(digits) != size:
        raise ValueError("WAVFRM? 二进制块长度与波形点数不一致。")
    payload = stream.read(size)
    terminator = stream.read(1) if len(payload) == size else b""
    if terminator == b"\r":
        terminator = stream.read(1)
    if terminator != b"\n":
        raise ValueError("WAVFRM? 二进制波形未完整接收。")

    time_s = _sample_times(points, interval, origin)
    voltage_v = np.frombuffer(payload, dtype=">i2").astype(float)
    voltage_v -= offset
    voltage_v *= gain
    voltage_v += zero
    if not np.isfinite(voltage_v).all():
        raise ValueError("WAVFRM? 的电压换算结果无效。")
    voltage_v.flags.writeable = False
    return ScopeFrame(time_s, voltage_v, interval, datetime.now().astimezone())


class ScopeConnection:
    def __init__(self, connection, stream):
        self.connection = connection
        self.stream = stream
        self.stop_point = None

    def send(self, command):
        self.connection.sendall((command + "\n").encode("ascii"))

    def query(self, command):
        self.send(command)
        try:
            raw = self.stream.readline(MAX_TEXT_BYTES + 1)
        except TimeoutError as error:
            raise TimeoutError(f"查询 {command} 超时。") from error
        if not raw:
            raise ConnectionError("示波器已断开连接。")
        if len(raw) > MAX_TEXT_BYTES or not raw.endswith(b"\n"):
            raise ValueError(f"{command} 应答过长或缺少换行。")
        return raw.decode("ascii", errors="replace").strip()

    def identify(self):
        identity = self.query("*IDN?")
        fields = identity.split(",", 3)
        if (not identity.isascii() or not identity.isprintable()
                or len(fields) != 4 or not all(part.strip() for part in fields)):
            raise ValueError("收到的数据不是有效的 *IDN? 身份应答。请检查 Socket Server 的 Protocol 设置。")
        return identity

    def configure_waveform(self):
        self.send("HEADER ON;:VERBOSE ON")
        selected = self.query(f"SELECT:{SCOPE_CHANNEL}?").rpartition(" ")[-1].upper()
        if selected not in ("ON", "1"):
            raise ValueError(f"请在示波器上开启 {SCOPE_CHANNEL} 并运行采集。")
        self.send(f"DATA:SOURCE {SCOPE_CHANNEL};:DATA:ENCDG RIBINARY;:DATA:WIDTH 2;:DATA:START 1")

    def read_frame(self):
        # Record length may change on the front panel while a frame is requested.
        for _ in range(2):
            length = self.query("HORIZONTAL:RECORDLENGTH?").split()
            try:
                points = int(length[-1])
            except (IndexError, ValueError) as error:
                raise ValueError("示波器返回的记录长度无效。") from error
            if not 20 <= points <= MAX_SCOPE_POINTS:
                raise ValueError(f"记录长度为 {points:,} 点。请设为 20 至 {MAX_SCOPE_POINTS:,} 点后重新连接。")
            if points != self.stop_point:
                self.send(f"DATA:STOP {points}")
                self.stop_point = points
            self.send("WAVFRM?")
            try:
                frame = read_waveform(self.stream)
            except TimeoutError as error:
                raise TimeoutError(f"读取波形超时，请确认 {SCOPE_CHANNEL} 已开启并采集到波形。") from error
            if frame.time_s.size == points:
                scale = self.query("HORIZONTAL:SCALE?").split()
                try:
                    scale_s = float(scale[-1])
                except (IndexError, ValueError) as error:
                    raise ValueError("示波器返回的水平标度无效。") from error
                if scale_s <= 0 or not isfinite(scale_s):
                    raise ValueError("示波器返回的水平标度无效。")
                return replace(frame, horizontal_scale_s=scale_s,
                               pulse_view_ns=_single_pulse_view_ns(frame, scale_s))
            self.stop_point = None
        raise ValueError("传回的波形点数与当前记录长度不一致，请重新连接示波器。")
