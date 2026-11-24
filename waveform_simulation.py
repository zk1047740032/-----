"""Generate, evaluate, and save simulated captures for the desktop frontend."""

from dataclasses import dataclass
from datetime import datetime
from ipaddress import ip_address
from itertools import product
import json
from math import ceil, isfinite
from pathlib import Path
from tempfile import NamedTemporaryFile, mkdtemp

import numpy as np

from fiber1100_algorithm import evaluate_shape_record
from waveform_core.timing_pulses import prepare_timed_waveform


DEFAULT_FREQUENCY = (80, 200, 20)
DEFAULT_WIDTH = (100, 800, 100)
DEFAULT_RAW_LIMIT = 20.0
MAX_CAPTURE_SAMPLES = 1_000_000


def scan_axis(start, end, step, label):
    if start <= 0 or end < start or step <= 0:
        raise ValueError(f"{label}须满足正数起始值、终止值不小于起始值且步进大于 0。")
    return range(start, end + 1, step)


def capture_grid(frequency_khz, width_ns):
    period_ns = 1_000_000 / frequency_khz
    interval_ns = min(0.2, width_ns / 500)
    return period_ns, interval_ns, ceil((2 * period_ns + 3 * width_ns) / interval_ns)


@dataclass(frozen=True)
class SimulationSettings:
    frequencies: range
    widths: range
    destination: Path
    scope_ip: str
    board_port: str
    alignment_tolerance_pct: float
    raw_limit_pct: float

    def __post_init__(self):
        if self.scope_ip:
            try:
                ip_address(self.scope_ip)
            except ValueError as error:
                raise ValueError("请输入有效的示波器 IP 地址。") from error
        if not isfinite(self.alignment_tolerance_pct) or self.alignment_tolerance_pct < 0:
            raise ValueError("红蓝高度允许偏差须为非负的有限百分比。")
        if not isfinite(self.raw_limit_pct) or self.raw_limit_pct <= 0:
            raise ValueError("不平度上限须为大于 0 的有限百分比。")
        if not self.frequencies or not self.widths:
            raise ValueError("至少需要一个工况。")
        if self.widths[-1] >= 1_000_000 / self.frequencies[-1]:
            raise ValueError("最大脉宽须小于最高重频对应的周期。")
        # The sample count can peak at either width endpoint as the interval
        # scales with narrow pulses and stays constant for wider pulses.
        for width in (self.widths[0], self.widths[-1]):
            if capture_grid(self.frequencies[0], width)[2] > MAX_CAPTURE_SAMPLES:
                raise ValueError("当前重频和脉宽需要过多的模拟样点，请调整测试范围。")

    @property
    def count(self):
        return len(self.frequencies) * len(self.widths)

    @property
    def thresholds(self):
        return dict(alignment_tolerance_pct=self.alignment_tolerance_pct,
                    raw_limit_pct=self.raw_limit_pct)


@dataclass(frozen=True)
class Measurement:
    index: int
    frequency_khz: int
    width_ns: int
    capture_path: Path
    report: dict
    pulse: dict

    @property
    def verdict(self):
        if self.report["passed"]:
            return "达标"
        if not self.report["shape"]["geometry_valid"]:
            return "待复核"
        return "未达标"


@dataclass(frozen=True)
class SimulationOutcome:
    status: str
    directory: Path | None
    saved: int
    error: str = ""
    condition: tuple[int, int] | None = None


def simulate_capture(frequency_khz, width_ns, seed):
    period_ns, interval_ns, count = capture_grid(frequency_khz, width_ns)
    t = -width_ns + np.arange(count) * interval_ns
    rng = np.random.default_rng(seed)
    voltage = rng.normal(0, 0.0002, count)
    ramp_ns = min(7.0, 0.07 * width_ns)
    dip = (0.0, -0.10, -0.24)[(frequency_khz // 20 + width_ns // 100 + 1) % 3]
    for cycle in range(3):
        local_time = t - cycle * period_ns
        profile = np.clip(np.minimum(local_time, width_ns - local_time) / ramp_ns, 0, 1)
        distortion = dip * np.exp(-0.5 * ((local_time / width_ns - 0.51) / 0.087) ** 2)
        voltage += profile * (1 + distortion)
    return t * 1e-9, voltage


def atomic_write(path, write):
    temporary = None
    try:
        with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            write(stream)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_summary(directory, summary):
    text = json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2)
    atomic_write(directory / "run.json", lambda stream: stream.write(text.encode("utf-8")))


def run_simulation(settings, cancelled, on_started, on_measurement, *, interval_s=0.20):
    """Run acquisition and I/O in the calling thread, publishing only saved captures."""
    directory = condition = None
    entries = []
    status = "running"
    error = ""
    now = datetime.now().astimezone()
    summary = dict(mode="simulation", status=status, started_at=now.isoformat(),
                   scope_ip=settings.scope_ip, board_port=settings.board_port,
                   thresholds=settings.thresholds, total=settings.count, measurements=entries)
    for name, axis in (("frequency_khz", settings.frequencies), ("width_ns", settings.widths)):
        summary[name] = dict(start=axis.start, end=axis.stop - 1, step=axis.step)
    try:
        settings.destination.mkdir(parents=True, exist_ok=True)
        directory = Path(mkdtemp(dir=settings.destination, prefix=now.strftime("%Y%m%d_%H%M%S_")))
        write_summary(directory, summary)
        on_started(directory)
        rng = np.random.default_rng()
        for index, (frequency, width) in enumerate(product(settings.frequencies, settings.widths), 1):
            if cancelled.wait(interval_s if index > 1 else 0):
                status = "stopped"
                break
            condition = (frequency, width)
            time_s, voltage_v = simulate_capture(frequency, width, int(rng.integers(2**32)))
            record = prepare_timed_waveform(time_s, voltage_v, point=condition)
            report = evaluate_shape_record(record, **settings.thresholds)
            if cancelled.is_set():
                status = "stopped"
                break
            pulse = next(pulse for pulse in record.pulses
                         if pulse["number"] == report["shape"]["representative_pulse"])
            path = directory / f"{index:04d}_{frequency}kHz_{width}ns.npz"
            report_json = json.dumps(report, ensure_ascii=False, allow_nan=False)
            atomic_write(path, lambda stream: np.savez_compressed(
                stream, time_s=time_s, voltage_v=voltage_v, evaluation_json=report_json))
            measurement = Measurement(index, frequency, width, path, report, pulse)
            entries.append(dict(frequency_khz=frequency, width_ns=width, capture=path.name,
                                verdict=measurement.verdict, passed=report["passed"],
                                alignment_pct=report["shape"]["score_pct"],
                                raw_flatness_pct=report["flatness"]["raw_max_pct"]))
            write_summary(directory, summary)
            on_measurement(measurement)
            condition = None
        else:
            status = "completed"
    except Exception as exc:
        status, error = "failed", str(exc) or type(exc).__name__

    if directory is not None:
        summary.update(status=status, finished_at=datetime.now().astimezone().isoformat())
        if error:
            summary.update(error=error, failed_condition=condition)
        try:
            write_summary(directory, summary)
        except OSError as exc:
            detail = f"运行汇总保存失败。{exc}"
            status, error = "failed", f"{error}\n{detail}" if error else detail
    return SimulationOutcome(status, directory, len(entries), error, condition if error else None)
