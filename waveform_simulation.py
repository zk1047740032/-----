"""Evaluate and save desktop waveforms from simulation, files or the scope."""

from dataclasses import dataclass
from datetime import datetime
from functools import cache
from ipaddress import ip_address
from itertools import product
import json
from math import ceil, isfinite
from pathlib import Path
from tempfile import NamedTemporaryFile, mkdtemp

import numpy as np

from fiber1100_algorithm import evaluate_shape_record
from scope_capture import SCOPE_CHANNEL, ScopeFrame
from waveform_core.timing_pulses import prepare_timed_waveform
from waveform_core.waveform_io import load_waveform


DEFAULT_FREQUENCY = (50, 150, 10)
DEFAULT_WIDTH = (100, 800, 50)
DEFAULT_RAW_LIMIT = 20.0
MAX_CAPTURE_SAMPLES = 1_000_000
TEMPLATE_INTERVAL_NS = 0.2
TEMPLATE_PATH = Path(__file__).with_name("waveform_core") / "data" / "simulation_pulses.npz"


def scan_axis(start, end, step, label):
    if start <= 0 or end < start or step <= 0:
        raise ValueError(f"{label}须满足正数起始值、终止值不小于起始值且步进大于 0。")
    return range(start, end + 1, step)


def capture_grid(frequency_khz, width_ns):
    period_ns = 1_000_000 / frequency_khz
    interval_ns = min(TEMPLATE_INTERVAL_NS, width_ns / 500)
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
class FileEvaluationSettings:
    source_file: Path
    destination: Path
    thresholds: dict[str, float]

    count = 1


@dataclass(frozen=True)
class ScopeEvaluationSettings:
    frame: ScopeFrame
    destination: Path
    thresholds: dict[str, float]
    scope_ip: str
    scope_identity: str

    count = 1


@dataclass(frozen=True)
class Measurement:
    index: int
    frequency_khz: int | None
    width_ns: int | None
    capture_path: Path
    report: dict
    pulse: dict
    source_file: Path | None = None

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


@cache
def _measured_pulses():
    """Load the bundled raw pulse excerpts, grouped by source condition."""
    pulses = {}
    with np.load(TEMPLATE_PATH, allow_pickle=False) as data:
        voltage = data["voltage_v"]
        offsets = data["offsets"]
        for point, start, stop, rise, width in zip(
                data["points"], offsets[:-1], offsets[1:], data["rise_samples"], data["width50_ns"],
                strict=True):
            pulses.setdefault(tuple(map(int, point)), []).append((voltage[start:stop], rise, width))
    return pulses


def simulate_capture(frequency_khz, width_ns, seed):
    """Replay measured pulse shapes at the requested repetition rate."""
    period_ns, interval_ns, count = capture_grid(frequency_khz, width_ns)
    t = -width_ns + np.arange(count) * interval_ns
    templates = _measured_pulses()
    source_point = min(templates, key=lambda point: (abs(point[0] - frequency_khz),
                                                     abs(point[1] - width_ns), point))
    source_pulses = templates[source_point]
    rng = np.random.default_rng(seed)
    first = rng.integers(max(1, len(source_pulses) - 2))
    pulses = [source_pulses[(first + i) % len(source_pulses)] for i in range(3)]
    # The first 20 ns of each excerpt is sampled from the scope's low state.
    baseline = np.concatenate([samples[:100] for samples, _, _ in pulses])
    voltage = rng.choice(baseline, size=count).astype(float)

    for cycle, (samples, rise_sample, measured_width) in enumerate(pulses):
        source_t = (np.arange(len(samples)) - rise_sample) * TEMPLATE_INTERVAL_NS
        target_width = min(measured_width * (width_ns / source_point[1]), period_ns - interval_ns)
        padding = (period_ns - target_width) / 2
        pre = min(padding, -source_t[0])
        post = min(padding, source_t[-1] - measured_width)
        trigger = round((width_ns + cycle * period_ns) / interval_ns)
        start = max(0, ceil(trigger - pre / interval_ns))
        end = min(count, 1 + int(trigger + (target_width + post) / interval_ns))
        local_t = (np.arange(start, end) - trigger) * interval_ns
        # For a different width, stretch the interior and keep the two edge times.
        edge = min(20.0, 0.2 * measured_width, 0.2 * target_width)
        measured_t = np.interp(local_t,
                               [source_t[0], edge, target_width - edge,
                                target_width + source_t[-1] - measured_width],
                               [source_t[0], edge, measured_width - edge, source_t[-1]])
        voltage[start:end] = np.interp(measured_t, source_t, samples)
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
    return run_evaluation(settings, cancelled, on_started, on_measurement, interval_s=interval_s)


def run_evaluation(settings: SimulationSettings | FileEvaluationSettings | ScopeEvaluationSettings,
                   cancelled, on_started, on_measurement, *, interval_s=0.20):
    """Read, evaluate and save in the calling thread, publishing only saved captures."""
    directory = condition = None
    entries = []
    status = "running"
    error = ""
    simulated = isinstance(settings, SimulationSettings)
    source_file = settings.source_file if isinstance(settings, FileEvaluationSettings) else None
    now = datetime.now().astimezone()
    summary = dict(status=status, started_at=now.isoformat(), thresholds=settings.thresholds,
                   total=settings.count, measurements=entries)
    if simulated:
        summary.update(mode="simulation", scope_ip=settings.scope_ip, board_port=settings.board_port)
        for name, axis in (("frequency_khz", settings.frequencies), ("width_ns", settings.widths)):
            summary[name] = dict(start=axis.start, end=axis.stop - 1, step=axis.step)
        conditions = product(settings.frequencies, settings.widths)
    else:
        if source_file is not None:
            summary.update(mode="file", source_file=str(source_file))
        else:
            summary.update(mode="scope", scope_ip=settings.scope_ip, scope_identity=settings.scope_identity,
                           channel=SCOPE_CHANNEL, frame_received_at=settings.frame.received_at.isoformat())
        conditions = (None,)
    try:
        settings.destination.mkdir(parents=True, exist_ok=True)
        directory = Path(mkdtemp(dir=settings.destination, prefix=now.strftime("%Y%m%d_%H%M%S_")))
        write_summary(directory, summary)
        on_started(directory)
        rng = np.random.default_rng()
        for index, point in enumerate(conditions, 1):
            if cancelled.wait(interval_s if index > 1 else 0):
                status = "stopped"
                break
            condition = point
            if simulated:
                frequency, width = condition
                time_s, voltage_v = simulate_capture(frequency, width, int(rng.integers(2**32)))
                record = prepare_timed_waveform(time_s, voltage_v, point=condition)
                filename = f"{index:04d}_{frequency}kHz_{width}ns.npz"
                source_fields = dict(frequency_khz=frequency, width_ns=width)
            else:
                frequency = width = None
                if source_file is not None:
                    time_s, voltage_v, interval = load_waveform(source_file)
                    filename = "0001_导入波形.npz"
                    source_fields = dict(source_file=str(source_file))
                else:
                    frame = settings.frame
                    time_s, voltage_v, interval = frame.time_s, frame.voltage_v, frame.sample_interval_s
                    filename = "0001_实测波形.npz"
                    source_fields = {}
                record = prepare_timed_waveform(time_s, voltage_v, sample_interval_s=interval,
                                                path=source_file)
            report = evaluate_shape_record(record, **settings.thresholds)
            if cancelled.is_set():
                status = "stopped"
                break
            pulse = next(pulse for pulse in record.pulses
                         if pulse["number"] == report["shape"]["representative_pulse"])
            path = directory / filename
            report_json = json.dumps(report, ensure_ascii=False, allow_nan=False)
            atomic_write(path, lambda stream: np.savez_compressed(
                stream, time_s=time_s, voltage_v=voltage_v, evaluation_json=report_json))
            measurement = Measurement(index, frequency, width, path, report, pulse, source_file)
            entries.append(dict(**source_fields, capture=path.name,
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
