"""Compare the three controlled parts of a pulse to its rear plateau.

The phase intervals are initial estimates from the annotated waveform. The
reference right edge ends the default hold interval, before the falling
shoulder. All intervals are locked together when a leveling session starts.
"""

import itertools
from collections.abc import Mapping

import numpy as np
from scipy.signal import savgol_filter

from .hold_comparison import validate_phase_window
from .shape_feedback import _region_assessment


DEFAULT_ALIGNMENT_TOLERANCE_PCT = 7.0
ALIGNMENT_TOLERANCE_EPS = 1e-12
AXIS_LABELS = {"y0": "左肩", "y1": "平顶前段", "y2": "平顶中段"}
# Phase 0 and 1 are the two half-height crossings. These positions are
# observation windows taken from the figure, not calibrated response limits.
OBSERVATION_WINDOWS = {
    "y0": (.04, .125),
    "y1": (.125, .215),
    "y2": (.37, .55),
    "reference": (.635, .755),
}
RAMP_SEARCH_END = OBSERVATION_WINDOWS["y1"][0]
MIN_REGION_SAMPLES = 3


def _ramp_end(pulse):
    """Find the first sustained slow patch, independently of its height."""
    t, v, width = pulse["time_ns"], pulse["voltage_v"], pulse["width50_ns"]
    dt = float(np.median(np.diff(t)))
    n = max(5, round(3.0/dt))
    n += n % 2 == 0
    if n > len(t) or not np.isfinite(pulse["rise10_ns"]):
        return None
    slope = savgol_filter(v, n, 3, deriv=1, delta=dt)
    rise = slope[(t >= pulse["rise10_ns"]) & (t <= 0.)]
    if not rise.size or rise.max() <= 0.:
        return None
    persistence = max(1, int(np.ceil(1.0/dt)))
    limit = 0.10 * rise.max()
    for i in np.flatnonzero((t >= 0.) & (t <= RAMP_SEARCH_END*width)):
        if i + persistence <= len(t) and np.all(np.abs(slope[i:i+persistence]) <= limit):
            return float(t[i]/width)
    return None


def _clip_regions(phase, red_windows):
    left, right = phase
    windows = OBSERVATION_WINDOWS.copy()
    if red_windows is not None:
        if not isinstance(red_windows, Mapping) or red_windows.keys() - AXIS_LABELS.keys():
            raise ValueError("红区端点须按 y0、y1、y2 设置。")
        for axis, interval in red_windows.items():
            try:
                windows[axis] = validate_phase_window(interval)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(f"{axis} 红区须满足 0 ≤ 左端点 < 右端点 ≤ 1。") from error
    regions = {}
    for name, (lo, hi) in windows.items():
        if name == "y0":
            # The annotated left shoulder can precede the hold start.
            regions[name] = (lo, hi)
            continue
        a, b = max(left, lo), min(right, hi)
        regions[name] = (a, b) if b-a >= (hi-lo)/2 else None
    return regions


def adjust_reference_window(record, window, *, confirmed_phase=None, red_windows=None):
    """Apply the blue reference geometry to a window prepared by the evaluator."""
    automatic = window.get("automatic_phase")
    ramps = [_ramp_end(pulse) for pulse in record.pulses] if automatic is not None else []
    ramp_ends = [end for end in ramps if end is not None]
    # Choose a shared start from the later end of the measured shoulders.
    # The range remains visible because individual noisy pulses can differ.
    slow_start = float(np.percentile(ramp_ends, 90)) if ramp_ends else None
    if automatic is not None:
        auto_left, auto_right = automatic
        auto_left = max(auto_left, slow_start) if slow_start is not None else auto_left
        auto_right = min(auto_right, OBSERVATION_WINDOWS["reference"][1])
        automatic = validate_phase_window((auto_left, auto_right))
    confirmed = confirmed_phase is not None or window.get("confirmed", False)
    manual_phase = window["phase"] if confirmed_phase is None else confirmed_phase
    phase = validate_phase_window(manual_phase if confirmed else automatic)
    regions = _clip_regions(phase, red_windows)
    bands = window.get("sensitivity_phase_bands")
    scenarios = [phase]
    sensitivity_available = bool(window.get("sensitivity_available") and bands is not None)
    if bands is not None and not confirmed:
        # Keep the previous estimate of left-edge sensitivity while shifting to
        # the stricter slow-slope criterion. The right edge cannot run beyond blue.
        offsets = np.asarray(bands[0])-window["phase"][0]
        left_band = np.clip(phase[0]+offsets, 0., 1.)
        if ramp_ends:
            early, late = np.percentile(ramp_ends, [5, 95])
            left_band = [min(left_band[0], early), max(left_band[1], late)]
        right_band = np.minimum(bands[1], OBSERVATION_WINDOWS["reference"][1])
        bands = [list(map(float, left_band)), right_band.tolist()]
    elif bands is not None and phase != tuple(window["phase"]):
        offset = (np.asarray(phase) - window["phase"])[:, None]
        bands = np.clip(np.asarray(bands) + offset, 0., 1.).tolist()
    if bands is not None and bands[0][1] < bands[1][0]:
        phases = [np.linspace(*band, 7) for band in bands]
        scenarios += list(itertools.product(*phases))
    else:
        sensitivity_available = False
    scenarios = list(dict.fromkeys(scenarios))
    return dict(window, phase=phase, automatic_phase=automatic, confirmed=confirmed,
                evidence="manual" if confirmed else window.get("evidence"),
                sensitivity_phase_bands=bands, sensitivity_windows=scenarios,
                sensitivity_available=sensitivity_available, alignment_windows=regions,
                alignment_window_basis="annotated_w50_v3",
                alignment_ramp=dict(supported_pulses=len(ramp_ends), total_pulses=len(record.pulses),
                                    later_than_hold=sum(end > phase[0] for end in ramp_ends),
                                    estimated_range=[min(ramp_ends), max(ramp_ends)] if ramp_ends else None),
                alignment_ramp_supported=confirmed or len(ramp_ends)*3 >= len(record.pulses)*2)


def _level(pulse, interval):
    if interval is None:
        return None
    phase = pulse["time_ns"]/pulse["width50_ns"]
    values = pulse["voltage_v"][(phase >= interval[0]) & (phase <= interval[1])]
    if len(values) < MIN_REGION_SAMPLES:
        return None
    return float(np.median(values)-pulse["baseline_v"])


def _tolerance_value(value):
    message = "红蓝允许偏差须为有限的非负百分比。"
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(message) from error
    if not np.isfinite(value) or value < 0:
        raise ValueError(message)
    return value


def validate_alignment_tolerance(tolerance_pct):
    if not isinstance(tolerance_pct, Mapping):
        return _tolerance_value(tolerance_pct)
    if tolerance_pct.keys() - AXIS_LABELS.keys():
        raise ValueError("红蓝允许偏差须按 y0、y1、y2 设置。")
    return {axis: _tolerance_value(tolerance_pct.get(axis, DEFAULT_ALIGNMENT_TOLERANCE_PCT))
            for axis in AXIS_LABELS}


def assess_alignment(record, report, *, tolerance_pct=DEFAULT_ALIGNMENT_TOLERANCE_PCT):
    """Measure every complete pulse against its own locked blue reference."""
    tolerance_pct = validate_alignment_tolerance(tolerance_pct)
    limits = tolerance_pct if isinstance(tolerance_pct, dict) else dict.fromkeys(AXIS_LABELS, tolerance_pct)

    window, flatness = report["window"], report["flatness"]
    rows = flatness["pulses"]
    if not rows or len(rows) != len(record.pulses):
        raise ValueError("保持期评价与完整脉冲数量不一致。")
    windows = window["alignment_windows"]
    pulse_regions, profiles, failed, missing = {}, [], [], []
    references = []
    for row, pulse in zip(rows, record.pulses, strict=True):
        number = row["number"]
        if number != pulse["number"]:
            raise ValueError("脉冲编号与评分不一致。")
        reference = _level(pulse, windows["reference"])
        if reference is not None and reference <= 0.:
            reference = None
        references.append(reference)
        block_levels = np.asarray(row["block_levels_v"])
        profiles.append(block_levels/np.median(block_levels))
        areas = {}
        for axis in AXIS_LABELS:
            height = _level(pulse, windows[axis])
            error = 100.*(height-reference)/reference if height is not None and reference is not None else None
            areas[axis] = dict(level_v=height, error_pct=error)
        if any(area["error_pct"] is None for area in areas.values()):
            missing.append(number)
        elif any(abs(area["error_pct"]) > limits[axis] + ALIGNMENT_TOLERANCE_EPS
                 for axis, area in areas.items()):
            failed.append(number)
        pulse_regions[number] = dict(reference_level_v=reference, areas=areas)

    regions, axis_scores = {}, {}
    for axis, label in AXIS_LABELS.items():
        errors = [pulse["areas"][axis]["error_pct"] for pulse in pulse_regions.values()]
        axis_scores[axis] = None if None in errors else max(abs(error) for error in errors)
        requested = limits[axis]
        voting = requested + ALIGNMENT_TOLERANCE_EPS
        requested_limits = [-requested, requested]
        voting_limits = [-voting, voting]
        result = _region_assessment(label, windows[axis], voting_limits, errors)
        result["reference_band_pct"] = requested_limits
        if result["direction"] == "hold":
            result["reason"] = "该区域在红蓝允许偏差内。"
        regions[axis] = result
    directions = {axis: region["direction"] for axis, region in regions.items()}

    profiles = np.asarray(profiles)
    distance = np.mean((profiles - np.median(profiles, axis=0))**2, axis=1)
    reference_values = [value for value in references if value is not None]
    coverage = flatness["pulse_coverage_check"]["status"]
    damaged = coverage == "early_loss_while_peers_hold" or any(row["geometry_recovered"] for row in rows)
    ramp_supported = window.get("alignment_ramp_supported", True)
    result = dict(status="uncertain", reason="", candidate=None, regions=regions,
                  directions=directions, score_pct=max(axis_scores.values()) if not missing else None,
                  axis_scores_pct=axis_scores,
                  tolerance_pct=tolerance_pct, pulse_regions=pulse_regions,
                  reference_level_v=float(np.median(reference_values)) if reference_values else None,
                  raw_passed=bool(flatness["raw_passed"]), shape_failed_pulses=failed,
                  shape_unavailable_pulses=missing, aligned=not failed and not missing,
                  geometry_valid=not damaged and ramp_supported and not missing,
                  representative_pulse=rows[int(np.argmin(distance))]["number"],
                  worst_pulse=max(rows, key=lambda row: row["raw_pp_pct"])["number"],
                  stop_basis="blue_reference_alignment_v1")
    if damaged:
        result["reason"] = "脉冲提前跌落或出现深凹陷，请核对保持期。"
    elif not ramp_supported:
        result["reason"] = "上升斜坡终点不稳定，请在图上确认保持期。"
    elif missing:
        result["reason"] = "锁定窗口内有红框或蓝框采样不足，无法完成对齐判定。"
    elif not failed:
        result.update(status="flat", reason="三个红框的电平均在蓝框的允许偏差内。")
        return result
    else:
        options = [axis for axis in regions if directions[axis] in ("increase", "decrease")]
        if options:
            axis = max(options, key=lambda key: regions[key]["excess_pct"])
            step = 10 if directions[axis] == "increase" else -10
            result.update(status="trial", candidate=dict(axis=axis, step=step),
                          reason=f"{regions[axis]['label']}偏离蓝框，先试 {axis} {step:+d}。")
            return result
        result["reason"] = "仍有红蓝偏差超限，脉冲之间的调整方向缺少一致证据。"
    for axis, region in regions.items():
        directions[axis] = region["direction"] = "uncertain"
        region["reason"] = result["reason"]
    result["aligned"] = False
    return result
