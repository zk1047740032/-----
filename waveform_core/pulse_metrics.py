"""Experimental numerical pulse metrics. No product acceptance limits are assumed.

All times are in seconds and all signal values are in volts. This module has no
file, device, GUI, or y-parameter dependencies. Positive, low-duty pulses only.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.stats import trim_mean


DEFAULT_DURATION_LEVEL = 0.99


def validate_duration_level(value):
    try:
        level = float(value)
    except (ValueError, TypeError) as error:
        raise ValueError("保持期电平比例必须大于0.1且小于1，例如0.95") from error
    if not np.isfinite(level) or not .1 < level < 1:
        raise ValueError("保持期电平比例必须大于0.1且小于1，例如0.95")
    return level


def runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def smooth(v, dt, duration_s=1e-9):
    count = max(1, int(round(duration_s / dt)))
    count += (count % 2 == 0)
    return uniform_filter1d(v, count, mode="nearest")


def locate_pulses(t, v, dt):
    if len(t) != len(v) or len(t) < 20 or not np.isfinite(v).all() or not np.isfinite(t).all():
        raise ValueError("Nonfinite, unequal, or insufficient waveform arrays")
    steps = np.diff(t)
    if dt <= 0 or np.any(steps <= 0):
        raise ValueError("Time must be strictly increasing; dt must be positive")
    if np.max(np.abs(steps - dt)) > max(.02 * dt, 1e-13):
        raise ValueError("Time axis is inconsistent with the sample interval")
    # Estimate noise from the lower state, not the whole record. The shortest
    # exports can contain a single 800 ns pulse in 2 us, so a global absolute-
    # deviation quantile is polluted by the pulse itself. 0.60281 is the SD of
    # the lower half of a unit Gaussian. This remains a positive-pulse bootstrap.
    lower = v[v <= np.median(v)]
    base = float(np.median(lower))
    noise = float(np.std(lower) / .602810)
    high = float(np.quantile(v, 0.999))
    amplitude = high - base
    if amplitude <= max(8 * noise, 1e-9):
        return [], {"baseline_v": base, "noise_v": noise, "bootstrap_amplitude_v": amplitude, "rejected_runs": 0}
    filtered = smooth(v, dt)
    starts, stops = runs(filtered > base + 0.5 * amplitude)
    valid, rejected = [], 0
    for start, stop in zip(starts, stops):
        # A 10 ns bootstrap rejection removes isolated noise; not an acceptance
        # width limit. The tested main conditions are 100--800 ns.
        if (stop - start) * dt < 10e-9 or start == 0 or stop == len(v):
            rejected += 1
            continue
        valid.append((int(start), int(stop)))
    return valid, {"baseline_v": base, "noise_v": noise, "bootstrap_amplitude_v": amplitude, "rejected_runs": rejected}


def crossings(t, v, level, rising):
    a, b = v[:-1], v[1:]
    indices = np.flatnonzero((a < level) & (b >= level) if rising else (a >= level) & (b < level))
    return t[indices] + (level - a[indices]) * (t[indices + 1] - t[indices]) / (b[indices] - a[indices])


def pulse_metrics(t, v, dt, start, stop, *, duration_level=DEFAULT_DURATION_LEVEL):
    duration_level = validate_duration_level(duration_level)
    coarse_width = (stop - start) * dt
    margin = max(80e-9, 0.6 * coarse_width)
    lo, hi = max(0, start - int(margin / dt)), min(len(v), stop + int(margin / dt))
    tx, raw = t[lo:hi], v[lo:hi]
    filtered = smooth(raw, dt)
    s, e = start - lo, stop - lo
    pre = raw[max(0, s - int(margin / dt)):max(1, s - int(max(10e-9, 0.08 * coarse_width) / dt))]
    if len(pre) < 10:
        raise ValueError("Insufficient pre-pulse baseline")
    baseline = float(np.median(pre))
    noise = float(np.std(pre))
    core = raw[s + int(0.2 * (e - s)):s + int(0.8 * (e - s))]
    amplitude = float(np.median(core) - baseline)
    if amplitude <= max(8 * noise, 1e-9):
        raise ValueError("Insufficient pulse amplitude relative to local baseline noise")
    midpoint = (tx[s] + tx[e - 1]) / 2
    r50s = crossings(tx, filtered, baseline + 0.5 * amplitude, True)
    f50s = crossings(tx, filtered, baseline + 0.5 * amplitude, False)
    r50 = float(r50s[r50s < midpoint][-1])
    f50 = float(f50s[f50s > midpoint][0])
    width = f50 - r50

    def edge(level, rising, before, anchor):
        values = crossings(tx, filtered, baseline + level * amplitude, rising)
        values = values[values <= anchor] if before else values[values >= anchor]
        return float(values[-1] if before else values[0]) if len(values) else np.nan

    r10 = edge(0.1, True, True, r50)
    r90 = edge(0.9, True, False, r50)
    f90 = edge(0.9, False, True, f50)
    f10 = edge(0.1, False, False, f50)
    duration_start = r50 if duration_level == .5 else edge(duration_level, True, duration_level < .5, r50)
    duration_end = f50 if duration_level == .5 else edge(duration_level, False, duration_level > .5, f50)
    top_mask = (tx >= r50 + 0.1 * width) & (tx <= r50 + 0.9 * width)
    top, top_t = raw[top_mask], (tx[top_mask] - r50) / width
    if len(top) < 20:
        raise ValueError("Insufficient top samples")
    slope, intercept = np.polyfit(top_t, top, 1)
    residual = top - (slope * top_t + intercept)
    structural = smooth(raw, dt, 5e-9)[top_mask]
    quantile_span = lambda values: float(np.quantile(values, .95) - np.quantile(values, .05))
    front = float(np.mean(top[top_t <= .3]))
    middle = float(np.mean(top[(top_t >= .4) & (top_t <= .6)]))
    rear = float(np.mean(top[top_t >= .7]))
    on = (tx >= r50) & (tx <= f50)
    near = (tx >= r50 - .1 * width) & (tx <= f50 + .3 * width)
    in_band = on & (np.abs(filtered - (baseline + amplitude)) <= .05 * amplitude)
    band_starts, band_stops = runs(in_band)
    longest_band = max(band_stops - band_starts, default=0) * dt / width
    tail_mask = (tx >= f50 + .1 * width) & (tx <= f50 + .3 * width)
    raw_peak = float(np.max(raw[near]))
    raw_trough = float(np.min(raw[near]))
    shape = {
        "baseline_v": baseline, "amplitude_v": amplitude, "baseline_noise_v": noise,
        "amplitude_trimmed_mean_v": float(trim_mean(core, .1) - baseline),
        "width50_ns": width * 1e9, "rise10_90_ns": (r90 - r10) * 1e9,
        "fall90_10_ns": (f10 - f90) * 1e9,
        "flat80_q95_q05_pct": quantile_span(top) / amplitude * 100,
        "flat80_raw_pp_pct": float(np.ptp(top)) / amplitude * 100,
        "flat80_smooth5ns_q_pct": quantile_span(structural) / amplitude * 100,
        "droop_full_width_pct": float(slope / amplitude * 100),
        "ripple80_detrended_q_pct": quantile_span(residual) / amplitude * 100,
        "ripple80_detrended_rms_pct": float(np.std(residual) / amplitude * 100),
        "bow_pct": (middle - .5 * (front + rear)) / amplitude * 100,
        "front_minus_rear_pct": (front - rear) / amplitude * 100,
        "shoulder_drop_pct": float((np.mean(raw[(tx >= r50 + .4 * width) & (tx <= r50 + .6 * width)]) - np.mean(raw[(tx >= r50 + .85 * width) & (tx <= r50 + .95 * width)])) / amplitude * 100),
        "overshoot_raw_pct": max(0., (raw_peak - baseline - amplitude) / amplitude * 100),
        "undershoot_raw_pct": max(0., (baseline - raw_trough) / amplitude * 100),
        "band5_total_coverage_pct": float(np.mean(in_band[on]) * 100),
        "band5_longest_coverage_pct": float(longest_band * 100),
        "tail_mean_pct": float((np.mean(raw[tail_mask]) - baseline) / amplitude * 100) if np.any(tail_mask) else np.nan,
        "snr_amplitude_sigma": amplitude / max(noise, 1e-12),
        "rise50_s": r50, "fall50_s": f50, "rise10_s": r10, "rise90_s": r90,
        "fall90_s": f90, "fall10_s": f10,
        "duration_start_s": duration_start, "duration_end_s": duration_end,
        "rise_time_ns": (duration_start - r10) * 1e9,
        "fall_time_ns": (f10 - duration_end) * 1e9,
        "rise_start_s": r10, "rise_end_s": duration_start,
        "fall_start_s": duration_end, "fall_end_s": f10,
    }
    crop = {"time_s": tx - r50, "voltage_v": raw, "filtered_v": filtered}
    return shape, crop
