"""Shape-only, single-record assessment. All levels are baseline-relative.

The provisional stop envelope comes from the 10 complete pulses in the user's
accepted 150ns.csv, scored in their shared automatic window.
Limits below are rounded outward from the worst observed pulse. This one
record supplies a starting point, not a production tolerance. No settings,
record identity, absolute voltage, or calibration file is used at inference.
"""

import numpy as np


# The arrows are near phase .08, .25 and .50 between the two W50 crossings.
# These neighborhoods support a trial. They do not describe independent gains.
# The accepted shoulder runs lower than the rest of its plateau.
OBSERVATIONS = {
    "y0": ("肩部", (0.04, 0.13), (-5.5, 1.5)),
    "y1": ("平顶前段", (0.17, 0.32), (-1.5, 1.5)),
    "y2": ("平顶中段", (0.44, 0.60), (-1.5, 1.5)),
}

# Raw sample deviations from median block level. The end blocks include
# shoulders. Every interior point still participates in the raw 15% check.
RAW_SHAPE_BANDS = ((-12., 4.), (-6., 6.), (-16., 4.))
RMS_LIMIT_PCT = 2.5                 # observed maximum 2.4424%
LOCAL_DEVIATION_LIMIT_PCT = 10.      # 3 ns average, observed maximum 9.6194%


def _inside_raw_shape_band(samples):
    sections = np.array_split(samples, 12)
    zones = (sections[0], np.concatenate(sections[1:-1]), sections[-1])
    return all(zone.min() >= low and zone.max() <= high
               for zone, (low, high) in zip(zones, RAW_SHAPE_BANDS, strict=True))


def _region_assessment(label, window, limits, pulse_errors):
    errors = np.asarray(pulse_errors, dtype=float)
    region = dict(label=label, phase_window=window, reference_band_pct=list(limits),
                  error_pct=None, excess_pct=0., direction="uncertain", reason="")
    if not len(errors) or not np.isfinite(errors).all():
        region["reason"] = "锁定窗口内该区域的有效采样不足。"
        return region
    error = float(np.median(errors))
    lo, hi = limits
    region["error_pct"] = error
    low = errors < lo
    high = errors > hi
    if low.any() and high.any():
        region["reason"] = "同次采集的脉冲在该区域给出相反意见。"
        return region
    if max(low.mean(), high.mean()) < 2 / 3:
        if low.any() or high.any():
            region["reason"] = "偏差只出现于部分脉冲。"
        else:
            region.update(direction="hold", reason="该区域处于暂定保持范围。")
        return region
    if low.mean() >= 2 / 3:
        region.update(direction="increase", excess_pct=float(lo - error), reason=f"{label}持续偏低。")
    else:
        region.update(direction="decrease", excess_pct=float(error - hi), reason=f"{label}持续偏高。")
    return region


def assess_shape(record, report):
    """Assess every complete pulse using one locked window and propose one trial.

    directions gives the observed direction for each region; candidate is the
    only step selected for this trial. Flat means shape-stop. The independent
    strict raw power check is returned even when it disagrees with shape-stop.
    """
    flatness = report["flatness"]
    rows = flatness["pulses"]
    if not rows or len(rows) != len(record.pulses):
        raise ValueError("保持期评价与完整脉冲数量不一致。")
    left, right = report["window"]["phase"]
    windows = {}
    for axis, (_, (start, end), _) in OBSERVATIONS.items():
        a, b = max(left, start), min(right, end)
        windows[axis] = [float(a), float(b)] if b - a >= 0.5 * (end - start) else None

    errors = {axis: [] for axis in OBSERVATIONS}
    profiles, failed = [], []
    for row, pulse in zip(rows, record.pulses, strict=True):
        if row["number"] != pulse["number"]:
            raise ValueError("脉冲编号与评分不一致。")
        levels = np.asarray(row["block_levels_v"])
        reference = float(np.median(levels))
        if not np.isfinite(reference) or reference <= 0:
            raise ValueError("保持期参考电平无效。")
        profiles.append(100 * (levels / reference - 1))
        t = pulse["time_ns"]
        phase = t / pulse["width50_ns"]
        deviation = 100 * ((pulse["voltage_v"] - pulse["baseline_v"]) / reference - 1)
        inside = (t >= row["start_ns"]) & (t <= row["end_ns"])
        if (row["block_rms_pct"] > RMS_LIMIT_PCT
                or max(row["local_high_pct"], row["local_low_pct"]) > LOCAL_DEVIATION_LIMIT_PCT
                or not _inside_raw_shape_band(deviation[inside])):
            failed.append(row["number"])
        for axis, window in windows.items():
            if window:
                region = deviation[inside & (phase >= window[0]) & (phase <= window[1])]
            else:
                region = np.empty(0)
            # Each third votes with its own mean to reject an isolated spike.
            thirds = np.array_split(region, 3)
            error = np.median([part.mean() for part in thirds]) if len(region) >= 6 else np.nan
            errors[axis].append(error)

    regions = {axis: _region_assessment(label, windows[axis], limits, errors[axis])
               for axis, (label, _, limits) in OBSERVATIONS.items()}
    directions = {axis: region["direction"] for axis, region in regions.items()}
    profiles = np.asarray(profiles)
    distance = np.mean((profiles - np.median(profiles, axis=0))**2, axis=1)
    result = dict(status="uncertain", reason="", candidate=None, regions=regions,
                  directions=directions, score_pct=float(flatness["rms_max_pct"]),
                  raw_passed=bool(flatness["raw_passed"]), shape_failed_pulses=failed,
                  representative_pulse=rows[int(np.argmin(distance))]["number"],
                  worst_pulse=max(rows, key=lambda row: row["raw_pp_pct"])["number"],
                  stop_basis="accepted_150ns_shape_v1")
    coverage = flatness["pulse_coverage_check"]["status"]
    if coverage == "early_loss_while_peers_hold" or any(row["geometry_recovered"] for row in rows):
        result["reason"] = "脉冲提前跌落或出现深凹陷，请核对波形覆盖范围。"
        for axis, region in regions.items():
            directions[axis] = region["direction"] = "uncertain"
            region["reason"] = result["reason"]
    elif not failed:
        result.update(status="flat", reason="完整脉冲均在认可样例的暂定形状范围内，停止试调。")
        for axis, region in regions.items():
            directions[axis] = region["direction"] = "hold"
            region["reason"] = "形状已满足暂定停止条件。"
    else:
        options = [axis for axis in regions if directions[axis] in ("increase", "decrease")]
        if options:
            axis = max(options, key=lambda key: regions[key]["excess_pct"])
            step = 10 if directions[axis] == "increase" else -10
            result.update(status="trial", candidate=dict(axis=axis, step=step),
                          reason=f"{regions[axis]['label']}偏差超过保持范围，先试 {axis} {step:+d}。")
        else:
            result["reason"] = "形状超出暂定范围，三个作用区未见足够一致的偏差可归因。"
    return result
