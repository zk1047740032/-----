"""Offline closed-loop leveling. A submitted step represents one actual change and a fresh capture."""

from __future__ import annotations

from copy import deepcopy
from math import isfinite
from numbers import Integral

import numpy as np

from .capture_sampling import shared_sample_interval_ns
from .hold_raw_band import classify_raw_band
from .operating_timing import frequency_check
from .reference_alignment import ALIGNMENT_TOLERANCE_EPS


AXES = ("y0", "y1", "y2")
STEP = 10
EPS = 1e-9


def action(kind, axis=None, step=0):
    return {"kind": kind, "axis": axis, "step": step}


def moved(offsets, move):
    values = list(offsets)
    if move["axis"] is not None:
        values[AXES.index(move["axis"])] += move["step"]
    return tuple(values)


def measurement(record, report, shape):
    if record.errors or not record.pulses or not record.waveform_fingerprint:
        raise ValueError("需要一份可识别且包含完整脉冲的新采集。")
    flat = report["flatness"]
    rms = float(flat["rms_median_pct"])
    raw = float(flat["raw_max_pct"])
    score = float(shape["score_pct"]) if shape.get("score_pct") is not None else None
    if not np.isfinite((rms, raw)).all() or (score is not None and not isfinite(score)):
        raise ValueError("保持期分数无效，请检查波形后重新采集。")
    interval = shared_sample_interval_ns([p["sample_interval_ns"] for p in flat["pulses"]])
    if interval is None:
        raise ValueError("同一采集中的有效采样间隔不一致。")
    widths = np.array([p["width50_ns"] for p in record.pulses], dtype=float)
    width = float(np.median(widths))
    if not isfinite(width) or width <= 0:
        raise ValueError("不能测出有效半高宽。")
    scan_count = len(report["window"]["sensitivity_windows"])
    scans = {}
    for key in ("rms_by_window_pct", "worst_rms_by_window_pct", "raw_by_window_pct"):
        values = flat.get(key)
        if values is not None and len(values) != scan_count:
            raise ValueError("边界扫描分数与窗口数目不一致。")
        scans[key] = [float(v) for v in values] if values and all(v is not None and isfinite(v) for v in values) else None
    thresholds = report.get("thresholds")
    geometry_valid = shape.get("geometry_valid", True)
    if thresholds is not None:
        scans["alignment"] = [score] if score is not None and geometry_valid else None
    return dict(path=str(record.path), fingerprint=record.waveform_fingerprint,
                voltage_fingerprint=(record.measurement_context or {}).get("voltage_fingerprint"), raw_pct=raw,
                raw_passed=bool(flat["raw_passed"]), raw_band=flat["raw_window_band"]["status"],
                raw_bounds=flat["raw_window_band"], thresholds=thresholds, passed=report.get("passed"),
                geometry_valid=geometry_valid,
                rms_pct=rms, worst_rms_pct=float(flat["rms_max_pct"]),
                shape_score_pct=score, axis_scores_pct=shape.get("axis_scores_pct"),
                shape_status=shape["status"], scans=scans,
                interval_ns=interval, width_ns=width, width_mad_ns=float(np.median(np.abs(widths-width))),
                timing=deepcopy(report.get("timing_check", {})),
                errors_pct={axis: shape.get("regions", {}).get(axis, {}).get("error_pct") for axis in AXES},
                coverage=flat.get("pulse_coverage_check", {}).get("status"),
                geometry_recovered=any(p.get("geometry_recovered", False) for p in record.pulses))


class LevelingSession:
    """One condition, one locked window, positions relative to the starting setting.

    If absolute bounds are available, pass initial_y and bounds in axis order.
    Bounds may also be a dictionary keyed by y0/y1/y2. Device communication is external.
    """

    def __init__(self, max_trials=12, max_repeats=2, initial_y=None, bounds=None):
        if any(not isinstance(n, Integral) or isinstance(n, bool) or n < 1 for n in (max_trials, max_repeats)):
            raise ValueError("试调与复测次数上限须为正整数。")
        self.max_trials = int(max_trials)
        self.max_repeats = int(max_repeats)
        self.initial_y = None
        if initial_y is not None:
            values = np.asarray(initial_y, dtype=float)
            if values.shape != (3,) or not np.isfinite(values).all():
                raise ValueError("实际初值须含三个有限数值。")
            self.initial_y = tuple(values.tolist())
        self.bounds = {}
        if bounds is not None:
            if self.initial_y is None:
                raise ValueError("核对绝对边界时需要三个实际初值。")
            if not isinstance(bounds, dict):
                bounds = dict(zip(AXES, bounds, strict=True))
            for axis, limits in bounds.items():
                limits = np.asarray(limits, dtype=float)
                if (axis not in AXES or limits.shape != (2,) or not np.isfinite(limits).all()
                        or not limits[0] <= self.initial_y[AXES.index(axis)] <= limits[1]):
                    raise ValueError("参数边界须覆盖当前实际初值。")
                self.bounds[axis] = tuple(limits.tolist())
        self.window = None
        self.offsets = (0, 0, 0)
        self.history = []
        self.responses = []
        self.best = None
        self.next_action = None
        self.pending_move = None
        self.limit_blocks = []
        self.status = "new"
        self.thresholds = None
        self.threshold_changes = []
        self.message = ""
        self.trials = 0
        self._seen = set()
        self._seen_voltages = set()
        self._tested = set()
        self._at_setting = {}
        self._probe = None
        self._repeats = 0

    def _set(self, status, message, next_move=None):
        self.status, self.message, self.next_action = status, message, next_move

    def snapshot(self):
        capture_offsets = moved(self.offsets, self.pending_move) if self.pending_move else None
        return deepcopy(dict(status=self.status, message=self.message, next_action=self.next_action,
                             offsets=list(self.offsets), pending_move=self.pending_move,
                             pending_offsets=list(capture_offsets) if capture_offsets is not None else None,
                             best=self.best, history=self.history,
                             limit_blocks=self.limit_blocks,
                             thresholds=self.thresholds, threshold_changes=self.threshold_changes,
                             responses=self.responses, trials=self.trials, max_trials=self.max_trials,
                             max_repeats=self.max_repeats, window=self.window))

    def _record(self, obs, move):
        obs.update(offsets=list(self.offsets), applied=dict(move))
        if self.initial_y is not None:
            obs["parameters"] = [y + delta for y, delta in zip(self.initial_y, self.offsets, strict=True)]
        self.history.append(obs)
        self._at_setting.setdefault(self.offsets, []).append(obs)
        self._seen.add(obs["fingerprint"])
        if obs["voltage_fingerprint"] is not None:
            self._seen_voltages.add(obs["voltage_fingerprint"])
        self._update_best()

    def _classification(self, obs):
        if self.thresholds is None:
            return obs["raw_passed"], obs["raw_band"], obs["shape_status"] == "flat"
        limit = self.thresholds["raw_limit_pct"]
        raw = [value for value in (obs["scans"].get("raw_by_window_pct") or ()) if value is not None]
        raw_passed = obs["raw_pct"] < limit
        observed_pass = raw_passed or any(value < limit for value in raw)
        flip = observed_pass and (not raw_passed or any(value >= limit for value in raw))
        bounds = obs["raw_bounds"]
        status = classify_raw_band(bounds["bounds_pct"], flip, observed_pass=observed_pass,
                                   lower_bound_exact=bounds.get("lower_bound_exact", False),
                                   limit_pct=limit)["status"]
        score = obs["shape_score_pct"]
        aligned = False
        if obs["geometry_valid"] and score is not None:
            tolerance = self.thresholds["alignment_tolerance_pct"]
            if isinstance(tolerance, dict):
                aligned = all(obs["axis_scores_pct"][axis] <= limit + ALIGNMENT_TOLERANCE_EPS
                              for axis, limit in tolerance.items())
            else:
                aligned = score <= tolerance + ALIGNMENT_TOLERANCE_EPS
        return raw_passed, status, aligned

    def _update_best(self):
        def rank(item):
            raw_passed, _, aligned = self._classification(item)
            if self.thresholds is not None:
                score = item["shape_score_pct"]
                return (not (raw_passed and aligned), score if score is not None else float("inf"),
                        item["raw_pct"], item["worst_rms_pct"])
            return (not raw_passed, not aligned, item["worst_rms_pct"], item["rms_pct"])

        eligible = (item for item in self.history
                    if (item["coverage"] != "early_loss_while_peers_hold"
                        and not item["geometry_recovered"] and item["geometry_valid"]))
        self.best = min(eligible, key=rank, default=None)

    def start(self, record, report, shape=None):
        if self.window is not None:
            raise ValueError("本轮已开始。请新建一轮以更换基准。")
        shape = shape or report["shape"]
        obs = measurement(record, report, shape)
        self.window = deepcopy(report["window"])
        self.thresholds = deepcopy(obs["thresholds"])
        self._record(obs, action("start"))
        self._choose(shape)
        self._log_decision()
        return self.snapshot()

    def _check_window(self, window):
        for key in ("phase", "sensitivity_available", "sensitivity_phase_bands", "sensitivity_windows",
                    "reference_fingerprint", "reference_operating_point", "alignment_windows"):
            if not np.array_equal(window.get(key), self.window.get(key)):
                raise ValueError("本轮参考保持期已经锁定。请在同一窗口下复评。")

    def reassess(self, record, report):
        """Apply new limits to the latest capture without accepting it a second time."""
        if not self.history or self.pending_move or self._probe:
            raise ValueError("请完成这一步的采集或回退后再调整阈值。")
        self._check_window(report["window"])
        obs = measurement(record, report, report["shape"])
        if obs["fingerprint"] != self.history[-1]["fingerprint"] or obs["thresholds"] is None:
            raise ValueError("请对本轮最新采集重新判定。")
        if self.thresholds != obs["thresholds"]:
            self.threshold_changes.append(dict(fingerprint=obs["fingerprint"], thresholds=obs["thresholds"],
                                               passed=obs["passed"]))
            self.thresholds = deepcopy(obs["thresholds"])
            self._update_best()
            self._choose(report["shape"], "已按新阈值判定当前波形。")
        return self.snapshot()

    def _check_context(self, obs, window):
        if obs["fingerprint"] in self._seen or obs["voltage_fingerprint"] in self._seen_voltages:
            raise ValueError("这份波形已用于本轮。请选择本次新采集的文件。")
        self._check_window(window)
        if obs["thresholds"] != self.thresholds:
            raise ValueError("本轮阈值已锁定，请按当前应用的阈值复评。")
        ref = self.history[0]
        if shared_sample_interval_ns([ref["interval_ns"], obs["interval_ns"]]) is None:
            raise ValueError("本次采样间隔与基准不同，请用同一采集设置复采。")
        if ref["timing"].get("status") == "regular":
            frequency = ref["timing"].get("observed_frequency_khz")
            if frequency_check(obs["timing"], frequency)["declared_check"] == "conflict":
                raise ValueError("两次波形的实测重复频率不同，请核对当前工况。")
        # Guard against changing files to a different nominal width. These are
        # tolerances for gross mismatch, not acceptance limits on the pulse.
        tolerance = max(.25*ref["width_ns"], 3*ref["width_mad_ns"], 2*ref["interval_ns"])
        if abs(obs["width_ns"]-ref["width_ns"]) > tolerance:
            raise ValueError("本次半高宽相对基准明显改变，请核对脉宽和采集设置。")

    def _physical_candidate(self, axis, step):
        move = action("trial", axis, step)
        target = moved(self.offsets, move)
        if (self.offsets, target) in self._tested or target in self._at_setting:
            return None
        if axis in self.bounds:
            i = AXES.index(axis)
            lower, upper = self.bounds[axis]
            value = self.initial_y[i]+target[i]
            if not lower <= value <= upper:
                self.limit_blocks.append(dict(axis=axis, step=step, requested=value, bounds=[lower, upper]))
                return None
        return move

    def _response_weight(self, move, regions):
        """Prefer steps whose measured regional response reduces current errors."""
        for response in reversed(self.responses):
            if response["axis"] != move["axis"] or response["verdict"] not in ("better", "worse"):
                continue
            distance = sum(abs(a-b) for a, b in zip(response["to_offsets"], self.offsets, strict=True))
            if distance > STEP:
                continue
            signs = move["step"] / response["step"]
            pairs = []
            for axis, region in regions.items():
                value = region.get("error_pct")
                limits = region.get("reference_band_pct")
                delta = response["region_delta_pct"].get(axis)
                if value is None or limits is None or delta is None:
                    continue
                low, high = limits
                predicted = value + delta*signs
                pairs.append((max(low-value, value-high, 0.), max(low-predicted, predicted-high, 0.)))
            if pairs:
                before = np.linalg.norm([error for error, _ in pairs])
                after = np.linalg.norm([error for _, error in pairs])
                return float((before-after) / max(before, EPS))
        return None

    def _choose(self, shape, prefix=""):
        self.limit_blocks = []
        if shape["status"] == "flat":
            if self.thresholds is not None and not self._classification(self.history[-1])[0]:
                self._set("aligned", prefix+"红蓝已对齐，整段不平度超限。保持三个 y，检查区内和框间波形。")
            else:
                message = "两项达标，保持三个 y，本轮结束。" if self.thresholds else "保持三个 y，本轮结束。"
                self._set("flat", prefix+message)
            return
        if self.trials >= self.max_trials:
            self._set("budget", prefix+"达到本轮试调次数上限，已记录本轮最佳观测。")
            return
        primary = shape.get("candidate")
        if shape["status"] != "trial" or not primary:
            self._set("uncertain", prefix+shape.get("reason", "当前没有足够清楚的试调方向。"))
            return
        choices = []
        for axis in AXES:
            direction = shape["directions"].get(axis)
            if direction not in ("increase", "decrease"):
                continue
            initial = STEP if direction == "increase" else -STEP
            for step in (initial, -initial):
                region = shape["regions"][axis]
                move = action("trial", axis, step)
                weight = self._response_weight(move, shape["regions"])
                if (weight is None and step != initial) or (weight is not None and weight <= EPS):
                    continue
                if self._physical_candidate(axis, step) is None:
                    continue
                choices.append(((weight is not None, weight or 0., axis == primary["axis"],
                                 region.get("excess_pct", 0.)), move))
        if not choices:
            self._set("exhausted", prefix+"可试方向已测过、达到边界或本轮实测响应不支持，保留本轮最佳观测。")
            return
        rank, move = max(choices, key=lambda pair: pair[0])
        basis = "依据本轮已测区域响应" if rank[0] else "依据保持期区域偏差"
        self._set("trial", prefix+f"{basis}，试调 {move['axis']} {move['step']:+d}，随后新采集。",
                  move)

    def _repeat(self, reason):
        self._set("repeat", reason+"保持当前参数，重新采集一次。", action("repeat"))

    def _matched_comparison(self, before, after, metric):
        arrays = [obs["scans"][metric] for obs in before+after]
        if any(values is None for values in arrays):
            return dict(gain_range_pp=None, noise_pp=None)
        a = np.asarray(arrays[:len(before)])
        b = np.asarray(arrays[len(before):])
        # Compare the same phase window, including all before/after file pairs.
        lo = float(np.min(a.min(axis=0)-b.max(axis=0)))
        hi = float(np.max(a.max(axis=0)-b.min(axis=0)))
        spreads = [float(np.max(np.ptp(group, axis=0))) for group in (a, b) if len(group) > 1]
        return dict(gain_range_pp=[lo, hi], noise_pp=max(spreads) if spreads else None)

    def _compare(self):
        probe = self._probe
        before = probe["before"]
        after = self._at_setting[self.offsets]
        checks = before + after
        evidence = dict(axis=probe["move"]["axis"], step=probe["move"]["step"],
                        from_offsets=before[-1]["offsets"], to_offsets=list(self.offsets),
                        before_files=[o["fingerprint"] for o in before],
                        after_files=[o["fingerprint"] for o in after], region_delta_pct={})
        evidence["rms"] = self._matched_comparison(before, after, "rms_by_window_pct")
        evidence.update(self._matched_comparison(before, after, "alignment") if self.thresholds
                        else evidence["rms"])
        evidence["worst_rms"] = self._matched_comparison(before, after, "worst_rms_by_window_pct")
        evidence["raw"] = self._matched_comparison(before, after, "raw_by_window_pct")
        for axis in AXES:
            a = [o["errors_pct"].get(axis) for o in before]
            b = [o["errors_pct"].get(axis) for o in after]
            evidence["region_delta_pct"][axis] = (None if None in a+b else float(np.median(b)-np.median(a)))
        before_checks = [self._classification(o) for o in before]
        after_checks = [self._classification(o) for o in after]
        if any(o["coverage"] == "early_loss_while_peers_hold" or o["geometry_recovered"]
               or not o["geometry_valid"] for o in checks):
            verdict = "shape_review"
        elif (all(band == "all_below_limit" for _, band, _ in before_checks)
              and all(band == "all_above_limit" for _, band, _ in after_checks)):
            verdict = "worse"
        elif (any(passed or band == "boundary_sensitive" for passed, band, _ in before_checks)
              and any(not passed or band == "boundary_sensitive" for passed, band, _ in after_checks)):
            verdict = "repeat"
        elif all(aligned for _, _, aligned in after_checks):
            # A new capture at the shape target stops tuning even if the
            # existing strict power limit is still unmet.
            verdict = "shape_target"
        elif ((self.thresholds is None and not self.window.get("sensitivity_available"))
              or evidence["gain_range_pp"] is None):
            verdict = "boundary_review"
        else:
            lo, hi = evidence["gain_range_pp"]
            noise = evidence["noise_pp"]
            if noise is not None and lo > noise+EPS:
                # A median improvement can conceal an obvious bad pulse.
                guard_metrics = [evidence["worst_rms"]]
                if any(not passed for passed, _, _ in after_checks):
                    guard_metrics.append(evidence["raw"])
                regressed = any(metric["noise_pp"] is not None
                                and metric["gain_range_pp"][1] < -metric["noise_pp"]-EPS
                                for metric in guard_metrics)
                verdict = "pulse_regression" if regressed else "better"
            elif noise is not None and hi < -noise-EPS:
                verdict = "worse"
            else:
                verdict = "repeat"
        evidence["verdict"] = verdict
        return evidence

    def mark_applied(self):
        """Record the operator's acknowledgement before checking the next file."""
        if not self.next_action or self.next_action["kind"] not in ("trial", "rollback"):
            raise ValueError("当前没有待确认的调参动作。")
        self.pending_move = dict(self.next_action)
        move = self.pending_move
        self._set("await_capture", f"已确认执行 {move['axis']} {move['step']:+d}，请选择这一步的新采集。",
                  action("capture"))
        return self.snapshot()

    def observe(self, record, report, shape=None, *, applied=True):
        if self.window is None or self.next_action is None:
            raise ValueError("当前没有待复评动作，请先开始新一轮。")
        shape = shape or report["shape"]
        obs = measurement(record, report, shape)
        self._check_context(obs, report["window"])
        if self.pending_move and not applied:
            raise ValueError("这一步已执行，请载入调整后的新采集。")
        move = dict(self.pending_move or self.next_action) if applied else action("repeat")
        self.pending_move = None
        previous = self.offsets
        self.offsets = moved(previous, move)
        if move["kind"] == "trial":
            self._tested.add((previous, self.offsets))
            self._probe = dict(move=move, before=list(self._at_setting[previous]))
            self.trials += 1
        self._repeats = self._repeats+1 if move["kind"] == "repeat" else 0
        self._record(obs, move)
        if move["kind"] == "rollback":
            self._probe = None
            self._choose(shape, "已采集回退后的波形。")
        elif self._probe is not None:
            response = self._compare()
            verdict = response["verdict"]
            if verdict == "shape_target":
                self.responses.append(response)
                self._probe = None
                self._choose(shape, "复评后保留这一步。")
            elif verdict == "better":
                self.responses.append(response)
                self._probe = None
                metric = "红蓝偏差" if self.thresholds else "同窗 RMS"
                self._choose(shape, f"{metric}改善超过本轮复采波动，保留这一步。")
            elif verdict in ("worse", "pulse_regression") or self._repeats >= self.max_repeats:
                self.responses.append(response)
                probe = self._probe["move"]
                undo = action("rollback", probe["axis"], -probe["step"])
                reasons = {"worse": "同窗复评变差。", "pulse_regression": "部分脉冲的平整度仍有明显退化。"}
                reason = reasons.get(verdict, "复采仍无法确认改善。")
                self._set("rollback", reason+f"将 {undo['axis']} 调 {undo['step']:+d} 回到试调前设置，再采集。",
                          undo)
            else:
                reason = {"repeat": "差异未超出已有同参数波动，或还缺少同参数复采。",
                          "boundary_review": "保持期边界变化会影响结论或扫描覆盖不足。",
                          "shape_review": "当前脉冲形状或保持期覆盖异常。"}[verdict]
                self._repeat(reason)
        elif shape["status"] == "flat":
            self._choose(shape)
        elif self._repeats >= self.max_repeats:
            self._set("retest_limit", "达到连续同参数复测上限，已保留本轮最佳观测。")
        else:
            self._choose(shape, "已记录同参数复采波动。")
        self._log_decision()
        return self.snapshot()

    def _log_decision(self):
        self.history[-1].update(decision=self.status, note=self.message, next_action=deepcopy(self.next_action),
                                limit_blocks=deepcopy(self.limit_blocks))

    def end(self):
        if self.window is not None and self.next_action is not None:
            message = ("本轮结束。最后一次已确认调参仍待新采集核对。" if self.pending_move
                       else "本轮结束。可保存相对偏移和本轮最佳观测。")
            self._set("stopped", message)
        return self.snapshot()
