"""离线评价波形的保持期、平整度和 y0/y1/y2 调整方向。"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from waveform_core.hold_comparison import score_comparison_record, validate_phase_window
from waveform_core.hold_flatness import derivative_window
from waveform_core.y_direction import DEFAULT_MODEL, recommend_y_direction
from waveform_core.timing_pulses import prepare_timed_file, prepare_timed_waveform
from waveform_core.paired_step import advise_paired
from waveform_core.hold_raw_band import raw_band_report
from waveform_core.trial_feedback import review_trial
from waveform_core.direction_band import check_direction_band
from waveform_core.operating_timing import assert_frequency_consistent
from waveform_core.pulse_coverage import check_pulse_coverage
from waveform_core.capture_sampling import pulse_sample_intervals_ns, shared_sample_interval_ns


# The nominal values reproduce the six reviewed windows on their labeled pulses.
# Change one design parameter at a time to measure sensitivity.
SLOPE_SETTINGS = ((3., .20), (2., .20), (4., .20), (3., .15), (3., .25))
AXES = ('y0', 'y1', 'y2')
MOVES = {'increase': 10, 'decrease': -10}
MODEL_SAMPLE_INTERVAL_NS = 0.2


def known_context(record):
    saved = DEFAULT_MODEL['source_catalog'].get(record.waveform_fingerprint,{}) if DEFAULT_MODEL else {}
    return dict(saved,**(record.measurement_context or {}))


def window_from_reference(record, confirmed_phase=None, *, operating_point=None, use_saved_context=True):
    """Find a common phase once, from the reference's complete pulses."""
    if record.errors or not record.pulses:
        raise ValueError('；'.join(record.errors) or '参考波形没有完整可评的脉冲')

    declared = known_context(record).get('operating_point') if use_saved_context else None
    if operating_point is not None and declared is not None and not np.array_equal(operating_point,declared):
        raise ValueError('参考波形的已知工况与声明的工况不一致')
    scope = operating_point if operating_point is not None else declared
    if scope is not None:
        scope = np.asarray(scope,dtype=float)
        if scope.shape!=(2,) or not np.isfinite(scope).all() or np.any(scope<=0):
            raise ValueError('参考窗工况需为有效的频率 kHz 和脉宽 ns')
        scope = scope.tolist()
    cadence = assert_frequency_consistent(record,scope)

    variants = []
    for smoothing, fraction in SLOPE_SETTINGS:
        phases = []
        for p in record.pulses:
            if (smoothing, fraction) == SLOPE_SETTINGS[0]:
                if not p['auto_hold']['valid']:
                    continue
                phases.append(p['auto_hold']['phase_window'])
                continue
            try:
                width = p['width50_ns']
                if not np.isfinite([p['rise10_ns'], p['fall10_ns']]).all():
                    continue
                w = derivative_window(p['time_ns'], p['voltage_v'], p['rise10_ns'],
                                      0., width, p['fall10_ns'], smoothing_ns=smoothing,
                                      fraction=fraction, persistence_ns=1.)
                if w['rise_peak_slope_v_per_ns'] > 0 > w['fall_peak_slope_v_per_ns']:
                    phases.append(validate_phase_window((w['start_ns']/width, w['end_ns']/width)))
            except ValueError:
                continue
        variants.append(np.asarray(phases, dtype=float).reshape(-1, 2))
    nominal = variants[0]
    if len(nominal):
        estimate = tuple(np.median(nominal, axis=0))
    else:
        estimate = None
    if confirmed_phase is None and estimate is None:
        raise ValueError('自动斜率法缺少有效边沿，请人工确认保持期')
    phase = validate_phase_window(confirmed_phase if confirmed_phase is not None else estimate)
    # The envelope describes design sensitivity and pulse dispersion under the
    # five settings. Transfer offsets around the locked phase, including manual locks.
    scenarios = [phase]
    bands = None
    if estimate is not None:
        percentiles = np.asarray([np.percentile(rows, [5, 95], axis=0) for rows in variants if len(rows)])
        bounds = np.array([percentiles[:, 0].min(axis=0), percentiles[:, 1].max(axis=0)])
        bounds += np.asarray(phase)-estimate
        bounds = np.clip(bounds, 0., 1.)
        bands = bounds.T.tolist()
        if bounds[1, 0] < bounds[0, 1]:
            starts, ends = [np.linspace(*bounds[:, edge], 7) for edge in (0, 1)]
            scenarios.extend(itertools.product(starts, ends))
    scenarios = list(dict.fromkeys(scenarios))
    return dict(phase=phase, first_pulse_phase=record.pulses[0]['suggested_phase'],
                automatic_phase=estimate, confirmed=confirmed_phase is not None,
                reference_operating_point=scope, reference_fingerprint=record.waveform_fingerprint,
                reference_timing_check=cadence,
                evidence='manual' if confirmed_phase is not None else 'automatic_candidate',
                pulse_count=len(record.pulses), variant_coverage=[len(p) for p in variants],
                sensitivity_phase_bands=bands, sensitivity_windows=scenarios,
                sensitivity_available=(all(len(rows) == len(record.pulses) for rows in variants)
                                       and bands[0][1] < bands[1][0]))


def coherent_hold_profile(signal, sample_interval_ns, mean_v):
    """Relative percent deviation of overlapping 3 ns averages inside the hold."""
    target_samples = round(3./sample_interval_ns, 9)
    count = min(len(signal), max(1, int(np.floor(target_samples+.5))))
    total = np.r_[0., np.cumsum(signal)]
    averages = (total[count:]-total[:-count])/count
    return 100.*(averages-mean_v)/mean_v, count


def score_locked(record, window):
    """Keep all raw hold samples and report coherent local deviations as diagnostics."""
    results = []
    rms_by_window = []
    worst_rms_by_window = []
    raw_by_window = []
    for phase in window['sensitivity_windows']:
        score = score_comparison_record(record, phase)
        rms_by_window.append(score['rms_median'] if score['valid'] else None)
        worst_rms_by_window.append(score['rms_max'] if score['valid'] else None)
        raw_by_window.append(score['raw_pp_max'] if score['valid'] else None)
        if score['valid']:
            results.append(score)
        elif not results:
            raise ValueError('；'.join(score['errors']))
    selected = results[0]
    for pulse, row in zip(record.pulses, selected['scores'], strict=True):
        t, v = pulse['time_ns'], pulse['voltage_v']
        inside = (t >= row['start_ns']) & (t <= row['end_ns'])
        hold_time = t[inside]
        dt = float(np.median(np.diff(hold_time)))
        row['sample_interval_ns'] = dt
        profile, count = coherent_hold_profile(v[inside]-pulse['baseline_v'], dt,
                                                row['mean_above_baseline_v'])
        high, low = int(np.argmax(profile)), int(np.argmin(profile))
        centers = (hold_time[:len(profile)]+hold_time[count-1:])/2.
        row['local_high_pct'] = float(profile[high])
        row['local_low_pct'] = float(-profile[low])
        row['local_high_ns'] = float(centers[high])
        row['local_low_ns'] = float(centers[low])
        row['local_average_ns'] = count*dt
        row['coherent_rms_pct'] = float(np.sqrt(np.mean(profile**2)))
        row['raw_passed'] = row['raw_pp_pct'] < 15.
        row['geometry_recovered'] = pulse.get('geometry_recovered', False)
    score_range = [s['rms_median'] for s in results]
    raw_range = [s['raw_pp_max'] for s in results]
    coherent = [s['coherent_rms_pct'] for s in selected['scores']]
    raw_limit_sensitive = min(raw_range) < 15. <= max(raw_range)
    band = window.get('sensitivity_phase_bands') if window.get('sensitivity_available') else None
    bounds = raw_band_report(record.pulses, band, raw_limit_sensitive,
                             observed_pass=min(raw_range)<15.)
    return dict(pulses=selected['scores'], rms_median_pct=selected['rms_median'],
                rms_max_pct=selected['rms_max'], rms_window_range_pct=[min(score_range), max(score_range)],
                coherent_rms_median_pct=float(np.median(coherent)), coherent_rms_max_pct=max(coherent),
                raw_max_pct=selected['raw_pp_max'], raw_passed=selected['raw_pp_max'] < 15.,
                raw_window_range_pct=[min(raw_range), max(raw_range)],
                raw_limit_sensitive=raw_limit_sensitive, raw_window_band=bounds,
                sensitivity_scores=len(results), rms_by_window_pct=rms_by_window,
                worst_rms_by_window_pct=worst_rms_by_window, raw_by_window_pct=raw_by_window,
                shape_passed=None, pulse_coverage_check=check_pulse_coverage(record,window),
                local_high_pct=max(s['local_high_pct'] for s in selected['scores']),
                local_low_pct=max(s['local_low_pct'] for s in selected['scores']))


def advise(record, window, *, point=None, parameters=None, repeat_margin_pp=None, model=DEFAULT_MODEL):
    """Use the saved profile model and check practical evidence for each fixed step."""
    coverage = window.get('pulse_coverage_check')
    if coverage is None:
        coverage = check_pulse_coverage(record,window)
    early = coverage['status']=='early_loss_while_peers_hold'
    recovered = any(p.get('geometry_recovered', False) for p in record.pulses)
    outside_sampling = (model is not None and model is DEFAULT_MODEL
                        and shared_sample_interval_ns([MODEL_SAMPLE_INTERVAL_NS,
                                                       *pulse_sample_intervals_ns(record)]) is None)
    if early or recovered or outside_sampling:
        if early:
            evidence = 'pulse_coverage_review'
            reason = '脉冲已降到低电平时多数邻脉冲仍在保持期，需复核覆盖范围'
        elif recovered:
            evidence = 'outside_calibrated_shape'
            reason = '存在跌破半高的深凹陷，超出当前 y 响应校准的波形形态'
        else:
            evidence = 'outside_calibrated_sampling'
            reason = '当前采样间隔与内置 y 方向模型的 0.2 ns 标定记录不同'
        rows = {axis:dict(direction='uncertain', nominal_model_direction='uncertain',
                          evidence=evidence, minimum_model_gain_pp=None,
                          votes=[], reasons=[reason]) for axis in AXES}
        return dict(per_parameter=rows, repeat_margin_pp=repeat_margin_pp,
                    priority_trial=None, exploratory_trial=None,
                    model_operating_point=model['operating_point'] if model else None,
                    device_command_allowed=False)
    nominal = recommend_y_direction(record, window['phase'], operating_point=point,
                                    parameters=parameters, model=model)
    contexts = [nominal]
    if nominal['pulses']:
        contexts.extend(recommend_y_direction(record, p, operating_point=point,
                                              parameters=parameters, model=model)
                        for p in window['sensitivity_windows'][1:])
    rows = {}
    current = np.asarray(nominal['parameters']) if nominal['pulses'] else None
    center = np.asarray(model['center_parameters']) if nominal['pulses'] else None
    limits = np.asarray(model['calibration_parameter_box']) if nominal['pulses'] else None
    for i, axis in enumerate(AXES):
        candidate = nominal['directions'][axis]
        votes = [c['directions'][axis] for c in contexts]
        reasons = []
        gain = None
        on_axis = current is not None and bool(np.all(np.delete(current-center, i) == 0))
        if candidate == 'uncertain':
            reasons.append(nominal['per_parameter'].get(axis, {}).get('reason', nominal['reason']))
        elif candidate in MOVES:
            gains = [p['per_parameter'][axis]['worst_model_improvement_pp']
                     for c in contexts for p in c['pulses']]
            finite = [g for g in gains if g is not None]
            gain = min(finite) if finite else None
            if not on_axis:
                reasons.append('当前位置与试调点涉及多轴组合，缺少对应响应校准')
            if gain is None or gain <= 0 or any(v != candidate for v in votes):
                reasons.append('移动窗口边界后，模型未保持同向改善')
        elif not on_axis or (current[i]-10 < limits[i, 0] or current[i]+10 > limits[i, 1]):
            reasons.append('现有校准没有覆盖当前位置的两侧试调')
        elif any(v != 'hold' for v in votes):
            reasons.append('移动窗口边界后，保持意见有变化')
        if not window['sensitivity_available']:
            reasons.append('窗口敏感性证据不足')
        proposal = candidate if not reasons else 'uncertain'
        rows[axis] = dict(direction=proposal, nominal_model_direction=candidate,
                          minimum_model_gain_pp=gain, votes=votes, reasons=reasons)

    wanted = {axis:row['direction'] for axis,row in rows.items() if row['direction']!='uncertain'}
    if wanted:
        bands = window.get('sensitivity_phase_bands')
        proofs = check_direction_band(record,bands,wanted,point=point,parameters=parameters,model=model) if bands is not None else {}
        for axis in wanted:
            row = rows[axis]
            proof = proofs.get(axis)
            row['window_band_check'] = proof
            if proof is None or not proof['stable']:
                row['direction'] = 'uncertain'
                row['grid_minimum_model_gain_pp'] = row['minimum_model_gain_pp']
                row['minimum_model_gain_pp'] = None
                row['reasons'].append('边界带样本组合出现不同意见或缺少完整检查')
            elif row['direction'] in MOVES:
                row['minimum_model_gain_pp'] = min(row['minimum_model_gain_pp'],proof['minimum_gain_pp'])

    for row in rows.values():
        proposal, gain, reasons = row['direction'], row['minimum_model_gain_pp'], row['reasons']
        if proposal in MOVES:
            if repeat_margin_pp is None:
                evidence = 'model_only'
                reasons.append('缺少本工况同参数复采噪声包络')
            elif gain <= repeat_margin_pp:
                evidence = 'within_repeat_variation'
                reasons.append('预测改善未超过已观察到的同参数波动')
            else:
                evidence = 'above_observed_repeat_variation'
        elif proposal == 'hold':
            evidence = 'no_step_improvement_in_model'
        else:
            evidence = 'insufficient'
        row['evidence'] = evidence
    eligible = [(item['minimum_model_gain_pp'], axis) for axis, item in rows.items()
                if item['evidence'] == 'above_observed_repeat_variation']
    exploratory = [(item['minimum_model_gain_pp'], axis) for axis, item in rows.items()
                   if item['direction'] in MOVES]

    def best(choices):
        if choices:
            _, axis = max(choices)
            return dict(axis=axis, step=MOVES[rows[axis]['direction']])
        return None

    return dict(per_parameter=rows, repeat_margin_pp=repeat_margin_pp,
                priority_trial=best(eligible), exploratory_trial=best(exploratory),
                model_operating_point=nominal['model_operating_point'], device_command_allowed=False)


def evaluate_record(record, *, window=None, confirmed_phase=None, point=None, parameters=None,
                    repeat_margin_pp=None, paired_calibration=None):
    if window is None:
        window = window_from_reference(record, confirmed_phase, operating_point=point)
    reference_scope = window.get('reference_operating_point')
    recorded_scope = known_context(record).get('operating_point')
    if reference_scope is not None:
        scope = point if point is not None else recorded_scope
        if (not np.array_equal(reference_scope,scope)
                or (recorded_scope is not None and not np.array_equal(reference_scope,recorded_scope))):
            raise ValueError('参考窗仅可在原工况复用，请核对当前频率和脉宽')
    cadence = assert_frequency_consistent(record,point if point is not None else recorded_scope)
    flatness = score_locked(record, window)
    direction_window = dict(window, pulse_coverage_check=flatness['pulse_coverage_check'],
                            sensitivity_available=(
        window['sensitivity_available']
        and flatness['sensitivity_scores'] == len(window['sensitivity_windows'])))
    result = dict(window=window, flatness=flatness, timing_check=cadence,
                  advice=advise(record, direction_window, point=point, parameters=parameters,
                                repeat_margin_pp=repeat_margin_pp))
    if paired_calibration is not None:
        known = known_context(record)
        current_point = point if point is not None else known.get('operating_point')
        current_y = parameters if parameters is not None else known.get('parameters')
        result['paired_step_advice'] = advise_paired(record,direction_window,current_point,current_y,paired_calibration)
    return result


def evaluate_shape_record(record, *, window=None, confirmed_phase=None):
    """Evaluate one capture from its samples, optionally on the session's locked window."""
    from waveform_core.shape_feedback import assess_shape

    if window is not None and confirmed_phase is not None:
        raise ValueError('锁定的比较窗和新框选的相位只能使用一项')
    if window is None:
        window = window_from_reference(record, confirmed_phase, use_saved_context=False)
    phase = validate_phase_window(window['phase'])
    if not window.get('sensitivity_windows') or not np.array_equal(phase, window['sensitivity_windows'][0]):
        raise ValueError('参考窗起止和首个评分窗口不一致')
    report = dict(window=window, flatness=score_locked(record, window),
                  timing_check=assert_frequency_consistent(record, window.get('reference_operating_point')))
    report['shape'] = assess_shape(record, report)
    return report


def evaluate_waveform(time_s, voltage_v, **options):
    """Numerical entry for one record. Arrays use seconds and volts."""
    record = prepare_timed_waveform(time_s, voltage_v, point=options.get('point'))
    if not (options.keys() - {'window', 'confirmed_phase'}):
        return evaluate_shape_record(record, **options)
    return evaluate_record(record, **options)


def run_leveling(capture, apply_step, *, initial_y, bounds, confirmed_phase=None,
                 max_trials=12, max_repeats=2, cancelled=None):
    """Drive the same feedback loop through acquisition and board callbacks.

    capture() saves fresh data to a CSV/NPZ and returns its path. apply_step()
    returns True after the single-axis command is acknowledged and settled.
    Machine limits and initial settings come from the caller's controller.
    """
    from waveform_core.leveling import AXES as axes, LevelingSession

    limits = bounds if isinstance(bounds, dict) else dict(zip(axes, bounds, strict=True))
    if set(limits) != set(axes):
        raise ValueError("程序下发试调前需要控制器提供三个 y 的实际范围。")
    session = LevelingSession(max_trials=max_trials, max_repeats=max_repeats, initial_y=initial_y, bounds=limits)
    action = None
    phase = "初始采集"
    try:
        if cancelled is not None and cancelled():
            return dict(session.snapshot(), status="stopped", message="在初始采集前停止。")
        record = prepare_timed_file(capture())
        report = evaluate_shape_record(record, confirmed_phase=confirmed_phase)
        state = session.start(record, report)
        while state["next_action"] is not None:
            phase = "等待下一步"
            if cancelled is not None and cancelled():
                return session.end()
            action = state["next_action"]
            if action["kind"] in ("trial", "rollback"):
                phase = "下发参数及确认应答"
                if apply_step(action["axis"], action["step"]) is not True:
                    raise ValueError("控制器未确认已执行并稳定。")
                session.mark_applied()
            phase = "调整后采集" if session.pending_move else "同设置复采"
            record = prepare_timed_file(capture())
            report = evaluate_shape_record(record, window=session.window)
            state = session.observe(record, report, applied=session.pending_move is not None)
        return state
    except Exception as error:
        # If the command's acknowledgement is lost, the caller must reconcile
        # device settings before resuming. The loop never retries the command.
        return dict(session.snapshot(), status="interrupted", next_action=None,
                    message=f"{phase}失败。{error}", interrupted_action=action)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('waveform', type=Path)
    reference = parser.add_mutually_exclusive_group()
    reference.add_argument('--reference', type=Path, help='Optional same-condition baseline waveform')
    reference.add_argument('--window-json', type=Path, help='Use a previously saved reference window')
    parser.add_argument('--phase', nargs=2, type=float, help='Human-reviewed relative W50 interval')
    parser.add_argument('--point', nargs=2, type=float, metavar=('KHZ', 'NS'))
    parser.add_argument('--y', nargs=3, type=float)
    parser.add_argument('--paired-calibration', type=Path, help='Optional saved ordered-step evidence')
    parser.add_argument('--before', type=Path, help='Optional waveform recorded immediately before a one-axis trial')
    parser.add_argument('--before-y', nargs=3, type=float)
    parser.add_argument('--output', type=Path, help='Save the result to a new JSON file')
    args = parser.parse_args()
    if args.window_json and args.phase:
        parser.error('--phase and --window-json describe different window sources')
    if args.before and (args.before_y is None or args.y is None or args.point is None):
        parser.error('--before requires --before-y, --y and --point')
    if args.before_y is not None and not args.before:
        parser.error('--before-y requires --before')
    calibrated = any(value is not None for value in
                     (args.point, args.y, args.before, args.paired_calibration))
    record = prepare_timed_file(args.waveform, point=args.point)
    margin = None
    if args.window_json:
        saved = json.loads(args.window_json.read_text(encoding='utf-8'))
        window = saved['window']
        calibration = saved.get('repeat_calibration')
        if calibration is not None:
            known = DEFAULT_MODEL['source_catalog'].get(record.waveform_fingerprint, {}) if DEFAULT_MODEL else {}
            actual_point = args.point or known.get('operating_point')
            if actual_point != calibration['operating_point']:
                parser.error('复用已保存的参考窗口需要同工况，请用 --point 确认频率和脉宽')
            reference_point = window.get('reference_operating_point')
            if reference_point is not None and not np.array_equal(reference_point,calibration['operating_point']):
                parser.error('参考窗与复采校准的工况不一致')
            window = dict(window, reference_operating_point=calibration['operating_point'])
            margin = calibration['max_observed_same_setting_rms_difference_pp']
    else:
        baseline = prepare_timed_file(args.reference, point=args.point) if args.reference else record
        window = window_from_reference(baseline, args.phase, operating_point=args.point,
                                       use_saved_context=calibrated)
    paired = json.loads(args.paired_calibration.read_text(encoding='utf-8')) if args.paired_calibration else None
    if calibrated:
        report = evaluate_record(record, window=window, point=args.point, parameters=args.y,
                                 repeat_margin_pp=margin, paired_calibration=paired)
    else:
        report = evaluate_shape_record(record, window=window)
    if args.before:
        before = prepare_timed_file(args.before, point=args.point)
        report['trial_review'] = review_trial(before,record,window,point=args.point,
                                              previous_y=args.before_y,current_y=args.y,
                                              paired_calibration=paired)
    result = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2)
    if args.output:
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(result)
    else:
        print(result)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError) as error:
        raise SystemExit(str(error))
