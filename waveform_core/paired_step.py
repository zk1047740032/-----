"""Ordered +10/-10 calibration for a fixed operating point and hold window."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .hold_comparison import score_comparison_record
from .pulse_coverage import check_pulse_coverage
from .capture_sampling import pulse_sample_intervals_ns, shared_sample_interval_ns
from .hold_raw_band import raw_band_report

SCHEMA = 'fiber1100-paired-step-v1'
AXES = ('y0','y1','y2')


def _window_scores(record, phases):
    if not record.pulses or record.errors or any(p.get('geometry_recovered',False) for p in record.pulses):
        raise ValueError('波形超出成对响应模型可评范围')
    scores = [score_comparison_record(record, phase) for phase in phases]
    for row in scores:
        if not row['valid']:
            raise ValueError('；'.join(row['errors']) or '保持期评分无效')
    return scores


def _profiles(scores):
    return np.asarray([[100.*(np.asarray(p['block_levels_v'])/p['mean_above_baseline_v']-1.)
                        for p in row['scores']] for row in scores])


def block_profiles(record, phases):
    return _profiles(_window_scores(record, phases))


def rms(profiles):
    return np.sqrt(np.mean(np.square(profiles),axis=-1))


def calibrate_step(*, capture_round, before, after, previous_y, next_y, phases, phase_bands=None):
    """Read an observed, ordered pair. The caller supplies actual acquisition order."""
    source, target = np.asarray(previous_y,dtype=float), np.asarray(next_y,dtype=float)
    if source.shape!=(3,) or target.shape!=(3,) or not np.isfinite([source,target]).all():
        raise ValueError('成对标定需要有效的起止 y0、y1、y2')
    changed = np.flatnonzero(target-source)
    if len(changed)!=1 or abs((target-source)[changed[0]])!=10:
        raise ValueError('成对标定须为实际单轴一步 10')
    axis = int(changed[0])
    start_hash, end_hash = before.waveform_fingerprint, after.waveform_fingerprint
    if not capture_round or not start_hash or not end_hash or start_hash==end_hash:
        raise ValueError('成对标定需要不同波形的身份和已核实轮次')
    interval = shared_sample_interval_ns(
        pulse_sample_intervals_ns(before) + pulse_sample_intervals_ns(after))
    if interval is None:
        raise ValueError('成对标定前后必须有一致的采样间隔')
    raw = []
    profiles = []
    for record in (before, after):
        scores = _window_scores(record, phases)
        profiles.append(_profiles(scores))
        values = [row['raw_pp_max'] for row in scores]
        band_status = raw_band_report(record.pulses, phase_bands, min(values) < 15 <= max(values),
                                     observed_pass=min(values) < 15)
        raw.append(dict(windows_pct=values, band=band_status))
    a, b = profiles
    return dict(round=capture_round, axis=AXES[axis], step=int((target-source)[axis]),
                from_y=source.tolist(), to_y=target.tolist(), fingerprints=[start_hash,end_hash],
                sample_interval_ns=interval,
                raw_qualification=raw, raw_phase_bands=np.asarray(phase_bands).tolist(),
                response_pp=(b.mean(axis=1)-a.mean(axis=1)).tolist(),
                measured_gain_pp=(np.median(rms(a),axis=1)-np.median(rms(b),axis=1)).tolist())


def build_calibration(point, phases, trials, repeat_margin_pp, *, fixed_sequence):
    scope = np.asarray(point,dtype=float)
    bounds = np.asarray(phases,dtype=float)
    noise = np.asarray(repeat_margin_pp,dtype=float)
    if scope.shape!=(2,) or not np.isfinite(scope).all() or min(scope)<=0:
        raise ValueError('成对标定工况无效')
    if (bounds.ndim!=2 or bounds.shape[1]!=2 or not len(bounds)
            or not np.isfinite(bounds).all() or np.any(bounds<0) or np.any(bounds>1)
            or np.any(bounds[:,0]>=bounds[:,1])):
        raise ValueError('成对标定窗口无效')
    if noise.shape!=(len(bounds),) or not np.isfinite(noise).all() or np.any(noise<0):
        raise ValueError('复采波动须与每个共同窗口对应')
    interval = shared_sample_interval_ns([trial.get('sample_interval_ns') for trial in trials])
    if trials and interval is None:
        raise ValueError('成对标定各轮需要可核实且一致的采样间隔')
    seen = defaultdict(lambda:(set(),set()))
    for trial in trials:
        key = (tuple(trial['from_y']),tuple(trial['to_y']))
        rounds, captures = seen[key]
        if trial['round'] in rounds or captures.intersection(trial['fingerprints']):
            raise ValueError('同一方向不能重复使用同轮或同条波形')
        rounds.add(trial['round'])
        captures.update(trial['fingerprints'])
    return dict(schema=SCHEMA, point=scope.tolist(), sensitivity_windows=bounds.tolist(), trials=trials,
                sample_interval_ns=interval,
                repeat_margin_pp=noise.tolist(), fixed_acquisition_sequence=bool(fixed_sequence))


def _raw_qualification_evidence(trial, window):
    scores = trial.get('raw_qualification')
    if not scores or len(scores) != 2:
        return 'missing_raw_calibration'
    before, after = (np.asarray(item['windows_pct'], dtype=float) for item in scores)
    if (before.shape != (len(window['sensitivity_windows']),) or after.shape != before.shape
            or not np.isfinite([before, after]).all()):
        return 'missing_raw_calibration'
    if np.any((before < 15) & (after >= 15)):
        return 'lost_raw_qualification'
    bands = window.get('sensitivity_phase_bands')
    if bands is None:
        return 'raw_grid_checked'
    if not np.array_equal(trial.get('raw_phase_bands'), bands):
        return 'raw_qualification_uncertain'
    previous, current = (item['band'] for item in scores)
    if previous['status'] == 'all_above_limit' or current['status'] == 'all_below_limit':
        return 'raw_band_checked'
    if ((previous['status'] == 'all_below_limit' and current['status'] in ('all_above_limit', 'boundary_sensitive'))
            or (current['status'] == 'all_above_limit' and previous['status'] == 'boundary_sensitive')):
        return 'lost_raw_qualification'
    return 'raw_qualification_uncertain'


def advise_paired(record, window, point, parameters, calibration):
    """Apply saved ordered responses to one waveform; runtime has no neighbor records."""
    axes = {axis:dict(direction='uncertain', reason='缺少当前位置的同工况有序一步复采',
                     steps=[]) for axis in AXES}
    report = dict(per_parameter=axes, exploratory_trial=None, robust_trial=None,
                  status='uncertain', device_command_allowed=False)
    phases = window['sensitivity_windows']
    if calibration.get('schema')!=SCHEMA:
        raise ValueError('成对标定文件的格式无效')
    if not np.array_equal(point,calibration['point']) or parameters is None:
        report.update(status='out_of_scope', reason='需匹配标定工况和当前三个 y')
        return report
    if (not window.get('confirmed') or not window.get('sensitivity_available')
            or not np.array_equal(phases,calibration['sensitivity_windows'])):
        report.update(status='out_of_scope',reason='成对响应需要原标定的确认窗口和完整敏感性评分')
        return report
    coverage = window.get('pulse_coverage_check')
    if coverage is None:
        coverage = check_pulse_coverage(record,window)
    if coverage['status']=='early_loss_while_peers_hold':
        report.update(status='pulse_coverage_review',reason='本条记录的保持期覆盖需复核')
        return report
    if calibration['trials']:
        expected_interval = calibration.get('sample_interval_ns')
        intervals = pulse_sample_intervals_ns(record)
        report['sampling'] = dict(calibration_interval_ns=expected_interval,
                                  current_interval_ns=shared_sample_interval_ns(intervals))
        recorded_intervals = [trial.get('sample_interval_ns') for trial in calibration['trials']]
        if shared_sample_interval_ns([expected_interval, *recorded_intervals, *intervals]) is None:
            report.update(status='outside_calibrated_sampling',
                          reason='当前采样间隔与有序复采标定不一致，或标定缺少采样间隔记录')
            return report
    try:
        profile = block_profiles(record,phases)
    except ValueError as error:
        report.update(status='outside_calibrated_shape',reason=str(error))
        return report
    grouped = defaultdict(list)
    excluded_rounds = {q['round'] for q in calibration['trials']
                       if record.waveform_fingerprint in q['fingerprints']}
    for trial in calibration['trials']:
        if np.array_equal(parameters,trial['from_y']) and trial['round'] not in excluded_rounds:
            grouped[trial['axis'],trial['step']].append(trial)

    repeat = np.asarray(calibration['repeat_margin_pp'])
    robust, exploratory = [], []
    shape_score = rms(profile)
    for (axis,step),trials in grouped.items():
        raw_evidence = [_raw_qualification_evidence(trial, window) for trial in trials]
        raw_blocker = next((code for code in ('lost_raw_qualification', 'missing_raw_calibration',
                                              'raw_qualification_uncertain') if code in raw_evidence), None)
        effects = np.asarray([q['response_pp'] for q in trials])
        measured = np.asarray([q['measured_gain_pp'] for q in trials])
        gains = shape_score[None]-rms(profile[None]+effects[:,:,None])
        nominal_mean = shape_score-rms(profile+effects.mean(axis=0)[:,None])
        weakest = float(np.min(gains))
        grade = 'insufficient_rounds'
        reason = '预测或已测同向改善证据不足'
        if len(trials)>=2:
            grade = 'conflicting_or_nonpositive'
            if raw_blocker:
                grade = raw_blocker
                reason = ('有序复采中该方向使原始不平度从低于 15% 变为超限'
                          if grade == 'lost_raw_qualification' else '缺少匹配窗口下的原始不平度证据')
            elif weakest>0 and np.min(measured)>0:
                grade = 'exploratory'
                exploratory.append((weakest,axis,step))
                if (not calibration['fixed_acquisition_sequence'] and np.all(gains>repeat[None,:,None])
                        and np.all(measured>repeat[None,:])):
                    grade = 'above_observed_repeat_variation'
                    robust.append((weakest,axis,step))
        axes[axis]['steps'].append(dict(step=step, rounds=len(trials), evidence=grade,
                                       raw_qualification_evidence=raw_evidence,
                                       nominal_mean_response_gain_pp=float(np.median(nominal_mean[0])),
                                       predicted_gain_range_pp=[weakest,float(np.max(gains))],
                                       calibration_gain_range_pp=[float(np.min(measured)),float(np.max(measured))]))
        axes[axis]['reason'] = reason

    for axis in AXES:
        moves = [q for q in exploratory if q[1]==axis]
        if moves:
            _,_,step = max(moves)
            axes[axis].update(direction='increase' if step>0 else 'decrease',
                              reason='保存的有序复采和当前波形的一步预测均同向改善')

    for label, pool in (('exploratory_trial',exploratory),('robust_trial',robust)):
        if pool:
            _, axis, step = max(pool)
            report[label] = dict(axis=axis,step=step)
            report['status'] = 'candidate'
    report['excluded_training_rounds'] = sorted(excluded_rounds)
    return report
