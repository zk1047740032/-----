"""Offline review of one observed fixed-10 trial on a confirmed hold window."""
from __future__ import annotations

import numpy as np

from .hold_raw_band import raw_band_report
from .paired_step import advise_paired
from .rms_gain_band import exact_gain_band
from .operating_timing import assert_frequency_consistent
from .pulse_coverage import check_pulse_coverage
from .capture_sampling import pulse_sample_intervals_ns, shared_sample_interval_ns
from .hold_comparison import score_comparison_record
from .y_direction import DEFAULT_MODEL

AXES = ('y0','y1','y2')


def _declared_context(record):
    saved = DEFAULT_MODEL['source_catalog'].get(record.waveform_fingerprint,{}) if DEFAULT_MODEL else {}
    context = record.measurement_context or {}
    return dict(saved,**context)


def _measure(record, phases, bands):
    rows = [score_comparison_record(record,phase) for phase in phases]
    if any(not row['valid'] for row in rows):
        raise ValueError('；'.join(error for row in rows for error in row['errors']) or '共同窗口评分失败')
    raw = np.asarray([r['raw_pp_max'] for r in rows])
    bands_report = raw_band_report(record.pulses,bands,raw.min()<15.<=raw.max(),observed_pass=raw.min()<15.)
    intervals = pulse_sample_intervals_ns(record)
    detail = dict(raw_pp_pct=float(raw[0]),raw_passed=bool(raw[0]<15.),
                  raw_window_band=bands_report, rms_pct=rows[0]['rms_median'],
                  sample_interval_ns_range=[min(intervals), max(intervals)])
    return detail,np.asarray([r['rms_median'] for r in rows])


def _matched_noise(calibration, point, phases, bands, previous, current, sample_interval_ns):
    if (not calibration or not np.array_equal(calibration['point'],point)
            or not np.array_equal(calibration['sensitivity_windows'],phases)
            or shared_sample_interval_ns([sample_interval_ns, calibration.get('sample_interval_ns')]) is None):
        return None,None
    # The repeat model applies directly only to recorded parameter groups.
    known = {tuple(q[end]) for q in calibration['trials'] for end in ('from_y','to_y')}
    allowed = {tuple(previous),tuple(current)}
    if not allowed<=known:
        return None,None
    full = calibration.get('full_band_repeatability',{})
    groups = {tuple(y) for y in full.get('repeat_parameter_groups',[])}
    valid_band = (allowed<=groups and np.array_equal(full.get('phase_bands'),bands)
                  and shared_sample_interval_ns([sample_interval_ns, full.get('sample_interval_ns')]) is not None)
    return calibration['repeat_margin_pp'],full.get('maximum_difference_pp') if valid_band else None


def review_trial(before, after, window, *, point, previous_y, current_y,
                 repeat_margin_pp=None, repeat_band_margin_pp=None, paired_calibration=None):
    """Use distinct measured records. Ordering is declared by the caller."""
    prior,current = np.asarray(previous_y,dtype=float), np.asarray(current_y,dtype=float)
    scope = np.asarray(point,dtype=float)
    if (prior.shape!=(3,) or current.shape!=(3,) or scope.shape!=(2,) or not np.isfinite([prior,current]).all()
            or not np.isfinite(scope).all() or np.any(scope<=0)):
        raise ValueError('复评需要相同工况以及试调前后的三个 y')
    difference = current-prior
    changed = np.flatnonzero(difference)
    if len(changed)!=1 or abs(difference[changed[0]])!=10:
        raise ValueError('复评只接受一次实际单轴 ±10 试调')
    index = int(changed[0])
    axis,step = AXES[index],int(difference[index])
    report = dict(status='undetermined', action='review_or_retest',
                  tested_step=dict(axis=axis,step=step), next_trial=None,
                  previous_y=prior.tolist(), current_y=current.tolist(), point=scope.tolist(),
                  waveform_fingerprints=[before.waveform_fingerprint,after.waveform_fingerprint],
                  order_evidence='supplied_before_after',device_command_allowed=False,direction_validated=False)

    def stop(status, reason):
        report.update(status=status,reason=reason)
        return report

    if (not before.waveform_fingerprint or not after.waveform_fingerprint
            or before.waveform_fingerprint==after.waveform_fingerprint):
        return stop('duplicate_or_unknown_capture','试调前后的原始波形必须来自可区分的采集')
    for record, setting in ((before,prior),(after,current)):
        known = _declared_context(record)
        if (('operating_point' in known and not np.array_equal(known['operating_point'],scope))
                or ('parameters' in known and not np.array_equal(known['parameters'],setting))):
            return stop('context_mismatch','声明的工况或参数与已知采集身份不一致')
    try:
        report['timing_checks'] = [assert_frequency_consistent(r,scope) for r in (before,after)]
    except ValueError as error:
        return stop('context_mismatch',str(error))
    phases = window['sensitivity_windows']
    if (not window.get('confirmed') or not window.get('sensitivity_available')
            or not phases or not np.array_equal(window['phase'],phases[0])):
        return stop('window_review_required','试调前后需要同一个已确认并可做敏感性检查的保持期')
    bands = window.get('sensitivity_phase_bands')
    try:
        measured_before,v0 = _measure(before,phases,bands)
        measured_after,v1 = _measure(after,phases,bands)
    except ValueError as error:
        return stop('invalid_waveform',str(error))
    gain = v0-v1
    raw_before = measured_before['raw_window_band']['status']
    raw_after = measured_after['raw_window_band']['status']
    passed, failed = 'all_below_limit', 'all_above_limit'
    report.update(before=measured_before,after=measured_after,nominal_rms_gain_pp=float(gain[0]),
                  scanned_rms_gain_range_pp=[float(gain.min()),float(gain.max())],
                  best_nominal_observation=('after' if
                    (not measured_after['raw_passed'],measured_after['rms_pct']) <
                    (not measured_before['raw_passed'],measured_before['rms_pct']) else 'before'))
    coverage = [check_pulse_coverage(r,window) for r in (before,after)]
    report['pulse_coverage_checks'] = coverage
    if any(c['status']=='early_loss_while_peers_hold' for c in coverage):
        return stop('pulse_coverage_review','试调前后存在保持期覆盖异常，须先核对波形')
    intervals = measured_before['sample_interval_ns_range'] + measured_after['sample_interval_ns_range']
    interval = shared_sample_interval_ns(intervals)
    if interval is None:
        return stop('sampling_context_mismatch','试调前后采样间隔不同，需用一致的采集设置复测')
    if raw_after==passed:
        report['action'] = 'retain_current'
        return stop('observed_below_15','试调后在确认窗口和边界带内均低于 15%')
    if raw_before==passed and raw_after==failed:
        report.update(action='rollback',next_trial=dict(axis=axis,step=-step))
        return stop('lost_raw_qualification','试调前整个边界带低于 15%，试调后超出限值')
    if raw_before!=failed or raw_after!=failed:
        return stop('raw_boundary_sensitive','边界带不足或 15% 判定随选窗变化，需要先复核当前窗口')

    try:
        exact = exact_gain_band(before,after,bands)
    except ValueError as error:
        return stop('invalid_waveform',str(error))
    report['rms_gain_band'] = exact
    lo,hi = exact['range_pp']
    tolerance = 1e-9
    if lo < -tolerance and hi > tolerance:
        return stop('rms_direction_window_sensitive','共同窗口移动会翻转本次 RMS 改善方向')

    recorded_noise, recorded_full = _matched_noise(paired_calibration,scope,phases,bands,prior,current,interval)
    if repeat_margin_pp is None:
        repeat_margin_pp = recorded_noise
    if repeat_margin_pp is None:
        return stop('missing_repeatability','缺少相同工况和参数范围的同参数复采波动')
    noise = np.asarray(repeat_margin_pp,dtype=float)
    if noise.ndim==0:
        if repeat_band_margin_pp is None:
            repeat_band_margin_pp = float(noise)
        noise = np.full(len(phases),noise)
    if noise.shape!=(len(phases),) or not np.isfinite(noise).all() or np.any(noise<0):
        raise ValueError('复采波动包络须与共同窗口对应')
    report['repeat_margin_range_pp'] = [float(noise.min()),float(noise.max())]
    if repeat_band_margin_pp is None:
        repeat_band_margin_pp = recorded_full
    if repeat_band_margin_pp is not None:
        band_margin = float(repeat_band_margin_pp)
        if not np.isfinite(band_margin) or band_margin<0:
            raise ValueError('整个边界带的复采波动界须为非负有限数')
    else:
        band_margin = None
    report['repeat_band_margin_pp'] = band_margin
    better, worse = np.all(gain>noise), np.all(gain < -noise)
    if (better or worse) and band_margin is None:
        return stop('missing_full_band_repeatability','规则扫描已有一致变化，完整边界带的复采波动仍缺证据')
    if better and lo>band_margin+tolerance:
        report['action'] = 'retain_current'
        if paired_calibration:
            next_review = advise_paired(after,window,scope,current,paired_calibration)
            candidate = next_review['robust_trial']
            if candidate==dict(axis=axis,step=step):
                report.update(action='consider_continue_same_direction',next_trial=candidate)
        return stop('observed_improvement_above_variation','两次都未达到 15%，全部样本窗口的 RMS 改善超过记录的复采波动')
    if worse and hi < -band_margin-tolerance:
        report.update(action='rollback',next_trial=dict(axis=axis,step=-step))
        return stop('observed_worsening_above_variation','全部样本窗口的 RMS 变差超过记录的复采波动')
    return stop('within_observed_repeat_variation','本次 RMS 变化未稳定超过同参数复采波动')
