"""Measure regular pulse cadence independently of declared settings."""
from __future__ import annotations

import numpy as np


# Gross inconsistency check for the offline prototype. The archived spacing
# and nominal-frequency offsets are far smaller, with no hardware tolerance set.
CHECK_FRACTION = .01


def observe_cadence(starts_s):
    starts = np.asarray(starts_s,dtype=float)
    gaps = np.diff(starts)
    report = dict(status='insufficient_intervals', interval_count=int(len(gaps)),
                  observed_frequency_khz=None,period_ns=None,maximum_period_deviation_pct=None)
    if len(gaps)==0 or np.any(gaps<=0) or not np.isfinite(gaps).all():
        return report
    median = float(np.median(gaps))
    spread = float(np.max(np.abs(gaps/median-1.)))
    report.update(observed_frequency_khz=1e-3/median,period_ns=median*1e9,
                  maximum_period_deviation_pct=100.*spread)
    if len(gaps)>=2:
        report['status'] = 'regular' if spread<CHECK_FRACTION else 'irregular'
    return report


def frequency_check(observed, declared_khz):
    report = dict(observed)
    if declared_khz is None or observed.get('status')!='regular':
        report['declared_check'] = 'unverified'
        return report
    freq = float(declared_khz)
    measured = observed['observed_frequency_khz']
    deviation = abs(measured/freq-1.)
    report.update(declared_frequency_khz=freq, deviation_pct=100.*deviation,
                  declared_check='conflict' if deviation>CHECK_FRACTION else 'matches_observed_band')
    multiple = freq/measured
    integer = round(multiple)
    if report['declared_check']=='conflict' and integer>1 and abs(multiple/integer-1.)<CHECK_FRACTION:
        report['possible_missing_or_subharmonic_pulses'] = True
    return report


def require_matching_frequency(observed, operating_point):
    if operating_point is not None:
        point = np.asarray(operating_point,dtype=float)
        if point.shape!=(2,) or not np.isfinite(point).all() or np.any(point<=0):
            raise ValueError('工况需要有效的频率 kHz 和脉宽 ns')
    check = frequency_check(observed,None if operating_point is None else point[0])
    if check['declared_check']=='conflict':
        raise ValueError(f"声明频率 {check['declared_frequency_khz']:g} kHz 与观察到的脉冲频率 "
                         f"{check['observed_frequency_khz']:.3f} kHz 不符，请核对采集或漏脉冲")
    return check


def assert_frequency_consistent(record, operating_point):
    known = record.measurement_context or {}
    report = require_matching_frequency(known.get('observed_cadence',{}),operating_point)
    if record.pulses:
        width = float(np.median([p['width50_ns'] for p in record.pulses]))
        report['observed_w50_median_ns'] = width
        if operating_point is not None:
            report['declared_width_ns'] = float(operating_point[1])
            report['w50_minus_declared_width_ns'] = width-report['declared_width_ns']
    return report
