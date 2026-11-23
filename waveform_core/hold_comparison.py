"""Shared numerical waveform entry point, file adapter and locked scoring."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .hold_flatness import derivative_window, flatness_descriptors
from .pulse_metrics import DEFAULT_DURATION_LEVEL, locate_pulses, pulse_metrics
from .y_direction import waveform_fingerprint


@dataclass
class ComparisonRecord:
    path: Path
    pulses: list
    errors: list
    ignored_boundary_pulses: int = 0
    waveform_fingerprint: str | None = None
    measurement_context: dict | None = None


def prepare_comparison_waveform(time_s, voltage_v, *, sample_interval_s=None, path=None):
    """Prepare numerical arrays without reading files, devices or GUI state."""
    t, v = np.asarray(time_s, dtype=float), np.asarray(voltage_v, dtype=float)
    if t.ndim != 1 or v.ndim != 1 or t.shape != v.shape or len(t) < 20:
        raise ValueError('时间/电压须为等长的一维数组，且至少有20个采样点')
    dt = float(np.median(np.diff(t))) if sample_interval_s is None else float(sample_interval_s)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('采样间隔须为正的有限秒数')
    locations, _ = locate_pulses(t, v, dt)
    pulses, errors, ignored = [], [], 0
    for number, (start, stop) in enumerate(locations, 1):
        width = stop-start
        before = max(int(10e-9/dt), int(.08*width))+10
        after = int(np.ceil(.3*width))+2
        if start < before or stop+after >= len(v):
            ignored += 1
            continue
        try:
            m, crop = pulse_metrics(t, v, dt, start, stop, duration_level=DEFAULT_DURATION_LEVEL)
            if not np.isfinite([m['baseline_v'], m['amplitude_v'], m['width50_ns']]).all():
                raise ValueError('脉冲定位或参考幅度无效')
            suggested = np.array([m['duration_start_s'], m['duration_end_s']])-m['rise50_s']
            suggested *= 1e9/m['width50_ns']
            initial = tuple(suggested) if np.isfinite(suggested).all() and 0 <= suggested[0] < suggested[1] <= 1 else None
            pulse = {'number': number, 'time_ns': crop['time_s'].copy()*1e9,
                     'voltage_v': crop['voltage_v'].copy(), 'baseline_v': m['baseline_v'],
                     'amplitude_v': m['amplitude_v'], 'width50_ns': m['width50_ns'],
                     'rise10_ns': (m['rise10_s']-m['rise50_s'])*1e9,
                     'fall10_ns': (m['fall10_s']-m['rise50_s'])*1e9,
                     'legacy99_phase': initial}
            pulse['auto_hold'] = suggest_hold_window(pulse)
            pulse['suggested_phase'] = pulse['auto_hold'].get('phase_window')
            pulses.append(pulse)
        except (ValueError, IndexError, FloatingPointError) as error:
            errors.append(f'脉冲{number}：{error}')
    if not pulses:
        errors.append('没有可完整定位的正脉冲')
    return ComparisonRecord(Path(path) if path is not None else Path('memory'), pulses, errors, ignored,
                            waveform_fingerprint(t, v))


def validate_phase_window(window):
    values = np.asarray(window, dtype=float)
    if values.shape != (2,) or not np.isfinite(values).all() or not 0 <= values[0] < values[1] <= 1:
        raise ValueError('保持期必须位于上升50%至下降50%之间，且起点小于终点')
    return tuple(float(x) for x in values)


def suggest_hold_window(pulse):
    """Research edge-slope initial selection; manual confirmation is required.

    Failure does not invalidate the underlying pulse or silently fall back to a
    voltage-threshold window. Parameters describe the existing research method,
    not hardware constants or a validated cross-condition acceptance rule.
    """
    settings = {'method': 'edge_slope20', 'smoothing_ns': 3.,
                'slope_fraction': .2, 'persistence_ns': 1.}
    try:
        t, v = np.asarray(pulse['time_ns']), np.asarray(pulse['voltage_v'])
        width = pulse['width50_ns']
        r10, f10 = pulse['rise10_ns'], pulse['fall10_ns']
        if (t.ndim != 1 or t.shape != v.shape or len(t) < 5
                or not np.isfinite(t).all() or not np.isfinite(v).all()
                or not np.all(np.diff(t) > 0)
                or not np.isfinite([r10, width, f10]).all()
                or not t[0] <= r10 < 0 < width < f10 <= t[-1]):
            raise ValueError('边沿上下文不足或采样无效，请手动框选')
        window = derivative_window(t, v, r10, 0., width, f10,
                                   smoothing_ns=settings['smoothing_ns'],
                                   fraction=settings['slope_fraction'],
                                   persistence_ns=settings['persistence_ns'])
        if window['rise_peak_slope_v_per_ns'] <= 0 or window['fall_peak_slope_v_per_ns'] >= 0:
            raise ValueError('未找到有效的上升/下降边沿斜率，请手动框选')
        phase = validate_phase_window((window['start_ns']/width, window['end_ns']/width))
        flatness_descriptors(t, v, pulse['baseline_v'], window['start_ns'], window['end_ns'])
        return dict(settings, valid=True, phase_window=phase, **window)
    except (ValueError, IndexError, FloatingPointError) as error:
        return dict(settings, valid=False, error=str(error))


def score_comparison_record(record, phase_window, bins=12):
    """只用锁定窗口内的扣基线样点计算平整度。"""
    left, right = validate_phase_window(phase_window)
    scores = []
    errors = list(record.errors)
    for pulse in record.pulses:
        try:
            a, b = left*pulse['width50_ns'], right*pulse['width50_ns']
            score = flatness_descriptors(pulse['time_ns'], pulse['voltage_v'],
                                         pulse['baseline_v'], a, b, bins=bins)
            scores.append(dict(score, start_ns=a, end_ns=b, number=pulse['number']))
        except (ValueError, FloatingPointError) as error:
            errors.append(f"脉冲{pulse['number']}：{error}")
    result = {'valid': bool(scores) and not errors, 'errors': errors,
              'scores': scores, 'phase_window': (left, right), 'pulse_count': len(record.pulses)}
    if result['valid']:
        result.update(rms_median=float(np.median([s['block_rms_pct'] for s in scores])),
                      rms_max=max(s['block_rms_pct'] for s in scores),
                      tilt_median=float(np.median([s['tilt_signed_pct'] for s in scores])),
                      raw_pp_max=max(s['raw_pp_pct'] for s in scores),
                      worst_index=int(np.argmax([s['block_rms_pct'] for s in scores])),
                      width_median=float(np.median([p['width50_ns'] for p in record.pulses])))
    return result
