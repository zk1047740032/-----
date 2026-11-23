"""Offline preparation for pulses with deep internal excursions."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .waveform_io import load_waveform
from .hold_comparison import ComparisonRecord, prepare_comparison_waveform, suggest_hold_window
from .pulse_metrics import locate_pulses, pulse_metrics, runs, smooth
from .y_direction import waveform_fingerprint
from .operating_timing import observe_cadence, require_matching_frequency


def _high_islands(filtered, dt, diagnostics):
    base = diagnostics['baseline_v']
    amplitude = diagnostics['bootstrap_amplitude_v']
    if amplitude <= max(8*diagnostics['noise_v'], 1e-9):
        return []
    starts, stops = runs(filtered > base+.5*amplitude)
    # A short but clearly high fragment can be the part before or after a dip.
    # A peak only just crossing 50% can be ringing on an ordinary edge.
    minimum = max(2, int(np.ceil(1e-9/dt)))
    return [(int(a),int(b)) for a,b in zip(starts,stops,strict=True)
            if b-a >= minimum and np.max(filtered[a:b]) >= base+.7*amplitude]


def _recover_pulse(t, v, dt, group, number):
    start, stop = group[0][0], group[-1][1]
    width = stop-start
    # Match the existing edge context and boundary-pulse rule.
    before = max(int(10e-9/dt), int(.08*width))+10
    after = int(np.ceil(.3*width))+2
    if start < before or stop+after >= len(v):
        return None
    margin = int(max(80e-9, .6*(width*dt))/dt)
    lo, hi = max(0,start-margin), min(len(v),stop+margin)
    raw = v[lo:hi]
    geometry = raw.copy()
    if len(group) > 1:
        plateau = np.median(np.concatenate([v[a:b] for a,b in group]))
        pad = int(np.ceil(1e-9/dt))
        for (_, end),(next_start, _) in zip(group,group[1:]):
            a = max(start+1,end-pad)-lo
            b = min(stop-1,next_start+pad)-lo
            geometry[a:b] = plateau
    metrics, crop = pulse_metrics(t[lo:hi], geometry, dt, start-lo, stop-lo)
    if len(crop['time_s']) != len(raw):
        raise ValueError('外沿定位采用的时间轴与原始采样不一致')
    if len(group)>1 and (metrics['rise50_s'] >= t[group[0][1]-1] or metrics['fall50_s'] <= t[group[-1][0]]):
        raise ValueError('50% 交点未分别落在外部片段两侧')
    width50 = metrics['width50_ns']
    if not np.isfinite(width50) or width50 <= 0:
        raise ValueError('外侧两个 50% 交点不完整')
    pulse = dict(number=number, time_ns=crop['time_s']*1e9, voltage_v=raw.copy(),
                 baseline_v=metrics['baseline_v'], amplitude_v=metrics['amplitude_v'],
                 width50_ns=width50, rise10_ns=(metrics['rise10_s']-metrics['rise50_s'])*1e9,
                 fall10_ns=(metrics['fall10_s']-metrics['rise50_s'])*1e9,
                 legacy99_phase=None, geometry_recovered=len(group)>1,
                 timing_start_s=metrics['rise50_s'])
    pulse['auto_hold'] = suggest_hold_window(pulse)
    pulse['suggested_phase'] = pulse['auto_hold'].get('phase_window')
    return pulse


def _with_cadence(record, starts, point, voltage):
    observed = observe_cadence(starts)
    require_matching_frequency(observed,point)
    record.measurement_context = dict(record.measurement_context or {}, observed_cadence=observed,
                                      voltage_fingerprint=waveform_fingerprint((), voltage))
    return record


def _ordinary_record(t,v,dt,path,locations,point):
    record = prepare_comparison_waveform(t,v,sample_interval_s=dt,path=path)
    starts = [t[locations[p['number']-1][0]] for p in record.pulses]
    return _with_cadence(record,starts,point,v)


def _group_islands(islands, split_interval):
    groups = []
    for island in islands:
        if not groups or island[0] - groups[-1][0][0] >= split_interval:
            groups.append([])
        groups[-1].append(island)
    return groups


def prepare_timed_waveform(time_s, voltage_v, *, point=None, sample_interval_s=None, path=None):
    t = np.asarray(time_s,dtype=float)
    v = np.asarray(voltage_v,dtype=float)
    if t.ndim != 1 or t.shape != v.shape or len(t)<20:
        raise ValueError('时间和电压需要等长、至少 20 点的一维数组')
    dt = float(np.median(np.diff(t))) if sample_interval_s is None else float(sample_interval_s)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('采样间隔须为正数')
    locations, diagnostics = locate_pulses(t,v,dt)
    strong_islands = _high_islands(smooth(v,dt),dt,diagnostics) if diagnostics['bootstrap_amplitude_v']>0 else []
    ordinary = set(locations)
    islands = sorted(ordinary.union(strong_islands))
    if point is None:
        # A close pair alone cannot distinguish two pulses from one with a dip.
        # Infer geometry only when complete peer pulses supply BOTH width and period.
        close = set()
        for a, b in zip(islands, islands[1:]):
            if b[0]-a[1] <= 1.5*max(a[1]-a[0], b[1]-b[0]):
                close.update((a, b))
        if not close:
            return _ordinary_record(t,v,dt,path,locations,point)
        peers = ordinary - close
        # Peers separated by an ambiguous pulse may be two or more cycles apart.
        consecutive = [b[0]-a[0] for a,b in zip(islands,islands[1:]) if a in peers and b in peers]
        message = '相邻高电平片段的外沿有歧义，请提供包含更多完整脉冲的采集。'
        if len(consecutive) < 2:
            raise ValueError(message)
        period = float(np.median(consecutive))
        widths = np.asarray([b-a for a,b in peers])
        width = float(np.median(widths))
        if (width >= .4*period or np.max(np.abs(widths/width-1)) > .15
                or np.max(np.abs(np.array(consecutive)/period-1)) > .01):
            raise ValueError(message)
        # Each separated fragment group must match one peer pulse's phase and width.
        groups = _group_islands(islands, .5*period)
        anchor = min(a for a, _ in peers)
        for group in groups:
            if not any(island in close for island in group):
                continue
            cycle = (group[0][0]-anchor)/period
            span = group[-1][1] - group[0][0]
            if (len(group) == 1 or abs(cycle-round(cycle))*period > .25*width
                    or abs(span/width-1) > .20):
                raise ValueError(message)
        result = ComparisonRecord(Path(path) if path is not None else Path('memory'), [], [],
                                  waveform_fingerprint=waveform_fingerprint(t,v))
        for i, group in enumerate(groups, 1):
            if len(group) == 1 and group[0] not in ordinary:
                continue
            try:
                pulse = _recover_pulse(t, v, dt, group, i)
                if pulse is None:
                    result.ignored_boundary_pulses += 1
                else:
                    result.pulses.append(pulse)
            except (ValueError, IndexError, FloatingPointError) as error:
                result.errors.append(f'脉冲 {i}：{error}')
        if not result.pulses:
            result.errors.append('没有完整的脉冲可供评价')
        return _with_cadence(result,[p['timing_start_s'] for p in result.pulses], None, v)
    timing = np.asarray(point,dtype=float)
    if timing.shape != (2,) or not np.isfinite(timing).all() or np.any(timing <= 0):
        raise ValueError('工况需要两个正数，频率 kHz 和脉宽 ns')
    require_matching_frequency(observe_cadence([t[a] for a,_ in locations]),timing)
    period_ns = 1e6/timing[0]
    nominal_width_ns = timing[1]
    groups = _group_islands(islands, .5*period_ns/(dt*1e9))
    if all(len(group)==1 for group in groups):
        return _ordinary_record(t,v,dt,path,locations,point)
    if nominal_width_ns >= .4*period_ns:
        raise ValueError('高电平片段接近且占空比过高，无法由低占空比时序可靠定位')
    result = ComparisonRecord(Path(path) if path is not None else Path('memory'), [], [],
                              waveform_fingerprint=waveform_fingerprint(t,v))
    for i,group in enumerate(groups,1):
        if len(group)==1 and group[0] not in ordinary:
            continue
        try:
            span_ns = (group[-1][1]-group[0][0])*dt*1e9
            if len(group)>1:
                longest_ns = max(b-a for a,b in group)*dt*1e9
                if not .65*nominal_width_ns <= span_ns <= 1.35*nominal_width_ns:
                    raise ValueError('高电平片段跨度与设定脉宽相差过大，外沿有歧义')
                if longest_ns >= .9*nominal_width_ns:
                    raise ValueError('一个片段已接近设定脉宽，周边片段可能来自振铃或附加脉冲')
            pulse = _recover_pulse(t,v,dt,group,i)
            if pulse is None:
                result.ignored_boundary_pulses += 1
            else:
                result.pulses.append(pulse)
        except (IndexError, ValueError, FloatingPointError) as error:
            result.errors.append(f'脉冲 {i}：{error}')
    if not result.pulses:
        result.errors.append('无完整的脉冲可供评价')
    return _with_cadence(result,[p['timing_start_s'] for p in result.pulses],point,v)


def prepare_timed_file(path, *, point=None):
    t,v,interval = load_waveform(path)
    return prepare_timed_waveform(t,v,point=point,sample_interval_s=interval,path=path)
