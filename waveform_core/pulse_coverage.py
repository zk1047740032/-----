"""Check for a pulse already low while most peers sustain their hold."""
from __future__ import annotations

import numpy as np


MIN_OVERLAP_NS = 1.0


def check_pulse_coverage(record, window):
    pulses = record.pulses
    report = dict(status='insufficient_peer_pulses', pulses=len(pulses), shortened_pulses=[])
    if len(pulses) < 3:
        return report
    widths = np.asarray([p['width50_ns'] for p in pulses])
    bands = window.get('sensitivity_phase_bands') if window.get('sensitivity_available') else None
    start, end = (bands[0][1], bands[1][0]) if bands is not None else window['phase']
    peer_starts, peer_ends = start * widths, end * widths
    # Past the median window end, a strict majority cannot still be in the window.
    latest_possible = float(np.median(peer_ends))
    majority = len(pulses) // 2 + 1
    report.update(status='no_early_loss_observed', support_required=majority,
                  peer_hold_phase=[float(start), float(end)])
    for pulse in pulses:
        fall10 = pulse.get('fall10_ns')
        if fall10 is None or not np.isfinite(fall10) or fall10 >= latest_possible:
            continue
        pulse_time = pulse['time_ns']
        mask = (pulse_time >= fall10) & (pulse_time <= latest_possible)
        trial_time = pulse_time[mask]
        if len(trial_time) < 2:
            continue
        low = pulse['voltage_v'][mask] < pulse['baseline_v'] + 0.5 * pulse['amplitude_v']
        support = np.zeros(len(trial_time), dtype=int)
        for peer, peer_start, peer_end in zip(pulses, peer_starts, peer_ends, strict=True):
            if peer is pulse:
                continue
            peer_time = peer['time_ns']
            within = ((trial_time >= max(peer_start, peer_time[0]))
                      & (trial_time <= min(peer_end, peer_time[-1])))
            if not np.any(within):
                continue
            q = trial_time[within]
            lo = np.searchsorted(peer_time, q, side='right') - 1
            hi = np.searchsorted(peer_time, q, side='left')
            volts = peer['voltage_v']
            half = peer['baseline_v'] + 0.5 * peer['amplitude_v']
            support[within] += (volts[lo] >= half) & (volts[hi] >= half)
        found = low & (support >= majority)
        edges = np.diff(np.r_[False, found, False].astype(int))
        starts = np.flatnonzero(edges == 1)
        stops = np.flatnonzero(edges == -1)
        runs = [(a, b) for a, b in zip(starts, stops, strict=True)
                if trial_time[b - 1] - trial_time[a] >= MIN_OVERLAP_NS]
        if runs:
            a, b = max(runs, key=lambda span: trial_time[span[1] - 1] - trial_time[span[0]])
            report['shortened_pulses'].append(dict(
                number=pulse['number'], observed_fall10_ns=float(fall10),
                w50_ns=float(pulse['width50_ns']), overlap_start_ns=float(trial_time[a]),
                overlap_end_ns=float(trial_time[b - 1]), minimum_high_peers=int(support[a:b].min()),
            ))
    if report['shortened_pulses']:
        report['status'] = 'early_loss_while_peers_hold'
    return report
