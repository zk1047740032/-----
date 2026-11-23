"""Exact sampled RMS range for two captures sharing a phase band."""
from __future__ import annotations

from itertools import combinations

import numpy as np
from .capture_sampling import pulse_sample_intervals_ns, shared_sample_interval_ns

BINS = 12


def phase_transitions(pulse, lower, upper, side):
    t = np.asarray(pulse['time_ns'])
    width = pulse['width50_ns']
    a,b = np.searchsorted(t,np.asarray([lower,upper])*width,side=side)
    samples = t[a:b]
    if not len(samples):
        return np.empty(0)

    # For nonnegative finite floats, the bits are monotone with the value.
    # Find the first floating phase at which each sample changes membership
    # under the original evaluator's multiply-and-compare inclusion rule.
    p = np.full(len(samples),np.float64(lower+0.).view(np.uint64),dtype=np.uint64)
    q = np.full(len(samples),np.float64(upper).view(np.uint64),dtype=np.uint64)
    while np.any(p<q):
        middle = p+(q-p)//2
        endpoint = middle.view(np.float64)*width
        changed = endpoint>samples if side=='left' else endpoint>=samples
        p = np.where(changed,p,middle+1)
        q = np.where(changed,middle,q)
    return p.view(np.float64)


def _pulse_rms_matrix(pulse, prefix, starts, stops):
    width = pulse['width50_ns']
    time = pulse['time_ns']
    left = np.searchsorted(time,starts*width,side='left')[:,None]
    right = np.searchsorted(time,stops*width,side='right')
    counts = right-left
    if np.any(counts<2*BINS):
        raise ValueError('存在可达保持期的样本数不足')
    mean = (prefix[right]-prefix[left])/counts
    if np.any(mean<=0) or not np.isfinite(mean).all():
        raise ValueError('存在可达保持期的扣基线均值无效')
    # This is np.array_split's rule: the first remainder blocks get one extra point.
    small, extra = np.divmod(counts,BINS)
    total = np.zeros_like(mean)
    previous = left
    for i in range(1,BINS+1):
        boundary = left+small*i+np.minimum(extra,i)
        averages = (prefix[boundary]-prefix[previous])/(boundary-previous)
        total += np.square(averages-mean)
        previous = boundary
    return 100.*np.sqrt(total/BINS)/mean


def exact_gain_band(before, after, bands):
    box = np.asarray(bands,dtype=float)
    if (box.shape!=(2,2) or not np.isfinite(box).all()
            or not 0<=box[0,0]<=box[0,1]<box[1,0]<=box[1,1]<=1):
        raise ValueError('复评需要有效且不交叠的起止边界带')
    if before.errors or after.errors or not before.pulses or not after.pulses:
        raise ValueError('复评需要两条完整可评分的记录')
    pulses = before.pulses+after.pulses
    points = [np.unique(np.r_[lo,*(phase_transitions(p,lo,hi,side) for p in pulses)])
              for (lo,hi),side in zip(box,('left','right'),strict=True)]
    starts,stops = points
    prefix = [np.r_[0.,np.cumsum(p['voltage_v']-p['baseline_v'])] for p in pulses]
    # Chunk the shared phase grid. A single row is processed at a time for
    # wide bands with numerous pulses, keeping memory bounded.
    chunk = max(1,65536//(len(stops)*len(pulses)))
    cross = len(before.pulses)
    low,high = float('inf'),float('-inf')
    for i in range(0,len(starts),chunk):
        choose = starts[i:i+chunk]
        values = [_pulse_rms_matrix(p,acc,choose,stops)
                  for p,acc in zip(pulses,prefix,strict=True)]
        gain = np.median(values[:cross],axis=0)-np.median(values[cross:],axis=0)
        low,high = min(low,float(gain.min())),max(high,float(gain.max()))
    return dict(range_pp=[low,high],window_count=int(len(starts)*len(stops)))


def same_setting_band_margin(groups, bands):
    """Upper envelope of observed same-setting capture-to-capture RMS change."""
    groups = [list(group) for group in groups]
    interval = shared_sample_interval_ns([
        dt for group in groups for record in group for dt in pulse_sample_intervals_ns(record)
    ])
    if interval is None:
        raise ValueError('同参数复采波动需要一致的采样间隔')
    worst = 0.
    count = 0
    for records in groups:
        for a,b in combinations(records,2):
            if not a.waveform_fingerprint or not b.waveform_fingerprint or a.waveform_fingerprint==b.waveform_fingerprint:
                raise ValueError('同参数复采证据需要独立文件身份')
            result = exact_gain_band(a,b,bands)
            worst = max(worst,*(abs(d) for d in result['range_pp']))
            count += 1
    if not count:
        raise ValueError('缺少成对的同参数复采')
    return dict(phase_bands=bands,maximum_difference_pp=worst,same_setting_pair_count=count,
                sample_interval_ns=interval)
