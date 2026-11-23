"""Bounds for raw peak-to-peak/mean over all sample windows in a phase band."""
from __future__ import annotations

import numpy as np


def pulse_range_in_band(pulse, bands):
    """Enumerate attained slice indices through prefix sums and extrema."""
    t = np.asarray(pulse['time_ns'])
    z = np.asarray(pulse['voltage_v'])-pulse['baseline_v']
    start, stop = np.asarray(bands)*pulse['width50_ns']
    first, left = np.searchsorted(t,start,side='left')
    right, last = np.searchsorted(t,stop,side='right')
    core = z[left:right]
    if len(core)<24:
        return None

    # Every choice contains the common core. It can also contain a suffix of
    # the left extension and a prefix of the right extension.
    a = z[first:left][::-1]
    b = z[right:last]
    l_sum = np.r_[0.,np.cumsum(a)]
    r_sum = np.r_[0.,np.cumsum(b)]
    l_min = np.r_[core.min(),np.minimum.accumulate(a)]
    r_min = np.r_[core.min(),np.minimum.accumulate(b)]
    l_max = np.r_[core.max(),np.maximum.accumulate(a)]
    r_max = np.r_[core.max(),np.maximum.accumulate(b)]
    lo, hi = float('inf'), float('-inf')
    # Limit peak memory for very wide bands while enumerating every choice.
    rows = max(1,131072//len(r_sum))
    r_count = np.arange(len(r_sum))
    for i in range(0,len(l_sum),rows):
        stop_i = min(i+rows,len(l_sum))
        take = slice(i,stop_i)
        count = len(core)+np.arange(i,stop_i)[:,None]+r_count
        mean = (core.sum()+l_sum[take,None]+r_sum)/count
        if np.any(mean<=0):
            return None
        smallest = np.minimum(np.minimum(l_min[take,None],r_min),core.min())
        largest = np.maximum(np.maximum(l_max[take,None],r_max),core.max())
        ranges = 100.*(largest-smallest)/mean
        lo,hi = min(lo,float(ranges.min())),max(hi,float(ranges.max()))
    return lo,hi


def raw_band_report(pulses, bands, sampled_flip, *, observed_pass=False):
    report = dict(status='insufficient_band', bounds_pct=None)
    if not pulses or bands is None:
        return report
    values = [pulse_range_in_band(p,bands) for p in pulses]
    if any(r is None for r in values):
        return report
    low,high = np.max(values,axis=0)
    # max(min(pulse)) bounds the worst-pulse score from below. The upper bound
    # max(max(pulse)) is attained even though all pulses share the same phase.
    # Separate near-limit rounding of accumulated sums from the strict nominal
    # check using its directly computed mean.
    tol = 1e-9
    proven_pass = observed_pass or (len(pulses)==1 and low<15.-tol)
    if sampled_flip or (proven_pass and high>15.+tol):
        status = 'boundary_sensitive'
    elif high<15.-tol:
        status = 'all_below_limit'
    elif low>15.+tol:
        status = 'all_above_limit'
    else:
        status = 'unresolved_within_band'
    return dict(status=status, bounds_pct=[float(low),float(high)],
                lower_bound_exact=len(pulses)==1)
