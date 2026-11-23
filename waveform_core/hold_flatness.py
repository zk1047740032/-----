"""Experimental hold-window geometry and flatness descriptors, not acceptance.

Boundaries are descriptive geometric estimates, not verified physical boundaries.
No learned historical voltage envelope, y directions, or production thresholds.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def derivative_window(time_ns, voltage, rise10_ns, rise50_ns, fall50_ns, fall10_ns,
                      *, smoothing_ns=3., fraction=.2, persistence_ns=1.):
    """Find the end/start of dominant fast edge slopes, retaining internal dips.

    The fraction is relative to slope, not voltage. It remains a research
    heuristic; sensitivity must be checked, not treated as a physical constant.
    """
    t, v = np.asarray(time_ns,float), np.asarray(voltage,float)
    dt = float(np.median(np.diff(t)))
    count = max(5,int(round(smoothing_ns/dt)))
    count += count%2 == 0
    if count > len(t):
        raise ValueError("Smoothing window exceeds samples")
    derivative = savgol_filter(v,count,3,deriv=1,delta=dt,mode="interp")
    rise = np.flatnonzero((t>=rise10_ns)&(t<=rise50_ns))
    fall = np.flatnonzero((t>=fall50_ns)&(t<=fall10_ns))
    if not len(rise) or not len(fall):
        raise ValueError("Missing edge contexts")
    rp = int(rise[np.argmax(derivative[rise])])
    fp = int(fall[np.argmin(derivative[fall])])
    center = int(np.searchsorted(t,(rise50_ns+fall50_ns)/2))
    persistence = max(1,int(np.ceil(persistence_ns/dt)))
    left = next((i for i in range(rp+1,center-persistence+1)
                 if np.all(derivative[i:i+persistence] <= fraction*derivative[rp])),None)
    right = next((i for i in range(fp-1,center+persistence-2,-1)
                  if np.all(derivative[i-persistence+1:i+1] >= fraction*derivative[fp])),None)
    if left is None or right is None or right<=left:
        raise ValueError("No persistent slow region between fast edges")
    return {"start_ns":float(t[left]),"end_ns":float(t[right]),
            "rise_peak_slope_v_per_ns":float(derivative[rp]),
            "fall_peak_slope_v_per_ns":float(derivative[fp])}


def flatness_descriptors(time_ns, voltage, baseline_v, start_ns, end_ns, bins=12):
    """Raw nonuniformity plus time-block shape descriptors, all in percent.

    The new RMS compares block means with ONE constant mean. No detrending:
    a smooth tilted top must still score worse. Raw spikes stay in raw_pp_pct.
    The score is not directly comparable with the legacy 15% limit.
    """
    t, v = np.asarray(time_ns,float), np.asarray(voltage,float)
    if not np.isfinite([baseline_v,start_ns,end_ns]).all() or end_ns<=start_ns:
        raise ValueError("Invalid hold interval")
    mask = (t>=start_ns)&(t<=end_ns)
    tx, z = t[mask], v[mask]-baseline_v
    if len(z)<2*bins or not np.isfinite(z).all():
        raise ValueError("Insufficient finite hold samples")
    mean = float(np.mean(z))
    if mean<=0:
        raise ValueError("Nonpositive baseline-subtracted hold mean")
    # Uniform scope samples: contiguous equal-point blocks approximate equal
    # duration to one sample, avoiding floating ties at ideal time-bin edges.
    # Never discard low/high interior points.
    levels = np.array([part.mean() for part in np.array_split(z,bins)])
    slope = float(np.polyfit((tx-start_ns)/(end_ns-start_ns),z,1)[0])
    return {"raw_pp_pct":float(np.ptp(z)/mean*100),
            "block_rms_pct":float(np.sqrt(np.mean((levels-mean)**2))/mean*100),
            "block_span_pct":float(np.ptp(levels)/mean*100),
            "block_max_abs_deviation_pct":float(np.max(np.abs(levels-mean))/mean*100),
            "tilt_signed_pct":slope/mean*100,"tilt_abs_pct":abs(slope)/mean*100,
            "raw_rms_pct":float(np.std(z)/mean*100),
            "duration_ns":float(end_ns-start_ns),"mean_above_baseline_v":mean,
            "samples":len(z),"bins":bins,"block_levels_v":levels.tolist()}
