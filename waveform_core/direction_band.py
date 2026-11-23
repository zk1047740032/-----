"""Check directional stability over the sample selections in a phase band."""
from __future__ import annotations

from dataclasses import replace
from itertools import product

import numpy as np

from .y_direction import recommend_y_direction


def sample_endpoints(samples, band, side):
    lo,hi = band
    interior = samples[(samples>=lo)&(samples<=hi)]
    after = np.inf if side=='left' else -np.inf
    points = np.r_[lo,hi,interior,np.nextafter(interior,after)]
    hull = points[(points>=lo)&(points<=hi)]
    indices = np.searchsorted(samples,hull,side=side)
    _, representative = np.unique(indices,return_index=True)
    return hull[representative]


def discrete_phase_windows(pulse, bands):
    box = np.asarray(bands,dtype=float)
    if (box.shape!=(2,2) or not np.isfinite(box).all()
            or not 0<=box[0,0]<=box[0,1]<box[1,0]<=box[1,1]<=1):
        raise ValueError('边界带需要两个不重叠且有效的起止范围')
    phase = np.asarray(pulse['time_ns'])/pulse['width50_ns']
    return product(sample_endpoints(phase,box[0],'left'),sample_endpoints(phase,box[1],'right'))


def check_direction_band(record, bands, wanted, *, point=None, parameters=None, model=None):
    proofs = {axis:dict(checked=0,stable=True,minimum_gain_pp=None,conflict=None) for axis in wanted}
    active = dict(wanted)
    for pulse in record.pulses:
        only = replace(record,pulses=[pulse])
        for left,right in discrete_phase_windows(pulse,bands):
            decision = recommend_y_direction(only,(left,right),operating_point=point,
                                             parameters=parameters,model=model)
            for axis,desired in list(active.items()):
                result = proofs[axis]
                result['checked'] += 1
                actual = decision['directions'][axis]
                ends = decision['pulses']
                gain = ends[0]['per_parameter'][axis]['worst_model_improvement_pp'] if ends else None
                if actual!=desired or (desired!='hold' and (gain is None or not np.isfinite(gain) or gain<=0)):
                    result.update(stable=False,conflict=dict(pulse=pulse['number'],
                                  phase=[float(left),float(right)],direction=actual,gain_pp=gain))
                    del active[axis]
                elif gain is not None and np.isfinite(gain):
                    old = result['minimum_gain_pp']
                    result['minimum_gain_pp'] = gain if old is None else min(old,gain)
            if not active:
                return proofs
    return proofs
