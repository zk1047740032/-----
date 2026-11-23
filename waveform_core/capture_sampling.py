"""Sample intervals observed in the original pulse time arrays."""
from __future__ import annotations

import numpy as np


def pulse_sample_intervals_ns(record):
    return [float(np.median(np.diff(pulse['time_ns']))) for pulse in record.pulses]


def shared_sample_interval_ns(intervals):
    values = np.asarray(intervals, dtype=float)
    if not values.size or not np.isfinite(values).all() or np.min(values) <= 0:
        return None
    if not np.isclose(np.min(values), np.max(values), rtol=1e-6, atol=0.):
        return None
    return float(np.median(values))
