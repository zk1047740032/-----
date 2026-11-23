"""读取带物理单位的示波器 CSV 或数组 NPZ。"""

import csv
from pathlib import Path

import numpy as np


def load_waveform(path):
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            t, v = data["time_s"], data["voltage_v"]
        return t, v, None
    if path.suffix.lower() != ".csv":
        raise ValueError("支持示波器CSV、time_s/voltage_v两列CSV或NPZ")
    metadata, skiprows, header = {}, None, None
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for line, fields in enumerate(csv.reader(stream), 1):
            if fields and fields[0].strip() in ("TIME", "time_s"):
                header, skiprows = [x.strip() for x in fields], line
                break
            if len(fields) >= 2:
                metadata[fields[0]] = fields[1]
            if line > 100:
                break
    if header == ["time_s", "voltage_v"]:
        dt = None
    elif header == ["TIME", "CH2"]:
        if metadata.get("Horizontal Units") != "s" or metadata.get("Vertical Units") != "V":
            raise ValueError("示波器CSV的单位必须明确为s和V")
        dt = float(metadata["Sample Interval"])
    else:
        raise ValueError("CSV表头须为time_s,voltage_v，或带s/V元数据的TIME,CH2")
    data = np.loadtxt(path, delimiter=",", skiprows=skiprows, ndmin=2, encoding="utf-8-sig")
    if data.shape[1] != 2:
        raise ValueError("CSV必须恰好包含时间、电压两列")
    if "Record Length" in metadata and len(data) != int(float(metadata["Record Length"])):
        raise ValueError("CSV实际点数与Record Length不一致")
    return data[:, 0], data[:, 1], dt
