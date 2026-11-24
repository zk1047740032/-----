# AGENTS.md

## 项目入口

- 项目在 Windows 上使用 Python 3.14，`waveform_app.py` 是 Tkinter 界面入口。
- `fiber1100_algorithm.py` 提供 `evaluate_waveform()` 和 `run_leveling()`。核心逻辑位于 `waveform_core/`，参考和标定数据位于 `waveform_core/data/`。
- 使用说明见 `README.md`，需求和设备协议见 `docs/`。

## 评价与调平

- 同一轮复评沿用起始波形的保持期和红蓝四区。红蓝偏差须在允许范围内，原始功率不平度须严格小于所设上限。默认红蓝允许偏差 7%，不平度上限 15%。
- 每步只改变一个 y，步长固定为 10。试调和回退后均用新采集的波形复评。

## 验证

改动后运行离线测试。涉及设备的改动，注明实际完成的验证范围。

```powershell
uv run --python 3.14 --with-requirements requirements.txt python -B -m unittest discover -s tests -v
```
