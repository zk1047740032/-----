"""方波调平工具。每轮保存一个共同窗口，并逐次接收实际采集。"""

import json
import sys
import tkinter as tk
from pathlib import Path
from queue import Empty, Queue
from tempfile import NamedTemporaryFile
from threading import Thread
from tkinter import filedialog, ttk

import numpy as np

from fiber1100_algorithm import evaluate_shape_record
from waveform_core.leveling import LevelingSession
from waveform_core.timing_pulses import prepare_timed_file


AXES = ("y0", "y1", "y2")
LABELS = {"y0": "肩部", "y1": "平顶前段", "y2": "平顶中段"}


class WaveformApp:
    def __init__(self, root):
        self.root = root
        self.path = self.record = self.report = self.session = self.result = None
        self.drag = self.plot_transform = None
        self.pending = Queue()
        self.loading_mode = None
        self.busy = False
        root.title("方波评价")
        root.geometry("990x720")
        root.minsize(775, 600)
        root.option_add("*Font", ("Microsoft YaHei UI", 10))
        style = ttk.Style(root)
        style.configure("Heading.TLabel", font=("Microsoft YaHei UI", 17, "bold"), foreground="#153947")
        style.configure("Summary.TLabel", font=("Microsoft YaHei UI", 11, "bold"), foreground="#153947")
        style.configure("Hint.TLabel", foreground="#50646e")

        page = ttk.Frame(root, padding=(20, 14))
        page.pack(fill="both", expand=True)
        bar = ttk.Frame(page)
        bar.pack(fill="x")
        ttk.Label(bar, text="方波调平", style="Heading.TLabel").pack(side="left")
        self.open_button = ttk.Button(bar, text="打开起始波形", command=lambda: self.choose("start"))
        self.save_button = ttk.Button(bar, text="保存记录", command=self.save)
        self.stop_button = ttk.Button(bar, text="结束本轮", command=self.stop)
        for button in (self.open_button, self.stop_button, self.save_button):
            button.pack(side="right", padx=(8, 0))
        self.filename = ttk.Label(page, text="打开示波器 CSV 或 NPZ 波形开始。", style="Hint.TLabel")
        self.filename.pack(fill="x", pady=(3, 8))

        self.summary = ttk.Label(page, text="", style="Summary.TLabel")
        self.summary.pack(fill="x")
        self.shape_note = ttk.Label(page, text="", style="Hint.TLabel", wraplength=780)
        self.shape_note.pack(fill="x", pady=(0, 5))
        preview = ttk.Frame(page)
        preview.pack(fill="x", pady=(5, 4))
        ttk.Label(preview, text="查看脉冲").pack(side="left")
        self.pulse = ttk.Combobox(preview, state="disabled", width=5)
        self.pulse.pack(side="left", padx=6)
        self.pulse.bind("<<ComboboxSelected>>", self.draw)
        self.worst_button = ttk.Button(preview, text="查看最大不平度", command=self.select_worst)
        self.worst_button.pack(side="left")
        self.window_hint = ttk.Label(preview, text="", style="Hint.TLabel")
        self.window_hint.pack(side="left", padx=8)
        self.auto_button = ttk.Button(preview, text="恢复自动框选", command=lambda: self.reframe(None))
        self.auto_button.pack(side="right")

        self.canvas = tk.Canvas(page, background="#ffffff", highlightbackground="#d7e2e6", highlightthickness=1)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self.draw)
        self.canvas.bind("<ButtonPress-1>", self.begin_drag)
        self.canvas.bind("<B1-Motion>", self.show_drag)
        self.canvas.bind("<ButtonRelease-1>", self.apply_drag)

        directions = ttk.Frame(page)
        directions.pack(fill="x", pady=(8, 3))
        self.direction_labels = []
        for i, axis in enumerate(AXES):
            directions.columnconfigure(i, weight=1, uniform="axis")
            label = ttk.Label(directions, text=f"{axis} · {LABELS[axis]}  —", justify="left", wraplength=240)
            label.grid(row=0, column=i, sticky="nw", padx=(0, 8))
            self.direction_labels.append(label)
        self.status = ttk.Label(page, text="", style="Summary.TLabel", wraplength=800, justify="left")
        self.status.pack(fill="x", pady=(4, 2))
        self.best = ttk.Label(page, text="", style="Hint.TLabel", wraplength=820, justify="left")
        self.best.pack(fill="x")
        self.notice = ttk.Label(page, text="", foreground="#935020", wraplength=800)
        self.notice.pack(fill="x")
        actions = ttk.Frame(page)
        actions.pack(fill="x", pady=(4, 0))
        self.applied_button = ttk.Button(actions, text="载入操作后的新采集", command=lambda: self.choose("applied"))
        self.repeat_button = ttk.Button(actions, text="保持设置，载入复采", command=lambda: self.choose("repeat"))
        self.applied_button.pack(side="left")
        self.repeat_button.pack(side="left", padx=10)
        self.refresh_controls()

    def refresh_controls(self):
        action = (self.result or {}).get("next_action")
        can_move = action is not None and action["kind"] in ("trial", "rollback")
        awaiting = action is not None and action["kind"] == "capture"
        if can_move:
            verb = "已回退" if action["kind"] == "rollback" else "已试调"
            self.applied_button.configure(text=f"{verb} {action['axis']} {action['step']:+d}，载入新采集")
        elif awaiting:
            self.applied_button.configure(text="载入本步新采集")
        else:
            self.applied_button.configure(text="载入操作后的新采集")
        self.open_button.configure(text="新开一轮" if self.path else "打开起始波形")
        buttons = {
            self.open_button: not awaiting,
            self.applied_button: can_move or awaiting,
            self.repeat_button: action is not None and not awaiting,
            self.stop_button: action is not None,
            self.save_button: self.result is not None,
            self.auto_button: not awaiting and self.report is not None and self.report["window"]["confirmed"],
            self.worst_button: self.report is not None and len(self.record.pulses) > 1,
        }
        for widget, enabled in buttons.items():
            widget.configure(state="normal" if enabled and not self.busy else "disabled")
        self.pulse.configure(state="readonly" if self.record and self.record.pulses and not self.busy else "disabled")

    def choose(self, mode):
        if self.busy:
            return
        if mode == "applied" and self.result["next_action"]["kind"] in ("trial", "rollback"):
            self.result = self.session.mark_applied()
            self.show_results()
            self.refresh_controls()
        titles = {"start": "选择本轮起始波形", "applied": "选择执行屏幕所示操作后的新采集",
                  "repeat": "选择保持当前设置的独立复采"}
        options = {"parent": self.root, "title": titles[mode], "filetypes": [("波形文件", "*.csv *.npz")]}
        if self.path:
            options["initialdir"] = self.path.parent
        path = filedialog.askopenfilename(**options)
        if path:
            self.open_path(path, mode=mode)

    def open_path(self, path, *, mode="start", phase=None):
        if self.busy:
            return
        if mode in ("applied", "repeat") and not (self.result or {}).get("next_action"):
            return
        self.busy = True
        self.loading_mode = mode
        self.notice.configure(text="正在评价波形…")
        self.refresh_controls()
        path = Path(path).resolve()
        Thread(target=self.work, args=(path, mode, phase), daemon=True).start()
        self.root.after(50, self.finish)

    def work(self, path, mode, phase):
        record = report = None
        try:
            record = prepare_timed_file(path)
            if mode == "start":
                session = LevelingSession()
                report = evaluate_shape_record(record, confirmed_phase=phase)
                result = session.start(record, report)
            else:
                session = self.session
                report = evaluate_shape_record(record, window=session.window)
                result = session.observe(record, report, applied=mode == "applied")
            self.pending.put((path, record, report, session, result, None))
        except Exception as error:
            self.pending.put((path, record, None, None, None, str(error) or type(error).__name__))

    def finish(self):
        try:
            path, record, report, session, result, error = self.pending.get_nowait()
        except Empty:
            self.root.after(50, self.finish)
            return
        self.busy = False
        if error:
            note = f"未接收 {path.name}。{error}"
            if self.result and self.result.get("status") == "await_capture":
                note += " 保持这一步的设置，重新载入本步新采集。"
            self.notice.configure(text=note)
            # A new baseline can still be manually framed if its automatic selection fails.
            if self.loading_mode == "start" and record is not None and record.pulses and (
                    self.session is None or self.path != path):
                self.path, self.record = path, record
                self.result = self.session = self.report = None
                self.filename.configure(text=path.name)
                for label in (self.summary, self.shape_note, self.status, self.best):
                    label.configure(text="")
                for axis, label in zip(AXES, self.direction_labels, strict=True):
                    label.configure(text=f"{axis} · {LABELS[axis]}  —")
                self.set_pulses(record.pulses[0]["number"])
                self.draw()
        else:
            self.path, self.record, self.report = path, record, report
            self.session, self.result = session, result
            self.filename.configure(text=path.name)
            self.notice.configure(text="")
            self.set_pulses(report["shape"]["representative_pulse"])
            self.show_results()
            self.draw()
        self.refresh_controls()

    def set_pulses(self, selected):
        numbers = [str(pulse["number"]) for pulse in self.record.pulses]
        self.pulse.configure(values=numbers)
        self.pulse.set(str(selected) if str(selected) in numbers else numbers[0])

    def stop(self):
        if self.session and not self.busy and self.result.get("next_action"):
            self.result = self.session.end()
            self.notice.configure(text="")
            self.show_results()
            self.refresh_controls()

    def save(self):
        if not self.result:
            return
        path = filedialog.asksaveasfilename(parent=self.root, title="保存本轮记录", defaultextension=".json",
                                            filetypes=[("调平记录", "*.json")])
        if not path:
            return
        destination = Path(path)
        temporary = None
        try:
            content = json.dumps({"window": self.session.window, "session": self.result},
                                 ensure_ascii=False, indent=2, allow_nan=False)
            if destination.resolve() == self.path or destination.suffix.lower() != ".json":
                raise ValueError("请选择 JSON 结果文件。")
            with NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent,
                                    delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(content)
            temporary.replace(destination)
            self.notice.configure(text=f"本轮记录已保存到 {destination.name}。")
        except (OSError, ValueError, TypeError) as error:
            self.notice.configure(text=f"保存失败。{error}")
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def reframe(self, phase):
        if self.path and not self.busy and (self.result or {}).get("status") != "await_capture":
            self.open_path(self.path, phase=phase)

    def select_worst(self):
        worst = max(self.report["flatness"]["pulses"], key=lambda row: row["raw_pp_pct"])
        self.pulse.set(str(worst["number"]))
        self.draw()

    def show_results(self):
        score = self.report["flatness"]
        shape = self.report["shape"]
        n = len(score["pulses"])
        bad = sum(not row["raw_passed"] for row in score["pulses"])
        verdict = "全部低于 15%" if bad == 0 else f"{bad}/{n} 个脉冲达到或超过 15%"
        self.summary.configure(text=f"最大不平度 {score['raw_max_pct']:.2f}%  ·  {verdict}"
                                    f"    RMS 中位数 {score['rms_median_pct']:.2f}%")
        shape_note = shape["reason"]
        if shape["status"] == "trial":
            shape_note = f"{len(shape['shape_failed_pulses'])}/{n} 个脉冲超出暂定形状范围。"
        if score["raw_window_band"]["status"] == "boundary_sensitive":
            shape_note += "  保持期边界变化会影响 15% 判定。"
        self.shape_note.configure(text=shape_note)
        action = self.result.get("next_action")
        for axis, label in zip(AXES, self.direction_labels, strict=True):
            row = shape["regions"].get(axis, {})
            error = row.get("error_pct")
            level = "" if error is None else f"    区域偏差 {error:+.2f}%"
            movement = "本步保持" if action else "保持"
            if action and action["axis"] == axis and action["step"]:
                movement = f"{'回退' if action['kind'] == 'rollback' else '试调'} {action['step']:+d}"
            elif self.result.get("pending_move") and self.result["pending_move"]["axis"] == axis:
                movement = f"已调 {self.result['pending_move']['step']:+d}，待复评"
            elif action is None and self.result["status"] != "flat":
                movement = "暂缓"
            label.configure(text=f"{axis} · {LABELS[axis]}    {movement}{level}")
        self.status.configure(text=self.result["message"])
        best = self.result.get("best")
        if best:
            parts = "，".join(f"{axis} {int(delta):+d}" for axis, delta in zip(AXES, best["offsets"], strict=True))
            self.best.configure(text=f"本轮最佳  {Path(best['path']).name}    相对起点 {parts}"
                                     f"    不平度 {best['raw_pct']:.2f}%    RMS {best['rms_pct']:.2f}%")
        else:
            self.best.configure(text="")

    def draw(self, _event=None):
        self.canvas.delete("all")
        self.plot_transform = None
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        if not self.record or not self.pulse.get():
            self.window_hint.configure(text="")
            return
        pulse = next(pulse for pulse in self.record.pulses if str(pulse["number"]) == self.pulse.get())
        t, v, w50 = pulse["time_ns"], pulse["voltage_v"], pulse["width50_ns"]
        xmin, xmax = max(t[0], -.18 * w50), min(t[-1], 1.18 * w50)
        indices = (t >= xmin) & (t <= xmax)
        t, v = t[indices], v[indices]
        ymin, ymax = min(v), max(v)
        if width < 160 or height < 90 or ymax <= ymin:
            return
        left, right, top, bottom = 61, width - 24, 21, height - 37
        xscale = (right - left) / (xmax - xmin)
        yscale = (bottom - top - 12) / (ymax - ymin)
        x0 = left - xmin * xscale
        self.plot_transform = x0, xscale, w50, top, bottom
        if self.report:
            a, b = self.report["window"]["phase"]
            xa, xb = x0 + a * w50 * xscale, x0 + b * w50 * xscale
            self.canvas.create_rectangle(xa, top, xb, bottom, outline="#458c87", fill="#e0f1ee")
            source = "手动窗" if self.report["window"]["confirmed"] else "自动窗"
            self.window_hint.configure(text=f"{source} {a*w50:.2f}–{b*w50:.2f} ns · 拖动改窗将重开本轮")
            action = self.result.get("next_action")
            if action and action["kind"] == "trial":
                region = self.report["shape"]["regions"][action["axis"]].get("phase_window")
                if region:
                    first, last = region
                    rx = x0 + first * w50 * xscale
                    ex = x0 + last * w50 * xscale
                    self.canvas.create_rectangle(rx, top, ex, bottom, fill="#ffedd0", outline="#b88039")
                    self.canvas.create_text((rx + ex) / 2, top + 8, text=action["axis"], fill="#754918")

        else:
            self.window_hint.configure(text="拖动曲线区域手动框选保持期")
        for fraction in (0., .25, .5, .75, 1.):
            x = left + fraction * (right - left)
            y = bottom - fraction * (bottom - top)
            self.canvas.create_line(x, top, x, bottom, fill="#e1eaed")
            self.canvas.create_line(left, y, right, y, fill="#e1eaed")
            self.canvas.create_text(x, bottom + 11, text=f"{xmin+(xmax-xmin)*fraction:.1f}", fill="#50646e")
        for level in np.linspace(ymin, ymax, 5):
            y = bottom - 6 - (level - ymin) * yscale
            self.canvas.create_text(left - 8, y, text=f"{level:.2g}", anchor="e", fill="#50646e")
        self.canvas.create_text(14, top - 7, text="V", fill="#50646e")
        self.canvas.create_text(right, bottom + 26, text="相对上升 50% 交点 (ns)", anchor="e", fill="#50646e")
        pixels = np.column_stack((x0 + t * xscale, bottom - 6 - (v - ymin) * yscale))
        self.canvas.create_line(*pixels.ravel().tolist(), fill="#174a66", width=2)

    def begin_drag(self, event):
        graph = self.plot_transform
        if (graph and not self.busy and (self.result or {}).get("status") != "await_capture"
                and graph[3] <= event.y <= graph[4]):
            self.drag = event.x

    def show_drag(self, event):
        if self.drag is not None and self.plot_transform:
            self.canvas.delete("drag")
            *_, top, bottom = self.plot_transform
            self.canvas.create_rectangle(self.drag, top, event.x, bottom, outline="#00766e", dash=(4, 3), tags="drag")

    def apply_drag(self, event):
        if self.drag is None or not self.plot_transform:
            return
        start, self.drag = self.drag, None
        self.canvas.delete("drag")
        if abs(start - event.x) < 5:
            return
        x0, scale, w50, *_ = self.plot_transform
        left, right = sorted(np.clip((np.array([start, event.x]) - x0) / (scale * w50), 0., 1.))
        if left < right:
            self.reframe((float(left), float(right)))


if __name__ == "__main__":
    root = tk.Tk()
    app = WaveformApp(root)
    if len(sys.argv) == 2:
        root.after_idle(lambda: app.open_path(Path(sys.argv[1])))
    root.mainloop()
