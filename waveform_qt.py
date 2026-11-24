"""Windows desktop frontend for simulated square-wave evaluation."""

from datetime import datetime
from pathlib import Path
import sys
from threading import Event

import pyqtgraph as pg
from PySide6.QtCore import QStandardPaths, QThread, Qt, Signal
from PySide6.QtGui import QFont
from PySide6 import QtWidgets as W
from serial.tools.list_ports import comports

from waveform_core.reference_alignment import DEFAULT_ALIGNMENT_TOLERANCE_PCT
from waveform_simulation import (
    DEFAULT_FREQUENCY,
    DEFAULT_RAW_LIMIT,
    DEFAULT_WIDTH,
    SimulationSettings,
    run_simulation,
    scan_axis,
)


STYLE = """
QMainWindow { background: #eff3f8; }
QWidget { color: #24384a; font-size: 13px; }
QWidget#controlFields { background: #ffffff; }
QFrame[card="true"] {
    background: #ffffff; border: 1px solid #dbe4ee; border-radius: 12px;
}
QLabel { background: transparent; }
QLabel[role="title"] { color: #132b40; font-size: 24px; font-weight: 700; }
QLabel[role="heading"] { color: #132b40; font-size: 16px; font-weight: 700; }
QLabel[role="section"] { color: #355674; font-weight: 700; }
QLabel[role="muted"] { color: #5b7086; }
QLabel[role="value"] { color: #132b40; font-size: 20px; font-weight: 700; }
QLabel[role="mode"] {
    color: #075a65; background: #e0f2f1; border: 1px solid #bce2df;
    border-radius: 12px; padding: 5px 12px; font-weight: 700;
}
QLabel[role="verdict"] {
    color: #3d5770; background: #edf2f7; border-radius: 9px; padding: 5px 10px;
    font-weight: 700;
}
QLabel[role="verdict"][state="pass"] { color: #17614c; background: #ddf2e9; }
QLabel[role="verdict"][state="fail"] { color: #a13631; background: #fee9e7; }
QLabel[role="verdict"][state="review"] { color: #79551b; background: #fff1d2; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPushButton {
    background: #f7f9fc; border: 1px solid #d8e3ee; border-radius: 6px;
    padding: 4px 8px; min-height: 22px;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QPushButton:focus {
    border-color: #21788b;
}
QPushButton:hover { background: #e9f1f6; }
QPushButton[primary="true"] { background: #087486; color: white; border-color: #087486; font-weight: 700; }
QPushButton[primary="true"]:hover { background: #066473; }
QPushButton:disabled, QPushButton[primary="true"]:disabled {
    background: #edf2f6; color: #708296; border-color: #e3eaf1;
}
QProgressBar { background: #e9eff6; border: none; border-radius: 3px; }
QProgressBar::chunk { background: #0b8191; border-radius: 3px; }
QPlainTextEdit { background: #f5f8fc; border: 1px solid #e2eaf2; border-radius: 7px; padding: 5px; }
QScrollArea { background: transparent; border: none; }
QSplitter::handle { background: #eff3f8; height: 10px; }
"""


def label(text, role=None):
    widget = W.QLabel(text)
    if role:
        widget.setProperty("role", role)
    return widget


def card():
    frame = W.QFrame()
    frame.setProperty("card", True)
    return frame


class SimulationThread(QThread):
    directory_ready = Signal(object)
    measurement_ready = Signal(object)

    def __init__(self, settings, parent):
        super().__init__(parent)
        self.settings = settings
        self.cancelled = Event()
        self.outcome = None

    def run(self):
        self.outcome = run_simulation(self.settings, self.cancelled,
                                      self.directory_ready.emit, self.measurement_ready.emit)


class WaveformWindow(W.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("方波自动化 · 波形评价")
        self.worker = None
        self.closing = False
        self.setStyleSheet(STYLE)
        self.setMinimumSize(820, 480)

        central = W.QWidget()
        self.setCentralWidget(central)
        root = W.QVBoxLayout(central)
        root.setContentsMargins(18, 12, 18, 14)
        root.setSpacing(10)
        header = W.QHBoxLayout()
        header.addWidget(label("方波自动化", "title"))
        header.addStretch()
        header.addWidget(label("模拟模式", "mode"))
        root.addLayout(header)

        splitter = W.QSplitter(Qt.Orientation.Vertical)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter, 1)
        upper = W.QWidget()
        grid = W.QHBoxLayout(upper)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(12)
        grid.addWidget(self.build_controls())
        grid.addWidget(self.build_waveform(), 1)
        splitter.addWidget(upper)
        splitter.addWidget(self.build_log())
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([530, 155])

        screen = self.screen().availableGeometry()
        self.resize(min(1240, screen.width() - 36), min(790, screen.height() - 52))
        self.clear_measurement()
        self.set_busy(False)
        self.append_log("等待开始模拟采集。")

    def build_controls(self):
        panel = card()
        panel.setFixedWidth(348)
        layout = W.QVBoxLayout(panel)
        layout.setContentsMargins(13, 12, 13, 12)
        layout.setSpacing(8)
        layout.addWidget(label("控制面板", "heading"))
        scroll = W.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        layout.addWidget(scroll, 1)

        self.form = W.QWidget()
        self.form.setObjectName("controlFields")
        scroll.setWidget(self.form)
        fields = W.QVBoxLayout(self.form)
        fields.setContentsMargins(1, 2, 3, 3)
        fields.setSpacing(7)
        fields.addWidget(label("连接与地址", "section"))

        connection = W.QFormLayout()
        connection.setHorizontalSpacing(9)
        connection.setVerticalSpacing(6)
        self.scope_ip = W.QLineEdit(placeholderText="例如 192.168.1.10")
        self.scope_ip.setAccessibleName("示波器 IP")
        connection.addRow("示波器 IP", self.scope_ip)
        self.board_port = W.QComboBox()
        self.board_port.setAccessibleName("脉冲板口")
        self.board_port.setMinimumWidth(0)
        self.board_port.setSizePolicy(W.QSizePolicy.Policy.Ignored, W.QSizePolicy.Policy.Fixed)
        refresh = W.QPushButton("刷新")
        refresh.clicked.connect(self.refresh_ports)
        connection.addRow("脉冲板口", self.field_with_button(self.board_port, refresh))
        self.refresh_ports()
        documents = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation)
        self.destination = W.QLineEdit(str((Path(documents) if documents else Path.home()) / "方波评价结果"))
        self.destination.setAccessibleName("保存地址")
        browse = W.QPushButton("浏览")
        browse.clicked.connect(self.choose_directory)
        connection.addRow("保存地址", self.field_with_button(self.destination, browse))
        fields.addLayout(connection)

        fields.addWidget(label("算法参数", "section"))
        thresholds = W.QFormLayout()
        thresholds.setHorizontalSpacing(9)
        thresholds.setVerticalSpacing(6)
        self.alignment = self.percentage(DEFAULT_ALIGNMENT_TOLERANCE_PCT, minimum=0.0)
        self.raw_limit = self.percentage(DEFAULT_RAW_LIMIT, minimum=0.01)
        thresholds.addRow("红蓝高度允许偏差 ±", self.alignment)
        thresholds.addRow("原始不平度上限 <", self.raw_limit)
        fields.addLayout(thresholds)

        fields.addWidget(label("测试参数", "section"))
        test_grid = W.QGridLayout()
        test_grid.setHorizontalSpacing(8)
        test_grid.setVerticalSpacing(5)
        test_grid.setColumnStretch(1, 1)
        test_grid.setColumnStretch(2, 1)
        test_grid.addWidget(label("重频 (kHz)", "muted"), 0, 1)
        test_grid.addWidget(label("脉宽 (ns)", "muted"), 0, 2)
        self.frequencies, self.widths = [], []
        for row, name in enumerate(("起始", "终止", "步进"), 1):
            test_grid.addWidget(label(name), row, 0)
            for column, (axis, defaults, widgets) in enumerate((
                ("重频", DEFAULT_FREQUENCY, self.frequencies), ("脉宽", DEFAULT_WIDTH, self.widths)
            ), 1):
                spin = W.QSpinBox(minimum=1, maximum=999_999, value=defaults[row - 1])
                spin.setAccessibleName(f"{name}{axis}")
                test_grid.addWidget(spin, row, column)
                widgets.append(spin)
        fields.addLayout(test_grid)
        fields.addStretch()

        actions = W.QHBoxLayout()
        actions.setSpacing(7)
        self.start_button = W.QPushButton("开始")
        self.start_button.setProperty("primary", True)
        self.start_button.clicked.connect(lambda: self.start_run(single=False))
        self.stop_button = W.QPushButton("停止")
        self.stop_button.clicked.connect(self.stop_run)
        self.single_button = W.QPushButton("单次评价")
        self.single_button.clicked.connect(lambda: self.start_run(single=True))
        for button in (self.start_button, self.stop_button, self.single_button):
            actions.addWidget(button, 1)
        layout.addLayout(actions)
        return panel

    @staticmethod
    def field_with_button(field, button):
        widget = W.QWidget()
        row = W.QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(5)
        row.addWidget(field, 1)
        row.addWidget(button)
        return widget

    @staticmethod
    def percentage(default, minimum):
        spin = W.QDoubleSpinBox(decimals=2, minimum=minimum, maximum=999_999,
                               singleStep=0.5, value=default, suffix=" %")
        return spin

    def build_waveform(self):
        panel = card()
        layout = W.QVBoxLayout(panel)
        layout.setContentsMargins(13, 12, 13, 10)
        layout.setSpacing(7)
        header = W.QHBoxLayout()
        header.addWidget(label("实时波形", "heading"))
        self.condition = label("", "muted")
        header.addWidget(self.condition)
        header.addStretch()
        self.progress_label = label("空闲", "muted")
        header.addWidget(self.progress_label)
        layout.addLayout(header)

        self.progress = W.QProgressBar(textVisible=False, maximum=1)
        self.progress.setFixedHeight(5)
        self.progress.setAccessibleName("工况扫描进度")
        layout.addWidget(self.progress)

        content = W.QHBoxLayout()
        content.setSpacing(12)
        frame = W.QVBoxLayout()
        frame.setSpacing(4)
        self.plot = pg.PlotWidget(background="#102537")
        self.plot.setMinimumSize(0, 140)
        self.plot.setLabel("bottom", "相对上升 50% 交点", units="ns", color="#bfd0de")
        self.plot.setLabel("left", "电压", units="V", color="#bfd0de")
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.plot.hideButtons()
        for axis in ("bottom", "left"):
            item = self.plot.getAxis(axis)
            item.setPen("#658096")
            item.setTextPen("#bfd0de")
            item.enableAutoSIPrefix(False)
        frame.addWidget(self.plot, 1)
        legend_row = W.QHBoxLayout()
        legend = label("保持期 · 红区 y0–y2 · 蓝区参考", "muted")
        legend.setWordWrap(True)
        legend_row.addWidget(legend, 1)
        self.zoom_button = W.QPushButton("重置缩放")
        self.zoom_button.clicked.connect(self.reset_zoom)
        legend_row.addWidget(self.zoom_button)
        frame.addLayout(legend_row)
        content.addLayout(frame, 1)

        metrics = W.QVBoxLayout()
        metrics.setContentsMargins(0, 3, 0, 2)
        metrics.setSpacing(5)
        self.pulse_caption = label("", "muted")
        self.pulse_caption.setWordWrap(True)
        self.verdict = label("待采集", "verdict")
        metrics.addWidget(label("整次采集评价", "section"))
        metrics.addWidget(self.verdict, alignment=Qt.AlignmentFlag.AlignLeft)
        metrics.addSpacing(5)
        metrics.addWidget(label("最大红蓝偏差", "muted"))
        self.alignment_value = label("—", "value")
        self.alignment_limit = label("", "muted")
        metrics.addWidget(self.alignment_value)
        metrics.addWidget(self.alignment_limit)
        metrics.addSpacing(3)
        metrics.addWidget(label("最大原始不平度", "muted"))
        self.raw_value = label("—", "value")
        self.raw_threshold = label("", "muted")
        metrics.addWidget(self.raw_value)
        metrics.addWidget(self.raw_threshold)
        metrics.addWidget(self.pulse_caption)
        self.reason = label("", "muted")
        self.reason.setWordWrap(True)
        metrics.addWidget(self.reason)
        metrics.addStretch()
        results = W.QWidget()
        results.setFixedWidth(176)
        results.setLayout(metrics)
        content.addWidget(results)
        layout.addLayout(content, 1)
        return panel

    def build_log(self):
        panel = card()
        layout = W.QVBoxLayout(panel)
        layout.setContentsMargins(13, 9, 13, 10)
        layout.setSpacing(5)
        layout.addWidget(label("运行日志", "heading"))
        self.log_view = W.QPlainTextEdit(readOnly=True, maximumBlockCount=1500)
        self.log_view.setMinimumHeight(46)
        layout.addWidget(self.log_view)
        return panel

    def append_log(self, message):
        self.log_view.appendPlainText(f"{datetime.now():%H:%M:%S}  {message}")

    def refresh_ports(self):
        saved = self.board_port.currentData()
        self.board_port.clear()
        self.board_port.addItem("未选择", "")
        self.board_port.setToolTip("")
        try:
            ports = sorted(comports())
        except OSError as error:
            self.board_port.setToolTip(f"串口列表读取失败。{error}")
            return
        for port in ports:
            self.board_port.addItem(port.device, port.device)
        previous = self.board_port.findData(saved)
        if previous >= 0:
            self.board_port.setCurrentIndex(previous)

    def choose_directory(self):
        directory = W.QFileDialog.getExistingDirectory(self, "选择结果保存目录", self.destination.text())
        if directory:
            self.destination.setText(str(Path(directory)))

    def set_busy(self, busy):
        self.form.setEnabled(not busy)
        self.start_button.setEnabled(not busy)
        self.single_button.setEnabled(not busy)
        self.stop_button.setEnabled(busy)

    def run_settings(self, single):
        destination = self.destination.text().strip()
        if not destination:
            raise ValueError("请选择保存目录。")

        def axis(widgets, name):
            if single:
                start = widgets[0].value()
                return range(start, start + 1)
            return scan_axis(*(widget.value() for widget in widgets), name)

        return SimulationSettings(axis(self.frequencies, "重频"), axis(self.widths, "脉宽"),
                                  Path(destination).expanduser().resolve(), self.scope_ip.text().strip(),
                                  self.board_port.currentData(), self.alignment.value(), self.raw_limit.value())

    def start_run(self, *, single):
        if self.worker is not None:
            return
        try:
            settings = self.run_settings(single)
        except (OSError, ValueError) as error:
            self.append_log(f"参数有误。{error}")
            self.progress_label.setText("请检查参数")
            return
        self.clear_measurement()
        self.progress.setMaximum(settings.count)
        self.progress_label.setText(f"0 / {settings.count} 个工况")
        self.set_verdict("采集中")
        self.alignment_limit.setText(f"允许 ±{settings.alignment_tolerance_pct:g}%")
        self.raw_threshold.setText(f"要求 <{settings.raw_limit_pct:g}%")
        self.set_busy(True)
        self.append_log("开始模拟单次评价。" if single else f"开始模拟扫描，共 {settings.count} 个工况。")
        worker = SimulationThread(settings, self)
        self.worker = worker
        worker.directory_ready.connect(self.show_directory)
        worker.measurement_ready.connect(self.show_measurement)
        worker.finished.connect(self.finish_run)
        worker.start()

    def show_directory(self, directory):
        self.append_log(f"保存目录  {directory}")

    def stop_run(self):
        if self.worker is None or self.worker.cancelled.is_set():
            return
        self.worker.cancelled.set()
        self.stop_button.setEnabled(False)
        self.progress_label.setText("正在停止…")
        self.append_log("已请求停止，等待当前操作结束。")

    def finish_run(self):
        worker = self.worker
        outcome = worker.outcome
        stopped = worker.cancelled.is_set()
        self.worker = None
        worker.deleteLater()
        if stopped:
            self.clear_measurement()
        if outcome.status == "failed":
            condition = outcome.condition
            point = f"  {condition[0]} kHz · {condition[1]} ns" if condition else ""
            self.progress_label.setText("运行失败")
            if self.progress.value() == 0:
                self.condition.setText(point.strip())
                self.set_verdict("待复核", "review")
                self.reason.setText(outcome.error)
            self.append_log(f"运行失败{point}。{outcome.error}")
        elif stopped or outcome.status == "stopped":
            self.progress_label.setText("已停止")
            self.append_log(f"流程已停止，已保存 {outcome.saved} 个工况。")
        else:
            self.progress_label.setText(f"已完成  {outcome.saved} / {worker.settings.count}")
            self.append_log(f"流程完成，已保存 {outcome.saved} 个工况。")
        self.set_busy(False)
        if self.closing:
            self.close()

    def set_verdict(self, text, state=""):
        self.verdict.setText(text)
        self.verdict.setProperty("state", state)
        self.verdict.style().unpolish(self.verdict)
        self.verdict.style().polish(self.verdict)

    def reset_zoom(self):
        self.plot.enableAutoRange()
        self.plot.autoRange()

    def clear_measurement(self):
        self.plot.clear()
        self.plot.setRange(xRange=(0, 100), yRange=(0, 1), padding=0)
        self.zoom_button.setEnabled(False)
        self.progress.setValue(0)
        self.condition.clear()
        self.set_verdict("待采集")
        for value in (self.alignment_value, self.raw_value):
            value.setText("—")
        for value in (self.alignment_limit, self.raw_threshold, self.pulse_caption, self.reason):
            value.clear()

    def show_measurement(self, measurement):
        report = measurement.report
        shape, flatness = report["shape"], report["flatness"]
        point = f"{measurement.frequency_khz} kHz · {measurement.width_ns} ns"
        alignment = shape["score_pct"]
        aligned_value = "—" if alignment is None else f"{alignment:.2f}%"
        raw_value = f"{flatness['raw_max_pct']:.2f}%"
        self.append_log(f"{point}  {measurement.verdict}  红蓝 {aligned_value}  不平度 {raw_value}  "
                        f"已保存 {measurement.capture_path.name}")
        if self.worker.cancelled.is_set():
            return
        self.condition.setText(point)
        self.progress.setValue(measurement.index)
        self.progress_label.setText(f"{measurement.index} / {self.worker.settings.count} 个工况")
        states = {"达标": "pass", "未达标": "fail", "待复核": "review"}
        self.set_verdict(measurement.verdict, states[measurement.verdict])
        self.alignment_value.setText(aligned_value)
        self.raw_value.setText(raw_value)
        pulse = measurement.pulse
        self.pulse_caption.setText(f"{len(flatness['pulses'])} 个完整脉冲 · 图示第 {pulse['number']} 个")
        self.reason.setText("保持期不平度超限。" if shape["status"] == "flat" and not report["passed"]
                            else shape["reason"])
        self.draw_pulse(pulse, report["window"])

    def draw_pulse(self, pulse, window):
        self.plot.clear()
        t, voltage, width = pulse["time_ns"], pulse["voltage_v"], pulse["width50_ns"]
        visible = (t >= -0.18 * width) & (t <= 1.18 * width)
        areas = [("hold", window["phase"]), *window["alignment_windows"].items()]
        for role, interval in areas:
            if interval is None:
                continue
            if role == "hold":
                fill, edge = (102, 164, 172, 30), (132, 184, 194, 125)
            elif role == "reference":
                fill, edge = (77, 155, 231, 95), (132, 193, 250, 185)
            else:
                fill, edge = (250, 120, 128, 70), (255, 156, 163, 180)
            region = pg.LinearRegionItem([part * width for part in interval], movable=False,
                                         brush=fill, pen=pg.mkPen(edge))
            self.plot.addItem(region, ignoreBounds=True)
        self.plot.plot(t[visible], voltage[visible], pen=pg.mkPen("#7fe0ee", width=2),
                       autoDownsample=True, downsampleMethod="peak")
        self.zoom_button.setEnabled(True)
        self.reset_zoom()

    def closeEvent(self, event):
        if self.worker is None:
            super().closeEvent(event)
        else:
            self.closing = True
            self.stop_run()
            event.ignore()


def main():
    app = W.QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Microsoft YaHei UI", 10))
    window = WaveformWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
