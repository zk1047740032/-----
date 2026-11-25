"""Windows desktop frontend for square-wave evaluation."""

from datetime import datetime
from ipaddress import ip_address
from pathlib import Path
import sys
from threading import Event

import pyqtgraph as pg
from PySide6.QtCore import QStandardPaths, QThread, Qt, Signal
from PySide6.QtGui import QColor, QFont, QTextCharFormat, QTextCursor
from PySide6 import QtWidgets as W
from serial.tools.list_ports import comports

from scope_probe import SCOPE_PORT, ScopeProbe
from waveform_core.reference_alignment import AXIS_LABELS, DEFAULT_ALIGNMENT_TOLERANCE_PCT
from waveform_simulation import (
    DEFAULT_FREQUENCY,
    DEFAULT_RAW_LIMIT,
    DEFAULT_WIDTH,
    FileEvaluationSettings,
    SimulationSettings,
    run_evaluation,
    scan_axis,
)


STYLE = """
QMainWindow { background: #f4f6f9; }
QWidget { color: #2c3e50; font-size: 13px; }
QWidget#controlFields { background: #ffffff; }
QFrame[card="true"] {
    background: #ffffff; border: 1px solid #e3e7ec; border-radius: 12px;
}
QLabel { background: transparent; }
QLabel[role="title"] { color: #34495e; font-size: 24px; font-weight: 700; }
QLabel[role="heading"] { color: #34495e; font-size: 16px; font-weight: 700; }
QLabel[role="section"] { color: #2c3e50; font-weight: 700; }
QLabel[role="muted"] { color: #6b7280; }
QLabel[role="value"] { color: #34495e; font-size: 20px; font-weight: 700; }
QLabel[role="mode"] {
    color: #2d6cdf; background: #eaf1fd; border: 1px solid #cdddf7;
    border-radius: 12px; padding: 5px 12px; font-weight: 700;
}
QLabel[role="verdict"] {
    color: #6b7280; background: #eef0f1; border-radius: 9px; padding: 5px 10px;
    font-weight: 700;
}
QLabel[role="verdict"][state="pass"] { color: #1e8449; background: #e9f6ef; }
QLabel[role="verdict"][state="fail"] { color: #c0392b; background: #fdf2f0; }
QLabel[role="verdict"][state="review"], QLabel[role="adjustment"][state="review"] {
    color: #9b640b; background: #fef6ec;
}
QLabel[role="adjustment"] {
    color: #6b7280; background: #eef0f1; border-radius: 6px; padding: 2px 8px; font-weight: 700;
}
QLabel[role="adjustment"][state="trial"] { color: #ffffff; background: #2d6cdf; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPushButton {
    background: #ffffff; border: 1px solid #cbd2dc; border-radius: 6px;
    padding: 4px 8px; min-height: 22px;
    selection-background-color: #2d6cdf;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QPushButton:focus {
    border-color: #2d6cdf;
}
QPushButton:hover { color: #2d6cdf; border-color: #2d6cdf; }
QPushButton[primary="true"] { background: #2d6cdf; color: white; border-color: #2d6cdf; font-weight: 700; }
QPushButton[primary="true"]:hover { background: #1f5bc0; }
QPushButton#stopButton { background: #fdedec; border-color: #f5c6cb; color: #c0392b; }
QPushButton#stopButton:hover { background: #f9d7da; color: #922b21; }
QPushButton:disabled, QPushButton[primary="true"]:disabled, QPushButton#stopButton:disabled {
    background: #f7f9fb; color: #8994a2; border-color: #eceff3;
}
QProgressBar { background: #eef0f1; border: none; border-radius: 3px; }
QProgressBar::chunk { background: #2d6cdf; border-radius: 3px; }
QPlainTextEdit { background: #ffffff; border: 1px solid #e3e7ec; border-radius: 7px; padding: 5px; }
QScrollArea { background: transparent; border: none; }
QSplitter::handle { background: #f4f6f9; height: 10px; }
"""

LOG_COLORS = {
    "info": "#2c3e50",
    "system": "#2d6cdf",
    "success": "#27ae60",
    "warning": "#9b640b",
    "error": "#e74c3c",
}


def label(text, role=None):
    widget = W.QLabel(text)
    if role:
        widget.setProperty("role", role)
    return widget


def card():
    frame = W.QFrame()
    frame.setProperty("card", True)
    return frame


class EvaluationThread(QThread):
    directory_ready = Signal(object)
    measurement_ready = Signal(object)

    def __init__(self, settings, parent):
        super().__init__(parent)
        self.settings = settings
        self.cancelled = Event()
        self.outcome = None

    def run(self):
        self.outcome = run_evaluation(self.settings, self.cancelled,
                                     self.directory_ready.emit, self.measurement_ready.emit)


class WaveformWindow(W.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("方波自动化 · 波形评价")
        self.worker = None
        self.scope_probe = None
        self.closing = False
        self.waveform_directory = ""
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
        self.mode = label("模拟模式", "mode")
        header.addWidget(self.mode)
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
        self.append_log("可开始模拟采集或打开波形文件。")

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
        self.scope_check_button = W.QPushButton("检查")
        self.scope_check_button.setAccessibleName("检查示波器通信")
        self.scope_check_button.setToolTip(f"使用 TCP {SCOPE_PORT} 查询示波器身份。")
        self.scope_check_button.clicked.connect(self.check_scope)
        connection.addRow("示波器 IP", self.field_with_button(self.scope_ip, self.scope_check_button))
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
        self.stop_button.setObjectName("stopButton")
        self.stop_button.clicked.connect(self.stop_run)
        self.single_button = W.QPushButton("单次评价")
        self.single_button.clicked.connect(lambda: self.start_run(single=True))
        for button in (self.start_button, self.stop_button, self.single_button):
            actions.addWidget(button, 1)
        layout.addLayout(actions)
        self.open_button = W.QPushButton("打开波形文件…")
        self.open_button.clicked.connect(self.open_waveform)
        layout.addWidget(self.open_button)
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
        self.plot = pg.PlotWidget(background="#ffffff")
        self.plot.setMinimumSize(0, 140)
        self.plot.setLabel("bottom", "相对上升 50% 交点", units="ns", color="#6b7280")
        self.plot.setLabel("left", "电压", units="V", color="#6b7280")
        self.plot.showGrid(x=True, y=True, alpha=0.22)
        self.plot.hideButtons()
        for axis in ("bottom", "left"):
            item = self.plot.getAxis(axis)
            item.setPen("#aab3bd")
            item.setTextPen("#6b7280")
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

        values = W.QGridLayout()
        values.setHorizontalSpacing(9)
        values.setVerticalSpacing(2)
        values.addWidget(label("最大红蓝偏差", "muted"), 0, 0)
        values.addWidget(label("最大原始不平度", "muted"), 0, 1)
        self.alignment_value = label("—", "value")
        self.raw_value = label("—", "value")
        self.alignment_limit = label("", "muted")
        self.raw_threshold = label("", "muted")
        values.addWidget(self.alignment_value, 1, 0)
        values.addWidget(self.raw_value, 1, 1)
        values.addWidget(self.alignment_limit, 2, 0)
        values.addWidget(self.raw_threshold, 2, 1)
        metrics.addLayout(values)

        metrics.addSpacing(4)
        metrics.addWidget(label("本步 y 调整建议", "section"))
        adjustments = W.QGridLayout()
        adjustments.setHorizontalSpacing(8)
        adjustments.setVerticalSpacing(4)
        self.axis_advice = {}
        for row, (axis, area) in enumerate(AXIS_LABELS.items()):
            adjustments.addWidget(label(f"{axis} · {area}"), row, 0)
            value = label("—", "adjustment")
            value.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.axis_advice[axis] = value
            adjustments.addWidget(value, row, 1)
        metrics.addLayout(adjustments)
        metrics.addWidget(self.pulse_caption)
        self.reason = label("", "muted")
        self.reason.setWordWrap(True)
        metrics.addWidget(self.reason)
        metrics.addStretch()
        results = W.QWidget()
        results.setFixedWidth(208)
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

    def append_log(self, message, *, level="info"):
        document = self.log_view.document()
        scrollbar = self.log_view.verticalScrollBar()
        following = scrollbar.value() == scrollbar.maximum()
        cursor = QTextCursor(document)
        cursor.movePosition(QTextCursor.MoveOperation.End)
        format_ = QTextCharFormat()
        format_.setForeground(QColor(LOG_COLORS[level]))
        cursor.beginEditBlock()
        if not document.isEmpty():
            cursor.insertBlock()
        cursor.insertText(f"{datetime.now():%H:%M:%S}  {message}", format_)
        cursor.endEditBlock()
        if following:
            scrollbar.setValue(scrollbar.maximum())

    def check_scope(self):
        if self.scope_probe is not None:
            return
        try:
            address = str(ip_address(self.scope_ip.text().strip()))
        except ValueError:
            self.append_log("请输入有效的示波器 IP 地址。", level="warning")
            self.scope_ip.setFocus()
            return

        self.scope_ip.setEnabled(False)
        self.scope_check_button.setEnabled(False)
        self.append_log(f"检查示波器 {address}（TCP {SCOPE_PORT}）。", level="system")
        probe = ScopeProbe(address, self)
        self.scope_probe = probe
        probe.finished.connect(self.finish_scope_check)
        probe.start()

    def finish_scope_check(self, success, detail):
        probe = self.scope_probe
        self.scope_probe = None
        probe.deleteLater()
        self.scope_ip.setEnabled(True)
        self.scope_check_button.setEnabled(True)
        if success:
            self.append_log(f"示波器 {probe.address} 通信正常。{detail}", level="success")
        else:
            self.append_log(f"示波器 {probe.address} 通信失败。{detail}", level="error")

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
        self.open_button.setEnabled(not busy)
        self.stop_button.setEnabled(busy)

    def destination_path(self):
        destination = self.destination.text().strip()
        if not destination:
            raise ValueError("请选择保存目录。")
        return Path(destination).expanduser().resolve()

    def run_settings(self, single):
        def axis(widgets, name):
            if single:
                start = widgets[0].value()
                return range(start, start + 1)
            return scan_axis(*(widget.value() for widget in widgets), name)

        return SimulationSettings(axis(self.frequencies, "重频"), axis(self.widths, "脉宽"),
                                  self.destination_path(), self.scope_ip.text().strip(),
                                  self.board_port.currentData(), self.alignment.value(), self.raw_limit.value())

    def open_waveform(self):
        if self.worker is not None:
            return
        try:
            destination = self.destination_path()
        except (OSError, ValueError) as error:
            self.append_log(f"参数有误。{error}", level="warning")
            self.progress_label.setText("请检查参数")
            return
        selected, _ = W.QFileDialog.getOpenFileName(
            self, "打开波形文件", self.waveform_directory, "波形文件 (*.csv *.npz)")
        if not selected:
            return
        source = Path(selected)
        self.waveform_directory = str(source.parent)
        thresholds = dict(alignment_tolerance_pct=self.alignment.value(), raw_limit_pct=self.raw_limit.value())
        self.start_evaluation(FileEvaluationSettings(source, destination, thresholds))

    def start_run(self, *, single):
        if self.worker is not None:
            return
        try:
            settings = self.run_settings(single)
        except (OSError, ValueError) as error:
            self.append_log(f"参数有误。{error}", level="warning")
            self.progress_label.setText("请检查参数")
            return
        self.start_evaluation(settings)

    def start_evaluation(self, settings):
        self.clear_measurement()
        self.progress.setMaximum(settings.count)
        file_mode = isinstance(settings, FileEvaluationSettings)
        self.mode.setText("文件评价" if file_mode else "模拟模式")
        if file_mode:
            source = settings.source_file
            name = self.condition.fontMetrics().elidedText(source.name, Qt.TextElideMode.ElideMiddle, 190)
            self.condition.setText(name)
            self.condition.setToolTip(str(source))
            self.progress_label.setText("读取文件…")
            self.set_verdict("评价中")
            self.append_log(f"打开波形文件  {source}", level="system")
        else:
            self.progress_label.setText(f"0 / {settings.count} 个工况")
            self.set_verdict("采集中")
            if settings.count == 1:
                self.append_log("开始模拟评价，共 1 个工况。", level="system")
            else:
                self.append_log(f"开始模拟扫描，共 {settings.count} 个工况。", level="system")
        thresholds = settings.thresholds
        self.alignment_limit.setText(f"允许 ±{thresholds['alignment_tolerance_pct']:g}%")
        self.raw_threshold.setText(f"要求 <{thresholds['raw_limit_pct']:g}%")
        self.set_busy(True)
        worker = EvaluationThread(settings, self)
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
        self.append_log("已请求停止，等待当前操作结束。", level="warning")

    def finish_run(self):
        worker = self.worker
        outcome = worker.outcome
        file_mode = isinstance(worker.settings, FileEvaluationSettings)
        stopped = worker.cancelled.is_set()
        self.worker = None
        worker.deleteLater()
        if stopped:
            self.clear_measurement()
        if outcome.status == "failed":
            condition = outcome.condition
            if file_mode:
                point = worker.settings.source_file.name
            else:
                point = f"{condition[0]} kHz · {condition[1]} ns" if condition else ""
            self.progress_label.setText("运行失败")
            if self.progress.value() == 0:
                if not file_mode:
                    self.condition.setText(point)
                self.set_verdict("待复核", "review")
                self.reason.setText(outcome.error)
            self.append_log(f"运行失败  {point}  {outcome.error}", level="error")
        elif stopped or outcome.status == "stopped":
            self.progress_label.setText("已停止")
            unit = "份文件结果" if file_mode else "个工况"
            self.append_log(f"流程已停止，已保存 {outcome.saved} {unit}。", level="warning")
        elif file_mode:
            self.progress_label.setText("文件评价完成")
            self.append_log("文件评价完成，已保存结果。")
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

    def show_axis_advice(self, shape=None):
        candidate = shape.get("candidate") if shape else None
        for axis, value in self.axis_advice.items():
            state = ""
            if shape is None:
                text = "—"
            elif candidate and axis == candidate["axis"]:
                step = candidate["step"]
                direction = "增大" if step > 0 else "减小"
                text, state = f"{direction} {step:+d}", "trial"
            elif candidate:
                text = "本步保持"
            elif shape["status"] == "flat":
                text = "保持"
            else:
                text, state = "待复核", "review"
            value.setText(text)
            value.setProperty("state", state)
            value.style().unpolish(value)
            value.style().polish(value)

    def reset_zoom(self):
        self.plot.enableAutoRange()
        self.plot.autoRange()

    def clear_measurement(self):
        self.plot.clear()
        self.plot.setRange(xRange=(0, 100), yRange=(0, 1), padding=0)
        self.zoom_button.setEnabled(False)
        self.progress.setValue(0)
        self.condition.clear()
        self.condition.setToolTip("")
        self.set_verdict("待采集")
        self.show_axis_advice()
        for value in (self.alignment_value, self.raw_value):
            value.setText("—")
        for value in (self.alignment_limit, self.raw_threshold, self.pulse_caption, self.reason):
            value.clear()

    def show_measurement(self, measurement):
        report = measurement.report
        shape, flatness = report["shape"], report["flatness"]
        file_mode = measurement.source_file is not None
        point = (measurement.source_file.name if file_mode
                 else f"{measurement.frequency_khz} kHz · {measurement.width_ns} ns")
        alignment = shape["score_pct"]
        aligned_value = "—" if alignment is None else f"{alignment:.2f}%"
        raw_value = f"{flatness['raw_max_pct']:.2f}%"
        verdict = measurement.verdict
        state, level = {"达标": ("pass", "success"), "未达标": ("fail", "error"),
                        "待复核": ("review", "warning")}[verdict]
        self.append_log(f"{point}  {verdict}  红蓝 {aligned_value}  不平度 {raw_value}", level=level)
        self.append_log(f"已保存 {measurement.capture_path.name}", level="success")
        if self.worker.cancelled.is_set():
            return
        self.progress.setValue(measurement.index)
        if not file_mode:
            self.condition.setText(point)
            self.progress_label.setText(f"{measurement.index} / {self.worker.settings.count} 个工况")
        self.set_verdict(verdict, state)
        self.alignment_value.setText(aligned_value)
        self.raw_value.setText(raw_value)
        self.show_axis_advice(shape)
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
                fill, edge = (45, 108, 223, 24), (45, 108, 223, 92)
            elif role == "reference":
                fill, edge = (106, 151, 237, 50), (106, 151, 237, 148)
            else:
                fill, edge = (223, 118, 128, 50), (223, 118, 128, 148)
            region = pg.LinearRegionItem([part * width for part in interval], movable=False,
                                         brush=fill, pen=pg.mkPen(edge))
            self.plot.addItem(region, ignoreBounds=True)
        self.plot.plot(t[visible], voltage[visible], pen=pg.mkPen("#2563eb", width=2),
                       autoDownsample=True, downsampleMethod="peak")
        self.zoom_button.setEnabled(True)
        self.reset_zoom()

    def closeEvent(self, event):
        if self.scope_probe is not None:
            self.scope_probe.stop()
            self.scope_probe.deleteLater()
            self.scope_probe = None
        if self.worker is None:
            super().closeEvent(event)
        else:
            self.closing = True
            self.stop_run()
            event.ignore()


def main():
    app = W.QApplication(sys.argv)
    app.setStyle("Fusion")
    app.styleHints().setColorScheme(Qt.ColorScheme.Light)
    app.setFont(QFont("Microsoft YaHei", 10))
    window = WaveformWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
