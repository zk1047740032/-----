"""Windows desktop frontend for square-wave evaluation."""

from datetime import datetime
from ipaddress import ip_address
from math import ceil
from pathlib import Path
import socket
import sys
from threading import Event, Lock
from time import monotonic

import pyqtgraph as pg
from PySide6.QtCore import QEvent, QRectF, QStandardPaths, QThread, Qt, Signal
from PySide6.QtGui import (QColor, QFont, QFontMetricsF, QIcon, QIntValidator, QPainter, QPen,
                          QTextCharFormat, QTextCursor)
from PySide6 import QtWidgets as W
from serial.tools.list_ports import comports

from pulse_board_port import BOARD_BAUDRATE, SHAPE_INDEX_D, SHAPE_REGISTER_MAX, PulseBoardPort
from scope_capture import SCOPE_CHANNEL, SCOPE_PORT, SCOPE_TIMEOUT_S, ScopeConnection
from waveform_core.reference_alignment import AXIS_LABELS, DEFAULT_ALIGNMENT_TOLERANCE_PCT
from waveform_simulation import (
    DEFAULT_FREQUENCY,
    DEFAULT_RAW_LIMIT,
    DEFAULT_WIDTH,
    FileEvaluationSettings,
    ScopeEvaluationSettings,
    SimulationSettings,
    run_evaluation,
    scan_axis,
)


PULSE_TIME_LABEL = "相对上升 50% 交点"
PULSE_LEGEND = "保持期 · 红区 y0–y2 · 蓝区参考"


STYLE = """
QMainWindow { background: #f4f6f9; }
QWidget { color: #2c3e50; font-size: 13px; }
QWidget#controlFields { background: #ffffff; }
QFrame[card="true"] {
    background: #ffffff; border: 1px solid #e3e7ec; border-radius: 12px;
}
QLabel { background: transparent; }
QLabel[role="heading"] { color: #34495e; font-size: 16px; font-weight: 700; }
QLabel[role="section"] { color: #2c3e50; font-weight: 700; }
QLabel[role="muted"] { color: #6b7280; }
QLabel[role="value"] { color: #34495e; font-size: 20px; font-weight: 700; }
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


class ScopeTimeAxis(pg.AxisItem):
    division_ns = None

    def set_scope_division(self, division_ns):
        self.division_ns = division_ns
        self.picture = None
        self.update()

    def tickSpacing(self, minVal, maxVal, size):
        spacing = self.division_ns
        span = maxVal - minVal
        if spacing is None or span < 2 * spacing:
            return super().tickSpacing(minVal, maxVal, size)
        metrics = QFontMetricsF(self.style["tickFont"] or W.QApplication.font())
        labels = self.tickStrings([minVal, maxVal], 1, spacing)
        label_width = max(metrics.horizontalAdvance(value) for value in labels)
        # Skip full divisions as needed to leave room for the time labels.
        label_spacing = span * (label_width + 10) / max(1, size)
        step = max(1, ceil(label_spacing / spacing))
        return [(spacing * step, 0)]


class ScopeVoltageAxis(pg.AxisItem):
    division_v = None
    grid_origin_v = 0.

    def set_scope_division(self, division_v, origin_v=0.):
        self.division_v = division_v
        self.grid_origin_v = origin_v
        self.picture = None
        self.update()

    def tickSpacing(self, minVal, maxVal, size):
        spacing = self.division_v
        if spacing is not None and 2 * spacing <= maxVal - minVal <= 16 * spacing:
            return [(spacing, self.grid_origin_v)]
        return super().tickSpacing(minVal, maxVal, size)

    def tickStrings(self, values, scale, spacing):
        if self.division_v is None:
            return super().tickStrings(values, scale, spacing)
        return [f"{0. if abs(value) < spacing * 1e-9 else value * scale:.8g}" for value in values]


class SimulationSwitch(W.QAbstractButton):
    def __init__(self):
        super().__init__()
        self.setCheckable(True)
        self.setFixedSize(56, 36)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAccessibleName("模拟模式")

    def paintEvent(self, _event):
        checked = self.isChecked()
        enabled = self.isEnabled()
        if enabled:
            fill, border = ("#23a653", "#168540") if checked else ("#d9dfe7", "#b5bfcb")
        else:
            fill, border = ("#b1d5bf", "#9dbfac") if checked else ("#edf0f4", "#d5dde6")
        track = QRectF(4, 6, 48, 24)
        knob_x = track.right() - 22 if checked else track.left() + 2
        knob = QRectF(knob_x, track.top() + 2, 20, 20)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self.hasFocus():
            painter.setPen(QPen(QColor("#2d6cdf"), 1.5))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(track.adjusted(-2.5, -2.5, 2.5, 2.5), 15, 15)
        painter.setPen(QPen(QColor(border)))
        painter.setBrush(QColor(fill))
        painter.drawRoundedRect(track, 12, 12)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 27))
        painter.drawEllipse(knob.translated(0, 1))
        painter.setPen(QPen(QColor("#d7dfe8")))
        painter.setBrush(QColor("#ffffff" if enabled else "#f6f8fa"))
        painter.drawEllipse(knob)
        painter.end()


class EvaluationThread(QThread):
    measurement_ready = Signal(object)

    def __init__(self, settings, parent):
        super().__init__(parent)
        self.settings = settings
        self.cancelled = Event()
        self.outcome = None

    def run(self):
        self.outcome = run_evaluation(self.settings, self.cancelled, self.measurement_ready.emit)


class ScopeThread(QThread):
    identified = Signal(str)
    frame_ready = Signal()
    frame_period_s = 1 / 60

    def __init__(self, address, live, parent):
        super().__init__(parent)
        self.address = address
        self.live = live
        self.cancelled = Event()
        self.connection = None
        self.identity = ""
        self._frame_lock = Lock()
        self._latest_frame = None
        self._frame_pending = False
        self.error = ""

    def run(self):
        try:
            with socket.create_connection((self.address, SCOPE_PORT), timeout=SCOPE_TIMEOUT_S) as connection:
                self.connection = connection
                if self.cancelled.is_set():
                    return
                with connection.makefile("rb") as stream:
                    scope = ScopeConnection(connection, stream)
                    self.identity = scope.identify()
                    self.identified.emit(self.identity)
                    if not self.live or self.cancelled.is_set():
                        return
                    scope.configure_waveform()
                    while not self.cancelled.is_set():
                        started = monotonic()
                        self._publish_frame(scope.read_frame())
                        self.cancelled.wait(max(0., self.frame_period_s - (monotonic() - started)))
        except (OSError, ValueError) as error:
            if not self.cancelled.is_set():
                self.error = str(error) or type(error).__name__
        finally:
            self.connection = None

    def _publish_frame(self, frame):
        with self._frame_lock:
            self._latest_frame = frame
            if self._frame_pending:
                return
            self._frame_pending = True
        self.frame_ready.emit()

    def take_latest_frame(self):
        with self._frame_lock:
            self._frame_pending = False
            return self._latest_frame

    def stop(self):
        self.cancelled.set()
        connection = self.connection
        if connection is not None:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


class BoardPortThread(QThread):
    def __init__(self, port, parent):
        super().__init__(parent)
        self.port = port
        self.connection = None
        self.error = ""

    def run(self):
        try:
            self.connection = PulseBoardPort(self.port)
        except (OSError, ValueError) as error:
            self.error = str(error)


class BoardShapeThread(QThread):
    def __init__(self, connection, values, parent):
        super().__init__(parent)
        self.connection = connection
        self.values = values
        self.error = None

    def run(self):
        try:
            self.connection.set_shape(*self.values)
        except (OSError, ValueError) as error:
            self.error = error


class WaveformWindow(W.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("方波自动化")
        self.worker = None
        self.scope_thread = None
        self.board_thread = None
        self.board_connection = None
        self.scope_frozen = False
        self.scope_evaluation_items = []
        self.scope_preview_range = None
        self.displayed_frame = None
        self.scope_curve = None
        self.scope_rendered = (None, None)
        self.closing = False
        self.waveform_directory = ""
        self.setStyleSheet(STYLE)
        self.setMinimumSize(820, 520)

        central = W.QWidget()
        self.setCentralWidget(central)
        root = W.QVBoxLayout(central)
        root.setContentsMargins(18, 12, 18, 14)

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
        splitter.setSizes([530, 197])

        screen = self.screen().availableGeometry()
        self.resize(min(1240, screen.width() - 36), min(790, screen.height() - 52))
        self.clear_measurement()
        self.update_controls()
        self.append_log("可连接示波器、打开波形文件，或开启模拟模式。")

    def build_controls(self):
        panel = card()
        panel.setFixedWidth(348)
        layout = W.QVBoxLayout(panel)
        layout.setContentsMargins(13, 12, 13, 12)
        layout.setSpacing(8)

        header = W.QHBoxLayout()
        header.addWidget(label("控制面板", "heading"))
        header.addStretch()
        header.addWidget(label("模拟", "section"))
        self.simulation_toggle = SimulationSwitch()
        self.simulation_toggle.setToolTip("关闭模拟模式后可连接示波器，实时显示并单次评价。")
        self.simulation_toggle.toggled.connect(self.toggle_simulation)
        header.addWidget(self.simulation_toggle)
        layout.addLayout(header)

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
        self.scope_ip = W.QLineEdit("192.168.7.12")
        self.scope_ip.setAccessibleName("示波器 IP")
        self.scope_button = W.QPushButton("连接")
        self.scope_button.clicked.connect(self.connect_scope)
        connection.addRow("示波器 IP", self.field_with_buttons(self.scope_ip, self.scope_button))
        self.board_port = W.QComboBox()
        self.board_port.setAccessibleName("脉冲板口")
        self.board_port.setMinimumWidth(0)
        self.board_port.setSizePolicy(W.QSizePolicy.Policy.Ignored, W.QSizePolicy.Policy.Fixed)
        self.board_port.addItem("未选择", "")
        self.board_refresh_button = W.QPushButton("刷新")
        self.board_refresh_button.setAccessibleName("刷新串口列表")
        self.board_refresh_button.clicked.connect(self.refresh_ports)
        self.board_button = W.QPushButton("连接")
        self.board_button.clicked.connect(self.connect_board)
        connection.addRow("脉冲板口", self.field_with_buttons(
            self.board_port, self.board_refresh_button, self.board_button))
        self.refresh_ports()
        documents = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation)
        self.screenshot_directory = W.QLineEdit(str((Path(documents) if documents else Path.home()) / "示波器截图"))
        self.screenshot_directory.setAccessibleName("截图目录")
        self.screenshot_directory.setToolTip("自动调平达标后保存示波器截图的目录。")
        browse = W.QPushButton("浏览")
        browse.clicked.connect(self.choose_screenshot_directory)
        connection.addRow("截图目录", self.field_with_buttons(self.screenshot_directory, browse))
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
        self.single_button.clicked.connect(self.single_evaluation)
        for button in (self.start_button, self.stop_button, self.single_button):
            actions.addWidget(button, 1)
        layout.addLayout(actions)
        self.open_button = W.QPushButton("打开波形文件…")
        self.open_button.clicked.connect(self.open_waveform)
        layout.addWidget(self.open_button)
        return panel

    @staticmethod
    def field_with_buttons(field, *buttons):
        widget = W.QWidget()
        row = W.QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(5)
        row.addWidget(field, 1)
        for button in buttons:
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
        self.time_axis = ScopeTimeAxis("bottom")
        self.voltage_axis = ScopeVoltageAxis("left")
        self.plot = pg.PlotWidget(background="#ffffff",
                                  axisItems={"bottom": self.time_axis, "left": self.voltage_axis})
        self.plot.setMinimumSize(0, 140)
        self.plot.setLabel("left", "电压", units="V", color="#6b7280")
        view = self.plot.getViewBox()
        view.sigRangeChangedManually.connect(self.pause_scope_follow)
        view.sigXRangeChanged.connect(self.render_scope_curve)
        self.plot.showGrid(x=True, y=True, alpha=0.22)
        self.plot.hideButtons()
        for axis in ("bottom", "left"):
            item = self.plot.getAxis(axis)
            item.setPen("#aab3bd")
            item.setTextPen("#6b7280")
            item.enableAutoSIPrefix(False)
        frame.addWidget(self.plot, 1)
        legend_row = W.QHBoxLayout()
        self.window_legend = label(PULSE_LEGEND, "muted")
        self.window_legend.setWordWrap(True)
        legend_row.addWidget(self.window_legend, 1)
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

        y_fields = W.QGridLayout()
        y_fields.setHorizontalSpacing(6)
        y_fields.setVerticalSpacing(0)
        self.y_inputs = []
        for column, (axis, default) in enumerate((("y0", 1200), ("y1", 1700), ("y2", 2400))):
            field = W.QLineEdit(str(default))
            field.setValidator(QIntValidator(0, SHAPE_REGISTER_MAX, field))
            field.setAccessibleName(axis)
            self.y_inputs.append(field)
            axis_label = label(axis, "muted")
            axis_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            axis_label.setBuddy(field)
            y_fields.addWidget(axis_label, 0, column)
            y_fields.addWidget(field, 1, column)
            y_fields.setColumnStretch(column, 1)
        metrics.addLayout(y_fields)
        self.set_y_button = W.QPushButton("设置")
        self.set_y_button.setAccessibleName("设置 y 值")
        self.set_y_button.clicked.connect(self.set_board_shape)
        metrics.addWidget(self.set_y_button)
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

    def connect_scope(self):
        if self.scope_thread is not None:
            if self.scope_thread.live:
                self.disconnect_scope()
            return
        try:
            address = str(ip_address(self.scope_ip.text().strip()))
        except ValueError:
            self.append_log("请输入有效的示波器 IP 地址。", level="warning")
            self.scope_ip.setFocus()
            return

        scope = ScopeThread(address, live=not self.simulation_toggle.isChecked(), parent=self)
        self.scope_thread = scope
        if scope.live:
            self.scope_frozen = False
            self.clear_measurement()
            self.condition.setText(SCOPE_CHANNEL)
            self.condition.setToolTip(address)
            self.progress_label.setText("正在连接示波器…")
            scope.frame_ready.connect(self.refresh_scope_frame, Qt.ConnectionType.QueuedConnection)
        self.append_log(f"正在连接示波器 {address}（TCP {SCOPE_PORT}）。", level="system")
        scope.identified.connect(self.show_scope_identity)
        scope.finished.connect(self.finish_scope_connection)
        self.update_controls()
        scope.start()

    def show_scope_identity(self, identity):
        scope = self.scope_thread
        if scope.cancelled.is_set():
            return
        self.append_log(f"示波器 {scope.address} 通信正常。{identity}", level="success")
        if scope.live and self.displayed_frame is None:
            self.progress_label.setText(f"等待 {SCOPE_CHANNEL} 波形…")

    def disconnect_scope(self):
        scope = self.scope_thread
        scope.stop()
        if scope.live:
            self.scope_frozen = False
            self.clear_measurement()
            self.progress_label.setText("已断开示波器")
            self.append_log(f"已断开示波器 {scope.address}。", level="system")
        self.update_controls()

    def finish_scope_connection(self):
        scope = self.scope_thread
        self.scope_thread = None
        scope.deleteLater()
        if scope.error and not scope.cancelled.is_set():
            self.append_log(f"示波器 {scope.address} 连接或读取失败。{scope.error}", level="error")
            if scope.live and not self.scope_frozen:
                self.clear_measurement()
                self.progress_label.setText("示波器读取中断" if scope.identity else "连接失败")
        self.update_controls()
        if self.closing:
            self.close()

    def refresh_scope_frame(self):
        scope = self.scope_thread
        if self.scope_frozen or scope is None or scope.cancelled.is_set():
            return
        frame = scope.take_latest_frame()
        if frame is None or frame is self.displayed_frame:
            return
        previous = self.displayed_frame
        first_frame = previous is None
        if first_frame:
            self.plot.setLabel("bottom", "相对触发时刻", units="ns", color="#6b7280")
        if self.scope_curve is None:
            self.scope_curve = self.plot.plot(pen=pg.mkPen("#2563eb", width=2), clipToView=True,
                                              autoDownsample=True, downsampleMethod="peak")
            self.zoom_button.setEnabled(True)
        scale_s, scale_v = frame.horizontal_scale_s, frame.vertical_scale_v
        if first_frame or (scale_s, scale_v) != (previous.horizontal_scale_s, previous.vertical_scale_v):
            scales = []
            if scale_s is not None:
                self.time_axis.set_scope_division(scale_s * 1e9)
                scales.append(f"水平 {pg.siFormat(scale_s, suffix='s', precision=6)}/格")
            if scale_v is not None:
                scales.append(f"{SCOPE_CHANNEL} 垂直 {pg.siFormat(scale_v, suffix='V', precision=6)}/格")
            self.window_legend.setText(" · ".join(scales))
        follow_pulse = self.scope_follow_pulse and (first_frame or frame.pulse_view_ns != previous.pulse_view_ns)
        rendered_frame = self.scope_rendered[0]
        self.displayed_frame = frame
        if rendered_frame is None or (frame.voltage_range_v, scale_v) != (
                rendered_frame.voltage_range_v, rendered_frame.vertical_scale_v):
            self.set_scope_voltage_view(frame)
        if follow_pulse:
            self.set_scope_view(frame)
        self.render_scope_curve()
        received = frame.received_at.strftime("%H:%M:%S.%f")[:-3]
        self.condition.setText(f"{SCOPE_CHANNEL} · {received}")
        self.progress_label.setText("实时显示" if frame.pulse_view_ns else "未识别到完整脉冲")
        if first_frame:
            self.set_verdict("待评价")
            self.update_controls()

    def refresh_ports(self):
        saved = self.board_port.currentData()
        self.board_port.setToolTip("")
        try:
            ports = sorted(comports())
        except OSError as error:
            self.board_port.setToolTip(f"串口列表读取失败。{error}")
            return
        self.board_port.clear()
        self.board_port.addItem("未选择", "")
        for port in ports:
            self.board_port.addItem(port.device, port.device)
        previous = self.board_port.findData(saved)
        if previous >= 0:
            self.board_port.setCurrentIndex(previous)

    def connect_board(self):
        if self.board_thread is not None:
            return
        if self.board_connection is not None:
            self.disconnect_board()
            self.update_controls()
            return
        port = self.board_port.currentData()
        if not port:
            self.append_log("请先选择脉冲板串口。", level="warning")
            self.board_port.setFocus()
            return

        thread = BoardPortThread(port, self)
        self.board_thread = thread
        thread.finished.connect(self.finish_board_connection)
        self.update_controls()
        thread.start()

    def finish_board_connection(self):
        thread = self.board_thread
        self.board_thread = None
        thread.deleteLater()
        if thread.connection is None:
            self.append_log(f"脉冲板串口 {thread.port} 打开失败。{thread.error}", level="error")
        else:
            self.board_connection = thread.connection
            self.append_log(f"脉冲板串口 {thread.port} 已打开，板卡应答待验证。", level="success")
        self.update_controls()
        if self.closing:
            self.close()

    def set_board_shape(self):
        if not self.set_y_button.isEnabled():
            return
        values = []
        for field in self.y_inputs:
            if not field.hasAcceptableInput():
                self.append_log(f"{field.accessibleName()} 须为 0–{SHAPE_REGISTER_MAX} 的整数。", level="warning")
                field.setFocus()
                return
            value, _ = field.validator().locale().toInt(field.text())
            values.append(value)
        thread = BoardShapeThread(self.board_connection, values, self)
        self.board_thread = thread
        thread.finished.connect(self.finish_board_shape)
        self.update_controls()
        thread.start()

    def finish_board_shape(self):
        thread = self.board_thread
        self.board_thread = None
        thread.deleteLater()
        if thread.error is None:
            y0, y1, y2 = thread.values
            self.append_log(f"脉冲板回读一致。y0={y0}，y1={y1}，y2={y2}，index_d={SHAPE_INDEX_D}。", level="success")
        else:
            self.append_log(f"脉冲板 y 值设置失败。{thread.error}", level="error")
        self.update_controls()
        if self.closing:
            self.close()

    def disconnect_board(self):
        connection = self.board_connection
        if connection is None:
            return
        try:
            connection.close()
        except OSError as error:
            self.append_log(f"关闭脉冲板串口 {connection.port} 失败。{error}", level="error")
            return
        self.board_connection = None
        self.append_log(f"已关闭脉冲板串口 {connection.port}。", level="system")

    def choose_screenshot_directory(self):
        directory = W.QFileDialog.getExistingDirectory(self, "选择示波器截图目录", self.screenshot_directory.text())
        if directory:
            self.screenshot_directory.setText(str(Path(directory)))

    def update_controls(self):
        busy = self.worker is not None
        available = not busy and not self.closing
        simulated = self.simulation_toggle.isChecked()
        scope = self.scope_thread
        preview = scope is not None and scope.live and not scope.cancelled.is_set()
        can_freeze = preview and self.displayed_frame is not None
        can_resume = self.scope_frozen and not self.closing
        board_ready = available and self.board_thread is None
        board_connected = self.board_connection is not None
        self.form.setEnabled(not busy)
        self.simulation_toggle.setEnabled(available)
        self.scope_ip.setEnabled(scope is None)
        self.scope_button.setText("断开" if preview else "连接")
        self.scope_button.setAccessibleName("断开示波器" if preview else "连接示波器")
        action = "查询型号" if simulated else f"实时读取 {SCOPE_CHANNEL} 波形"
        self.scope_button.setToolTip("断开示波器连接。" if preview else f"通过 TCP {SCOPE_PORT} {action}。")
        self.scope_button.setEnabled(available and (scope is None or preview))
        self.board_port.setEnabled(board_ready and not board_connected)
        self.board_refresh_button.setEnabled(board_ready and not board_connected)
        self.board_button.setEnabled(board_ready)
        self.board_button.setText("断开" if board_connected else "连接")
        self.board_button.setAccessibleName("断开脉冲板串口" if board_connected else "连接脉冲板串口")
        self.set_y_button.setEnabled(board_ready and board_connected and not simulated)
        self.set_y_button.setToolTip("关闭模拟并连接脉冲板后可设置 y 值。" if simulated or not board_connected
                                     else f"下发三个 y 值，index_d 固定为 {SHAPE_INDEX_D}。")
        self.start_button.setEnabled(available and simulated)
        self.single_button.setText("恢复" if self.scope_frozen else "单次评价")
        self.single_button.setEnabled(can_resume or (available and (simulated or can_freeze)))
        self.open_button.setEnabled(available)
        self.stop_button.setEnabled(busy and not self.worker.cancelled.is_set())

    def toggle_simulation(self, enabled):
        if self.scope_thread is not None:
            self.disconnect_scope()
        elif self.scope_frozen:
            self.scope_frozen = False
            self.clear_measurement()
            self.progress_label.setText("空闲")
        state = "开启" if enabled else "关闭"
        self.update_controls()
        self.append_log(f"模拟模式已{state}。", level="system")

    def evaluation_thresholds(self):
        return dict(alignment_tolerance_pct=self.alignment.value(), raw_limit_pct=self.raw_limit.value())

    def run_settings(self, single):
        def axis(widgets, name):
            if single:
                start = widgets[0].value()
                return range(start, start + 1)
            return scan_axis(*(widget.value() for widget in widgets), name)

        return SimulationSettings(axis(self.frequencies, "重频"), axis(self.widths, "脉宽"),
                                  **self.evaluation_thresholds())

    def open_waveform(self):
        if self.worker is not None:
            return
        selected, _ = W.QFileDialog.getOpenFileName(
            self, "打开波形文件", self.waveform_directory, "波形文件 (*.csv *.npz)")
        if not selected:
            return
        source = Path(selected)
        self.waveform_directory = str(source.parent)
        if self.scope_thread is not None and self.scope_thread.live:
            self.disconnect_scope()
        self.start_evaluation(FileEvaluationSettings(source, self.evaluation_thresholds()))

    def single_evaluation(self):
        if self.scope_frozen:
            self.scope_frozen = False
            if self.scope_thread is None:
                self.clear_measurement()
                self.progress_label.setText("请连接示波器")
            else:
                preview_range = self.scope_preview_range
                self.displayed_frame = None
                self.clear_evaluation()
                if preview_range is not None:
                    x_range, y_range = preview_range
                    self.plot.setRange(xRange=x_range, yRange=y_range, padding=0)
                    self.scope_curve.show()
                self.refresh_scope_frame()
            self.update_controls()
            return
        if self.simulation_toggle.isChecked():
            self.start_run(single=True)
            return
        scope = self.scope_thread
        frame = self.displayed_frame
        if self.worker is not None or frame is None or scope is None or scope.cancelled.is_set():
            return
        self.scope_frozen = True
        self.start_evaluation(ScopeEvaluationSettings(frame, self.evaluation_thresholds()))

    def start_run(self, *, single):
        if self.worker is not None or not self.simulation_toggle.isChecked():
            return
        try:
            settings = self.run_settings(single)
        except ValueError as error:
            self.append_log(f"参数有误。{error}", level="warning")
            self.progress_label.setText("请检查参数")
            return
        self.start_evaluation(settings)

    def start_evaluation(self, settings):
        scope_mode = isinstance(settings, ScopeEvaluationSettings)
        if scope_mode:
            self.clear_evaluation()
        else:
            self.scope_frozen = False
            self.clear_measurement()
        self.progress.setMaximum(settings.count)
        if scope_mode:
            self.progress_label.setText("已固定 · 评价中…")
            self.set_verdict("评价中")
            self.append_log(f"固定 {self.condition.text()} 波形并评价。", level="system")
        elif isinstance(settings, FileEvaluationSettings):
            source = settings.source_file
            name = self.condition.fontMetrics().elidedText(source.name, Qt.TextElideMode.ElideMiddle, 190)
            self.condition.setText(name)
            self.condition.setToolTip(str(source))
            self.progress_label.setText("读取文件…")
            self.set_verdict("评价中")
            self.append_log(f"打开波形文件  {source}", level="system")
        else:
            self.progress_label.setText(f"模拟 0 / {settings.count} 个工况")
            self.set_verdict("采集中")
            if settings.count == 1:
                self.append_log("开始模拟评价，共 1 个工况。", level="system")
            else:
                self.append_log(f"开始模拟扫描，共 {settings.count} 个工况。", level="system")
        thresholds = settings.thresholds
        self.alignment_limit.setText(f"允许 ±{thresholds['alignment_tolerance_pct']:g}%")
        self.raw_threshold.setText(f"要求 <{thresholds['raw_limit_pct']:g}%")
        worker = EvaluationThread(settings, self)
        self.worker = worker
        worker.measurement_ready.connect(self.show_measurement)
        worker.finished.connect(self.finish_run)
        self.update_controls()
        worker.start()

    def stop_run(self):
        if self.worker is None or self.worker.cancelled.is_set():
            return
        self.worker.cancelled.set()
        self.stop_button.setEnabled(False)
        if self.scope_frozen or not isinstance(self.worker.settings, ScopeEvaluationSettings):
            self.progress_label.setText("正在停止…")
        self.append_log("已请求停止，等待当前操作结束。", level="warning")

    def finish_run(self):
        worker = self.worker
        outcome = worker.outcome
        file_mode = isinstance(worker.settings, FileEvaluationSettings)
        stopped = worker.cancelled.is_set()
        self.worker = None
        worker.deleteLater()
        if isinstance(worker.settings, ScopeEvaluationSettings):
            self.finish_scope_evaluation(outcome, stopped)
            self.update_controls()
            if self.closing:
                self.close()
            return
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
            unit = "个文件" if file_mode else "个工况"
            self.append_log(f"流程已停止，已评价 {outcome.evaluated} {unit}。", level="warning")
        elif file_mode:
            self.progress_label.setText("文件评价完成")
            self.append_log("文件评价完成。")
        else:
            self.progress_label.setText(f"模拟完成  {outcome.evaluated} / {worker.settings.count}")
            self.append_log(f"流程完成，已评价 {outcome.evaluated} 个工况。")
        self.update_controls()
        if self.closing:
            self.close()

    def finish_scope_evaluation(self, outcome, stopped):
        if outcome.status == "failed":
            state = "评价失败"
            self.append_log(f"实机单次评价失败。{outcome.error}", level="error")
            if self.scope_frozen:
                self.set_verdict("待复核", "review")
                self.reason.setText(outcome.error)
        elif stopped or outcome.status == "stopped":
            state = "评价已停止"
            self.append_log("实机单次评价已停止。", level="warning")
            if self.scope_frozen:
                self.set_verdict("已停止")
        else:
            state = "评价完成"
            self.append_log("实机单次评价完成。")
        if self.scope_frozen:
            self.progress_label.setText(f"已固定 · {state}")

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

    def pause_scope_follow(self, axes):
        if self.scope_curve is not None and self.scope_preview_range is None and axes[0]:
            self.scope_follow_pulse = False

    def set_scope_view(self, frame):
        if frame.pulse_view_ns is None:
            self.plot.enableAutoRange(x=True)
        else:
            self.plot.setXRange(*frame.pulse_view_ns, padding=0)

    def set_scope_voltage_view(self, frame):
        if frame.voltage_range_v is None:
            self.voltage_axis.set_scope_division(None)
            self.plot.enableAutoRange(y=True)
            return
        low_v, high_v = frame.voltage_range_v
        self.voltage_axis.set_scope_division(frame.vertical_scale_v, low_v)
        self.plot.setYRange(low_v, high_v, padding=0)

    def render_scope_curve(self):
        frame = self.displayed_frame
        if frame is None or self.scope_preview_range is not None:
            return
        view = self.plot.getViewBox()
        samples = slice(None)
        if not view.autoRangeEnabled()[0]:
            left, right = view.viewRange()[0]
            first, stop = frame.time_s.searchsorted((left * 1e-9, right * 1e-9))
            # Keep samples outside the viewport for the line and peak downsampling at its edges.
            pad = max(1, ceil((stop - first) / max(1, view.width())))
            samples = slice(max(0, first - pad), min(frame.time_s.size, stop + pad))

        previous_frame, previous_samples = self.scope_rendered
        if previous_frame is frame and previous_samples == samples:
            return
        if previous_frame is not None and previous_frame.time_s is frame.time_s and previous_samples == samples:
            time_ns, _ = self.scope_curve.getOriginalDataset()
        else:
            time_ns = frame.time_s[samples] * 1e9
        self.scope_rendered = (frame, samples)
        self.scope_curve.setData(time_ns, frame.voltage_v[samples])

    def reset_zoom(self):
        frame = self.displayed_frame
        if frame is not None:
            self.set_scope_voltage_view(frame)
            if self.scope_preview_range is not None:
                x_range, _ = self.plot.getViewBox().childrenBounds()
                self.plot.setXRange(*x_range)
            else:
                self.scope_follow_pulse = True
                self.set_scope_view(frame)
                self.render_scope_curve()
            return
        self.plot.enableAutoRange()
        self.plot.autoRange()

    def clear_measurement(self):
        self.clear_evaluation()
        self.scope_curve = None
        self.displayed_frame = None
        self.scope_rendered = (None, None)
        self.plot.clear()
        self.time_axis.set_scope_division(None)
        self.voltage_axis.set_scope_division(None)
        self.plot.setLabel("bottom", PULSE_TIME_LABEL, units="ns", color="#6b7280")
        self.plot.setRange(xRange=(0, 100), yRange=(0, 1), padding=0)
        self.scope_follow_pulse = True
        self.window_legend.setText(PULSE_LEGEND)
        self.zoom_button.setEnabled(False)
        self.condition.clear()
        self.condition.setToolTip("")

    def clear_evaluation(self):
        for item in self.scope_evaluation_items:
            self.plot.removeItem(item)
        self.scope_evaluation_items.clear()
        self.scope_preview_range = None
        self.progress.setValue(0)
        self.set_verdict("待采集")
        self.show_axis_advice()
        for value in (self.alignment_value, self.raw_value):
            value.setText("—")
        for value in (self.alignment_limit, self.raw_threshold, self.pulse_caption, self.reason):
            value.clear()

    def show_measurement(self, measurement):
        report = measurement.report
        shape, flatness = report["shape"], report["flatness"]
        settings = self.worker.settings
        scope_mode = isinstance(settings, ScopeEvaluationSettings)
        file_mode = measurement.source_file is not None
        if scope_mode:
            point = f"{SCOPE_CHANNEL} · {settings.frame.received_at:%H:%M:%S.%f}"[:-3]
        elif file_mode:
            point = measurement.source_file.name
        else:
            point = f"{measurement.frequency_khz} kHz · {measurement.width_ns} ns"
        alignment = shape["score_pct"]
        aligned_value = "—" if alignment is None else f"{alignment:.2f}%"
        raw_value = f"{flatness['raw_max_pct']:.2f}%"
        verdict = measurement.verdict
        state, level = {"达标": ("pass", "success"), "未达标": ("fail", "error"),
                        "待复核": ("review", "warning")}[verdict]
        self.append_log(f"{point}  {verdict}  红蓝偏差 {aligned_value}  不平度 {raw_value}", level=level)
        if self.worker.cancelled.is_set() or (scope_mode and not self.scope_frozen):
            return
        self.progress.setValue(measurement.index)
        if not (file_mode or scope_mode):
            self.condition.setText(point)
            self.progress_label.setText(f"模拟 {measurement.index} / {settings.count} 个工况")
        self.set_verdict(verdict, state)
        self.alignment_value.setText(aligned_value)
        self.raw_value.setText(raw_value)
        self.show_axis_advice(shape)
        pulse = measurement.pulse
        pulse_count = f"{len(flatness['pulses'])} 个完整脉冲"
        self.pulse_caption.setText(f"{pulse_count} · 图示第 {pulse['number']} 个")
        self.reason.setText("保持期不平度超限。" if shape["status"] == "flat" and not report["passed"]
                            else shape["reason"])
        self.draw_pulse(pulse, report["window"], scope_mode=scope_mode)

    def add_evaluation_regions(self, window, pulse_geometry):
        regions = []
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
            pen = pg.mkPen(edge)
            for rise50_ns, width_ns in pulse_geometry:
                region = pg.LinearRegionItem([rise50_ns + part * width_ns for part in interval],
                                             movable=False, brush=fill, pen=pen)
                region.setZValue(-1)
                self.plot.addItem(region, ignoreBounds=True)
                regions.append(region)
        return regions

    def draw_pulse(self, pulse, window, *, scope_mode=False):
        if scope_mode:
            self.scope_preview_range = self.plot.viewRange()
            self.scope_curve.hide()
            self.time_axis.set_scope_division(None)
            self.plot.setLabel("bottom", PULSE_TIME_LABEL, units="ns", color="#6b7280")
            scale_v = self.displayed_frame.vertical_scale_v
            legend = PULSE_LEGEND
            if scale_v is not None:
                legend = f"{SCOPE_CHANNEL} 垂直 {pg.siFormat(scale_v, suffix='V', precision=6)}/格 · {legend}"
            self.window_legend.setText(legend)
        else:
            self.plot.clear()
        t, voltage, width = pulse["time_ns"], pulse["voltage_v"], pulse["width50_ns"]
        visible = (t >= -0.18 * width) & (t <= 1.18 * width)
        regions = self.add_evaluation_regions(window, ((0., width),))
        curve = self.plot.plot(t[visible], voltage[visible], pen=pg.mkPen("#2563eb", width=2),
                               autoDownsample=True, downsampleMethod="peak")
        if scope_mode:
            self.scope_evaluation_items = [curve, *regions]
        self.zoom_button.setEnabled(True)
        self.reset_zoom()

    def closeEvent(self, event):
        if self.board_thread is None:
            self.disconnect_board()
        if self.scope_thread is None and self.worker is None and self.board_thread is None:
            super().closeEvent(event)
            return
        self.closing = True
        if self.scope_thread is not None:
            self.scope_thread.stop()
        self.stop_run()
        self.update_controls()
        event.ignore()


def main():
    app = W.QApplication(sys.argv)
    app.setStyle("Fusion")
    app.styleHints().setColorScheme(Qt.ColorScheme.Light)
    app.setFont(QFont("Microsoft YaHei", 10))
    app.setWindowIcon(QIcon(str(Path(__file__).resolve().with_name("PreciLasers-app.ico"))))
    window = WaveformWindow()
    window.show()
    exit_code = app.exec()
    window.deleteLater()
    app.sendPostedEvents(window, QEvent.Type.DeferredDelete)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
