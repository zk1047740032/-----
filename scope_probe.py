"""Read a Tektronix oscilloscope's SCPI identification over its socket server."""

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtNetwork import QHostAddress, QTcpSocket


SCOPE_PORT = 4000
TIMEOUT_MS = 3000
MAX_REPLY_BYTES = 512


class ScopeProbe(QObject):
    finished = Signal(bool, str)

    def __init__(self, address, parent=None):
        super().__init__(parent)
        self.address = address
        self.connected = False
        self.active = False
        self.socket = QTcpSocket(self)
        self.socket.setReadBufferSize(MAX_REPLY_BYTES + 1)
        self.socket.connected.connect(self.query_identity)
        self.socket.readyRead.connect(self.read_identity)
        self.socket.errorOccurred.connect(self.socket_error)
        self.timer = QTimer(self, singleShot=True, interval=TIMEOUT_MS)
        self.timer.timeout.connect(self.timed_out)

    def start(self):
        self.active = True
        self.timer.start()
        self.socket.connectToHost(QHostAddress(self.address), SCOPE_PORT)

    def stop(self):
        self.active = False
        self.timer.stop()
        self.socket.abort()

    def finish(self, success, detail):
        self.stop()
        self.finished.emit(success, detail)

    def query_identity(self):
        if not self.active:
            return
        self.connected = True
        if self.socket.write(b"*IDN?\n") < 0 and self.active:
            self.finish(False, f"发送 *IDN? 失败。{self.socket.errorString()}")

    def read_identity(self):
        if not self.active:
            return
        if self.socket.canReadLine():
            raw = bytes(self.socket.readLine(MAX_REPLY_BYTES + 1))
            identity = raw.decode("ascii", errors="replace").rstrip("\r\n").strip()
            fields = identity.split(",", 3)
            if (len(raw) <= MAX_REPLY_BYTES and raw.endswith(b"\n")
                    and identity.isascii() and identity.isprintable()
                    and len(fields) == 4 and all(part.strip() for part in fields)):
                self.finish(True, identity)
            else:
                self.finish(False, "收到的数据不是有效的 *IDN? 身份应答。请检查 Socket Server 的 Protocol 设置。")
        elif self.socket.bytesAvailable() >= MAX_REPLY_BYTES:
            self.finish(False, "*IDN? 应答过长或缺少换行。")

    def timed_out(self):
        detail = "*IDN? 应答超时。" if self.connected else "TCP 连接超时。"
        self.finish(False, detail)

    def socket_error(self, _error):
        if not self.active:
            return
        if self.socket.canReadLine():
            self.read_identity()
            return
        detail = "接收 *IDN? 应答失败。" if self.connected else "TCP 连接失败。"
        self.finish(False, f"{detail}{self.socket.errorString()}")
