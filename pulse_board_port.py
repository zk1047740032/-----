"""Hold a Windows COM port with the line settings already configured in Windows."""

import ctypes
import re

from serial import win32


class PulseBoardPort:
    def __init__(self, port):
        if not re.fullmatch(r"COM[1-9][0-9]*", port, re.IGNORECASE):
            raise ValueError("请选择有效的 COM 串口。")
        self.port = port
        self.handle = win32.CreateFile(
            "\\\\.\\" + port, win32.GENERIC_READ | win32.GENERIC_WRITE, 0,
            None, win32.OPEN_EXISTING, 0, None,
        )
        if self.handle == win32.INVALID_HANDLE_VALUE:
            raise ctypes.WinError()

    def close(self):
        if not win32.CloseHandle(self.handle):
            raise ctypes.WinError()
