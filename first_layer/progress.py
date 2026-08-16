"""
简单控制台进度条（线程安全，原地刷新）

用法:
    bar = ProgressBar(total=9, label="OCR")
    for i in range(9):
        bar.update(i + 1, detail=f"frame_{i+1}.jpg")
    bar.finish()
"""

from __future__ import annotations

import sys
import threading


class ProgressBar:
    """用 \\r 原地刷新进度条，避免打印大量逐行日志。

    仅依赖标准库，Windows/PowerShell 下可用（不使用 ANSI 转义序列）。
    多线程场景下通过内部锁保证同一时刻只有一个进度条在写终端。
    """

    _LINE_WIDTH = 88  # 固定行宽，用于覆盖上一行的残留字符

    def __init__(
        self,
        total: int,
        label: str = "",
        width: int = 30,
        stream=None,
    ):
        self.total = max(int(total), 1)
        self.label = label
        self.width = width
        self.stream = stream if stream is not None else sys.stderr
        self._lock = threading.Lock()
        self._finished = False

    def _render(self, current: int, detail: str = "") -> str:
        ratio = min(current / self.total, 1.0)
        filled = int(self.width * ratio)
        bar = "#" * filled + "-" * (self.width - filled)
        pct = int(ratio * 100)
        head = f"{self.label} " if self.label else ""
        line = f"\r{head}[{bar}] {pct:3d}% ({current}/{self.total})"
        if detail:
            line += f"  {detail}"
        # 右侧补空格，覆盖上一次更长内容留下的字符
        return line.ljust(self._LINE_WIDTH)

    def update(self, current: int, detail: str = "") -> None:
        with self._lock:
            self.stream.write(self._render(current, detail))
            self.stream.flush()

    def finish(self, message: str = "") -> None:
        with self._lock:
            if self._finished:
                return
            self._finished = True
            # 清空当前行并换行，避免与后续日志粘连
            self.stream.write("\r" + " " * self._LINE_WIDTH + "\r")
            if message:
                self.stream.write(message + "\n")
            else:
                self.stream.write("\n")
            self.stream.flush()
