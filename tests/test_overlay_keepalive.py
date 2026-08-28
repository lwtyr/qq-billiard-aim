"""overlay 置顶保活回归测试（bug5）。

bug5 现场：v3.6.3 的保活代码在 draw() 里引用了模块级不存在的
wintypes（本模块为 __init__ 局部导入），每帧抛 NameError，
导致置顶保活从未执行 → QQ 游戏窗口（同为 TOPMOST）把选框整层
盖住，用户看到「瞄准线消失了」。本测试用假 WinAPI 验证保活
逻辑本身正确：不抛异常、节流正常、每 15 帧重申一次置顶。
"""
import numpy as np
from ctypes import wintypes

from aimtool import native_overlay as no

HWND_TOPMOST = -1


def _fake_layer():
    layer = no.NativeLayer.__new__(no.NativeLayer)   # 绕过 Windows-only 构造
    layer.sh, layer.sw = 8, 8
    layer._hwnd = object()                           # 非 None 即可
    layer._last_push = 0.0
    layer._push_interval = 0.0                       # 每次调用都放行
    layer._bits = np.zeros((8, 8, 4), np.uint8)
    layer.drawn_mask = None
    layer._push_calls = []
    layer._push = lambda: layer._push_calls.append(1)
    layer._render = lambda scene, region: np.zeros((8, 8, 4), np.uint8)
    layer._wintypes = wintypes

    class _U32:
        def __init__(self):
            self.calls = []

        def SetWindowPos(self, hwnd, after, x, y, cx, cy, flags):
            self.calls.append(after)
            return 1

    layer._u32 = _U32()
    return layer


def _topmost_value():
    """与 draw() 内部一致：wintypes.HWND(-1) 的无符号值（平台无关）。"""
    return wintypes.HWND(HWND_TOPMOST).value


def test_topmost_keepalive_runs_without_nameerror(monkeypatch):
    """保活路径不得抛 NameError（bug5 根因），且不破坏推送节流。"""
    monkeypatch.setattr(no, "_PIL_OK", True)
    layer = _fake_layer()
    for _ in range(20):
        layer.draw({})                    # 修复前：tick=15 处 NameError
    assert len(layer._push_calls) == 20   # 每帧都推送
    assert layer._topmost_tick == 20
    assert [c.value for c in layer._u32.calls] == [_topmost_value()]  # 仅第 15 帧


def test_topmost_keepalive_periodic(monkeypatch):
    """每 _TOPMOST_REFRESH_FRAMES 帧恰好重申一次置顶。"""
    monkeypatch.setattr(no, "_PIL_OK", True)
    layer = _fake_layer()
    for _ in range(45):
        layer.draw({})
    assert [c.value for c in layer._u32.calls] == [_topmost_value()] * 3
