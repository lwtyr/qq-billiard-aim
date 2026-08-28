"""屏幕捕获：截取全屏或指定区域，返回 BGR numpy 帧。

Windows 上用 ctypes GDI BitBlt（仅 SRCCOPY，不带 CAPTUREBLT）：
CAPTUREBLT 会把分层窗口（我们自己的 overlay 瞄准线、台面框）一起
截进画面 → 识别管线把白线当成台面边缘/白球 → 检测结果抖动，
悬浮框「闪个不停」。不带 CAPTUREBLT 时分层窗口不会被捕获，问题根除。

优化：
- 模块级预声明 Win32 API 签名与 BITMAPINFOHEADER 结构体，消除每帧类定义开销；
- 缓存复用 (mem_dc, bmp, buf, bmi)，消除重复构造与内存分配；
- 使用 OpenCV SIMD 向量化 cv2.COLOR_BGRA2BGR 替代 Python 跨步切片 copy，单帧加速数十倍。

GDI 失败时回退 mss（其实现带 CAPTUREBLT，仅作兜底）。
非 Windows 平台直接用 mss。
"""
from __future__ import annotations

import sys
import threading
from typing import List, Optional

import cv2
import numpy as np

_local = threading.local()
_use_gdi = sys.platform == "win32"

if _use_gdi:
    import ctypes
    from ctypes import wintypes

    _handle = ctypes.c_void_p
    _user32 = ctypes.windll.user32
    _gdi32 = ctypes.windll.gdi32

    _user32.GetDC.argtypes = [_handle]
    _user32.GetDC.restype = _handle
    _user32.ReleaseDC.argtypes = [_handle, _handle]
    _user32.ReleaseDC.restype = ctypes.c_int

    _gdi32.CreateCompatibleDC.argtypes = [_handle]
    _gdi32.CreateCompatibleDC.restype = _handle
    _gdi32.CreateCompatibleBitmap.argtypes = [_handle, ctypes.c_int, ctypes.c_int]
    _gdi32.CreateCompatibleBitmap.restype = _handle
    _gdi32.SelectObject.argtypes = [_handle, _handle]
    _gdi32.SelectObject.restype = _handle
    _gdi32.BitBlt.argtypes = [
        _handle, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        _handle, ctypes.c_int, ctypes.c_int, wintypes.DWORD
    ]
    _gdi32.BitBlt.restype = wintypes.BOOL
    _gdi32.DeleteObject.argtypes = [_handle]
    _gdi32.DeleteObject.restype = wintypes.BOOL
    _gdi32.DeleteDC.argtypes = [_handle]
    _gdi32.DeleteDC.restype = wintypes.BOOL

    class BITMAPINFOHEADER(ctypes.Structure):
        _fields_ = [
            ("biSize", wintypes.DWORD),
            ("biWidth", ctypes.c_long),
            ("biHeight", ctypes.c_long),   # 负值=自顶向下行序
            ("biPlanes", wintypes.WORD),
            ("biBitCount", wintypes.WORD),
            ("biCompression", wintypes.DWORD),
            ("biSizeImage", wintypes.DWORD),
            ("biXPelsPerMeter", ctypes.c_long),
            ("biYPelsPerMeter", ctypes.c_long),
            ("biClrUsed", wintypes.DWORD),
            ("biClrImportant", wintypes.DWORD),
        ]

    _gdi32.GetDIBits.argtypes = [
        _handle, _handle, wintypes.UINT, wintypes.UINT,
        ctypes.c_void_p, ctypes.POINTER(BITMAPINFOHEADER), wintypes.UINT
    ]
    _gdi32.GetDIBits.restype = ctypes.c_int
else:
    ctypes = None
    _user32 = None
    _gdi32 = None


# ---------- Windows GDI 直截（不含分层窗口） ----------

def _gdi_release() -> None:
    """释放当前线程缓存的 GDI 截屏资源（尺寸变化/失败/回退时）。"""
    st = getattr(_local, "gdi", None)
    if st is None:
        return
    mem_dc, bmp = st[0], st[1]
    try:
        if mem_dc:
            _gdi32.DeleteDC(mem_dc)
    finally:
        if bmp:
            _gdi32.DeleteObject(bmp)
    _local.gdi = None


def _gdi_grab(x: int, y: int, w: int, h: int) -> np.ndarray:
    """BitBlt 截取屏幕区域，返回 BGR uint8。不捕获分层窗口。

    mem DC / 兼容位图 / 行缓冲 / BITMAPINFOHEADER 按（线程, 尺寸）缓存复用。
    """
    hdc_screen = _user32.GetDC(None)
    if not hdc_screen:
        raise OSError("GetDC(None) failed")
    try:
        st = getattr(_local, "gdi", None)
        if st is not None and (st[2] != w or st[3] != h):
            _gdi_release()
            st = None
        if st is None:
            mem_dc = _gdi32.CreateCompatibleDC(hdc_screen)
            bmp = _gdi32.CreateCompatibleBitmap(hdc_screen, w, h)
            if not mem_dc or not bmp:
                if mem_dc:
                    _gdi32.DeleteDC(mem_dc)
                if bmp:
                    _gdi32.DeleteObject(bmp)
                raise OSError("CreateCompatibleDC/CreateCompatibleBitmap failed")
            buf = ctypes.create_string_buffer(w * h * 4)

            bmi = BITMAPINFOHEADER()
            bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
            bmi.biWidth = w
            bmi.biHeight = -h            # 负值=自顶向下行序，免翻转
            bmi.biPlanes = 1
            bmi.biBitCount = 32
            bmi.biCompression = 0        # BI_RGB

            st = (mem_dc, bmp, w, h, buf, bmi)
            _local.gdi = st
        mem_dc, bmp, _, _, buf, bmi = st

        old_bmp = _gdi32.SelectObject(mem_dc, bmp)
        if not old_bmp:
            raise OSError("SelectObject failed")
        try:
            # 仅 SRCCOPY：不含 CAPTUREBLT → 不捕获分层窗口（overlay）
            if not _gdi32.BitBlt(mem_dc, 0, 0, w, h, hdc_screen, x, y, 0x00CC0020):
                raise OSError("BitBlt failed")
        finally:
            _gdi32.SelectObject(mem_dc, old_bmp)

        if _gdi32.GetDIBits(mem_dc, bmp, 0, h, buf, ctypes.byref(bmi), 0) <= 0:
            raise OSError("GetDIBits failed")

        bgra = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
        # 使用 OpenCV SIMD 快速转换 BGRA -> BGR，比 Python 跨步切片 copy 快数十倍
        return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
    except Exception:
        _gdi_release()
        raise
    finally:
        _user32.ReleaseDC(None, hdc_screen)


# ---------- mss（回退 / 非 Windows） ----------

def _get_mss():
    """取当前线程的 mss 实例；失效时重建一次（显示器热插拔等场景）。"""
    sct = getattr(_local, "sct", None)
    if sct is None:
        import mss
        sct = _local.sct = mss.mss()
    return sct


def _mss_grab(region: Optional[List[int]] = None) -> np.ndarray:
    for attempt in (0, 1):
        try:
            sct = _get_mss()
            if region:
                x, y, w, h = region
                monitor = {"left": int(x), "top": int(y), "width": int(w), "height": int(h)}
            else:
                monitor = sct.monitors[1]  # 主显示器
            shot = sct.grab(monitor)
            bgra = np.asarray(shot)
            return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
        except Exception:
            if attempt == 1:
                raise
            _local.sct = None


def grab(region: Optional[List[int]] = None) -> np.ndarray:
    """截屏。region = [x, y, w, h]（屏幕坐标）；None = 主屏全屏。

    返回 BGR 三通道 uint8 数组。
    """
    global _use_gdi
    gdi_error = None
    if _use_gdi:
        if region:
            x, y, w, h = region
        else:
            w = _user32.GetSystemMetrics(0)
            h = _user32.GetSystemMetrics(1)
            x, y = 0, 0
        try:
            return _gdi_grab(int(x), int(y), int(w), int(h))
        except Exception as exc:
            gdi_error = exc
            _use_gdi = False
    try:
        return _mss_grab(region)
    except Exception as exc:
        if gdi_error is not None:
            raise RuntimeError(
                f"GDI 截屏失败: {gdi_error}; mss 回退失败: {exc}") from exc
        raise


def monitor_size() -> tuple:
    if _use_gdi:
        return (_user32.GetSystemMetrics(0), _user32.GetSystemMetrics(1))
    sct = _get_mss()
    m = sct.monitors[1]
    return (m["width"], m["height"])
