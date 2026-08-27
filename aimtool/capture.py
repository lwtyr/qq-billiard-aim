"""屏幕捕获：截取全屏或指定区域，返回 BGR numpy 帧。

Windows 上用 ctypes GDI BitBlt（仅 SRCCOPY，不带 CAPTUREBLT）：
CAPTUREBLT 会把分层窗口（我们自己的 overlay 瞄准线、台面框）一起
截进画面 → 识别管线把白线当成台面边缘/白球 → 检测结果抖动，
悬浮框「闪个不停」。不带 CAPTUREBLT 时分层窗口不会被捕获，问题根除。

GDI 失败时回退 mss（其实现带 CAPTUREBLT，仅作兜底）。
非 Windows 平台直接用 mss。
"""
from __future__ import annotations

import sys
import threading
from typing import List, Optional

import numpy as np

_local = threading.local()
_use_gdi = sys.platform == "win32"


# ---------- Windows GDI 直截（不含分层窗口） ----------

def _gdi_grab(x: int, y: int, w: int, h: int) -> np.ndarray:
    """BitBlt 截取屏幕区域，返回 BGR uint8。不捕获分层窗口。"""
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32

    # ctypes 默认把 Win32 返回值当作 32 位 c_int；64 位 Windows 上
    # HDC/HBITMAP 是指针大小句柄，必须显式声明，否则句柄可能被截断。
    handle = ctypes.c_void_p
    user32.GetDC.argtypes = [handle]
    user32.GetDC.restype = handle
    user32.ReleaseDC.argtypes = [handle, handle]
    user32.ReleaseDC.restype = ctypes.c_int
    gdi32.CreateCompatibleDC.argtypes = [handle]
    gdi32.CreateCompatibleDC.restype = handle
    gdi32.CreateCompatibleBitmap.argtypes = [handle, ctypes.c_int, ctypes.c_int]
    gdi32.CreateCompatibleBitmap.restype = handle
    gdi32.SelectObject.argtypes = [handle, handle]
    gdi32.SelectObject.restype = handle
    gdi32.BitBlt.argtypes = [handle, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                             ctypes.c_int, handle, ctypes.c_int, ctypes.c_int,
                             wintypes.DWORD]
    gdi32.BitBlt.restype = wintypes.BOOL
    gdi32.DeleteObject.argtypes = [handle]
    gdi32.DeleteObject.restype = wintypes.BOOL
    gdi32.DeleteDC.argtypes = [handle]
    gdi32.DeleteDC.restype = wintypes.BOOL

    hdc_screen = user32.GetDC(None)
    if not hdc_screen:
        raise OSError("GetDC(None) failed")
    mem_dc = gdi32.CreateCompatibleDC(hdc_screen)
    bmp = gdi32.CreateCompatibleBitmap(hdc_screen, w, h)
    if not mem_dc or not bmp:
        if mem_dc:
            gdi32.DeleteDC(mem_dc)
        if bmp:
            gdi32.DeleteObject(bmp)
        user32.ReleaseDC(None, hdc_screen)
        raise OSError("CreateCompatibleDC/CreateCompatibleBitmap failed")
    old_bmp = None
    try:
        old_bmp = gdi32.SelectObject(mem_dc, bmp)
        if not old_bmp:
            raise OSError("SelectObject failed")
        # 仅 SRCCOPY：不含 CAPTUREBLT → 不捕获分层窗口（overlay）
        if not gdi32.BitBlt(mem_dc, 0, 0, w, h, hdc_screen, x, y, 0x00CC0020):
            raise OSError("BitBlt failed")

        class BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [("biSize", wintypes.DWORD),
                        ("biWidth", ctypes.c_long),
                        ("biHeight", ctypes.c_long),   # 正值=自底向上
                        ("biPlanes", wintypes.WORD),
                        ("biBitCount", wintypes.WORD),
                        ("biCompression", wintypes.DWORD),
                        ("biSizeImage", wintypes.DWORD),
                        ("biXPelsPerMeter", ctypes.c_long),
                        ("biYPelsPerMeter", ctypes.c_long),
                        ("biClrUsed", wintypes.DWORD),
                        ("biClrImportant", wintypes.DWORD)]

        gdi32.GetDIBits.argtypes = [handle, handle, wintypes.UINT, wintypes.UINT,
                                    ctypes.c_void_p,
                                    ctypes.POINTER(BITMAPINFOHEADER), wintypes.UINT]
        gdi32.GetDIBits.restype = ctypes.c_int

        bmi = BITMAPINFOHEADER()
        bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.biWidth = w
        bmi.biHeight = -h            # 负值=自顶向下行序，免翻转
        bmi.biPlanes = 1
        bmi.biBitCount = 32
        bmi.biCompression = 0        # BI_RGB

        # GetDIBits 要求目标位图不能仍被选入 DC。旧实现把
        # SelectObject 放到了 finally，导致这里在 Windows 上经常返回 0，
        # 随后 grab() 永久退回 mss；mss 又可能把分层 Overlay 一起截进来，
        # 形成「Overlay -> 识别 -> Overlay」的自反馈抖动。
        if not gdi32.SelectObject(mem_dc, old_bmp):
            raise OSError("restore SelectObject failed")
        old_bmp = None

        buf = ctypes.create_string_buffer(w * h * 4)
        if gdi32.GetDIBits(mem_dc, bmp, 0, h, buf, ctypes.byref(bmi), 0) <= 0:
            raise OSError("GetDIBits failed")
        frame = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
        return frame[:, :, :3].copy()   # BGRA → BGR
    finally:
        if old_bmp:
            gdi32.SelectObject(mem_dc, old_bmp)
        gdi32.DeleteObject(bmp)
        gdi32.DeleteDC(mem_dc)
        user32.ReleaseDC(None, hdc_screen)


# ---------- mss（回退 / 非 Windows） ----------

_local = threading.local()


def _get_mss():
    """取当前线程的 mss 实例；失效时重建一次（显示器热插拔等场景）。"""
    sct = getattr(_local, "sct", None)
    if sct is None:
        import mss
        sct = _local.sct = mss.mss()
    return sct


def _mss_grab(region: Optional[List[int]] = None) -> np.ndarray:
    for attempt in (0, 1):                      # 第二次尝试=重建实例
        try:
            sct = _get_mss()
            if region:
                x, y, w, h = region
                monitor = {"left": int(x), "top": int(y), "width": int(w), "height": int(h)}
            else:
                monitor = sct.monitors[1]  # 主显示器
            shot = sct.grab(monitor)
            return np.asarray(shot)[:, :, :3].copy()
        except Exception:
            if attempt == 1:
                raise
            _local.sct = None                   # 置空，下一轮重建


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
            # 主显示器尺寸（物理像素，与 DPI aware 行为一致）
            import ctypes
            w = ctypes.windll.user32.GetSystemMetrics(0)
            h = ctypes.windll.user32.GetSystemMetrics(1)
            x, y = 0, 0
        try:
            return _gdi_grab(int(x), int(y), int(w), int(h))
        except Exception as exc:
            gdi_error = exc
            _use_gdi = False            # GDI 异常时本次回退 mss，后续也走 mss
    try:
        return _mss_grab(region)
    except Exception as exc:
        if gdi_error is not None:
            raise RuntimeError(
                f"GDI 截屏失败: {gdi_error}; mss 回退失败: {exc}") from exc
        raise


def monitor_size() -> tuple:
    if _use_gdi:
        import ctypes
        return (ctypes.windll.user32.GetSystemMetrics(0),
                ctypes.windll.user32.GetSystemMetrics(1))
    sct = _get_mss()
    m = sct.monitors[1]
    return (m["width"], m["height"])
