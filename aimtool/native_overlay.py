"""全屏透明瞄准层：原生分层窗口 + UpdateLayeredWindow 逐像素 alpha 直绘。

为什么不用 Tk 的 -transparentcolor：
  系统对分层窗口的命中测试只认「逐像素 alpha」（UpdateLayeredWindow +
  AC_SRC_ALPHA）：alpha=0 的像素系统不发送鼠标消息（文档行为，跨进程），
  点击自动落到下层窗口。而 Tk 的 -transparentcolor 走的是颜色键
  （LWA_COLORKEY），命中测试仍按整个窗口矩形计算——全屏置顶时会把
  全系统点击吞掉（症状：所有窗口假死、左右键失效、键盘正常）。
  WM_NCHITTEST → HTTRANSPARENT 子类化跨进程/不同系统版本行为不稳定，
  不能作为唯一手段。因此本模块：每帧把瞄准画面用 PIL 画进 32 位 BGRA
  位图（背景 alpha=0，只有线条/文字像素 alpha=255），经 UpdateLayeredWindow
  推送到全屏分层窗口。透明像素天然穿透（系统级），线条像素才可点。

环境变量：
  QQ_AIM_TK_OVERLAY=1  强制退回 Tk 透明色窗口（诊断用）

非 Windows 环境：本模块不做任何事（demo/测试路径不依赖 tk）。
"""
from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

_PIL_OK = False
try:                                        # pragma: no cover - 环境差异
    from PIL import Image, ImageDraw, ImageFont
    _PIL_OK = True
except Exception:
    _PIL_OK = False

# 绘制配色（与旧 Tk 渲染保持一致）
C_AIM = "#22c55e"             # 母球→鬼球（绿）
C_TARGET = "#f97316"          # 鬼球→目标球（橙）
C_POCKET = "#facc15"          # 目标球→袋口（黄）
C_KICK = "#38bdf8"            # 库边反弹段（天蓝）
C_GHOST = "#e2e8f0"           # 鬼球虚线圆
C_EDGE = "#ffffff"
C_TEXT = "#ffffff"
C_TEXT_BG = "#111827"
C_HINT = "#f87171"

_WS_EX_LAYERED = 0x00080000
_WS_EX_TOPMOST = 0x00000008
_WS_EX_NOACTIVATE = 0x08000000
_WS_EX_TOOLWINDOW = 0x00000080
_WS_EX_TRANSPARENT = 0x00000020   # 整窗命中穿透：鼠标消息交给下层窗口
_WS_POPUP = 0x80000000

_SM_CXSCREEN = 0
_SM_CYSCREEN = 1

_ULW_ALPHA = 0x00000002
_AC_SRC_OVER = 0x00000000
_AC_SRC_ALPHA = 0x00000001
_SW_HIDE = 0
_SW_SHOWNOACTIVATE = 4
_DIB_RGB_COLORS = 0

_FONT_CANDIDATES = (
    "C:/Windows/Fonts/msyhbd.ttc",   # 微软雅黑 粗体
    "C:/Windows/Fonts/msyh.ttc",     # 微软雅黑
    "C:/Windows/Fonts/simhei.ttf",   # 黑体
    "C:/Windows/Fonts/msyh.ttf",
)

_coord = Tuple[int, int]


def _bgra_premultiplied(rgba: np.ndarray) -> np.ndarray:
    """RGBA(0..255) 直通 alpha → BGRA 预乘 alpha（UpdateLayeredWindow 要求）。"""
    out = np.empty_like(rgba)
    a = rgba[:, :, 3:4].astype(np.float32) / 255.0
    out[:, :, 0] = rgba[:, :, 2] * a[:, :, 0]   # B
    out[:, :, 1] = rgba[:, :, 1] * a[:, :, 0]   # G
    out[:, :, 2] = rgba[:, :, 0] * a[:, :, 0]   # R
    out[:, :, 3] = rgba[:, :, 3]
    return np.ascontiguousarray(out, dtype=np.uint8)


class NativeLayer:
    """一个全屏置顶的逐像素透明层（UpdateLayeredWindow）。

    只在本模块为 Windows 且 PIL 可用时使用（非 Windows 下构造抛
    UnsupportedError，调用方退回 Tk 路径）。
    """

    def __init__(self, width: int, height: int):
        if os.name != "nt" or not _PIL_OK:
            raise RuntimeError("NativeLayer 需要 Windows + Pillow")
        import ctypes
        from ctypes import wintypes, windll

        self.sw = int(width)
        self.sh = int(height)
        self._ctypes = ctypes
        self._wintypes = wintypes
        self._u32 = windll.user32
        self._gdi32 = windll.gdi32
        self._k32 = windll.kernel32
        self._hwnd = None
        self._hdc_screen = None
        self._memdc = None
        self._hbmp = None
        self._bits = None            # numpy 视图（shape sh,sw,4），直接写屏
        self.drawn_mask: Optional[np.ndarray] = None   # 实际画过的像素（alpha>0），供截屏自清理
        self._lock = None
        self._last_push = 0.0
        try:
            overlay_fps = float(os.environ.get("QQ_AIM_OVERLAY_FPS", "30"))
        except (TypeError, ValueError):
            overlay_fps = 30.0
        # Overlay 的旧 15fps 上限会额外引入最多约 66ms 的点位延迟；
        # 默认与 App 的 30fps UI tick 对齐，仍允许低功耗机器显式降频。
        self._push_interval = 1.0 / max(1.0, min(60.0, overlay_fps))
        self._create_window()
        self._create_dib()
        # 字体
        self._fonts: Dict[int, object] = {}
        self._load_fonts()

    # ---------- Win32 初始化 ----------
    def _create_window(self) -> None:
        ctypes, wintypes = self._ctypes, self._wintypes
        u32 = self._u32
        u32.CreateWindowExW.argtypes = [
            wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE, ctypes.c_void_p]
        u32.CreateWindowExW.restype = wintypes.HWND
        ex = (_WS_EX_LAYERED | _WS_EX_TOPMOST | _WS_EX_NOACTIVATE
              | _WS_EX_TOOLWINDOW | _WS_EX_TRANSPARENT)
        k32 = self._k32
        k32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
        k32.GetModuleHandleW.restype = wintypes.HMODULE
        hwnd = u32.CreateWindowExW(ex, "STATIC", "QQ2D桌球瞄准器 Overlay",
                                   _WS_POPUP,
                                   0, 0, self.sw, self.sh,
                                   None, None, k32.GetModuleHandleW(None), None)
        if not hwnd:
            raise ctypes.WinError(ctypes.get_last_error())
        self._hwnd = hwnd
        # 确保置顶并立即应用（TOPWINDOW 顺序；不动尺寸/激活状态）
        u32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int,
                                     ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                     wintypes.UINT]
        u32.SetWindowPos.restype = wintypes.BOOL
        u32.SetWindowPos(hwnd, wintypes.HWND(-1), 0, 0, 0, 0,
                         0x0001 | 0x0002 | 0x0010)   # NOSIZE|NOMOVE|NOACTIVATE
        u32.ShowWindow(hwnd, _SW_SHOWNOACTIVATE)

    def _create_dib(self) -> None:
        ctypes, wintypes = self._ctypes, self._wintypes
        u32, gdi32 = self._u32, self._gdi32
        u32.GetDC.argtypes = [wintypes.HWND]
        u32.GetDC.restype = wintypes.HDC
        u32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
        u32.ReleaseDC.restype = ctypes.c_int
        gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
        gdi32.CreateCompatibleDC.restype = wintypes.HDC
        gdi32.DeleteDC.argtypes = [wintypes.HDC]
        gdi32.DeleteDC.restype = ctypes.c_int
        gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
        gdi32.DeleteObject.restype = ctypes.c_int

        self._hdc_screen = u32.GetDC(None)        # 全屏 DC
        if not self._hdc_screen:
            raise ctypes.WinError(ctypes.get_last_error())
        self._memdc = gdi32.CreateCompatibleDC(self._hdc_screen)
        if not self._memdc:
            raise ctypes.WinError(ctypes.get_last_error())

        class BITMAPINFOHEADER(ctypes.Structure):       # pragma: no cover
            _fields_ = [("biSize", wintypes.DWORD),
                        ("biWidth", ctypes.c_long),
                        ("biHeight", ctypes.c_long),
                        ("biPlanes", wintypes.WORD),
                        ("biBitCount", wintypes.WORD),
                        ("biCompression", wintypes.DWORD),
                        ("biSizeImage", wintypes.DWORD),
                        ("biXPelsPerMeter", ctypes.c_long),
                        ("biYPelsPerMeter", ctypes.c_long),
                        ("biClrUsed", wintypes.DWORD),
                        ("biClrImportant", wintypes.DWORD)]

        class BITMAPINFO(ctypes.Structure):             # pragma: no cover
            _fields_ = [("bmiHeader", BITMAPINFOHEADER)]

        bmi = BITMAPINFO()
        bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.bmiHeader.biWidth = self.sw
        bmi.bmiHeader.biHeight = -self.sh          # 负数 = 自上而下
        bmi.bmiHeader.biPlanes = 1
        bmi.bmiHeader.biBitCount = 32
        gdi32.CreateDIBSection.argtypes = [wintypes.HDC, ctypes.c_void_p,
                                           wintypes.UINT,
                                           ctypes.POINTER(ctypes.c_void_p),
                                           wintypes.HANDLE, wintypes.DWORD]
        gdi32.CreateDIBSection.restype = wintypes.HBITMAP
        bits_ptr = ctypes.c_void_p()
        self._hbmp = gdi32.CreateDIBSection(
            self._memdc, ctypes.byref(bmi), _DIB_RGB_COLORS,
            ctypes.byref(bits_ptr), None, 0)
        if not self._hbmp:
            raise ctypes.WinError(ctypes.get_last_error())
        gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
        gdi32.SelectObject.restype = wintypes.HGDIOBJ
        if not gdi32.SelectObject(self._memdc, self._hbmp):
            raise ctypes.WinError(ctypes.get_last_error())
        # 位图内存 → numpy 视图（每帧直接原位写入）
        arr_type = ctypes.c_uint8 * (self.sh * self.sw * 4)
        buf = ctypes.cast(bits_ptr, ctypes.POINTER(arr_type)).contents
        self._bits = np.ctypeslib.as_array(buf).reshape(self.sh, self.sw, 4)
        self._bits.fill(0)

        # UpdateLayeredWindow 参数
        class POINT(ctypes.Structure):                  # pragma: no cover
            _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]
        class SIZE(ctypes.Structure):                   # pragma: no cover
            _fields_ = [("cx", ctypes.c_long), ("cy", ctypes.c_long)]
        class BLENDFUNCTION(ctypes.Structure):          # pragma: no cover
            _fields_ = [("BlendOp", ctypes.c_ubyte),
                        ("BlendFlags", ctypes.c_ubyte),
                        ("SourceConstantAlpha", ctypes.c_ubyte),
                        ("AlphaFormat", ctypes.c_ubyte)]
        self._pt_dst = POINT(0, 0)
        self._size = SIZE(self.sw, self.sh)
        self._pt_src = POINT(0, 0)
        self._blend = BLENDFUNCTION(_AC_SRC_OVER, 0, 255, _AC_SRC_ALPHA)
        self._u32.UpdateLayeredWindow.argtypes = [
            wintypes.HWND, wintypes.HDC, ctypes.c_void_p, ctypes.c_void_p,
            wintypes.HDC, ctypes.c_void_p, wintypes.COLORREF, ctypes.c_void_p,
            wintypes.DWORD]
        self._u32.UpdateLayeredWindow.restype = wintypes.BOOL

    def _load_fonts(self) -> None:
        try:
            for size, path in ((18, _FONT_CANDIDATES[0]),
                               (14, _FONT_CANDIDATES[0]),
                               (11, _FONT_CANDIDATES[1])):
                for cand in (path, *_FONT_CANDIDATES):
                    try:
                        self._fonts[size] = ImageFont.truetype(cand, size)
                        break
                    except Exception:
                        continue
        except Exception:
            self._fonts = {}

    def _font(self, size: int):
        return self._fonts.get(size) or ImageFont.load_default()

    # ---------- 对外 ----------
    def show(self) -> None:
        if self._hwnd:
            self._u32.ShowWindow(self._hwnd, _SW_SHOWNOACTIVATE)

    def hide(self) -> None:
        self.drawn_mask = None          # 隐藏期间窗口不可见，无需自清理
        if self._hwnd:
            self._u32.ShowWindow(self._hwnd, _SW_HIDE)

    def draw(self, scene: Dict, region: Optional[Tuple[int, int, int, int]] = None) -> None:
        """按场景直绘一帧并推送（带 15fps 节流）。"""
        if not self._hwnd or not _PIL_OK:
            return
        now = time.monotonic()
        if now - self._last_push < self._push_interval:
            return
        try:
            if scene is None:
                # 尚无场景：也推一帧全透明，让逐像素穿透从第一帧就生效
                self._bits.fill(0)
                self.drawn_mask = np.zeros((self.sh, self.sw), dtype=np.uint8)
                self._push()
                self._last_push = now
                return
            rgba = self._render(scene, region)
            pm = _bgra_premultiplied(rgba)
            self._bits[:] = pm
            # 登记自绘像素：截屏端拿它把自家画的线填回台呢色，避免
            # 「自己画的内容被识别管线当成遮挡/异色块」。
            self.drawn_mask = (self._bits[:, :, 3] > 0).astype(np.uint8)
            self._push()
            self._last_push = now
        except Exception as exc:
            print(f"[overlay] 直绘异常: {type(exc).__name__}: {exc}", flush=True)

    def _push(self) -> None:
        ctypes = self._ctypes
        self._u32.UpdateLayeredWindow(
            self._hwnd, self._hdc_screen, ctypes.byref(self._pt_dst),
            ctypes.byref(self._size), self._memdc, ctypes.byref(self._pt_src),
            0, ctypes.byref(self._blend), _ULW_ALPHA)

    def close(self) -> None:
        self.drawn_mask = None
        if self._hwnd:
            self._u32.DestroyWindow(self._hwnd)
            self._hwnd = None
        if self._hbmp:
            self._gdi32.DeleteObject(self._hbmp)
            self._hbmp = None
        if self._memdc:
            self._gdi32.DeleteDC(self._memdc)
            self._memdc = None
        if self._hdc_screen:
            self._u32.ReleaseDC(None, self._hdc_screen)
            self._hdc_screen = None

    # ---------- 绘制 ----------
    def _render(self, scene: Dict, region: Optional[Tuple[int, int, int, int]]):
        img = Image.new("RGBA", (self.sw, self.sh), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)

        # R 键框选矩形（实时）
        if region:
            x0, y0, x1, y1 = region
            self._dashed(d, [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)],
                         "#ffffff", 2, (6, 4))
            d.text(((x0 + x1) // 2, min(y0, y1) - 14), "松开左键完成框选",
                   font=self._font(14), fill="#ffffff")

        # 台面边框
        quad = scene.get("table_quad")
        if quad:
            pts = [tuple(p) for p in quad] + [tuple(quad[0])]
            d.line(pts, fill=C_EDGE, width=2, joint="curve")

        # 袋口：只画灰色小点，不画选中袋口的橙色圈（影响视觉；
        # 选中袋口由左上状态文字「袋口N」标明）。
        for p in scene.get("pockets", []):
            if p.get("sel"):
                continue
            x, y = int(p["x"]), int(p["y"])
            d.ellipse([x - 4, y - 4, x + 4, y + 4], outline="#94a3b8", width=2)

        # 瞄准线段。黄色「目标球→袋口」是预测路径不是瞄准方向，
        # 降低透明度以免用户误以为是对着它打。
        for seg in scene.get("segments", []):
            pts = [(int(x), int(y)) for x, y in seg["pts"]]
            color = seg.get("color", C_AIM)
            width = seg.get("width", 5)
            if seg.get("color") == C_POCKET:
                fill = (250, 204, 21, 150)          # 预测路径：半透明
            else:
                fill = color
            if seg.get("dash"):
                self._dashed(d, pts, fill, width, (12, 8))
            else:
                d.line(pts, fill=fill, width=width, joint="curve")

        # 球体
        cue = scene.get("cue")
        target = scene.get("target")
        for b in scene.get("balls", []):
            r = max(4.0, float(b.get("r", 12)))
            x, y = float(b["x"]), float(b["y"])
            is_cue = cue is not None and abs(x - cue["x"]) < r and abs(y - cue["y"]) < r
            is_tgt = (target is not None
                      and abs(x - target["x"]) < r and abs(y - target["y"]) < r)
            if is_cue:
                e = r * 0.9
                d.ellipse([x - e, y - e, x + e, y + e],
                          fill="#ffffff", outline=C_EDGE, width=2)
                e = r * 1.5
                d.ellipse([x - e, y - e, x + e, y + e], outline=C_AIM, width=2)
            elif is_tgt:
                e = r * 1.5
                d.ellipse([x - e, y - e, x + e, y + e], outline=C_TARGET, width=3)

        # 鬼球（绿色虚线圆 + 中心十字）—— 唯一瞄准目标：白球中心
        # 要穿过虚线圆的圆心，而不是打目标球表面上的点。
        g = scene.get("ghost")
        if g:
            r = float(g.get("r", scene.get("ball_r", 12)))
            gx, gy = float(g["x"]), float(g["y"])
            circ = self._circle_points(gx, gy, r, 64)
            self._dashed(d, circ, C_AIM, 2, (10, 7))
            arm = max(5.0, min(10.0, 0.35 * r))
            d.line([(gx - arm, gy), (gx + arm, gy)], fill=C_GHOST, width=2)
            d.line([(gx, gy - arm), (gx, gy + arm)], fill=C_GHOST, width=2)

        # 接触点只画成目标球上的小点，供检查目标方向；实际瞄准仍看
        # 鬼球中心十字，不能把这个点当作母球中心。
        contact = scene.get("contact")
        if contact:
            cx, cy = float(contact["x"]), float(contact["y"])
            cr = max(2.0, min(4.0, 0.22 * float(contact.get("r", 12))))
            d.ellipse([cx - cr, cy - cr, cx + cr, cy + cr],
                      fill=C_TARGET, outline=C_TARGET)

        # 状态文字（左上深色底）。底色带 alpha：面板内部点击仍穿透，
        # 只有文字笔画像素挡鼠标（逐像素 alpha 命中测试）。
        status = scene.get("status")
        if status:
            d.rectangle([16, 12, 560, 64], fill=(17, 24, 39, 236))
            d.text((24, 38), status, font=self._font(18), fill=C_TEXT, anchor="lm")
        hint = scene.get("hint")
        if hint:
            d.rectangle([16, 74, 780, 108], fill=(127, 29, 29, 236))
            d.text((24, 91), hint, font=self._font(14), fill=C_HINT, anchor="lm")
        help_ = scene.get("help")
        if help_:
            d.text((16, self.sh - 14), help_, font=self._font(11),
                   fill="#94a3b8", anchor="lm")

        return np.asarray(img, dtype=np.uint8)

    @staticmethod
    def _circle_points(cx: float, cy: float, r: float, n: int) -> List[_coord]:
        import math
        return [(int(cx + r * math.cos(2 * math.pi * i / n)),
                 int(cy + r * math.sin(2 * math.pi * i / n))) for i in range(n)]

    @staticmethod
    def _dashed(d, pts: List[_coord], color: str, width: int,
                dash: Tuple[int, int]) -> None:
        """沿折线按 dash 间隔分段画实线/虚线段。"""
        if not pts:
            return
        segs: List[Tuple[_coord, _coord]] = []
        i = 0
        while i < len(pts) - 1:
            (x1, y1), (x2, y2) = pts[i], pts[i + 1]
            length = max(1e-6, ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
            for s in range(0, int(length), dash[0] + dash[1]):
                if s >= length:
                    break
                t0 = s / length
                t1 = min((s + dash[0]) / length, 1.0)
                segs.append(((x1 + (x2 - x1) * t0, y1 + (y2 - y1) * t0),
                             (x1 + (x2 - x1) * t1, y1 + (y2 - y1) * t1)))
            i += 1
        if segs:
            for a, b in segs:
                d.line([a, b], fill=color, width=width)
