"""桌面悬浮瞄准线 Overlay（tkinter 全屏置顶窗口）。

Windows 默认使用原生分层窗口（native_overlay.NativeLayer）：
  UpdateLayeredWindow + 逐像素 alpha —— alpha=0 的像素系统不送鼠标
  消息（文档行为，跨进程），点击天然穿透，彻底避免「全屏置顶层吞掉
  全系统点击」的假死。QQ_AIM_TK_OVERLAY=1 可强制退回本文件里保留的
  Tk -transparentcolor 窗口路径（诊断用；旧路径的鼠标穿透靠
  WM_NCHITTEST 子类化，跨进程/不同系统版本行为不稳定，仅供参考）。
其他平台退化为半透明黑底窗口。
"""
from __future__ import annotations

import os
import tkinter as tk
from tkinter import font as tkfont
from typing import Callable, Dict, List, Optional

from aimtool.native_overlay import NativeLayer

SENTINEL = "#0d0e0f"          # 透明色（Windows）
TRANSPARENT_ON_NT = os.name == "nt"
# 关键：Tk 的窗口过程对 WM_NCHITTEST 返回 HTCLIENT，WS_EX_TRANSPARENT
# 对 Tk 窗口的鼠标穿透无效（该样式仅影响绘制顺序），必须子类化窗口过程
# 把命中测试改为 HTTRANSPARENT，否则全屏置顶透明层会把全系统鼠标点击
# 全部吞掉（症状：所有窗口假死、左右键失效、键盘正常、任务管理器能打开）。
# 默认开启；个别 Tk 构建若子类化崩溃，用 QQ_AIM_NATIVE_HITTEST=0 关闭，
# 此时鼠标穿透只能回退到 WS_EX_TRANSPARENT（Tk 下不可靠，不推荐）。
ENABLE_NATIVE_HIT_TEST = (
    TRANSPARENT_ON_NT
    and os.environ.get("QQ_AIM_NATIVE_HITTEST", "").strip().lower()
    not in {"0", "off", "false", "no"}
)

# 绘制配色
C_AIM = "#22c55e"             # 母球→鬼球（绿）
C_TARGET = "#f97316"          # 鬼球→目标球（橙）
C_POCKET = "#facc15"          # 目标球→袋口（黄）
C_KICK = "#38bdf8"            # 库边反弹段（天蓝）
C_GHOST = "#ffffff"           # 鬼球白色虚线圆
C_EDGE = "#ffffff"
C_TEXT = "#ffffff"
C_TEXT_BG = "#111827"
C_HINT = "#f87171"


class Overlay:
    """全屏置顶透明层。"""

    def __init__(self, on_key: Callable[[str], None],
                 on_click: Callable[[int, int], None],
                 on_drag: Callable[[int, int], None],
                 on_drag_end: Callable[[int, int], None]):
        self.on_key = on_key
        self.on_click = on_click
        self.on_drag = on_drag
        self.on_drag_end = on_drag_end

        # DPI 感知：必须在创建 Tk root 之前声明。否则高 DPI 缩放屏上
        # winfo_screenwidth 返回逻辑分辨率，而 mss 截图是物理像素，
        # 两套坐标系错位 → 瞄准线整体偏移。
        if TRANSPARENT_ON_NT:
            try:
                import ctypes
                try:
                    ctypes.windll.shcore.SetProcessDpiAwareness(2)   # PER_MONITOR_DPI_AWARE
                except Exception:
                    ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass

        # ---- 显示层选择：默认原生分层窗口（逐像素 alpha 穿透），
        #      失败自动退回 Tk 透明色窗口；QQ_AIM_TK_OVERLAY=1 强制 Tk。
        self._native = None
        use_tk = True
        if TRANSPARENT_ON_NT:
            force_tk = os.environ.get("QQ_AIM_TK_OVERLAY", "").strip().lower()
            if force_tk not in {"1", "true", "yes", "on"}:
                try:
                    import ctypes
                    from ctypes import wintypes
                    u32 = ctypes.windll.user32
                    u32.GetSystemMetrics.argtypes = [ctypes.c_int]
                    u32.GetSystemMetrics.restype = ctypes.c_int
                    sw = u32.GetSystemMetrics(0)     # SM_CXSCREEN
                    sh = u32.GetSystemMetrics(1)     # SM_CYSCREEN
                    self._native = NativeLayer(sw, sh)
                    use_tk = False
                    print("[overlay] 原生分层窗口 OK：逐像素透明，"
                          "点击由系统自动穿透（假死问题应消失）", flush=True)
                except Exception as exc:
                    self._native = None
                    use_tk = True
                    print(f"[overlay] 原生分层窗口初始化失败: "
                          f"{type(exc).__name__}: {exc}", flush=True)
                    print("[overlay] 退回 Tk 透明色窗口路径；"
                          "若仍被吞点击请按 Esc 退出", flush=True)

        # Tk root 仅作事件调度器：原生模式下 withdraw 隐藏，不进命中测试
        self.root = tk.Tk()
        if use_tk:
            self.root.title("QQ2D桌球瞄准器 Overlay")
            self.root.overrideredirect(True)
            self.root.attributes("-topmost", True)
            sw = self.root.winfo_screenwidth()
            sh = self.root.winfo_screenheight()
            self.root.geometry(f"{sw}x{sh}+0+0")
        else:
            self.root.withdraw()

        if TRANSPARENT_ON_NT and use_tk:
            try:
                self.root.attributes("-transparentcolor", SENTINEL)
                bg = SENTINEL
            except tk.TclError as exc:
                # 某些 Tk 构建不带 transparentcolor 支持；保留可见的
                # 半透明回退层，避免初始化异常直接让程序退出且无窗口。
                print(f"[overlay] transparentcolor unavailable: {exc}", flush=True)
                self.root.attributes("-alpha", 0.85)
                bg = "#101820"
        else:
            self.root.attributes("-alpha", 0.6)
            bg = "#000000"

        if use_tk:
            # Overlay 不接管游戏指针。固定 crosshair 会让用户误以为游戏
            # 进入了另一种瞄准状态，而且在覆盖层绘制内容上点击时容易把
            # 点击留在 Overlay；真正的瞄准状态由 QQ 游戏自己显示。
            self.canvas = tk.Canvas(self.root, width=sw, height=sh,
                                    highlightthickness=0, bg=bg, cursor="arrow")
            self.canvas.pack(fill="both", expand=True)
            try:
                self.font_status = tkfont.Font(family="Microsoft YaHei UI",
                                               size=18, weight="bold")
                self.font_hint = tkfont.Font(family="Microsoft YaHei UI",
                                             size=14, weight="bold")
                self.font_label = tkfont.Font(family="Microsoft YaHei UI",
                                              size=11)
            except Exception:
                self.font_status = ("TkDefaultFont", 18, "bold")
                self.font_hint = ("TkDefaultFont", 14, "bold")
                self.font_label = ("TkDefaultFont", 11)

        # 注意：<Key> 只在「全局热键不可用」时才绑定——Windows 上热键线程
        # 是唯一通道；若同时绑定 tk <Key>，窗口激活时按一次键会触发两次。
        self._drag_start: Optional[tuple] = None
        self._drag_rect_id: Optional[int] = None
        self._click_through = False
        self._hwnd = None
        self._old_wndproc = None
        self._wndproc = None
        self._visible = True
        self._show_balls = False        # 默认极简：只画母球/目标球/瞄准线
        self._verify_left = 0           # 映射后复查次数（Windows 上在下方赋值）
        self._key_queue = None          # 延迟创建（见下）
        if TRANSPARENT_ON_NT:
            # 热键：独立线程（PeekMessage 建消息队列后 GetAsyncKeyState 才可靠）
            import queue as _queue
            self._key_queue: "_queue.Queue[str]" = _queue.Queue()
            import threading as _threading
            _threading.Thread(target=self._hotkey_thread, daemon=True).start()
            self.root.after(50, self._drain_keys)
        else:
            # 非 Windows：GetAsyncKeyState 不可用，降级为 tk bind 单通道
            print("[热键] 非 Windows 环境，使用 tk 键盘事件（需窗口焦点）", flush=True)
            self.root.bind("<Key>", self._key)
            self.root.focus_force()
        # 鼠标事件：Windows 上透明区域天然穿透收不到，仅作降级路径；
        # 真正的框选/手动录入用 GetCursorPos 轮询（见 _poll_* 方法）
        if use_tk:
            self.canvas.bind("<Button-1>", self._click)
            self.canvas.bind("<B1-Motion>", self._motion)
            self.canvas.bind("<ButtonRelease-1>", self._release)
        # R 键框选 / M 键手动录入：transparentcolor 透明区域鼠标天然穿透，
        # overlay 收不到鼠标事件，改用 GetCursorPos + 左键状态轮询
        # （鼠标直接在游戏窗口上点/拖即可）。非 Windows 无此 API，不启动。
        self._region_active = False
        self._region_dragging = False
        self._region_start: Optional[tuple] = None
        self._region_current: Optional[tuple] = None
        if TRANSPARENT_ON_NT:
            # 框选是短时交互；50ms 轮询容易错过快速按下/松开，
            # 尤其是在用户按 R 后立即开始拖拽时。
            self.root.after(10, self._poll_frame_select)
            self._manual_active = False
            self._manual_pressed = False
            self.root.after(50, self._poll_manual_clicks)
            if use_tk:
                # 从第一帧起就必须整窗命中穿透，不能依赖任何后续调用：
                # main.py 的 set_click_through(True) 只是幂等重复。勾住
                # WM_NCHITTEST 让包括瞄准线在内的所有像素都不拦截系统鼠标。
                self.set_click_through(True)
                # 首次映射时 Tk 可能重建顶层 HWND，映射后延迟复查并重装穿透
                self._verify_left = 30
                self.root.after(250, self._verify_after_map)
            else:
                # 原生分层窗口：alpha=0 像素系统自动穿透，无需子类化
                self.set_click_through(True)

    def begin_region(self) -> None:
        """进入 R 键框选模式（由 App.on_key 调用）。"""
        self._manual_active = False
        self._region_active = True
        self._region_dragging = False
        self._region_start = None
        self._region_current = None

    def begin_manual(self) -> None:
        """进入 M 键手动录入模式：轮询全局左键点击并上报坐标。"""
        self._region_active = False
        self._manual_pressed = False
        self._manual_active = True

    def stop_manual(self) -> None:
        self._manual_active = False
        self._manual_pressed = False

    def _poll_manual_clicks(self) -> None:
        """手动录入轮询：检测完整的「按下→松开」点击序列，上报松开位置。"""
        if self._manual_active:
            try:
                import ctypes
                from ctypes import wintypes
                u32 = ctypes.windll.user32
                lbtn = bool(u32.GetAsyncKeyState(0x01) & 0x8000)   # VK_LBUTTON
                pt = wintypes.POINT()
                u32.GetCursorPos(ctypes.byref(pt))
                x, y = int(pt.x), int(pt.y)
                if lbtn:
                    self._manual_pressed = True
                elif self._manual_pressed:            # 松开 → 完成一次点击
                    self._manual_pressed = False
                    try:
                        self.on_click(x, y)
                    except Exception as e:
                        print(f"[手动录入] 点击回调出错: {e}", flush=True)
            except Exception as e:
                print(f"[手动录入] 轮询异常: {e}", flush=True)
        self.root.after(50, self._poll_manual_clicks)

    def _poll_frame_select(self) -> None:
        """框选轮询：检测鼠标左键按下/拖动/松开，实时更新选框并上报结束位置。"""
        if self._region_active:
            try:
                import ctypes
                from ctypes import wintypes
                u32 = ctypes.windll.user32
                lbtn = bool(u32.GetAsyncKeyState(0x01) & 0x8000)   # VK_LBUTTON
                pt = wintypes.POINT()
                u32.GetCursorPos(ctypes.byref(pt))
                x, y = int(pt.x), int(pt.y)
                if lbtn and not self._region_dragging:
                    self._region_dragging = True
                    self._region_start = (x, y)
                    self._region_current = (x, y)
                    print(f"[框选] 鼠标按下: ({x},{y})", flush=True)
                elif lbtn and self._region_dragging:
                    self._region_current = (x, y)
                elif not lbtn and self._region_dragging:
                    self._region_dragging = False
                    self._region_active = False
                    print(f"[框选] 鼠标释放: start={self._region_start} "
                          f"end=({x},{y})", flush=True)
                    try:
                        self.on_drag_end(x, y)
                    except Exception as e:
                        print(f"[框选] 完成回调出错: {e}", flush=True)
            except Exception as e:
                print(f"[框选] 轮询异常: {e}", flush=True)
        self.root.after(10, self._poll_frame_select)

    # ---------- 全局热键（Windows） ----------
    _HOTKEYS = {
        "1": "1", "2": "2", "3": "3", "4": "4", "5": "5", "6": "6",
        "0": "0", "g": "g", "m": "m", "r": "r", "k": "k", "p": "p", "o": "o", "b": "b",
        "x": "x", "t": "t", "c": "c", "q": "q", "w": "w",
    }
    _VK_EXTRA = {0x1B: "escape", 0x7B: "f12"}

    def _hotkey_thread(self) -> None:
        import time
        import ctypes
        from ctypes import wintypes
        u32 = ctypes.windll.user32
        # 关键：GetAsyncKeyState 要求调用线程有 Windows 消息队列，
        # 否则一直返回 0（这就是"按了键没反应"的根因）。
        msg = wintypes.MSG()
        u32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 0x0001)  # PM_NOREMOVE 建队列
        prev: Dict[str, bool] = {}
        while True:
            try:
                fired: List[str] = []
                for ch, ks in self._HOTKEYS.items():
                    down = bool(u32.GetAsyncKeyState(ord(ch.upper())) & 0x8000)
                    if down and not prev.get(ch, False):
                        fired.append(ks)
                    prev[ch] = down
                for vk, ks in self._VK_EXTRA.items():
                    down = bool(u32.GetAsyncKeyState(vk) & 0x8000)
                    if down and not prev.get(f"vk{vk}", False):
                        fired.append(ks)
                    prev[f"vk{vk}"] = down
                for ks in fired:
                    print(f"[热键] 检测到 {ks} (thread)", flush=True)
                    self._key_queue.put(ks)
            except Exception as e:
                print(f"[热键] 线程异常: {e}", flush=True)
            time.sleep(0.03)

    def _drain_keys(self) -> None:
        import queue as _queue
        try:
            while True:
                ks = self._key_queue.get_nowait()
                try:
                    self.on_key(ks)
                except Exception as e:
                    print(f"[热键] 处理 {ks} 出错: {e}", flush=True)
        except _queue.Empty:
            pass
        self.root.after(50, self._drain_keys)

    # ---------- 事件 ----------
    def _key(self, ev):
        keysym = (ev.keysym or "").lower()
        if ev.char and ev.char.isprintable():
            keysym = ev.char.lower()
        self.on_key(keysym)

    def _click(self, ev):
        # Windows 透明层上的手动/框选操作由全局鼠标轮询统一处理。
        # 若这里也响应 Tk 事件，同一物理点击可能被上报两次。
        if TRANSPARENT_ON_NT and (self._manual_active or self._region_active):
            return
        self.on_click(ev.x_root, ev.y_root)
        self._drag_start = (ev.x_root, ev.y_root)

    def _motion(self, ev):
        if TRANSPARENT_ON_NT and (self._manual_active or self._region_active):
            return
        self.on_drag(ev.x_root, ev.y_root)
        if self._drag_start:
            if self._drag_rect_id is not None:
                self.canvas.delete(self._drag_rect_id)
            x0, y0 = self._drag_start
            self._drag_rect_id = self.canvas.create_rectangle(
                x0, y0, ev.x_root, ev.y_root, outline="#ffffff", width=2, dash=(6, 4))

    def _release(self, ev):
        if TRANSPARENT_ON_NT and (self._manual_active or self._region_active):
            return
        if self._drag_start:
            self.on_drag_end(ev.x_root, ev.y_root)
            if self._drag_rect_id is not None:
                self.canvas.delete(self._drag_rect_id)
                self._drag_rect_id = None
            self._drag_start = None

    # ---------- 控制 ----------
    def _window_proc(self, hwnd, msg, wparam, lparam):
        """让穿透状态下的所有像素（包括线条和边框）都不命中窗口。"""
        # WM_NCHITTEST = 0x84, HTTRANSPARENT = -1
        if msg == 0x0084 and self._click_through:
            return -1
        import ctypes
        from ctypes import wintypes
        if self._old_wndproc:
            call = ctypes.windll.user32.CallWindowProcW
            call.argtypes = [ctypes.c_void_p, wintypes.HWND, wintypes.UINT,
                             wintypes.WPARAM, wintypes.LPARAM]
            call.restype = ctypes.c_ssize_t
            return call(self._old_wndproc, hwnd, msg, wparam, lparam)
        return ctypes.windll.user32.DefWindowProcW(hwnd, msg, wparam, lparam)

    def _install_hit_test_filter(self) -> None:
        """安装一次窗口子类处理，避免 WS_EX_TRANSPARENT 只影响绘制顺序。

        带验证（v2.5）：安装后读回 GWLP_WNDPROC 与写入地址比对，任何
        失败/不一致都打印到 runtime.log，不再静默吞掉——全屏置顶层若
        穿透失效会把全系统鼠标点击吞掉（所有窗口假死、左右键失效、
        键盘正常、Esc 能退出），必须一眼能看出子类是否装成功。
        """
        if (not TRANSPARENT_ON_NT or not ENABLE_NATIVE_HIT_TEST
                or self._wndproc is not None):
            return
        try:
            import ctypes
            from ctypes import wintypes
            u32 = ctypes.windll.user32
            hwnd = int(self.root.winfo_id())
            self._hwnd = hwnd
            WNDPROC = ctypes.WINFUNCTYPE(
                ctypes.c_ssize_t, wintypes.HWND, wintypes.UINT,
                wintypes.WPARAM, wintypes.LPARAM)
            self._wndproc = WNDPROC(self._window_proc)
            get_proc = getattr(u32, "GetWindowLongPtrW", u32.GetWindowLongW)
            set_proc = getattr(u32, "SetWindowLongPtrW", u32.SetWindowLongW)
            get_proc.argtypes = [wintypes.HWND, ctypes.c_int]
            get_proc.restype = ctypes.c_void_p
            set_proc.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_void_p]
            set_proc.restype = ctypes.c_void_p
            old = get_proc(hwnd, -4)       # GWLP_WNDPROC
            self._old_wndproc = old
            if not old:
                print(f"[overlay] 命中测试子类化失败: hwnd={hwnd} 读不到旧窗口过程",
                      flush=True)
                self._wndproc = None
                return
            new_addr = ctypes.cast(self._wndproc, ctypes.c_void_p)
            set_proc(hwnd, -4, new_addr)
            back = get_proc(hwnd, -4)
            ok = bool(back) and new_addr.value == back
            print(f"[overlay] 命中测试子类化 hwnd={hwnd} "
                  f"old={old:#x} new={new_addr.value:#x} back={back:#x} ok={ok}",
                  flush=True)
            if not ok:
                print("[overlay] 警告: 子类未能生效，全屏层可能会吞点击；"
                      "卡住时按 Esc 退出并告诉我这条警告", flush=True)
                self._wndproc = None
        except Exception as exc:
            print(f"[overlay] 命中测试子类化异常: {type(exc).__name__}: {exc}",
                  flush=True)
            # 样式切换仍可工作；失败时保留原 Tk 窗口过程，避免影响启动。
            self._wndproc = None

    def _verify_after_map(self) -> None:
        """窗口首次映射后复查穿透状态（最多 30 次 ≈ 12 秒）。

        Tk 在首次显示时可能重建顶层 HWND：若重建，旧 hwnd 上的子类和
        扩展样式全部失效，全屏置顶层就会恢复吞点击（系统假死症状）。
        本方法延迟复查：HWND 变了 / 子类丢了就立刻在新 HWND 上重装。
        """
        try:
            self._verify_left -= 1
            if self._verify_left <= 0:
                return
            if TRANSPARENT_ON_NT:
                import ctypes
                from ctypes import wintypes
                u32 = ctypes.windll.user32
                hwnd_now = int(self.root.winfo_id())
                if hwnd_now != self._hwnd:
                    print(f"[overlay] HWND 变化 {self._hwnd} -> {hwnd_now}"
                          f"，重装穿透…", flush=True)
                    self._hwnd = None
                    self._old_wndproc = None
                    self._wndproc = None
                    self.set_click_through(self._click_through)
                elif (ENABLE_NATIVE_HIT_TEST and self._wndproc is not None):
                    get_proc = getattr(u32, "GetWindowLongPtrW", u32.GetWindowLongW)
                    get_proc.argtypes = [wintypes.HWND, ctypes.c_int]
                    get_proc.restype = ctypes.c_void_p
                    cur = get_proc(hwnd_now, -4)
                    want = ctypes.cast(self._wndproc, ctypes.c_void_p).value
                    if cur != want:
                        print(f"[overlay] 子类丢失 cur={cur:#x} want={want:#x}，重装…",
                              flush=True)
                        self._wndproc = None
                        self.set_click_through(self._click_through)
        except Exception as exc:
            print(f"[overlay] 穿透复查异常: {type(exc).__name__}: {exc}", flush=True)
        self.root.after(400, self._verify_after_map)

    def set_click_through(self, on: bool) -> None:
        self._click_through = on
        if getattr(self, "_native", None) is not None:
            # 原生分层窗口：alpha=0 像素由系统自动穿透，无需扩展样式/子类化
            print(f"[overlay] click_through={on} (原生层：整窗恒定命中穿透)",
                  flush=True)
            return
        try:
            self.canvas.configure(cursor="arrow")
        except Exception:
            pass
        if not TRANSPARENT_ON_NT:
            return
        try:
            import ctypes
            from ctypes import wintypes
            u32 = ctypes.windll.user32
            GWL_EXSTYLE = -20
            WS_EX_LAYERED = 0x00080000
            WS_EX_TRANSPARENT = 0x00000020
            WS_EX_NOACTIVATE = 0x08000000
            if ENABLE_NATIVE_HIT_TEST:
                self._install_hit_test_filter()
            hwnd = self._hwnd or int(self.root.winfo_id())
            # 64 位下 HWND/句柄必须按指针宽传入，否则高 32 位被截断，
            # SetWindowLongW 会静默失败（样式没设上=整窗吞点击，且无报错）。
            u32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
            u32.GetWindowLongW.restype = ctypes.c_long
            u32.SetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_long]
            u32.SetWindowLongW.restype = ctypes.c_long
            u32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int,
                                         ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                         wintypes.UINT]
            u32.SetWindowPos.restype = wintypes.BOOL
            style = u32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            if on:
                style |= WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE
            else:
                style &= ~WS_EX_TRANSPARENT
            u32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)
            # 让新扩展样式立即参与命中测试，不改变窗口位置和大小。
            u32.SetWindowPos(hwnd, 0, 0, 0, 0, 0,
                             0x0001 | 0x0002 | 0x0004 | 0x0010 | 0x0020)
            # 读回验证：样式没设上 = 整窗吞点击且无任何报错（64 位句柄
            # 截断会静默失败），把实际状态写进 runtime.log 供排查。
            style_back = u32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            ok = (bool(style_back & WS_EX_TRANSPARENT) if on
                  else not (bool(style_back & WS_EX_TRANSPARENT)))
            print(f"[overlay] click_through={on} hwnd={hwnd} "
                  f"exstyle={style_back:#x} ok={ok}", flush=True)
            if not ok:
                print("[overlay] 警告: 扩展样式未生效；若点击被吞请按 Esc 退出"
                      "并把这行 warning 告诉我", flush=True)
        except Exception as exc:
            print(f"[overlay] set_click_through 异常: {type(exc).__name__}: {exc}",
                  flush=True)

    def toggle_visible(self) -> bool:
        if self._native is not None:
            if self._visible:
                self._native.hide()
                self._visible = False
            else:
                self._native.show()
                self._visible = True
            return self._visible
        if self._visible:
            self.root.withdraw()
            self._visible = False
        else:
            self.root.deiconify()
            self.root.attributes("-topmost", True)
            self.root.focus_force()
            self._visible = True
        return self._visible

    def toggle_balls(self) -> bool:
        """切换球标注模式：极简（仅母球/目标球）↔ 全部球细空心标注。"""
        self._show_balls = not self._show_balls
        return self._show_balls

    # ---------- 渲染 ----------
    def render(self, scene: Dict) -> None:
        if self._native is not None:
            # 原生分层窗口：PIL 直接画在位图上，alpha=0 像素点击恒穿透
            region = None
            if self._region_active and self._region_start and self._region_current:
                x0, y0 = self._region_start
                x1, y1 = self._region_current
                region = (x0, y0, x1, y1)
            try:
                self._native.draw(scene, region)
            except Exception as exc:
                print(f"[overlay] 原生渲染异常: {type(exc).__name__}: {exc}",
                      flush=True)
            return
        c = self.canvas
        c.delete("all")

        # R 键框选矩形（实时更新，鼠标在游戏窗口上拖动即可）
        if self._region_active and self._region_start and self._region_current:
            x0, y0 = self._region_start
            x1, y1 = self._region_current
            c.create_rectangle(x0, y0, x1, y1, outline="#ffffff", width=2, dash=(6, 4))
            c.create_text((x0 + x1) // 2, min(y0, y1) - 14, text="松开左键完成框选",
                          fill="#ffffff", font=self.font_label)

        # 台面边框
        quad = scene.get("table_quad")
        if quad:
            c.create_polygon(*[v for p in quad for v in p], outline=C_EDGE,
                             fill="", width=2)

        # 袋口：只画灰色小点，选中袋口的橙色圈不再画（影响视觉；
        # 选中袋口由状态文字「袋口N」标明）。
        for p in scene.get("pockets", []):
            if p.get("sel"):
                continue
            c.create_oval(p["x"] - 4, p["y"] - 4, p["x"] + 4, p["y"] + 4,
                          outline="#94a3b8", width=2)

        # 瞄准线段
        for seg in scene.get("segments", []):
            pts = seg["pts"]
            color = seg.get("color", C_AIM)
            width = seg.get("width", 5)
            dash = (12, 8) if seg.get("dash") else ()
            if len(pts) == 2:
                (x1, y1), (x2, y2) = pts
                c.create_line(x1, y1, x2, y2, fill=color, width=width, dash=dash,
                              capstyle="round")
            else:
                for i in range(len(pts) - 1):
                    (x1, y1), (x2, y2) = pts[i], pts[i + 1]
                    c.create_line(x1, y1, x2, y2, fill=color, width=width,
                                  dash=dash, capstyle="round")

        # 球：极简模式只高亮 母球 + 当前目标球；其他球默认不画（避免遮挡游戏），
        # 按 B 键可切换为「全部球细空心标注」。
        cue = scene.get("cue")
        target = scene.get("target")
        for b in scene.get("balls", []):
            r = max(4.0, b.get("r", 12))
            is_cue = cue is not None and abs(b["x"] - cue["x"]) < r and abs(b["y"] - cue["y"]) < r
            is_tgt = target is not None and abs(b["x"] - target["x"]) < r and abs(b["y"] - target["y"]) < r
            if is_cue:
                # 母球：白色实心 + 外圈
                c.create_oval(b["x"] - r * 0.9, b["y"] - r * 0.9,
                              b["x"] + r * 0.9, b["y"] + r * 0.9,
                              fill="#ffffff", outline=C_EDGE, width=2)
                c.create_oval(b["x"] - r * 1.5, b["y"] - r * 1.5,
                              b["x"] + r * 1.5, b["y"] + r * 1.5,
                              outline=C_AIM, width=2)
            elif is_tgt:
                # 目标球亮橙外圈隐藏；contact 小点仍在下方单独绘制。
                pass
            elif self._show_balls:
                # 全标注模式：细空心小圆
                c.create_oval(b["x"] - r * 0.6, b["y"] - r * 0.6,
                              b["x"] + r * 0.6, b["y"] + r * 0.6,
                              outline=b.get("color", "#94a3b8"), width=1)

        # 鬼球（绿色虚线圆 + 中心十字）—— 唯一瞄准目标
        g = scene.get("ghost")
        if g:
            r = g.get("r", scene.get("ball_r", 12))
            c.create_oval(g["x"] - r, g["y"] - r, g["x"] + r, g["y"] + r,
                          outline=C_GHOST, width=2, dash=(10, 7))
            # 鬼球中心是母球中心的瞄准点，用十字标出，避免只看虚线圆
            # 时无法判断应对准哪个像素。
            arm = max(5.0, min(10.0, 0.35 * r))
            c.create_line(g["x"] - arm, g["y"], g["x"] + arm, g["y"],
                          fill=C_GHOST, width=2)
            c.create_line(g["x"], g["y"] - arm, g["x"], g["y"] + arm,
                          fill=C_GHOST, width=2)

        # 接触点只画成目标球上的小点，便于检查几何方向；真正的瞄准
        # 目标仍是上面的鬼球中心十字，不是这个点。
        contact = scene.get("contact")
        if contact:
            cr = max(2.0, min(4.0, 0.22 * contact.get("r", 12)))
            c.create_oval(contact["x"] - cr, contact["y"] - cr,
                          contact["x"] + cr, contact["y"] + cr,
                          fill=C_TARGET, outline=C_TARGET)

        # 状态文字
        status = scene.get("status")
        if status:
            c.create_rectangle(16, 12, 560, 64, fill=C_TEXT_BG, outline="")
            c.create_text(24, 38, text=status, anchor="w", fill=C_TEXT,
                          font=self.font_status)
        hint = scene.get("hint")
        if hint:
            c.create_rectangle(16, 74, 780, 108, fill="#7f1d1d", outline="")
            c.create_text(24, 91, text=hint, anchor="w", fill=C_HINT,
                          font=self.font_hint)

        # 底部热键提示
        help = scene.get("help")
        if help:
            c.create_text(16, self.root.winfo_screenheight() - 14, text=help,
                          anchor="w", fill="#94a3b8", font=self.font_label)

    def run(self) -> None:
        self.root.mainloop()

    def destroy(self) -> None:
        if self._native is not None:
            try:
                self._native.close()
            except Exception as exc:
                print(f"[overlay] 原生层关闭异常: {type(exc).__name__}: {exc}",
                      flush=True)
            self.root.destroy()
            return
        if TRANSPARENT_ON_NT and self._old_wndproc and self._hwnd:
            try:
                import ctypes
                from ctypes import wintypes
                u32 = ctypes.windll.user32
                set_proc = getattr(u32, "SetWindowLongPtrW", u32.SetWindowLongW)
                set_proc(self._hwnd, -4, self._old_wndproc)
            except Exception:
                pass
        self.root.destroy()
