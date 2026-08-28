"""QQ 2D桌球 辅助瞄准器 主程序。

模式：
  python main.py                  # 正常启动：截屏识别 + 悬浮瞄准线
  python main.py --demo           # 无界面自检（合成台面跑通全流程）
  python main.py --frame x.png    # 分析一张游戏截图并打印瞄准方案
  python main.py --region X Y W H # 预设捕获区域（可被 R 键重新框选）
  python main.py --manual         # 以手动录入模式启动

热键（Overlay 窗口激活时）：
  1-6 选择袋口 · 0 自动选袋口 · G 点选目标球 · M 手动录入(母球→目标球→袋口) · R 框选球桌区域
  K 库边解围开关 · P 自动袋口开关 · Q 红/彩切换 · X 鼠标穿透开关 · T 隐藏/显示 · C 重新识别 · Esc 退出
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


_RUNTIME_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runtime.log")


def _ensure_stdio() -> None:
    """保证所有输出都能被记录。

    - pythonw（无控制台）：stdout/stderr 为 None，接到 runtime.log（UTF-8）。
    - 控制台 + QQ_AIM_HEADLESS=1（start.bat 隐藏窗口模式）：控制台窗口会被
      立即隐藏，stdout 是隐藏控制台的流，同样强制改接到 runtime.log，否则
      用户和我们都看不到任何日志。
    文件为空时先写 BOM，让记事本等编辑器能自动识别编码。
    """
    force_log = os.environ.get("QQ_AIM_HEADLESS", "") == "1"
    if not force_log and sys.stdout is not None and sys.stderr is not None:
        return
    try:
        stream = open(_RUNTIME_LOG, "a", encoding="utf-8", buffering=1)
        if stream.tell() == 0:
            stream.write("\ufeff")   # UTF-8 BOM
        if sys.stdout is None or force_log:
            sys.stdout = stream
        if sys.stderr is None or force_log:
            sys.stderr = stream
    except OSError:
        pass


def _hide_console() -> None:
    """QQ_AIM_HEADLESS=1 时把本进程的控制台窗口隐藏（SW_HIDE）。

    启动器用控制台 Python 启动（pythonw 静默启动失败无法诊断），
    窗口在程序第一条语句处就藏掉，用户只看到 cmd 一闪而过。
    """
    if os.name != "nt" or os.environ.get("QQ_AIM_HEADLESS", "") != "1":
        return
    try:
        import ctypes
        hwnd = ctypes.windll.kernel32.GetConsoleWindow()
        if hwnd:
            ctypes.windll.user32.ShowWindow(hwnd, 0)   # SW_HIDE
    except Exception:
        pass


def _startup_marker() -> None:
    """main.py 被加载的独立证据，写入 startup.log（与 runtime.log 分开，
    避免 runtime.log 被占用/权限问题掩盖"程序是否启动"的事实）。"""
    try:
        marker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "startup.log")
        with open(marker, "a", encoding="utf-8") as f:
            f.write(f"[main] loaded pid={os.getpid()} "
                    f"headless={os.environ.get('QQ_AIM_HEADLESS', '')} "
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    except Exception:
        pass


def _valid_saved_region(region: Optional[List[int]]) -> Optional[List[int]]:
    """Drop a persisted capture rectangle that cannot fit the current screen."""
    if not region:
        return None
    try:
        x, y, w, h = (int(v) for v in region)
        if x < 0 or y < 0 or w <= 40 or h <= 40:
            return None
        if os.name == "nt":
            import ctypes
            sw = int(ctypes.windll.user32.GetSystemMetrics(0))
            sh = int(ctypes.windll.user32.GetSystemMetrics(1))
            if sw <= 0 or sh <= 0 or x + w > sw or y + h > sh:
                return None
        return [x, y, w, h]
    except (TypeError, ValueError, OSError):
        return None


_ensure_stdio()
_hide_console()
if __name__ == "__main__":
    _startup_marker()


try:
    import cv2
    import numpy as np
    from aimtool import capture, config as config_mod, physics, snooker, tracking, vision
except ModuleNotFoundError as exc:
    missing = exc.name or "未知模块"
    raise SystemExit(
        f"[启动失败] 缺少 Python 依赖: {missing}\n"
        "请在项目目录运行: python -m pip install -r requirements.txt"
    )

APP_VERSION = "3.6.0"

HELP_TEXT = ("1-6 选袋口 | 0 自动 | G 点选目标球 | M 手动录入 | R 框选区域 | K 库边解围 | "
             "Q 红/彩切换 | O 兼容切换 | P 自动袋口 | B 球标注 | X 穿透 | T 隐藏 | C 重识别 | Esc 退出")

COLOR_HEX = {
    "白球": "#ebebef",
    "黄球": "#fdb813",
    "绿球": "#2f9e44",
    "棕球": "#8a5a2b",
    "蓝球": "#1971c2",
    "粉球": "#f06595",
    "黑球": "#2b2b2b",
    "红球": "#e03131",
}


_warn_state = {"key": None, "t": 0.0}
_white_state = {"n": 0}
_save_state = {"key": None, "t": 0.0}
# blank_self_mask 的自绘像素还原信息。_save_bad_frame 写盘前据此把
# 已被填回台呢色的像素恢复成原始画面（见 vision.blank_self_mask）。
_self_paint: Dict = {"restore": None}
_BAD_FRAME_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_frames")
_SAVE_INTERVAL = 30.0   # 同一异常 30 秒内只存一张现场帧，避免刷盘
_BAD_FRAME_PREFIX = "bad2_"  # 新格式；保留旧 bad_HHMMSS.png 诊断文件
_BAD_FRAME_KEEP = 100
_INSTANCE_MUTEX = None


def _kill_stale_instances() -> bool:
    """强制结束运行本脚本的其它 python 进程（防重复启动被打死的情况）。

    只在「互斥体被占、且找不到任何旧实例窗口」时调用——正常实例一定有
    可见窗口，找不到窗口说明是老版本的僵尸进程（原生窗口无标题时代遗留，
    或旧版崩溃后残留）。按命令行匹配 snoke/main.py 并排除自身。
    """
    try:
        import subprocess
        ps = (
            "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
            f"Where-Object {{ $_.CommandLine -match 'snoke.*main\\.py' "
            f"-and $_.ProcessId -ne {os.getpid()} }} | "
            "ForEach-Object { taskkill /F /PID $_.ProcessId | Out-Null }"
        )
        subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
            timeout=30, creationflags=0x08000000)   # CREATE_NO_WINDOW：不闪控制台
        # 等 1 秒让进程真正退干净
        time.sleep(1.0)
        return True
    except Exception:
        return False


def _acquire_instance() -> bool:
    """防止重复点击启动器创建多个全屏置顶 Overlay。

    已有实例时先尝试关闭它的 Overlay 窗口：上次异常退出可能留下一个
    看不见的全屏置顶透明层（鼠标穿透失效时它会吞掉系统全部点击），
    必须先把它关掉再继续启动，否则新实例退出后僵尸层还在。
    """
    global _INSTANCE_MUTEX
    if os.name != "nt":
        return True
    try:
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.windll.kernel32
        user32 = ctypes.windll.user32
        kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p]
        kernel32.CreateMutexW.restype = ctypes.c_void_p
        kernel32.GetLastError.restype = ctypes.c_ulong

        def create_mutex() -> tuple:
            """返回 (句柄, last_error)；句柄为空表示无法创建。"""
            h = kernel32.CreateMutexW(None, True, "Local\\QQ2D_Billiard_Aim_Tool")
            err = kernel32.GetLastError() if h else -1
            return h, err

        handle, err = create_mutex()
        if not handle:
            # 互斥体创建失败只是失去防重复保护，不影响正常启动。
            return True
        if err != 183:                          # 非 ERROR_ALREADY_EXISTS
            _INSTANCE_MUTEX = handle            # 持有到进程退出
            return True
        kernel32.CloseHandle(handle)

        # 已有实例：多半是上次异常退出留下的僵尸 Overlay（或用户重复双击）。
        # 找到它的 Overlay 窗口发 WM_CLOSE，等它退出后重试互斥体。
        user32.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
        user32.FindWindowW.restype = wintypes.HWND
        user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT,
                                        wintypes.WPARAM, wintypes.LPARAM]
        user32.PostMessageW.restype = wintypes.BOOL
        hwnd = user32.FindWindowW(None, "QQ2D桌球瞄准器 Overlay")
        if hwnd:
            print("[启动] 检测到已有实例的 Overlay 窗口，正在关闭旧实例…", flush=True)
            user32.PostMessageW(hwnd, 0x0010, 0, 0)   # WM_CLOSE
        else:
            print("[启动] 互斥体被占用但窗口不可见（残留进程？），等待其退出…", flush=True)
        window_found = bool(hwnd)
        # 无论找没找到窗口都等它释放互斥体，最多 ~5s（旧实例可能正在退出）。
        for _ in range(25):
            time.sleep(0.2)
            handle, err = create_mutex()
            if handle and err != 183:
                _INSTANCE_MUTEX = handle
                return True
            if handle:
                kernel32.CloseHandle(handle)
        if not window_found and _kill_stale_instances():
            # 从没找到过任何窗口 ⇒ 持锁者是不可见的僵尸进程（原生窗口
            # 无标题时代的残留/崩溃残留），强制结束它再续试互斥体。
            print("[启动] 已强制结束卡死的旧实例，续试互斥体…", flush=True)
            for _ in range(15):
                time.sleep(0.2)
                handle, err = create_mutex()
                if handle and err != 183:
                    _INSTANCE_MUTEX = handle
                    return True
                if handle:
                    kernel32.CloseHandle(handle)
        print("[启动] 已有实例未能退出，本次启动中止。", flush=True)
        return False
    except Exception:
        # 互斥体只是防重复保护，失败时不影响正常启动。
        return True


def _save_bad_frame(frame: np.ndarray) -> Optional[str]:
    """保存异常帧并限制本版本生成的文件数量。

    文件名包含纳秒时间和进程号，避免同一秒内不同异常互相覆盖。
    只清理带新前缀的文件，用户已有的旧诊断帧保持不动。
    """
    try:
        import cv2
        os.makedirs(_BAD_FRAME_DIR, exist_ok=True)
        stamp = time.time_ns()
        filename = (f"{_BAD_FRAME_PREFIX}{time.strftime('%Y%m%d_%H%M%S')}_"
                    f"{os.getpid()}_{stamp}.png")
        path = os.path.join(_BAD_FRAME_DIR, filename)
        restore = _self_paint.get("restore")
        if restore is not None:
            ys, xs, pixels = restore
            if (len(ys.shape) == 1 and len(xs.shape) == 1
                    and ys.size == xs.size == pixels.shape[0]
                    and pixels.shape[1] == frame.shape[2]
                    and (ys.size == 0 or
                         (ys.max() < frame.shape[0] and xs.max() < frame.shape[1]))):
                # 在副本上还原后存盘：不污染分析中的帧（叠加线恢复可见，
                # 诊断遮挡误报时恰需要看到自家画线与真实画面的叠加关系）。
                frame = frame.copy()
                frame[ys, xs] = pixels
        if not cv2.imwrite(path, frame):
            return None

        files = []
        for entry in os.scandir(_BAD_FRAME_DIR):
            if entry.is_file() and entry.name.startswith(_BAD_FRAME_PREFIX) \
                    and entry.name.lower().endswith(".png"):
                try:
                    files.append((entry.stat().st_mtime_ns, entry.path))
                except OSError:
                    pass
        files.sort(reverse=True)
        for _, stale in files[_BAD_FRAME_KEEP:]:
            try:
                os.remove(stale)
            except OSError:
                pass
        return path
    except Exception:
        return None


def _warn_once(msg: str, frame: Optional[np.ndarray] = None,
               key: Optional[str] = None) -> None:
    """节流警告：同一类别 5 秒内不重复打印，避免实机刷屏。

    key 是节流类别（如 "occlusion"）；消息正文里的每帧变化数值
    （面积/球数/明细）不再参与节流判断。旧实现以整条消息做 key，
    遮挡消息含每帧变化的 bbox/fill → key 恒新 → 节流失效，
    遮挡期间每帧打印+每帧存 PNG（实测 30 次/秒的 IO 风暴）。
    附带 frame 时存原始帧到 debug_frames/，同一类别 30 秒内只存一次。
    """
    now = time.monotonic()
    key = msg if key is None else key
    if key != _warn_state["key"] or now - _warn_state["t"] > 5.0:
        print(msg, flush=True)
        _warn_state["key"] = key
        _warn_state["t"] = now
        if frame is not None and (key != _save_state["key"]
                                  or now - _save_state["t"] > _SAVE_INTERVAL):
            _save_state["key"] = key
            _save_state["t"] = now
            path = _save_bad_frame(frame)
            if path:
                print(f"[识别异常] 原始帧已存至 {path}", flush=True)


def _ema_point(state: Dict, key: str, cur: physics.Point,
               jump: float, alpha: float = 0.5) -> physics.Point:
    """带跳变保护的逐帧 EMA。球被切换/窗口移动时直接跟随，避免把真位移糊掉。"""
    prev = state.get(key)
    if prev is None or physics.dist(cur, prev) > jump:
        state[key] = cur
        return cur
    blended = (alpha * cur[0] + (1.0 - alpha) * prev[0],
               alpha * cur[1] + (1.0 - alpha) * prev[1])
    state[key] = blended
    return blended


def _ema_scalar(state: Dict, key: str, cur: float, alpha: float = 0.5,
                jump_frac: float = 0.12) -> float:
    """标量 EMA（球径等）。漂移超过比例即视为换台面/比例，直接跟随。"""
    prev = state.get(key)
    if prev is None or abs(cur - prev) > jump_frac * max(abs(prev), 1e-6):
        state[key] = cur
        return cur
    blended = alpha * cur + (1.0 - alpha) * prev
    state[key] = blended
    return blended


def _geometry_radius(ball, fallback: float, cfg) -> float:
    """Return a conservative per-ball radius for collision geometry.

    A single-frame ellipse fit is useful for center refinement but its radius
    is label/lighting dependent.  Blend it only when it is a subpixel fit and
    keep the result close to the frame-wide median, so one clipped highlight
    cannot move the ghost ball by several pixels.
    """
    fallback = max(float(fallback), 1e-6)
    if not bool(getattr(ball, "subpixel", False)):
        return fallback
    try:
        observed = float(ball.radius) * float(getattr(cfg, "ball_radius_scale", 1.0))
    except (TypeError, ValueError):
        return fallback
    if not np.isfinite(observed):
        return fallback
    lo = 0.88 * fallback
    hi = 1.12 * fallback
    observed = max(lo, min(hi, observed))
    weight = float(getattr(cfg, "ball_radius_instance_weight", 0.35))
    weight = max(0.0, min(1.0, weight))
    return (1.0 - weight) * fallback + weight * observed


@dataclass
class _AnalysisContext:
    """analyze() 的帧级上下文：输入参数 + 各阶段共享产物。

    原 530 行 analyze() 拆成 7 个 _stage_* 纯函数，ctx 负责在阶段间
    传递中间产物；任一阶段返回 scene 即提前结束（顺序见 analyze()）。
    """
    # ---- 输入参数（原 analyze 签名，顺序一致）----
    frame: np.ndarray
    cfg: config_mod.Config
    manual_cue: Optional[physics.Point] = None
    manual_target: Optional[physics.Point] = None
    manual_pocket_idx: Optional[int] = None
    picked_target: Optional[physics.Point] = None
    tracker: Optional[vision.TableTracker] = None
    prefer_target: Optional[physics.Point] = None
    pocket_tracker: Optional[vision.PocketTracker] = None
    smooth: Optional[Dict] = None
    self_mask: Optional[np.ndarray] = None
    ball_tracker: Optional[tracking.BallTracker] = None
    table_state: Optional[tracking.TableStateTracker] = None
    turn_tracker: Optional[snooker.TurnTracker] = None
    captured_at: Optional[float] = None
    # ---- 标准台面尺寸 ----
    W: int = 0
    H: int = 0
    # ---- _stage_find_table 产物 ----
    quad: Optional[np.ndarray] = None
    Hm: Optional[np.ndarray] = None
    Hinv: Optional[np.ndarray] = None
    scale: float = 1.0
    analysis_w: int = 0
    analysis_h: int = 0
    r_analysis: float = 0.0
    warped: Optional[np.ndarray] = None
    warped_hsv: Optional[np.ndarray] = None
    warped_gray: Optional[np.ndarray] = None
    felt_hsv: Optional[tuple] = None
    protect_masks: Optional[Dict] = None
    foreign_mask: Optional[np.ndarray] = None
    # ---- _stage_pockets 产物 ----
    pockets_analysis: List[physics.Point] = field(default_factory=list)
    pockets_t: List[physics.Point] = field(default_factory=list)
    # ---- _stage_balls 产物 ----
    balls_t: List = field(default_factory=list)
    r: float = 0.0
    # ---- _stage_scene 产物 ----
    scene: Dict = field(default_factory=dict)
    manual_override: bool = False
    table_phase: Any = None
    # ---- _stage_targets 产物 ----
    cue_t: Optional[physics.Point] = None
    cue_b: Any = None
    target_t: Optional[physics.Point] = None
    target_b: Any = None
    others: List[physics.Point] = field(default_factory=list)
    cue_radius: float = 0.0
    target_radius: float = 0.0
    ghost_offset: tuple = (0.0, 0.0)
    break_mode: bool = False
    rule_text: str = ""


def _stage_find_table(ctx: _AnalysisContext) -> Optional[Dict]:
    """阶段①：自绘清理 + 台面四边形锁定 + 透视变换与帧级共享视觉产物。"""
    frame = ctx.frame
    cfg = ctx.cfg
    W, H = ctx.W, ctx.H
    tracker = ctx.tracker
    self_mask = ctx.self_mask
    captured_at = ctx.captured_at
    W, H = cfg.table_w, cfg.table_h
    captured_at = time.monotonic() if captured_at is None else captured_at
    # 自截屏清理：叠加层画过的像素先填回台呢色。全屏顶层透明窗的
    # 画线会被截屏一起抓进帧里，遮挡检测把自家的线误判为弹窗
    # （实测：命中块 99% 像素是画线颜色）。
    _self_paint["restore"] = vision.blank_self_mask(frame, self_mask, cfg)
    if tracker is not None:
        quad = tracker.update(frame)
    else:
        quad = vision.find_table(frame, cfg)
    if quad is None:
        return {
            "status": "未检测到台面",
            "hint": "按 R 框选球桌区域，或按 M 手动录入球位",
            "help": HELP_TEXT,
        }

    # 物理与绘制仍用 2000x1000 标准坐标，但视觉在接近原始桌面像素
    # 尺寸下运行。把约 1000px 的真实台面硬插值到 2000px 不会增加信息，
    # 却会把 Hough/形态学工作量放大约四倍。
    Hm = vision.homography(quad, W, H)
    Hinv = np.linalg.inv(Hm)
    source_w = float(np.linalg.norm(quad[1] - quad[0]))
    analysis_w = int(round(source_w * float(getattr(cfg, "analysis_scale", 0.85))))
    analysis_w = max(int(getattr(cfg, "analysis_min_width", 720)), analysis_w)
    analysis_w = min(int(getattr(cfg, "analysis_max_width", 960)), analysis_w)
    analysis_w = max(320, analysis_w)
    analysis_h = max(160, int(round(analysis_w * H / W)))
    Hm_analysis = vision.homography(quad, analysis_w, analysis_h)
    warped = vision.warp_table(frame, Hm_analysis, analysis_w, analysis_h)
    r_analysis = cfg.ball_radius_ratio * analysis_w
    # These full-frame products are shared by occlusion, transient-UI and
    # ball detection.  Reusing them matters on small Windows laptops where a
    # fresh HSV conversion plus felt scan can consume most of one frame.
    warped_hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    felt_hsv = vision.estimate_felt_hsv(warped, cfg, hsv=warped_hsv)
    # 帧级颜色掩膜字典：台呢清理（_protect_mask）、外来检测（红球保护）、
    # UI 排除（白球预判）共用同一份，代替原先同一组全帧布尔运算×3 重复。
    _h_ch, _s_ch, _v_ch = cv2.split(warped_hsv)
    protect_masks = vision.compute_label_masks(_h_ch, _s_ch, _v_ch)
    foreign_mask = vision.compute_foreign_mask(
        warped, cfg, r_analysis, hsv=warped_hsv, felt_hsv=felt_hsv,
        label_masks=protect_masks)



    ctx.quad = quad
    ctx.Hm = Hm
    ctx.Hinv = Hinv
    ctx.scale = np.hypot(*(quad[1] - quad[0])) / W
    ctx.analysis_w = analysis_w
    ctx.analysis_h = analysis_h
    ctx.r_analysis = r_analysis
    ctx.warped = warped
    ctx.warped_hsv = warped_hsv
    ctx.warped_gray = warped_gray
    ctx.felt_hsv = felt_hsv
    ctx.protect_masks = protect_masks
    ctx.foreign_mask = foreign_mask
    return None


def _stage_pockets(ctx: _AnalysisContext) -> Optional[Dict]:
    """阶段②：袋口定位（默认 → 暗域精修 → 跟踪器锁定），无早退。"""
    cfg = ctx.cfg
    W, H = ctx.W, ctx.H
    pocket_tracker = ctx.pocket_tracker
    analysis_w, analysis_h = ctx.analysis_w, ctx.analysis_h
    warped = ctx.warped
    r_analysis = ctx.r_analysis
    # 袋口必须先于球检测：clean_background 按袋口位置挖洞，
    # 若用默认角点 (0,0) 等，真实内缩袋口区域不被涂灰，会被当成
    # 白球/黑球候选（实测每帧多出 4~10 个假球）。
    pockets_analysis: List[physics.Point] = physics.default_pockets(analysis_w, analysis_h)
    locked_pockets = pocket_tracker.current() if pocket_tracker is not None else None
    if cfg.pocket_refine and locked_pockets is None:
        pockets_analysis = vision.refine_pockets(
            warped, pockets_analysis, r_analysis,
            dark_delta=float(getattr(cfg, "pocket_dark_delta", 18.0)),
            min_dark_area_ratio=float(
                getattr(cfg, "pocket_min_dark_area_ratio", 0.02)),
            pin_area_ratio=float(getattr(cfg, "pocket_pin_area_ratio", 0.70)),
            search_ratio=float(getattr(cfg, "pocket_search_ratio", 4.0)),
        )
    pockets_t: List[physics.Point] = [
        (float(x * W / analysis_w), float(y * H / analysis_h))
        for x, y in pockets_analysis
    ]
    if locked_pockets is not None:
        # After the table homography is locked, pocket coordinates are already
        # in the same normalized table space. Re-running dark-component
        # refinement every frame only adds jitter and several milliseconds.
        pockets_t = locked_pockets
    elif pocket_tracker is not None:
        pockets_t = pocket_tracker.update(pockets_t)
    # 用稳定但尚未标定偏移的袋心参与视觉清理；标定偏移只应用到物理规划。
    pockets_analysis = [
        (float(x * analysis_w / W), float(y * analysis_h / H))
        for x, y in pockets_t
    ]



    ctx.pockets_analysis = pockets_analysis
    ctx.pockets_t = pockets_t
    return None


def _stage_balls(ctx: _AnalysisContext) -> Optional[Dict]:
    """阶段③：遮挡检测（早退）→ UI 排除 → 台呢清理 → 球检测 → 袋口标定 → 半径估计。"""
    frame = ctx.frame
    cfg = ctx.cfg
    W, H = ctx.W, ctx.H
    quad, Hm, Hinv = ctx.quad, ctx.Hm, ctx.Hinv
    r_analysis = ctx.r_analysis
    warped = ctx.warped
    warped_hsv, warped_gray = ctx.warped_hsv, ctx.warped_gray
    felt_hsv = ctx.felt_hsv
    protect_masks = ctx.protect_masks
    foreign_mask = ctx.foreign_mask
    pockets_analysis, pockets_t = ctx.pockets_analysis, ctx.pockets_t
    analysis_w, analysis_h = ctx.analysis_w, ctx.analysis_h
    table_state = ctx.table_state
    captured_at = ctx.captured_at
    smooth = ctx.smooth
    # QQ 菜单/设置窗口/提示弹窗会把大块白灰面板送进白球掩膜，
    # 造成大量假球。污染帧必须暂停瞄准，不能沿用上一帧方案继续显示。
    # 不再把任何一帧的外来像素立即“学习”为静态背景：旧实现会把真菜单
    # 在下一帧放行。大面板暂停，小型连击文字由 ui_mask 从球检测中排除。
    occlusion = vision.detect_table_occlusion(
        warped, cfg, r_analysis, hsv=warped_hsv, foreign=foreign_mask,
        felt_hsv=felt_hsv)
    if occlusion is not None:
        # 诊断日志：真实遮挡 vs 球杆/反光误报，把触发块的特征记下来
        _warn_once(f"[遮挡] 检测到遮挡: {occlusion}", frame, key="occlusion")
        scale = ctx.scale
        ball_r_screen = max(4.0, cfg.ball_radius_ratio * W * scale)
        scene = {
            "table_quad": quad.tolist(),
            "pockets": [
                {"x": x, "y": y, "sel": False}
                for (x, y) in [vision.point_table_to_screen(p, Hinv) for p in pockets_t]
            ],
            "balls": [],
            "ball_r": ball_r_screen,
            "help": HELP_TEXT,
            "rule": "",
            "H": Hm,
            "Hinv": Hinv,
            "invalid": True,
            "occluded": True,
            "status": "台面被菜单/弹窗遮挡，已暂停瞄准",
            "hint": "关闭游戏设置、提示框或右键菜单后按 C 重新识别",
        }
        if table_state is not None:
            table_state.update((), captured_at, occluded=True)
        return scene
    ui_mask = vision.transient_ui_mask(
        warped, cfg, r_analysis, hsv=warped_hsv, gray=warped_gray,
        foreign=foreign_mask, felt_hsv=felt_hsv, label_masks=protect_masks)
    clean = vision.clean_background(warped, cfg, r_analysis, pockets_analysis,
                                    exclude_mask=ui_mask, hsv=warped_hsv,
                                    felt_hsv=felt_hsv, label_masks=protect_masks)
    balls_analysis = vision.detect_balls(
        warped, r_analysis, cfg=cfg, pockets=pockets_analysis, clean=clean,
        warped_hsv=warped_hsv, warped_gray=warped_gray,
    )
    balls_t = [vision.Ball(
        b.label,
        (float(b.pos[0] * W / analysis_w), float(b.pos[1] * H / analysis_h)),
        float(b.radius * W / analysis_w), b.subpixel, b.confidence, b.track_id,
    ) for b in balls_analysis]
    offsets = getattr(cfg, "pocket_offsets", [])
    if isinstance(offsets, list) and len(offsets) == len(pockets_t):
        calibrated_pockets = []
        for pocket, offset in zip(pockets_t, offsets):
            if isinstance(offset, (list, tuple)) and len(offset) == 2:
                calibrated_pockets.append((pocket[0] + float(offset[0]),
                                           pocket[1] + float(offset[1])))
            else:
                calibrated_pockets.append(pocket)
        pockets_t = calibrated_pockets
    r = cfg.ball_radius_ratio * W
    # 保险过滤：贴在袋口嘴上的球无法击打，且多为袋口反光误检
    balls_t = [b for b in balls_t
               if all(physics.dist(b.pos, p) > 2.0 * r for p in pockets_t)]
    # 透视校正和窗口缩放会让配置半径与当前帧的实际球径有小偏差。
    # 用多个非红球的亚像素半径估计物理半径，鬼球偏移和碰撞清空
    # 因此不再固定依赖 ball_radius_ratio。
    r = vision.estimate_ball_radius(balls_t, r)
    r *= float(getattr(cfg, "ball_radius_scale", 1.0))
    if smooth is not None:
        r = _ema_scalar(smooth, "r", r)



    ctx.balls_t = balls_t
    ctx.pockets_t = pockets_t
    ctx.r = r
    return None


def _stage_scene(ctx: _AnalysisContext) -> Optional[Dict]:
    """阶段④：屏幕坐标上下文 + 跟踪器更新 + 运动状态机（非 READY 早退）。"""
    cfg = ctx.cfg
    W, H = ctx.W, ctx.H
    quad, Hm, Hinv = ctx.quad, ctx.Hm, ctx.Hinv
    r = ctx.r
    pockets_t = ctx.pockets_t
    balls_t = ctx.balls_t
    ball_tracker = ctx.ball_tracker
    table_state = ctx.table_state
    captured_at = ctx.captured_at
    analysis_w, analysis_h = ctx.analysis_w, ctx.analysis_h
    manual_cue, manual_target = ctx.manual_cue, ctx.manual_target
    manual_pocket_idx = ctx.manual_pocket_idx
    # 先建立坐标上下文。即使球识别失败，M 手动录入也仍需要这套
    # 屏幕→台面矩阵；旧逻辑在球数校验前返回，导致手动兜底无法启动。
    scale = np.hypot(*(quad[1] - quad[0])) / W
    ball_r_screen = max(4.0, r * scale)
    scene: Dict = {
        "table_quad": quad.tolist(),
        "pockets": [
            {"x": x, "y": y, "sel": False}
            for (x, y) in [vision.point_table_to_screen(p, Hinv) for p in pockets_t]
        ],
        "balls": [],
        "ball_r": ball_r_screen,
        "help": HELP_TEXT,
        "rule": "",
        # H: 屏幕局部坐标 → 台面坐标；Hinv: 台面坐标 → 屏幕局部坐标。
        "H": Hm,
        "Hinv": Hinv,
    }

    # 手动指定球位时，允许跳过自动球数/颜色校验；否则 M 模式在
    # 识别失败画面上无法提供真正的手动兜底。
    manual_override = (manual_cue is not None or manual_target is not None
                       or manual_pocket_idx is not None)
    if ball_tracker is not None:
        balls_t = ball_tracker.update(balls_t, captured_at)
    table_phase = tracking.TableState.READY
    if table_state is not None:
        table_phase = table_state.update(balls_t, captured_at)
        if table_phase == tracking.TableState.MOVING and ball_tracker is not None:
            ball_tracker.reset_smoothing()
    scene["table_state"] = str(table_phase.value)
    scene["analysis_size"] = (analysis_w, analysis_h)
    scene["balls"] = [
        {"label": b.label, "x": px, "y": py,
         "r": _geometry_radius(b, r, cfg) * scale,
         "color": COLOR_HEX.get(b.label, "#94a3b8")}
        for b in balls_t
        for (px, py) in [vision.point_table_to_screen(b.pos, Hinv)]
    ]

    if not manual_override and table_phase != tracking.TableState.READY:
        if table_phase == tracking.TableState.MOVING:
            status = "球仍在运动，已隐藏瞄准线"
            hint = "等待球完全静止后自动汇总多帧球位"
        else:
            status = "正在确认稳定球位…"
            hint = "连续帧确认完成后自动显示瞄准线"
        scene.update({"status": status, "hint": hint, "invalid": True})
        return scene



    ctx.scene = scene
    ctx.manual_override = manual_override
    ctx.table_phase = table_phase
    ctx.balls_t = balls_t
    ctx.scale = scale
    return None


def _stage_validate(ctx: _AnalysisContext) -> Optional[Dict]:
    """阶段⑤：对局画面校验（球数/白球数/红球数三道早退闸门）。"""
    cfg = ctx.cfg
    frame = ctx.frame
    balls_t = ctx.balls_t
    manual_override = ctx.manual_override
    scene = ctx.scene
    # 对局画面校验：候选球数异常说明当前画面不是斯诺克台面
    # （大厅界面/桌面浅色区域被 watershed 切碎成大量"球"候选）。
    # 白球数量是最可靠信号（只可能有 1 个）；总球数放宽到 60——
    # 开局红球三角在真实画面上 watershed 碎块候选可能偏多（合成图外）。
    n_white = sum(1 for b in balls_t if b.label == "白球")
    if not manual_override and (n_white > 8 or len(balls_t) > cfg.detect_max_balls):
        from collections import Counter
        detail = dict(Counter(b.label for b in balls_t))
        _warn_once(f"[识别异常] 球数={len(balls_t)} 白球候选={n_white} "
                   f"上限={cfg.detect_max_balls} 明细={detail}", frame,
                   key="balls_too_many")
        scene.update({
            "status": "自动识别异常，请按 R 重新框选",
            "hint": "框选范围尽量紧贴绿色台面（别框进四周界面）。"
                    "若已框选仍异常，把控制台 [识别异常] 日志发给我",
            "invalid": True,
        })
        return scene
    if not manual_override and len(balls_t) < 2:
        # 连母球+1颗目标球都不到：识别基本失效，继续跑只会给出错误瞄准
        scene.update({
            "status": "识别到的球太少，画面可能不是对局台面",
            "hint": "按 R 框选球桌区域，或按 C 重新识别",
            "invalid": True,
        })
        return scene
    n_reds = sum(1 for b in balls_t if b.label == "红球")
    if not manual_override and n_reds == 0 and len(balls_t) > 7:
        # 清彩阶段最多 母球+6彩=7 颗；无红球却检出更多 → 红球被误分类/漏检，
        # 决策层会按清彩顺序打错球（实战送分），拒绝输出方案。
        from collections import Counter
        detail = dict(Counter(b.label for b in balls_t))
        _warn_once(f"[识别异常] 无红球但总数={len(balls_t)} 明细={detail}", frame,
                   key="no_red_balls")
        scene.update({
            "status": "自动识别异常，请按 R 重新框选",
            "hint": "疑似红球漏检（球被遮挡或框选偏移）。框选紧贴台面后按 C 重试",
            "invalid": True,
        })
        return scene



    return None


def _stage_targets(ctx: _AnalysisContext) -> Optional[Dict]:
    """阶段⑥：母球/目标球选择（手动点选吸附 → 决策层 → 开局解球兜底）+ break 模式渲染早退。"""
    cfg = ctx.cfg
    W, H = ctx.W, ctx.H
    r = ctx.r
    balls_t = ctx.balls_t
    pockets_t = ctx.pockets_t
    scene = ctx.scene
    manual_cue, manual_target = ctx.manual_cue, ctx.manual_target
    picked_target = ctx.picked_target
    prefer_target = ctx.prefer_target
    turn_tracker = ctx.turn_tracker
    table_phase = ctx.table_phase
    ball_tracker = ctx.ball_tracker
    smooth = ctx.smooth
    scale = ctx.scale
    Hinv = ctx.Hinv
    # 母球 / 目标球（斯诺克决策层；支持手动录入覆盖）
    # prefer_target：上一帧目标只用于规则键完全打平时的稳定性；自动袋口
    # 现在始终按同一套切角/离袋距离优先级重新选择，不沿用旧袋口。
    cue_t = manual_cue
    cue_b = None
    if cue_t is None:
        whites = [b for b in balls_t if b.label == "白球"]
        if len(whites) > 1:
            # 只在数量变化时打印（位置每帧微变，按内容节流会失效刷屏）
            if len(whites) != _white_state["n"]:
                _white_state["n"] = len(whites)
                print(f"[白球] 检测到 {len(whites)} 个白球候选: "
                      f"{[(round(b.pos[0], 1), round(b.pos[1], 1), round(b.radius, 1)) for b in whites[:5]]}...",
                      flush=True)
        cb = vision.pick_cue(balls_t)
        cue_b = cb
        cue_t = cb.pos if cb else None
    target_t = manual_target
    target_b = None
    break_mode = False
    rule_text = ""
    ball_on = None
    if turn_tracker is not None:
        ball_on = turn_tracker.update(balls_t, table_phase == tracking.TableState.READY)
    if target_t is None and picked_target is not None and cue_t is not None and balls_t:
        # G 键点选的目标球优先于自动决策：把点击点吸附到最近的检测球，
        # 每帧重新吸附（球会动，点选的是「这颗球」而非固定坐标）
        near = min(balls_t, key=lambda b: (b.pos[0] - picked_target[0]) ** 2
                   + (b.pos[1] - picked_target[1]) ** 2)
        if physics.dist(near.pos, picked_target) <= 2.5 * r:
            target_b = near
            target_t = near.pos
    if target_t is None and cue_t is not None and balls_t:
        if cue_b is None:
            cue_b = min(balls_t, key=lambda b: (b.pos[0] - cue_t[0]) ** 2
                        + (b.pos[1] - cue_t[1]) ** 2)
        tb, _phase, rule_text = snooker.choose_target(
            balls_t, cue_b, pockets_t, r, W, H, cfg, prefer=prefer_target,
            ball_on=ball_on)
        target_b = tb
        target_t = tb.pos if tb else None
        if (target_t is None
                and bool(getattr(cfg, "opening_break_fallback", True))
                and ball_on in (None, "red")
                and cue_b is not None):
            break_b = snooker.opening_break_target(balls_t, cue_b, r)
            if break_b is not None:
                target_b = break_b
                target_t = break_b.pos
                break_mode = True
                rule_text = "开局解球：瞄准母球方向最外层红球，先撞散球架"
    scene["rule"] = rule_text
    if cue_t is None:
        scene["status"] = "未找到母球（白球）"
        scene["hint"] = "按 M 手动点击母球位置，或按 C 重新识别"
        return scene
    if target_t is None:
        scene["status"] = "未找到可行目标球"
        scene["hint"] = rule_text or "按 M 手动点击目标球位置"
        return scene

    # 帧间 EMA：从源头稳住母球/目标球位置。显示层 _smooth_segments 只
    # 治标——球心噪声经鬼球几何放大成线的晃动；源头平滑才治本。
    # 跳变保护（4r）保证换球/窗口移动时立即跟随，不会把真位移糊掉。
    if smooth is not None and ball_tracker is None:
        cue_t = _ema_point(smooth, "cue", cue_t, 4.0 * r)
        target_t = _ema_point(smooth, "target", target_t, 4.0 * r)

    # 障碍球 = 除母球/目标球外的所有球
    others: List[physics.Point] = []
    if balls_t:
        if cue_b is None:
            cue_b = min(balls_t, key=lambda b: (b.pos[0] - cue_t[0]) ** 2
                        + (b.pos[1] - cue_t[1]) ** 2)
        if target_b is None:
            target_b = min(balls_t, key=lambda b: (b.pos[0] - target_t[0]) ** 2
                           + (b.pos[1] - target_t[1]) ** 2)
        others = [b.pos for b in balls_t if b is not cue_b and b is not target_b]

    cue_radius = (_geometry_radius(cue_b, r, cfg)
                  if cue_b is not None else r)
    target_radius = (_geometry_radius(target_b, r, cfg)
                     if target_b is not None else r)
    ghost_offset = (
        float(getattr(cfg, "aim_offset_x", 0.0)),
        float(getattr(cfg, "aim_offset_y", 0.0)),
    )

    if break_mode:
        ghost_t = physics.impact_ghost(
            cue_t, target_t, r,
            cue_radius=cue_radius, target_radius=target_radius,
            offset=ghost_offset,
        )
        if ghost_t is None:
            scene["status"] = "开局解球点位退化，请重新识别"
            scene["hint"] = "按 C 重新识别球位"
            scene["invalid"] = True
            return scene
        cue_s = vision.point_table_to_screen(cue_t, Hinv)
        target_s = vision.point_table_to_screen(target_t, Hinv)
        ghost_s = vision.point_table_to_screen(ghost_t, Hinv)
        contact_t = physics.impact_contact_pos(
            cue_t, target_t, r, target_radius=target_radius)
        contact_s = (vision.point_table_to_screen(contact_t, Hinv)
                     if contact_t is not None else target_s)
        scene["segments"] = [
            {"pts": [cue_s, ghost_s], "color": "#22c55e", "width": 6},
            {"pts": [ghost_s, target_s], "color": "#f97316", "width": 5,
             "dash": True},
        ]
        scene["ghost"] = {"x": ghost_s[0], "y": ghost_s[1],
                           "r": cue_radius * scale}
        scene["contact"] = {"x": contact_s[0], "y": contact_s[1],
                             "r": target_radius * scale}
        scene["aim_geometry"] = {
            "mode": "opening_break",
            "cue_radius": float(cue_radius),
            "target_radius": float(target_radius),
        }
        scene["cue"] = {"x": cue_s[0], "y": cue_s[1]}
        scene["target"] = {"x": target_s[0], "y": target_s[1]}
        scene["shot_key"] = {"cue": cue_t, "target": target_t,
                              "pocket": None}
        scene["status"] = "开局解球 | 目标外层红球 | 不代表入袋路线"
        scene["hint"] = "把白球中心对准虚线圆（绿线末端）中心打，先撞散红球架"
        return scene



    ctx.cue_t, ctx.cue_b = cue_t, cue_b
    ctx.target_t, ctx.target_b = target_t, target_b
    ctx.others = others
    ctx.cue_radius = cue_radius
    ctx.target_radius = target_radius
    ctx.ghost_offset = ghost_offset
    ctx.break_mode = break_mode
    ctx.rule_text = rule_text
    return None


def _stage_plan(ctx: _AnalysisContext) -> Optional[Dict]:
    """阶段⑦：路线规划（手动/自动）+ 力度推荐 + 线段渲染 + 最终场景。"""
    cfg = ctx.cfg
    W, H = ctx.W, ctx.H
    r = ctx.r
    pockets_t = ctx.pockets_t
    scene = ctx.scene
    cue_t, target_t = ctx.cue_t, ctx.target_t
    others = ctx.others
    Hinv = ctx.Hinv
    scale = ctx.scale
    cue_radius, target_radius = ctx.cue_radius, ctx.target_radius
    ghost_offset = ctx.ghost_offset
    manual_pocket_idx = ctx.manual_pocket_idx
    # 瞄准方案：自动模式与 snooker.choose_target 使用同一套路线排序。
    rail_inset = max(0.0, float(getattr(cfg, "rail_inset_ratio", 1.0)) * r)
    pocket_clearance = 1.35 * r
    sel_idx = manual_pocket_idx if manual_pocket_idx is not None else cfg.selected_pocket
    if sel_idx is not None and 0 <= sel_idx < len(pockets_t):
        p = pockets_t[sel_idx]
        direct = physics.direct_shot(
            cue_t, target_t, p, r, others,
            cue_radius=cue_radius, target_radius=target_radius,
            ghost_offset=ghost_offset,
        )
        plans: List[physics.Shot] = [direct] if direct.valid and not direct.blocked else []
        if cfg.allow_kicks and (not direct.valid or direct.blocked):
            for rails in physics.KICK_SEQUENCES:
                if len(rails) > cfg.max_kicks:
                    continue
                k = physics.kick_shot(
                    cue_t, target_t, p, r, rails, W, H, others,
                    rail_inset=rail_inset, pockets=pockets_t,
                    pocket_clearance=pocket_clearance,
                    cue_radius=cue_radius, target_radius=target_radius,
                    ghost_offset=ghost_offset,
                )
                if k.valid and not k.blocked:
                    plans.append(k)
                    break
        shot = (physics.best_shot(plans, W, table_height=float(H),
                                  pocket_radius=cfg.pocket_accept_ratio * r,
                                  ball_radius=r, pockets=pockets_t)
                if plans else None)
    else:
        shot = None
        plans = physics.plan_shots(cue_t, target_t, pockets_t, r, W, H,
                                   others, cfg.allow_kicks, cfg.max_kicks,
                                   rail_inset=rail_inset,
                                   pocket_clearance=pocket_clearance,
                                   cue_radius=cue_radius,
                                   target_radius=target_radius,
                                   ghost_offset=ghost_offset)
        if plans:
            # 自动选球规则要求切角和目标到袋距离优先于总路程；沿用
            # 决策层的同一排序，避免“选中一颗球”和“显示另一条路线”。
            ranked = snooker.rank_target_shots(plans)
            if ranked:
                best = ranked[0]
                shot = best
            if ranked and cfg.auto_pocket and len(ranked) > 1:
                alts = ranked[1:4]
                alt_txt = "  ".join(
                    f"袋{idx + 1}·{s.label}" for s in alts
                    for idx, _ in [(pockets_t.index(s.pocket), s)]
                )
                scene["hint"] = "备选: " + alt_txt
            scene["route_options"] = [
                {
                    "pocket": pockets_t.index(s.pocket),
                    "label": s.label,
                    "cut_deg": s.cut_deg,
                    "total": s.total,
                    "score": physics.route_score(
                        s, W, table_height=float(H),
                        pocket_radius=cfg.pocket_accept_ratio * r,
                        ball_radius=r, pockets=pockets_t),
                }
                for s in (ranked[:6] if len(plans) > 1 else ranked)
            ]

    if shot is None:
        scene["status"] = "无可行方案"
        scene["hint"] = "该球暂无直球/解围路线，试试换目标球或袋口"
        return scene

    # 推荐力度（切角补偿 + 库边能量损耗补偿）
    power = physics.power_suggestion(shot.total, W, cfg.power_dref_ratio,
                                     cfg.power_min_pct, cfg.power_curve,
                                     cut_deg=shot.cut_deg,
                                     rails=len(shot.bounce_points),
                                     rail_loss=cfg.rail_energy_loss,
                                     gain=float(getattr(cfg, "power_gain", 1.0)),
                                     bias=float(getattr(cfg, "power_bias", 0.0)))
    pidx = pockets_t.index(shot.pocket)
    scene["pockets"][pidx]["sel"] = True

    # 绘制线段（屏幕坐标）
    segs: List[Dict] = []
    cue_s = vision.point_table_to_screen(cue_t, Hinv)
    target_s = vision.point_table_to_screen(target_t, Hinv)
    ghost_s = vision.point_table_to_screen(shot.ghost, Hinv)
    pocket_s = vision.point_table_to_screen(shot.pocket, Hinv)
    contact_t = physics.contact_pos(target_t, shot.pocket, r,
                                    target_radius=target_radius)
    contact_s = (vision.point_table_to_screen(contact_t, Hinv)
                 if contact_t is not None else target_s)

    if shot.bounce_points:
        path_s = [cue_s]
        for b in shot.bounce_points:
            path_s.append(vision.point_table_to_screen(b, Hinv))
        path_s.append(ghost_s)
        segs.append({"pts": path_s, "color": "#38bdf8", "width": 5, "dash": False})
    else:
        segs.append({"pts": [cue_s, ghost_s], "color": "#22c55e", "width": 6})
    segs.append({"pts": [ghost_s, target_s], "color": "#f97316", "width": 5, "dash": True})
    segs.append({"pts": [target_s, pocket_s], "color": "#facc15", "width": 4, "dash": True})
    # 白球切线轨迹：碰后母球沿切线方向滚动；指向袋口（摔袋）时用红色警示
    tdir, tfrac = physics.cue_tangent(shot)
    if tfrac > 0.17:
        L = physics.clamp(W * 0.55 * (power / 100.0) * tfrac, 0.12 * W, 0.8 * W)
        end_t = (shot.ghost[0] + tdir[0] * L, shot.ghost[1] + tdir[1] * L)
        end_s = vision.point_table_to_screen(end_t, Hinv)
        risk = physics.scratch_risk(shot, r, cfg.pocket_accept_ratio * r,
                                    pockets_t, max(W, H))
        segs.append({"pts": [ghost_s, end_s], "width": 3, "dash": True,
                     "color": "#ef4444" if risk > 0.25 else "#22d3ee",
                     "label": "白球切线" + ("（摔袋风险！）" if risk > 0.25 else "")})
    scene["segments"] = segs
    scene["ghost"] = {"x": ghost_s[0], "y": ghost_s[1],
                       "r": cue_radius * scale}
    scene["contact"] = {"x": contact_s[0], "y": contact_s[1],
                         "r": target_radius * scale}
    scene["aim_geometry"] = {
        "cue_radius": float(cue_radius),
        "target_radius": float(target_radius),
        "pocket": {"x": pocket_s[0], "y": pocket_s[1]},
    }
    # 唯一瞄准操作指引：白球中心对准虚线圆（鬼球）圆心，沿绿线方向击打。
    # 目标球表面上的点/预测路径不是瞄准目标，对着它们打切角球必偏。
    prev_hint = scene.get("hint", "")
    aim_hint = "把白球对准虚线圆（绿线末端）中心打"
    if prev_hint.startswith("备选"):
        scene["hint"] = aim_hint + " | " + prev_hint
    else:
        scene["hint"] = aim_hint
    # 母球 / 目标球屏幕坐标（Overlay 高亮用，避免把 22 颗球全画出来遮挡画面）
    scene["cue"] = {"x": cue_s[0], "y": cue_s[1]}
    scene["target"] = {"x": target_s[0], "y": target_s[1]}
    # 本帧方案键：App 记忆后传给下一帧，稳定目标/袋口选择
    scene["shot_key"] = {"cue": cue_t, "target": target_t, "pocket": shot.pocket}

    blocked = "被挡" if shot.blocked else "通畅"
    scene["status"] = (f"袋口{pidx + 1} | {shot.label} {blocked} | 切角 {shot.cut_deg:.0f}° "
                       f"| 力度 {power}%")
    return scene


    return None


def analyze(frame: np.ndarray, cfg: config_mod.Config,
            manual_cue: Optional[physics.Point] = None,
            manual_target: Optional[physics.Point] = None,
            manual_pocket_idx: Optional[int] = None,
            picked_target: Optional[physics.Point] = None,
            tracker: Optional[vision.TableTracker] = None,
            prefer_target: Optional[physics.Point] = None,
            pocket_tracker: Optional[vision.PocketTracker] = None,
            smooth: Optional[Dict] = None,
            self_mask: Optional[np.ndarray] = None,
            ball_tracker: Optional[tracking.BallTracker] = None,
            table_state: Optional[tracking.TableStateTracker] = None,
            turn_tracker: Optional[snooker.TurnTracker] = None,
            captured_at: Optional[float] = None) -> Dict:
    """一帧 → 场景描述（屏幕坐标，供 Overlay 直接绘制）。"""
    started_at = time.perf_counter()

    def _stamp_ms(s: Dict) -> Dict:
        """所有提前返回的 scene 也带上耗时：性能日志不再打出「分析 ?ms」。"""
        s["analysis_ms"] = round((time.perf_counter() - started_at) * 1000.0, 1)
        return s

    ctx = _AnalysisContext(
        frame=frame, cfg=cfg, manual_cue=manual_cue, manual_target=manual_target,
        manual_pocket_idx=manual_pocket_idx, picked_target=picked_target,
        tracker=tracker, prefer_target=prefer_target,
        pocket_tracker=pocket_tracker, smooth=smooth, self_mask=self_mask,
        ball_tracker=ball_tracker, table_state=table_state,
        turn_tracker=turn_tracker, captured_at=captured_at)
    ctx.W, ctx.H = cfg.table_w, cfg.table_h
    ctx.captured_at = time.monotonic() if captured_at is None else captured_at
    for _stage in (_stage_find_table, _stage_pockets, _stage_balls,
                   _stage_scene, _stage_validate, _stage_targets):
        _scene = _stage(ctx)
        if _scene is not None:
            return _stamp_ms(_scene)
    return _stamp_ms(_stage_plan(ctx))


class App:
    def __init__(self, cfg: config_mod.Config, region: Optional[List[int]] = None,
                 start_manual: bool = False):
        self.cfg = cfg
        self.region = (list(region) if region is not None else
                       (list(cfg.capture_region) if cfg.capture_region else None))
        # Every frame must be analyzed in the capture coordinate system it was
        # taken from.  The generation rejects packets that crossed an R-key or
        # automatic re-anchor region change while waiting for analysis.
        self._capture_generation = 0
        # A configured region is user-owned and must remain fixed.  Automatic
        # framing is only allowed for the first successful full-screen detect;
        # a moved window is recovered by the no-table full-screen fallback.
        self._auto_region_enabled = region is None and not cfg.capture_region
        self.mode = "manual" if start_manual else "auto"
        self.pick_mode = False                              # G 键：点选目标球中
        self.picked_target: Optional[physics.Point] = None  # 用户点选的目标球（台面坐标）
        self.manual_cue: Optional[physics.Point] = None      # 台面坐标
        self.manual_target: Optional[physics.Point] = None
        self.manual_pocket_idx: Optional[int] = None
        self.frame: Optional[np.ndarray] = None
        self.frame_store = tracking.FrameStore()
        self.scene: Dict = {"status": "启动中…", "help": HELP_TEXT}
        self.running = True
        self.overlay = None
        self.tracker = vision.TableTracker(cfg) if cfg.table_lock else None
        self.pocket_tracker = vision.PocketTracker(cfg)
        self.ball_tracker = tracking.BallTracker(cfg)
        self.table_state = tracking.TableStateTracker(cfg)
        self.turn_tracker = snooker.TurnTracker()
        self._last_shot: Optional[Dict] = None     # 上一帧方案（目标/袋口记忆）
        self._last_Hm: Optional[np.ndarray] = None  # 最近一帧 屏幕→台面单应矩阵
        self._smooth_state: Dict = {}              # 台面坐标 EMA 平滑状态（analyze 用）
        self.frame_mask: Optional[np.ndarray] = None   # 截屏时刻的叠加层自绘掩膜
        self._last_packet_sequence = -1
        self._last_analysis_at = 0.0
        self._last_perf_report = 0.0
        self._capture_err: Optional[str] = None
        self._key_handlers: Optional[Dict[str, Callable[[], None]]] = None

    # ---------- 后台线程 ----------
    def _capture_context(self) -> Tuple[Optional[Tuple[int, int, int, int]], int]:
        """Return an immutable snapshot of the current capture context."""
        region = (tuple(int(value) for value in self.region)
                  if self.region else None)
        return region, int(getattr(self, "_capture_generation", 0))

    def _advance_capture_generation(self) -> None:
        """Invalidate frames captured before a region/coordinate change."""
        self._capture_generation = int(getattr(self, "_capture_generation", 0)) + 1

    def _packet_matches_current(self, packet) -> bool:
        region, generation = self._capture_context()
        return (getattr(packet, "capture_region", None) == region
                and int(getattr(packet, "capture_generation", 0)) == generation)

    def _capture_loop(self) -> None:
        last_error = None
        last_report = 0.0
        while self.running:
            try:
                capture_region, capture_generation = self._capture_context()
                capture_arg = (list(capture_region)
                               if capture_region is not None else None)
                self.frame = capture.grab(capture_arg)
                # 同步叠加层自绘掩膜：与刚抓到的帧同一时刻的画面内容，
                # analyze() 用它把自家画线填回台呢色，杜绝自截屏干扰。
                self.frame_mask = self._snapshot_overlay_mask(capture_region)
                # 一个 packet 同时封存帧和遮罩，检测线程不会再拿到新帧+旧遮罩。
                store = getattr(self, "frame_store", None)
                if store is None:
                    store = self.frame_store = tracking.FrameStore()
                store.publish(self.frame, self.frame_mask,
                              capture_region, capture_generation)
                if self._capture_err:
                    print("[截屏] 已恢复", flush=True)
                self._capture_err = None
                last_error = None
            except Exception as e:                      # pragma: no cover
                # 截屏失败不应终止 UI：mss/GDI 可能在窗口切换、权限变化或
                # 显示器初始化期间短暂失败，保留 Overlay 让用户能按 R/C/Esc。
                message = f"{type(e).__name__}: {e}" or type(e).__name__
                self._capture_err = message
                now = time.monotonic()
                if message != last_error or now - last_report >= 5.0:
                    print(f"[截屏] {message}；1 秒后重试", flush=True)
                    last_error = message
                    last_report = now
                time.sleep(1.0)
                continue
            time.sleep(1.0 / max(1.0, self.cfg.capture_fps))

    def _snapshot_overlay_mask(
            self, capture_region: Optional[Tuple[int, int, int, int]] = None
    ) -> Optional[np.ndarray]:
        """当前叠加层画过的像素（屏幕坐标 → 截屏区域坐标）。

        读取不抛异常：叠加层可能尚未初始化/已销毁/是 Tk 回退路径。
        区域模式只保留区域内的掩膜，与帧的 (h, w) 对齐。
        """
        try:
            overlay = self.overlay
            native = getattr(overlay, "_native", None) if overlay else None
            dm = getattr(native, "drawn_mask", None) if native else None
            if dm is None:
                return None
            if dm.dtype != np.uint8:
                dm = (dm > 0).astype(np.uint8)
            if capture_region is not None:
                rx, ry, rw, rh = (int(v) for v in capture_region)
                if rx < 0 or ry < 0 or rx + rw > dm.shape[1] or ry + rh > dm.shape[0]:
                    return None
                return np.ascontiguousarray(dm[ry:ry + rh, rx:rx + rw])
            return dm
        except Exception:
            return None

    def _detect_loop(self) -> None:
        while self.running:
            store = getattr(self, "frame_store", None)
            if store is None:
                time.sleep(0.1)
                continue
            # 等发布通知代替 250Hz 轮询：新帧一到立即分析（延迟↓），
            # 同时消除检测线程的空转唤醒与 FrameStore 锁竞争。
            packet = store.wait_for_new(
                getattr(self, "_last_packet_sequence", -1), timeout=0.05)
            now = time.monotonic()
            analysis_fps = max(1.0, float(getattr(self.cfg, "analysis_fps", 30.0)))
            fresh = packet is not None and packet.sequence != getattr(self, "_last_packet_sequence", -1)
            due = now - getattr(self, "_last_analysis_at", 0.0) >= 1.0 / analysis_fps
            if fresh and due:
                self._last_packet_sequence = packet.sequence
                # Region changes are handled by the UI/capture thread while
                # analysis may be busy.  Never attach the current region
                # origin or tracker state to an older frame.
                if not self._packet_matches_current(packet):
                    continue
                self._last_analysis_at = now
                if self.mode == "region":
                    # 手动框选模式：不覆盖场景，等用户拖框（否则框选提示被冲掉）
                    pass
                else:
                    try:
                        prefer_t = self._last_shot["target"] if self._last_shot else None
                        scene = analyze(
                            packet.frame, self.cfg,
                            manual_cue=self.manual_cue, manual_target=self.manual_target,
                            manual_pocket_idx=self.manual_pocket_idx,
                            picked_target=self.picked_target,
                            tracker=self.tracker,
                            pocket_tracker=self.pocket_tracker,
                            prefer_target=prefer_t,
                            smooth=self._smooth_state,
                            self_mask=packet.self_mask,
                            ball_tracker=self.ball_tracker,
                            table_state=self.table_state,
                            turn_tracker=self.turn_tracker,
                            captured_at=packet.captured_at,
                        )
                        # 捕获区域模式：识别输出的是区域局部坐标，而 Overlay
                        # 全屏窗口从 (0,0) 画起——不补偿 region 原点会让所有
                        # 绘制整体偏移 (rx, ry)。先把场景平移到全屏坐标系。
                        if packet.capture_region is not None:
                            self._offset_scene(
                                scene, float(packet.capture_region[0]),
                                float(packet.capture_region[1]))
                        # A region can change during the relatively expensive
                        # OpenCV pass.  Drop that result before it can update
                        # the visible scene or automatic framing state.
                        if not self._packet_matches_current(packet):
                            continue
                        if self.pick_mode:
                            # 点选模式提示不被每帧识别结果冲掉
                            scene["hint"] = "点选目标：点击要打的球（母球/袋口自动选），再按 G 取消"
                        self._last_Hm = scene.get("H")
                        scene["latency_ms"] = round((time.monotonic() - packet.captured_at) * 1000.0, 1)
                        self.scene = scene
                        if scene.get("occluded"):
                            self._last_shot = None
                        elif scene.get("shot_key"):
                            self._last_shot = scene["shot_key"]
                        # 球位已经由 BallTracker 的中位数窗口稳定；再做显示层
                        # EMA 会把快速更新重新变成滞后线，因此 v2 不再二次平滑。
                        # 自动框选：全屏模式下首次检测到台面即收缩截屏区域（稳定识别）
                        if scene.get("table_quad"):
                            self._auto_region(scene["table_quad"])
                        # 台面丢失且此前框选了区域：回到全屏重新找（窗口可能移动了）
                        if scene.get("status") == "未检测到台面" and self.region:
                            self.region = None
                            self.cfg.capture_region = None
                            self._auto_region_enabled = True
                            self._advance_capture_generation()
                            self._reset_tracking()
                        if now - self._last_perf_report >= 5.0:
                            self._last_perf_report = now
                            print(f"[性能] 分析 {scene.get('analysis_ms', '?')}ms | "
                                  f"端到端 {scene['latency_ms']}ms | "
                                  f"状态 {scene.get('table_state', '?')}", flush=True)
                    except Exception as e:
                        # 分析帧失败必须可见：以前静默吞掉，画面空白却查不到日志
                        _warn_once(f"[识别异常] 分析帧失败: {type(e).__name__}: {e}",
                                   packet.frame,
                                   key=f"analyze_fail:{type(e).__name__}")
                        self.scene = {"status": f"识别异常: {e}", "help": HELP_TEXT}
            elif fresh:
                # 新帧但未到期（analysis_fps 节流）：睡到到期即可，
                # 到期后 wait_for_new 立刻返回当前帧，不再额外轮询。
                time.sleep(max(0.0, min(
                    0.05,
                    (1.0 / analysis_fps) - (now - getattr(self, "_last_analysis_at", 0.0)))))

    @staticmethod
    def _offset_scene(scene: Dict, dx: float, dy: float) -> None:
        """把场景内所有屏幕坐标（区域局部）平移到全屏全局。shot_key 存的是
        台面坐标，不动。"""
        if not dx and not dy:
            return
        q = scene.get("table_quad")
        if q:
            scene["table_quad"] = [[x + dx, y + dy] for x, y in q]
        for p in scene.get("pockets", []):
            p["x"] += dx
            p["y"] += dy
        for b in scene.get("balls", []):
            b["x"] += dx
            b["y"] += dy
        for seg in scene.get("segments", []):
            seg["pts"] = [(x + dx, y + dy) for x, y in seg["pts"]]
        for k in ("ghost", "contact", "cue", "target"):
            v = scene.get(k)
            if v:
                v["x"] += dx
                v["y"] += dy

    def _reset_tracking(self) -> None:
        """捕获区域变更后丢弃旧坐标系的 tracker 和瞄准历史。"""
        self.tracker = vision.TableTracker(self.cfg) if self.cfg.table_lock else None
        self.pocket_tracker = vision.PocketTracker(self.cfg)
        self.ball_tracker = tracking.BallTracker(self.cfg)
        self.table_state = tracking.TableStateTracker(self.cfg)
        self.turn_tracker = snooker.TurnTracker()
        self._last_shot = None
        self._last_Hm = None
        self._smooth_state = {}
        self._last_packet_sequence = -1

    def _auto_region(self, quad: List) -> None:
        """在一次可靠的全屏检测后建立固定捕获区域。

        捕获区域一旦由用户 R 键选定，或由首次全屏检测建立，就不再由
        每帧的四边形候选改写。窗口移动后当前区域会失去台面，检测循环
        再统一回到全屏并重新建立区域，避免单帧误检把锁定框带走。
        """
        if not getattr(self, "_auto_region_enabled", True) or self.region:
            return
        q = np.asarray(quad, dtype=float)
        pad = 25
        x0 = max(0, int(q[:, 0].min()) - pad)
        y0 = max(0, int(q[:, 1].min()) - pad)
        x1 = int(q[:, 0].max()) + pad
        y1 = int(q[:, 1].max()) + pad
        self.region = [x0, y0, x1 - x0, y1 - y0]
        self.cfg.capture_region = self.region
        self.cfg.save()
        self._auto_region_enabled = False
        self._advance_capture_generation()
        print(f"[自动框选] 首次锁定区域: {self.region}", flush=True)
        # 区域变了=坐标系原点变了，历史跟踪全部作废
        self._reset_tracking()

    # ---------- Overlay 交互 ----------
    # ---------- 按键分发（表驱动：新热键在 _build_key_handlers 注册即可） ----------

    def _build_key_handlers(self) -> Dict[str, Callable[[], None]]:
        """keysym → 处理函数映射；所有热键行为实现在 _key_* 方法。"""
        handlers: Dict[str, Callable[[], None]] = {
            "escape": self.quit,
            "0": self._key_auto_pocket,
            "g": self._key_pick_toggle,
            "m": self._key_manual_mode,
            "r": self._key_reselect_region,
            "k": self._key_toggle_kicks,
            "p": self._key_toggle_auto_pocket,
            "q": self._key_toggle_turn,
            "o": self._key_toggle_turn,
            "x": self._key_toggle_click_through,
            "t": self._key_toggle_overlay,
            "b": self._key_toggle_balls,
            "c": self._redetect,
            "f12": self._key_dump_state,
        }
        for i in range(6):                       # 1-6 手动指定袋口
            handlers[str(i + 1)] = (
                lambda idx=i: self._key_select_pocket(idx))
        return handlers

    def on_key(self, keysym: str) -> None:
        handlers = getattr(self, "_key_handlers", None)
        if handlers is None:                    # 惰性构建（测试会轻量构造 App）
            handlers = self._key_handlers = self._build_key_handlers()
        handler = handlers.get(keysym)
        if handler is not None:
            handler()

    def _key_select_pocket(self, idx: int) -> None:
        """1-6：手动指定袋口，关闭自动选袋。"""
        self.cfg.selected_pocket = idx
        self.cfg.auto_pocket = False
        self.cfg.save()
        self._redetect()

    def _key_auto_pocket(self) -> None:
        """0：恢复自动选袋。"""
        self.cfg.selected_pocket = -1
        self.cfg.auto_pocket = True
        self.cfg.save()
        self._redetect()

    def _key_pick_toggle(self) -> None:
        """G：点选目标开关——点哪颗球立即算哪颗的方案（母球/袋口自动）。"""
        self.pick_mode = not self.pick_mode
        if self.pick_mode:
            self._set_click_through(False)
            if self.overlay:
                self.overlay.begin_manual()
            self.scene["hint"] = "点选目标：点击要打的球（母球/袋口自动选），再按 G 取消"
        else:
            self.picked_target = None
            if self.overlay:
                self.overlay.stop_manual()
            self._set_click_through(True)
        self._redetect()

    def _key_manual_mode(self) -> None:
        """M：手动录入模式开关（依次点击 母球 → 目标球 → 袋口）。"""
        entering = self.mode != "manual"
        self.mode = "manual" if entering else "auto"
        self.manual_cue = self.manual_target = None
        self.manual_pocket_idx = None
        # transparentcolor 透明区收不到 tk 鼠标事件，手动录入的点击
        # 由 overlay 的 GetCursorPos 轮询捕获（与 R 键框选同机制）
        if self.overlay:
            if entering:
                self.overlay.begin_manual()
            else:
                self.overlay.stop_manual()
        self._set_click_through(self.mode != "manual")
        self.scene["hint"] = ("手动录入：依次点击 母球 → 目标球 → 袋口" if self.mode == "manual"
                              else "")
        self._redetect()

    def _key_reselect_region(self) -> None:
        """R：重新框选捕获区域（始终从全屏开始，脱离旧坐标系）。"""
        # 重新框选必须脱离旧捕获区域，否则旧区域越界/错位时，
        # 新框选完成后仍可能继续截取错误位置。R 模式始终从全屏开始。
        old_region = self.region or self.cfg.capture_region
        self.region = None
        self.cfg.capture_region = None
        self._auto_region_enabled = False
        self._advance_capture_generation()
        print(f"[框选] 进入全屏选择，清除旧区域: {old_region}", flush=True)
        # 即使原本就是全屏模式，也要丢弃旧四边形和旧瞄准线；否则
        # 框选期间用户会看到上一套坐标系的白色边框仍在漂移。
        self._reset_tracking()
        try:
            self.cfg.save()
        except OSError as exc:
            print(f"[框选] 无法保存全屏捕获状态: {exc}", flush=True)
        self.mode = "region"
        self._set_click_through(False)
        if self.overlay:
            self.overlay.begin_region()
        self.scene = {
            "status": "等待框选球桌区域",
            "hint": "已切换全屏捕获：框选绿色台面（可稍大一点），按住左键拖到对角松开",
            "help": HELP_TEXT,
        }

    def _key_toggle_kicks(self) -> None:
        """K：允许/禁止库边解球。"""
        self.cfg.allow_kicks = not self.cfg.allow_kicks
        self.cfg.save()
        self._redetect()

    def _key_toggle_auto_pocket(self) -> None:
        """P：自动/手动选袋开关。"""
        self.cfg.auto_pocket = not self.cfg.auto_pocket
        self.cfg.selected_pocket = -1
        self.cfg.save()
        self._redetect()

    def _key_toggle_turn(self) -> None:
        """Q/O：红球阶段手动切换红/彩球权（O 为旧版兼容热键）。

        换手/失误无法可靠地由单帧视觉判断，允许用户显式切换当前球权。
        """
        balls = [type("BallRef", (), {"label": b.get("label")})()
                 for b in self.scene.get("balls", [])]
        ball_on = self.turn_tracker.toggle_red_color(balls)
        self.scene["hint"] = ("规则状态：打红球（Q 可切换为彩球）" if ball_on == "red"
                              else "规则状态：红球后选彩球（Q 可切回红球）"
                              if ball_on == "color"
                              else "规则状态：清彩顺序 黄→绿→棕→蓝→粉→黑")
        self._redetect()

    def _key_toggle_click_through(self) -> None:
        """X：点击穿透开关。"""
        on = not getattr(self.overlay, "_click_through", False)
        self._set_click_through(on)

    def _key_toggle_overlay(self) -> None:
        """T：显示/隐藏瞄准层。"""
        if self.overlay:
            self.overlay.toggle_visible()

    def _key_toggle_balls(self) -> None:
        """B：全球标注 / 极简模式切换。"""
        if self.overlay:
            on = self.overlay.toggle_balls()
            self.scene["hint"] = "全球标注：开" if on else "极简模式：仅母球/目标球"
            self._redetect()

    def _key_dump_state(self) -> None:
        """F12：打印当前运行状态。"""
        print(f"[状态] scene={self.scene.get('status')} | region={self.region} "
              f"| mode={self.mode} | last_shot={self._last_shot is not None}", flush=True)

    def _set_click_through(self, on: bool) -> None:
        if self.overlay:
            self.overlay.set_click_through(on)

    def _redetect(self) -> None:
        # 让下一张已捕获帧立即重跑；不再依赖旧版检测循环计数器。
        self._last_packet_sequence = -1

    def on_click(self, x: int, y: int) -> None:
        if self.mode == "region":
            self._region_start = (x, y)
            return
        if self.pick_mode:
            # G 键点选模式：把点击点转台面坐标，analyze() 每帧把它吸附到
            # 最近检测球（≤2.5r），立即出该球的瞄准方案（母球/袋口自动）
            Hm = self._last_Hm
            if Hm is None:
                self.scene["hint"] = "点选失败：尚未识别到台面，先等识别稳定或按 R 框选"
                return
            sx, sy = x, y
            if self.region:
                sx -= self.region[0]
                sy -= self.region[1]
            self.picked_target = vision.point_screen_to_table((sx, sy), Hm)
            self.pick_mode = False
            if self.overlay:
                self.overlay.stop_manual()
            self._set_click_through(True)
            self.scene["hint"] = "已点选目标球，正在计算击球点位…（再按 G 取消锁定）"
            self._redetect()
            return
        if self.mode != "manual":
            return
        # 手动录入：把屏幕点转成台面坐标
        Hm = self._last_Hm
        if Hm is None:
            return
        sx, sy = x, y
        if self.region:
            # 捕获区域模式：点击是全屏坐标，识别是区域局部坐标，先扣原点
            sx -= self.region[0]
            sy -= self.region[1]
        pt = vision.point_screen_to_table((sx, sy), Hm)
        if self.manual_cue is None:
            self.manual_cue = pt
            self.scene["hint"] = "已设母球，请点击目标球"
        elif self.manual_target is None:
            self.manual_target = pt
            self.scene["hint"] = "已设目标球，请点击袋口"
        else:
            # 选最近袋口
            W, H = self.cfg.table_w, self.cfg.table_h
            pockets = physics.default_pockets(W, H)
            idx = min(range(len(pockets)), key=lambda i:
                      (pockets[i][0] - pt[0]) ** 2 + (pockets[i][1] - pt[1]) ** 2)
            self.manual_pocket_idx = idx
            self.mode = "auto"
            self._set_click_through(True)
            self.scene["hint"] = f"已选袋口 {idx + 1}"
        self._redetect()

    def on_drag(self, x: int, y: int) -> None:
        pass

    def on_drag_end(self, x: int, y: int) -> None:
        if self.mode == "region":
            # 起点：优先用 overlay 轮询记录的起点（Windows 框选不经过 tk 点击事件，
            # 之前 _region_start 从未被设置，导致起点=终点 → "区域太小"）
            sx, sy = (x, y)
            if self.overlay and self.overlay._region_start:
                sx, sy = self.overlay._region_start
            elif getattr(self, "_region_start", None):
                sx, sy = self._region_start
            x0, x1 = sorted((sx, x))
            y0, y1 = sorted((sy, y))
            # GetCursorPos 使用屏幕坐标；拖拽过程中若越过屏幕边缘，
            # 先裁剪再保存，避免产生下一次启动无法使用的越界区域。
            if os.name == "nt":
                try:
                    import ctypes
                    sw = int(ctypes.windll.user32.GetSystemMetrics(0))
                    sh = int(ctypes.windll.user32.GetSystemMetrics(1))
                    if sw > 0 and sh > 0:
                        x0 = max(0, min(x0, sw - 1))
                        y0 = max(0, min(y0, sh - 1))
                        x1 = max(0, min(x1, sw))
                        y1 = max(0, min(y1, sh))
                except (AttributeError, OSError, TypeError, ValueError):
                    pass
            try:
                if x1 - x0 > 40 and y1 - y0 > 40:
                    self.region = [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]
                    self.cfg.capture_region = self.region
                    self._auto_region_enabled = False
                    self._advance_capture_generation()
                    # 区域变了=坐标系变了，旧跟踪/单应阵全部作废
                    self._reset_tracking()
                    try:
                        self.cfg.save()
                    except OSError as exc:
                        # 保存失败不应让本次框选卡在 region 模式；当前进程
                        # 仍可使用刚选的区域，下次启动会回到全屏并提示重选。
                        print(f"[框选] 区域已应用但保存失败: {exc}", flush=True)
                        self.scene["hint"] = f"捕获区域已设为 {self.region}（本次有效，保存失败）"
                    else:
                        self.scene["hint"] = f"捕获区域已设为 {self.region}（已保存）"
                    print(f"[框选] 完成: start=({sx},{sy}) end=({x},{y}) "
                          f"region={self.region}", flush=True)
                else:
                    self.scene["hint"] = "区域太小，未修改"
                    print(f"[框选] 忽略过小区域: start=({sx},{sy}) end=({x},{y})",
                          flush=True)
            finally:
                # 无论配置文件是否可写，都必须结束当前一次选择并恢复自动模式。
                self.mode = "auto"
                self._region_start = None
                self._set_click_through(True)
                self._redetect()

    def quit(self) -> None:
        self.running = False
        if self.overlay:
            self.overlay.destroy()

    # ---------- 主循环 ----------
    def _tick(self) -> None:
        if not self.running:
            return
        # 原生分层窗口被外部关闭（新实例的互斥体恢复流程发的 WM_CLOSE）
        # ⇒ 本实例应当正常退出并释放互斥体，否则会让新实例永远卡死。
        if self.overlay and os.name == "nt":
            try:
                native = getattr(self.overlay, "_native", None)
                hwnd = getattr(native, "_hwnd", None) if native else None
                if hwnd is not None:
                    import ctypes
                    if not ctypes.windll.user32.IsWindow(hwnd):
                        print("[启动] Overlay 窗口被关闭，本实例退出。", flush=True)
                        self.quit()
                        return
            except Exception:
                pass
        if self._capture_err:
            self.scene = {"status": f"截屏失败: {self._capture_err}",
                          "hint": "请确认 mss 可用（Windows 上运行），按 Esc 退出"}
        if self.overlay:
            try:
                self.overlay.render(self.scene)
            except Exception as e:
                # 渲染异常不能让 after 链断掉（否则界面冻结、热键失效）
                self.scene = {"status": f"渲染异常: {e}", "help": HELP_TEXT}
                print(f"[render] 异常: {e}", flush=True)
            self.overlay.root.after(33, self._tick)

    def run(self) -> None:
        from aimtool.overlay import Overlay          # 延迟导入：无显示环境（demo）不依赖 tk

        self.overlay = Overlay(self.on_key, self.on_click, self.on_drag, self.on_drag_end)
        self._set_click_through(self.mode != "manual")
        threading.Thread(target=self._capture_loop, daemon=True).start()
        threading.Thread(target=self._detect_loop, daemon=True).start()
        self._tick()
        try:
            self.overlay.run()
        except KeyboardInterrupt:
            self.quit()


# ---------- 无界面模式 ----------

def demo() -> int:
    """用合成台面跑通 识别→瞄准 全流程（无显示环境可运行的自检）。"""
    import synth

    print("=== QQ 2D桌球瞄准器 demo（合成台面自检）===")
    cfg = config_mod.Config()
    ok = True
    for seed in range(3):
        img, meta = synth.snooker_layout(seed)
        # 真值：synth 的球位就是画布坐标（与台面同坐标系），直接线性映射到标准坐标
        fx0, fy0, fx1, fy1 = meta["felt"]

        def canvas_to_table(p):
            return ((p[0] - fx0) * cfg.table_w / (fx1 - fx0),
                    (p[1] - fy0) * cfg.table_h / (fy1 - fy0))

        truth = [(b["label"], canvas_to_table(b["pos"])) for b in meta["balls"]]
        quad = vision.find_table(img, cfg)
        if quad is None:
            print(f"[seed {seed}] 台面识别失败")
            ok = False
            continue
        Hm = vision.homography(quad, cfg.table_w, cfg.table_h)
        warped = vision.warp_table(img, Hm, cfg.table_w, cfg.table_h)
        balls = vision.detect_balls(warped, cfg.ball_radius_ratio * cfg.table_w, cfg)
        truth = [(b["label"], canvas_to_table(b["pos"])) for b in meta["balls"]]
        print(f"[seed {seed}] 检出 {len(balls)} 球 / 真值 {len(truth)} 球")
        used = set()
        for label, t in truth:
            cands = [(i, b) for i, b in enumerate(balls)
                     if b.label == label and i not in used]
            if not cands:
                print(f"   MISS {label}")
                ok = False
                continue
            i, b = min(cands, key=lambda x: np.hypot(*(np.array(x[1].pos) - np.array(t))))
            used.add(i)
            err = np.hypot(*(np.array(b.pos) - np.array(t)))
            # 开局红球三角内部球无可见边缘（相切重叠），合成极限 ~25px；
            # 实战中先打边缘球、一杆后散开即回到 <5px。红球阈值放宽到 30px。
            limit = 30.0 if label == "红球" else 12.0
            flag = "OK " if err < limit else "BAD"
            if err >= limit:
                ok = False
            print(f"   {flag} {label} 误差 {err:.1f}px")
        # 瞄准方案
        cue = next((b for b in balls if b.label == "白球"), None)
        if cue:
            scene = analyze(img, cfg)
            print(f"   方案: {scene.get('rule')} | {scene.get('status')}")
        else:
            print("   MISS 母球")
            ok = False
    print("=== 结果:", "PASS" if ok else "FAIL", "===")
    return 0 if ok else 1


def analyze_frame_file(path: str) -> int:
    import cv2

    img = cv2.imread(path)
    if img is None:
        print(f"无法读取图片: {path}")
        return 1
    cfg = config_mod.Config.load()   # 与实机运行同一套配置，便于离线排查
    scene = analyze(img, cfg)
    print("status:", scene.get("status"))
    if scene.get("hint"):
        print("hint:", scene["hint"])
    for b in scene.get("balls", []):
        print(f"  球 {b['label']} @ ({b['x']:.0f},{b['y']:.0f})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="QQ 2D桌球 辅助瞄准器")
    ap.add_argument("--demo", action="store_true", help="无界面自检（合成台面）")
    ap.add_argument("--frame", metavar="PATH", help="分析一张游戏截图")
    ap.add_argument("--region", nargs=4, type=int, metavar=("X", "Y", "W", "H"),
                    help="预设捕获区域")
    ap.add_argument("--manual", action="store_true", help="以手动录入模式启动")
    ap.add_argument("--fps", type=float, default=None, help="截屏帧率")
    args = ap.parse_args()

    if not args.demo and not args.frame:
        print("=" * 52, flush=True)
        print(f"  QQ 2D桌球 斯诺克瞄准器  v{APP_VERSION} PID={os.getpid()} "
              f"{time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        print("  本行若能写入 runtime.log，说明程序已常驻", flush=True)
        print("  日志编码 UTF-8；若显示乱码请用记事本打开 runtime.log", flush=True)
        print("  Esc 退出 · F12 状态 · T 隐藏/显示瞄准线", flush=True)
        print("=" * 52, flush=True)

    if args.demo:
        return demo()
    if args.frame:
        return analyze_frame_file(args.frame)

    if not _acquire_instance():
        print("[启动] 程序已经运行，忽略本次重复启动。", flush=True)
        return 0

    cfg = config_mod.Config.load()
    if not args.region:
        saved_region = cfg.capture_region
        cfg.capture_region = _valid_saved_region(saved_region)
        if saved_region and cfg.capture_region is None:
            print(f"[配置] 丢弃越过当前屏幕的旧捕获区域: {saved_region}", flush=True)
            try:
                cfg.save()
            except OSError as exc:
                print(f"[配置] 无法保存全屏回退: {exc}", flush=True)
    if args.fps:
        cfg.capture_fps = args.fps
    region = list(args.region) if args.region else None
    if region:
        cfg.capture_region = region
        cfg.save()
    App(cfg, region=region, start_manual=args.manual).run()
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        # 双击 .py 或启动器异常退出时保留完整 traceback，避免控制台关闭后
        # 无法判断是 Tk、Win32 截屏还是配置初始化失败。
        traceback.print_exc()
        try:
            error_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "startup_error.log")
            with open(error_path, "w", encoding="utf-8") as f:
                traceback.print_exc(file=f)
            print(f"[启动失败] 详细错误已写入: {error_path}", flush=True)
        except Exception as log_exc:
            print(f"[启动失败] 无法写入错误日志: {log_exc}", flush=True)
        raise
