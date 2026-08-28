"""主程序交互回归测试（不创建 Tk 窗口）。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import cv2
import synth

from aimtool import config, overlay as overlay_mod, tracking, snooker, vision
import main as main_mod
from main import App


W, H = 2000.0, 1000.0


def test_manual_clicks_use_region_local_homography():
    """手动点击的全屏坐标应扣除捕获区域原点后再映射到台面。"""
    app = App.__new__(App)
    app.cfg = config.Config()
    app.mode = "manual"
    app.pick_mode = False
    app.region = [137, 241, 2200, 1200]
    quad = np.array([
        [31.0, 27.0], [2034.0, 41.0], [2020.0, 1042.0], [18.0, 1026.0]
    ], dtype=np.float32)
    app._last_Hm = vision.homography(quad, W, H)
    app.manual_cue = None
    app.manual_target = None
    app.manual_pocket_idx = None
    app.overlay = None
    app.scene = {}
    app._redetect = lambda: None
    app._set_click_through = lambda _on: None

    Hinv = np.linalg.inv(app._last_Hm)

    def click_table(pt):
        local = vision.point_table_to_screen(pt, Hinv)
        app.on_click(round(local[0] + app.region[0]),
                     round(local[1] + app.region[1]))

    cue = (420.0, 730.0)
    target = (1230.0, 410.0)
    click_table(cue)
    click_table(target)
    click_table((0.0, 0.0))

    assert np.hypot(*(np.array(app.manual_cue) - cue)) < 2.0
    assert np.hypot(*(np.array(app.manual_target) - target)) < 2.0
    assert app.manual_pocket_idx == 0
    assert app.mode == "auto"


def test_bad_frame_names_are_unique_and_legacy_files_are_preserved(tmp_path, monkeypatch):
    """异常帧不应同秒覆盖，保留上限也不能删除旧格式诊断文件。"""
    monkeypatch.setattr(main_mod, "_BAD_FRAME_DIR", str(tmp_path))
    monkeypatch.setattr(main_mod, "_BAD_FRAME_KEEP", 2)
    legacy = tmp_path / "bad_123456.png"
    legacy.write_bytes(b"legacy")

    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    paths = [main_mod._save_bad_frame(frame) for _ in range(3)]
    assert all(paths)
    assert len(set(paths)) == 3
    assert len(list(tmp_path.glob("bad2_*.png"))) == 2
    assert legacy.exists()


def test_analyze_pauses_when_table_is_covered_by_ui():
    """弹窗覆盖台面时，主流程不能输出鬼球。"""
    img, _ = synth.random_layout(seed=10)
    cv2.rectangle(img, (760, 390), (1240, 610), (235, 235, 235), -1)
    scene = main_mod.analyze(img, config.Config())
    assert scene.get("occluded") is True
    assert scene.get("ghost") is None


def test_overlay_does_not_force_crosshair_cursor(monkeypatch):
    """穿透层应交还游戏指针，不能把鼠标固定成 Tk 十字光标。"""
    class Canvas:
        def __init__(self):
            self.options = {}

        def configure(self, **kwargs):
            self.options.update(kwargs)

    ov = overlay_mod.Overlay.__new__(overlay_mod.Overlay)
    ov.canvas = Canvas()
    ov._click_through = False
    monkeypatch.setattr(overlay_mod, "TRANSPARENT_ON_NT", False)
    ov.set_click_through(True)
    assert ov._click_through is True
    assert ov.canvas.options["cursor"] == "arrow"


def test_capture_loop_retries_without_stopping_app(monkeypatch):
    """一次截屏失败只应显示错误并重试，不能让 Overlay 生命周期结束。"""
    app = App.__new__(App)
    app.running = True
    app.region = None
    app.cfg = config.Config(capture_fps=1000.0)
    app.frame = None
    app._capture_err = None
    calls = []

    def grab(_region):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("temporary capture error")
        app.running = False
        return np.zeros((2, 2, 3), dtype=np.uint8)

    monkeypatch.setattr(main_mod.capture, "grab", grab)
    monkeypatch.setattr(main_mod.time, "sleep", lambda _seconds: None)
    app._capture_loop()

    assert len(calls) == 2
    assert app.frame.shape == (2, 2, 3)
    assert app._capture_err is None


def test_r_key_clears_old_capture_region_before_selection(monkeypatch):
    """重新框选不能继续使用上一次错误/越界的捕获区域。"""
    app = App.__new__(App)
    app.cfg = config.Config(capture_region=[732, 668, 1036, 573])
    app.region = list(app.cfg.capture_region)
    app.mode = "auto"
    app.scene = {}
    app.overlay = None
    app._set_click_through = lambda _on: None
    app._reset_tracking = lambda: None
    app._redetect = lambda: None
    monkeypatch.setattr(app.cfg, "save", lambda: None)

    app.on_key("r")

    assert app.mode == "region"
    assert app.region is None
    assert app.cfg.capture_region is None
    assert "全屏捕获" in app.scene["hint"]


def test_region_selection_uses_overlay_start_and_finishes(monkeypatch):
    """Windows 轮询提供的起点应生成正确区域并退出框选模式。"""
    app = App.__new__(App)
    app.cfg = config.Config()
    app.region = None
    app.mode = "region"
    app.scene = {}
    app._region_start = None
    app.overlay = type("OverlayRef", (), {"_region_start": (100, 120)})()
    app._set_click_through = lambda _on: None
    app._reset_tracking = lambda: None
    app._redetect = lambda: None
    monkeypatch.setattr(app.cfg, "save", lambda: None)

    app.on_drag_end(700, 620)

    assert app.region == [100, 120, 600, 500]
    assert app.cfg.capture_region == [100, 120, 600, 500]
    assert app.mode == "auto"
    assert app._region_start is None


def test_q_key_toggles_red_and_color_target_mode(monkeypatch):
    """Q 键应立即切换红球阶段的 ball-on 状态。"""
    app = App.__new__(App)
    app.cfg = config.Config()
    app.scene = {"balls": [{"label": "白球"}, {"label": "红球"},
                            {"label": "黑球"}]}
    app.turn_tracker = snooker.TurnTracker()
    app.overlay = None
    app._redetect = lambda: None

    app.on_key("q")
    assert app.turn_tracker.ball_on == "color"
    assert "彩球" in app.scene["hint"]
    app.on_key("q")
    assert app.turn_tracker.ball_on == "red"
    assert "红球" in app.scene["hint"]


def test_analyze_hides_aim_until_ball_positions_settle():
    """跨帧确认期间只能显示球位，READY 后才允许输出瞄准线。"""
    img, _ = synth.random_layout(seed=12)
    cfg = config.Config(track_confirm_frames=1, settle_seconds=0.20)
    table_tracker = vision.TableTracker(cfg)
    pocket_tracker = vision.PocketTracker(cfg)
    ball_tracker = tracking.BallTracker(cfg)
    table_state = tracking.TableStateTracker(cfg)
    turn_tracker = snooker.TurnTracker()

    scenes = [main_mod.analyze(
        img.copy(), cfg, tracker=table_tracker,
        pocket_tracker=pocket_tracker, ball_tracker=ball_tracker,
        table_state=table_state, turn_tracker=turn_tracker,
        captured_at=stamp,
    ) for stamp in (0.0, 0.10, 0.40)]

    assert scenes[0]["table_state"] == "stabilizing"
    assert scenes[1]["invalid"] is True
    assert scenes[1].get("ghost") is None
    assert scenes[2]["table_state"] == "ready"
    assert scenes[2].get("ghost") is not None


def test_analyze_shows_explicit_opening_break_aim_for_dense_rack():
    """开局红球架没有入袋路线时仍给出明确的解球碰撞点。"""
    img, _ = synth.snooker_layout(seed=4)
    scene = main_mod.analyze(img, config.Config())
    assert scene["aim_geometry"]["mode"] == "opening_break"
    assert scene["status"].startswith("开局解球")
    assert scene.get("ghost") is not None
    assert len(scene.get("segments", [])) == 2
    assert all(not pocket["sel"] for pocket in scene["pockets"])
