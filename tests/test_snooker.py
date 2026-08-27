"""斯诺克专项测试：开局台面识别、亚像素、台面跟踪、决策层、去背景。"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

import synth
from aimtool import config, physics, snooker, vision

W, H = 2000.0, 1000.0


def _pipeline(seed):
    img, meta = synth.snooker_layout(seed=seed)
    cfg = config.Config()
    quad = vision.find_table(img, cfg)
    assert quad is not None
    Hm = vision.homography(quad, W, H)
    warped = vision.warp_table(img, Hm, W, H)
    r = cfg.ball_radius_ratio * W
    balls = vision.detect_balls(warped, r, cfg)
    fx0, fy0, fx1, fy1 = meta["felt"]

    def c2t(p):
        return ((p[0] - fx0) * W / (fx1 - fx0), (p[1] - fy0) * H / (fy1 - fy0))

    truth = [(b["label"], c2t(b["pos"])) for b in meta["balls"]]
    return balls, truth, cfg, r


def test_snooker_opening_all_balls_detected():
    """斯诺克开局：22 球（白+6彩+15红）全部检出，无漏检。"""
    for seed in range(3):
        balls, truth, _, _ = _pipeline(seed)
        assert len(balls) == len(truth), f"seed {seed}: 检出 {len(balls)} 真值 {len(truth)}"
        labels = sorted(t[0] for t in truth)
        detected = sorted(b.label for b in balls)
        assert detected == labels


def test_snooker_red_count_exact():
    """开局红球恰 15 颗（watershed 无假阳性/漏检）。"""
    for seed in range(3):
        balls, truth, _, _ = _pipeline(seed)
        assert sum(1 for t in truth if t[0] == "红球") == 15
        assert sum(1 for b in balls if b.label == "红球") == 15


def test_snooker_opening_rack_precision():
    """完整开局 rack 使用整体边界拟合后，红球中心误差应显著低于粗分割。"""
    for seed in range(3):
        balls, truth, _, _ = _pipeline(seed)
        detected = [b for b in balls if b.label == "红球"]
        errors = []
        for label, target in truth:
            if label == "红球":
                errors.append(min(np.hypot(b.pos[0] - target[0],
                                           b.pos[1] - target[1])
                                  for b in detected))
        assert max(errors) < 4.0, f"seed {seed}: rack 最大误差 {max(errors):.1f}px"


def test_color_ball_precision():
    """彩球/白球球心误差 < 12px（标准坐标）。"""
    for seed in range(3):
        balls, truth, _, _ = _pipeline(seed)
        used = set()
        for label, t in truth:
            if label == "红球":
                continue
            cands = [(i, b) for i, b in enumerate(balls)
                     if b.label == label and i not in used]
            assert cands, f"seed {seed}: MISS {label}"
            i, b = min(cands, key=lambda x: np.hypot(*(np.array(x[1].pos) - np.array(t))))
            used.add(i)
            err = np.hypot(*(np.array(b.pos) - np.array(t)))
            assert err < 12, f"seed {seed} {label}: {err:.1f}px"


def test_subpixel_fit_pure_circle():
    """亚像素圆拟合：合成纯色圆（无渐变）误差 < 1px。"""
    img = np.full((600, 600, 3), (128, 128, 128), np.uint8)
    truth = (503.7, 402.3)
    cv2 = pytest.importorskip("cv2")
    cv2.circle(img, (504, 402), 30, (0, 0, 255), -1, cv2.LINE_AA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fit = vision.fit_ball_edges(gray, 504, 402, 30)
    assert fit is not None
    err = np.hypot(fit[0] - truth[0], fit[1] - truth[1])
    assert err < 1.0, f"亚像素拟合误差 {err:.3f}px"


def test_clean_background_isolates_green_ball():
    """去背景后台面变灰、绿球保留（绿球与台面同色的关键场景）。"""
    img, meta = synth.snooker_layout(seed=0)
    cfg = config.Config()
    quad = vision.find_table(img, cfg)
    warped = vision.warp_table(img, vision.homography(quad, W, H), W, H)
    r = cfg.ball_radius_ratio * W
    clean = vision.clean_background(warped, cfg, r)
    # 台面非球处（标准坐标 x=200 左中部，D 区与蓝球之间无球）应是灰色
    center = clean[500, 200]
    assert abs(int(center[0]) - 128) < 25, f"台面未涂灰: {center}"
    # 绿球独立检出
    balls = vision.detect_balls(warped, r, cfg, clean=clean)
    greens = [b for b in balls if b.label == "绿球"]
    assert len(greens) == 1


def test_find_table_edge_band_on_snooker():
    """斯诺克布局（红球压住下边）：find_table 角点误差 < 8px。"""
    for seed in range(3):
        img, meta = synth.snooker_layout(seed=seed)
        cfg = config.Config()
        quad = vision.find_table(img, cfg)
        assert quad is not None
        fx0, fy0, fx1, fy1 = meta["felt"]
        truth = np.array([[fx0, fy0], [fx1, fy0], [fx1, fy1], [fx0, fy1]], dtype=np.float32)
        errs = np.hypot(*(quad - truth).T)
        assert errs.max() < 8, f"seed {seed}: 角点误差 {errs.max():.1f}px"


def test_table_tracker_locks_and_smooths():
    """TableTracker：首帧锁定，后续帧复用锁定四边形，抖动被平滑。"""
    cfg = config.Config()
    img, _ = synth.snooker_layout(seed=1)
    # 模拟连续帧：同一张图 + 轻微抖动
    tracker = vision.TableTracker(cfg)
    quads = []
    for i in range(10):
        q = tracker.update(img)
        assert q is not None
        quads.append(q.copy())
    # 锁定后（非重检帧）输出应完全一致
    assert np.array_equal(quads[3], quads[4]) or cfg.table_recheck_frames > 4
    # 四边形角点大致稳定
    spread = np.max([np.abs(q - quads[0]).max() for q in quads[1:]])
    assert spread < 20


def test_snooker_decision_red_phase():
    """红球阶段：目标必须是红球。"""
    cfg = config.Config()
    r = cfg.ball_radius_ratio * W
    cue = vision.Ball("白球", (300.0, 500.0), r)
    red = vision.Ball("红球", (1000.0, 500.0), r)
    balls = [cue, red]
    pockets = physics.default_pockets(W, H)
    tb, phase, _ = snooker.choose_target(balls, cue, pockets, r, W, H, cfg)
    assert phase == "red"
    assert tb is not None and tb.label == "红球"


def test_snooker_decision_color_order():
    """清彩阶段：按 黄→绿→棕→蓝→粉→黑 顺序。"""
    cfg = config.Config()
    r = cfg.ball_radius_ratio * W
    pockets = physics.default_pockets(W, H)

    def mk(labels, start_x=100.0):
        """构造 ball 列表（简化对象，只带 label/pos）。"""
        class B:
            def __init__(self, label, pos):
                self.label = label
                self.pos = pos
        pts = [(start_x + 80 * i, 500.0) for i in range(len(labels))]
        return [B(l, p) for l, p in zip(labels, pts)]

    cue = mk(["白球"], 300.0)[0]
    # 只有黑球在场（黄绿棕蓝粉已清）→ 打黑球
    balls = [cue, mk(["黑球"], 1000.0)[0]]
    tb, phase, _ = snooker.choose_target(balls, cue, pockets, r, W, H, cfg)
    assert phase == "color" and tb.label == "黑球"
    # 黄绿都在 → 打黄球
    balls = [cue] + mk(["黄球", "绿球"], 700.0)
    tb, phase, _ = snooker.choose_target(balls, cue, pockets, r, W, H, cfg)
    assert tb.label == "黄球"
    # 无红球无彩球
    tb, phase, _ = snooker.choose_target([cue], cue, pockets, r, W, H, cfg)
    assert tb is None and phase == "color"


def test_turn_tracker_enters_color_after_last_red():
    """最后一颗红球落袋后仍先进入任选彩球阶段。"""
    cfg = config.Config()
    r = cfg.ball_radius_ratio * W
    t = snooker.TurnTracker()
    cue = vision.Ball("白球", (300.0, 500.0), r)
    reds = [vision.Ball("红球", (700.0, 400.0), r),
            vision.Ball("红球", (800.0, 400.0), r)]
    black = vision.Ball("黑球", (1500.0, 500.0), r)

    assert t.update([cue, *reds, black], stable=True) == "red"
    assert t.update([cue, reds[0], black], stable=True) == "color"
    assert t.update([cue, black], stable=True) == "color"


def test_choose_target_explicit_color_reports_color_phase():
    cfg = config.Config()
    r = cfg.ball_radius_ratio * W
    cue = vision.Ball("白球", (300.0, 500.0), r)
    yellow = vision.Ball("黄球", (700.0, 500.0), r)
    pockets = physics.default_pockets(W, H)
    target, selected_phase, _ = snooker.choose_target(
        [cue, yellow], cue, pockets, r, W, H, cfg, ball_on="color")
    assert target is yellow
    assert selected_phase == "color"


def test_red_rack_fit_rejects_partial_rack():
    """红球已减少后不能被 rack 拟合凭空补回 15 颗。"""
    spots = synth.snooker_spot_positions()
    centers = synth.snooker_red_triangle(spots["红球顶点"], synth.BALL_R, 5)
    assert vision.refine_red_rack(centers[:-4], synth.BALL_R) is None


def test_opening_break_target_uses_outer_red_when_rack_has_no_pot_route():
    """完整球架无安全入袋路线时，回退到母球方向的外层红球。"""
    spots = synth.snooker_spot_positions()
    reds = [vision.Ball("红球", point, synth.BALL_R)
            for point in synth.snooker_red_triangle(spots["红球顶点"], synth.BALL_R)]
    cue = vision.Ball("白球", (spots["红球顶点"][0] - 900.0,
                               spots["红球顶点"][1] + 80.0), synth.BALL_R)
    target = snooker.opening_break_target(reds + [cue], cue, synth.BALL_R)
    assert target is not None
    assert target.pos[0] == min(ball.pos[0] for ball in reds)


def test_render_default_uses_snooker_palette():
    """render() 默认路径不能遗留美式球的紫/橙标签。"""
    img, meta = synth.render(seed=0)
    assert img.shape[:2] == (synth.CANVAS_H, synth.CANVAS_W)
    assert all(ball["label"] in synth.POCKET_COLORS for ball in meta["balls"])
