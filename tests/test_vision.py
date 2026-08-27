"""视觉识别测试：合成台面 → 台面/球/袋口识别。"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import pytest

import synth
from aimtool import config, physics, vision

W, H = 2000.0, 1000.0


def test_find_table_on_synth():
    img, meta = synth.random_layout(seed=1)
    cfg = config.Config()
    quad = vision.find_table(img, cfg)
    assert quad is not None
    assert quad.shape == (4, 2)
    # 台面四边形应近似覆盖台呢区域
    fx0, fy0, fx1, fy1 = meta["felt"]
    tl, br = quad[0], quad[2]
    assert abs(tl[0] - fx0) < 40 and abs(tl[1] - fy0) < 40
    assert abs(br[0] - fx1) < 40 and abs(br[1] - fy1) < 40


def test_detect_all_balls():
    img, meta = synth.random_layout(seed=2)
    cfg = config.Config()
    quad = vision.find_table(img, cfg)
    Hm = vision.homography(quad, W, H)
    warped = vision.warp_table(img, Hm, W, H)
    r = cfg.ball_radius_ratio * W
    balls = vision.detect_balls(warped, r)

    fx0, fy0, fx1, fy1 = meta["felt"]
    def canvas_to_table(p):
        return ((p[0] - fx0) * W / (fx1 - fx0), (p[1] - fy0) * H / (fy1 - fy0))

    truth = {b["label"]: canvas_to_table(b["pos"]) for b in meta["balls"]}
    assert len(balls) >= len(truth) - 1, f"检出 {len(balls)} 球，真值 {len(truth)} 球"
    found = 0
    for label, tp in truth.items():
        hit = [b for b in balls if b.label == label
               and np.hypot(b.pos[0] - tp[0], b.pos[1] - tp[1]) < 15]
        if hit:
            found += 1
    assert found >= len(truth) - 1, f"正确检出 {found}/{len(truth)}"


def test_cue_is_white():
    img, meta = synth.random_layout(seed=3)
    cfg = config.Config()
    quad = vision.find_table(img, cfg)
    Hm = vision.homography(quad, W, H)
    warped = vision.warp_table(img, Hm, W, H)
    balls = vision.detect_balls(warped, cfg.ball_radius_ratio * W)
    cue = vision.pick_cue(balls)
    assert cue is not None and cue.label == "白球"


def test_refine_pockets():
    img, meta = synth.random_layout(seed=4)
    cfg = config.Config()
    quad = vision.find_table(img, cfg)
    Hm = vision.homography(quad, W, H)
    warped = vision.warp_table(img, Hm, W, H)
    r = cfg.ball_radius_ratio * W
    expected = physics.default_pockets(W, H)
    refined = vision.refine_pockets(warped, expected, r)
    assert len(refined) == 6
    # 袋口应在台面图边缘附近
    for (px, py) in refined:
        assert 0 <= px <= W and 0 <= py <= H
    # 边界方向不能被裁切后的暗区质心拉进台面。
    for (got, exp) in zip(refined, expected):
        if exp[0] in (0.0, W):
            assert got[0] == exp[0]
        if exp[1] in (0.0, H):
            assert got[1] == exp[1]
    # 回归：精修结果应仍贴近合成图中的真实袋口，不能被附近黑球/库边
    # 的暗色连通域拖走。角袋的真实中心在台面边缘内缩约一个球半径。
    fx0, fy0, fx1, fy1 = meta["felt"]
    truth = [((px - fx0) * W / (fx1 - fx0),
              (py - fy0) * H / (fy1 - fy0))
             for px, py in meta["pockets"]]
    errors = [min(np.hypot(rx - tx, ry - ty) for rx, ry in refined)
              for tx, ty in truth]
    assert max(errors) < 1.25 * r, f"袋口精修偏移过大: {max(errors):.1f}px"


def test_refine_pockets_finds_inset_dark_centers():
    """真实风格的袋洞向台内偏移时，不能回退到绿色边界角点。"""
    width, height = 800, 400
    r = 9.0
    image = np.full((height, width, 3), synth.FELT_BGR, dtype=np.uint8)
    expected = physics.default_pockets(width, height)
    offsets = [(18.0, 14.0), (-18.0, 14.0),
               (18.0, -14.0), (-18.0, -14.0),
               (0.0, 11.0), (0.0, -11.0)]
    for (px, py), (ox, oy) in zip(expected, offsets):
        cv2.circle(image, (round(px + ox), round(py + oy)),
                   round(1.25 * r), (8, 8, 8), -1, cv2.LINE_AA)

    refined = vision.refine_pockets(image, expected, r)
    truth = [(px + ox, py + oy)
             for (px, py), (ox, oy) in zip(expected, offsets)]
    errors = [np.hypot(got[0] - want[0], got[1] - want[1])
              for got, want in zip(refined, truth)]
    assert max(errors) < 3.0, f"内缩袋口中心偏移过大: {max(errors):.2f}px"


def test_point_mapping_roundtrip():
    img, meta = synth.random_layout(seed=5)
    cfg = config.Config()
    quad = vision.find_table(img, cfg)
    Hm = vision.homography(quad, W, H)
    Hinv = np.linalg.inv(Hm)
    pt = (1234.5, 678.9)
    scr = vision.point_table_to_screen(pt, Hinv)
    back = vision.point_screen_to_table(scr, Hm)
    assert np.hypot(back[0] - pt[0], back[1] - pt[1]) < 1.0


def test_detect_table_occlusion_rejects_large_ui_panel():
    """台面上的大块设置/提示面板不能进入球检测管线。"""
    img, _ = synth.random_layout(seed=9)
    cfg = config.Config()
    cv2.rectangle(img, (760, 390), (1240, 610), (235, 235, 235), -1)
    quad = vision.find_table(img, cfg)
    assert quad is not None
    warped = vision.warp_table(img, vision.homography(quad, W, H), W, H)
    r = cfg.ball_radius_ratio * W
    assert vision.detect_table_occlusion(warped, cfg, r) is not None

    clean_img, _ = synth.random_layout(seed=9)
    clean_quad = vision.find_table(clean_img, cfg)
    clean_warped = vision.warp_table(
        clean_img, vision.homography(clean_quad, W, H), W, H
    )
    assert vision.detect_table_occlusion(clean_warped, cfg, r) is None


def test_scattered_ball_mask_centers_are_stable():
    """颜色外轮廓拟合不能被球面高光拉偏。"""
    img, meta = synth.random_layout(seed=0)
    cfg = config.Config()
    quad = vision.find_table(img, cfg)
    Hm = vision.homography(quad, W, H)
    warped = vision.warp_table(img, Hm, W, H)
    r = cfg.ball_radius_ratio * W
    balls = vision.detect_balls(warped, r, cfg)
    fx0, fy0, fx1, fy1 = meta["felt"]
    errors = []
    for truth_ball in meta["balls"]:
        if truth_ball["label"] == "红球":
            continue
        truth = ((truth_ball["pos"][0] - fx0) * W / (fx1 - fx0),
                 (truth_ball["pos"][1] - fy0) * H / (fy1 - fy0))
        found = min((b for b in balls if b.label == truth_ball["label"]),
                    key=lambda b: np.hypot(b.pos[0] - truth[0], b.pos[1] - truth[1]))
        errors.append(np.hypot(found.pos[0] - truth[0], found.pos[1] - truth[1]))
    assert float(np.mean(errors)) < 2.5


def test_estimate_ball_radius_uses_current_frame_scale():
    """当前帧球径估计应能纠正合成台面缩放带来的固定半径偏差。"""
    img, _ = synth.random_layout(seed=0)
    cfg = config.Config()
    quad = vision.find_table(img, cfg)
    warped = vision.warp_table(img, vision.homography(quad, W, H), W, H)
    fallback = cfg.ball_radius_ratio * W
    balls = vision.detect_balls(warped, fallback, cfg)
    estimated = vision.estimate_ball_radius(balls, fallback)
    assert estimated > fallback + 0.5
    assert estimated < 1.2 * fallback


def test_table_tracker_rejects_single_bad_recheck(monkeypatch):
    """一次斜边误检不能把锁定框拉走，连续确认后才接受窗口移动。"""
    cfg = config.Config()
    cfg.table_recheck_frames = 1
    cfg.table_move_confirmations = 3
    base = np.array([[10, 10], [990, 10], [990, 500], [10, 500]], dtype=np.float32)
    bad = np.array([[10, 10], [990, 10], [1030, 500], [10, 500]], dtype=np.float32)
    values = iter((base, bad, base))
    monkeypatch.setattr(vision, "find_table", lambda _frame, _cfg: next(values))
    tracker = vision.TableTracker(cfg)
    frame = np.zeros((500, 1000, 3), dtype=np.uint8)

    first = tracker.update(frame).copy()
    second = tracker.update(frame).copy()
    third = tracker.update(frame).copy()
    assert np.array_equal(first, second)
    assert np.array_equal(first, third)


def test_table_tracker_does_not_accumulate_subthreshold_noise(monkeypatch):
    """静止球桌的连续小幅边缘噪声不能把锁定框逐步推走。"""
    cfg = config.Config()
    cfg.table_recheck_frames = 1
    cfg.table_recheck_max_shift = 7.0
    cfg.table_move_confirmations = 3
    base = np.array([[10, 10], [990, 10], [990, 500], [10, 500]], dtype=np.float32)
    noisy = np.array([[13, 12], [993, 12], [993, 503], [13, 503]], dtype=np.float32)
    values = iter((base, noisy, noisy, noisy, noisy))
    monkeypatch.setattr(vision, "find_table", lambda _frame, _cfg: next(values))
    tracker = vision.TableTracker(cfg)
    frame = np.zeros((500, 1000, 3), dtype=np.uint8)

    first = tracker.update(frame).copy()
    outputs = [tracker.update(frame).copy() for _ in range(4)]

    assert all(np.array_equal(first, got) for got in outputs)


def test_pocket_tracker_holds_single_candidate_jump():
    """单帧袋口候选跳点不应改变瞄准袋口。"""
    cfg = config.Config()
    tracker = vision.PocketTracker(cfg)
    base = [(0.0, 0.0), (2000.0, 0.0), (0.0, 1000.0),
            (2000.0, 1000.0), (1000.0, 0.0), (1000.0, 1000.0)]
    jump = list(base)
    jump[1] = (1960.0, 60.0)

    assert tracker.update(base)[1] == base[1]
    assert tracker.update(jump)[1] == base[1]
    assert tracker.update(jump)[1] == base[1]
    assert tracker.update(jump)[1] == jump[1]


def test_end_to_end_plan():
    """合成台面 → 识别 → 出瞄准方案（直球或解围）→ 力度合理。"""
    img, meta = synth.random_layout(seed=6)
    from main import analyze
    cfg = config.Config()
    scene = analyze(img, cfg)
    assert scene.get("status")
    # 至少识别出台面与球
    assert scene.get("table_quad") is not None
    assert scene.get("balls")
    assert scene.get("ghost") is not None
    assert scene.get("contact") is not None


def test_combo_text_is_masked_without_losing_cue_ball():
    """连击文字和球杆不能分别造成黑球假阳性或白球漏检。"""
    layout = [
        ("白球", (300.0, 300.0)),
        ("黑球", (1600.0, 300.0)),
        ("红球", (1000.0, 700.0)),
        ("黄球", (500.0, 700.0)),
        ("蓝球", (1500.0, 700.0)),
    ]
    img, _ = synth.render(layout, seed=123)
    # The cue touches the white ball at its edge but has a different hue.
    cv2.line(img, (320, 300), (450, 800), (150, 120, 100), 8, cv2.LINE_AA)
    cv2.putText(img, "2 COMBO", (820, 510), cv2.FONT_HERSHEY_SIMPLEX,
                2.0, (20, 20, 20), 7, cv2.LINE_AA)
    cv2.putText(img, "2 COMBO", (820, 510), cv2.FONT_HERSHEY_SIMPLEX,
                2.0, (245, 245, 245), 2, cv2.LINE_AA)

    cfg = config.Config()
    quad = vision.find_table(img, cfg)
    aw = int(round(np.linalg.norm(quad[1] - quad[0])))
    aw = max(cfg.analysis_min_width, min(cfg.analysis_max_width, aw))
    ah = int(round(aw * H / W))
    warped = vision.warp_table(img, vision.homography(quad, aw, ah), aw, ah)
    r = cfg.ball_radius_ratio * aw
    pockets = vision.refine_pockets(warped, physics.default_pockets(aw, ah), r)
    ui = vision.transient_ui_mask(warped, cfg, r)
    clean = vision.clean_background(warped, cfg, r, pockets, ui)
    balls = vision.detect_balls(warped, r, cfg, pockets, clean, ui)

    assert ui.any()
    assert sorted(b.label for b in balls) == sorted(label for label, _ in layout)
    assert sum(b.label == "黑球" for b in balls) == 1
    assert sum(b.label == "白球" for b in balls) == 1


def test_hough_is_skipped_when_color_masks_have_enough_candidates(monkeypatch):
    """Normal complete color detections should stay on the fast path."""
    img, _ = synth.random_layout(seed=13)
    cfg = config.Config()
    quad = vision.find_table(img, cfg)
    warped = vision.warp_table(img, vision.homography(quad, W, H), W, H)

    def unexpected_hough(*args, **kwargs):
        raise AssertionError("Hough should be a low-candidate fallback")

    monkeypatch.setattr(cv2, "HoughCircles", unexpected_hough)
    balls = vision.detect_balls(warped, cfg.ball_radius_ratio * W, cfg)
    assert len(balls) >= 8


def test_analysis_size_does_not_upscale_small_source_table():
    """分析图宽度应受当前台面像素限制，不固定放大到标准 2000px。"""
    img, _ = synth.random_layout(seed=11)
    cfg = config.Config(analysis_scale=1.0, analysis_max_width=1280)
    from main import analyze
    scene = analyze(img, cfg)
    source_width = float(synth.FELT_X1 - synth.FELT_X0)
    analysis_w, analysis_h = scene["analysis_size"]
    assert analysis_w <= source_width
    assert analysis_w < cfg.table_w
    assert analysis_h == int(round(analysis_w * cfg.table_h / cfg.table_w))
