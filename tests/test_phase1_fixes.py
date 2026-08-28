"""阶段1修复回归测试：节流节流键、存证还原、rack 防复活、贴库球覆盖率。

对应修复：
- _warn_once 增加类别 key：消息正文含每帧变化数值时旧实现 key 恒新，
  节流失效（遮挡期间 IO 风暴）。
- blank_self_mask 返回自绘像素还原信息，_save_bad_frame 存盘前还原，
  保住异常帧诊断证据。
- refine_red_rack 增加逐点最远距离校验，防止先验网格"复活"已进球。
- circle_edge_coverage 越界 bin 不计入分母，贴库球不再被静默剔除。
- analyze 所有提前返回路径带 analysis_ms。
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import io
import contextlib
import numpy as np

import main
from aimtool import config, vision


# ---------- _warn_once 类别节流 ----------

def test_warn_once_throttles_by_category_key():
    main._warn_state.update({"key": None, "t": 0.0})
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        for i in range(20):  # 正文每帧变化（模拟遮挡 dict / 球数明细）
            main._warn_once(f"[遮挡] 检测到遮挡: {{'bbox': ({i},1,2,3)}}",
                            None, key="occlusion")
    lines = [l for l in buf.getvalue().splitlines() if l.startswith("[遮挡]")]
    assert len(lines) == 1


def test_warn_once_different_keys_both_print():
    main._warn_state.update({"key": None, "t": 0.0})
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main._warn_once("[识别异常] 球数=31", None, key="balls_too_many")
        main._warn_once("[识别异常] 无红球", None, key="no_red_balls")
    out = buf.getvalue()
    assert "球数=31" in out and "无红球" in out


def test_warn_once_backwards_compatible_without_key():
    main._warn_state.update({"key": None, "t": 0.0})
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main._warn_once("固定消息A")
        main._warn_once("固定消息A")
        main._warn_once("固定消息B")
    assert buf.getvalue().count("固定消息") == 2


# ---------- blank_self_mask 还原 + 存证 ----------

def _painted_frame():
    frame = np.full((60, 80, 3), (40, 120, 60), np.uint8)
    frame[20:28, 30:40] = (220, 220, 220)   # 模拟截屏带进的叠加层白线
    self_mask = np.zeros((60, 80), np.uint8)
    self_mask[20:28, 30:40] = 255
    return frame, self_mask


def test_blank_self_mask_returns_restore_info():
    frame, self_mask = _painted_frame()
    orig = frame.copy()
    restore = vision.blank_self_mask(frame, self_mask, config.Config())
    assert restore is not None
    ys, xs, pixels = restore
    assert ys.size == xs.size == pixels.shape[0] == 80
    assert not np.array_equal(frame, orig)           # 已涂台呢色
    assert np.array_equal(pixels, orig[ys, xs])      # 还原信息与原像素一致


def test_save_bad_frame_restores_self_mask_pixels():
    import cv2
    frame, self_mask = _painted_frame()
    orig = frame.copy()
    main._self_paint["restore"] = vision.blank_self_mask(
        frame, self_mask, config.Config())
    try:
        path = main._save_bad_frame(frame)
        assert path is not None
        saved = cv2.imread(path)
        assert bool((saved[20:28, 30:40] > 180).all())   # 白线在存档中可见
        assert not np.array_equal(frame, orig)            # 分析帧不被存证污染
        os.remove(path)
    finally:
        main._self_paint["restore"] = None


# ---------- refine_red_rack 防复活 ----------

def _rack_grid(r=20.0, cx=400.0, cy=300.0):
    return [(cx + (j - i / 2.0) * 2 * r, cy + i * (3 ** 0.5) * r)
            for i in range(5) for j in range(i + 1)]


def test_refine_red_rack_rejects_revival_with_far_candidate():
    r = 20.0
    grid = _rack_grid(r)
    # 14 颗真球 + 1 颗远处多余候选补位：平均距离能过，最远点超 1.4r
    cands = grid[:-1] + [(grid[-1][0] + 6 * r, grid[-1][1])]
    assert vision.refine_red_rack(cands, r) is None


def test_refine_red_rack_accepts_full_rack_with_jitter():
    r = 20.0
    cands = [(x + 3, y + 2) for x, y in _rack_grid(r)]
    grid = vision.refine_red_rack(cands, r)
    assert grid is not None and len(grid) == 15


def test_refine_red_rack_rejects_fourteen_candidates():
    assert vision.refine_red_rack(_rack_grid()[:-4], 20.0) is None


# ---------- circle_edge_coverage 贴库球 ----------

def _rail_image(gap, r=20.0):
    """球心距左边缘 gap*r 的合成帧：白盘+暗环+边缘 6px 灰带（warp 边界）。"""
    img = np.full((200, 300), 100, np.uint8)
    yy, xx = np.mgrid[:200, :300]
    d2 = (xx - gap * r) ** 2 + (yy - 100) ** 2
    img[d2 <= r * r] = 210
    img[(d2 > r * r) & (d2 <= 1.15 * r * r)] = 150
    img[:, :6] = 128
    return img


def test_circle_edge_coverage_rail_ball_passes():
    for gap in (1.00, 1.05, 1.10):
        img = _rail_image(gap)
        cov = vision.circle_edge_coverage(img, (gap * 20.0, 100.0), 20.0)
        assert cov >= 0.42, f"贴库球覆盖率 {cov:.3f} 低于门限"


def test_circle_edge_coverage_mid_ball_unchanged():
    img = _rail_image(7.5)   # 远离边缘，无越界 bin
    cov = vision.circle_edge_coverage(img, (7.5 * 20.0, 100.0), 20.0)
    assert cov >= 0.95


def test_circle_edge_coverage_too_short_arc_rejected():
    # 小画幅+大半径：所有采样环必然越界 → 无有效 bin → 按不合格处理。
    # （现实几何里单边库边最多挡任约 1/3 圆周，几乎不可能触发下限，
    # 这里用极端参数直接验证防护分支本身。）
    img = np.full((40, 40), 100, np.uint8)
    assert vision.circle_edge_coverage(img, (20.0, 20.0), 30.0) == 0.0
    assert vision.circle_edge_coverage(img, (20.0, 20.0), 2.0) == 0.0  # r<3 防抖


# ---------- analyze 提前返回路径的 analysis_ms ----------

def test_analyze_early_returns_have_analysis_ms():
    cfg = config.Config()
    scene = main.analyze(np.full((200, 300, 3), 128, np.uint8), cfg)
    assert scene.get("status") == "未检测到台面"
    assert isinstance(scene.get("analysis_ms"), float)
    assert scene["analysis_ms"] >= 0.0
