# -*- coding: utf-8 -*-
"""白框歪斜/抖动修复（bug2）回归测试。

bug2 现象：左库被绿色污染拉斜 8px（skew=0.0154），恰好落在旧版
axis_like 阈值 0.015 与 edge_skew 拒绝阈值 0.02 之间——歪四边形逃过
正则化被 TableTracker 锁定固化，白框长期歪斜且帧间抖动。

修复：find_table 对所有通过歪斜检查的四边形一律做外接矩形正则化；
TableTracker 重检时发现锁定框歪、候选正（历史遗留）直接换新。
"""
from pathlib import Path

import cv2
import numpy as np
import pytest

from aimtool import config, vision

ASSETS = Path(__file__).resolve().parent / "assets"


def _rect_quad(x0, y0, x1, y1):
    return np.array(
        [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


def _assert_axis_aligned(quad, tol=1.0):
    """台面必然轴对齐：对边端点坐标差不超过 tol 像素。"""
    assert quad is not None
    assert abs(float(quad[1][1] - quad[0][1])) <= tol   # 上边水平
    assert abs(float(quad[2][1] - quad[3][1])) <= tol   # 下边水平
    assert abs(float(quad[3][0] - quad[0][0])) <= tol   # 左边垂直
    assert abs(float(quad[2][0] - quad[1][0])) <= tol   # 右边垂直


def _synth_frame_with_left_blob():
    """绿色台面 + 左侧相连的绿色污染块（复刻 bug2 场景）。"""
    frame = np.zeros((600, 1000, 3), dtype=np.uint8)
    cv2.rectangle(frame, (100, 100), (900, 500), (40, 160, 40), -1)
    # 左侧污染块：与台面连片，把左边界拉到 x=94 并污染边带拟合
    #（更宽的污染块会被 table_max_edge_skew 直接拒绝，那是另一条防线）
    cv2.rectangle(frame, (94, 220), (104, 460), (40, 160, 40), -1)
    return frame


def test_find_table_regularizes_skewed_left_edge():
    """左侧污染拉歪的台面输出必须被正则化为轴对齐矩形。"""
    cfg = config.Config()
    quad = vision.find_table(_synth_frame_with_left_blob(), cfg)
    _assert_axis_aligned(quad)


def test_find_table_clean_frame_still_axis_aligned():
    """无污染的普通帧同样输出轴对齐矩形（回归保护）。"""
    frame = np.zeros((600, 1000, 3), dtype=np.uint8)
    cv2.rectangle(frame, (100, 100), (900, 500), (40, 160, 40), -1)
    quad = vision.find_table(frame, config.Config())
    _assert_axis_aligned(quad)


@pytest.mark.skipif(
    not (ASSETS / "bug2.png").exists(), reason="需要 bug2.png 实拍样本")
def test_find_table_on_bug2_screenshot():
    """真实 bug2 截图：左库歪 8px 的历史场景必须输出正交四边形。"""
    img = cv2.imread(str(ASSETS / "bug2.png"))
    assert img is not None
    quad = vision.find_table(img, config.Config())
    _assert_axis_aligned(quad)


def test_table_tracker_replaces_legacy_skewed_quad(monkeypatch):
    """旧版锁定的歪四边形：重检候选是正框时直接替换，不等连续确认。"""
    cfg = config.Config()
    tracker = vision.TableTracker(cfg)
    legacy = np.array(
        [[28, 54], [1011, 52], [1046, 572], [36, 571]], dtype=np.float32)
    tracker.quad = legacy.copy()
    tracker.frame = cfg.table_recheck_frames - 1   # 下次 update 触发重检

    good = _rect_quad(30, 52, 1011, 572)
    monkeypatch.setattr(vision, "find_table", lambda frame, c: good.copy())
    out = tracker.update(np.zeros((648, 1152, 3), dtype=np.uint8))
    assert np.allclose(out, good)


def test_table_tracker_keeps_stable_lock(monkeypatch):
    """正框锁定后，2px 内的重检噪声保持旧框（不引入新抖动）。"""
    cfg = config.Config()
    tracker = vision.TableTracker(cfg)
    locked = _rect_quad(30, 52, 1011, 572)
    tracker.quad = locked.copy()
    tracker.frame = cfg.table_recheck_frames - 1

    drifted = _rect_quad(32, 52, 1011, 572)        # 2px 平移候选
    monkeypatch.setattr(vision, "find_table", lambda frame, c: drifted.copy())
    out = tracker.update(np.zeros((648, 1152, 3), dtype=np.uint8))
    assert np.allclose(out, locked)


def test_rank_target_shots_drop_blocked():
    """阻挡路线永不进入推荐（自动选球优先级第一环）。"""
    from aimtool import physics, snooker

    def shot(blocked):
        return physics.Shot(
            pocket=(0.0, 0.0), ghost=(10.0, 10.0), aim_dir=(1.0, 0.0),
            cue_to_contact=100.0, target_to_pocket=200.0, total=300.0,
            cut_deg=5.0, valid=True, blocked=blocked, label="直球")

    clear, blocked_shot = shot(False), shot(True)
    assert snooker.rank_target_shots([blocked_shot, clear]) == [clear]
    assert snooker.best_target_shot([blocked_shot]) is None
