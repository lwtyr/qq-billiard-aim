"""常驻 UI 学习与遮挡误报修复的回归测试（bug6）。

bug6：QQ 2D 瞄准时球杆一直搭在台面上，细长斜跨（fill≈0.19-0.24）
绕过旧三道判定（bbox 短边/fill<0.18/暗色），被当成弹窗遮挡 →
大多数帧早退 → 用户看到「偶尔瞄一下，大多数没辅助线」。

修复两层：
1. detect_table_occlusion 加最小外接旋转矩形判定（细长条放行），
   并在返回 dict 里带命中块 "mask"；
2. main 侧同位置连续命中 >=3 帧判定为游戏常驻 UI（力度条/按钮区），
   学习为 static_mask 后放行；偶发弹窗维持暂停保护。
"""
import numpy as np
import cv2

from aimtool import vision as vision_mod
from aimtool import config as config_mod
from main import _bbox_iou, _occ_streak_hit


def _felt_frame(w=1400, h=800):
    img = np.full((h, w, 3), (80, 180, 80), np.uint8)   # BGR 亮绿台呢
    return img


def test_skewed_cue_stick_not_occlusion():
    """斜跨球杆（细长棕条）不再被误判为弹窗遮挡。"""
    img = _felt_frame()
    cv2.line(img, (60, 740), (1340, 90), (96, 64, 0), 8)   # 棕色杆身斜跨
    occ = vision_mod.detect_table_occlusion(img, _cfg(), 20.0)
    assert occ is None, f"斜杆被误判遮挡: {occ}"


def _cfg():
    return config_mod.Config()


def test_solid_panel_still_occlusion():
    """真弹窗（实心白面板）仍判遮挡，且返回块 mask。"""
    img = _felt_frame()
    cv2.rectangle(img, (742, 372), (1259, 629), (250, 250, 250), -1)
    occ = vision_mod.detect_table_occlusion(img, _cfg(), 20.0)
    assert occ is not None
    bx, by, bw, bh = occ["bbox"]
    # 命中块覆盖面板区域（贴边圆角允许略大）
    assert bx <= 742 and by <= 372 and bx + bw >= 1259 and by + bh >= 629
    mask = occ["mask"]
    assert mask.shape == img.shape[:2]
    # mask 至少覆盖面板中心
    assert mask[500, 1000] == 255


def test_static_mask_replays_clear():
    """学习到的块 mask 作为 static_mask 回放：同一帧不再判遮挡。"""
    img = _felt_frame()
    cv2.rectangle(img, (742, 372), (1259, 629), (250, 250, 250), -1)
    occ = vision_mod.detect_table_occlusion(img, _cfg(), 20.0)
    assert occ is not None
    occ2 = vision_mod.detect_table_occlusion(
        img, _cfg(), 20.0, static_mask=occ["mask"])
    assert occ2 is None, "学习后仍误报"


def test_bbox_iou():
    assert _bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert _bbox_iou((0, 0, 10, 10), (20, 20, 10, 10)) == 0.0
    # 半重叠：交 5x10=50，并 100+100-50=150
    assert abs(_bbox_iou((0, 0, 10, 10), (5, 0, 10, 10)) - 50.0 / 150.0) < 1e-6


def test_occ_streak_same_pos_accumulates_and_resets():
    state = {"bbox": None, "n": 0, "static": None}
    assert _occ_streak_hit(state, (100, 100, 50, 30)) == 1
    assert _occ_streak_hit(state, (101, 100, 50, 30)) == 2   # IoU>0.6 连续
    assert _occ_streak_hit(state, (102, 101, 50, 30)) == 3
    # 换位置（不重叠）→ 重置为 1
    assert _occ_streak_hit(state, (900, 700, 50, 30)) == 1
    assert state["bbox"] == (900, 700, 50, 30)


def test_dark_shade_still_skipped():
    """纯暗大块（球群阴影）回归：不判遮挡。"""
    img = _felt_frame()
    cv2.rectangle(img, (400, 300), (900, 600), (10, 20, 10), -1)
    occ = vision_mod.detect_table_occlusion(img, _cfg(), 20.0)
    assert occ is None
