"""阶段3精度优化回归测试。

- 袋口挂球：中袋正下方 / 角袋斜口的红球必须检出且误差 <1.5px
  （此前袋口涂灰圆 1.35r 会把球体像素一起涂掉：≤1.1r 直接漏检、
  1.3r 质心下拉 4px+）；
- 黑球挂袋：靠亮高光核心区分袋洞与黑球，必须检出；
- 空袋口：洞必须仍被涂灰，不产生假黑球。
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np

import synth
from synth import POCKET_COLORS
from aimtool import config, vision

CFG = config.Config()
AW = 960.0


def _warped_empty_felt(seed=42):
    """合成台面（带标准袋口画在 felt 边界），返回 warped 图 + 球半径。"""
    img, meta = synth.random_layout(seed=seed)
    fx0, fy0, fx1, fy1 = meta["felt"]
    rr = float(meta["ball_r"])
    felt_col = tuple(int(c) for c in img[fy0 + 40, (fx0 + fx1) // 2])
    img[fy0:fy1, fx0:fx1] = felt_col
    pr = int(round(1.15 * rr))
    for (px, py) in [(fx0, fy0), (fx1, fy0), (fx0, fy1), (fx1, fy1),
                     ((fx0 + fx1) / 2, fy0), ((fx0 + fx1) / 2, fy1)]:
        cv2.circle(img, (int(px), int(py)), pr, (12, 12, 14), -1, cv2.LINE_AA)
    crop = img[fy0:fy1, fx0:fx1]
    scale = AW / crop.shape[1]
    warped = cv2.resize(crop, None, fx=scale, fy=scale,
                        interpolation=cv2.INTER_AREA)
    return warped, rr * scale, (fx0, fy0, fx1, fy1), scale


def _place_and_detect(color, dist_r, x_frac=0.5):
    warped, wr, (fx0, fy0, fx1, fy1), scale = _warped_empty_felt()
    cv2.circle(warped, (int(warped.shape[1] / 2), 0),
               int(1.15 * wr), (12, 12, 14), -1, cv2.LINE_AA)  # 顶中袋洞
    cx = (fx1 - fx0) * x_frac * scale
    cy = dist_r * wr
    synth.draw_ball(warped, (cx, cy), wr, POCKET_COLORS[color])
    found = vision.detect_balls(warped, wr, CFG)
    truth = (cx, cy)
    hits = [b for b in found if b.label == color]
    return truth, hits


# ---------- 中袋正下方挂球 ----------

def test_red_ball_hanging_under_middle_pocket_all_distances():
    # 修复前：1.0r/1.1r 漏检，1.3r 偏 4.28px
    for dist_r, tol in ((1.0, 1.5), (1.1, 1.5), (1.3, 1.5), (1.6, 1.0)):
        truth, hits = _place_and_detect("红球", dist_r)
        assert hits, f"{dist_r}r 中袋挂红球漏检"
        bx, by = hits[0].pos
        err = ((bx - truth[0]) ** 2 + (by - truth[1]) ** 2) ** 0.5
        assert err < tol, f"{dist_r}r 中袋挂红球误差 {err:.2f}px >= {tol}px"


def test_black_ball_hanging_under_middle_pocket_detected():
    # 黑球与袋洞同色：靠亮高光核心保护后必须能检出。但「洞+球」在黑掩膜里
    # 连成一个连通域，分水岭分割后质心仍略偏（球物理上也被洞遮一部分），
    # 容差 5px；红/彩色球（常规局面）不受此影响（<1.5px）。
    for dist_r in (1.1, 1.3, 1.6):
        truth, hits = _place_and_detect("黑球", dist_r)
        assert hits, f"{dist_r}r 中袋挂黑球漏检"
        bx, by = hits[0].pos
        err = ((bx - truth[0]) ** 2 + (by - truth[1]) ** 2) ** 0.5
        assert err < 5.0, f"{dist_r}r 黑球误差 {err:.2f}px"


def test_ball_in_corner_jaw_detected():
    # 角袋洞在 (0,0)，球挂在斜口方向（对角线上）
    for dist_r in (1.3, 1.6):
        warped2, wr2, _, _ = _warped_empty_felt()
        diag = dist_r * wr2 * 0.7071
        synth.draw_ball(warped2, (diag, diag), wr2, POCKET_COLORS["红球"])
        found = vision.detect_balls(warped2, wr2, CFG)
        hits = [b for b in found if b.label == "红球"]
        assert hits, f"{dist_r}r 角袋挂球漏检"
        bx, by = hits[0].pos
        err = ((bx - diag) ** 2 + (by - diag) ** 2) ** 0.5
        assert err < 1.5, f"{dist_r}r 角袋误差 {err:.2f}px"


# ---------- 空袋口不产生假黑球 ----------

def test_empty_pockets_produce_no_fake_black_ball():
    warped, wr, _, _ = _warped_empty_felt(seed=3)
    found = vision.detect_balls(warped, wr, CFG)
    blacks = [b for b in found if b.label == "黑球"]
    assert not blacks, f"空袋口产生假黑球: {blacks}"


def test_clean_background_still_greys_the_hole():
    from aimtool.physics import default_pockets
    warped, wr, _, _ = _warped_empty_felt()
    clean = vision.clean_background(warped, CFG, wr,
                                    pockets=default_pockets(warped.shape[1],
                                                            warped.shape[0]))
    # 顶中袋洞中心像素应被涂成中性灰 (128,128,128)
    cx = int(warped.shape[1] / 2)
    px = clean[2, cx]
    assert all(abs(int(c) - 128) <= 30 for c in px), \
        f"袋洞未被涂灰: {tuple(int(c) for c in px)}"
