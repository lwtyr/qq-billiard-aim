# -*- coding: utf-8 -*-
"""v3.10.2 回归：transient UI 掩膜不得吞掉真实彩球。

实机根因（user_report_blue.png 全链复现）：游戏自带瞄准轨迹/覆盖注释
与合法球成组后被判成界面，clean_background 把整颗球涂成中性灰 → 彩球
漏检 → 清彩严格顺序被"看不到的球"跳过。修复 = 形状校验过的彩球实心盘
从界面掩膜整体豁免。
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import synth
from aimtool import config, physics, vision

W, H = 2000, 1000


def _scene_with_annotations():
    layout = [
        ("白球", (300.0, 300.0)),
        ("蓝球", (900.0, 500.0)),
        ("粉球", (1040.0, 500.0)),
        ("黑球", (1700.0, 400.0)),
    ]
    img, _ = synth.render(layout, seed=20260902)
    # 游戏自带瞄准轨迹（虚线穿过蓝球、粉球）+ 连击计分文字贴着蓝球：
    # 成组后远大于一颗球，旧逻辑会把整颗球划成"界面"。
    for x in range(620, 1360, 46):
        cv2.line(img, (x, 497), (x + 30, 503), (235, 235, 235), 3,
                 cv2.LINE_AA)
    cv2.putText(img, "x9 COMBO", (790, 470), cv2.FONT_HERSHEY_SIMPLEX,
                1.6, (250, 250, 250), 4, cv2.LINE_AA)
    cfg = config.Config()
    quad = vision.find_table(img, cfg)
    aw = int(round(np.linalg.norm(quad[1] - quad[0])))
    aw = max(cfg.analysis_min_width, min(cfg.analysis_max_width, aw))
    ah = int(round(aw * H / W))
    warped = vision.warp_table(img, vision.homography(quad, aw, ah), aw, ah)
    r = cfg.ball_radius_ratio * aw
    scale = aw / W
    pts = {label: (xy[0] * scale, xy[1] * scale) for label, xy in layout}
    return warped, cfg, aw, ah, r, pts


def _mask_circle(mask, xy, rr):
    h, w = mask.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    return ((xx - xy[0]) ** 2 + (yy - xy[1]) ** 2) <= rr * rr


def test_annotation_overlay_cannot_eat_balls():
    """UI 注释成组覆盖球身时，彩球/白球仍全部检出，且球盘不被涂灰。"""
    warped, cfg, aw, ah, r, pts = _scene_with_annotations()
    ui = vision.transient_ui_mask(warped, cfg, r)
    assert ui.any(), "场景里确有界面注释，掩膜不应为空"
    for name in ("蓝球", "粉球"):
        disc = _mask_circle(ui, pts[name], 0.8 * r)
        assert not ui[disc].any(), f"{name} 球盘核心不许被界面掩膜覆盖"
    pockets = vision.refine_pockets(
        warped, physics.default_pockets(aw, ah), r)
    clean = vision.clean_background(warped, cfg, r, pockets, ui)
    balls = vision.detect_balls(warped, r, cfg, pockets, clean, ui)
    labels = sorted(b.label for b in balls)
    assert labels == sorted(["白球", "蓝球", "粉球", "黑球"]), labels
    # 合成台四角存在 warp 透视畸变（既有特性），位置容差放宽到 3r；
    # 球-label 多重集上面已精确断言，此处只防"识别到错误的另一颗球"。
    for b in balls:
        exp = pts[b.label]
        dist = (abs(b.pos[0] - exp[0]) ** 2 + abs(b.pos[1] - exp[1]) ** 2) ** 0.5
        assert dist < 3.0 * r, (b.label, tuple(b.pos), exp)


def test_ui_mask_still_covers_real_annotation_pixels():
    """真实 UI 像素仍须被屏蔽：球被豁免不能把注释放归检测。"""
    warped, cfg, aw, ah, r, pts = _scene_with_annotations()
    ui = vision.transient_ui_mask(warped, cfg, r)
    scale = (aw / W)
    # 连击文字的字形墨区（远离任何球的保护半径）必须仍在掩膜里。
    x0, x1 = int(796 * scale), int(826 * scale)
    y0, y1 = int(444 * scale), int(468 * scale)
    roi = ui[y0:y1, x0:x1]
    assert float((roi > 0).mean()) > 0.3


def test_verified_colored_mask_geometry():
    """辅助函数：实心彩球盘全豁免；黑球/袋洞状暗盘永不豁免。"""
    img = np.full((400, 400, 3), (70, 120, 60), np.uint8)
    cv2.circle(img, (120, 120), 16, (200, 120, 60), -1, cv2.LINE_AA)  # 蓝
    cv2.circle(img, (260, 260), 16, (20, 20, 20), -1, cv2.LINE_AA)    # 类袋洞暗盘
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    keep = vision._verified_colored_ball_mask(hsv, 14.0)
    assert keep[120, 120] > 0, "蓝球盘必须被豁免"
    disc = _mask_circle(keep, (120, 120), 0.9 * 14.0)
    assert float((keep[disc] > 0).mean()) > 0.9
    assert keep[260, 260] == 0, "黑球/袋洞状暗盘不得豁免"
    near = _mask_circle(keep, (260, 260), 1.2 * 14.0)
    assert not (keep[near] > 0).any()


def test_verified_colored_mask_rejects_glyphs():
    """星形/文字状的黄绿色杂点不得被当作球豁免。"""
    img = np.full((400, 400, 3), (70, 120, 60), np.uint8)
    cv2.putText(img, "88", (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                (60, 240, 240), 3, cv2.LINE_AA)  # 黄色文字
    cv2.drawMarker(img, (280, 260), (80, 230, 80), cv2.MARKER_STAR,
                   22, 2, cv2.LINE_AA)  # 绿色星标
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    keep = vision._verified_colored_ball_mask(hsv, 14.0)
    assert not keep.any(), "文字与星标不得通过实心盘形状门控"
