# -*- coding: utf-8 -*-
"""阶段 3 精度提升的回归测试：库边能量损耗、袋口宽度/摔袋风险评分、切线轨迹、坐标映射守卫。

几何约定：cue_tangent 的法线取「袋口→鬼球」方向（与实现一致），
碰撞方向 i = 母球→鬼球，切线 t = i - (i·n)n 与 |t| = sin(夹角)。
构造自洽几何时直接以单位向量反推母球位置，避免 cut_deg 标签与
实际几何不一致。
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from aimtool.physics import (Point, Shot, cue_tangent, default_pockets,
                             power_suggestion, route_score, scratch_risk)
from aimtool import vision

W, H = 900.0, 450.0
POCKETS = default_pockets(W, H)


def _shot(ghost: Point, pocket: Point, aim: Point, total: float = 350.0,
          cut_deg: float = 0.0, bounces: int = 0) -> Shot:
    return Shot(pocket=pocket, ghost=ghost, aim_dir=aim,
                cue_to_contact=200.0, target_to_pocket=150.0, total=total,
                cut_deg=cut_deg, valid=True,
                bounce_points=[(450.0, 0.0)] * bounces)


# ---------- 力度：库边能量损耗 ----------

def test_power_equivalent_to_stretching_total():
    """一库损耗 22% 等价于总路程 ×1.22（实现无关的精确断言）。"""
    assert power_suggestion(350.0, W, 2.2, 10.0, rails=1) == \
        power_suggestion(350.0 * 1.22, W, 2.2, 10.0)
    assert power_suggestion(350.0, W, 2.2, 10.0, rails=2) == \
        power_suggestion(350.0 * 1.44, W, 2.2, 10.0)


def test_power_monotonic_in_rails():
    base = power_suggestion(350.0, W, 2.2, 10.0)
    one = power_suggestion(350.0, W, 2.2, 10.0, rails=1)
    two = power_suggestion(350.0, W, 2.2, 10.0, rails=2)
    assert base <= one <= two <= 100.0


def test_power_cut_and_rails_combine_not_overwrite():
    """回归：大切角 + 多库时，库边补偿必须保留（曾被 cut 分支用 total 覆盖）。"""
    cut_only = power_suggestion(350.0, W, 2.2, 10.0, cut_deg=60.0)
    both = power_suggestion(350.0, W, 2.2, 10.0, cut_deg=60.0, rails=1)
    assert both > cut_only


def test_power_zero_rail_loss_matches_legacy():
    assert power_suggestion(350.0, W, 2.2, 10.0, rails=3, rail_loss=0.0) == \
        power_suggestion(350.0, W, 2.2, 10.0)


# ---------- 切线方向 ----------

def _tangent_setup(cut_sin: float) -> tuple:
    """构造 45° 切角的自洽几何：ghost (600,225)，袋口 (900,0)。

    n = normalize(袋口→鬼球) = -(0.8,-0.6)；i = cos·n' + sin·m'（n'=鬼球→袋口）。
    返回 (shot, 期望切线单位向量, 期望 tn)。
    """
    ghost, pocket = (600.0, 225.0), (900.0, 0.0)
    n = ((pocket[0] - ghost[0]), (pocket[1] - ghost[1]))          # 鬼球→袋口
    nn = math.hypot(*n)
    n = (n[0] / nn, n[1] / nn)                                    # (0.8, -0.6)
    m = (-n[1], n[0])                                             # 垂直方向 (0.6, 0.8)
    c, s = math.sqrt(1.0 - cut_sin ** 2), cut_sin
    aim = (c * n[0] + s * m[0], c * n[1] + s * m[1])
    return _shot(ghost, pocket, aim, cut_deg=math.degrees(math.asin(cut_sin))), \
        (s * m[0] + 0.0, s * m[1] + 0.0), s


def test_cue_tangent_perpendicular_to_normal():
    shot, _, _ = _tangent_setup(cut_sin=math.sin(math.radians(45.0)))
    tdir, tfrac = cue_tangent(shot)
    assert 0.0 < tfrac < 1.0
    # 法线（鬼球→袋口）与切线垂直
    nx, ny = shot.pocket[0] - shot.ghost[0], shot.pocket[1] - shot.ghost[1]
    n = math.hypot(nx, ny)
    assert abs(tdir[0] * (nx / n) + tdir[1] * (ny / n)) < 1e-9
    # 切向份额 = sin(实际夹角) = sin45
    assert abs(tfrac - math.sin(math.radians(45.0))) < 1e-9


def test_cue_tangent_degenerate_on_full_ball_hit():
    """碰撞方向与球心连线共线（正碰）→ 切线为 0。"""
    ghost, pocket = (600.0, 225.0), (900.0, 0.0)
    aim = ((pocket[0] - ghost[0]), (pocket[1] - ghost[1]))
    nn = math.hypot(*aim)
    shot = _shot(ghost, pocket, (aim[0] / nn, aim[1] / nn))
    tdir, tfrac = cue_tangent(shot)
    assert tfrac < 1e-9 and tdir == (0.0, 0.0)


# ---------- 摔袋风险 ----------

def test_scratch_risk_none_for_dead_center_shot():
    """近正碰（切向份额≈0）→ 无摔袋风险。"""
    ghost, pocket = (600.0, 225.0), (900.0, 0.0)
    aim = ((pocket[0] - ghost[0]), (pocket[1] - ghost[1]))
    nn = math.hypot(*aim)
    shot = _shot(ghost, pocket, (aim[0] / nn, aim[1] / nn))
    assert scratch_risk(shot, 10.0, 14.5, POCKETS, max(W, H)) == 0.0


def test_scratch_risk_high_when_tangent_aims_pocket():
    """45° 切角、切线正前方放一个袋口 → 高风险。"""
    shot, tdir, _ = _tangent_setup(cut_sin=math.sin(math.radians(45.0)))
    gx, gy = shot.ghost
    on_ray = (gx + 80.0 * tdir[0], gy + 80.0 * tdir[1])          # 切线正前方 80px
    risk = scratch_risk(shot, 10.0, 14.5, [on_ray], max(W, H))
    assert risk > 0.5


def test_scratch_risk_zero_when_tangent_away():
    """切线方向上没有袋口（袋口在侧后方）→ 风险为 0。"""
    shot, _, _ = _tangent_setup(cut_sin=math.sin(math.radians(45.0)))
    gx, gy = shot.ghost
    behind = (gx - 200.0 * 0.6, gy - 200.0 * 0.8)                # 切线反方向
    assert scratch_risk(shot, 10.0, 14.5, [behind], max(W, H)) == 0.0


def test_scratch_risk_ignores_far_pockets():
    shot, tdir, _ = _tangent_setup(cut_sin=math.sin(math.radians(45.0)))
    gx, gy = shot.ghost
    far = (gx + 5000.0 * tdir[0], gy + 5000.0 * tdir[1])         # 超出 1.2×span
    assert scratch_risk(shot, 10.0, 14.5, [far], max(W, H)) == 0.0


# ---------- 评分：袋口宽度 + 摔袋 ----------

def _mid_pocket_shots() -> tuple:
    """中袋 (450,0)：正对入射 vs 贴库斜抹，total 相同以隔离变量。"""
    straight = _shot(ghost=(450.0, 200.0), pocket=(450.0, 0.0), aim=(0.0, -1.0))
    shallow = _shot(ghost=(700.0, 80.0), pocket=(450.0, 0.0), aim=(-0.954, -0.305))
    return straight, shallow


def test_route_score_penalizes_narrow_pocket_angle():
    """回归：cos_in 必须以「台心→袋口」为开口方向（曾取反导致全员满罚）。"""
    straight, shallow = _mid_pocket_shots()
    kw = dict(table_height=H, pocket_radius=14.5, ball_radius=10.0, pockets=POCKETS)
    s_straight = route_score(straight, W, **kw)
    s_shallow = route_score(shallow, W, **kw)
    assert s_shallow > s_straight
    # 正对入射不应吃入射角惩罚（只含 total + 摔袋项，几何上无切线袋口）
    assert s_straight <= straight.total + 1e-6


def test_route_score_penalizes_scratch_risk():
    """同一条 Shot，切线正前方有袋口时评分必须更差。"""
    shot, tdir, _ = _tangent_setup(cut_sin=math.sin(math.radians(45.0)))
    gx, gy = shot.ghost
    on_ray = (gx + 80.0 * tdir[0], gy + 80.0 * tdir[1])
    kw = dict(table_height=H, pocket_radius=14.5, ball_radius=10.0)
    assert route_score(shot, W, pockets=[on_ray], **kw) > \
        route_score(shot, W, pockets=[(5000.0, 5000.0)], **kw)


def test_route_score_backward_compatible_without_geometry():
    """不传几何参数时与旧评分公式完全一致（回归保护）。"""
    shot = _shot(ghost=(600.0, 225.0), pocket=(900.0, 0.0), aim=(1.0, 0.0),
                 cut_deg=30.0)
    legacy = shot.total + 0.60 * W * (30.0 / 85.0) ** 2.4
    assert abs(route_score(shot, W) - legacy) < 1e-9


# ---------- 坐标映射守卫 ----------

def test_point_mapping_raises_on_degenerate_homography():
    bad = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    with pytest.raises(ValueError):
        vision.point_screen_to_table((100.0, 100.0), bad)
    with pytest.raises(ValueError):
        vision.point_table_to_screen((100.0, 100.0), bad)


def test_point_mapping_roundtrip_still_works():
    hm = np.array([[1.0, 0.0, 50.0], [0.0, 1.0, 20.0], [0.0, 0.0, 1.0]])
    pt = vision.point_screen_to_table((150.0, 70.0), hm)
    assert pt == (200.0, 90.0)                                    # 平移 +50/+20
    assert vision.point_table_to_screen(pt, np.linalg.inv(hm)) == (150.0, 70.0)
