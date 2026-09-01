# -*- coding: utf-8 -*-
"""v3.10 瞄准优化回归：袋口容错让点 + 进球成功率排序。

语义锁定：
  * Shot.pocket 永远是袋口中心（归类/索引/高亮用）；Shot.aim_point 是
    容错最优瞄准点（出球方向终点），未启用让点时为 None；
  * 正对袋口时让点退化为中心（与旧几何一致）；
  * 斜切时让点使出球方向到两侧袋角的角余量严格变大；
  * 入射角硬过滤改为按真实出球方向（ghost→aim_point）计算，但袋口
    归类仍按袋口中心——让点永远不会绕过过滤；
  * 成功率排序：近袋大切可以胜远袋小切；薄切/库边单调惩罚。
"""
import math

import pytest

from aimtool import config, physics, snooker

W, H = 2000.0, 1000.0
R = 22.5


def _mk_shot(pocket, ghost=(10.0, 10.0), c2c=600.0, ttp=500.0, cut=5.0,
             rails=(), aim_point=None, blocked=False):
    return physics.Shot(
        pocket=pocket, ghost=ghost, aim_dir=(1.0, 0.0),
        cue_to_contact=c2c, target_to_pocket=ttp, total=c2c + ttp,
        cut_deg=cut, valid=True, blocked=blocked,
        bounce_points=list((float(i), float(i)) for i in range(len(rails))),
        rail_seq=tuple(rails), label="直球" if not rails else f"{len(rails)}库",
        aim_point=aim_point,
    )


def test_aim_point_stays_center_for_straight_approach():
    """正对袋口：让点退化为袋口中心/鬼球不变（与旧几何一致）。"""
    cue = (1000.0, 720.0)
    target = (1000.0, 470.0)
    pocket = (1000.0, 0.0)
    plain = physics.direct_shot(cue, target, pocket, R)
    opt = physics.direct_shot(cue, target, pocket, R,
                              aim_half_width=1.45 * R, table_size=(W, H))
    assert plain.valid and opt.valid
    assert opt.aim_point is not None
    assert abs(opt.aim_point[0] - pocket[0]) < 0.5
    assert abs(opt.aim_point[1] - pocket[1]) < 0.5
    assert abs(opt.ghost[0] - plain.ghost[0]) < 0.5
    assert abs(opt.ghost[1] - plain.ghost[1]) < 0.5
    assert abs(opt.cut_deg - plain.cut_deg) < 0.05


def _jaw_margin_deg(target, travel_dir, pocket, half_width, u):
    """出球方向到开口线两端点的最小夹角（度）：越大越扛瞄准误差。"""
    pa = (pocket[0] - half_width * u[0], pocket[1] - half_width * u[1])
    pb = (pocket[0] + half_width * u[0], pocket[1] + half_width * u[1])

    def ang(p):
        d = physics.normalize(physics.sub(p, target))
        return math.degrees(math.acos(physics.clamp(
            physics.dot(d, travel_dir), -1.0, 1.0)))

    return min(ang(pa), ang(pb))


def test_aim_point_shifts_to_bisector_for_angled_mid_pocket():
    """斜着进中袋：瞄准点自动让位，出球方向到两袋角的角余量变大。"""
    cue = (330.0, 640.0)
    target = (400.0, 415.0)
    pocket = (1000.0, 0.0)          # 上中袋，开口法线 (0,-1)
    plain = physics.direct_shot(cue, target, pocket, R)
    opt = physics.direct_shot(cue, target, pocket, R,
                              aim_half_width=1.45 * R, table_size=(W, H))
    assert plain.valid and opt.valid and opt.aim_point is not None
    assert opt.aim_point != pocket
    # 目标球在开口中点左侧、浅角度接近：瞄准点应向左侧让位（x 变小），
    # 且让位幅度必然小于开口半宽
    assert pocket[0] - 1.45 * R < opt.aim_point[0] < pocket[0]

    a = 1.45 * R
    u = (1.0, 0.0)
    d_plain = physics.normalize(physics.sub(pocket, target))
    d_opt = physics.normalize(physics.sub(opt.aim_point, target))
    m_plain = _jaw_margin_deg(target, d_plain, pocket, a, u)
    m_opt = _jaw_margin_deg(target, d_opt, pocket, a, u)
    assert m_opt > m_plain


def test_aim_point_bisects_corner_pocket_mouth():
    """角袋斜进：让点后到两袋角的角余量趋于相等（平分线性质）。"""
    target = (400.0, 415.0)
    pocket = (2000.0, 0.0)          # 右上角袋
    aim = physics.pocket_aim_point(target, pocket, W, H, 1.45 * R, R)
    assert aim != pocket
    frame = physics._pocket_frame(pocket, W, H, R)
    n, u, mid = frame
    assert not mid
    a = 1.45 * R
    pa = (pocket[0] - a * u[0], pocket[1] - a * u[1])
    pb = (pocket[0] + a * u[0], pocket[1] + a * u[1])
    d_opt = physics.normalize(physics.sub(aim, target))
    da = physics.normalize(physics.sub(pa, target))
    db = physics.normalize(physics.sub(pb, target))
    ang_a = physics.clamp(physics.dot(d_opt, da), -1.0, 1.0)
    ang_b = physics.clamp(physics.dot(d_opt, db), -1.0, 1.0)
    # 平分线：到两端点方向夹角近似相等
    assert abs(ang_a - ang_b) < 1e-6


def test_entry_angle_filter_survives_aim_refinement():
    """让点后入射角硬过滤仍然生效：中袋大斜角依旧判不可行。"""
    cue = (80.0, 178.0)
    target = (170.0, 150.0)
    pocket = (1000.0, 0.0)
    shot = physics.direct_shot(cue, target, pocket, R,
                               aim_half_width=1.45 * R, table_size=(W, H))
    assert shot.valid
    cos_in = physics.pocket_entry_cos(shot, W, H, R)
    assert cos_in is not None and cos_in < physics.MID_POCKET_MIN_COS
    assert not physics.pocket_entry_ok(shot, W, H, R)
    plans = physics.plan_shots(cue, target, [pocket], R, W, H,
                               allow_kicks=False, pocket_aim_half=1.45 * R)
    assert plans == []


def test_entry_cos_uses_real_travel_direction_not_center():
    """pocket_entry_cos 的方向必须取自出球方向（aim_point），而非袋口中心。"""
    # 构造鬼球-袋口中心方向正对、但真实出球方向（aim_point）更斜的方案
    shot = _mk_shot(pocket=(1000.0, 0.0), ghost=(1000.0, 300.0),
                    aim_point=(800.0, 0.0))
    cos_in = physics.pocket_entry_cos(shot, W, H, R)
    # ghost→pocket 正对（cos=1.0）；ghost→aim_point 明显偏斜
    expected = physics.dot(physics.normalize((-200.0, -300.0)), (0.0, -1.0))
    assert cos_in == pytest.approx(expected, abs=1e-6)
    legacy = _mk_shot(pocket=(1000.0, 0.0), ghost=(1000.0, 300.0))
    assert physics.pocket_entry_cos(legacy, W, H, R) == pytest.approx(1.0)


def test_success_prob_prefers_nearer_pocket_over_slightly_smaller_cut():
    """成功率排序：近袋稍大切胜远袋更小切（旧切角优先做不到的区分）。"""
    cfg = config.Config()
    near = _mk_shot(pocket=(0.0, 0.0), ttp=150.0, cut=6.0)
    far = _mk_shot(pocket=(0.0, 0.0), ttp=700.0, cut=4.0)
    p_near = physics.pot_success_prob(near, W, H, R, cfg)
    p_far = physics.pot_success_prob(far, W, H, R, cfg)
    assert 0.0 < p_far < p_near <= 1.0


def test_success_prob_penalizes_thin_cuts_and_cushions():
    """薄切与每库反弹都被单调惩罚；正常方案概率落在 (0,1]。"""
    cfg = config.Config()
    base = _mk_shot(pocket=(0.0, 0.0), ttp=400.0, cut=10.0)
    thin = _mk_shot(pocket=(0.0, 0.0), ttp=400.0, cut=65.0)
    kick = _mk_shot(pocket=(0.0, 0.0), ttp=400.0, cut=10.0, rails=("top",))
    kick2 = _mk_shot(pocket=(0.0, 0.0), ttp=400.0, cut=10.0,
                     rails=("top", "bottom"))
    p0 = physics.pot_success_prob(base, W, H, R, cfg)
    assert 0.0 < p0 <= 1.0
    assert physics.pot_success_prob(thin, W, H, R, cfg) < p0
    p1 = physics.pot_success_prob(kick, W, H, R, cfg)
    p2 = physics.pot_success_prob(kick2, W, H, R, cfg)
    assert 0.0 < p2 < p1 < p0


def test_target_shot_key_success_first_and_classic_fallback():
    """带几何参数：成功率优先（近袋大切胜远袋小切）；不带：旧切角优先。"""
    cfg = config.Config()
    good = _mk_shot(pocket=(0.0, 0.0), ttp=120.0, cut=12.0)
    bad = _mk_shot(pocket=(0.0, 0.0), ttp=800.0, cut=3.0)
    # 旧规则（无几何/关开关）：切角 3° 的远袋方案排前
    assert snooker.target_shot_key(bad) < snooker.target_shot_key(good)
    cfg_off = config.Config(rank_by_success=False)
    assert snooker.target_shot_key(bad, W, H, R, cfg_off) < \
        snooker.target_shot_key(good, W, H, R, cfg_off)
    # 新规则：成功率优先，近袋方案排前
    assert snooker.target_shot_key(good, W, H, R, cfg) < \
        snooker.target_shot_key(bad, W, H, R, cfg)
    # 无效/被挡方案永远垫底
    dead = _mk_shot(pocket=(0.0, 0.0), blocked=True)
    assert snooker.target_shot_key(dead, W, H, R, cfg)[0] == float("inf")
    assert snooker.target_shot_key(dead)[0] == float("inf")


def test_plan_shots_sets_aim_point_and_prob_in_range():
    """集成：开启让点后所有方案带 aim_point，成功率在 [0,1]。"""
    cfg = config.Config()
    cue = (300.0, 500.0)
    target = (700.0, 300.0)
    on = physics.plan_shots(cue, target, physics.default_pockets(W, H),
                            R, W, H, allow_kicks=False,
                            pocket_aim_half=cfg.pocket_accept_ratio * R)
    assert on
    assert all(s.aim_point is not None for s in on)
    probs = [physics.pot_success_prob(s, W, H, R, cfg) for s in on]
    assert all(p is not None and 0.0 <= p <= 1.0 for p in probs)
    # 关闭让点（不传参数）：维持旧行为
    off = physics.plan_shots(cue, target, physics.default_pockets(W, H),
                             R, W, H, allow_kicks=False)
    assert off
    assert all(s.aim_point is None for s in off)
    # 让点只平移在开口线上（不从台面飞走）：同一袋口的鬼球位移有限
    for s_on in on:
        s_off = next(s for s in off if s.pocket == s_on.pocket)
        d = physics.dist(s_on.ghost, s_off.ghost)
        assert d <= 1.5 * R, f"让点鬼球位移 {d:.1f} 超过 1.5R"
