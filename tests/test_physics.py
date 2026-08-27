"""物理引擎单元测试：鬼球、切角、一库/两库反弹、障碍、力度。"""
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aimtool import physics


def approx(a, b, tol=1e-6):
    assert abs(a - b) < tol, f"{a} != {b}"


def test_ghost_pos():
    # 目标球 (1000,500) 打进上方中袋 (1000,0)，r=25：鬼球应在目标球下方 50
    g = physics.ghost_pos((1000, 500), (1000, 0), 25)
    approx(g[0], 1000)
    approx(g[1], 550)


def test_ghost_pos_degenerate():
    # target == pocket → normalize 返回 None，ghost_pos 应返回 None 而非崩溃
    assert physics.ghost_pos((100, 50), (100, 50), 25) is None


def test_contact_pos_is_on_target_surface():
    target, pocket, r = (1000, 500), (1000, 0), 25
    assert physics.contact_pos(target, pocket, r) == (1000.0, 525.0)


def test_ghost_and_contact_use_per_ball_radii_and_offset():
    target, pocket = (1000.0, 500.0), (1000.0, 0.0)
    ghost = physics.ghost_pos(
        target, pocket, 25.0, cue_radius=20.0, target_radius=30.0,
        offset=(2.0, -3.0),
    )
    assert ghost == (1002.0, 547.0)
    assert physics.contact_pos(target, pocket, 25.0,
                               target_radius=30.0) == (1000.0, 530.0)


def test_impact_ghost_and_contact_follow_cue_to_target_line():
    cue, target, r = (100.0, 500.0), (500.0, 500.0), 25.0
    assert physics.impact_ghost(cue, target, r) == (450.0, 500.0)
    assert physics.impact_contact_pos(cue, target, r) == (475.0, 500.0)


def test_direct_shot_passes_per_ball_radii_to_ghost_geometry():
    shot = physics.direct_shot(
        (1000.0, 800.0), (1000.0, 500.0), (1000.0, 0.0), 25.0,
        cue_radius=20.0, target_radius=30.0,
    )
    assert shot.valid
    assert shot.ghost == (1000.0, 550.0)


def test_direct_shot_geometry():
    # 目标球 (1000,500) 打进上方中袋 (1000,0)，r=25：鬼球在 (1000,550)。
    # 母球放在鬼球正下方 → 正碰（切角 0），出发方向正上。
    cue, target, pocket, r = (1000, 800), (1000, 500), (1000, 0), 25
    s = physics.direct_shot(cue, target, pocket, r)
    assert s.valid
    approx(s.aim_dir[0], 0.0)
    approx(s.aim_dir[1], -1.0)
    approx(s.cut_deg, 0.0)
    # 总路程 = 250(母球到鬼球) + 500(目标球到袋口)
    approx(s.total, 750, 1e-3)
    # 鬼球位置校验
    approx(s.ghost[0], 1000)
    approx(s.ghost[1], 550)


def test_direct_shot_cut_angle():
    # 斜切：母球偏下，切角应落在 (0, 90)
    cue, target, pocket, r = (400, 700), (1000, 500), (1000, 0), 25
    s = physics.direct_shot(cue, target, pocket, r)
    assert s.valid
    assert 0 < s.cut_deg < 90
    assert s.aim_dir[1] < 0


def test_direct_shot_extreme_cut_rejected():
    # 近乎极限薄切：接触时沿袋口方向分速度≈0，物理上进不了球，
    # 必须判 invalid（否则瞄准器会推荐必丢方案）
    target, pocket, r = (1900, 500), (2000, 500), 25
    ghost = physics.ghost_pos(target, pocket, r)   # ≈ (1850, 500)
    cue = (1855, 900)                              # 从鬼球正下方极近处打 → 切角接近90°
    s = physics.direct_shot(cue, target, pocket, r, max_cut_deg=85.0)
    if s.cut_deg > 85.0:
        assert not s.valid


def test_direct_shot_blocked():
    # 正碰构型（切角0°）：障碍球恰好在母球→鬼球线段上 → 被挡
    cue, target, pocket, r = (1000, 800), (1000, 500), (1000, 0), 25
    blocker = (1000, 650)
    s = physics.direct_shot(cue, target, pocket, r, others=[blocker])
    assert s.valid and s.blocked
    s2 = physics.direct_shot(cue, target, pocket, r, others=[])
    assert not s2.blocked


def test_target_to_pocket_path_blocked():
    # 目标球到袋口之间的障碍也会导致进球失败，不能只检查母球路线。
    cue, target, pocket, r = (1000, 800), (1000, 500), (1000, 0), 25
    blocker = (1000, 250)
    s = physics.direct_shot(cue, target, pocket, r, others=[blocker])
    assert s.valid and s.blocked


def test_ghost_unreachable():
    # 母球与鬼球几乎重合 → 不可打（无明确出发方向）
    target, pocket, r = (1000, 500), (1000, 0), 25
    g = physics.ghost_pos(target, pocket, r)
    s = physics.direct_shot(g, target, pocket, r)
    assert not s.valid


def test_one_rail_kick():
    # 一库解围：展开法给出方向后仿真应确认恰好一次 top 反弹并穿过鬼球。
    # （回归测试：旧版仿真缺「射线穿过鬼球」终止条件，visited 序列
    #   会多记反弹导致所有 kick 永远判无效。）
    cue, target, pocket, r = (200, 900), (1500, 300), (1000, 1000), 25
    s = physics.kick_shot(cue, target, pocket, r, ("top",), 2000, 1000)
    assert s.valid, f"一库解围应有效: cut={s.cut_deg:.1f}"
    assert len(s.bounce_points) == 1
    bx, by = s.bounce_points[0]
    approx(by, 0.0)                     # 反弹点在 top 库（y=0）
    assert 0 <= bx <= 2000
    assert s.aim_dir[1] < 0             # 出发朝上
    # 总路程必须按实际折线路径计算（> 直线距离），否则力度系统性偏小
    straight = math.hypot(cue[0] - s.ghost[0], cue[1] - s.ghost[1]) + \
        math.hypot(target[0] - pocket[0], target[1] - pocket[1])
    assert s.total > straight + 1e-6
    # 障碍检查覆盖全部真实路径段：挡在第一腿 / 目标球→袋口腿都要能检出
    b1 = physics.kick_shot(cue, target, pocket, r, ("top",), 2000, 1000,
                           others=[(700, 450)])
    assert b1.blocked and "700" in str(b1.blocked_by)
    b2 = physics.kick_shot(cue, target, pocket, r, ("top",), 2000, 1000,
                           others=[(1325, 545)])
    assert b2.blocked                   # 目标球→袋口腿上的障碍


def test_kick_wrong_sequence_invalid():
    # 该构型一库(top)后射线即穿过鬼球；若强行要求两库(top,bottom)，
    # 实际反弹序列只有 top → 序列不匹配，必须判 invalid
    cue, target, pocket, r = (200, 900), (1500, 300), (1000, 1000), 25
    s = physics.kick_shot(cue, target, pocket, r, ("top", "bottom"), 2000, 1000)
    assert not s.valid


def test_two_rail_kick_perpendicular():
    # 垂直两库 (top,right)：展开法+仿真应给出精确两次反弹的可行解。
    # 反弹点分别落在对应库上；总路程为实际三段折线之和。
    cue, target, pocket, r = (150, 150), (1900, 500), (2000, 1000), 25
    s = physics.kick_shot(cue, target, pocket, r, ("top", "right"), 2000, 1000)
    assert s.valid, f"垂直两库应有效: cut={s.cut_deg:.1f}"
    assert len(s.bounce_points) == 2
    assert abs(s.bounce_points[0][1]) < 1e-6            # 第一反弹在 top
    assert abs(s.bounce_points[1][0] - 2000) < 1e-6     # 第二反弹在 right
    assert not s.blocked


def test_two_rail_parallel_sequence_checked():
    # 平行两库 (left,right)：镜像顺序对轴对齐库可交换，但序列/越界校验
    # 仍必须生效——反弹点若不在对应库上或超出边界即判无效。
    for rails in (("left", "right"), ("top", "bottom")):
        s = physics.kick_shot((300, 500), (1540, 370), (2000, 0), 25, rails, 2000, 1000)
        if s.valid:
            assert len(s.bounce_points) == len(rails)


def test_plan_shots_priority():
    # 无障碍时直球应排第一
    cue, target, pockets, r = (400, 500), (1000, 500), physics.default_pockets(2000, 1000), 25
    plans = physics.plan_shots(cue, target, pockets, r, 2000, 1000)
    assert plans
    assert plans[0].label == "直球"
    assert not plans[0].bounce_points
    # 排序规则：直球(未挡) < 直球(被挡) < 一库 < 两库，同级按总路程升序
    keys = [(0 if not p.bounce_points else 2 + len(p.bounce_points), p.total)
            for p in plans]
    assert keys == sorted(keys)


def test_plan_shots_omits_blocked_routes_when_clear_kick_exists():
    # 直球被障碍挡住，但一库路线畅通时，自动规划不能把直球排在前面。
    cue = (1696.3605234707295, 681.3073248918987)
    target = (1054.0447094707295, 840.8070487341598)
    pocket = (0.0, 1000.0)
    blocker = (1397.450309910817, 757.6971054914331)
    plans = physics.plan_shots(cue, target, [pocket], 22.5, 2000, 1000,
                               [blocker], allow_kicks=True, max_kicks=2)
    assert plans
    assert all(not shot.blocked for shot in plans)
    assert plans[0].bounce_points


def test_best_shot_penalizes_thin_cut_over_shorter_route():
    """自动路线不能只看总路程，极薄切应让位于容错更高的路线。"""
    safe = physics.Shot(
        pocket=(2000.0, 500.0), ghost=(1000.0, 500.0), aim_dir=(1.0, 0.0),
        cue_to_contact=900.0, target_to_pocket=600.0, total=1500.0,
        cut_deg=18.0, valid=True,
    )
    thin = physics.Shot(
        pocket=(2000.0, 500.0), ghost=(1000.0, 500.0), aim_dir=(1.0, 0.0),
        cue_to_contact=500.0, target_to_pocket=400.0, total=900.0,
        cut_deg=83.0, valid=True,
    )
    assert physics.best_shot([thin, safe], 2000.0) is safe


def test_power_suggestion():
    W = 2000
    p0 = physics.power_suggestion(0, W)
    assert p0 >= 10
    p1 = physics.power_suggestion(2.2 * W, W)
    assert p1 >= 95
    pbig = physics.power_suggestion(10 * W, W)
    assert pbig <= 100
    assert 10 <= p0 <= p1 <= pbig


def test_power_cut_compensation():
    # 同样路程下切角越大需要越高杆速（动量传递≈cos²cut）
    W = 2000
    base = physics.power_suggestion(600, W, cut_deg=0.0)
    mid = physics.power_suggestion(600, W, cut_deg=50.0)
    big = physics.power_suggestion(600, W, cut_deg=75.0)
    assert 10 <= base < mid < big <= 100
    # ≤30° 不补偿
    assert physics.power_suggestion(600, W, cut_deg=20.0) == base


def test_pockets_default():
    ps = physics.default_pockets(2000, 1000)
    assert len(ps) == 6
    assert (0, 0) in ps and (2000, 0) in ps and (1000, 0) in ps


def test_reflect_across():
    assert physics.reflect_across((100, 100), "top", 2000, 1000) == (100, -100)
    assert physics.reflect_across((100, 100), "bottom", 2000, 1000) == (100, 1900)
    assert physics.reflect_across((100, 100), "left", 2000, 1000) == (-100, 100)
    assert physics.reflect_across((100, 100), "right", 2000, 1000) == (3900, 100)


def test_rail_inset_uses_ball_center_collision_line():
    """库边内缩只改变球心反弹线，不改变默认 inset=0 的兼容行为。"""
    assert physics.reflect_across((100, 100), "top", 2000, 1000, inset=25) == (100, -50)
    t = physics.ray_rail_t((100, 100), (0, -1), "top", 2000, 1000, inset=25)
    assert t == 75.0
    assert physics.rail_crossing((100, 100), (0, -1), "top", 2000, 1000,
                                 inset=25) == (100.0, 25.0)
