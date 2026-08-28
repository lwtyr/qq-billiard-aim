"""中袋入射角过滤（pocket_entry_ok）回归测试。

背景 bug3：目标球在 (305,77)、上中袋 (500,0) 时入射角约 70°，
物理上必撞袋角弹出，但切角校验放行导致辅助推荐该路线 → 打丢。
"""
from aimtool import physics


def _straight_cue(target, pocket, r, back=300.0):
    """把母球放在 target→pocket 延长线的鬼球正后方（正碰，cut≈0）。"""
    g = physics.ghost_pos(target, pocket, r, r, r)
    d = physics.normalize(physics.sub(pocket, g))
    return physics.add(g, physics.mul(d, -back))


def test_mid_pocket_shallow_angle_rejected():
    """bug3 场景：中袋 ~70° 大斜角必须被判不可行。"""
    w, h, r = 1000.0, 500.0, 21.0
    target, pocket = (305.0, 77.0), (500.0, 0.0)
    cue = _straight_cue(target, pocket, r)
    shot = physics.direct_shot(cue, target, pocket, r)
    assert shot.valid, "切角校验应放行（拒绝必须来自入射角检查）"
    assert not physics.pocket_entry_ok(shot, w, h, r)


def test_mid_pocket_head_on_ok():
    """正对中袋的球必须照常推荐。"""
    w, h, r = 1000.0, 500.0, 21.0
    target, pocket = (500.0, 150.0), (500.0, 0.0)
    cue = (500.0, 380.0)
    shot = physics.direct_shot(cue, target, pocket, r)
    assert shot.valid
    assert physics.pocket_entry_ok(shot, w, h, r)


def test_corner_along_rail_ok():
    """角袋沿库边抹进（cos≈0.8）必须保留。"""
    w, h, r = 1000.0, 500.0, 21.0
    target, pocket = (150.0, 30.0), (0.0, 0.0)
    cue = _straight_cue(target, pocket, r)
    shot = physics.direct_shot(cue, target, pocket, r)
    assert shot.valid
    assert physics.pocket_entry_ok(shot, w, h, r)


def test_ghost_near_pocket_passes():
    """鬼球距袋口 < 2r 时方向几何不稳，放行。"""
    shot = physics.Shot(
        pocket=(500.0, 0.0), ghost=(505.0, 30.0),
        aim_dir=(0.0, 1.0), cue_to_contact=100.0,
        target_to_pocket=30.0, total=130.0, cut_deg=10.0, valid=True,
    )
    assert physics.pocket_entry_ok(shot, 1000.0, 500.0, 21.0)


def test_plan_shots_filters_mid_pocket():
    """plan_shots 集成：中袋大斜角路线被淘汰，同场景角袋路线保留。"""
    w, h, r = 1000.0, 500.0, 21.0
    pockets = physics.default_pockets(w, h)
    target, pocket_mid = (305.0, 77.0), (500.0, 0.0)
    cue = _straight_cue(target, pocket_mid, r)
    plans = physics.plan_shots(cue, target, pockets, r, w, h,
                               allow_kicks=False)
    assert plans, "右上角袋直球路线应保留"
    assert all(s.pocket != pocket_mid for s in plans), \
        "上中袋大斜角路线必须被过滤"
