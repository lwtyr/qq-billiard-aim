# -*- coding: utf-8 -*-
"""TurnTracker 人工驱动策略（v3.7）回归测试。

用户方案：无人工干预一律瞄红球；要打彩球时按 Q（一次性脉冲，
下一杆瞄彩球），该杆打完自动回落瞄红球；红球清完自动严格清彩。
视觉无法可靠区分「进红/没进/换手」，状态推断全部废除，只保留
击杆周期（球动过 = 打过一杆）驱动脉冲回落。
"""
from aimtool import snooker


def _balls(reds=0, colors=()):
    out = [type("B", (), {"label": "白球"})()]
    out += [type("B", (), {"label": "红球"})() for _ in range(reds)]
    out += [type("B", (), {"label": c})() for c in colors]
    return out


def test_default_always_red():
    """无人工干预：一律瞄红球，任何球数变化都不改变。"""
    t = snooker.TurnTracker()
    assert t.update(_balls(reds=15), stable=True) == "red"
    assert t.update(_balls(reds=14), stable=True) == "red"
    assert t.update(_balls(reds=13), stable=True) == "red"
    # 打过一杆（动→停）依然是红球
    assert t.update(_balls(reds=13), stable=False) == "red"
    assert t.update(_balls(reds=13), stable=True) == "red"
    # 新开一把（红球满桌）依然是红球 —— 无状态残留问题
    assert t.update(_balls(reds=15), stable=True) == "red"


def test_q_pulse_last_one_shot_then_back_to_red():
    """Q 脉冲：下一杆瞄彩球，该杆打完（动→停）自动回落红球。"""
    t = snooker.TurnTracker()
    balls = _balls(reds=14)
    assert t.update(balls, stable=True) == "red"
    assert t.pulse_color(balls) == "color"
    # 未开杆前一直保持彩球
    assert t.update(balls, stable=True) == "color"
    # 打这一杆：动 → 停 → 回落红球
    assert t.update(balls, stable=False) == "color"
    assert t.update(balls, stable=True) == "red"


def test_q_pulse_while_balls_moving_survives_current_shot():
    """打进红后球还在滚时按 Q（最常见按法）：脉冲必须存活到下一杆。

    时序：按 Q 时球在动 → 先等当前这杆结束（脉冲不清）→
    下一杆瞄彩球 → 该杆结束才回落红球。
    """
    t = snooker.TurnTracker()
    balls = _balls(reds=14)
    assert t.update(balls, stable=True) == "red"
    # 打进红球：球开始滚
    assert t.update(balls, stable=False) == "red"
    # 球在滚时按 Q
    assert t.pulse_color(balls) == "color"
    # 当前这杆结束：脉冲刚生效，不回落
    assert t.update(balls, stable=True) == "color"
    # 下一杆瞄彩球，打完回落
    assert t.update(balls, stable=False) == "color"
    assert t.update(balls, stable=True) == "red"


def test_no_reds_defaults_to_strict_clearance():
    """红球清完：自动严格清彩，无需任何按键。"""
    t = snooker.TurnTracker()
    assert t.update(_balls(reds=15), stable=True) == "red"
    assert t.update(_balls(reds=0, colors=("黄球", "黑球")),
                    stable=True) == "clear"
    # 清彩阶段打一杆没进：依然是清彩
    assert t.update(_balls(reds=0, colors=("黄球", "黑球")),
                    stable=False) == "clear"
    assert t.update(_balls(reds=0, colors=("黄球", "黑球")),
                    stable=True) == "clear"


def test_next_color_uses_first_color_in_clearance_order():
    """清彩目标始终取黄→绿→棕→蓝→粉→黑中第一颗在场彩球。"""
    balls = _balls(reds=0, colors=("黑球", "蓝球", "黄球", "粉球"))
    assert snooker.next_color(balls).label == "黄球"
    balls = _balls(reds=0, colors=("黑球", "粉球", "蓝球"))
    assert snooker.next_color(balls).label == "蓝球"


def test_q_pulse_without_reds_allows_one_free_color_shot():
    """最后一颗红后的那一杆本就是任选彩球：Q 脉冲在无红球时覆盖清彩一杆。"""
    t = snooker.TurnTracker()
    clear_balls = _balls(reds=0, colors=("黄球", "黑球"))
    assert t.update(clear_balls, stable=True) == "clear"
    # 刚打进最后一颗红，按 Q → 下一杆任选彩球
    assert t.pulse_color(clear_balls) == "color"
    assert t.update(clear_balls, stable=True) == "color"
    # 该杆结束 → 回落严格清彩
    assert t.update(clear_balls, stable=False) == "color"
    assert t.update(clear_balls, stable=True) == "clear"


def test_reset_clears_pulse():
    t = snooker.TurnTracker()
    balls = _balls(reds=5)
    t.pulse_color(balls)
    assert t.ball_on == "color"
    t.reset()
    assert t.update(balls, stable=True) == "red"


def test_w_pulse_forces_red_and_clears_pending_color_pulse():
    """W 强制切回红球，并取消当前及待生效的彩球脉冲。"""
    t = snooker.TurnTracker()
    balls = _balls(reds=5)
    t.update(balls, stable=False)
    assert t.pulse_color(balls) == "color"
    assert t.pulse_red() == "red"
    assert t._color_pulse is False
    assert t._pulse_pending is False
    assert t.update(balls, stable=True) == "red"


def test_w_pulse_returns_to_clearance_when_no_reds_remain():
    """W 只在仍有红球时生效；红球清完后自动回到严格清彩。"""
    t = snooker.TurnTracker()
    clear_balls = _balls(reds=0, colors=("黄球", "黑球"))
    assert t.update(clear_balls, stable=True) == "clear"
    assert t.pulse_red() == "red"
    assert t.update(clear_balls, stable=True) == "clear"
    assert t._force_red is False


def test_w_pulse_forces_red_until_reds_are_gone():
    """W 在仍有红球时强制红球，红球清完的下一次更新进入清彩。"""
    t = snooker.TurnTracker()
    red_balls = _balls(reds=2, colors=("黄球", "黑球"))
    assert t.pulse_red() == "red"
    assert t.update(red_balls, stable=True) == "red"
    assert t._force_red is True
    assert t.update(_balls(reds=0, colors=("黄球", "黑球")), stable=True) == "clear"
    assert t._force_red is False


def test_q_pulse_in_clearance_recovers_to_clear():
    """Q 在清彩阶段只覆盖一杆，击杆结束后恢复严格清彩。"""
    t = snooker.TurnTracker()
    balls = _balls(reds=0, colors=("黄球", "绿球", "黑球"))
    assert t.update(balls, stable=True) == "clear"
    assert t.pulse_color(balls) == "color"
    assert t.update(balls, stable=True) == "color"
    assert t.update(balls, stable=False) == "color"
    assert t.update(balls, stable=True) == "clear"
