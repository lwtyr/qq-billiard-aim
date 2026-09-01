# -*- coding: utf-8 -*-
"""TurnTracker 人工驱动策略（v3.9）回归测试。

语义（v3.9 起）：
  * 无人工干预一律瞄红球；红球清完无条件严格清彩（黄→绿→…→黑），
    Q 在清彩阶段无效（最后一颗红后想挑其他彩球，用 G 键手动点选）；
  * Q（仅红球在场时）= 一次性脉冲：下一杆瞄彩球，打完自动回落红球；
  * 「一杆」由球位位移判定：同色球跨更新最近邻匹配，最大位移
    > 8 台面单位记为击球开始，连续 2 次更新无位移记为击球结束；
  * 遮挡（occluded=True）/零信息帧冻结计数——漏看帧、UI 覆盖、
    检测抖动都不得让任选态泄漏到后续杆，也不得提前失效；
  * 打进红后球还在滚时按 Q：当前这杆不消耗脉冲，下一杆打完才回落。
"""
from aimtool import snooker


class B:
    """带位置的最小球桩（TurnTracker 只读 label/pos）。"""

    def __init__(self, label, pos):
        self.label = label
        self.pos = pos


def _balls(reds=0, colors=()):
    """白球 + 红×n + 彩球，位置沿 x 轴排开（间距大于位移阈值）。"""
    out = [B("白球", (300.0, 500.0))]
    x = 600.0
    for _ in range(reds):
        out.append(B("红球", (x, 500.0)))
        x += 90.0
    for c in colors:
        out.append(B(c, (x, 500.0)))
        x += 90.0
    return out


def _move(balls, dx=120.0):
    """整体平移所有球（单次位移 > 阈值 = 一击开始）。"""
    for b in balls:
        b.pos = (b.pos[0] + dx, b.pos[1])


def _nudge(balls, dx=3.0):
    """整体微移（位移阈值以下 = 检测噪声，不得计为一杆）。"""
    for b in balls:
        b.pos = (b.pos[0] + dx, b.pos[1])


def _play_one_shot(t, balls):
    """模拟真实一杆：先位移，再原地停稳两次，返回最后一次 update 结果。"""
    _move(balls)
    t.update(balls, stable=False)
    t.update(balls, stable=True)
    return t.update(balls, stable=True)


def test_default_always_red():
    """无人工干预：一律瞄红球；击球周期、新局铺球都不改变。"""
    t = snooker.TurnTracker()
    balls = _balls(reds=15)
    assert t.update(balls, stable=True) == "red"
    assert t.update(balls, stable=True) == "red"
    # 打过一杆依然是红球
    assert _play_one_shot(t, balls) == "red"
    # 红球减少依然是红球
    balls.pop()
    assert t.update(balls, stable=True) == "red"


def test_q_pulse_one_shot_then_back_to_red():
    """Q 脉冲：下一杆瞄彩球，该杆打完（位移→停稳）自动回落红球。"""
    t = snooker.TurnTracker()
    balls = _balls(reds=14)
    assert t.update(balls, stable=True) == "red"
    assert t.pulse_color(balls) == "color"
    # 未开杆前一直保持彩球
    assert t.update(balls, stable=True) == "color"
    assert t.update(balls, stable=True) == "color"
    # 打这一杆 → 回落红球
    assert _play_one_shot(t, balls) == "red"
    assert t.update(balls, stable=True) == "red"


def test_q_pulse_pressed_while_balls_rolling_survives_current_shot():
    """打进红后球还在滚时按 Q（最常见按法）：脉冲存活到下一杆。

    时序：按 Q 时球在动 → 当前这杆结束不消耗脉冲（guard）→
    下一杆瞄彩球 → 该杆结束才回落红球。
    """
    t = snooker.TurnTracker()
    balls = _balls(reds=14)
    assert t.update(balls, stable=True) == "red"
    # 打进红球：球开始滚
    _move(balls)
    assert t.update(balls, stable=False) == "red"
    # 球在滚时按 Q
    assert t.pulse_color(balls) == "color"
    _move(balls)
    assert t.update(balls, stable=False) == "color"   # 还在滚
    # 当前这杆停稳：guard 抵消，脉冲保留
    assert t.update(balls, stable=True) == "color"    # 停 1/2
    assert t.update(balls, stable=True) == "color"    # 停 2/2：本杆结束但脉冲不清
    # 下一杆（脉冲杆）打完才回落
    assert _play_one_shot(t, balls) == "red"


def test_q_pulse_never_leaks_when_shot_window_is_missed():
    """漏看击球过程（检测中断、_stage_targets 没被跑到）：

    以击球期间没有任何 update 调用来模拟。恢复调用后脉冲照常生效
    且仍只覆盖一杆——不会永久卡在彩球任选（v3.8 及以前的清彩
    跳序 bug 根因）。
    """
    t = snooker.TurnTracker()
    balls = _balls(reds=14)
    assert t.update(balls, stable=True) == "red"
    assert t.pulse_color(balls) == "color"
    _move(balls)  # 用户打了这一杆，期间没有任何 update 被调用
    assert _play_one_shot(t, balls) == "red"


def test_occluded_and_empty_frames_freeze_pulse_lifecycle():
    """UI 遮挡 / 零信息帧：冻结计数，不消耗脉冲也不判定击杆结束。"""
    t = snooker.TurnTracker()
    balls = _balls(reds=5)
    assert t.update(balls, stable=True) == "red"
    assert t.pulse_color(balls) == "color"
    _move(balls)
    assert t.update(balls, stable=False) == "color"    # 击球开始
    # 遮挡期间多帧：不许把「画面冻结不动」累积成击球结束
    for _ in range(3):
        assert t.update(balls, stable=False, occluded=True) == "color"
    assert t.update([], stable=False, occluded=True) == "color"   # 空帧
    # 遮挡结束：正常停稳两次才释放脉冲
    assert t.update(balls, stable=True) == "color"     # 停 1/2
    assert t.update(balls, stable=True) == "red"       # 停 2/2：释放


def test_tiny_jitter_does_not_count_as_shot():
    """亚阈值抖动（检测噪声）不得触发击球判定、不得消耗脉冲。"""
    t = snooker.TurnTracker()
    balls = _balls(reds=5)
    assert t.update(balls, stable=True) == "red"
    assert t.pulse_color(balls) == "color"
    _nudge(balls)
    assert t.update(balls, stable=True) == "color"
    _nudge(balls)
    assert t.update(balls, stable=True) == "color"
    _nudge(balls)
    assert t.update(balls, stable=True) == "color"
    # 只有真打一杆才回落
    assert _play_one_shot(t, balls) == "red"


def test_no_reds_strict_clearance_and_q_is_ignored():
    """红球清完：无条件严格清彩；Q 不再产生任选彩球（红后挑球用 G）。"""
    t = snooker.TurnTracker()
    assert t.update(_balls(reds=15), stable=True) == "red"
    clear_balls = _balls(reds=0, colors=("黄球", "黑球"))
    assert t.update(clear_balls, stable=True) == "clear"
    assert t.pulse_color(clear_balls) == "clear"       # Q 无效
    assert t.update(clear_balls, stable=True) == "clear"
    # 清彩阶段打一杆没进：依然是清彩
    assert _play_one_shot(t, clear_balls) == "clear"


def test_next_color_uses_first_color_in_clearance_order():
    """清彩目标始终取黄→绿→棕→蓝→粉→黑中第一颗在场彩球。"""
    balls = _balls(reds=0, colors=("黑球", "蓝球", "黄球", "粉球"))
    assert snooker.next_color(balls).label == "黄球"
    balls = _balls(reds=0, colors=("黑球", "粉球", "蓝球"))
    assert snooker.next_color(balls).label == "蓝球"


def test_reset_clears_pulse():
    t = snooker.TurnTracker()
    balls = _balls(reds=5)
    t.update(balls, stable=True)
    t.pulse_color(balls)
    assert t.ball_on == "color"
    t.reset()
    assert t.update(balls, stable=True) == "red"


def test_w_pulse_forces_red_and_clears_pending_color_pulse():
    """W 强制切回红球，并取消当前及带 guard 的彩球脉冲。"""
    t = snooker.TurnTracker()
    balls = _balls(reds=5)
    t.update(balls, stable=True)
    _move(balls)
    t.update(balls, stable=False)                      # 击球进行中
    assert t.pulse_color(balls) == "color"
    assert t._pulse_guard is True                      # 滚动中按 Q → guard 置位
    assert t.pulse_red() == "red"
    assert t._color_pulse is False
    assert t._pulse_guard is False
    assert t.update(balls, stable=True) == "red"


def test_w_pulse_returns_to_clearance_when_no_reds_remain():
    """W 只在仍有红球时生效；红球清完后的下一次更新自动回到严格清彩。"""
    t = snooker.TurnTracker()
    clear_balls = _balls(reds=0, colors=("黄球", "黑球"))
    assert t.update(clear_balls, stable=True) == "clear"
    assert t.pulse_red() == "red"          # 立即反馈；场上状态由 update 裁决
    assert t.update(clear_balls, stable=True) == "clear"
    assert t._force_red is False


def test_w_pulse_forces_red_until_reds_are_gone():
    """W 在仍有红球时强制红球，红球清完的下一次更新进入清彩。"""
    t = snooker.TurnTracker()
    red_balls = _balls(reds=2, colors=("黄球", "黑球"))
    assert t.pulse_red() == "red"
    assert t.update(red_balls, stable=True) == "red"
    assert t._force_red is True
    cleared = _balls(reds=0, colors=("黄球", "黑球"))
    assert t.update(cleared, stable=True) == "clear"
    assert t._force_red is False


def test_q_in_clearance_keeps_strict_order_after_shots():
    """清彩阶段连打几杆（含误按 Q）：目标始终严格在低分彩球一侧。"""
    t = snooker.TurnTracker()
    balls = _balls(reds=0, colors=("黄球", "绿球", "黑球"))
    assert t.update(balls, stable=True) == "clear"
    assert t.pulse_color(balls) == "clear"
    assert _play_one_shot(t, balls) == "clear"
    # 黄球进袋（移除黄球）：轮到绿球，仍是严格清彩
    balls = [balls[0]] + balls[2:]
    assert t.update(balls, stable=True) == "clear"
    assert t.pulse_color(balls) == "clear"
    assert t.update(balls, stable=True) == "clear"
