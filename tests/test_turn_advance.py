# -*- coding: utf-8 -*-
"""TurnTracker 击杆周期感知 + 红球减少双帧确认 回归测试。

修复两个实战问题：
  1. 打进红球后仍继续瞄红球：红球数识别抖动（球架遮挡 ±1）会把基线
     拉低，之后真进球触发不了「红球减少」事件。现在需连续两个稳定帧
     确认，确认前不同步基线。
  2. 清彩阶段跳过黄球直接瞄绿球：红球清完后若「红后任选彩球」那一杆
     没进球，状态永远停在任选彩球（按切角任选）。现在利用击杆周期
     （稳定→非稳定→稳定）感知「打过一杆且没进球」，据此推进状态。
"""
from aimtool import config, vision, snooker

W, H = 1680, 840


def _balls(reds=0, colors=()):
    cue = vision.Ball("白球", (300.0, 500.0), 20.0)
    out = [cue]
    for i in range(reds):
        out.append(vision.Ball("红球", (600.0 + 30.0 * i, 400.0), 20.0))
    palette = ("黄球", "绿球", "棕球", "蓝球", "粉球", "黑球")
    for i, name in enumerate(colors):
        out.append(vision.Ball(name, (1100.0 + 40.0 * i, 500.0), 20.0))
    return out


def test_potted_red_then_color_miss_falls_back_to_red():
    """红球还在：红后彩球没进 → 换手继续打红球。"""
    t = snooker.TurnTracker()
    assert t.update(_balls(reds=14, colors=("黑球",)), stable=True) == "red"
    # 进一颗红（双帧确认）→ 红后任选彩球
    assert t.update(_balls(reds=13, colors=("黑球",)), stable=True) == "red"
    assert t.update(_balls(reds=13, colors=("黑球",)), stable=True) == "color"
    # 一杆没进（球动了又停，球数未变）→ 回到红球
    assert t.update(_balls(reds=13, colors=("黑球",)), stable=False) == "color"
    assert t.update(_balls(reds=13, colors=("黑球",)), stable=True) == "red"


def test_last_red_color_miss_enters_strict_clearance():
    """红球已清：红后彩球没进 → 直接进入严格清彩顺序（不跳号）。"""
    t = snooker.TurnTracker()
    assert t.update(_balls(reds=1, colors=("黄球", "绿球")), stable=True) == "red"
    assert t.update(_balls(colors=("黄球", "绿球")), stable=True) == "red"
    assert t.update(_balls(colors=("黄球", "绿球")), stable=True) == "color"
    # 打黄球没进 → 不能停在「任选彩球」，必须按顺序打黄球（clear）
    assert t.update(_balls(colors=("黄球", "绿球")), stable=False) == "color"
    assert t.update(_balls(colors=("黄球", "绿球")), stable=True) == "clear"


def test_clearance_miss_stays_clear():
    """清彩阶段没进球 → 继续打当前最低分彩球，状态不变。"""
    t = snooker.TurnTracker()
    assert t.update(_balls(colors=("黄球", "绿球")), stable=True) == "clear"
    assert t.update(_balls(colors=("黄球", "绿球")), stable=False) == "clear"
    assert t.update(_balls(colors=("黄球", "绿球")), stable=True) == "clear"


def test_red_count_single_frame_drop_is_ignored():
    """单帧红球数减少（识别抖动）不切换；基线不被低值污染。"""
    t = snooker.TurnTracker()
    assert t.update(_balls(reds=15), stable=True) == "red"
    # 抖动一帧：14 → pending，返回 red，基线仍 15
    assert t.update(_balls(reds=14), stable=True) == "red"
    # 恢复 15：基线不被拉低
    assert t.update(_balls(reds=15), stable=True) == "red"
    # 真进一颗：双帧确认 → color
    assert t.update(_balls(reds=14), stable=True) == "red"
    assert t.update(_balls(reds=14), stable=True) == "color"


def test_miss_advance_skipped_during_manual_override():
    """Q 手动覆盖期间，没进球的杆不推进（用户明确指定的状态优先）。"""
    t = snooker.TurnTracker()
    balls = _balls(reds=10, colors=("黄球",))
    assert t.update(balls, stable=True) == "red"
    assert t.toggle_red_color(balls) == "color"
    assert t.update(balls, stable=False) == "color"
    assert t.update(balls, stable=True) == "color"   # 没进，仍保持 Q 指定


# ---------- 新局检测（v3.6.3 回归：上一把残留 clear → 新一把全程无可行方案） ----------

def test_new_frame_after_clear_restores_red():
    """上一把打完停在 clear，新一把红球重现 → 立即回红球阶段。"""
    t = snooker.TurnTracker()
    # 上一把：清彩阶段（只剩黑球）
    assert t.update(_balls(reds=0, colors=("黑球",)), stable=True) == "clear"
    # 新开一把：红球满桌（红球从 0 → 15，clear 阶段不该有红球）
    assert t.update(_balls(reds=15, colors=("黄球", "绿球", "棕球",
                                            "蓝球", "粉球", "黑球")),
                    stable=True) == "red"


def test_new_frame_color_state_red_surge_restores_red():
    """任选彩球阶段红球大增（一杆最多少 1-2 颗）→ 新一把，回红球阶段。"""
    t = snooker.TurnTracker()
    assert t.update(_balls(reds=2, colors=("黑球",)), stable=True) == "red"
    assert t.update(_balls(reds=1, colors=("黑球",)), stable=True) == "red"
    assert t.update(_balls(reds=1, colors=("黑球",)), stable=True) == "color"
    # 新一把：红球 1 → 15（+14 ≥ 3）
    assert t.update(_balls(reds=15, colors=("黄球", "绿球", "棕球",
                                            "蓝球", "粉球", "黑球")),
                    stable=True) == "red"


def test_manual_override_survives_new_frame_detection():
    """Q 手动覆盖期间不做新局推进（用户指定优先）。"""
    t = snooker.TurnTracker()
    assert t.update(_balls(reds=5, colors=("黑球",)), stable=True) == "red"
    t.cycle_manual(_balls(reds=5, colors=("黑球",)))  # 切到红后任选彩球
    # 新一把：红球 5 → 15（+10），但手动覆盖优先，不推进
    assert t.update(_balls(reds=15, colors=("黄球",)), stable=True) == "color"
