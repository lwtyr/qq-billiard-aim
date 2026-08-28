"""斯诺克决策层：目标球选择 + 局面说明。

斯诺克规则（区别于八球"打最近的球"）：
  * 红球阶段：台面上还有红球时，目标是「某颗红球」（选最好进且不被挡的）；
  * 清彩阶段：红球清完后，必须按分值顺序打彩球：黄(2)→绿(3)→棕(4)→蓝(5)→粉(6)→黑(7)。

只依赖识别出的球列表 + 标准 6 袋 + 台面尺寸，不依赖彩球点位知识
（彩球复位后位置由每帧全量识别自动跟踪）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from aimtool import physics

# 清彩顺序（分值升序）
COLOR_ORDER: Tuple[str, ...] = ("黄球", "绿球", "棕球", "蓝球", "粉球", "黑球")
COLOR_VALUE = {"黄球": 2, "绿球": 3, "棕球": 4, "蓝球": 5, "粉球": 6, "黑球": 7}
VALUE_LABEL = {v: k for k, v in COLOR_VALUE.items()}


@dataclass
class TurnTracker:
    """跨稳定局面维护当前允许击打的球类。

    单帧看到红球并不代表本杆必须打红球：刚进红后应任选彩球。视觉
    无法可靠地从“未进球/对方换手”中推断球权，因此在这种歧义下保留
    手动切换入口，而不是假装能从一张截图知道游戏内部状态。
    """

    ball_on: Optional[str] = None  # red / color / clear
    _reds: Optional[int] = None
    _colors: Optional[int] = None
    # Q/O is a session override, separate from the visual rule inference.  A
    # A stable frame with the same ball counts must not undo a user's toggle;
    # the override is released when the visible ball counts actually change.
    _manual_ball_on: Optional[str] = None
    _manual_counts: Optional[Tuple[int, int]] = None
    # 击杆周期感知：台面从稳定进入非稳定（球在动）即一杆开始，恢复稳
    # 定时若球数与杆前相同＝这杆没进球。视觉无法区分「还没打」与「打
    # 了没进」，但 MOVING→READY 周期本身就是「打过一杆」的证据，必须
    # 据此推进状态，否则红后彩球没进会永远停在「任选彩球」（跳过黄
    # 球直接磁切角好的绿球）。
    _pending_shot: bool = False
    _pre_shot_counts: Optional[Tuple[int, int]] = None
    # 红球数识别偶发 ±1 抖动（球架遮挡）：单帧减少不确认，连续两个稳
    # 定帧仍少才切换到红后彩球；确认前不同步基线，避免基线被低值污
    # 染后真进球再也触发不了。
    _red_drop: int = 0

    def reset(self) -> None:
        self.ball_on = None
        self._reds = None
        self._colors = None
        self._manual_ball_on = None
        self._manual_counts = None
        self._pending_shot = False
        self._pre_shot_counts = None
        self._red_drop = 0

    def update(self, balls: Sequence, stable: bool) -> str:
        reds = reds_remaining(balls)
        colors = sum(1 for b in balls if b.label in COLOR_ORDER)
        if not stable:
            if self._reds is not None and not self._pending_shot:
                self._pre_shot_counts = (self._reds, self._colors or 0)
            self._pending_shot = True
            return self.ball_on or ("clear" if reds == 0 else "red")

        counts = (reds, colors)
        if self._pending_shot:
            # 一杆结束（MOVING/STABILIZING → READY）。球数与杆前相同＝
            # 这一杆没进球：
            #   * 红后任选彩球没进 → 红球还在则换手打红；红球已清则
            #     进入严格清彩顺序；
            #   * 红球/清彩阶段没进 → 本来就继续打同类，无需变化。
            # Q 手动覆盖期间不推进（用户明确指定的状态优先）。
            self._pending_shot = False
            if (self._pre_shot_counts is not None
                    and counts == self._pre_shot_counts
                    and self._manual_ball_on is None
                    and self.ball_on == "color"):
                self.ball_on = "red" if reds > 0 else "clear"
            self._pre_shot_counts = None

        if self._manual_ball_on is not None:
            counts = (reds, colors)
            if reds == 0:
                self._manual_ball_on = None
                self._manual_counts = None
                if self._reds is None:
                    # Q pressed before the first frame, and that first frame
                    # is already a clear-colour position.
                    self.ball_on = "clear"
                    self._reds, self._colors = counts
                    return self.ball_on
            elif self._manual_counts is None:
                # Q may have been pressed before the first stable detection.
                self._manual_counts = counts
            elif counts != self._manual_counts:
                # A real ball-count change marks the end of the one-shot user
                # override; resume the normal red-after-pot/color progression.
                self._manual_ball_on = None
                self._manual_counts = None
            else:
                self.ball_on = self._manual_ball_on
                self._reds, self._colors = counts
                return self.ball_on

        if self._reds is None:
            # Q 可能在首个稳定帧前按下；已有手动状态时保留它，
            # 否则按当前台面初始化默认状态。
            if self.ball_on is None:
                self.ball_on = "clear" if reds == 0 else "red"
        elif reds < self._reds:
            # 红球数减少（包括最后一颗红球）后，下一杆都必须先选彩球。
            # 不能在 reds==0 时直接进入 clear，否则会跳过最后一颗红后的
            # 任选彩球阶段。单帧减少不算：连续两个稳定帧仍少才确认，
            # 确认前不同步基线，防止遮挡抖动把基线拉低后真进球失灵。
            if self._red_drop + 1 >= 2:
                self.ball_on = "color"
                self._red_drop = 0
            else:
                self._red_drop += 1
                return self.ball_on or "red"
        elif self.ball_on == "color" and colors < (self._colors or 0):
            # 颜色在进红后的当前杆被打掉（短暂未复位的画面）。
            self.ball_on = "red" if reds > 0 else "clear"
        elif reds == 0 and self.ball_on is None:
            self.ball_on = "clear"
        elif self.ball_on is None:
            self.ball_on = "red"
        self._reds, self._colors = reds, colors
        self._red_drop = 0
        return self.ball_on or "red"

    def cycle_manual(self, balls: Sequence) -> str:
        """手动切换红球/彩球目标，并同步当前画面的计数基线。

        视觉只能看到球是否还在台面上，不能知道上一杆是未进球、犯规还是
        成功入袋。同步 ``_reds``/``_colors`` 很重要：否则在首次识别完成前
        按键，下一次 ``update`` 会把刚切换的状态重新初始化掉。
        """
        if balls:
            reds = reds_remaining(balls)
            colors = sum(1 for b in balls if b.label in COLOR_ORDER)
        else:
            # 识别线程可能正处于稳定确认/换帧窗口，使用最近一次可靠计数，
            # 不让一次空场景把 Q 的即时切换错误地变成清彩阶段。
            reds = self._reds
            colors = self._colors or 0
        if reds == 0:
            self.ball_on = "clear"
            self._manual_ball_on = None
            self._manual_counts = None
            if balls:
                self._reds, self._colors = reds, colors
            return self.ball_on

        current = self.ball_on if self.ball_on in ("red", "color") else "red"
        self._manual_ball_on = "color" if current == "red" else "red"
        self._manual_counts = ((reds, colors) if balls or self._reds is not None
                               else None)
        self.ball_on = self._manual_ball_on
        if balls:
            self._reds, self._colors = reds, colors
        return self.ball_on

    def toggle_red_color(self, balls: Sequence) -> str:
        """Q 键入口：红球仍在时，在红球与红后选彩之间切换。

        无红球时不能回到红球目标，始终保持清彩阶段的严格顺序。
        ``cycle_manual`` 保留为旧 O 键的兼容入口。
        """
        return self.cycle_manual(balls)


def phase(balls: Sequence) -> str:
    """当前局面阶段：'red' 红球阶段 / 'color' 清彩阶段 / 'none' 无球。"""
    if any(b.label == "红球" for b in balls):
        return "red"
    if any(b.label in COLOR_ORDER for b in balls):
        return "color"
    return "none"


def reds_remaining(balls: Sequence) -> int:
    return sum(1 for b in balls if b.label == "红球")


def next_color(balls: Sequence) -> Optional[object]:
    """清彩阶段下一颗该打的彩球（按分值顺序的第一颗在场彩球）。"""
    for label in COLOR_ORDER:
        for b in balls:
            if b.label == label:
                return b
    return None


def _others(balls: Sequence, cue, target) -> List[physics.Point]:
    return [b.pos for b in balls if b is not cue and b is not target]


def opening_break_target(balls: Sequence, cue, r: float) -> Optional[object]:
    """在完整、紧凑的红球架上选朝向母球一侧的外层红球。

    入袋规划器会正确地拒绝球架内部的目标球，因为目标球到袋口的轨迹
    会穿过相邻红球。开局第一杆的目的通常是解球/撞散球架，不应因此让
    画面完全没有击球点；该回退只接受接近标准三角尺寸的密集球群，并且
    由母球方向选择最外层红球，避免对散开局面做未经验证的推荐。
    """
    reds = [b for b in balls if b.label == "红球"]
    if len(reds) < 14 or r <= 0:
        return None
    points = np.asarray([b.pos for b in reds], dtype=float)
    span_x = float(points[:, 0].max() - points[:, 0].min())
    span_y = float(points[:, 1].max() - points[:, 1].min())
    # The standard triangle spans 8r in its long direction and sqrt(3)*4r
    # in its short direction, so the latter is slightly below 7r.
    if not (6.5 * r <= span_x <= 11.5 * r
            and 5.8 * r <= span_y <= 11.5 * r):
        return None
    nearest = []
    for i, point in enumerate(points):
        distances = np.hypot(points[:, 0] - point[0], points[:, 1] - point[1])
        distances[i] = np.inf
        nearest.append(float(distances.min()))
    if float(np.median(nearest)) > 2.45 * r:
        return None

    cue_pos = getattr(cue, "pos", cue)
    center = (float(points[:, 0].mean()), float(points[:, 1].mean()))
    approach = physics.normalize(physics.sub(cue_pos, center))
    if approach is None:
        return None
    return max(
        reds,
        key=lambda b: (
            physics.dot(physics.sub(b.pos, center), approach),
            -physics.dist(b.pos, cue_pos),
        ),
    )


def _plan(cue, target, pockets: Sequence[physics.Point], r: float,
          w: float, h: float, others: Sequence[physics.Point], cfg):
    """Apply the same calibrated cushion constraints to every rule branch."""
    cue_r = float(getattr(cue, "radius", r) or r)
    target_r = float(getattr(target, "radius", r) or r)
    ghost_offset = (
        float(getattr(cfg, "aim_offset_x", 0.0)),
        float(getattr(cfg, "aim_offset_y", 0.0)),
    )
    return physics.plan_shots(
        cue.pos, target.pos, pockets, r, w, h, others,
        cfg.allow_kicks, cfg.max_kicks,
        rail_inset=max(0.0, float(getattr(cfg, "rail_inset_ratio", 1.0)) * r),
        pocket_clearance=1.35 * r,
        cue_radius=cue_r,
        target_radius=target_r,
        ghost_offset=ghost_offset,
    )


def target_shot_key(shot: physics.Shot) -> Tuple[float, float, int, float]:
    """自动选球的优先级：切角、目标到袋距离、库数、总路程。

    ``plan_shots`` 已经会过滤被挡路线；这里再次检查 ``blocked``，让决策
    层即使收到测试替身或其它规划器的结果，也绝不会把有障碍的路线推荐给
    用户。切角和目标到袋距离严格排在总路程之前，符合实战选球顺序。
    """
    if not shot.valid or shot.blocked:
        return (float("inf"), float("inf"), 999, float("inf"))
    return (
        float(shot.cut_deg),
        float(shot.target_to_pocket),
        len(shot.bounce_points),
        float(shot.total),
    )


def rank_target_shots(plans: Sequence[physics.Shot]) -> List[physics.Shot]:
    """返回没有障碍的路线，并按自动选球规则排序。"""
    clear = [s for s in plans if s.valid and not s.blocked]
    return sorted(clear, key=target_shot_key)


def best_target_shot(plans: Sequence[physics.Shot]) -> Optional[physics.Shot]:
    """取一颗目标球对应的最佳无障碍路线。"""
    ranked = rank_target_shots(plans)
    return ranked[0] if ranked else None


def choose_target(balls: Sequence, cue, pockets: Sequence[physics.Point],
                  r: float, w: float, h: float, cfg,
                  prefer: Optional[physics.Point] = None,
                  ball_on: Optional[str] = None) -> Tuple[Optional[object], str, str]:
    """选择目标球。

    返回 (target_ball, phase, 说明文字)；target_ball 为 None 表示该阶段无可行目标。
    红球阶段：逐颗红球生成方案，只保留无障碍路线，按「切角小 → 目标离袋近
    → 库数少 → 总路程短」选择；红后任选彩球阶段仍按这套自动优先级，红球
    清完后的清彩阶段才按分值顺序只尝试当前最低分值的在场彩球。

    prefer：上一帧选定的目标球位置（台面坐标），仅作为完全同级候选的
    最后平局条件，不能压过新的切角/袋口优先级。
    """
    selected_on = ball_on or ("red" if phase(balls) == "red" else "clear")
    if selected_on == "red":
        reds = [b for b in balls if b.label == "红球"]
        if not reds:
            # 显式状态可能比检测帧滞后一帧；不要让目标记忆分支对空列表
            # 调用 min() 崩溃，退回到清彩阶段的规则判断。
            selected_on = "clear" if not any(
                b.label in COLOR_ORDER for b in balls) else "color"
        else:
            best: Optional[Tuple[Tuple[float, ...], object]] = None
            for tb in reds:
                plans = _plan(cue, tb, pockets, r, w, h,
                              _others(balls, cue, tb), cfg)
                shot = best_target_shot(plans)
                if shot is None:
                    continue
                key = target_shot_key(shot)
                if prefer is not None:
                    # 只在前面的规则完全打平时保持上一颗目标，减少识别
                    # 噪声造成的跳变；距离放在规则键之后，不改变优先级。
                    key = key + (physics.dist(tb.pos, prefer),)
                else:
                    key = key + (float(tb.pos[0]), float(tb.pos[1]))
                if best is None or key < best[0]:
                    best = (key, tb)
            if best is not None:
                n = reds_remaining(balls)
                return best[1], "red", f"红球阶段（剩 {n} 颗红球）：打红球"
            return None, "red", "红球阶段：所有红球暂无可行方案"
    if selected_on == "color":
        # 红球仍在时，红后彩球是任选目标；按自动选球优先级在所有
        # 彩球中挑选。红球清完后会由 clear 状态进入严格清彩顺序。
        colors = [b for b in balls if b.label in COLOR_ORDER]
        best: Optional[Tuple[Tuple[float, ...], object]] = None
        for tb in colors:
            plans = _plan(cue, tb, pockets, r, w, h,
                          _others(balls, cue, tb), cfg)
            shot = best_target_shot(plans)
            if shot is None:
                continue
            key = target_shot_key(shot)
            if prefer is not None:
                key = key + (physics.dist(tb.pos, prefer),)
            else:
                key = key + (float(tb.pos[0]), float(tb.pos[1]))
            if best is None or key < best[0]:
                best = (key, tb)
        if best is not None:
            return best[1], "color", f"红后任选彩球：打{best[1].label}"
        return None, "color", "红后任选彩球：所有彩球暂无可行方案"

    # 清彩阶段必须严格按分值顺序：只考虑「下一颗该打的彩球」。
    # 原实现会跳过无方案的黄球直接打绿球——实战中这是犯规送分。
    tb = next_color(balls)
    if tb is None:
        return None, "color", "清彩阶段：场上无彩球"
    plans = _plan(cue, tb, pockets, r, w, h,
                  _others(balls, cue, tb), cfg)
    v = COLOR_VALUE[tb.label]
    if best_target_shot(plans) is not None:
        return tb, "color", f"清彩阶段：打{tb.label}（{v} 分）"
    return None, "color", f"清彩阶段：{tb.label}（{v} 分）暂无可行方案（被挡或切角过大）"
