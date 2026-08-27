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

    def reset(self) -> None:
        self.ball_on = None
        self._reds = None
        self._colors = None

    def update(self, balls: Sequence, stable: bool) -> str:
        reds = reds_remaining(balls)
        colors = sum(1 for b in balls if b.label in COLOR_ORDER)
        if not stable:
            return self.ball_on or ("clear" if reds == 0 else "red")
        if self._reds is None:
            self.ball_on = "clear" if reds == 0 else "red"
        elif reds < self._reds:
            # 红球数减少（包括最后一颗红球）后，下一杆都必须先选彩球。
            # 不能在 reds==0 时直接进入 clear，否则会跳过最后一颗红后的
            # 任选彩球阶段。
            self.ball_on = "color"
        elif self.ball_on == "color" and colors < (self._colors or 0):
            # 颜色在进红后的当前杆被打掉（短暂未复位的画面）。
            self.ball_on = "red" if reds > 0 else "clear"
        elif reds == 0 and self.ball_on is None:
            self.ball_on = "clear"
        elif self.ball_on is None:
            self.ball_on = "red"
        self._reds, self._colors = reds, colors
        return self.ball_on or "red"

    def cycle_manual(self, balls: Sequence) -> str:
        """处理无法由视觉确定的失误/换手情况。"""
        if reds_remaining(balls) == 0:
            self.ball_on = "clear"
        elif self.ball_on == "red":
            self.ball_on = "color"
        else:
            self.ball_on = "red"
        return self.ball_on


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


def choose_target(balls: Sequence, cue, pockets: Sequence[physics.Point],
                  r: float, w: float, h: float, cfg,
                  prefer: Optional[physics.Point] = None,
                  ball_on: Optional[str] = None) -> Tuple[Optional[object], str, str]:
    """选择目标球。

    返回 (target_ball, phase, 说明文字)；target_ball 为 None 表示该阶段无可行目标。
    红球阶段：逐颗红球生成击球方案，选「未挡优先 + 总路程短」的最优；
    清彩阶段：按分值顺序找第一颗有可行方案的在场彩球。

    prefer：上一帧选定的目标球位置（台面坐标）。若该球仍在场且有可行方案，
    则沿用（避免 15 颗红球时目标在帧间跳变导致瞄准线乱动）。
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
            # 目标记忆：上一帧选的红球仍可打则沿用
            if prefer is not None:
                cand = min(reds, key=lambda b: (b.pos[0] - prefer[0]) ** 2 + (b.pos[1] - prefer[1]) ** 2)
                if physics.dist(cand.pos, prefer) < 2.0 * r:
                    plans = _plan(cue, cand, pockets, r, w, h,
                                  _others(balls, cue, cand), cfg)
                    if plans:
                        n = reds_remaining(balls)
                        return cand, "red", f"红球阶段（剩 {n} 颗红球）：打红球"
            best: Optional[Tuple[Tuple[float, float], object]] = None
            for tb in reds:
                plans = _plan(cue, tb, pockets, r, w, h,
                              _others(balls, cue, tb), cfg)
                if not plans:
                    continue
                s = physics.best_shot(plans, w)
                if s is None:
                    continue
                # 目标球选择也使用路线风险分数：自动模式不会为了几厘米
                # 的短路程去选极薄切或高风险多库球。
                key = (physics.route_score(s, w), s.total)
                if best is None or key < best[0]:
                    best = (key, tb)
            if best is not None:
                n = reds_remaining(balls)
                return best[1], "red", f"红球阶段（剩 {n} 颗红球）：打红球"
            return None, "red", "红球阶段：所有红球暂无可行方案"
    if selected_on == "color":
        colors = [b for b in balls if b.label in COLOR_ORDER]
        best: Optional[Tuple[Tuple[float, float], object]] = None
        for tb in colors:
            plans = _plan(cue, tb, pockets, r, w, h,
                          _others(balls, cue, tb), cfg)
            if not plans:
                continue
            shot = physics.best_shot(plans, w)
            if shot is None:
                continue
            key = (physics.route_score(shot, w), shot.total)
            if best is None or key < best[0]:
                best = (key, tb)
        if best is not None:
            return best[1], "color", f"红球后选彩球：打{best[1].label}"
        return None, "color", "红球后选彩球：所有彩球暂无可行方案"

    # 清彩阶段必须严格按分值顺序：只考虑「下一颗该打的彩球」。
    # 原实现会跳过无方案的黄球直接打绿球——实战中这是犯规送分。
    tb = next_color(balls)
    if tb is None:
        return None, "color", "清彩阶段：场上无彩球"
    plans = _plan(cue, tb, pockets, r, w, h,
                  _others(balls, cue, tb), cfg)
    v = COLOR_VALUE[tb.label]
    if plans:
        return tb, "color", f"清彩阶段：打{tb.label}（{v} 分）"
    return None, "color", f"清彩阶段：{tb.label}（{v} 分）暂无可行方案（被挡或切角过大）"
