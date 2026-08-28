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
    """红/彩目标策略（v3.7 起人工驱动，不再猜状态）。

    视觉无法可靠区分「进红 / 没进球 / 犯规换手」，此前基于球数事件的
    状态推断连出一串误判。新策略把歧义交给用户一键控制：

    * 无人工干预：一律瞄红球；
    * 红球清完：自动进入严格清彩顺序（黄→绿→棕→蓝→粉→黑，硬规则）；
    * Q 键 = 一次性脉冲：下一杆改瞄彩球（红后任选，自动挑最佳彩球），
      这一杆打完（球动过又恢复稳定）自动回落瞄红球。

    时序细节：打进红球后球还在滚时按 Q 是最常见的按法，脉冲必须
    存活到「下一杆」结束 —— 用 _pulse_pending 区分「按下 Q 时球
    正在动」：先等当前这杆结束（不清脉冲），再等脉冲杆结束才回落。
    """

    ball_on: Optional[str] = None      # red / color / clear
    _color_pulse: bool = False         # Q 脉冲：下一杆瞄彩球
    _pulse_pending: bool = False       # 按下 Q 时球正在动：先等这杆结束
    _pending_shot: bool = False        # 击杆周期：球动过 = 打过一杆

    def reset(self) -> None:
        self.ball_on = None
        self._color_pulse = False
        self._pulse_pending = False
        self._pending_shot = False

    def update(self, balls: Sequence, stable: bool) -> str:
        reds = reds_remaining(balls)
        if not stable:
            self._pending_shot = True
            return self.ball_on or "red"
        if self._pending_shot:
            # 一杆结束（MOVING/STABILIZING → READY）。
            self._pending_shot = False
            if self._pulse_pending:
                # 这是「按下 Q 时正在滚的那杆」的结束：脉冲才刚生效，
                # 不在本杆回落。
                self._pulse_pending = False
            else:
                self._color_pulse = False   # 脉冲杆打完，回落瞄红球
        if reds == 0:
            # 红球清完默认严格清彩（硬规则）；Q 脉冲可覆盖一杆——
            # 真实规则里「最后一颗红后那一杆」本就是任选彩球。
            self.ball_on = "color" if self._color_pulse else "clear"
        elif self._color_pulse:
            self.ball_on = "color"
        else:
            self.ball_on = "red"
        return self.ball_on

    def pulse_color(self, balls: Sequence) -> str:
        """Q 键入口：下一杆改瞄彩球（任选）；打完该杆自动回落。

        红球还在：覆盖默认的红球目标（打进红后的那一杆）。
        红球已清：覆盖清彩顺序一杆 —— 真实规则里「最后一颗红后
        的那一杆」本就是任选彩球，此后恢复严格顺序。
        """
        self._color_pulse = True
        # 按下时球正在滚（刚打进红）→ 等这杆结束才算脉冲杆开始。
        self._pulse_pending = self._pending_shot
        self.ball_on = "color"
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
            import copy
            use_kicks = bool(getattr(cfg, "allow_kicks", True))
            cfg_fast = copy.copy(cfg)
            cfg_fast.allow_kicks = False if use_kicks else cfg.allow_kicks

            best: Optional[Tuple[Tuple[float, ...], object]] = None
            # 第一阶段：极速扫描所有红球的直球路线（纯直球耗时 <1ms）
            for tb in reds:
                plans = _plan(cue, tb, pockets, r, w, h,
                              _others(balls, cue, tb), cfg_fast)
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

            # 第二阶段：仅当全部红球均无直球可行路线且开启库边时，才回退到全量库边反弹扫描
            if best is None and use_kicks:
                for tb in reds:
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
                n = reds_remaining(balls)
                return best[1], "red", f"红球阶段（剩 {n} 颗红球）：打红球"

            # 如果是完整开局紧凑红球三角架，交给 opening_break 专门处理开局解球碰撞点
            if opening_break_target(balls, cue, r) is not None:
                return None, "red", "红球阶段：开局球架"

            return None, "red", "红球阶段：所有红球暂无可行方案"
    if selected_on == "color":
        # 红球仍在时，红后彩球是任选目标；按自动选球优先级在所有
        # 彩球中挑选。红球清完后会由 clear 状态进入严格清彩顺序。
        colors = [b for b in balls if b.label in COLOR_ORDER]
        import copy
        use_kicks = bool(getattr(cfg, "allow_kicks", True))
        cfg_fast = copy.copy(cfg)
        cfg_fast.allow_kicks = False if use_kicks else cfg.allow_kicks

        best: Optional[Tuple[Tuple[float, ...], object]] = None
        # 第一阶段：极速扫描所有彩球直球
        for tb in colors:
            plans = _plan(cue, tb, pockets, r, w, h,
                          _others(balls, cue, tb), cfg_fast)
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

        # 第二阶段：无直球时回退库边
        if best is None and use_kicks:
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

    # 清彩阶段必须严格按分值顺序：绝不允许跳过低分球打高分球（实战犯规送分）。
    tb = next_color(balls)
    if tb is None:
        return None, "color", "清彩阶段：场上无彩球"
    v = COLOR_VALUE[tb.label]
    plans = _plan(cue, tb, pockets, r, w, h,
                  _others(balls, cue, tb), cfg)
    if best_target_shot(plans) is not None:
        return tb, "color", f"清彩阶段：打{tb.label}（{v} 分）"

    # 若直球没有通畅线路，尝试库边解围
    if not cfg.allow_kicks:
        import copy
        cfg_kicks = copy.copy(cfg)
        cfg_kicks.allow_kicks = True
        p_kicks = _plan(cue, tb, pockets, r, w, h,
                        _others(balls, cue, tb), cfg_kicks)
        if best_target_shot(p_kicks) is not None:
            return tb, "color", f"清彩阶段：打{tb.label}（{v} 分·库边解围）"

    # 目标球仍严格锁定为下一颗法定彩球（绝不跳顺序瞄准高分球）
    return tb, "color", f"清彩阶段：{tb.label}（{v} 分）暂无安全进袋线路，建议做球/防守"
