"""斯诺克决策层：目标球选择 + 局面说明。

斯诺克规则（区别于八球"打最近的球"）：
  * 红球阶段：台面上还有红球时，目标是「某颗红球」（选最好进且不被挡的）；
  * 清彩阶段：红球清完后，必须按分值顺序打彩球：黄(2)→绿(3)→棕(4)→蓝(5)→粉(6)→黑(7)。
    清彩顺序是硬规则，任何滞留/过期的上游状态都不允许把目标带离
    「下一颗最低分彩球」；最后一颗红后想挑其他彩球，用 G 键手动点选。

只依赖识别出的球列表 + 标准 6 袋 + 台面尺寸，不依赖彩球点位知识
（彩球复位后位置由每帧全量识别自动跟踪）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar, Dict, List, Optional, Sequence, Tuple

import numpy as np

from aimtool import physics

# 清彩顺序（分值升序）
COLOR_ORDER: Tuple[str, ...] = ("黄球", "绿球", "棕球", "蓝球", "粉球", "黑球")
COLOR_VALUE = {"黄球": 2, "绿球": 3, "棕球": 4, "蓝球": 5, "粉球": 6, "黑球": 7}
VALUE_LABEL = {v: k for k, v in COLOR_VALUE.items()}


@dataclass
class TurnTracker:
    """红/彩目标策略（v3.9 起：位移驱动脉冲 + 清彩恒严格）。

    视觉无法可靠区分「进红 / 没进球 / 犯规换手」，基于球数事件的
    状态推断早已废除，目标策略由用户一键控制：

    * 无人工干预：一律瞄红球；
    * 红球清完：无条件严格清彩（黄→绿→棕→蓝→粉→黑，硬规则），
      Q 在此阶段无效——最后一颗红后想挑其他彩球，用 G 键手动点选；
    * Q 键 = 一次性脉冲（仅红球在场时有效）：下一杆改瞄彩球
      （红后任选，自动挑最佳彩球），该杆打完自动回落瞄红球；
    * W 键 = 强制保持红球目标，直到 Q、红球清零或重置。

    脉冲的「一杆」由球位位移判定，不再依赖帧稳定信号：
    逐次更新对同色球做最近邻匹配，最大位移超过 _SHOT_MOTION_UNITS
    记为击球开始，连续 _STILL_UPDATES 次更新无位移记为击杆结束。
    遮挡（occluded=True）或整帧零检测时冻结全部计数——因此漏看
    几帧、UI 覆盖台面、检测抖动都不会让「任选彩球」泄漏到之后的杆，
    也不会提前失效（旧版脉冲依赖「恰好被观察到的动→停稳定信号」，
    击球过程漏看时脉冲永久卡死在任选态，清彩阶段会直奔黑球）。

    时序细节：打进红球后球还在滚时按 Q 是最常见的按法，脉冲必须
    存活到「下一杆」结束 —— 按下 Q 时若已有击球在进行（或最近
    一次更新画面未就绪），置 _pulse_guard，当前这杆结束不消耗脉冲。
    """

    # 中位数平滑后的静止球位抖动 <1 个台面单位，真实击球至少移动
    # 数十个单位；8（≈0.36 球半径）离两侧都有充足距离。
    _SHOT_MOTION_UNITS: ClassVar[float] = 8.0
    # 连续多少次更新无位移视为一杆结束（30fps 分析下约 67ms 静止）。
    _STILL_UPDATES: ClassVar[int] = 2

    ball_on: Optional[str] = None      # red / color / clear
    _color_pulse: bool = False         # Q 脉冲：下一杆瞄彩球
    _pulse_guard: bool = False         # 按下 Q 时正在进行的一杆不消耗脉冲
    _force_red: bool = False           # W 强制保持红球目标，直到 Q、红球清零或重置
    _shot_active: bool = False         # 位移判定：一杆进行中
    _still_updates: int = 0            # 连续无位移更新计数
    _ready_hint: bool = True           # 最近一次 update 的画面就绪信号（仅 Q 时序用）
    _prev_points: Optional[Dict[str, List[Tuple[float, float]]]] = None

    def reset(self) -> None:
        self.ball_on = None
        self._color_pulse = False
        self._pulse_guard = False
        self._force_red = False
        self._shot_active = False
        self._still_updates = 0
        self._ready_hint = True
        self._prev_points = None

    @staticmethod
    def _default_on(reds: int) -> str:
        return "clear" if reds <= 0 else "red"

    @staticmethod
    def _points_by_label(balls: Sequence) -> Dict[str, List[Tuple[float, float]]]:
        groups: Dict[str, List[Tuple[float, float]]] = {}
        for b in balls:
            pos = getattr(b, "pos", None)
            if pos is None:
                continue
            groups.setdefault(b.label, []).append((float(pos[0]), float(pos[1])))
        return groups

    @staticmethod
    def _max_displacement(prev: Dict[str, List[Tuple[float, float]]],
                          cur: Dict[str, List[Tuple[float, float]]]) -> float:
        """两帧球位的最大位移（同色最近邻匹配；红球互为等价）。

        新进/消失的球不参与位移（落袋由母球等其他在场球的位移代为
        体现），因此单帧误检、丢球不会被误判为一杆。
        """
        worst = 0.0
        for label, points in cur.items():
            rest = list(prev.get(label) or [])
            for x, y in points:
                if not rest:
                    break
                i = min(range(len(rest)),
                        key=lambda k: (rest[k][0] - x) ** 2 + (rest[k][1] - y) ** 2)
                px, py = rest.pop(i)
                d = math.hypot(px - x, py - y)
                if d > worst:
                    worst = d
        return worst

    def update(self, balls: Sequence, stable: bool = True,
               occluded: bool = False) -> str:
        """每帧调用一次，返回 red / color / clear。

        stable 仅是「按 Q 时画面是否可能在滚」的时序辅助信号，不再
        决定脉冲生死；occluded=True（UI 覆盖台面）或整帧零检测时
        冻结所有计数（不在没有事实依据时消耗脉冲或判定击杆）。
        """
        self._ready_hint = bool(stable)
        reds = reds_remaining(balls)
        if occluded:
            return self.ball_on or self._default_on(reds)
        cur = self._points_by_label(balls)
        if not cur:
            # 零信息帧（检测全丢/全遮挡）：禁止任何决策翻转。
            return self.ball_on or self._default_on(reds)
        if self._prev_points is not None:
            if self._max_displacement(self._prev_points, cur) > self._SHOT_MOTION_UNITS:
                self._shot_active = True
                self._still_updates = 0
            else:
                self._still_updates += 1
                if self._shot_active and self._still_updates >= self._STILL_UPDATES:
                    self._shot_active = False    # 一杆结束
                    self._still_updates = 0
                    if self._pulse_guard:
                        # 按下 Q 时正在滚的那一杆：脉冲刚生效，不回落
                        self._pulse_guard = False
                    elif self._color_pulse:
                        self._color_pulse = False    # 脉冲杆打完，回落瞄红球
        self._prev_points = cur

        if reds <= 0:
            # 红球清完：无条件严格清彩（硬规则）；任何滞留的脉冲 /
            # 强制红球状态都不允许把目标带离「下一颗最低分彩球」。
            self._color_pulse = False
            self._pulse_guard = False
            self._force_red = False
            self.ball_on = "clear"
        elif self._force_red:
            self.ball_on = "red"
        elif self._color_pulse:
            self.ball_on = "color"
        else:
            self.ball_on = "red"
        return self.ball_on

    def pulse_color(self, balls: Sequence) -> str:
        """Q 键入口：下一杆改瞄彩球（任选）；打完该杆自动回落。

        仅红球在场时生效。红球清完后一律严格清彩，Q 不再改变目标
        （最后一颗红后想挑其他彩球，直接用 G 键点选）；还什么都没
        检测到的空帧允许先押注，下一帧真实球况会自动校正（无红球
        则立即回到严格清彩）。
        """
        balls = list(balls)
        if balls and reds_remaining(balls) <= 0:
            self._color_pulse = False
            self._pulse_guard = False
            self.ball_on = "clear"
            return self.ball_on
        self._force_red = False
        self._color_pulse = True
        # 按下时球正在滚（刚打进红）→ 当前这杆不消耗脉冲。
        self._pulse_guard = self._shot_active or not self._ready_hint
        self.ball_on = "color"
        return self.ball_on

    def pulse_red(self) -> str:
        """W键入口：强制切回红球，取消彩球脉冲（红球清完后由 update 自动回到清彩）。"""
        self._color_pulse = False
        self._pulse_guard = False
        self._force_red = True
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
    aim_half = (float(getattr(cfg, "pocket_accept_ratio", 1.45)) * r
                if getattr(cfg, "pocket_aim_optimize", True) else 0.0)
    return physics.plan_shots(
        cue.pos, target.pos, pockets, r, w, h, others,
        cfg.allow_kicks, cfg.max_kicks,
        rail_inset=max(0.0, float(getattr(cfg, "rail_inset_ratio", 1.0)) * r),
        pocket_clearance=1.35 * r,
        cue_radius=cue_r,
        target_radius=target_r,
        ghost_offset=ghost_offset,
        pocket_aim_half=aim_half,
    )


def target_shot_key(shot: physics.Shot, w: Optional[float] = None,
                    h: Optional[float] = None, r: Optional[float] = None,
                    cfg=None) -> Tuple[float, ...]:
    """自动选球的优先级（v3.10）：进球成功率 → 切角 → 距袋 → 库数 → 总路程。

    成功率由 physics.pot_success_prob 估计（袋口角余量 × 出球方向误差 ×
    切角/库数惩罚），把「远袋小切」和「近袋稍大切」的真实难度区分开。
    未传几何参数/成功率不可算时回退旧规则（切角优先，首项 0.0 恒定）。
    ``plan_shots`` 已经会过滤被挡路线；这里再次检查 ``blocked``，让决策
    层即使收到测试替身或其它规划器的结果，也绝不会把有障碍的路线推荐给
    用户。
    """
    if not shot.valid or shot.blocked:
        return (float("inf"),) * 5
    tail = (float(shot.cut_deg), float(shot.target_to_pocket),
            float(len(shot.bounce_points)), float(shot.total))
    if (cfg is not None and getattr(cfg, "rank_by_success", True)
            and w and h and r):
        p = physics.pot_success_prob(shot, float(w), float(h), float(r), cfg)
        if p is not None:
            return (-p,) + tail
    return (0.0,) + tail


def rank_target_shots(plans: Sequence[physics.Shot], w: Optional[float] = None,
                      h: Optional[float] = None, r: Optional[float] = None,
                      cfg=None) -> List[physics.Shot]:
    """返回没有障碍的路线，并按自动选球规则排序。"""
    clear = [s for s in plans if s.valid and not s.blocked]
    return sorted(clear, key=lambda s: target_shot_key(s, w, h, r, cfg))


def best_target_shot(plans: Sequence[physics.Shot], w: Optional[float] = None,
                     h: Optional[float] = None, r: Optional[float] = None,
                     cfg=None) -> Optional[physics.Shot]:
    """取一颗目标球对应的最佳无障碍路线。"""
    ranked = rank_target_shots(plans, w, h, r, cfg)
    return ranked[0] if ranked else None


def choose_target(balls: Sequence, cue, pockets: Sequence[physics.Point],
                  r: float, w: float, h: float, cfg,
                  prefer: Optional[physics.Point] = None,
                  ball_on: Optional[str] = None) -> Tuple[Optional[object], str, str]:
    """选择目标球。

    返回 (target_ball, phase, 说明文字)；target_ball 为 None 表示该阶段无可行目标。
    红球阶段：逐颗红球生成方案，只保留无障碍路线，按「进球成功率 → 切角小
    → 目标离袋近 → 库数少 → 总路程短」选择（rank_by_success=False 回退旧的
    切角优先）；红后任选彩球（Q 脉冲，仅红球仍在场时合法）
    仍按这套自动优先级；红球清完后的清彩阶段只尝试分值顺序上当前最低的
    在场彩球，绝不跳序。

    prefer：上一帧选定的目标球位置（台面坐标），仅作为完全同级候选的
    最后平局条件，不能压过新的切角/袋口优先级。
    """
    selected_on = ball_on or ("red" if phase(balls) == "red" else "clear")
    if selected_on == "color" and not any(b.label == "红球" for b in balls):
        # 结构性保险：任选彩球只在红球仍在场时合法（红后一杆语义）。
        # 红球清完后，不管上游状态如何过期/异常，一律落入严格清彩。
        selected_on = "clear"
    if selected_on == "red":
        reds = [b for b in balls if b.label == "红球"]
        if not reds:
            # 显式状态可能比检测帧滞后一帧；红球已不在场时落入严格清彩
            # （旧版会错误地降级到「任选彩球」，滞后帧可能直接瞄准黑球），
            # 同时避免目标记忆分支对空列表调用 min() 崩溃。
            selected_on = "clear"
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
                shot = best_target_shot(plans, w, h, r, cfg)
                if shot is None:
                    continue
                key = target_shot_key(shot, w, h, r, cfg)
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
                    shot = best_target_shot(plans, w, h, r, cfg)
                    if shot is None:
                        continue
                    key = target_shot_key(shot, w, h, r, cfg)
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
        # 红球仍在场、按 Q 后的一杆：任选彩球（红后规则），仍按自动
        # 选球优先级在所有彩球中挑选。红球清完的状态在上面已被强制
        # 收敛到严格清彩，不可能进入本分支。
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
            shot = best_target_shot(plans, w, h, r, cfg)
            if shot is None:
                continue
            key = target_shot_key(shot, w, h, r, cfg)
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
                shot = best_target_shot(plans, w, h, r, cfg)
                if shot is None:
                    continue
                key = target_shot_key(shot, w, h, r, cfg)
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
    if best_target_shot(plans, w, h, r, cfg) is not None:
        return tb, "color", f"清彩阶段：打{tb.label}（{v} 分）"

    # 若直球没有通畅线路，尝试库边解围
    if not cfg.allow_kicks:
        import copy
        cfg_kicks = copy.copy(cfg)
        cfg_kicks.allow_kicks = True
        p_kicks = _plan(cue, tb, pockets, r, w, h,
                        _others(balls, cue, tb), cfg_kicks)
        if best_target_shot(p_kicks, w, h, r, cfg) is not None:
            return tb, "color", f"清彩阶段：打{tb.label}（{v} 分·库边解围）"

    # 目标球仍严格锁定为下一颗法定彩球（绝不跳顺序瞄准高分球）
    return tb, "color", f"清彩阶段：{tb.label}（{v} 分）暂无安全进袋线路，建议做球/防守"
