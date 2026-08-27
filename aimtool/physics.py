"""2D 桌球瞄准几何（纯 Python，无第三方依赖）。

坐标系约定：台面标准坐标 —— 原点在左上角，宽 W、高 H（像素或任意长度单位，
比例一致即可）。袋口为点，球为圆（半径 r，全台统一）。

包含：
  * 鬼球瞄准法（ghost ball）：母球中心应到达的位置、出发方向、切角；
  * 库边反弹：一库/两库解围（unfolding + 实空间仿真校验），返回真实反弹点；
  * 障碍判定：瞄准路径是否被其他球挡住；
  * 力度建议：根据总路程给出 0-100 的推荐力度。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

Point = Tuple[float, float]

# ---------- 基础向量工具 ----------

def sub(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1])


def add(a: Point, b: Point) -> Point:
    return (a[0] + b[0], a[1] + b[1])


def mul(a: Point, s: float) -> Point:
    return (a[0] * s, a[1] * s)


def dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1]


def norm(a: Point) -> float:
    return math.hypot(a[0], a[1])


def dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def normalize(a: Point) -> Optional[Point]:
    n = norm(a)
    if n < 1e-9:
        return None
    return (a[0] / n, a[1] / n)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def seg_point_dist(p: Point, a: Point, b: Point) -> float:
    """点 p 到线段 ab 的最短距离。"""
    ab = sub(b, a)
    ab2 = dot(ab, ab)
    if ab2 < 1e-12:
        return dist(p, a)
    t = clamp(dot(sub(p, a), ab) / ab2, 0.0, 1.0)
    proj = add(a, mul(ab, t))
    return dist(p, proj)


# ---------- 台面/库边 ----------

RAILS = ("top", "bottom", "left", "right")
# 库边线：轴 + 坐标。left/right 为竖直（x=c），top/bottom 为水平（y=c）。
RAIL_LINE = {
    "top": ("y", 0.0),
    "bottom": ("y", 1.0),
    "left": ("x", 0.0),
    "right": ("x", 1.0),
}


def _rail_coordinate(rail: str, w: float, h: float, inset: float = 0.0) -> Tuple[str, float]:
    """库边对应的球心反弹线；inset=0 保持旧几何兼容。"""
    inset = max(0.0, min(float(inset), 0.45 * min(w, h)))
    axis, side = RAIL_LINE[rail]
    if axis == "x":
        return axis, inset if side == 0.0 else w - inset
    return axis, inset if side == 0.0 else h - inset


def reflect_across(p: Point, rail: str, w: float, h: float,
                   inset: float = 0.0) -> Point:
    """把点 p 关于库边 rail 镜像（用于 unfolding）。"""
    axis, c = _rail_coordinate(rail, w, h, inset)
    if axis == "x":
        return (2.0 * c - p[0], p[1])
    return (p[0], 2.0 * c - p[1])


def ray_rail_t(p: Point, d: Point, rail: str, w: float, h: float,
               inset: float = 0.0) -> Optional[float]:
    """从 p 沿方向 d 的射线，到达库边 rail 的参数 t（>0 且有限）。"""
    axis, c = _rail_coordinate(rail, w, h, inset)
    if axis == "x":
        if abs(d[0]) < 1e-12:
            return None
        t = (c - p[0]) / d[0]
    else:
        if abs(d[1]) < 1e-12:
            return None
        t = (c - p[1]) / d[1]
    return t if t > 1e-9 else None


def reflect_dir(d: Point, rail: str) -> Point:
    """速度方向关于库边反射（理想库边，无能量损失）。"""
    axis, _ = RAIL_LINE[rail]
    if axis == "x":
        return (-d[0], d[1])
    return (d[0], -d[1])


def rail_crossing(p: Point, d: Point, rail: str, w: float, h: float,
                  inset: float = 0.0) -> Optional[Point]:
    """射线 p+t*d 与库边 rail 的交点（若在射线正方向上）。"""
    t = ray_rail_t(p, d, rail, w, h, inset)
    if t is None:
        return None
    return (p[0] + t * d[0], p[1] + t * d[1])


# ---------- 鬼球瞄准 ----------

def _resolve_radii(r: float, cue_radius: Optional[float],
                   target_radius: Optional[float]) -> Tuple[float, float]:
    """Resolve optional per-ball radii while preserving the old API."""
    fallback = max(float(r), 1e-6)
    cue = fallback if cue_radius is None else max(float(cue_radius), 1e-6)
    target = (fallback if target_radius is None
              else max(float(target_radius), 1e-6))
    return cue, target


def ghost_pos(target: Point, pocket: Point, r: float,
              cue_radius: Optional[float] = None,
              target_radius: Optional[float] = None,
              offset: Point = (0.0, 0.0)) -> Optional[Point]:
    """鬼球位置：目标球沿「目标球→袋口」方向退两球半径之和。

    旧调用只传 ``r`` 时仍得到 ``target - 2r*d``。实机调用可传入母球和
    目标球各自的有效半径；``offset`` 是零默认的经验校准量，避免把检测
    球心偏差和游戏几何偏差混成同一个参数。

    target 与 pocket 重合（袋口精修吸附到贴库球等退化情况）时返回 None。
    """
    d = normalize(sub(pocket, target))
    if d is None:
        return None
    cue_r, target_r = _resolve_radii(r, cue_radius, target_radius)
    base = add(target, mul(d, -(cue_r + target_r)))
    return add(base, (float(offset[0]), float(offset[1])))


def impact_ghost(cue: Point, target: Point, r: float,
                 cue_radius: Optional[float] = None,
                 target_radius: Optional[float] = None,
                 offset: Point = (0.0, 0.0)) -> Optional[Point]:
    """母球沿当前连线正碰目标球时的鬼球位置。

    这和入袋用的 ``ghost_pos`` 不同：开局解球没有目标袋口，鬼球应位于
    目标球朝向母球的一侧。``offset`` 保持和入袋几何相同的经验校准语义。
    """
    d = normalize(sub(target, cue))
    if d is None:
        return None
    cue_r, target_r = _resolve_radii(r, cue_radius, target_radius)
    base = add(target, mul(d, -(cue_r + target_r)))
    return add(base, (float(offset[0]), float(offset[1])))


def contact_pos(target: Point, pocket: Point, r: float,
                target_radius: Optional[float] = None) -> Optional[Point]:
    """目标球与母球的实际接触点（目标球表面上的点）。"""
    d = normalize(sub(pocket, target))
    if d is None:
        return None
    _, target_r = _resolve_radii(r, None, target_radius)
    return add(target, mul(d, -target_r))


def impact_contact_pos(cue: Point, target: Point, r: float,
                       target_radius: Optional[float] = None) -> Optional[Point]:
    """正碰/解球时目标球朝向母球一侧的实际接触点。"""
    d = normalize(sub(target, cue))
    if d is None:
        return None
    _, target_r = _resolve_radii(r, None, target_radius)
    return add(target, mul(d, -target_r))


@dataclass
class Shot:
    """一次击球方案（可能为直球或库边解围）。"""
    pocket: Point
    ghost: Point
    aim_dir: Point                       # 母球出发方向（单位向量，屏幕/台面坐标）
    cue_to_contact: float                # 母球 → 接触点
    target_to_pocket: float              # 目标球 → 袋口
    total: float                         # 总路程
    cut_deg: float                       # 切角（度），0=正碰
    valid: bool
    blocked: bool = False
    blocked_by: Optional[str] = None
    bounce_points: List[Point] = field(default_factory=list)  # 库边反弹点（空=直球）
    rail_seq: Tuple[str, ...] = ()
    label: str = "直球"


def _cut_angle(approach: Point, target_dir: Point) -> float:
    cos = clamp(dot(approach, target_dir), -1.0, 1.0)
    return math.degrees(math.acos(cos))


MAX_CUT_DEG = 85.0
"""可行切角上限（度）。切角接近 90° 时母球传递给目标球沿袋口方向的
动量趋近 0，物理上不可能进球；超过上限的方案直接判不可行。"""


def _path_blocked(segments: Sequence[Tuple[Point, Point]],
                  others: Sequence[Point], r: float, tolerance: float = 1.0) -> Tuple[bool, Optional[str]]:
    """判断路径各段是否被 other 球阻挡（球心距线段 < 2r 视为挡）。"""
    for o in others:
        for a, b in segments:
            if seg_point_dist(o, a, b) < 2.0 * r * tolerance:
                return True, f"({o[0]:.0f},{o[1]:.0f})"
    return False, None


def direct_shot(cue: Point, target: Point, pocket: Point, r: float,
                others: Sequence[Point] = (),
                max_cut_deg: float = MAX_CUT_DEG,
                cue_radius: Optional[float] = None,
                target_radius: Optional[float] = None,
                ghost_offset: Point = (0.0, 0.0)) -> Shot:
    """直球：鬼球法。返回瞄准方案；不可打（切角超限等）时 valid=False。"""
    cue_r, target_r = _resolve_radii(r, cue_radius, target_radius)
    g = ghost_pos(target, pocket, r, cue_r, target_r, ghost_offset)
    if g is None:
        return Shot(pocket, target, (0.0, 0.0), 0.0, 0.0, 0.0, 0.0, False)
    d = normalize(sub(g, cue))
    if d is None:
        return Shot(pocket, g, (0.0, 0.0), 0.0, 0.0, 0.0, 0.0, False)
    tdir = normalize(sub(pocket, target))
    cut = _cut_angle(d, tdir)
    if cut > max_cut_deg:
        # 薄切不可行：接触时沿目标球→袋口方向的分速度≈0，进不了球
        return Shot(pocket, g, d, dist(cue, g), dist(target, pocket),
                    dist(cue, g) + dist(target, pocket), cut, False, label="直球")
    cue_to_contact = dist(cue, g)
    target_to_pocket = dist(target, pocket)
    # 母球只走 cue→ghost；目标球被击出后才走 target→pocket。
    # g→target 是两球接触几何，不是任何球心的运动路径，不能当作
    # 障碍段，否则 rack/贴球场景会被大量误判为被挡。
    collision_r = max(float(r), cue_r, target_r)
    blocked, by = _path_blocked([(cue, g), (target, pocket)], others, collision_r)
    return Shot(
        pocket=pocket, ghost=g, aim_dir=d,
        cue_to_contact=cue_to_contact, target_to_pocket=target_to_pocket,
        total=cue_to_contact + target_to_pocket, cut_deg=cut,
        valid=True, blocked=blocked, blocked_by=by, label="直球",
    )


def kick_shot(cue: Point, target: Point, pocket: Point, r: float,
               rails: Sequence[str], w: float, h: float,
               others: Sequence[Point] = (),
              max_cut_deg: float = MAX_CUT_DEG,
              rail_inset: float = 0.0,
              pockets: Sequence[Point] = (),
              pocket_clearance: float = 0.0,
              cue_radius: Optional[float] = None,
              target_radius: Optional[float] = None,
              ghost_offset: Point = (0.0, 0.0)) -> Shot:
    """库边解围：unfolding 求出发方向，再在实空间仿真验证反弹序列。

    rails 为期望依次触碰的库边，如 ("top",) 一库，("right", "bottom") 两库。
    仿真校验：实际反弹序列必须与 rails 完全一致，且最终到达真实鬼球附近。
    """
    cue_r, target_r = _resolve_radii(r, cue_radius, target_radius)
    g = ghost_pos(target, pocket, r, cue_r, target_r, ghost_offset)
    if g is None:
        return Shot(pocket, target, (0.0, 0.0), 0.0, 0.0, 0.0, 0.0, False, rail_seq=tuple(rails))
    # unfolding：把鬼球依次镜像到展开空间。
    # 必须按触碰顺序的「逆序」镜像（平行轴反射不可交换）：
    # 路径 cue→rail1→rail2→ghost 等价于直线打「先过 rail2 再过 rail1 镜像」的鬼球。
    gf = g
    for rail in reversed(rails):
        gf = reflect_across(gf, rail, w, h, rail_inset)
    d = normalize(sub(gf, cue))
    if d is None:
        return Shot(pocket, g, (0.0, 0.0), 0.0, 0.0, 0.0, 0.0, False, rail_seq=tuple(rails))

    # 实空间仿真：记录真实反弹点与反弹序列。
    # 每步先检查「当前射线是否穿过鬼球」——这是到达判定：原实现缺少该
    # 检查，最后一库后轨迹明明穿过鬼球，循环却继续记录多余反弹，
    # 导致 visited 序列永不匹配、解围永远判无效（P0）。
    pos = cue
    vel = d
    visited: List[str] = []
    bounce_points: List[Point] = []
    arrived = False
    for _ in range(len(rails) + 2):
        to_g = sub(g, pos)
        t_closest = dot(to_g, vel)          # vel 为单位向量：投影即最近点参数
        if t_closest > 0:
            closest = add(pos, mul(vel, t_closest))
            if dist(closest, g) < 0.6 * r:   # 射线穿过鬼球 → 到达
                arrived = True
                break
        best_t, best_rail = None, None
        for rail in RAILS:
            t = ray_rail_t(pos, vel, rail, w, h, rail_inset)
            if t is not None and (best_t is None or t < best_t):
                best_t, best_rail = t, rail
        if best_t is None:
            break
        hit = (pos[0] + best_t * vel[0], pos[1] + best_t * vel[1])
        # 反弹点必须落在库边线段范围内（略放宽）：落在角/袋口外=射线打到
        # 库边延长线，实际会先进袋，序列无效
        eps = 1e-6
        if not (rail_inset - eps <= hit[0] <= w - rail_inset + eps
                and rail_inset - eps <= hit[1] <= h - rail_inset + eps):
            break
        if pocket_clearance > 0 and any(dist(hit, p) < pocket_clearance for p in pockets):
            break
        visited.append(best_rail)
        bounce_points.append(hit)
        pos = hit
        vel = reflect_dir(vel, best_rail)
    # 反弹序列必须精确匹配（不多不少），且确实穿过鬼球
    seq_ok = tuple(visited) == tuple(rails)
    if not seq_ok or not arrived or len(bounce_points) != len(rails):
        return Shot(pocket, g, (0.0, 0.0), 0.0, 0.0, 0.0, 0.0, False, rail_seq=tuple(rails))
    # 切角用最终接近方向近似
    final_dir = normalize(sub(g, pos))
    tdir_ok = normalize(sub(pocket, target))
    cut = _cut_angle(final_dir, tdir_ok) if (final_dir and tdir_ok) else 0.0
    if cut > max_cut_deg:
        return Shot(pocket, g, d, 0.0, dist(target, pocket), 0.0, cut, False,
                    bounce_points=bounce_points, rail_seq=tuple(rails), label=f"{len(rails)}库")
    cue_to_contact = dist(cue, bounce_points[0]) if bounce_points else dist(cue, g)
    # 总路程 = 实际多库路径长（cue→各反弹点→鬼球）+ 目标球→袋口。
    # 之前用 dist(cue,g) 直线距离会系统性低估 → 力度偏小、排序失真。
    path_pts: List[Point] = [cue] + bounce_points + [g]
    kick_path = sum(dist(path_pts[i], path_pts[i + 1]) for i in range(len(path_pts) - 1))
    total = kick_path + dist(target, pocket)
    # 母球到鬼球的各段不能提前碰到目标球；目标球碰撞后的路线还要
    # 检查其它球。目标球本身只在最后的 g→target 段作为终点，不算障碍。
    cue_legs: List[Tuple[Point, Point]] = list(zip(path_pts[:-1], path_pts[1:]))
    collision_r = max(float(r), cue_r, target_r)
    blocked, by = _path_blocked(cue_legs, list(others) + [target], collision_r)
    if not blocked:
        # 目标球从自己的球心沿 target→pocket 方向运动；g→target
        # 只是接触关系，不是目标球或母球的运动轨迹。
        object_legs = [(target, pocket)]
        blocked, by = _path_blocked(object_legs, others, collision_r)
    label = f"{len(rails)}库"
    return Shot(
        pocket=pocket, ghost=g, aim_dir=d,
        cue_to_contact=cue_to_contact, target_to_pocket=dist(target, pocket),
        total=total, cut_deg=cut, valid=True, blocked=blocked, blocked_by=by,
        bounce_points=bounce_points, rail_seq=tuple(rails),
        label=label,
    )


# ---------- 力度建议 ----------

def power_suggestion(total: float, table_width: float,
                     dref_ratio: float = 2.2, min_pct: float = 10.0,
                     curve: float = 1.0, max_pct: float = 100.0,
                     cut_deg: float = 0.0, gain: float = 1.0,
                     bias: float = 0.0) -> int:
    """总路程 → 推荐力度 0-100。

    dref_ratio：把「多少倍台面宽度」视为满杆距离（默认 2.2 倍）。
    curve > 1 让大力度段更保守。留 min_pct 的底力避免轻推走不动。

    cut_deg：切角补偿。切角越大，母球传给目标球的动量份额越少
    （≈cos²cut），需要更高杆速；等效路程按 1/cos(cut) 放大
    （切角≤30° 不补偿，≥80° 按上限 1/0.25=4 倍封顶）。
    """
    eff = total
    if cut_deg > 30.0:
        c = max(math.cos(math.radians(min(cut_deg, 80.0))), 0.25)
        eff = total / c
    dref = dref_ratio * table_width
    raw = clamp(eff / dref, 0.0, 1.0)
    pct = bias + gain * (min_pct + (max_pct - min_pct) * (raw ** curve))
    return int(round(clamp(pct, 1.0, max_pct)))


# ---------- 袋口与方案排序 ----------

def default_pockets(w: float, h: float) -> List[Point]:
    """标准 6 袋：四角 + 两长边中点（QQ 2D桌球 台面）。"""
    return [
        (0.0, 0.0), (w, 0.0), (0.0, h), (w, h),     # 四角
        (w / 2.0, 0.0), (w / 2.0, h),               # 长边中点
    ]


KICK_SEQUENCES: Tuple[Tuple[str, ...], ...] = (
    ("top",), ("bottom",), ("left",), ("right",),          # 一库
    ("top", "bottom"), ("bottom", "top"),                  # 两库
    ("left", "right"), ("right", "left"),
    ("top", "left"), ("top", "right"), ("bottom", "left"), ("bottom", "right"),
    ("left", "top"), ("left", "bottom"), ("right", "top"), ("right", "bottom"),
)


def plan_shots(cue: Point, target: Point, pockets: Sequence[Point], r: float,
               w: float, h: float, others: Sequence[Point] = (),
               allow_kicks: bool = True, max_kicks: int = 2,
               rail_inset: float = 0.0,
               pocket_clearance: float = 0.0,
               cue_radius: Optional[float] = None,
               target_radius: Optional[float] = None,
               ghost_offset: Point = (0.0, 0.0)) -> List[Shot]:
    """对每个袋口生成可执行的直球/库边方案并按优先级与路程排序。

    排序：直球 < 一库 < 两库；同级按总路程升序。
    被其它球阻挡的方案只作为底层诊断结果，不返回给上层推荐。
    """
    plans: List[Shot] = []
    for p in pockets:
        s = direct_shot(cue, target, p, r, others,
                        cue_radius=cue_radius,
                        target_radius=target_radius,
                        ghost_offset=ghost_offset)
        if s.valid:
            plans.append(s)
        if allow_kicks:
            for rails in KICK_SEQUENCES:
                if len(rails) > max_kicks:
                    continue
                k = kick_shot(cue, target, p, r, rails, w, h, others,
                              rail_inset=rail_inset, pockets=pockets,
                              pocket_clearance=pocket_clearance,
                              cue_radius=cue_radius,
                              target_radius=target_radius,
                              ghost_offset=ghost_offset)
                if k.valid:
                    plans.append(k)
    def key(s: Shot) -> Tuple[int, float]:
        if not s.bounce_points:
            base = 0 if not s.blocked else 2
        else:
            base = 3 + len(s.bounce_points)
        return (base, s.total)
    plans = [s for s in plans if not s.blocked]
    plans.sort(key=key)
    return plans


def route_score(shot: Shot, table_width: float,
                max_cut_deg: float = MAX_CUT_DEG) -> float:
    """估计一条可行路线的失误风险，分数越低越适合自动推荐。

    ``plan_shots`` 保留「直球/一库/两库」的传统分组顺序，方便诊断和
    兼容旧调用；自动选择则不能只看总路程。薄切虽然距离短，但有效袋口
    宽度和容错都很小；库边路线也应随反弹次数增加风险。这里把切角和
    反弹数转成可比较的惩罚，不改变几何有效性判断。
    """
    if not shot.valid or shot.blocked:
        return float("inf")
    cut_ratio = clamp(shot.cut_deg / max(1e-6, max_cut_deg), 0.0, 1.0)
    rails = len(shot.bounce_points)
    # 切角风险在 60 度以后明显上升；库边每多一库都会引入反弹误差。
    cut_penalty = table_width * 0.60 * (cut_ratio ** 2.4)
    rail_penalty = table_width * (0.16 * rails + 0.08 * rails * rails)
    return float(shot.total + cut_penalty + rail_penalty)


def rank_shots(plans: Sequence[Shot], table_width: float,
               max_cut_deg: float = MAX_CUT_DEG) -> List[Shot]:
    """按自动推荐质量排序，稳定保留原列表的顺序作为平局顺序。"""
    return sorted(plans, key=lambda s: route_score(s, table_width, max_cut_deg))


def best_shot(plans: Sequence[Shot], table_width: float,
              max_cut_deg: float = MAX_CUT_DEG) -> Optional[Shot]:
    """从已验证的候选路线中选最适合实际击打的一条。"""
    ranked = rank_shots(plans, table_width, max_cut_deg)
    return ranked[0] if ranked else None
