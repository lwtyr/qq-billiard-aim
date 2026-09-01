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
    # 袋口容错最优瞄准点（v3.10）：目标球实际出球方向终点位于袋口开口区间
    # 的角平分线上；None = 瞄准袋口中心（旧行为/未启用让点）。
    aim_point: Optional[Point] = None


def _cut_angle(approach: Point, target_dir: Point) -> float:
    cos = clamp(dot(approach, target_dir), -1.0, 1.0)
    return math.degrees(math.acos(cos))


MAX_CUT_DEG = 85.0
"""可行切角上限（度）。切角接近 90° 时母球传递给目标球沿袋口方向的
动量趋近 0，物理上不可能进球；超过上限的方案直接判不可行。"""


def _path_blocked(segments: Sequence[Tuple[Point, Point]],
                  others: Sequence[Point], r: float, tolerance: float = 1.0) -> Tuple[bool, Optional[str]]:
    """判断路径各段是否被 other 球阻挡。

    严格约束：绿线（cue→ghost）与黄虚线（target→pocket）前进路径上绝不允许有任何碰撞球。
    使用扫掠球（Swept-Sphere Capsule）进行精确碰撞判定：
    - 位于球体前进路径前方且侧向间距 < 2r 的球严密判定为阻挡；
    - 位于起点球体后方（已背向运动）的相切球不产生虚假阻挡。
    """
    r_coll = 2.0 * r * tolerance
    r_coll2 = r_coll * r_coll
    for o in others:
        for a, b in segments:
            vx = b[0] - a[0]
            vy = b[1] - a[1]
            L = math.hypot(vx, vy)
            if L < 1e-9:
                continue
            ux = vx / L
            uy = vy / L
            wx = o[0] - a[0]
            wy = o[1] - a[1]
            s = wx * ux + wy * uy
            d_perp2 = (wx * wx + wy * wy) - (s * s)
            if d_perp2 >= r_coll2:
                continue
            half_chord = math.sqrt(max(0.0, r_coll2 - d_perp2))
            t_enter = s - half_chord
            t_exit = s + half_chord
            # 碰撞区间 [t_enter, t_exit] 必须在运动前进区间 [0, L] 内发生
            if t_enter < L and t_exit > 0.05 * r_coll and s > -0.2 * r_coll:
                return True, f"({o[0]:.0f},{o[1]:.0f})"
    return False, None


def direct_shot(cue: Point, target: Point, pocket: Point, r: float,
                others: Sequence[Point] = (),
                max_cut_deg: float = MAX_CUT_DEG,
                cue_radius: Optional[float] = None,
                target_radius: Optional[float] = None,
                ghost_offset: Point = (0.0, 0.0),
                aim_half_width: float = 0.0,
                table_size: Optional[Tuple[float, float]] = None) -> Shot:
    """直球：鬼球法。返回瞄准方案；不可打（切角超限等）时 valid=False。

    aim_half_width>0 且给出 table_size 时，入袋瞄准点从袋口中心移到
    「袋口开口区间角平分线」位置（斜切球自动让点，方向余量最大化）。
    shot.pocket 始终保存袋口中心（袋口归类/索引/绘制用），实际瞄准点
    存于 shot.aim_point；入射角过滤按真实出球方向 ghost→aim_point。
    """
    cue_r, target_r = _resolve_radii(r, cue_radius, target_radius)
    aim = pocket
    if aim_half_width > 0.0 and table_size is not None:
        aim = pocket_aim_point(target, pocket, table_size[0], table_size[1],
                               aim_half_width, max(float(r), 1e-6))
    g = ghost_pos(target, aim, r, cue_r, target_r, ghost_offset)
    if g is None:
        return Shot(pocket, target, (0.0, 0.0), 0.0, 0.0, 0.0, 0.0, False)
    d = normalize(sub(g, cue))
    if d is None:
        return Shot(pocket, g, (0.0, 0.0), 0.0, 0.0, 0.0, 0.0, False)
    tdir = normalize(sub(aim, target))
    cut = _cut_angle(d, tdir)
    if cut > max_cut_deg:
        # 薄切不可行：接触时沿目标球→袋口方向的分速度≈0，进不了球
        return Shot(pocket, g, d, dist(cue, g), dist(target, aim),
                    dist(cue, g) + dist(target, aim), cut, False, label="直球")
    cue_to_contact = dist(cue, g)
    target_to_pocket = dist(target, aim)
    # 母球只走 cue→ghost；目标球被击出后才走 target→aim_point（袋口开口
    # 区间的容错最优处）。g→target 是两球接触几何，不是任何球心的运动
    # 路径，不能当作障碍段，否则 rack/贴球场景会被大量误判为被挡。
    collision_r = max(float(r), cue_r, target_r)
    blocked, by = _path_blocked([(cue, g), (target, aim)], others, collision_r)
    return Shot(
        pocket=pocket, ghost=g, aim_dir=d,
        cue_to_contact=cue_to_contact, target_to_pocket=target_to_pocket,
        total=cue_to_contact + target_to_pocket, cut_deg=cut,
        valid=True, blocked=blocked, blocked_by=by, label="直球",
        aim_point=(aim if aim_half_width > 0.0 else None),
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
              ghost_offset: Point = (0.0, 0.0),
              aim_half_width: float = 0.0) -> Shot:
    """库边解围：unfolding 求出发方向，再在实空间仿真验证反弹序列。

    rails 为期望依次触碰的库边，如 ("top",) 一库，("right", "bottom") 两库。
    仿真校验：实际反弹序列必须与 rails 完全一致，且最终到达真实鬼球附近。
    aim_half_width>0 时鬼球按袋口容错最优瞄准点计算（同 direct_shot 让点）。
    """
    cue_r, target_r = _resolve_radii(r, cue_radius, target_radius)
    aim = pocket
    if aim_half_width > 0.0:
        aim = pocket_aim_point(target, pocket, w, h, aim_half_width,
                               max(float(r), 1e-6))
    g = ghost_pos(target, aim, r, cue_r, target_r, ghost_offset)
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

    # 第一库几何快速剪枝：初速度方向背离首库时，物理上必不可能优先触碰该库
    r0 = rails[0]
    if (r0 == "top" and d[1] >= 0) or \
       (r0 == "bottom" and d[1] <= 0) or \
       (r0 == "left" and d[0] >= 0) or \
       (r0 == "right" and d[0] <= 0):
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
        # 仅检测射线运动方向可能迎面相交的两条库边（水平+垂直各一），避免无谓遍历反向库边
        r_h = "right" if vel[0] > 1e-12 else ("left" if vel[0] < -1e-12 else None)
        r_v = "bottom" if vel[1] > 1e-12 else ("top" if vel[1] < -1e-12 else None)
        for rail in (r_h, r_v):
            if rail is None:
                continue
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
    tdir_ok = normalize(sub(aim, target))
    cut = _cut_angle(final_dir, tdir_ok) if (final_dir and tdir_ok) else 0.0
    if cut > max_cut_deg:
        return Shot(pocket, g, d, 0.0, dist(target, aim), 0.0, cut, False,
                    bounce_points=bounce_points, rail_seq=tuple(rails), label=f"{len(rails)}库")
    cue_to_contact = dist(cue, bounce_points[0]) if bounce_points else dist(cue, g)
    # 总路程 = 实际多库路径长（cue→各反弹点→鬼球）+ 目标球→袋口。
    # 之前用 dist(cue,g) 直线距离会系统性低估 → 力度偏小、排序失真。
    path_pts: List[Point] = [cue] + bounce_points + [g]
    kick_path = sum(dist(path_pts[i], path_pts[i + 1]) for i in range(len(path_pts) - 1))
    total = kick_path + dist(target, aim)
    # 母球到鬼球的各段不能提前碰到目标球；目标球碰撞后的路线还要
    # 检查其它球。目标球本身只在最后的 g→target 段作为终点，不算障碍。
    cue_legs: List[Tuple[Point, Point]] = list(zip(path_pts[:-1], path_pts[1:]))
    collision_r = max(float(r), cue_r, target_r)
    blocked, by = _path_blocked(cue_legs, list(others) + [target], collision_r)
    if not blocked:
        # 目标球从自己的球心沿 target→aim_point 方向运动；g→target
        # 只是接触关系，不是目标球或母球的运动轨迹。
        object_legs = [(target, aim)]
        blocked, by = _path_blocked(object_legs, others, collision_r)
    label = f"{len(rails)}库"
    return Shot(
        pocket=pocket, ghost=g, aim_dir=d,
        cue_to_contact=cue_to_contact, target_to_pocket=dist(target, aim),
        total=total, cut_deg=cut, valid=True, blocked=blocked, blocked_by=by,
        bounce_points=bounce_points, rail_seq=tuple(rails),
        label=label,
        aim_point=(aim if aim_half_width > 0.0 else None),
    )


# ---------- 力度建议 ----------

def power_suggestion(total: float, table_width: float,
                     dref_ratio: float = 2.2, min_pct: float = 10.0,
                     curve: float = 1.0, max_pct: float = 100.0,
                     cut_deg: float = 0.0, gain: float = 1.0,
                     bias: float = 0.0, rails: int = 0,
                     rail_loss: float = 0.22) -> int:
    """总路程 → 推荐力度 0-100。

    dref_ratio：把「多少倍台面宽度」视为满杆距离（默认 2.2 倍）。
    curve > 1 让大力度段更保守。留 min_pct 的底力避免轻推走不动。

    cut_deg：切角补偿。切角越大，母球传给目标球的动量份额越少
    （≈cos²cut），需要更高杆速；等效路程按 1/cos(cut) 放大
    （切角≤30° 不补偿，≥80° 按上限 1/0.25=4 倍封顶）。

    rails/rail_loss：库边能量损耗补偿。每次反弹损失一部分动能
    （实测 2D 桌球每库约损失 20-25% 速度），等效路程按
    (1 + rail_loss × 库数) 放大，与切角补偿相乘。
    """
    eff = total * (1.0 + rail_loss * max(0, rails))
    if cut_deg > 30.0:
        c = max(math.cos(math.radians(min(cut_deg, 80.0))), 0.25)
        eff = eff / c          # 注意：在库边补偿后的 eff 基础上再放大，不能用 total 覆盖
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


MID_POCKET_MIN_COS = 0.55
"""中袋入射角限制：进袋方向与袋口内法线夹角 ≤ ≈57°。中袋开口窄，
大斜角必撞袋角弹出；QQ 2D 无「运气球」，直接判不可行。"""
CORNER_POCKET_MIN_COS = 0.42
"""角袋入射角限制（≤ ≈65°）。角袋沿库边抹进仍可行（cos≈0.7），阈值放宽。"""


def _pocket_frame(p: Point, w: float, h: float,
                  edge_tolerance: float = 0.0
                  ) -> Optional[Tuple[Point, Point, bool]]:
    """袋口几何帧：(开口外法线 n, 开口线方向 u, 是否中袋)；非上下边袋位 None。

    与入射角过滤共用的袋口归类：中袋 |p.x-w/2| < 0.25w、法线垂直库边；
    角袋法线取对角方向。袋口允许精修/用户偏移（edge_tolerance 容差内
    仍按原归类），让点/偏移永远不会使袋口逃出归类（v3.8 前的过滤
    绕过缺陷不因此复发）。
    """
    edge_tolerance = max(0.0, float(edge_tolerance))
    if not (p[1] <= edge_tolerance or p[1] >= h - edge_tolerance):
        return None
    at_top = p[1] <= edge_tolerance
    if abs(p[0] - w / 2.0) < 0.25 * w:
        return (0.0, -1.0 if at_top else 1.0), (1.0, 0.0), True
    nx = -1.0 if p[0] < w / 2.0 else 1.0
    ny = -1.0 if at_top else 1.0
    length = math.hypot(nx, ny)
    return (nx / length, ny / length), (-ny / length, nx / length), False


def pocket_entry_cos(shot: Shot, w: float, h: float,
                     edge_tolerance: float = 0.0) -> Optional[float]:
    """Return the signed pocket-entry cosine used by the geometry filter.

    The value is 1 for an approach along the pocket opening normal and 0 for
    a tangential approach.  ``None`` means this is not one of the standard
    top/bottom pocket positions, so the entry angle is intentionally not
    constrained by this model.

    真实出球方向是 ghost→aim_point（v3.10 起瞄准点可能从袋口中心让位到
    容错最优处）；袋口归类始终基于 shot.pocket（袋口中心），因此让点
    永远不会绕过入射角过滤。
    """
    frame = _pocket_frame(shot.pocket, w, h, edge_tolerance)
    if frame is None:
        return None
    n = frame[0]
    travel = shot.aim_point or shot.pocket
    d = sub(travel, shot.ghost)
    length = math.hypot(d[0], d[1])
    if length < 1e-9:
        return 1.0
    return clamp((d[0] * n[0] + d[1] * n[1]) / length, -1.0, 1.0)


def pocket_entry_limit(shot: Shot, w: float, h: float,
                       edge_tolerance: float = 0.0,
                       mid_cos: float = MID_POCKET_MIN_COS,
                       corner_cos: float = CORNER_POCKET_MIN_COS) -> Optional[float]:
    """Return the minimum entry cosine, or ``None`` for unconstrained sides."""
    p = shot.pocket
    edge_tolerance = max(0.0, float(edge_tolerance))
    if not (p[1] <= edge_tolerance or p[1] >= h - edge_tolerance):
        return None
    return mid_cos if abs(p[0] - w / 2.0) < 0.25 * w else corner_cos


def pocket_entry_ok(shot: Shot, w: float, h: float, r: float,
                    mid_cos: float = MID_POCKET_MIN_COS,
                    corner_cos: float = CORNER_POCKET_MIN_COS) -> bool:
    """目标球进袋的入射方向是否在该袋口的可捕获范围内。

    目标球被击后沿「鬼球 → 袋口」方向运动。中袋只能正对进
    （与内法线夹角小），大斜角穿袋口必撞袋角弹出；角袋可沿库边
    抹进，阈值更宽。鬼球距袋口 < 2r 时方向几何不稳，放行。
    """
    p, g = shot.pocket, shot.ghost
    if dist(g, p) < 2.0 * r:
        return True
    # Pocket coordinates are allowed to be inset by calibration offsets.  Use
    # a radius-sized edge tolerance so a refined pocket at y=1.2r still gets
    # the same angle filter as the nominal top-edge pocket.
    limit = pocket_entry_limit(shot, w, h, r, mid_cos, corner_cos)
    if limit is None:
        return True
    cos_in = pocket_entry_cos(shot, w, h, r)
    return cos_in is None or cos_in >= limit


def plan_shots(cue: Point, target: Point, pockets: Sequence[Point], r: float,
               w: float, h: float, others: Sequence[Point] = (),
               allow_kicks: bool = True, max_kicks: int = 2,
               rail_inset: float = 0.0,
               pocket_clearance: float = 0.0,
               cue_radius: Optional[float] = None,
               target_radius: Optional[float] = None,
               ghost_offset: Point = (0.0, 0.0),
               pocket_aim_half: float = 0.0) -> List[Shot]:
    """对每个袋口生成可执行的直球/库边方案并按优先级与路程排序。

    排序：直球 < 一库 < 两库；同级按总路程升序。
    被其它球阻挡的方案只作为底层诊断结果，不返回给上层推荐。
    """
    plans: List[Shot] = []
    for p in pockets:
        s = direct_shot(cue, target, p, r, others,
                        cue_radius=cue_radius,
                        target_radius=target_radius,
                        ghost_offset=ghost_offset,
                        aim_half_width=pocket_aim_half, table_size=(w, h))
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
                              ghost_offset=ghost_offset,
                              aim_half_width=pocket_aim_half)
                if k.valid:
                    plans.append(k)
    def key(s: Shot) -> Tuple[int, float]:
        if not s.bounce_points:
            base = 0 if not s.blocked else 2
        else:
            base = 3 + len(s.bounce_points)
        return (base, s.total)
    plans = [s for s in plans if not s.blocked]
    # 中袋大斜角物理上进不了（撞袋角弹出），入射角超限直接淘汰。
    plans = [s for s in plans if pocket_entry_ok(s, w, h, r)]
    plans.sort(key=key)
    return plans


def cue_tangent(shot: Shot) -> Tuple[Point, float]:
    """母球碰撞后的切线方向（单位向量）与切向速度份额 sin(cut)。

    碰撞瞬间母球速度分解：沿球心连线（传给目标球）+ 切线分量
    （母球自己带走）。切线方向 = aim 方向去掉法向分量。近正碰
    （sin(cut) 小）时切线分量趋近 0，母球基本留在原地，不画。
    库边解球用最后一库反弹点 → 鬼球方向作为碰撞时进杆方向。
    """
    if shot.bounce_points:
        bx, by = shot.bounce_points[-1]
        ix, iy = shot.ghost[0] - bx, shot.ghost[1] - by
    else:
        ix, iy = shot.aim_dir
    ilen = math.hypot(ix, iy)
    if ilen < 1e-9:
        return (0.0, 0.0), 0.0
    ix, iy = ix / ilen, iy / ilen
    # 法线 = 鬼球 → 目标球（球心连线），直球时 ≈ 瞄准点 → 鬼球方向
    _aim = shot.aim_point or shot.pocket
    nx, ny = shot.ghost[0] - _aim[0], shot.ghost[1] - _aim[1]
    nlen = math.hypot(nx, ny)
    if nlen < 1e-9:
        return (0.0, 0.0), 0.0
    nx, ny = nx / nlen, ny / nlen
    tx = ix - (ix * nx + iy * ny) * nx
    ty = iy - (ix * nx + iy * ny) * ny
    tn = math.hypot(tx, ty)
    if tn < 1e-9:
        return (0.0, 0.0), 0.0
    return (tx / tn, ty / tn), tn


def scratch_risk(shot: Shot, ball_radius: float, pocket_radius: float,
                 pockets: Sequence[Point], span: float) -> float:
    """母球沿切线滚动后的摔袋风险 0..1（0=无风险）。

    从鬼球沿切线方向射射线，检查各袋口中与射线的距离；同时要求
    袋口在切线前方且距离不超过 1.2 倍台面跨度（太远的袋口母球
    到不了）。目标袋本身权重更高（白球跟进目标袋最常见）。
    """
    tdir, tfrac = cue_tangent(shot)
    if tfrac < 0.17:
        return 0.0
    gx, gy = shot.ghost
    tx, ty = tdir
    reach = pocket_radius + 0.3 * ball_radius
    best = 0.0
    for (cx, cy) in pockets:
        sx, sy = cx - gx, cy - gy
        s = sx * tx + sy * ty          # 沿切线前进距离
        if s <= 0.0 or s > 1.2 * span:
            continue
        d = abs(sx * ty - sy * tx)     # 到射线的垂距
        if d >= reach:
            continue
        risk = (1.0 - d / reach) * (1.0 - 0.45 * min(1.0, s / span))
        if (cx, cy) == tuple(shot.pocket):
            risk *= 1.35
        best = max(best, min(1.0, risk))
    return float(best)


def route_score(shot: Shot, table_width: float,
                max_cut_deg: float = MAX_CUT_DEG,
                table_height: Optional[float] = None,
                pocket_radius: Optional[float] = None,
                ball_radius: Optional[float] = None,
                pockets: Optional[Sequence[Point]] = None) -> float:
    """估计一条可行路线的失误风险，分数越低越适合自动推荐。

    ``plan_shots`` 保留「直球/一库/两库」的传统分组顺序，方便诊断和
    兼容旧调用；自动选择则不能只看总路程。薄切虽然距离短，但有效袋口
    宽度和容错都很小；库边路线也应随反弹次数增加风险。这里把切角和
    反弹数转成可比较的惩罚，不改变几何有效性判断。

    可选精度项（传入 table_height/pocket_radius/pockets 后启用）：
    - 袋口入射角：目标球接近方向偏离袋口中线越多，有效接受宽度越窄；
    - 摔袋风险：母球切线指向某袋口时加重大惩罚（白球跟进=送分）。
    """
    if not shot.valid or shot.blocked:
        return float("inf")
    cut_ratio = clamp(shot.cut_deg / max(1e-6, max_cut_deg), 0.0, 1.0)
    rails = len(shot.bounce_points)
    # 切角风险在 60 度以后明显上升；库边每多一库都会引入反弹误差。
    cut_penalty = table_width * 0.60 * (cut_ratio ** 2.4)
    rail_penalty = table_width * (0.16 * rails + 0.08 * rails * rails)
    score = shot.total + cut_penalty + rail_penalty
    if table_height and pocket_radius and ball_radius:
        # 袋口入射角惩罚：接近方向偏离袋口中线 → 有效接受宽度收窄。
        # 口法向取「台心指向袋口」（开口朝向）：合法击球的接近方向
        # （鬼球→袋口）应与开口方向大致同向 → cos_in 接近 1；
        # 贴着库边斜着抹进去的球接近方向与开口近乎垂直 → cos_in 接近 0。
        gx, gy = shot.ghost
        px, py = shot.pocket
        ax, ay = px - gx, py - gy
        na = math.hypot(ax, ay)
        nx, ny = (px - table_width / 2.0, py - table_height / 2.0)
        nn = math.hypot(nx, ny)
        if na > 1e-6 and nn > 1e-6:
            cos_in = clamp((ax * nx + ay * ny) / (na * nn), 0.0, 1.0)
            full = 2.0 * max(1e-6, pocket_radius - ball_radius)
            narrow = 1.0 - cos_in
            # 窄到一半以上才明显罚，避免与切角惩罚重复计账
            if narrow > 0.5:
                score += table_width * 0.45 * ((narrow - 0.5) * 2.0) ** 1.5
        # 白球摔袋风险：切线指向袋口 = 送给对手机会
        if pockets:
            risk = scratch_risk(shot, ball_radius, pocket_radius,
                                pockets, max(table_width, table_height))
            score += table_width * 1.1 * risk
    return float(score)


def rank_shots(plans: Sequence[Shot], table_width: float,
               max_cut_deg: float = MAX_CUT_DEG, **kw) -> List[Shot]:
    """按自动推荐质量排序，稳定保留原列表的顺序作为平局顺序。"""
    return sorted(plans, key=lambda s: route_score(s, table_width,
                                                   max_cut_deg, **kw))


def best_shot(plans: Sequence[Shot], table_width: float,
              max_cut_deg: float = MAX_CUT_DEG, **kw) -> Optional[Shot]:
    """从已验证的候选路线中选最适合实际击打的一条。"""
    ranked = rank_shots(plans, table_width, max_cut_deg, **kw)
    return ranked[0] if ranked else None


# ---------- 袋口容错瞄准点与进球成功率（v3.10） ----------


def pocket_aim_point(target: Point, pocket: Point, w: float, h: float,
                     half_width: float, edge_tolerance: float = 0.0) -> Point:
    """袋口容错最优瞄准点：从目标球看开口区间 [A,B] 的内角平分线。

    正对袋口：平分线自然落在袋口中心，与旧行为一致；斜切进袋：瞄准点
    自动在开口线上让位，使出球方向到两侧袋角的夹角余量相等（最大）。
    瞄准点即方向上容错最强的入袋点。非标准袋位/退化情形返回袋口中心。
    """
    frame = _pocket_frame(pocket, w, h, edge_tolerance)
    if frame is None:
        return pocket
    n, u, _mid = frame
    a = max(float(half_width), 1e-6)
    pa = (pocket[0] - a * u[0], pocket[1] - a * u[1])
    pb = (pocket[0] + a * u[0], pocket[1] + a * u[1])
    da = normalize(sub(pa, target))
    db = normalize(sub(pb, target))
    if da is None or db is None:
        return pocket
    d = normalize((da[0] + db[0], da[1] + db[1]))
    if d is None:
        return pocket
    # 射线 target + t·d 与开口线 pocket + s·u 求交
    v = sub(pocket, target)
    cross_du = d[0] * u[1] - d[1] * u[0]
    if abs(cross_du) < 1e-9:
        return pocket
    t = (v[0] * u[1] - v[1] * u[0]) / cross_du
    s = (v[0] * d[1] - v[1] * d[0]) / cross_du
    if t <= 1e-9:
        return pocket
    # 平分线必在开口区间内；数值保护夹紧并预留少许袋角余量
    s = clamp(s, -0.92 * a, 0.92 * a)
    aim = (pocket[0] + s * u[0], pocket[1] + s * u[1])
    if dist(aim, target) < 1e-9:
        return pocket
    return aim


def pot_success_prob(shot: Shot, w: float, h: float, r: float,
                     cfg=None) -> Optional[float]:
    """估计进球概率 0..1（排序用相对值；-blocked/invalid 方案为 0）。

    启发式模型（σ 可由真实进球反馈标定，见 PRECISION_ANALYSIS.md）：

    * 方向误差 σθ：球心/映射定位误差 σ_u 分别经「母球行程」与「接触
      法线杠杆(≈2r)」放大，加执行对齐误差 σ_exec；薄切随切角增大
      更敏感（经验系数 (1+(cut/75°)²)）；
    * 袋口余量 margin：出球方向与开口线交点到最近袋角的距离，再乘
      |cos_in| 折算——斜射时同样的垂直误差在开口线上被放大；
    * 进球率 = P(|落点误差| < margin) = erf(margin / (√2·σθ·目标球行程))；
    * 贴近入射角阈值的路线乘 0.7+0.3·软度系数（硬过滤照旧，只是不再
      把“刚过阈值”和“正对入袋”当同难度）；每库反弹乘 kick_reliability。

    只影响路线/目标排序，不改变任何几何有效性判定。
    """
    if not shot.valid or shot.blocked:
        return 0.0
    sigma_u = float(getattr(cfg, "aim_sigma_units", 1.0)) if cfg else 1.0
    sigma_exec = float(getattr(cfg, "exec_sigma_rad", 0.004)) if cfg else 0.004
    kick_rel = float(getattr(cfg, "kick_reliability", 0.92)) if cfg else 0.92
    r_sum = max(2.0 * float(r), 1e-6)
    d1 = max(float(shot.cue_to_contact), r_sum)
    dir_sigma = math.sqrt((sigma_u / d1) ** 2 + (sigma_u / r_sum) ** 2
                          + sigma_exec ** 2)
    dir_sigma *= 1.0 + (min(float(shot.cut_deg), 85.0) / 75.0) ** 2

    accept = float(getattr(cfg, "pocket_accept_ratio", 1.45)) if cfg else 1.45
    edge_tol = max(float(r), 1e-6)
    frame = _pocket_frame(shot.pocket, w, h, edge_tol)
    cos_in = pocket_entry_cos(shot, w, h, edge_tol)
    if frame is None or cos_in is None:
        margin = 0.8 * accept * r
    else:
        n, u, _mid = frame
        aim = shot.aim_point or shot.pocket
        s = (aim[0] - shot.pocket[0]) * u[0] + (aim[1] - shot.pocket[1]) * u[1]
        margin = (max(accept * r - abs(s), 0.05 * accept * r)
                  * max(abs(cos_in), 0.05))
    l2 = max(float(shot.target_to_pocket), r_sum)
    sigma_m = dir_sigma * l2
    if sigma_m < 1e-9:
        return 1.0
    p = math.erf(margin / (math.sqrt(2.0) * sigma_m))
    limit = pocket_entry_limit(shot, w, h, edge_tol)
    if limit is not None and cos_in is not None:
        hardness = clamp((cos_in - limit) / 0.10, 0.0, 1.0)
        p *= 0.7 + 0.3 * hardness
    if shot.bounce_points:
        p *= kick_rel ** len(shot.bounce_points)
    return float(clamp(p, 0.0, 1.0))
