"""OpenCV 视觉（斯诺克版）：从一帧屏幕画面里找出球桌、球、袋口。

与美式八球版的关键差异：
  1. 球色是斯诺克 7 色（红×15、黄/绿/棕/蓝/粉/黑 + 白球），没有橙/紫；
  2. 台呢是绿色 —— 绿球与台面同色，必须先「台呢去背景」才能独立检出绿球；
  3. 红球三角相切摆放 —— 连通域可能粘连多球，用距离变换峰值分离；
  4. 袋口涂灰 —— 防止黑球检测把袋洞当球；
  5. 亚像素球心拟合 —— 边缘点最小二乘圆拟合，把球心精度提到 ±0.1px 级
     （斯诺克袋口比八球小，对坐标误差更敏感）。

流程：
  1. find_table：绿色台呢 → 最大轮廓 → 四边形四角 → 透视矩阵 H（屏幕 → 台面标准坐标）；
  2. warp_table：台面区域校正为 W×H 标准图，几何都在标准坐标里算；
  3. clean_background：台呢色（直方图峰值）+ 袋口 → 涂中性灰，只留球；
  4. detect_balls：按颜色掩膜 + 轮廓/距离变换分离 + 亚像素圆拟合；
  5. refine_pockets：找暗色袋口斑块精修袋口中心。

依赖：numpy + opencv-python（headless 亦可）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

Point = Tuple[float, float]

# 标准台面尺寸（检测与计算都用它）
TABLE_W = 2000.0
TABLE_H = 1000.0

# ---------- 斯诺克球色（唯一判定来源：BALL_HSV_RULES 表） ----------

# 每球 HSV 判定规则：h 允许段列表（None=不限制；uint8 色相 0-179）
# + (s_lo, s_hi) + (v_lo, v_hi) 闭区间。标量判色（classify_pixel）与
# 全帧向量化掩膜（_mask_for_label）都由这张表生成，改动阈值只改这里。
BALL_HSV_RULES: Dict[str, Tuple[Optional[List[Tuple[int, int]]],
                                 Tuple[int, int], Tuple[int, int]]] = {
    "白球": (None, (0, 79), (151, 255)),
    "黑球": (None, (0, 129), (0, 61)),
    "红球": ([(0, 10), (170, 179)], (131, 255), (121, 255)),
    "粉球": ([(150, 176)], (71, 255), (161, 255)),
    "黄球": ([(18, 45)], (111, 255), (191, 255)),
    "绿球": ([(45, 100)], (181, 255), (91, 255)),
    "棕球": ([(8, 32)], (71, 255), (40, 165)),
    "蓝球": ([(100, 130)], (141, 255), (81, 255)),
}
# 白色：高亮度（v>150 排除去背景后的灰色残留 128）+ 低饱和（s<80 容忍轻微染色）。
# 彩球高光同为白色但面积小，由 _split_blobs 的面积/圆度过滤排除。
# 黑色：纯黑球 v<62，收紧排除库边/阴影/UI 深色（v 50-120 常见）；真实黑球 v≈45；
# 袋口已由 clean_background 涂灰（v=128），不在此范围。
# 绿色：鲜绿球高饱和（s>180）；真实台面饱和 ~155 且会被去背景涂灰，收紧阈值
# 防止台面/UI 绿色未涂净时误入绿球掩膜。判据优先级见 PRIORITY（同色相区域
# 红/粉、棕/黄先判更特异的）。

# 参考色为 BGR（cv2 约定），仅用于显示/调试，判定逻辑看 BALL_HSV_RULES。
BALL_PALETTE: Dict[str, Tuple[int, int, int]] = {
    "白球": (255, 255, 255),
    "黑球": (40, 40, 40),
    "红球": (0, 0, 255),
    "粉球": (180, 105, 255),
    "黄球": (40, 200, 255),
    "绿球": (0, 160, 30),
    "棕球": (30, 80, 150),
    "蓝球": (255, 0, 0),
}
PRIORITY = ["白球", "黑球", "红球", "粉球", "黄球", "绿球", "棕球", "蓝球"]


def _match_rule(label: str, h: int, s: int, v: int) -> bool:
    """标量版规则匹配（单像素判色用）。"""
    rule = BALL_HSV_RULES.get(label)
    if rule is None:
        return False
    h_ranges, (s_lo, s_hi), (v_lo, v_hi) = rule
    if not (s_lo <= s <= s_hi and v_lo <= v <= v_hi):
        return False
    if h_ranges is None:
        return True
    return any(lo <= h <= hi for (lo, hi) in h_ranges)


def classify_pixel(hsv_px: np.ndarray) -> str:
    """单像素（或中心均值）按优先级判色。"""
    h, s, v = int(hsv_px[0]), int(hsv_px[1]), int(hsv_px[2])
    for label in PRIORITY:
        if _match_rule(label, h, s, v):
            return label
    return "未知"


@dataclass
class Ball:
    label: str
    pos: Point          # 台面标准坐标（球心，亚像素）
    radius: float       # 标准坐标半径
    subpixel: bool = False   # 是否经亚像素拟合
    confidence: float = 1.0  # 圆形/实心盘验证质量，供跨帧跟踪排序
    track_id: int = -1       # 由 tracking.BallTracker 填充


# ---------- 台面检测 ----------

def order_corners(pts: np.ndarray) -> np.ndarray:
    """4 点 → [TL, TR, BR, BL]（左上、右上、右下、左下）。"""
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)          # y - x
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _lines_intersection(l1: Tuple[float, float, float, float],
                        l2: Tuple[float, float, float, float]) -> Optional[Tuple[float, float]]:
    vx1, vy1, x1, y1 = l1
    vx2, vy2, x2, y2 = l2
    det = vx1 * (-vy2) - (-vx2) * vy1
    if abs(det) < 1e-9:
        return None
    t1 = ((x2 - x1) * (-vy2) - (-vx2) * (y2 - y1)) / det
    return (float(x1 + t1 * vx1), float(y1 + t1 * vy1))


def _fit_quad_by_edges(pts: np.ndarray, band: float = 0.12) -> Optional[List[Tuple[float, float, float, float]]]:
    """PCA 主轴 + 边带极值分组，拟合矩形 4 条边。

    对「某条边被物体咬出缺口」的掩膜（如红球三角压住台面下边）很稳：
    缺口处没有点，剩余边带点仍拟合出正确的边线。
    返回 4 条线 [(vx,vy,x0,y0), ...]。
    """
    pts2 = pts - pts.mean(axis=0)
    cov = pts2.T @ pts2
    w, V = np.linalg.eigh(cov)
    v1 = V[:, 1]  # 主轴（长边方向）
    v2 = V[:, 0]  # 短轴（短边方向）
    p1 = pts2 @ v1
    p2 = pts2 @ v2
    t1 = band * (p1.max() - p1.min())
    t2 = band * (p2.max() - p2.min())
    bands = {
        "a": pts[p2 <= p2.min() + t2],        # 上长边
        "b": pts[p2 >= p2.max() - t2],        # 下长边
        "c": pts[p1 <= p1.min() + t1],        # 左短边
        "d": pts[p1 >= p1.max() - t1],        # 右短边
    }
    lines: List[Tuple[float, float, float, float]] = []
    for name, sel in bands.items():
        if len(sel) < 25:
            return None
        vx, vy, x0, y0 = cv2.fitLine(sel.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
        lines.append((float(vx), float(vy), float(x0), float(y0)))
    return lines


def find_table(frame: np.ndarray, cfg) -> Optional[np.ndarray]:
    """在帧里找绿色台面，返回有序四角 [TL,TR,BR,BL]（屏幕坐标，亚像素）。

    方法：绿色掩膜 → 最大轮廓 → PCA 主轴边带拟合 4 条边 → 两两求交，
    取离质心最近的 4 个交点（对边交点离质心很远，自动排除）→ 排序。
    对「台面某边被球/阴影咬出缺口」的情况稳健（缺口处无点，剩余边带仍准）。
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 饱和度下限从 45 提到 60：过宽的绿色掩膜会把含绿色调的桌面壁纸
    # 也吸进来（实测壁纸 H≈104 落在默认区间内），全屏连片后被
    # 「占屏>97%」门丢弃 → 整帧找不到台面。真实台呢 s 远大于 60。
    mask = cv2.inRange(hsv, (cfg.green_hue_lo, 60, 45), (cfg.green_hue_hi, 255, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    frame_area = frame.shape[0] * frame.shape[1]
    # 台面候选：面积达标，且不是「整个画面」（屏幕边框/全屏绿色界面连片——
    # 周围 UI/桌面若是青绿色调会与台面连成一块占屏 100%，那不是台面）。
    cands = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < cfg.min_table_area_ratio * frame_area:
            continue
        if area > 0.97 * frame_area:
            # 全屏巨型轮廓：掩膜把壁纸等大块区域吸进来时会出现。
            # 不直接丢弃——降级到子层级轮廓重试一次，外框内部的
            # 子轮廓往往才是真正的台面（bug2 全桌面截图场景）。
            subs, _ = cv2.findContours(mask, cv2.RETR_CCOMP,
                                       cv2.CHAIN_APPROX_NONE)
            for sc in subs:
                sa = cv2.contourArea(sc)
                if not (cfg.min_table_area_ratio * frame_area <= sa
                        <= 0.97 * frame_area):
                    continue
                sx, sy, sw, sh = cv2.boundingRect(sc)
                if sw < 30 or sh < 20:
                    continue
                sratio = sw / max(1, sh)
                cands.append(((abs(sratio - 2.0), -sa), sc))
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w < 30 or h < 20:
            continue
        # 台面长宽比约 2:1（斯诺克）；排序改为面积优先、长宽比次之——
        # 原「长宽比优先」在多干扰源时会选中细长的误报区域
        ratio = w / max(1, h)
        score = (-area, abs(ratio - 2.0))
        cands.append((score, c))
    if not cands:
        return None
    cands.sort(key=lambda t: t[0])
    c = cands[0][1]
    pts = c.reshape(-1, 2).astype(np.float32)
    lines = _fit_quad_by_edges(pts)
    if lines is None:
        return None
    m = cv2.moments(c)
    cx0 = m["m10"] / m["m00"]
    cy0 = m["m01"] / m["m00"]
    cands: List[Tuple[float, float]] = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            ip = _lines_intersection(lines[i], lines[j])
            if ip is not None:
                cands.append(ip)
    cands.sort(key=lambda p: (p[0] - cx0) ** 2 + (p[1] - cy0) ** 2)
    if len(cands) < 4:
        return None
    quad = np.array(cands[:4], dtype=np.float32)
    quad = order_corners(quad)
    if not _quad_sane(quad):
        return None

    # QQ 2D 的台面通常是屏幕轴对齐的。此时绿色轮廓的外接边界比
    # fitLine 的边带均值更接近实际像素边界；后者会因抗锯齿、台面纹理
    # 和球造成约 1~2px 的系统性内缩。保留一般四边形路径，只对几乎
    # 水平/垂直的桌面采用边界盒，避免牺牲旋转或透视画面的兼容性。
    x, y, bw, bh = cv2.boundingRect(c)
    # 先拒绝明显的「一边斜」候选。Overlay 白边、库边弹窗或右侧菜单
    # 混入绿色掩膜时，PCA 仍可能拟合出一个数学上平行的梯形；若把它
    # 当作首帧台面，后续自动框选会把错误区域保存下来。QQ 2D 桌面
    # 本身轴对齐，真实边缘斜率通常远小于该阈值。
    tl, tr, br, bl = quad
    edge_skew = max(
        abs(float(tr[1] - tl[1])) / max(1.0, bw),
        abs(float(br[1] - bl[1])) / max(1.0, bw),
        abs(float(bl[0] - tl[0])) / max(1.0, bh),
        abs(float(br[0] - tr[0])) / max(1.0, bh),
        abs(float(np.linalg.norm(tr - tl) - np.linalg.norm(br - bl)))
        / max(1.0, bw),
    )
    if edge_skew > float(getattr(cfg, "table_max_edge_skew", 0.02)):
        return None
    axis_like = (
        max(abs(float(quad[1, 1] - quad[0, 1])),
            abs(float(quad[2, 1] - quad[3, 1]))) < 0.015 * max(1, bw)
        and max(abs(float(quad[3, 0] - quad[0, 0])),
                abs(float(quad[2, 0] - quad[1, 0]))) < 0.015 * max(1, bh)
    )
    if axis_like:
        box_quad = np.array(
            [[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]],
            dtype=np.float32,
        )
        if _quad_sane(box_quad):
            quad = box_quad
    return quad


def _quad_sane(quad: np.ndarray) -> bool:
    """四边形合理性校验：近似矩形（对边平行、长宽比合理）。

    框选范围偏大（混入 UI）时 find_table 可能拟合出畸形四边形
    （负坐标/旋转），warp 后 UI 会被当成球 → 球数爆炸。这里拒绝畸形。
    """
    tl, tr, br, bl = quad
    w1 = float(np.linalg.norm(tr - tl))
    w2 = float(np.linalg.norm(br - bl))
    h1 = float(np.linalg.norm(bl - tl))
    h2 = float(np.linalg.norm(br - tr))
    if min(w1, w2, h1, h2) < 25:
        return False

    def _parallel(a, b):
        n = np.linalg.norm(a) * np.linalg.norm(b)
        return abs(float(np.dot(a, b)) / n) > 0.88 if n > 1e-9 else False

    if not _parallel(tr - tl, br - bl):      # 两条长边平行
        return False
    if not _parallel(bl - tl, br - tr):      # 两条短边平行
        return False
    # 长宽比：斯诺克台面 2:1，允许 1.2~3.2
    long_side = max(w1, w2)
    short_side = max(h1, h2)
    ratio = long_side / short_side
    if not (1.15 < ratio < 3.3):
        return False
    return True


def homography(quad: np.ndarray, w: float = TABLE_W, h: float = TABLE_H) -> np.ndarray:
    """屏幕四边形 → 标准台面坐标的透视矩阵 H（3x3）。"""
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    return cv2.getPerspectiveTransform(quad, dst)


def warp_table(frame: np.ndarray, H: np.ndarray,
               w: float = TABLE_W, h: float = TABLE_H) -> np.ndarray:
    warped = cv2.warpPerspective(frame, H, (int(w), int(h)), flags=cv2.INTER_LINEAR)
    # 台面边界一圈涂灰：quad 边缘常压在库边顶端白线/残留 overlay 框线上，
    # warp 插值会把它们拉成整条白带 → watershed 切出一串假白球。
    # 台面边界本身不可能是球，直接排除。
    m = max(6, int(0.012 * min(int(w), int(h))))   # ≈12px
    warped[:m, :] = 128
    warped[-m:, :] = 128
    warped[:, :m] = 128
    warped[:, -m:] = 128
    return warped


# ---------- 坐标映射 ----------

def point_screen_to_table(pt: Point, H: np.ndarray) -> Point:
    x, y = pt
    denom = H[2, 0] * x + H[2, 1] * y + H[2, 2]
    if abs(denom) < 1e-9:
        # 单应矩阵退化（H 非法或点落在投影极线上）：
        # 返回原坐标会让调用方拿到一个"看似台面实则屏幕"的错坐标去算路线，
        # 必须让调用方显式失败而不是静默错下去。
        raise ValueError(f"point_screen_to_table: 退化单应（denom={denom:.3g}，pt={pt!r}）")
    sx = (H[0, 0] * x + H[0, 1] * y + H[0, 2]) / denom
    sy = (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / denom
    return (float(sx), float(sy))


def point_table_to_screen(pt: Point, Hinv: np.ndarray) -> Point:
    x, y = pt
    denom = Hinv[2, 0] * x + Hinv[2, 1] * y + Hinv[2, 2]
    if abs(denom) < 1e-9:
        raise ValueError(f"point_table_to_screen: 退化单应（denom={denom:.3g}，pt={pt!r}）")
    sx = (Hinv[0, 0] * x + Hinv[0, 1] * y + Hinv[0, 2]) / denom
    sy = (Hinv[1, 0] * x + Hinv[1, 1] * y + Hinv[1, 2]) / denom
    return (float(sx), float(sy))


# ---------- 台呢去背景 ----------

def estimate_felt_hsv(warped: np.ndarray, cfg,
                      hsv: Optional[np.ndarray] = None) -> Tuple[int, int, int]:
    """估计台呢色：绿色掩膜内像素的 HSV 中位数。

    台面（台呢）占绿色区域绝大多数，中位数对球/UI/噪声鲁棒。
    不能用「各通道独立直方图 argmax」——h/s/v 峰值可能来自不同像素，
    拼出不存在于画面的颜色，导致涂灰失败（台面不涂灰 → 色掩膜爆炸）。
    """
    if hsv is None:
        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    # The felt is spatially smooth and occupies most of the table.  Sampling
    # every few pixels avoids building a full-frame boolean index on every
    # analysis frame while retaining enough samples for a robust median.
    stride = 4 if max(hsv.shape[:2]) >= 600 else 2
    sample = hsv[::stride, ::stride]
    green = cv2.inRange(
        sample, (cfg.green_hue_lo, 45, 45), (cfg.green_hue_hi, 255, 255))
    if cv2.countNonZero(green) < 100:
        return (cfg.green_hue_lo + cfg.green_hue_hi) // 2, 120, 120
    pts = sample[green > 0].reshape(-1, 3).astype(np.int32, copy=False)
    fh = int(np.median(pts[:, 0]))
    fs = int(np.median(pts[:, 1]))
    fv = int(np.median(pts[:, 2]))
    # 中位数可能被高光/阴影拉偏：限制在合理范围
    fh = max(0, min(179, fh))
    fs = max(30, min(255, fs))
    fv = max(40, min(255, fv))
    return fh, fs, fv


def felt_like_mask(hsv: np.ndarray, cfg,
                   felt_hsv: Tuple[int, int, int]) -> np.ndarray:
    """Return a uint8 mask for pixels close to the current felt color."""
    fh, fs, fv = (int(felt_hsv[0]), int(felt_hsv[1]), int(felt_hsv[2]))
    hue_lo = max(0, fh - int(cfg.felt_hue_tol))
    hue_hi = min(179, fh + int(cfg.felt_hue_tol))
    sat_lo = max(0, fs - int(cfg.felt_sv_tol))
    sat_hi = min(255, fs + int(cfg.felt_sv_tol))
    val_lo = max(0, fv - int(cfg.felt_sv_tol))
    val_hi = min(255, fv + int(cfg.felt_sv_tol))
    return cv2.inRange(
        hsv, (hue_lo, sat_lo, val_lo), (hue_hi, sat_hi, val_hi))


def blank_self_mask(frame: np.ndarray, self_mask: Optional[np.ndarray],
                    cfg) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """把「叠加层自绘像素」填回台呢色，消除自截屏干扰。

    全屏顶层透明窗画出的瞄准线会被 BitBlt 截屏一起抓进帧里，遮挡
    检测会把自家的线误判为「弹窗面板」。叠加层每帧登记实际画过的
    像素（drawn_mask），截屏端同步快照、分析前原位填回台呢色，
    识别管线就再也看不到自家画的线/点/面板了。

    返回 (ys, xs, 原像素) 还原信息：_save_bad_frame 写盘前据此把
    填回的像素恢复成原始画面。否则存下的「原始帧」已被涂成台呢色，
    诊断时看不到真实遮挡/异常长什么样。
    """
    if self_mask is None or frame is None:
        return None
    if self_mask.shape[:2] != frame.shape[:2]:
        return None
    mask = self_mask > 0
    if not bool(mask.any()):
        return None
    ys, xs = np.nonzero(mask)
    restore = (ys, xs, frame[ys, xs].copy())
    fh, fs, fv = estimate_felt_hsv(frame, cfg)
    bgr = cv2.cvtColor(np.uint8([[[fh, fs, fv]]]), cv2.COLOR_HSV2BGR)[0][0]
    frame[mask] = bgr
    return restore


def _ball_color_protect(h: Optional[np.ndarray], s: Optional[np.ndarray],
                         v: Optional[np.ndarray], cfg,
                         label_masks: Optional[Dict[str, np.ndarray]] = None
                         ) -> np.ndarray:
    """所有「确定是球色」的像素（uint8 0/255），不含黑球。

    袋口涂灰时用它保护袋口挂球：红/黄/绿/棕/蓝/粉/白掩膜都不会命中
    袋洞（洞是暗色），可无条件保护；唯独黑球掩膜与袋洞同色，单独由
    clean_background 里「亮高光核心膨胀」判定（黑球有高光，洞没有）。
    """
    out = np.zeros(h.shape if h is not None else
                   list(label_masks.values())[0].shape[:2], dtype=np.uint8)
    for label in PRIORITY:
        if label == "黑球":
            continue
        if label_masks is not None:
            m = label_masks[label]
        elif label == "绿球":
            m = (h >= 45) & (h <= 100) & (s > 180) & (v > 90)
            m = m.astype(np.uint8)
            np.multiply(m, 255, out=m)
        else:
            m = _mask_for_label(h, s, v, label)
        np.bitwise_or(out, m, out=out)
    return out


def _protect_mask(h: Optional[np.ndarray], s: Optional[np.ndarray],
                  v: Optional[np.ndarray],
                  pockets: Sequence[Point], r: float, cfg,
                  felt_hsv: Optional[Tuple[int, int, int]] = None,
                  felt_like: Optional[np.ndarray] = None,
                  label_masks: Optional[Dict[str, np.ndarray]] = None
                  ) -> np.ndarray:
    """球像素保护掩膜：任何球色覆盖的像素都不被台呢去背景涂掉。

    关键处理：
    - 绿球与台面同色（宽绿掩膜会把整张台面罩住），所以绿球用高饱和紧掩膜
      （s>180）——台面不命中，只有鲜亮的绿球命中；
    - 黑球掩膜会罩住袋洞（深色），保护时把袋口圆形区域挖掉；
    - 若台面本身是浅色低饱和（真实游戏可能如此，v>150 且 s<80 会命中白球
      掩膜），保护掩膜必须排除「与台呢色相近」的像素，否则死循环：
      台面被保护 → 不涂灰 → 整片台面被当成白球候选。
    """
    protect = np.zeros(h.shape if h is not None else
                       list(label_masks.values())[0].shape[:2],
                       dtype=np.uint8)
    for label in PRIORITY:
        if label_masks is not None:
            # 帧级共享字典（见 compute_label_masks）。绿球条目即检测用的
            # 高饱和紧掩膜，逐位一致，可直接复用。
            m = label_masks[label]
        else:
            m = _mask_for_label(h, s, v, label)
        np.bitwise_or(protect, m, out=protect)
    # 排除与台呢色相近的像素（浅色台面防误保护）
    if felt_like is None and felt_hsv is not None:
        felt_like = felt_like_mask(
            cv2.merge((h, s, v)), cfg, felt_hsv)
    if felt_like is not None:
        protect[felt_like > 0] = 0
    # 挖掉袋口（黑球掩膜覆盖袋洞）
    for (px, py) in pockets:
        cv2.circle(protect, (int(px), int(py)), int(1.35 * r), 0, -1)
    return protect > 0


def clean_background(warped: np.ndarray, cfg, r: float,
                     pockets: Sequence[Point] = (),
                     exclude_mask: Optional[np.ndarray] = None,
                     hsv: Optional[np.ndarray] = None,
                     felt_hsv: Optional[Tuple[int, int, int]] = None,
                     label_masks: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
    """把台呢与袋口涂成中性灰，只留球，供颜色掩膜检测。

    - 台呢色用直方图峰值自适应估计（合成图 / 真实游戏 / 不同亮度都适用）；
    - 球像素保护：任何球色覆盖的像素都不涂（绿球用高饱和紧掩膜，
      避免把同色的整张台面误保护；浅色台面用「非台呢色」排除防死循环）；
      袋口按标准位置涂灰防黑球误检。
    """
    if hsv is None:
        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    if label_masks is None:
        # 帧级共享：保护掩膜 + 球色保护一次算完，避免逐色重复全帧运算。
        label_masks = compute_label_masks(h, s, v)
    felt_hsv = (felt_hsv if felt_hsv is not None else
                estimate_felt_hsv(warped, cfg, hsv=hsv))
    felt = felt_like_mask(hsv, cfg, felt_hsv) > 0
    protect = _protect_mask(h, s, v, pockets, r, cfg,
                             felt_hsv=felt_hsv, felt_like=felt,
                             label_masks=label_masks)
    felt = felt & ~(protect > 0)
    # 袋口挂球保护：红/黄/…/白球色像素 + 带亮高光核心的黑球。
    # 详见 _ball_color_protect 注释——袋洞与黑球同色，只能靠高光区分。
    ballish = _ball_color_protect(h, s, v, cfg, label_masks)
    dark_ball = label_masks["黑球"]
    k_core = max(3, int(round(0.7 * r)) | 1)
    bright = (v > int(felt_hsv[2]) + 55).astype(np.uint8)
    np.multiply(bright, 255, out=bright)
    bright = cv2.dilate(bright, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (k_core, k_core)))
    # 黑球暗色主体仅在其高光附近被保护；袋洞无高光 → 仍可涂灰。
    dark_ball = cv2.bitwise_and(dark_ball, bright)
    np.bitwise_or(ballish, dark_ball, out=ballish)
    # 球缘 AA 混色像素不在紧掩膜内，外扩一圈防袋口涂灰吃掉球边。
    ballish = cv2.dilate(ballish, np.ones((3, 3), np.uint8))
    ballish &= ~(felt > 0)
    felt = felt & (ballish == 0)
    clean = warped.copy()
    clean[felt] = (128, 128, 128)
    # 袋口涂灰：只涂洞，不涂袋口挂球
    hole = np.zeros(clean.shape[:2], np.uint8)
    for (px, py) in pockets:
        cv2.circle(hole, (int(px), int(py)), int(1.35 * r), 255, -1, cv2.LINE_AA)
    clean[(hole > 0) & (ballish == 0)] = (128, 128, 128)
    if exclude_mask is not None and exclude_mask.shape == clean.shape[:2]:
        # 连击字样、提示条等已确认 UI 区域不是球候选。直接中性化，后续
        # 所有颜色掩膜和 Hough 分支都会同时看不到它。
        clean[exclude_mask > 0] = (128, 128, 128)
    return clean


def compute_foreign_mask(warped: np.ndarray, cfg, r: float,
                         hsv: Optional[np.ndarray] = None,
                         felt_hsv: Optional[Tuple[int, int, int]] = None,
                         label_masks: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
    """台面内部「外来像素」掩膜（未闭运算）。

    遮挡检测和常驻界面学习共用：只查台面内部（忽略库边/袋口外围），
    排除红球 rack（合法单色大连通域）。
    """
    if warped is None or warped.size == 0:
        return np.zeros((0, 0), np.uint8)
    if hsv is None:
        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    if label_masks is None:
        h, s, v = cv2.split(hsv)
    felt_hsv = (felt_hsv if felt_hsv is not None else
                estimate_felt_hsv(warped, cfg, hsv=hsv))
    felt_like = felt_like_mask(hsv, cfg, felt_hsv)
    # 只在台面内部检查，库边、袋口和 warp 边缘不是界面遮挡。
    margin = max(2.5 * r, 24.0)
    x0, x1 = int(margin), int(warped.shape[1] - margin)
    y0, y1 = int(margin), int(warped.shape[0] - margin)
    foreign = cv2.bitwise_not(felt_like)
    if x1 <= x0 or y1 <= y0:
        return np.zeros_like(foreign)
    foreign[:y0, :] = 0
    foreign[y1:, :] = 0
    foreign[:, :x0] = 0
    foreign[:, x1:] = 0
    # 红球 rack 是合法的单色大连通域，不能把它误判为弹窗；其余颜色
    # 在一帧内通常各只有一颗，超大连通域才有明显的界面特征。
    red_mask = (label_masks["红球"] if label_masks is not None
                else _mask_for_label(h, s, v, "红球"))
    # 球面的阴影和抗锯齿边缘可能已不再满足红色阈值，但仍会与红球
    # 主体连成一个“大块”。只扩张红色掩膜一个小球半径来保护邻域，
    # 不直接删除所有 foreign，避免真正覆盖台面的白/灰弹窗被放行。
    protect_radius = max(3, int(round(0.75 * r)) | 1)
    red_neighborhood = cv2.dilate(
        red_mask,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (protect_radius, protect_radius)),
    )
    foreign[red_neighborhood > 0] = 0
    return foreign


def detect_table_occlusion(warped: np.ndarray, cfg, r: float,
                           static_mask: Optional[np.ndarray] = None,
                           hsv: Optional[np.ndarray] = None,
                           foreign: Optional[np.ndarray] = None,
                           felt_hsv: Optional[Tuple[int, int, int]] = None) -> Optional[Dict]:
    """检测台面内部是否被 QQ 菜单/弹窗等大块界面覆盖。

    弹窗的白色或灰色面板会被白球掩膜和 watershed 切成大量假球。
    这类帧即使球数校验侥幸通过，也不应继续计算瞄准线。真实球、红球
    rack 和台面标线都远小于一个 UI 面板，因此只检查台面内部的大型
    非台呢连通域，并返回诊断信息供上层显示。

    static_mask: 上层积累的「常驻外来区域」（游戏固定界面元素），
    掩膜>0 的位置不再视为遮挡——一次性学会的常驻 UI 不误报，
    临时出现的真弹窗仍会命中。
    """
    if warped is None or warped.size == 0:
        return None
    if foreign is None:
        foreign = compute_foreign_mask(warped, cfg, r, hsv=hsv,
                                       felt_hsv=felt_hsv)
    if foreign.size == 0:
        return None
    if static_mask is not None and static_mask.shape == foreign.shape:
        foreign = foreign.copy()
        foreign[static_mask > 0] = 0
    # 颜色球会留下小块噪声；闭运算只连接面板内部的抗锯齿/文字间隙，
    # 不把相邻球扩大成一整块。
    foreign = cv2.morphologyEx(
        foreign, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
    )
    n, lab, stats, _ = cv2.connectedComponentsWithStats(foreign, 8)
    min_area = max(3.0 * np.pi * r * r, 0.01 * warped.shape[0] * warped.shape[1])
    for idx in range(1, n):
        area = float(stats[idx, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        bx = int(stats[idx, cv2.CC_STAT_LEFT])
        by = int(stats[idx, cv2.CC_STAT_TOP])
        bw = int(stats[idx, cv2.CC_STAT_WIDTH])
        bh = int(stats[idx, cv2.CC_STAT_HEIGHT])
        if min(bw, bh) < 3.0 * r:
            # 细条状不是界面遮挡：UI 面板两个方向都要够大。原判断只
            # 跳过「两个方向都小」的块，水平/垂直瞄准时游戏的球杆是
            # 长条矩形（fill≈1），会被误判成弹窗——这就是无遮挡却
            # 提示「被菜单/弹窗遮挡」的原因。
            continue
        fill = area / max(1.0, float(bw * bh))
        if fill < 0.18:
            continue
        # 纯暗大块是球群阴影/袋口/黑球，不是界面；真菜单是白/灰亮面板。
        comp_px = warped[lab == idx]
        if float(comp_px.max(axis=1).mean()) < 60.0:
            continue
        return {
            "area": area,
            "bbox": (bx, by, bw, bh),
            "fill": fill,
        }
    return None


def transient_ui_mask(warped: np.ndarray, cfg, r: float,
                      hsv: Optional[np.ndarray] = None,
                      gray: Optional[np.ndarray] = None,
                      foreign: Optional[np.ndarray] = None,
                      felt_hsv: Optional[Tuple[int, int, int]] = None,
                      label_masks: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
    """返回应从球检测中排除的台面 UI 像素。

    计分连击、文字提示由许多小笔画组成，单个笔画并不会触发“大面板”
    遮挡保护，却足以被黑球 HSV 掩膜和 Hough 误认。这里仅把彼此接近
    且整体明显大于一颗球的非台呢区域合并为 UI；正常单球不受影响。
    """
    if foreign is None:
        foreign = compute_foreign_mask(warped, cfg, r, hsv=hsv,
                                       felt_hsv=felt_hsv,
                                       label_masks=label_masks)
    if foreign.size == 0:
        return foreign
    if hsv is None:
        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    if gray is None:
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    # 白球验证（连通域+距离变换）只有候选框真与白色像素相交才需要：
    # 绝大多数帧没有 UI 文字，白球验证可整帧跳过（曾是每帧数毫秒
    # 的固定成本）。无相交时 protected≡全零，与旧实现逐位一致。
    white = (label_masks["白球"] if label_masks is not None
             else _mask_for_label(hsv[:, :, 0], hsv[:, :, 1],
                                  hsv[:, :, 2], "白球"))
    protected: Optional[np.ndarray] = None
    # Close the raw foreign pixels before deciding whether a component is UI.
    # Keep the grouping kernel below a ball diameter: it joins character
    # strokes but does not merge a nearby cue stick with the cue ball.
    k = max(3, int(round(float(getattr(cfg, "ui_group_kernel_ratio", 0.80)) * r)) | 1)
    grouped = cv2.morphologyEx(
        foreign, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
    )
    n, labels, stats, _ = cv2.connectedComponentsWithStats(grouped, 8)
    ui = np.zeros_like(foreign)
    ball_area = np.pi * r * r
    ui_boxes = []
    for idx in range(1, n):
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        width = int(stats[idx, cv2.CC_STAT_WIDTH])
        height = int(stats[idx, cv2.CC_STAT_HEIGHT])
        # Work inside the component bbox.  A full-frame labels==idx and
        # nonzero() pair for every glyph is disproportionately expensive at
        # the 800-1000px analysis size.
        component = labels[y:y + height, x:x + width] == idx
        box = _component_ui_box(component, x, y, r, ball_area)
        if box is None:
            continue
        if (protected is None
                and (white[y:y + height, x:x + width] > 0).any()):
            protected = _verified_white_ball_mask(
                warped, hsv, gray, r, cfg, white_mask=white)
        if protected is not None:
            comp2 = component & ~(
                protected[y:y + height, x:x + width] > 0)
            box2 = _component_ui_box(comp2, x, y, r, ball_area)
            if box2 is None:
                continue
            component, box = comp2, box2
        ui_roi = ui[y:y + height, x:x + width]
        ui_roi[component] = 255
        ui_boxes.append(box)
    if ui_boxes:
        # A leading digit or glyph can remain a separate raw component when
        # its gap is just wider than the close kernel.  Join only components
        # with non-circular/text-like geometry; a real adjacent ball is kept.
        raw_n, raw_labels, raw_stats, _ = cv2.connectedComponentsWithStats(foreign, 8)
        for idx in range(1, raw_n):
            x = float(raw_stats[idx, cv2.CC_STAT_LEFT])
            y = float(raw_stats[idx, cv2.CC_STAT_TOP])
            bw = float(raw_stats[idx, cv2.CC_STAT_WIDTH])
            bh = float(raw_stats[idx, cv2.CC_STAT_HEIGHT])
            area = float(raw_stats[idx, cv2.CC_STAT_AREA])
            ix, iy = int(x), int(y)
            iw, ih = int(bw), int(bh)
            component = (raw_labels[iy:iy + ih, ix:ix + iw] == idx).astype(np.uint8)
            contours, _ = cv2.findContours(
                component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            perimeter = cv2.arcLength(contours[0], True)
            circularity = (4.0 * np.pi * area / (perimeter * perimeter)
                           if perimeter > 1e-6 else 0.0)
            text_like = circularity < 0.68 or area < 0.75 * ball_area
            if not text_like:
                continue
            for ux, uy, uw, uh in ui_boxes:
                gap_x = max(0.0, x - (ux + uw), ux - (x + bw))
                gap_y = max(0.0, y - (uy + uh), uy - (y + bh))
                if gap_x <= 1.5 * r and gap_y <= 1.5 * r:
                    ui[iy:iy + ih, ix:ix + iw][component > 0] = 255
                    break
    if not ui.any():
        return ui
    ui = cv2.dilate(ui, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    # Dilation must not grow back over a verified cue ball.
    if protected is not None:
        ui[protected > 0] = 0
    return ui


def _component_ui_box(component: np.ndarray, x: int, y: int,
                      r: float, ball_area: float
                      ) -> Optional[Tuple[float, float, float, float]]:
    """组件若呈 UI 几何特征（大面积或宽长条）→ 局部外接框，否则 None。"""
    ys, xs = np.nonzero(component)
    if len(xs) == 0:
        return None
    area = float(len(xs))
    local_x = float(xs.min())
    local_y = float(ys.min())
    bw = float(xs.max() - local_x + 1.0)
    bh = float(ys.max() - local_y + 1.0)
    # 多字符 UI 通常是宽而高的组合体；单颗球的扩张面积仍接近 πr²。
    is_large = area >= 2.6 * ball_area
    fill = area / max(1.0, bw * bh)
    is_wide = (max(bw, bh) >= 4.0 * r
               and min(bw, bh) >= 1.15 * r
               and fill >= 0.20)
    if not (is_large or is_wide):
        return None
    return (x + local_x, y + local_y, bw, bh)


def _verified_white_ball_mask(warped: np.ndarray, hsv: np.ndarray,
                              gray: np.ndarray, r: float, cfg,
                              white_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Return disks of white balls that are safe to preserve from a UI mask.

    White outlined glyphs can satisfy a simple occupancy test.  The component
    geometry and distance peak gates below require a near-full-size circular
    disk, so a word such as COMBO is not mistaken for the cue ball.
    """
    protected = np.zeros(hsv.shape[:2], dtype=np.uint8)
    if r < 3.0:
        return protected
    white = (white_mask if white_mask is not None
             else _mask_for_label(hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2], "白球"))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(white, 8)
    edge_threshold = float(getattr(cfg, "circle_min_edge_coverage", 0.42))
    for idx in range(1, n):
        x, y, bw, bh, area = (int(stats[idx, cv2.CC_STAT_LEFT]),
                               int(stats[idx, cv2.CC_STAT_TOP]),
                               int(stats[idx, cv2.CC_STAT_WIDTH]),
                               int(stats[idx, cv2.CC_STAT_HEIGHT]),
                               int(stats[idx, cv2.CC_STAT_AREA]))
        if area < 0.75 * np.pi * r * r:
            continue
        aspect = max(bw, bh) / max(1.0, min(bw, bh))
        if min(bw, bh) < 1.45 * r or aspect > 1.30 or max(bw, bh) > 2.7 * r:
            continue
        component = (labels[y:y + bh, x:x + bw] == idx).astype(np.uint8)
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        perimeter = cv2.arcLength(contours[0], True)
        circularity = (4.0 * np.pi * area / (perimeter * perimeter)
                       if perimeter > 1e-6 else 0.0)
        if circularity < 0.72:
            continue
        distance = cv2.distanceTransform(component, cv2.DIST_L2, 5)
        _, peak, _, local = cv2.minMaxLoc(distance)
        if peak < 0.72 * float(r):
            continue
        cx, cy = x + int(local[0]), y + int(local[1])
        pos = (float(cx), float(cy))
        if not _solid_white_disc(hsv, pos, r):
            continue
        if circle_edge_coverage(gray, pos, r) < edge_threshold:
            continue
        cv2.circle(protected, (cx, cy), int(round(1.35 * r)), 255, -1)
    return protected


# ---------- 亚像素球心拟合 ----------

def _kasa_fit(points: np.ndarray, weights: Optional[np.ndarray]) -> Optional[Tuple[float, float, float]]:
    """Kasa 最小二乘圆拟合：解 2x + 2y + c = x²+y² 的线性系统。返回 (cx, cy, radius)。"""
    if len(points) < 5:
        return None
    x, y = points[:, 0], points[:, 1]
    A = np.stack([2 * x, 2 * y, np.ones_like(x)], axis=1)
    b = x * x + y * y
    if weights is not None:
        w = weights.astype(np.float64)
        A = A * w[:, None]
        b = b * w
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy, c = float(sol[0]), float(sol[1]), float(sol[2])
    r2 = c + cx * cx + cy * cy
    if r2 <= 0:
        return None
    return (cx, cy, float(np.sqrt(r2)))


def fit_ball_edges(gray: np.ndarray, cx: float, cy: float, r: float,
                   window: float = 1.6, edge_min: float = 8.0) -> Optional[Tuple[float, float, float]]:
    """在粗定位球心 (cx,cy) 邻域内提取强梯度边缘点，做加权圆拟合。

    关键：只保留「圆环带」内的边缘点（距粗心 0.65r~1.5r）——排除球内部
    径向渐变/高光产生的伪边缘，也排除远离球体的背景干扰。
    返回亚像素 (cx, cy, r)；边缘点不足或拟合异常时返回 None（调用方回退粗定位）。
    """
    win = max(6, int(r * window))
    x0, x1 = max(0, int(cx) - win), min(gray.shape[1], int(cx) + win + 1)
    y0, y1 = max(0, int(cy) - win), min(gray.shape[0], int(cy) + win + 1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    patch = gray[y0:y1, x0:x1].astype(np.float32)
    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    thr = max(30.0, float(mag.mean()) + 1.5 * float(mag.std()))
    ys, xs = np.nonzero(mag > thr)
    if len(xs) < edge_min:
        return None
    pts = np.stack([xs.astype(np.float64) + x0, ys.astype(np.float64) + y0], axis=1)
    wgt = mag[ys, xs].astype(np.float64)
    # 圆环带过滤
    d = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
    ring = (d > 0.65 * r) & (d < 1.5 * r)
    pts, wgt = pts[ring], wgt[ring]
    if len(pts) < edge_min:
        return None
    fit = _kasa_fit(pts, wgt)
    if fit is None:
        return None
    fcx, fcy, fr = fit
    # 合理性校验：球心不能离粗定位太远，半径不能离谱
    if abs(fcx - cx) > 0.85 * r or abs(fcy - cy) > 0.85 * r:
        return None
    if fr < 0.45 * r or fr > 1.7 * r:
        return None
    return (fcx, fcy, fr)


def fit_ball_mask(mask: np.ndarray, cx: float, cy: float, r: float,
                  max_shift: float = 0.35) -> Optional[Tuple[float, float, float]]:
    """用球色掩膜的外轮廓拟合球心，返回亚像素中心与半径。

    灰度梯度里同时包含球面高光、渐变环和库边反光，直接对全部梯度
    做圆拟合会把中心拉向高光。颜色掩膜的连通域只保留球的外轮廓，
    对真实实心球更稳定；灰度圆拟合作为轮廓不足时的后备路径。
    """
    win = max(8, int(1.8 * r))
    x0, x1 = max(0, int(cx) - win), min(mask.shape[1], int(cx) + win + 1)
    y0, y1 = max(0, int(cy) - win), min(mask.shape[0], int(cy) + win + 1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    crop = mask[y0:y1, x0:x1]
    contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    target_area = np.pi * r * r
    candidates = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 0.35 * target_area:
            continue
        m = cv2.moments(contour)
        if m["m00"] < 1e-6:
            continue
        mx = float(m["m10"] / m["m00"] + x0)
        my = float(m["m01"] / m["m00"] + y0)
        candidates.append((np.hypot(mx - cx, my - cy), contour, mx, my, area))
    if not candidates:
        return None
    _, contour, mx, my, area = min(candidates, key=lambda item: item[0])

    # 轮廓过大通常说明候选落在粘连区域，不能把整块 rack 当作单球拟合。
    if area > 2.4 * target_area or len(contour) < 5:
        return None
    points = contour.astype(np.float32)
    ellipse = cv2.fitEllipse(points)
    ex, ey = float(ellipse[0][0] + x0), float(ellipse[0][1] + y0)
    er = 0.25 * (float(ellipse[1][0]) + float(ellipse[1][1]))
    # 对局部遮挡/粘连轮廓，质心比椭圆中心更保守。
    if np.hypot(ex - mx, ey - my) > 0.25 * r:
        ex, ey = mx, my
    if np.hypot(ex - cx, ey - cy) > max_shift * r:
        return None
    if not (0.65 * r <= er <= 1.45 * r):
        return None
    return (ex, ey, er)


# ---------- 球检测 ----------

def _mask_for_label(h: np.ndarray, s: np.ndarray, v: np.ndarray, label: str) -> np.ndarray:
    """单色向量化掩膜（由 BALL_HSV_RULES 生成，与 classify_pixel 同一规则表）。"""
    # All label predicates are comparisons, so converting every full-frame
    # channel to int16 here only creates needless temporaries.  Callers that
    # need signed subtraction (felt matching) perform that conversion once at
    # their boundary.
    hh, ss, vv = h, s, v
    rule = BALL_HSV_RULES.get(label)
    if rule is None:
        return np.zeros(hh.shape, dtype=np.uint8)
    h_ranges, (s_lo, s_hi), (v_lo, v_hi) = rule
    ok = (ss >= s_lo) & (ss <= s_hi) & (vv >= v_lo) & (vv <= v_hi)
    if h_ranges is not None:
        hm = np.zeros(hh.shape, dtype=bool)
        for lo, hi in h_ranges:
            hm |= (hh >= lo) & (hh <= hi)
        ok &= hm
    # 每帧 20+ 次调用曾是热点：(ok * 255) 生成 int64 临时数组，
    # astype 再拷一份。bool 与 uint8 等宽，视图重解释后原地 ×255
    # 零新增分配，结果与旧实现逐位一致（0/255）。
    if not ok.flags.c_contiguous:
        ok = np.ascontiguousarray(ok)
    out = ok.view(np.uint8)
    np.multiply(out, 255, out=out)
    return out


def compute_label_masks(h: np.ndarray, s: np.ndarray, v: np.ndarray
                        ) -> Dict[str, np.ndarray]:
    """一次算出全部 8 个颜色掩膜，供台呢清理/外来检测/球检测共用。

    同一帧内 _protect_mask、compute_foreign_mask、detect_balls 各自
    调用 _mask_for_label 会把同一组全帧布尔运算重复三遍；按帧算
    一份共享字典后每帧只算一遍。
    """
    return {label: _mask_for_label(h, s, v, label) for label in PRIORITY}


def _split_blobs(mask: np.ndarray, color: np.ndarray, r: float,
                 allow_large: bool = False) -> List[Tuple[float, float]]:
    """把（可能粘连多球的）掩膜连通域拆成单球粗定位。

    对每块独立连通域：
      - 面积 ~ 单球且圆度好 → 直接质心；
      - 面积明显 > 单球（粘连/相切，如红球三角）→ watershed 分水岭：
        距离变换峰值做种子，按彩色梯度精确分割每颗球，返回各区域质心。
    返回 (cx, cy) 列表（整数像素级粗定位，亚像素在 fit 阶段做）。
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: List[Tuple[float, float]] = []
    area_min = 0.45 * np.pi * r * r
    single_max = 1.55 * np.pi * r * r
    for c in contours:
        area = cv2.contourArea(c)
        if area < area_min:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        if not allow_large:
            # 游戏提示条、右键菜单和设置窗口会形成远大于单球的
            # 连通域。若把它们送入 watershed，会被切成几十个假球。
            # 非红球不需要处理相切 rack，因此直接丢弃大块/细长块。
            if area > 4.0 * np.pi * r * r:
                continue
            aspect = max(bw, bh) / max(1.0, min(bw, bh))
            if aspect > 2.4 and area > 1.4 * np.pi * r * r:
                continue
        peri = cv2.arcLength(c, True)
        circ = 4.0 * np.pi * area / (peri * peri) if peri > 0 else 0.0
        # 单球：圆度达标且面积在正常范围 → 质心
        if area <= single_max and circ >= 0.55:
            m = cv2.moments(c)
            if m["m00"] < 1e-6:
                continue
            out.append((float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"])))
            continue
        # 粘连多球：watershed 分割
        sub_mask = mask[y:y + bh, x:x + bw]
        sub_color = color[y:y + bh, x:x + bw]
        dist = cv2.distanceTransform((sub_mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
        dil = cv2.dilate(dist, np.ones((3, 3), np.uint8))
        peaks = ((dist >= dil) & (dist > 0.4 * r)).astype(np.uint8)
        if peaks.sum() < 2:
            m = cv2.moments(c)
            if m["m00"] > 1e-6:
                out.append((float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"])))
            continue
        # 峰值膨胀簇化：平顶等值会产生像素级碎块，膨胀后每颗球合并成单一 marker
        ksize = max(3, int(0.5 * r) | 1)
        peaks = cv2.dilate(peaks, np.ones((ksize, ksize), np.uint8))
        n, markers = cv2.connectedComponents(peaks)
        # 注：曾试验「0=未知、1=背景」的标准三段式标记，实测在低梯度
        # 纯色球上整片被背景吞掉；现有「掩膜内全为种子/背景」方案
        # 由 watershed 只在种子间划界，效果稳定，保留。
        markers = markers + 1                       # 峰值簇 → 2..n+1
        markers[sub_mask == 0] = 1                  # 掩膜外=确定背景
        try:
            cv2.watershed(sub_color, markers)
        except cv2.error:
            m = cv2.moments(c)
            if m["m00"] > 1e-6:
                out.append((float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"])))
            continue
        for label in range(2, n + 2):
            region = markers == label
            ys, xs = np.nonzero(region)
            if len(xs) < 8:
                continue
            out.append((float(xs.mean() + x), float(ys.mean() + y)))
    return out


def _detect_label(clean: np.ndarray, hsv: np.ndarray, label: str, r: float,
                  subpixel: bool, cfg,
                  gray: Optional[np.ndarray] = None,
                  mask: Optional[np.ndarray] = None) -> List[Ball]:
    if mask is None:
        h, s, v = cv2.split(hsv)
        mask = _mask_for_label(h, s, v, label)
    if gray is None:
        gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
    balls: List[Ball] = []
    for (cx, cy) in _split_blobs(mask, clean, r, allow_large=(label == "红球")):
        pos, radius = (cx, cy), r
        sub = False
        if subpixel:
            # 优先用颜色外轮廓，避免灰度高光把球心拉偏。红球 rack
            # 的内部边缘不可见，保留 watershed 粗中心交给整体网格拟合。
            # 红球仍沿用 rack 管线的粗候选，避免相切球在透视边界微调后
            # 被局部拟合误合并；当前球径由彩球/白球估计已足够稳定。
            fit = None if label == "红球" else fit_ball_mask(mask, cx, cy, r)
            if fit is None and label != "红球":
                fit = fit_ball_edges(gray, cx, cy, r,
                                     cfg.subpixel_window, cfg.subpixel_edges_min)
            if fit is not None:
                pos, radius, sub = (fit[0], fit[1]), fit[2], True
        balls.append(Ball(label, pos, float(radius), sub))
    return balls


def refine_red_rack(centers: Sequence[Point], r: float,
                    bounds: Optional[Tuple[float, float, float, float]] = None,
                    mask: Optional[np.ndarray] = None
                    ) -> Optional[List[Point]]:
    """红球三角（斯诺克开局）行结构拟合精修。

    相切/重叠的红球内部无可见边缘，局部定位误差大；但开局三角是
    规则排列（行水平、行距 sqrt(3)*r、球心距 2r），利用该先验精修
    精确网格。只在完整的 15 红球候选上启用，避免把已经打掉的红球
    错误重建回来；若候选不呈规则三角（球已散开）则返回 None。
    """
    if (bounds is None and len(centers) != 15) or (bounds is not None and len(centers) < 15):
        return None
    import math

    # 完整 rack 的红色掩膜是一个连通整体，外接边界比内部 watershed
    # 种子更可靠。宽度约为 10r（两端各一个球半径 + 4 个球心间距），
    # 因此可同时吸收透视校正后的真实半径和固定配置误差。
    if bounds is not None:
        min_x, max_x, min_y, max_y = bounds
        # 透视校正后的合成/实机截图可能仍有轻微纵横比例误差，
        # 横向球径和纵向球径分别由 rack 外框估计。
        radius_x = (max_x - min_x + 1.0) / 10.0
        radius_y = (max_y - min_y + 1.0) / (2.0 + 4.0 * math.sqrt(3.0))
        if not (0.70 * r <= radius_x <= 1.45 * r
                and 0.70 * r <= radius_y <= 1.45 * r):
            return None
        cx0 = 0.5 * (min_x + max_x)
        y0 = min_y + radius_y
        row_h = math.sqrt(3.0) * radius_y
        grid = [
            (float(cx0 + (j - i / 2.0) * 2.0 * radius_x),
             float(y0 + i * row_h))
            for i in range(5) for j in range(i + 1)
        ]
        if mask is not None:
            # 每个预测球心附近都必须确实存在红色像素；这是防止
            # 缺球/散开局面被完整 rack 先验凭空补回的关键校验。
            # 局部小窗代替全帧 mgrid：15 个探针 × 全帧布尔运算曾是
            # rack 帧的显著热点（探针半径只有 ~0.4r，全帧运算几乎
            # 全是浪费）。
            probe_r = 0.42 * min(radius_x, radius_y)
            pr = max(1, int(math.ceil(probe_r)) + 1)
            mh, mw = mask.shape[:2]
            for gx, gy in grid:
                x0 = max(0, int(math.floor(gx)) - pr)
                x1 = min(mw, int(math.floor(gx)) + pr + 1)
                y0 = max(0, int(math.floor(gy)) - pr)
                y1 = min(mh, int(math.floor(gy)) + pr + 1)
                if x1 <= x0 or y1 <= y0:
                    return None
                ly, lx = np.mgrid[y0:y1, x0:x1]
                inside = ((lx - gx) ** 2 + (ly - gy) ** 2
                          <= probe_r * probe_r)
                if not inside.any():
                    return None
                if float((mask[y0:y1, x0:x1][inside] > 0).mean()) < 0.65:
                    return None
        return grid

    row_h = math.sqrt(3.0) * r
    # 1. 按 y 聚成行（行均值迭代，容差 0.6*row_h）
    rows: List[List[Point]] = []
    for p in sorted(centers, key=lambda q: q[1]):
        best_i, best_d = -1, 1e18
        for i, row in enumerate(rows):
            my = np.mean([q[1] for q in row])
            d = abs(my - p[1])
            if d < best_d:
                best_d, best_i = d, i
        if best_i >= 0 and best_d < 0.6 * row_h:
            rows[best_i].append(p)
        else:
            rows.append([p])
    rows.sort(key=lambda row: np.mean([q[1] for q in row]))
    n_rows = len(rows)
    if n_rows != 5:
        return None
    ys = [float(np.mean([q[1] for q in row])) for row in rows]
    for i in range(1, n_rows):
        if abs(ys[i] - ys[i - 1] - row_h) > 0.45 * row_h:
            return None
    # 3. 行内候选数校验：行 i 期望 i+1 颗；允许行内冗余候选（≤ e+3），缺失过多则失败
    expected = list(range(1, n_rows + 1))
    counts = [len(row) for row in rows]
    if any(c > e + 3 or c < e - 1 for c, e in zip(counts, expected)):
        return None
    # 4. 重建网格：行 y 用「等距拟合」（y0 + i*row_h），吸收 watershed
    #    Voronoi 区域的不均匀偏移；顶点 x 用所有行中心均值（三角对称）。
    y0 = float(np.mean([ys[i] - i * row_h for i in range(n_rows)]))
    cx0 = float(np.mean([np.mean([q[0] for q in row]) for row in rows]))
    grid: List[Point] = []
    for i in range(n_rows):
        n_ball = expected[i]
        y_row = y0 + i * row_h
        for j in range(n_ball):
            x = cx0 + (j - (n_ball - 1) / 2.0) * 2.0 * r
            grid.append((float(x), y_row))
    # 5. 整体吻合校验：网格点与候选点平均距离 < 1.4r 才采纳
    cand = list(centers)
    used = [False] * len(cand)
    total = 0.0
    worst = 0.0
    for gx, gy in grid:
        best, bi = 1e18, -1
        for i, (px, py) in enumerate(cand):
            if used[i]:
                continue
            d = (px - gx) ** 2 + (py - gy) ** 2
            if d < best:
                best, bi = d, i
        if bi < 0:
            return None
        used[bi] = True
        dist = best ** 0.5
        total += dist
        worst = max(worst, dist)
    # 平均距离与最远点双重校验。只查平均时，15 个候选里混入一颗远处
    # 的多余候选、某颗真球缺失也能通过（其余 14 颗距离极小把平均值
    # 拉低），先验网格会把已打掉的红球“复活”出来指向不存在的球。
    # 任一网格点 1.4r 内无候选 → 整体拒绝重建。
    if len(grid) != 15 or not all(used):
        return None
    if total / len(grid) > 1.4 * r or worst > 1.4 * r:
        return None
    return grid


def detect_balls(warped: np.ndarray, r: float, cfg=None,
                 pockets: Sequence[Point] = (),
                 clean: Optional[np.ndarray] = None,
                 exclude_mask: Optional[np.ndarray] = None,
                 warped_hsv: Optional[np.ndarray] = None,
                 warped_gray: Optional[np.ndarray] = None) -> List[Ball]:
    """在标准台面图上检测全部球（斯诺克色板）。

    - clean 已由调用方去背景时直接复用；否则内部调用 clean_background。
    - cfg 可空（用默认 Config）；pockets 为空时按标准 6 袋位置涂灰。
    - 红球若呈规则三角（开局），用行结构拟合精修球心。
    """
    if cfg is None:
        from aimtool.config import Config
        cfg = Config()
    from aimtool.physics import default_pockets
    if not pockets:
        pockets = default_pockets(warped.shape[1], warped.shape[0])
    if clean is None:
        clean = clean_background(warped, cfg, r, pockets, exclude_mask)
    elif exclude_mask is not None and exclude_mask.shape == clean.shape[:2]:
        clean = clean.copy()
        clean[exclude_mask > 0] = (128, 128, 128)
    hsv = cv2.cvtColor(clean, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    label_masks = compute_label_masks(h, s, v)
    clean_gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
    found: List[Ball] = []
    for label in PRIORITY:
        found.extend(_detect_label(clean, hsv, label, r, cfg.subpixel, cfg,
                                   gray=clean_gray, mask=label_masks[label]))
    # 保留合并前的红球候选。watershed 在相切 rack 的接触缝附近可能
    # 产生冗余种子，先合并会把某一颗真实球一起吞掉；完整 rack 的整体
    # 边界拟合可以直接利用这些冗余候选并用掩膜覆盖率校验。
    raw_reds = [b for b in found if b.label == "红球"]

    # Hough is useful for a heavily occluded or very small scene, but it is
    # expensive and redundant when the color masks already found enough balls.
    # Keep it as a conservative fallback so normal frames stay on the fast
    # path without losing recovery on low-candidate frames.
    hough_enabled = bool(getattr(cfg, "hough_fallback", True))
    hough_trigger = max(0, int(getattr(cfg, "hough_trigger_ball_count", 4)))
    need_hough = (hough_enabled and
                  (len(found) < hough_trigger or
                   not any(b.label == "白球" for b in found)))
    if need_hough:
        gray = cv2.GaussianBlur(clean_gray, (5, 5), 0)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=2.0 * r,
            param1=100, param2=26,
            minRadius=int(0.6 * r), maxRadius=int(1.45 * r),
        )
        if circles is not None:
            for cx, cy, cr in circles[0]:
                px = max(0, min(clean.shape[1] - 1, int(cx)))
                py = max(0, min(clean.shape[0] - 1, int(cy)))
                label = classify_pixel(hsv[py, px])
                if label != "未知" and label != "红球":
                    # Hough 也会在白色弹窗/灰色按钮上找到规则圆。候选中心
                    # 对应的颜色连通域若远大于单球，说明它属于界面而非球。
                    hmask = label_masks[label]
                    _, labs, cstats, _ = cv2.connectedComponentsWithStats(hmask, 8)
                    lid = int(labs[py, px])
                    if (lid > 0 and
                            float(cstats[lid, cv2.CC_STAT_AREA]) > 4.0 * np.pi * r * r):
                        continue
                    found.append(Ball(label, (float(cx), float(cy)), float(cr)))

    # 合并去重：同一位置只保留半径更接近 r 的。红球 watershed
    # 会在相切边缘产生多个相距约 1r 的候选，保留更多候选交给 rack
    # 整体拟合；非红球仍使用较宽阈值抑制 Hough 重复圆。
    # 红球单独使用更窄的合并阈值，避免相切 rack 的碎块候选被提前吞掉；
    # 最终是否为完整 rack 由整体边界和掩膜覆盖率再次校验。
    merged: List[Ball] = []
    # Hough 仅用于补漏；同一位置已有颜色外轮廓/边缘亚像素结果时，
    # 不能因为 Hough 半径碰巧更接近配置值就把高置信中心替换掉。
    for b in sorted(found, key=lambda x: (0 if x.subpixel else 1,
                                          abs(x.radius - r))):
        merge_r = 1.1 * r if b.label == "红球" else 1.3 * r
        if all(abs(b.pos[0] - m.pos[0]) > merge_r or abs(b.pos[1] - m.pos[1]) > merge_r
               for m in merged):
            merged.append(b)

    # 红球三角行拟合精修（斯诺克开局）。只有完整 15 球且红色区域
    # 仍是一个大连通 rack 时才使用整体边界；散开球或缺球不重建。
    reds = [b for b in merged if b.label == "红球"]
    if cfg.rack_fit and len(raw_reds) >= 15:
        # 上方 label_masks 已在 clean HSV 上算过红球掩膜，这里不再重算。
        red_mask = cv2.morphologyEx(
            label_masks["红球"], cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        n_red, red_labels, red_stats, _ = cv2.connectedComponentsWithStats(red_mask, 8)
        red_bounds = None
        if n_red > 1:
            largest = 1 + int(np.argmax(red_stats[1:, cv2.CC_STAT_AREA]))
            total_area = float(red_stats[1:, cv2.CC_STAT_AREA].sum())
            largest_area = float(red_stats[largest, cv2.CC_STAT_AREA])
            if total_area > 0 and largest_area / total_area >= 0.72:
                ys, xs = np.nonzero(red_labels == largest)
                if len(xs) > 0:
                    red_bounds = (float(xs.min()), float(xs.max()),
                                  float(ys.min()), float(ys.max()))
        rack_source = raw_reds if len(raw_reds) >= 15 else reds
        grid = refine_red_rack([b.pos for b in rack_source], r, red_bounds, red_mask)
        if grid is not None:
            others = [b for b in merged if b.label != "红球"]
            merged = others + [Ball("红球", g, float(r), False) for g in grid]

    # 实心盘校验：真球是实心色盘；台面白色标线（D 区弧、开球线）切出的
    # 假白球、游戏提示气泡（「普通击球，交换选手」等）橙色文字笔画切出的
    # 假棕球，都是细线/笔画碎块，圆盘内该颜色占比低 → 剔除。
    # 不做校验的后果：假白球抢母球锁定；假棕球堆高总数触发「识别异常」拒出方案。
    hsv_w = warped_hsv if warped_hsv is not None else cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    gray_w = warped_gray if warped_gray is not None else cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    kept: List[Ball] = []
    for b in merged:
        if b.label == "白球":
            if _solid_white_disc(hsv_w, b.pos, b.radius):
                coverage = circle_edge_coverage(gray_w, b.pos, b.radius)
                if coverage >= float(getattr(cfg, "circle_min_edge_coverage", 0.42)):
                    b.confidence = coverage
                    kept.append(b)
            continue
        if b.label == "红球":
            kept.append(b)      # 红球有 rack 拟合兜底，且三角碎块占比低但真实
            continue
        min_ratio = 0.56 if b.label == "黑球" else 0.40
        if _solid_label_disc(hsv_w, b.pos, b.radius, b.label, min_ratio):
            # 黑/白球最容易和文字、标线混淆；绿色球与台呢的灰度差
            # 很小，不能用同一圆周梯度门，否则会把真实绿球拒掉。
            if b.label in {"黑球", "白球"}:
                coverage = circle_edge_coverage(gray_w, b.pos, b.radius)
                if coverage < float(getattr(cfg, "circle_min_edge_coverage", 0.42)):
                    continue
                b.confidence = coverage
            kept.append(b)
    return kept


def estimate_ball_radius(balls: Sequence[Ball], fallback: float) -> float:
    """从多个亚像素球候选估计当前台面的实际球半径。

    透视校正、缩放和游戏皮肤会让固定配置半径与实际像素半径有小偏差。
    使用非红球的中位数可避免开局 rack 的重建半径和红球粘连候选污染物理计算。
    """
    vals = [float(b.radius) for b in balls
            if b.subpixel and b.label != "红球"
            and 0.70 * fallback <= b.radius <= 1.35 * fallback]
    if len(vals) < 3:
        vals = [float(b.radius) for b in balls
                if b.subpixel and 0.70 * fallback <= b.radius <= 1.35 * fallback]
    if not vals:
        return float(fallback)
    median = float(np.median(vals))
    # 中位数已经很稳，但再用 MAD 排除极端拟合值。
    dev = np.abs(np.asarray(vals) - median)
    mad = float(np.median(dev))
    if mad > 1e-6:
        good = np.asarray(vals)[dev <= max(1.0, 3.0 * mad)]
        if len(good) >= 3:
            median = float(np.median(good))
    return float(np.clip(median, 0.80 * fallback, 1.25 * fallback))


def _solid_label_disc(hsv: np.ndarray, pos: Point, r: float, label: str,
                      min_ratio: float = 0.40) -> bool:
    """彩色球实心盘校验：圆盘内该分类颜色的像素占比 ≥ min_ratio。

    真球实心色盘占比 ≈0.85+（高光/阴影边缘略降）；文字笔画、线条
    碎块只在小部分面积上匹配颜色 → 剔除。
    """
    h, w = hsv.shape[:2]
    x, y = int(round(pos[0])), int(round(pos[1]))
    rr = max(int(round(r)) - 2, 3)
    x0, x1 = max(0, x - rr), min(w, x + rr + 1)
    y0, y1 = max(0, y - rr), min(h, y + rr + 1)
    if x1 <= x0 or y1 <= y0:
        return False
    patch = hsv[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    inside = (xx - x) ** 2 + (yy - y) ** 2 <= rr * rr
    mask = (_mask_for_label(patch[:, :, 0], patch[:, :, 1], patch[:, :, 2], label) > 0) & inside
    total = int(inside.sum())
    if total == 0:
        return False
    return int(mask.sum()) / total >= min_ratio


def _solid_white_disc(hsv: np.ndarray, pos: Point, r: float) -> bool:
    """白球候选实心盘校验：剔除台面白色标线产生的假白球。

    真实白球 = 实心白圆盘：圆内白像素占比高，且白像素呈圆团分布
    （主轴/次轴比 ≈1）。D 区弧线、开球线等细白线切出的候选：
    圆内白占比低（<0.5）或白像素沿一条线拉长（主轴比 >2.5）。
    """
    h, w = hsv.shape[:2]
    x, y = int(round(pos[0])), int(round(pos[1]))
    rr = max(int(round(r)) - 2, 3)
    x0, x1 = max(0, x - rr), min(w, x + rr + 1)
    y0, y1 = max(0, y - rr), min(h, y + rr + 1)
    if x1 <= x0 or y1 <= y0:
        return False
    patch = hsv[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    inside = (xx - x) ** 2 + (yy - y) ** 2 <= rr * rr
    white = (patch[:, :, 1] < 80) & (patch[:, :, 2] > 150) & inside
    n = int(white.sum())
    total = int(inside.sum())
    if total == 0 or n / total < 0.55:
        return False                      # 圆内大半不是白 → 线状假球
    if n < 12:
        return True                       # 白像素太少无法算主轴，占比已过则放行
    ys, xs = np.nonzero(white)
    cx, cy = xs.mean(), ys.mean()
    dx, dy = xs - cx, ys - cy
    cov = np.array([[ (dx*dx).mean(), (dx*dy).mean()],
                    [ (dx*dy).mean(), (dy*dy).mean()]])
    eig = np.linalg.eigvalsh(cov)
    minor = max(float(eig[0]), 1e-3)
    ratio = float(eig[1]) / minor
    return ratio <= 2.5                   # 白像素拉成线 → 不是球


def circle_edge_coverage(gray: np.ndarray, pos: Point, r: float,
                         bins: int = 48) -> float:
    """量化候选周围的圆周边缘覆盖率（0..1）。

    真实球在接近整圈的位置都有“球面→台呢”的亮度差；文字和 UI 图标
    虽然可能恰好有一个近圆的局部，却不能连续覆盖一圈。采样内外两条
    圆环的最大差值，避免球面高光和阴影影响中心颜色判断。

    两条采样环都越界的 bin 不计入分母：贴库球朝向库边的采样环必然
    落在图像边界外，旧实现按 0 计会把真球覆盖率拉到门限以下而静默
    剔除（白球贴库后从画面消失）。可见弧过短（<1/4 圆周）时样本
    已无判别力，直接按不合格处理。
    """
    if gray is None or r < 3:
        return 0.0
    h, w = gray.shape[:2]
    cx, cy = float(pos[0]), float(pos[1])
    contrasts: List[float] = []
    for i in range(bins):
        angle = 2.0 * np.pi * i / bins
        co, si = float(np.cos(angle)), float(np.sin(angle))
        samples = []
        for inner, outer in ((0.58, 1.12), (0.72, 1.30)):
            x0, y0 = int(round(cx + inner * r * co)), int(round(cy + inner * r * si))
            x1, y1 = int(round(cx + outer * r * co)), int(round(cy + outer * r * si))
            if 0 <= x0 < w and 0 <= y0 < h and 0 <= x1 < w and 0 <= y1 < h:
                samples.append(abs(int(gray[y0, x0]) - int(gray[y1, x1])))
        if not samples:
            continue                      # 该方向两条环均越界：贴库球，剔除出分母
        contrasts.append(float(max(samples)))
    if len(contrasts) < max(8, bins // 4):
        return 0.0
    # The absolute threshold is deliberately modest: QQ balls have gradients
    # and anti-aliased rims, while score glyphs only light up isolated bins.
    return float(np.mean(np.asarray(contrasts) >= 16.0))


def pick_cue(balls: Sequence[Ball]) -> Optional[Ball]:
    """母球 = 白球；有多个白球时取最接近标准半径者。"""
    whites = [b for b in balls if b.label == "白球"]
    if not whites:
        return None
    whites.sort(key=lambda b: abs(b.radius - (0.01125 * TABLE_W)))
    return whites[0]



# ---------- 袋口 ----------

def refine_pockets(warped: np.ndarray, expected: List[Point], r: float,
                   pocket_w_ratio: float = 0.045,
                   dark_delta: float = 18.0,
                   min_dark_area_ratio: float = 0.02,
                   pin_area_ratio: float = 0.70,
                   search_ratio: float = 4.0) -> List[Point]:
    """在每个期望袋口附近找暗色斑块，精修袋口中心。

    QQ 2D 的绿色台呢明度会随截图、缩放和位置变化，固定 ``gray < 70``
    在真实帧里会把台呢边缘连成一大片。这里改为逐袋口局部阈值，并对
    暗部做加权质心：

    * 阈值相对局部台呢背景计算，亮度变化不会改变袋口候选；
    * 每个 ROI 单独做连通域，库边不会把六个袋口连成一个大区域；
    * 允许角袋出现完整暗圆，不再用「两个方向都大」误杀真实袋洞；
    * 被 ``warp`` 边界裁掉的合成暗弧面积很小，回退到几何角点，避免把
      裁切后的可见质心当成袋心。

    低置信度时保留 ``expected``，因为错误的袋口比没有精修更容易把
    鬼球方向带偏。
    """
    h, w = warped.shape[:2]
    if h == 0 or w == 0 or not expected or r <= 0:
        return [(float(x), float(y)) for x, y in expected]

    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    # ROI 至少覆盖当前配置的窗口宽度，也至少覆盖 4 个球半径；真实 QQ
    # 角袋从台面边界向内缩约 2~3r，过小的搜索窗会直接丢失洞心。
    win = max(int(round(float(pocket_w_ratio) * w)),
              int(round(max(2.5, float(search_ratio)) * r)))
    max_shift = max(2.0 * r,
                    (max(2.5, float(search_ratio)) - 0.25) * r)
    min_area = max(8.0, float(min_dark_area_ratio) * np.pi * r * r)
    pin_area = max(0.0, float(pin_area_ratio)) * np.pi * r * r
    dark_delta = max(4.0, float(dark_delta))
    refined: List[Point] = []

    for ex, ey in expected:
        ex, ey = float(ex), float(ey)
        x0 = max(0, int(np.floor(ex - win)))
        x1 = min(w, int(np.ceil(ex + win + 1)))
        y0 = max(0, int(np.floor(ey - win)))
        y1 = min(h, int(np.ceil(ey + win + 1)))
        patch = gray[y0:y1, x0:x1]
        if patch.size == 0:
            refined.append((ex, ey))
            continue

        ly, lx = np.mgrid[:patch.shape[0], :patch.shape[1]]
        dx = lx.astype(np.float32) + x0 - ex
        dy = ly.astype(np.float32) + y0 - ey
        local_dist = np.hypot(dx, dy)

        # 用期望点外侧的环估计台呢。排除黑洞和 warp 涂灰边界后，
        # 中位数对台呢纹理/球影比固定全局阈值稳定。
        band = (local_dist >= 2.4 * r) & (local_dist <= 4.5 * r)
        background = patch[band]
        background = background[(background > 20) & (background < 125)]
        if len(background) < 20:
            background = patch[(patch > 20) & (patch < 125)]
        if len(background) == 0:
            baseline = float(np.median(patch))
        else:
            baseline = float(np.median(background))
        threshold = max(20.0, min(150.0, baseline - dark_delta))
        dark = (patch < threshold).astype(np.uint8)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            dark * 255, 8)

        # 期望袋口在四条边上。对边界方向施加单向约束，可以排除 ROI
        # 里反方向的黑球/文字；中袋只需要约束其垂直方向。
        edge_x = ex <= 0.08 * w or ex >= 0.92 * w
        edge_y = ey <= 0.08 * h or ey >= 0.92 * h
        inward_x = 1.0 if ex <= 0.08 * w else (-1.0 if ex >= 0.92 * w else 0.0)
        inward_y = 1.0 if ey <= 0.08 * h else (-1.0 if ey >= 0.92 * h else 0.0)
        corner = edge_x and edge_y
        candidates: List[Tuple[float, float, float, float]] = []

        for idx in range(1, n_labels):
            area = float(stats[idx, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            ys, xs = np.nonzero(labels == idx)
            if len(xs) == 0:
                continue
            weights = np.maximum(threshold - patch[ys, xs].astype(np.float32), 0.0)
            weight_sum = float(weights.sum())
            if weight_sum <= 1e-6:
                cx = float(x0 + np.mean(xs))
                cy = float(y0 + np.mean(ys))
            else:
                cx = float(x0 + np.average(xs, weights=weights))
                cy = float(y0 + np.average(ys, weights=weights))
            distance = float(np.hypot(cx - ex, cy - ey))
            if distance > max_shift:
                continue
            if edge_x and (cx - ex) * inward_x < -0.45 * r:
                continue
            if edge_y and (cy - ey) * inward_y < -0.45 * r:
                continue

            cw = float(stats[idx, cv2.CC_STAT_WIDTH])
            ch = float(stats[idx, cv2.CC_STAT_HEIGHT])
            if min(cw, ch) < 0.35 * r:
                continue
            # 中袋的暗线可能与一小段库边相连，但不应吞掉整个 ROI；
            # 角袋应保持近似圆形，长条 UI/边线直接排除。
            if corner and max(cw, ch) > 5.0 * r:
                continue
            if not corner and max(cw, ch) > 10.0 * r:
                continue

            area_ratio = area / max(1.0, np.pi * r * r)
            strength = float(weights.mean()) / dark_delta
            compact = min(cw, ch) / max(cw, ch)
            # 面积/暗度优先，距离和形状作轻度约束；这样比“离 expected
            # 最近”更能区分真实内缩袋洞与边界残影。
            score = (1.45 * min(area_ratio, 2.0)
                     + 0.45 * min(strength, 2.0)
                     + 0.25 * compact
                     - 0.25 * distance / max(r, 1e-6))
            candidates.append((score, cx, cy, area))

        if not candidates:
            refined.append((ex, ey))
            continue

        _, cx, cy, area = max(candidates, key=lambda item: item[0])
        # 合成图的角袋/中袋暗洞被 warp 的灰边裁切后，只剩一个小月牙，
        # 可见质心会稳定地向台内偏移；该情况下几何角点才是真值。
        # 真实 QQ 袋洞通常留下接近完整的暗区，面积足够大时保留精修。
        if area < pin_area:
            refined.append((ex, ey))
        else:
            refined.append((float(cx), float(cy)))
    return refined


# ---------- 台面四边形跟踪（帧间锁定 + 重检） ----------

class TableTracker:
    """台面四边形跟踪：首帧锁定，周期重检，移动须经稳定确认。

    消除「每帧重新 find_table 的抖动」（四边形抖动会直接变成球坐标
    角度误差）。只有同一个刚性偏移连续出现，才接受为窗口移动；单边
    误检、逐步漂移和短暂漏检都继续使用原锁定框。
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.quad: Optional[np.ndarray] = None
        self.miss = 0
        self.frame = 0
        self._jump = 0          # 连续大偏移计数（真移动 vs 偶发斜检）
        self._jump_samples: List[np.ndarray] = []

    def update(self, frame: np.ndarray) -> Optional[np.ndarray]:
        self.frame += 1
        locked = self.quad is not None and self.miss < self.cfg.table_max_miss
        recheck = self.frame % max(1, self.cfg.table_recheck_frames) == 0
        if locked and not recheck:
            return self.quad
        q = find_table(frame, self.cfg)
        if q is None:
            self.miss += 1
            self._jump = 0
            self._jump_samples = []
            if self.miss >= self.cfg.table_max_miss:
                self.quad = None          # 解锁，强制重新检测
            return self.quad
        self.miss = 0
        if self.quad is None:
            self.quad = q
            self._jump = 0
            self._jump_samples = []
        else:
            # 跳变拒绝：单次重检的角点大偏移多半是边带拟合被污染
            # （库边白线/球贴边/旧版 mss 截到 Overlay）造成的「一边斜」
            # 坏四边形。保持旧框，连续多次出现同一刚性偏移才认为是窗口
            # 真的移动；不能把每次逐步变大的误检当作连续确认。
            shift = float(np.abs(q - self.quad).max())
            jump_thresh = max(3.0, float(getattr(self.cfg, "table_recheck_max_shift", 7.0)))
            if shift > jump_thresh:
                # 真正平移时四个角点的 dx/dy 应近似相同；单边被白线、球
                # 或 UI 污染的候选直接丢弃。
                delta = q - self.quad
                rigid_tol = max(
                    0.75, float(getattr(self.cfg, "table_stable_deadband", 2.0)))
                delta_center = np.median(delta, axis=0)
                if float(np.abs(delta - delta_center).max()) > rigid_tol:
                    self._jump = 0
                    self._jump_samples = []
                    return self.quad

                # 候选必须围绕同一绝对位置聚集。只比较相邻候选会把
                # (+8),(+13),(+18) 这种静态误检漂移错误地确认成移动。
                consensus_tol = max(
                    0.75, float(getattr(self.cfg, "table_stable_deadband", 2.0)))
                if self._jump_samples:
                    center = np.median(np.asarray(self._jump_samples), axis=0)
                    if float(np.abs(q - center).max()) <= consensus_tol:
                        self._jump_samples.append(q.copy())
                    else:
                        self._jump_samples = [q.copy()]
                else:
                    self._jump_samples = [q.copy()]
                self._jump = len(self._jump_samples)
                confirmations = max(
                    1, int(getattr(self.cfg, "table_move_confirmations", 3)))
                if self._jump < confirmations:
                    return self.quad       # 丢弃本次可疑检测
                self.quad = np.median(
                    np.asarray(self._jump_samples), axis=0).astype(np.float32)
                self._jump = 0
                self._jump_samples = []
            else:
                # 锁定框是识别基准，不应把每次重检的 2~7px 边缘噪声
                # 通过 EMA 累积成可见漂移。只有候选四边形越过
                # jump_thresh，并连续确认，才在上面的分支中更新锁定框。
                # 真实窗口移动会在后续重检中继续累积到该阈值。
                self._jump = 0
                self._jump_samples = []
        return self.quad


class PocketTracker:
    """锁定台面坐标中的 6 个袋口，过滤暗色连通域误选造成的跳点。

    袋口经过单应校正后本应固定在台面坐标中。每帧重新从暗色像素取
    质心会把库边、反光或球影带来的候选变化直接显示出来，因此这里对
    小抖动保持原值，对大跳变要求连续确认。
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.pockets: Optional[np.ndarray] = None
        self._jump: Optional[np.ndarray] = None
        self._jump_count = np.zeros(0, dtype=np.int32)

    def reset(self) -> None:
        self.pockets = None
        self._jump = None
        self._jump_count = np.zeros(0, dtype=np.int32)

    def current(self) -> Optional[List[Point]]:
        """返回最近一次已锁定的袋口；尚未初始化时返回 None。"""
        if self.pockets is None:
            return None
        return [tuple(map(float, p)) for p in self.pockets]

    def update(self, observed: Sequence[Point]) -> List[Point]:
        """返回稳定袋口坐标；输入输出均为台面标准坐标。"""
        obs = np.asarray(observed, dtype=np.float32)
        if obs.ndim != 2 or obs.shape[1] != 2 or len(obs) == 0:
            return [tuple(map(float, p)) for p in observed]
        if self.pockets is None or self.pockets.shape != obs.shape:
            self.pockets = obs.copy()
            self._jump = obs.copy()
            self._jump_count = np.zeros(len(obs), dtype=np.int32)
            return [tuple(map(float, p)) for p in self.pockets]

        deadband = max(
            0.5, float(getattr(self.cfg, "pocket_stable_deadband", 2.5)))
        max_shift = max(
            deadband, float(getattr(self.cfg, "pocket_move_max_shift", 18.0)))
        alpha = float(np.clip(
            getattr(self.cfg, "pocket_smooth_alpha", 0.35), 0.05, 1.0))
        confirmations = max(
            1, int(getattr(self.cfg, "pocket_move_confirmations", 3)))

        for i, p in enumerate(obs):
            current = self.pockets[i]
            shift = float(np.linalg.norm(p - current))
            if shift <= deadband:
                self._jump_count[i] = 0
                self._jump[i] = p
                continue
            if shift > max_shift:
                prior = self._jump[i]
                if float(np.linalg.norm(p - prior)) <= max_shift:
                    self._jump_count[i] += 1
                else:
                    self._jump_count[i] = 1
                self._jump[i] = p
                if self._jump_count[i] < confirmations:
                    continue
                self.pockets[i] = p
                self._jump_count[i] = 0
                continue
            self._jump_count[i] = 0
            self._jump[i] = p
            self.pockets[i] = alpha * p + (1.0 - alpha) * current

        return [tuple(map(float, p)) for p in self.pockets]
