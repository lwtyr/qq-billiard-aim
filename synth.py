"""合成 QQ 2D桌球 台面图（测试与 demo 用）：绿色台呢 + 库边 + 6 袋 + 彩球。

返回 (image, meta)，meta 含真值：felt 矩形、球位（画布坐标）、袋口（画布坐标）。
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# 画布即球桌：外圈库边，内圈台呢
CANVAS_W, CANVAS_H = 2000, 1000
CUSHION = 70                      # 库边宽度
FELT_X0, FELT_Y0 = CUSHION, CUSHION
FELT_X1, FELT_Y1 = CANVAS_W - CUSHION, CANVAS_H - CUSHION
BALL_R = 22.5                     # 球半径（px）

FELT_BGR = (60, 140, 93)          # 台呢绿（斯诺克青绿：h≈72，与翠绿球 h≈54 拉开色相）
CUSHION_BGR = (48, 62, 118)       # 库边深棕（接近真实球桌，且不干扰绿色台面识别）
WOOD_BGR = (40, 40, 46)           # 外框

# 斯诺克 7 色（黄绿棕蓝粉黑）+ 白球。注意：全部是 BGR 顺序（cv2 用 BGR）！
POCKET_COLORS: Dict[str, Tuple[int, int, int]] = {
    "白球": (235, 235, 238),
    "黄球": (40, 200, 255),       # RGB(255,200,40)
    "绿球": (0, 160, 30),         # RGB(30,160,0)
    "棕球": (30, 80, 150),        # RGB(150,80,30)
    "蓝球": (255, 0, 0),
    "粉球": (180, 105, 255),      # RGB(255,105,180)
    "黑球": (45, 45, 45),
    "红球": (0, 0, 255),
}


def draw_ball(img: np.ndarray, center: Tuple[float, float], r: float,
              bgr: Tuple[int, int, int]) -> None:
    """画一颗球：底色 + 轻微径向渐变 + 高光，模拟真实渲染。"""
    cx, cy = int(round(center[0])), int(round(center[1]))
    rr = int(round(r))
    cv2.circle(img, (cx, cy), rr, bgr, -1, cv2.LINE_AA)
    # 渐变（向圆心方向提亮一点）。overlay 只改本球区域，
    # 然后以 0.35 透明度叠回原图——非本球区域保持原样。
    overlay = img.copy()
    for k in range(3):
        rr2 = max(1, rr - k * max(1, rr // 6))
        bright = tuple(min(255, int(c) + 14 - k * 4) for c in bgr)
        cv2.circle(overlay, (cx, cy), rr2, bright, 2, cv2.LINE_AA)
    img[:] = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
    # 高光
    hx, hy = cx - rr // 3, cy - rr // 3
    cv2.circle(img, (hx, hy), max(2, rr // 5), (235, 235, 235), -1, cv2.LINE_AA)


def _rng_layout(rng: np.random.Generator, labels: List[str]) -> List[Tuple[str, Tuple[float, float]]]:
    """在台呢内随机放置互不重叠的球。"""
    balls: List[Tuple[str, Tuple[float, float]]] = []
    x0, y0, x1, y1 = FELT_X0 + 1.6 * BALL_R, FELT_Y0 + 1.6 * BALL_R, FELT_X1 - 1.6 * BALL_R, FELT_Y1 - 1.6 * BALL_R
    tries = 0
    for label in labels:
        while tries < 4000:
            px = rng.uniform(x0, x1)
            py = rng.uniform(y0, y1)
            if all(((px - bx) ** 2 + (py - by) ** 2) > (2.6 * BALL_R) ** 2 for _, (bx, by) in balls):
                balls.append((label, (float(px), float(py))))
                break
            tries += 1
    return balls


# ---------- 斯诺克台面（标准点位 + 红球三角） ----------
# 坐标系：y=0 为开球端（D 区），y=H 为黑球端。
# 布局（贴近 QQ2D桌球 实际观感，保证球与球不重叠）：
#   D 线 1/6H（黄绿棕），蓝 1/2H；
#   红球三角顶点 0.66H，底边=顶点+4*行距；
#   黑球紧贴三角底边后方（+2.4r），粉球紧贴顶点前方（-2.4r）。
# 彩球点位/三角顶点可经 config 微调（见 snooker_spot_positions）。

def snooker_spot_positions(w: float = float(FELT_X1 - FELT_X0),
                           h: float = float(FELT_Y1 - FELT_Y0)) -> Dict[str, Tuple[float, float]]:
    """斯诺克 6 彩球标准点位 + 红球三角顶点（画布坐标）。"""
    x0, y0 = FELT_X0, FELT_Y0
    cx = x0 + w / 2.0
    line_y = y0 + h / 6.0          # D 线（黄绿棕）
    d_off = min(w, h) / 6.0        # D 区半径（台宽 1/6）
    row_h = math.sqrt(3.0) * BALL_R
    apex_y = y0 + 0.66 * h         # 红球三角顶点（foot spot）
    black_y = apex_y + 4.0 * row_h + 2.4 * BALL_R   # 紧贴三角底边后方
    pink_y = apex_y - 2.4 * BALL_R                  # 紧贴三角顶点前方
    return {
        "黄球": (cx + d_off, line_y),
        "绿球": (cx - d_off, line_y),
        "棕球": (cx, line_y),
        "蓝球": (cx, y0 + h / 2.0),
        "粉球": (cx, pink_y),
        "黑球": (cx, black_y),
        "红球顶点": (cx, apex_y),
    }


def snooker_red_triangle(apex: Tuple[float, float], r: float,
                         rows: int = 5) -> List[Tuple[float, float]]:
    """红球等边三角 rack：第 i 行 i+1 颗，球心相切（间距 2r），行距 sqrt(3)*r。"""
    pts: List[Tuple[float, float]] = []
    ax, ay = apex
    row_h = math.sqrt(3.0) * r
    for i in range(rows):
        n = i + 1
        y = ay + i * row_h
        for j in range(n):
            x = ax + (j - (n - 1) / 2.0) * 2.0 * r
            pts.append((float(x), float(y)))
    return pts


def snooker_layout(seed: Optional[int] = None, reds: int = 15,
                   cue_at_d: bool = True) -> Tuple[np.ndarray, Dict]:
    """生成标准斯诺克开局台面：白球 D 区 + 6 彩球点位 + 红球三角。

    reds=0 时生成「清彩阶段」台面（无红球），用于决策层测试。
    """
    rng = np.random.default_rng(seed)
    spots = snooker_spot_positions()
    layout: List[Tuple[str, Tuple[float, float]]] = []
    for label, pos in spots.items():
        if label == "红球顶点":
            continue
        layout.append((label, pos))
    if reds > 0:
        rows = int(math.ceil((math.sqrt(8 * reds + 1) - 1) / 2))  # n(n+1)/2 >= reds
        apex = spots["红球顶点"]
        for pos in snooker_red_triangle(apex, BALL_R, rows)[:reds]:
            layout.append(("红球", pos))
    # 白球放 D 区（棕球右侧偏上，模拟开球）
    if cue_at_d:
        bx, by = spots["棕球"]
        layout.append(("白球", (bx + 1.2 * BALL_R, by - 2.6 * BALL_R)))
    img, meta = render(layout, seed)
    meta["spots"] = spots
    meta["cue_side"] = "top"
    return img, meta


def render(layout: Optional[List[Tuple[str, Tuple[float, float]]]] = None,
           seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
    """生成一帧合成台面。layout=None 时随机（可用 seed 复现）。"""
    rng = np.random.default_rng(seed)
    img = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    img[:] = WOOD_BGR
    # 库边
    cv2.rectangle(img, (0, 0), (CANVAS_W - 1, CANVAS_H - 1), CUSHION_BGR, -1)
    # 台呢
    cv2.rectangle(img, (FELT_X0, FELT_Y0), (FELT_X1 - 1, FELT_Y1 - 1), FELT_BGR, -1)
    # 袋口（台呢四角 + 两长边中点，画在台呢与库边交界处）
    pocket_centers = [
        (FELT_X0, FELT_Y0), (FELT_X1, FELT_Y0), (FELT_X0, FELT_Y1), (FELT_X1, FELT_Y1),
        ((FELT_X0 + FELT_X1) / 2, FELT_Y0), ((FELT_X0 + FELT_X1) / 2, FELT_Y1),
    ]
    pr = int(round(1.15 * BALL_R))
    for (px, py) in pocket_centers:
        cv2.circle(img, (int(px), int(py)), pr, (12, 12, 14), -1, cv2.LINE_AA)

    if layout is None:
        labels = ["白球", "黄球", "蓝球", "红球", "粉球", "绿球", "棕球", "黑球"]
        layout = _rng_layout(rng, labels)
    for label, center in layout:
        draw_ball(img, center, BALL_R, POCKET_COLORS[label])

    # 轻度噪声 + 亮度渐变（贴近真实截图，测试鲁棒性）
    noise = rng.normal(0, 2.0, img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    grad = np.linspace(0.92, 1.08, CANVAS_H, dtype=np.float32).reshape(-1, 1, 1)
    img = np.clip(img.astype(np.float32) * grad, 0, 255).astype(np.uint8)

    meta = {
        "felt": (FELT_X0, FELT_Y0, FELT_X1, FELT_Y1),
        "ball_r": BALL_R,
        "balls": [{"label": label, "pos": (x, y)} for label, (x, y) in layout],
        "pockets": pocket_centers,
    }
    return img, meta


def random_layout(seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
    """随机斯诺克色台面（识别鲁棒性测试用）：白球 + 2 红 + 6 彩。"""
    labels = ["白球", "红球", "红球", "黄球", "绿球", "棕球", "蓝球", "粉球", "黑球"]
    return render(_rng_layout(np.random.default_rng(seed), labels), seed)


if __name__ == "__main__":
    import sys
    img, meta = random_layout(seed=42)
    out = sys.argv[1] if len(sys.argv) > 1 else "synth_table.png"
    cv2.imwrite(out, img)
    print(f"saved {out} {img.shape[1]}x{img.shape[0]}")
    print("balls:", [(b["label"], tuple(round(v, 1) for v in b["pos"])) for b in meta["balls"]])
