"""合成数据精度基准：评估视觉管线的定位误差和几何噪声敏感性。

三个实验：
  A. 识别精度实测：合成散开台面 → 检测球心 vs 真值，统计误差分布。
     （散开局面代表游戏进行中的绝大多数对局；开局三角阶段另见 rack 说明）
  B. 命中率-噪声曲线：对真值坐标注入高斯噪声 σ，ghost ball 法算瞄准方向，
     与袋口角度容差比较 → 命中率随 σ 的变化。
  C. 误差代入：把实验 A 的 X/Y 坐标 σ 代入曲线 B，得到合成噪声模型结果；
     该结果不代表 QQ 游戏真实命中率，也不能证明内存级精度。

用法：
  python bench_precision.py [--seeds 100] [--analysis-width 843]
                            [--pocket-ratio 1.30] [--out results.json]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import synth
from aimtool import config, physics, vision

Point = Tuple[float, float]


# ---------- 实验 A：识别精度实测 ----------

def _match_balls(detected: Sequence, truth: Sequence[Tuple[str, Point]],
                 max_distance: float) -> Tuple[List[Tuple[int, int, float]],
                                                List[int], List[int]]:
    """按颜色和距离做确定性的一对一匹配。

    先按距离排序再占用两侧索引，避免同一检测球同时计入两个真值；
    超过门限的候选分别计为假阳性和漏检，而不是被最近邻悄悄吞掉。
    """
    edges: List[Tuple[float, int, int]] = []
    for di, observed in enumerate(detected):
        for ti, (label, target) in enumerate(truth):
            if observed.label != label:
                continue
            distance = float(np.hypot(observed.pos[0] - target[0],
                                      observed.pos[1] - target[1]))
            if distance <= max_distance:
                edges.append((distance, di, ti))
    used_detected = set()
    used_truth = set()
    matches: List[Tuple[int, int, float]] = []
    for distance, di, ti in sorted(edges):
        if di in used_detected or ti in used_truth:
            continue
        used_detected.add(di)
        used_truth.add(ti)
        matches.append((di, ti, distance))
    false_positive = [i for i in range(len(detected)) if i not in used_detected]
    false_negative = [i for i in range(len(truth)) if i not in used_truth]
    return matches, false_positive, false_negative


def _localization_stats(errors: Sequence[Tuple[float, float]]) -> Dict:
    """Return signed X/Y statistics plus radial percentiles.

    ``sigma_x`` and ``sigma_y`` are coordinate-axis standard deviations.  The
    radial error is reported separately and is never reused as a coordinate
    sigma, because its distribution is not the same measurement.
    """
    if not errors:
        zeros = {
            "n": 0, "x_mean": 0.0, "x_std": 0.0, "y_mean": 0.0,
            "y_std": 0.0, "sigma_x": 0.0, "sigma_y": 0.0, "sigma_xy": 0.0,
            "radial_mean": 0.0, "radial_p50": 0.0, "radial_p95": 0.0,
            "radial_max": 0.0,
        }
        return zeros
    arr = np.asarray(errors, dtype=float)
    radial = np.hypot(arr[:, 0], arr[:, 1])
    sigma_x = float(arr[:, 0].std())
    sigma_y = float(arr[:, 1].std())
    return {
        "n": int(len(arr)),
        "x_mean": float(arr[:, 0].mean()),
        "x_std": sigma_x,
        "y_mean": float(arr[:, 1].mean()),
        "y_std": sigma_y,
        "sigma_x": sigma_x,
        "sigma_y": sigma_y,
        "sigma_xy": float(math.sqrt((sigma_x * sigma_x + sigma_y * sigma_y) / 2.0)),
        "radial_mean": float(radial.mean()),
        "radial_p50": float(np.percentile(radial, 50)),
        "radial_p95": float(np.percentile(radial, 95)),
        "radial_max": float(radial.max()),
    }


def _classification_stats(tp: int, fp: int, fn: int) -> Dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "tp": int(tp), "fp": int(fp), "fn": int(fn),
        "precision": float(precision), "recall": float(recall),
    }


def measure_detection_error(seeds: int, match_distance: Optional[float] = None,
                            analysis_width: Optional[int] = None) -> Dict:
    """合成散开台面上的一对一评估，可按生产分析宽度运行。"""
    cfg = config.Config()
    if analysis_width is not None:
        analysis_width = int(analysis_width)
        if analysis_width < 320:
            raise ValueError("analysis_width 必须至少为 320")
        analysis_height = max(
            160, int(round(analysis_width * cfg.table_h / cfg.table_w)))
    else:
        analysis_width = int(cfg.table_w)
        analysis_height = int(cfg.table_h)
    if match_distance is None:
        match_distance = 1.5 * cfg.ball_radius_ratio * cfg.table_w
    all_errors: List[Tuple[float, float]] = []
    per_label_errors: Dict[str, List[Tuple[float, float]]] = {}
    label_counts: Dict[str, Dict[str, int]] = {}
    total_truth = 0
    total_detected = 0
    true_positive = 0
    for seed in range(seeds):
        img, meta = synth.random_layout(seed=seed)
        fx0, fy0, fx1, fy1 = meta["felt"]

        def c2t(p):
            return ((p[0] - fx0) * cfg.table_w / (fx1 - fx0),
                    (p[1] - fy0) * cfg.table_h / (fy1 - fy0))

        truth = [(b["label"], c2t(b["pos"])) for b in meta["balls"]]
        total_truth += len(truth)
        quad = vision.find_table(img, cfg)
        balls = []
        if quad is not None:
            Hm = vision.homography(quad, analysis_width, analysis_height)
            warped = vision.warp_table(img, Hm, analysis_width, analysis_height)
            r = cfg.ball_radius_ratio * analysis_width
            detected = vision.detect_balls(warped, r, cfg)
            balls = [vision.Ball(
                b.label,
                (float(b.pos[0] * cfg.table_w / analysis_width),
                 float(b.pos[1] * cfg.table_h / analysis_height)),
                float(b.radius * cfg.table_w / analysis_width),
                b.subpixel, b.confidence, b.track_id,
            ) for b in detected]
        total_detected += len(balls)
        matches, false_positive, false_negative = _match_balls(
            balls, truth, float(match_distance))
        true_positive += len(matches)
        for di, ti, _ in matches:
            label = truth[ti][0]
            dx = float(balls[di].pos[0] - truth[ti][1][0])
            dy = float(balls[di].pos[1] - truth[ti][1][1])
            all_errors.append((dx, dy))
            per_label_errors.setdefault(label, []).append((dx, dy))
        for i in false_positive:
            label = balls[i].label
            label_counts.setdefault(label, {"tp": 0, "fp": 0, "fn": 0})["fp"] += 1
        for i in false_negative:
            label = truth[i][0]
            label_counts.setdefault(label, {"tp": 0, "fp": 0, "fn": 0})["fn"] += 1
        for _, ti, _ in matches:
            label_counts.setdefault(truth[ti][0], {"tp": 0, "fp": 0, "fn": 0})["tp"] += 1

    cls = _classification_stats(true_positive,
                                total_detected - true_positive,
                                total_truth - true_positive)
    loc = _localization_stats(all_errors)
    labels = sorted(set(label_counts) | set(per_label_errors))
    per_label = {}
    for label in labels:
        counts = label_counts.get(label, {"tp": 0, "fp": 0, "fn": 0})
        per_label[label] = {
            **_classification_stats(counts["tp"], counts["fp"], counts["fn"]),
            "localization": _localization_stats(per_label_errors.get(label, [])),
        }
    return {
        **cls,
        **loc,
        "total": int(total_truth),
        "detections": int(total_detected),
        "matches": int(true_positive),
        "match_distance": float(match_distance),
        "ball_radius": cfg.ball_radius_ratio * cfg.table_w,
        "analysis_size": [int(analysis_width), int(analysis_height)],
        "per_label": per_label,
    }


# ---------- 实验 B：命中率-噪声曲线 ----------

def _aim_error(cue: Point, target: Point, pocket: Point, r: float,
               cue_n: Point, target_n: Point, pocket_n: Point) -> Optional[float]:
    """理想与带噪坐标下的瞄准方向夹角（度）。"""
    g0 = physics.ghost_pos(target, pocket, r)
    d0 = physics.normalize(physics.sub(g0, cue))
    g1 = physics.ghost_pos(target_n, pocket_n, r)
    d1 = physics.normalize(physics.sub(g1, cue_n))
    if d0 is None or d1 is None:
        return None
    cos = max(-1.0, min(1.0, physics.dot(d0, d1)))
    return math.degrees(math.acos(cos))


def hit_rate_vs_noise(seeds: int, sigma_range: List[float],
                      pocket_ratio: float) -> Dict:
    """真值坐标注入高斯噪声，统计瞄准方向误差 vs 袋口容差的命中率。

    袋口容差：目标球心可偏离袋口中线 ±(pocket_r - ball_r)，
    角度容差 = asin((pocket_r - ball_r) / 目标球到袋口距离)。
    """
    cfg = config.Config()
    r = cfg.ball_radius_ratio * cfg.table_w
    W, H = cfg.table_w, cfg.table_h
    scenarios: List[Tuple[Point, Point, Point, float]] = []   # (cue, target, pocket, tol_deg)
    rng = np.random.default_rng(42)
    for seed in range(seeds):
        _, meta = synth.random_layout(seed=seed)
        fx0, fy0, fx1, fy1 = meta["felt"]

        def c2t(p):
            return ((p[0] - fx0) * W / (fx1 - fx0), (p[1] - fy0) * H / (fy1 - fy0))

        balls = [c2t(b["pos"]) for b in meta["balls"]]
        cue = next((p for b, p in zip(meta["balls"], balls) if b["label"] == "白球"), None)
        targets = [p for b, p in zip(meta["balls"], balls) if b["label"] != "白球"]
        if cue is None or not targets:
            continue
        target = targets[0]
        for pocket in physics.default_pockets(W, H):
            td = physics.dist(target, pocket)
            if td < 2 * r:
                continue
            tol = math.degrees(math.asin((pocket_ratio - 1.0) * r / td))
            scenarios.append((cue, target, pocket, tol))

    sigma_hits = {}
    for sigma in sigma_range:
        hits = 0
        n = 0
        for cue, target, pocket, tol in scenarios:
            for _ in range(5):                      # 每场景 5 次噪声采样
                cue_n = (cue[0] + rng.normal(0, sigma), cue[1] + rng.normal(0, sigma))
                t_n = (target[0] + rng.normal(0, sigma), target[1] + rng.normal(0, sigma))
                p_n = (pocket[0] + rng.normal(0, sigma), pocket[1] + rng.normal(0, sigma))
                err = _aim_error(cue, target, pocket, r, cue_n, t_n, p_n)
                if err is None:
                    continue
                n += 1
                if err <= tol:
                    hits += 1
        sigma_hits[sigma] = {"hit_rate": hits / n if n else 0.0, "n": n}
    return {"sigma_range": sigma_range, "results": sigma_hits,
            "pocket_ratio": pocket_ratio}


def _fit_sigma_to_rate(detection: Dict) -> float:
    """由 X/Y 轴误差估算等效坐标 σ（取两轴较大者，偏保守）。"""
    return max(float(detection.get("sigma_x", 0.0)),
               float(detection.get("sigma_y", 0.0)))


# ---------- 主流程 ----------

def main() -> int:
    ap = argparse.ArgumentParser(description="QQ2D桌球（斯诺克）视觉精度基准")
    ap.add_argument("--seeds", type=int, default=80, help="随机台面数量")
    ap.add_argument("--analysis-width", type=int, default=None,
                    help="分析图宽度；不填则使用 2000px 标准坐标")
    ap.add_argument("--pocket-ratio", type=float, default=1.30,
                    help="袋口直径/球直径（斯诺克约 1.2-1.4，默认 1.30）")
    ap.add_argument("--out", default="bench_results.json")
    args = ap.parse_args()

    print("=== 实验 A：识别精度实测（散开台面）===")
    det = measure_detection_error(args.seeds, analysis_width=args.analysis_width)
    print(f"球半径 {det['ball_radius']:.1f}px；TP={det['tp']} "
          f"truth={det['total']} detections={det['detections']} "
          f"FP={det['fp']} FN={det['fn']}")
    print(f"分类：precision={det['precision'] * 100:.2f}% "
          f"recall={det['recall'] * 100:.2f}%；"
          f"坐标 sigma_x={det['sigma_x']:.3f}px "
          f"sigma_y={det['sigma_y']:.3f}px "
          f"径向 p95={det['radial_p95']:.3f}px")
    for label, s in det["per_label"].items():
        loc = s["localization"]
        print(f"  {label}: TP={s['tp']} FP={s['fp']} FN={s['fn']} "
              f"P={s['precision'] * 100:.1f}% R={s['recall'] * 100:.1f}% "
              f"p95={loc['radial_p95']:.2f}px")

    print("\n=== 实验 B：命中率-噪声曲线 ===")
    sigmas = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
    curve = hit_rate_vs_noise(args.seeds, sigmas, args.pocket_ratio)
    for sigma in sigmas:
        res = curve["results"][sigma]
        bar = "#" * int(res["hit_rate"] * 40)
        print(f"  σ={sigma:>4.2f}px: 命中率 {res['hit_rate']*100:6.2f}%  {bar}")

    print("\n=== 实验 C：实测误差代入 ===")
    sigma_est = _fit_sigma_to_rate(det)
    res = curve["results"].get(sigma_est) or _interp(curve, sigma_est)
    actual_rate = res["hit_rate"] if isinstance(res, dict) else res
    print(f"实测坐标 σ≈{sigma_est:.2f}px → 合成噪声模型命中率 ≈ {actual_rate*100:.1f}%")
    print("（该结果只代表合成几何噪声模型，不是 QQ 游戏真实命中率；"
          "内存级读取也不能由此直接推断。）")

    out = {"detection": det, "hit_curve": curve,
           "conclusion": {
               "measured_sigma": sigma_est,
               "modeled_hit_rate": actual_rate,
               "pocket_ratio": args.pocket_ratio,
           }}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到 {args.out}")
    return 0


def _interp(curve: Dict, sigma: float) -> Dict:
    """在曲线上做线性插值（sigma 不在离散点时）。"""
    sigmas = curve["sigma_range"]
    res = curve["results"]
    if sigma <= sigmas[0]:
        return res[sigmas[0]]
    if sigma >= sigmas[-1]:
        return res[sigmas[-1]]
    for a, b in zip(sigmas, sigmas[1:]):
        if a <= sigma <= b:
            ra, rb = res[a]["hit_rate"], res[b]["hit_rate"]
            t = (sigma - a) / (b - a)
            return {"hit_rate": ra + t * (rb - ra), "n": res[a]["n"]}
    return res[sigmas[-1]]


if __name__ == "__main__":
    raise SystemExit(main())
