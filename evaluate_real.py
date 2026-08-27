"""Evaluate ball detection on manually labelled real screenshots.

Manifest format::

    {
      "coordinate_space": "table",  // default; or "image"
      "match_distance": 35,
      "frames": [
        {"image": "debug_frames/example.png",
         "balls": [{"label": "白球", "pos": [1000, 500]}]}
      ]
    }

Table coordinates are the standard 2000x1000 coordinates used by the physics
engine.  For ``coordinate_space=image``, positions are pixels in the image and
are transformed through the detected table homography before matching.
This tool never writes to the source image; it only writes JSON when --out is
provided.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from bench_precision import _classification_stats, _localization_stats, _match_balls
from aimtool import config, physics, vision

Point = Tuple[float, float]


def evaluate_manifest(manifest_path: str, cfg: Optional[config.Config] = None) -> Dict:
    """Run the production detector against every labelled manifest frame."""
    cfg = cfg or config.Config()
    root = Path(manifest_path).resolve().parent
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("frames"), list):
        raise ValueError("manifest 必须包含 frames 数组")
    coordinate_space = manifest.get("coordinate_space", "table")
    if coordinate_space not in {"table", "image"}:
        raise ValueError("coordinate_space 只能是 table 或 image")
    default_distance = float(manifest.get("match_distance", 1.5 *
                                 cfg.ball_radius_ratio * cfg.table_w))

    total = {"tp": 0, "fp": 0, "fn": 0}
    errors: List[Tuple[float, float]] = []
    per_label = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "errors": []})
    frame_results = []
    for item in manifest["frames"]:
        if not isinstance(item, dict) or not item.get("image"):
            raise ValueError("每个 frame 必须包含 image")
        image_path = Path(item["image"])
        if not image_path.is_absolute():
            image_path = root / image_path
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise FileNotFoundError(str(image_path))
        quad = vision.find_table(frame, cfg)
        detected: List[vision.Ball] = []
        truth: List[Tuple[str, Point]] = []
        analysis_size = None
        occluded = False
        if quad is not None:
            H = vision.homography(quad, cfg.table_w, cfg.table_h)
            aw = int(round(np.linalg.norm(quad[1] - quad[0]) *
                           float(getattr(cfg, "analysis_scale", 0.85))))
            aw = max(int(getattr(cfg, "analysis_min_width", 720)), aw)
            aw = min(int(getattr(cfg, "analysis_max_width", 960)), aw)
            ah = max(160, int(round(aw * cfg.table_h / cfg.table_w)))
            analysis_size = [aw, ah]
            Ha = vision.homography(quad, aw, ah)
            warped = vision.warp_table(frame, Ha, aw, ah)
            r = cfg.ball_radius_ratio * aw
            pockets = vision.refine_pockets(
                warped, physics.default_pockets(aw, ah), r,
                dark_delta=float(getattr(cfg, "pocket_dark_delta", 18.0)),
                min_dark_area_ratio=float(
                    getattr(cfg, "pocket_min_dark_area_ratio", 0.02)),
                pin_area_ratio=float(
                    getattr(cfg, "pocket_pin_area_ratio", 0.70)),
                search_ratio=float(getattr(cfg, "pocket_search_ratio", 4.0)),
            ) if cfg.pocket_refine else physics.default_pockets(aw, ah)
            warped_hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
            warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            felt_hsv = vision.estimate_felt_hsv(warped, cfg, hsv=warped_hsv)
            foreign = vision.compute_foreign_mask(
                warped, cfg, r, hsv=warped_hsv, felt_hsv=felt_hsv)
            occluded = vision.detect_table_occlusion(
                warped, cfg, r, hsv=warped_hsv, foreign=foreign,
                felt_hsv=felt_hsv) is not None
            if not occluded:
                ui = vision.transient_ui_mask(
                    warped, cfg, r, hsv=warped_hsv, gray=warped_gray,
                    foreign=foreign, felt_hsv=felt_hsv)
                clean = vision.clean_background(
                    warped, cfg, r, pockets, ui, warped_hsv, felt_hsv)
                detected_a = vision.detect_balls(
                    warped, r, cfg, pockets, clean, ui,
                    warped_hsv=warped_hsv, warped_gray=warped_gray)
                detected = [vision.Ball(
                    ball.label,
                    (ball.pos[0] * cfg.table_w / aw, ball.pos[1] * cfg.table_h / ah),
                    ball.radius * cfg.table_w / aw, ball.subpixel,
                    ball.confidence, ball.track_id,
                ) for ball in detected_a]
            for ball in item.get("balls", []):
                label = str(ball["label"])
                px, py = float(ball["pos"][0]), float(ball["pos"][1])
                if coordinate_space == "image":
                    tx, ty = vision.point_screen_to_table((px, py), H)
                else:
                    tx, ty = px, py
                truth.append((label, (tx, ty)))
        else:
            for ball in item.get("balls", []):
                truth.append((str(ball["label"]),
                              (float(ball["pos"][0]), float(ball["pos"][1]))))

        distance = float(item.get("match_distance", default_distance))
        matches, fp, fn = _match_balls(detected, truth, distance)
        frame_counts = _classification_stats(len(matches), len(fp), len(fn))
        total["tp"] += frame_counts["tp"]
        total["fp"] += frame_counts["fp"]
        total["fn"] += frame_counts["fn"]
        for di, ti, _ in matches:
            label = truth[ti][0]
            error = (detected[di].pos[0] - truth[ti][1][0],
                     detected[di].pos[1] - truth[ti][1][1])
            errors.append(error)
            per_label[label]["tp"] += 1
            per_label[label]["errors"].append(error)
        for di in fp:
            per_label[detected[di].label]["fp"] += 1
        for ti in fn:
            per_label[truth[ti][0]]["fn"] += 1
        frame_results.append({
            "image": str(image_path), "table_found": quad is not None,
            "occluded": occluded, "analysis_size": analysis_size,
            "detected": len(detected), "truth": len(truth),
            **frame_counts, "localization": _localization_stats([
                (detected[di].pos[0] - truth[ti][1][0],
                 detected[di].pos[1] - truth[ti][1][1])
                for di, ti, _ in matches
            ]),
        })

    result = {
        **_classification_stats(total["tp"], total["fp"], total["fn"]),
        "frames": len(frame_results),
        "match_distance": default_distance,
        "localization": _localization_stats(errors),
        "per_label": {},
        "frame_results": frame_results,
    }
    for label, values in sorted(per_label.items()):
        result["per_label"][label] = {
            **_classification_stats(values["tp"], values["fp"], values["fn"]),
            "localization": _localization_stats(values["errors"]),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="评估真实截图上的斯诺克球检测")
    parser.add_argument("manifest", help="标注 manifest JSON")
    parser.add_argument("--out", help="可选的 JSON 输出路径")
    args = parser.parse_args()
    result = evaluate_manifest(args.manifest)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
