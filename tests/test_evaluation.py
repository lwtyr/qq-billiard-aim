"""Precision/recall and real-frame manifest evaluator tests."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2

import bench_precision
import synth
from aimtool import config, vision
from evaluate_real import evaluate_manifest


def test_match_balls_is_one_to_one_and_distance_bounded():
    r = config.Config().ball_radius_ratio * 2000.0
    detected = [
        vision.Ball("红球", (100.0, 100.0), r),
        vision.Ball("红球", (300.0, 300.0), r),
    ]
    truth = [("红球", (101.0, 100.0)), ("红球", (500.0, 500.0))]
    matches, false_positive, false_negative = bench_precision._match_balls(
        detected, truth, max_distance=5.0)
    assert len(matches) == 1
    assert false_positive == [1]
    assert false_negative == [1]


def test_real_manifest_evaluator_on_synthetic_frame(tmp_path):
    image, meta = synth.random_layout(seed=0)
    image_path = tmp_path / "frame.png"
    manifest_path = tmp_path / "manifest.json"
    assert cv2.imwrite(str(image_path), image)
    fx0, fy0, fx1, fy1 = meta["felt"]
    truth = []
    for ball in meta["balls"]:
        truth.append({
            "label": ball["label"],
            "pos": [
                (ball["pos"][0] - fx0) * 2000.0 / (fx1 - fx0),
                (ball["pos"][1] - fy0) * 1000.0 / (fy1 - fy0),
            ],
        })
    manifest_path.write_text(json.dumps({
        "frames": [{"image": "frame.png", "balls": truth}],
    }, ensure_ascii=False), encoding="utf-8")

    result = evaluate_manifest(str(manifest_path))
    assert result["frames"] == 1
    assert result["tp"] == len(truth)
    assert result["fp"] == 0
    assert result["fn"] == 0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert "x_std" in result["localization"]


def test_synthetic_benchmark_supports_production_analysis_width():
    result = bench_precision.measure_detection_error(2, analysis_width=843)
    assert result["analysis_size"] == [843, 422]
    assert result["tp"] == result["total"]
    assert result["fp"] == 0
    assert result["fn"] == 0
