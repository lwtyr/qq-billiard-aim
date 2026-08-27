"""Realtime frame, ball identity, and settle-state regression tests."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from aimtool import config, tracking, vision


def test_frame_store_replaces_old_packet_with_latest():
    store = tracking.FrameStore()
    first = store.publish(np.zeros((2, 2, 3), dtype=np.uint8), None)
    second_frame = np.full((2, 2, 3), 7, dtype=np.uint8)
    second = store.publish(second_frame, np.ones((2, 2), dtype=np.uint8))

    latest = store.latest()
    assert latest is second
    assert latest.sequence == first.sequence + 1
    assert np.array_equal(latest.frame, second_frame)
    assert latest.self_mask is not None


def test_ball_tracker_confirms_tracks_and_deduplicates_unique_color():
    cfg = config.Config(track_confirm_frames=3, track_history_frames=5,
                        track_max_misses=1)
    tracker = tracking.BallTracker(cfg)
    r = cfg.ball_radius_ratio * cfg.table_w
    output = []
    for i in range(3):
        observations = [
            vision.Ball("黄球", (300.0 + i * 0.4, 400.0), r, confidence=0.9),
        ]
        if i == 1:
            observations.append(vision.Ball("黄球", (1500.0, 700.0), r,
                                             confidence=0.4))
        output = tracker.update(observations, now=float(i))
        if i < 2:
            assert output == []

    assert len(output) == 1
    assert output[0].track_id > 0
    assert np.hypot(output[0].pos[0] - 300.4, output[0].pos[1] - 400.0) < 2.0


def test_ball_tracker_drops_motion_history_before_aim_position():
    """停球后不能继续输出运动阶段的中位数位置。"""
    cfg = config.Config(track_confirm_frames=1, track_history_frames=7,
                        track_stable_window_frames=3, stationary_speed=18.0)
    tracker = tracking.BallTracker(cfg)
    r = cfg.ball_radius_ratio * cfg.table_w

    tracker.update([vision.Ball("白球", (100.0, 300.0), r)], now=0.0)
    # 模拟动画阶段：每帧位移明显超过 stationary_speed * dt。
    for i in range(1, 6):
        output = tracker.update(
            [vision.Ball("白球", (100.0 + 20.0 * i, 300.0), r)],
            now=i / 30.0,
        )
        assert output[0].pos[0] == 100.0 + 20.0 * i

    # 球停在 x=200；即使 association history 仍包含旧位置，稳定输出
    # 也只能来自停球后的样本。
    for i in range(6, 9):
        output = tracker.update(
            [vision.Ball("白球", (200.0, 300.0), r)], now=i / 30.0
        )
    assert abs(output[0].pos[0] - 200.0) < 1e-6


def test_table_state_hides_until_settled_and_handles_motion_and_occlusion():
    cfg = config.Config(stationary_speed=18.0, moving_speed=85.0,
                        settle_seconds=0.20)
    state = tracking.TableStateTracker(cfg)
    r = cfg.ball_radius_ratio * cfg.table_w

    def balls(x):
        return [
            vision.Ball("白球", (x, 300.0), r, track_id=1),
            vision.Ball("红球", (1000.0, 500.0), r, track_id=2),
        ]

    assert state.update(balls(300.0), now=0.0) == tracking.TableState.STABILIZING
    assert state.update(balls(420.0), now=0.10) == tracking.TableState.MOVING
    assert state.update(balls(420.0), now=0.20) == tracking.TableState.STABILIZING
    assert state.update(balls(420.0), now=0.45) == tracking.TableState.READY
    assert state.update(balls(420.0), now=0.46, occluded=True) == tracking.TableState.UI_BLOCKED
