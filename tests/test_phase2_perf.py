"""阶段2性能优化回归测试。

对应优化：
- _mask_for_label 零分配输出（bool 视图 × 原地 255），与旧 (ok*255).astype 逐位一致；
- compute_label_masks 帧级掩膜字典：clean_background / compute_foreign_mask /
  transient_ui_mask / detect_balls 共享同一份全帧布尔运算结果；
- transient_ui_mask 白球验证按需化：无 UI 候选或候选不与白色像素相交时
  整帧跳过 _verified_white_ball_mask，输出与旧实现逐位一致；
- FrameStore.wait_for_new：发布即唤醒，代替检测线程 250Hz 轮询；
- refine_red_rack 探针局部化（行为由 test_phase1_fixes 覆盖）。
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import threading
import time
import numpy as np

from aimtool import config, tracking, vision
import synth


CFG = config.Config()


def _warped_with_ui():
    """带台面 + 球的合成台面图（裁到台面区域，等效生产 warp 后视角）。

    生产管线 warp_table 只输出台面矩形；直接用 random_layout 原图会把
    台面外围木边框环（felt 从 y=70 才开始，margin 2.5r≈50px 切不掉）
    当成外来像素，不是真实路径。
    """
    img, meta = synth.random_layout(seed=7)
    fx0, fy0, fx1, fy1 = (int(v) for v in meta["felt"])
    return img[fy0:fy1, fx0:fx1].copy(), meta


def _ball_r(meta, img):
    return float(meta["ball_r"]) if "ball_r" in meta else CFG.ball_radius_ratio * img.shape[1]


# ---------- 掩膜字典 ----------

def test_compute_label_masks_covers_priority_and_is_binary():
    img, _ = _warped_with_ui()
    hsv = vision.cv2.cvtColor(img, vision.cv2.COLOR_BGR2HSV)
    h, s, v = vision.cv2.split(hsv)
    masks = vision.compute_label_masks(h, s, v)
    assert set(masks) == set(vision.PRIORITY)
    for label, m in masks.items():
        assert m.dtype == np.uint8
        assert set(np.unique(m)) <= {0, 255}
        # 与单色调用逐位一致
        assert np.array_equal(m, vision._mask_for_label(h, s, v, label))


def test_clean_background_label_masks_equivalent():
    img, meta = _warped_with_ui()
    r = _ball_r(meta, img)
    from aimtool.physics import default_pockets
    pockets = default_pockets(img.shape[1], img.shape[0])
    a = vision.clean_background(img, CFG, r, pockets)
    hsv = vision.cv2.cvtColor(img, vision.cv2.COLOR_BGR2HSV)
    h, s, v = vision.cv2.split(hsv)
    lm = vision.compute_label_masks(h, s, v)
    b = vision.clean_background(img, CFG, r, pockets, hsv=hsv,
                                label_masks=lm)
    assert np.array_equal(a, b)


# ---------- 白球验证按需化 ----------

def test_transient_ui_mask_still_excludes_text_ui():
    img, meta = _warped_with_ui()
    r = _ball_r(meta, img)
    # 台面中央画一条白色宽字（模拟连击提示）
    ui = img.copy()
    h, w = ui.shape[:2]
    y0 = int(h / 2 - r * 0.6)
    y1 = int(h / 2 + r * 0.6)
    x0 = int(w * 0.2)
    x1 = int(w * 0.8)
    ui[y0:y1, x0:x1] = (240, 240, 240)
    mask = vision.transient_ui_mask(ui, CFG, r)
    assert mask[y0:y1, x0:x1].max() == 255


def test_transient_ui_mask_clean_frame_is_zero():
    img, meta = _warped_with_ui()
    r = _ball_r(meta, img)
    mask = vision.transient_ui_mask(img, CFG, r)
    assert int(mask.max(initial=0)) == 0


# ---------- FrameStore 通知 ----------

def test_frame_store_wait_for_new_wakes_on_publish():
    store = tracking.FrameStore()
    t0 = time.perf_counter()
    out = {}

    def consume():
        pkt = store.wait_for_new(-1, timeout=3.0)
        out["seq"] = pkt.sequence
        out["dt"] = time.perf_counter() - t0

    th = threading.Thread(target=consume)
    th.start()
    time.sleep(0.15)
    store.publish(np.zeros((4, 4, 3), np.uint8), None)
    th.join(timeout=3.0)
    assert out.get("seq") == 1
    assert 0.1 < out["dt"] < 1.0          # 被发布唤醒，而不是超时返回


def test_frame_store_wait_for_new_timeout_returns_latest():
    store = tracking.FrameStore()
    p = store.publish(np.zeros((4, 4, 3), np.uint8), None)
    t0 = time.perf_counter()
    pkt = store.wait_for_new(p.sequence, timeout=0.05)
    dt = time.perf_counter() - t0
    assert pkt.sequence == p.sequence
    assert dt >= 0.04                      # 确实等待了，不是忙轮询
    assert store.latest().sequence == 1    # latest() 兼容不变
