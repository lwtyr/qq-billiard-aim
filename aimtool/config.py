"""配置：JSON 持久化（~/.qq-billiard-aim/config.json），启动时加载。"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".qq-billiard-aim")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")

_SAVE_LOCK = threading.Lock()


@dataclass
class Config:
    # 配置版本。v6 新增瞄准让点/成功率评分参数。
    pipeline_version: int = 6
    # 台面标准坐标
    table_w: float = 2000.0
    table_h: float = 1000.0
    # 球半径 = ratio * table_w（QQ 2D桌球 球径约台宽 2.25%）
    ball_radius_ratio: float = 0.0225 / 2.0

    # 屏幕捕获：None = 全屏自动找台；否则 [x, y, w, h]
    capture_region: Optional[List[int]] = None
    capture_fps: float = 60.0
    analysis_fps: float = 30.0            # 最多分析多少张最新帧；绝不排队处理旧帧
    analysis_scale: float = 0.85          # 相对屏幕台面宽度的分析分辨率
    analysis_min_width: int = 720
    analysis_max_width: int = 960
    hough_fallback: bool = True            # 低候选帧才运行 Hough 圆兜底
    hough_trigger_ball_count: int = 4      # 掩膜候选少于该数时启用兜底

    # 力度模型
    power_dref_ratio: float = 2.2         # 多少倍台宽视为满杆
    power_min_pct: float = 10.0
    power_curve: float = 1.0

    # 自动袋口：True=自动选最佳袋口；False=手动选
    auto_pocket: bool = True
    selected_pocket: int = -1             # -1=自动；0..5 指定
    allow_kicks: bool = True              # 直球被挡时是否给出库边解围
    max_kicks: int = 2
    # 鬼球虚线圆和中心十字默认隐藏；需要检查瞄准点时可按 V 临时显示。
    show_ghost: bool = False

    # 视觉调参（一般不用动）
    # 台呢绿色：实测 QQ2D桌球 台呢 H≈57。范围必须收窄——过宽会把含
    # 绿色调的桌面壁纸/背景（H 可达 ~104）也吸进来，全屏截图时掩膜
    # 占满整帧被「占屏>97%」门丢弃 → 找不到台面。
    green_hue_lo: int = 35
    green_hue_hi: int = 80
    min_table_area_ratio: float = 0.02    # 台面至少占捕获区域比例（窗口模式台面占比小，放宽到 2%）
    pocket_refine: bool = True            # 在台面图上精修袋口中心
    detect_max_balls: int = 60            # 球数异常上限（开局红球三角碎块候选可能偏多）

    # 亚像素球心拟合：True=用边缘圆拟合把球心精度提到 ±0.1px 级
    subpixel: bool = True
    subpixel_edges_min: float = 8.0       # 有效边缘点下限（太少则退回粗定位）
    subpixel_window: float = 1.6          # 拟合窗口 = 粗半径 * 该值
    circle_min_edge_coverage: float = 0.42  # 非红球圆周边缘覆盖率，过滤文字/图标
    ui_group_kernel_ratio: float = 0.80     # 把相邻 UI 文字连接为排除区域的核大小

    # 台呢去背景：直方图峰值估计台呢色后按容差涂灰（绿球与台面同色时必须）
    felt_hue_tol: int = 18                # 台呢色相容差（±）
    felt_sv_tol: int = 75                 # 台呢饱和/明度容差（±）

    # 红球三角行拟合：开局红球相切重叠、内部球无可见边缘，用规则行结构重建网格
    rack_fit: bool = True
    # 完整红球架没有安全入袋路线时，显示明确的开局解球碰撞点
    opening_break_fallback: bool = True

    # 台面锁定：首帧检测后锁定四边形，周期性重检防窗口移动
    table_lock: bool = True
    table_recheck_frames: int = 30        # 每 N 帧重检一次四边形
    table_max_miss: int = 5               # 连续检测失败 N 帧后强制解锁重检
    table_recheck_max_shift: float = 7.0  # 单次重检允许的像素偏移；更大需连续确认
    table_stable_deadband: float = 2.0    # 移动候选的刚性/绝对位置容差
    table_move_confirmations: int = 3    # 大偏移连续出现几次才接受为窗口移动
    table_max_edge_skew: float = 0.02     # QQ 2D 轴对齐台面允许的边缘斜率/透视

    # 袋口跟踪：袋口暗部容易被库边/反光分割成不同连通域，不能每帧
    # 直接替换质心，否则白色袋口标记会跳动。
    pocket_smooth_alpha: float = 0.35
    pocket_stable_deadband: float = 2.5
    pocket_move_max_shift: float = 18.0
    pocket_move_confirmations: int = 3
    pocket_dark_delta: float = 18.0       # 相对局部台呢的暗度阈值
    pocket_min_dark_area_ratio: float = 0.02
    pocket_pin_area_ratio: float = 0.70   # 暗区过小则保留几何边界
    pocket_search_ratio: float = 4.0      # 袋口搜索半径（球半径倍数）

    # 跨帧球跟踪与静止判定。坐标单位为标准台面坐标/秒。
    track_confirm_frames: int = 3
    track_history_frames: int = 7
    track_stable_window_frames: int = 3  # 停球后只用最近几帧做中位数
    track_max_misses: int = 2
    stationary_speed: float = 18.0
    moving_speed: float = 85.0
    # 静止多久后才认为局面就绪（打完一杆球停稳后出线的延迟）。
    # 0.15s 在 60fps 下约 9 帧，兼顾防误击与响应速度。
    settle_seconds: float = 0.15

    # 经验物理标定。默认不改变直球几何；实测后可在配置文件中调整。
    ball_radius_scale: float = 1.0
    # 鬼球中心的经验校准，单位为标准台面坐标。默认零，不改变理想几何；
    # 若多次实测同方向偏差，可按标注结果微调，而不是再改检测球心。
    aim_offset_x: float = 0.0
    aim_offset_y: float = 0.0
    ball_radius_instance_weight: float = 0.35
    pocket_offsets: List[List[float]] = field(
        default_factory=lambda: [[0.0, 0.0] for _ in range(6)]
    )
    rail_inset_ratio: float = 1.0         # 库边碰撞以球心轨迹计，默认离台呢边 r
    power_gain: float = 1.0
    power_bias: float = 0.0
    rail_energy_loss: float = 0.22        # 每库能量损耗：等效路程增幅 (1+0.22×库数)
    pocket_accept_ratio: float = 1.45     # 袋口可接受半径 / 球半径（评分与瞄准让点共用）

    # v3.10 瞄准优化：容错让点 + 成功率评分
    pocket_aim_optimize: bool = True      # 斜切球瞄准点自动让到开口区间角平分线
    rank_by_success: bool = True          # 选球/选线按进球成功率优先；False=旧切角优先
    aim_sigma_units: float = 1.0          # 综合球心/映射定位误差 σ（标准台面单位）
    exec_sigma_rad: float = 0.004         # 执行对齐误差 σ（弧度，≈0.23°）
    kick_reliability: float = 0.92        # 每库反弹成功率系数（评分用，需实机标定）

    def save(self) -> None:
        # 检测线程与 UI 线程都可能触发保存，加锁防并发写坏文件
        with _SAVE_LOCK:
            os.makedirs(CONFIG_DIR, exist_ok=True)
            tmp = CONFIG_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(asdict(self), f, ensure_ascii=False, indent=2)
            os.replace(tmp, CONFIG_FILE)

    @classmethod
    def load(cls) -> "Config":
        cfg = cls()
        migrated = False
        try:
            with open(CONFIG_FILE, encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("配置根节点不是对象")
            for k, v in data.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
            # v2 配置的默认分析宽度仍会在常见 1000px 桌面上放大到 992px；
            # v3 以约 800~850px 作为实时默认。只迁移仍等于旧默认值的字段，
            # 不覆盖用户已经手动调过的分辨率。
            version = int(data.get("pipeline_version", 1))
            if version < 2:
                cfg.pipeline_version = 2
                cfg.capture_fps = max(60.0, float(cfg.capture_fps))
                migrated = True
            if version < 3:
                if "analysis_scale" not in data or float(data.get("analysis_scale", 1.0)) == 1.0:
                    cfg.analysis_scale = 0.85
                    migrated = True
                if "analysis_max_width" not in data or int(data.get("analysis_max_width", 1280)) == 1280:
                    cfg.analysis_max_width = 960
                    migrated = True
                cfg.pipeline_version = 3
            if version < 4:
                cfg.pipeline_version = 4
                migrated = True
            if version < 5:
                cfg.pipeline_version = 5
                migrated = True
            if version < 6:
                cfg.pipeline_version = 6
                migrated = True
        except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
            if not isinstance(exc, FileNotFoundError):
                print(f"[配置] {exc}，使用默认配置", flush=True)
        if migrated:
            try:
                cfg.save()
                print(f"[配置] 已迁移到 v{cfg.pipeline_version} 精度参数", flush=True)
            except OSError as exc:
                print(f"[配置] v3 参数仅本次生效，无法保存: {exc}", flush=True)
        return cfg
