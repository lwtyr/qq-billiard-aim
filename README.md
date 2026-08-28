# QQ 2D桌球 斯诺克视觉瞄准器（qq-billiard-aim）

针对 **QQ游戏大厅《2D桌球》斯诺克模式** 的纯视觉辅助瞄准器。
**不读取游戏进程内存**（对标内存外挂 Taiqiu.exe 的另一种技术路线）：
截屏 → 视觉识别球/台面/袋口 → 物理计算瞄准线 → 悬浮窗 Overlay 提示。

## 与内存读取的边界

本项目不读取游戏进程内存，只从截图恢复台面几何和球位。内存读取可以直接得到
游戏内部坐标，纯视觉无法在信息已经被缩放、抗锯齿、遮挡或 UI 覆盖后保证同等精度，
因此不能把本项目宣称为“内存级精度”。v3 的目标是把可见场景下的延迟、抖动、误检
和漏检降到可测、可调的范围，实际效果必须用真实截图 manifest 验收。

`bench_precision.py` 只生成合成基准：它使用颜色+距离的一对一匹配，报告 TP/FP/FN、
Precision/Recall、X/Y 轴误差和径向 P50/P95。合成结果不能替代真实 QQ 游戏命中率。
使用 `--analysis-width` 可以把合成检测降到和生产一致的分析尺寸；不指定时仍以
2000×1000 标准图运行，便于和旧结果比较。

## 斯诺克专项能力

- **斯诺克色板**：红×15、黄/绿/棕/蓝/粉/黑 + 白球（替换原美式八球的橙/紫）
- **台呢去背景**：绿球与绿色台面同色 → 直方图峰值估计台呢色 + 球像素保护涂灰
- **红球三角分离**：相切重叠球 → 距离峰值种子 + watershed 分水岭 + 行结构网格拟合
- **瞄准点显示**：鬼球中心十字 + 目标球实际接触点 + 当前帧球径估计
- **遮挡保护**：设置窗口、提示框、右键菜单覆盖台面时暂停输出瞄准线，避免假球误导
- **实时管线**：捕获线程只发布最新帧，分析线程按 `analysis_fps` 消费，避免旧帧排队造成滞后；默认以约 800～850px 分析，低候选帧才启用 Hough 兜底
- **跨帧确认**：球候选经过多帧确认和中位数稳定；球运动、稳定等待、UI 遮挡分别阻断瞄准线
- **决策层**（`aimtool/snooker.py`）：自动目标先过滤击球线/下球线上的障碍，
  再按切角小、目标离袋近、库数少和总路程短选择；清彩阶段按
  黄(2)→绿(3)→棕(4)→蓝(5)→粉(6)→黑(7) 分值顺序
- **开局解球回退**：完整红球架没有安全入袋路线时，明确显示母球方向最外层
  红球的碰撞点；该线是撞散球架提示，不伪装成入袋路线
- **球权状态**：最后一颗红球落袋后保留“红后选彩”阶段；红球仍在时可按 `Q`
  在红球/彩球目标间即时切换，`O` 继续作为兼容热键
- **台面锁定**（`TableTracker`）：首帧检测后 EMA 平滑角点、周期重检，
  小噪声保持原框，大跳变连续确认后才接受；袋口也在台面坐标中锁定，
  消除逐帧四边形/袋口抖动
- **自动路线**：遍历六个袋口、直球和一库/两库路线，过滤击球线与下球线障碍后，
  严格按切角小、目标离袋近、库数少和总路程短选择；`route_options` 保留备选路线；
  库边轨迹按球心内缩，支持 `rail_inset_ratio`、袋口偏移和力度增益标定

## 安装与使用

```bash
pip install -r requirements.txt        # 运行依赖 + pytest 测试依赖
python main.py                         # 正常启动（截屏+Overlay，需 Windows+显示环境）
python main.py --demo                  # 无界面自检（合成斯诺克台面，可无头跑）
python main.py --frame 截图.png        # 分析一张游戏截图
python main.py --region X Y W H        # 预设捕获区域
python bench_precision.py --seeds 80 --analysis-width 843  # 按生产分析宽度评估
python evaluate_real.py labels.json --out real_results.json  # 真实截图标注评估
python -m pytest tests/ -q             # 单元测试
```

Overlay 热键：`1-6` 选袋口 · `0` 自动 · `G` 点选目标球 · `M` 手动录入 ·
`R` 框选区域 · `K` 库边解围 · `P` 自动袋口 · `B` 球标注 · `X` 鼠标穿透 ·
`Q` 红/彩切换 · `O` 兼容切换 · `T` 隐藏 · `C` 重识别 · `Esc` 退出

### 真实截图评估

先保存不包含本辅助层的游戏截图，人工标注球心。manifest 中的 `pos` 默认使用物理
管线的标准台面坐标 `(0..2000, 0..1000)`；如果标注的是截图局部像素，设置
`"coordinate_space": "image"`。同色红球可重复，评估器会进行一对一匹配。

```json
{
  "coordinate_space": "table",
  "match_distance": 35,
  "frames": [
    {
      "image": "debug_frames/example.png",
      "balls": [
        {"label": "白球", "pos": [1000, 500]},
        {"label": "黑球", "pos": [1600, 300]}
      ]
    }
  ]
}
```

运行 `evaluate_real.py` 后重点看每色 `precision/recall`、漏球/假球数量和
`localization.x_std/y_std`。建议按“无 UI、连击文字、球杆接触、运动中、窗口移动、
不同缩放”分组采样，再分别调阈值；不要用单张截图的命中结果作为结论。

实时性能调节主要看 `analysis_scale`、`analysis_min_width`、`analysis_max_width`；
若低清或遮挡场景经常漏球，可保留 `hough_fallback`，并把
`hough_trigger_ball_count` 调高。高分析宽度会近似按像素面积增加 CPU 开销。

## 架构

```
capture.py   屏幕捕获（Windows GDI，SRCCOPY 不含分层 Overlay；mss 回退）
vision.py    视觉：find_table（PCA 边带拟合亚像素四边形）→ warp →
             clean_background（台呢/袋口涂灰）→ detect_balls（颜色掩膜 +
             watershed 粘连分离 + 颜色外轮廓亚像素球心 + 红球三角网格拟合）
             TableTracker（四边形锁定/EMA/重检）
tracking.py  最新帧交换、球 ID 跟踪、唯一彩球去重、运动/稳定/UI 状态机
physics.py   几何：鬼球瞄准、库边解围（unfolding+仿真）、力度建议、可行路线评分
snooker.py   斯诺克决策：红球/清彩目标选择、分值顺序
main.py      analyze() 管线 + Overlay 应用
synth.py     合成斯诺克台面（15红三角+点位彩球，测试/bench 基准）
bench_precision.py  精度基准（识别误差实测 + 命中率-噪声曲线）
evaluate_real.py    真实截图 manifest 评估（Precision/Recall + X/Y 误差）
```

## 已知限制

1. **开局红球三角内部球**：相切重叠、内部球无可见边缘，合成基准通过 rack
   外框/掩膜约束将误差压到 1px 内；真实游戏仍建议先打外围球，散开后再重新识别。
2. **颜色判定依赖 HSV 阈值**：真实游戏的光照/球色若与合成差异大，需先采集并用
   `evaluate_real.py` 统计，再微调 `config.py` 中的色板和台呢参数。
3. **规则状态不是完整游戏状态**：截图无法可靠判断失误、换手和球是否入袋；红球仍在时按 `Q`
   （`O` 兼容）切换红球/彩球目标，避免因错误状态推荐犯规球。
4. `bench_precision.py` 只用于合成数据回归和参数比较；真实截图应使用
   `evaluate_real.py`，菜单/弹窗会被主动识别为遮挡，关闭后按 `C` 重新识别。
5. 纯视觉路线同样属于游戏辅助工具，腾讯 ACE 的行为检测（鼠标瞬移/异常命中率）
   可能识别，账号风险自负。

## 主要文件变更记录

- `aimtool/vision.py`：斯诺克色板、台呢去背景、UI 掩膜、watershed 粘连分离、亚像素圆拟合、
  PCA 边带 find_table、TableTracker、红球三角行拟合
- `aimtool/tracking.py`：新增最新帧交换、球跟踪和静止状态机
- `aimtool/snooker.py`：新增斯诺克决策层和跨帧球权状态
- `aimtool/config.py`：斯诺克/亚像素/tracker/rack/实时管线/物理标定配置项
- `synth.py`：修复 draw_ball 叠加 bug；斯诺克台面（三角+点位）
- `main.py`：最新帧实时分析、运动状态阻断、tracker 接入、demo 修复
- `bench_precision.py`、`evaluate_real.py`：合成/真实截图评估
- `tests/`：视觉、物理、规则、实时状态和评估回归
