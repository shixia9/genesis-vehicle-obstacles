# `mobile_robot` 视觉/控制需求执行报告

> 执行日期：2026-09-08
>
> 对应任务：[TASK.md](../TASK.md)
>
> 当前交付范围：简易策略行驶 + 沿途多物体视觉识别 + 日志/可视化 + 安全降级
>
> LLM：不属于本次实现前置，独立任务见 [TASK_LLM.md](../TASK_LLM.md)

## 1. 执行结论

已完成视觉/控制主线的代码实现和可验证部分测试，形成以下新入口：

```text
room_navigation_vision.py
    ├── 既有 waypoint + LiDAR 安全控制
    ├── vision_route_showcase 沿途多物体场景
    ├── disabled / ground_truth / yolo 三种感知模式
    ├── FramePacket → VisionResult → tracker
    ├── 颜色、可选深度辅助和检测框叠加
    ├── vision_results.jsonl / telemetry.csv / sensor JSONL
    └── 模型缺失或推理异常时安全降级
```

纯 Python 的视觉协议、投影、跟踪、颜色/深度辅助和 LiDAR 控制测试已通过。在受限沙箱中 Genesis 无法创建 OpenGL context，但经授权在宿主环境完成了完整 ground-truth 视觉/控制回归。真实 YOLO 权重仍未在仓库中提供，因此本报告不把 ground-truth 联调结果冒充 YOLO 识别结果。

## 2. 需求分析结果

本次将需求按两个工作流处理：

| 工作流        | 本次状态 | 说明                                                   |
| ------------- | -------- | ------------------------------------------------------ |
| 视觉/小车控制 | 本次主线 | 固定目的地、简易策略、沿途持续识别、记录和安全降级     |
| LLM 自然语言  | 独立并行 | 只约定`TaskSpec`/技能/事件接口，详见 `TASK_LLM.md` |

当前阶段的验收故事被明确为：

> 小车按现有 waypoint + LiDAR 简易策略驶向既定目的地，在整个行驶过程中持续识别不同物体，并在第一视角与日志中展示识别结果，最终安全到达。

本阶段视觉识别是旁路能力，不直接替代 waypoint、LiDAR 或轮速控制，避免视觉模型问题掩盖驾驶控制问题。

## 3. 已实现功能

### 3.1 沿途多物体场景

在 `room_navigation_observable.py` 中新增：

- `vision_route_showcase` 场景；
- 贯穿房间的多段 waypoint 路线；
- 黄色静态车 `yellow_car_01`；
- 红色箱体 `red_box_01`；
- 蓝色圆柱 `blue_cylinder_01`；
- `SemanticObjectSpec` 和 `semantic_objects_for_scenario()`；
- 语义物体 footprint 并入碰撞检查；
- 物体真值仅供 ground-truth/标注/评估使用。

### 3.2 实时视觉入口

新增 [room_navigation_vision.py](../room_navigation_vision.py)：

- `--perception-mode disabled`：控制基线；
- `--perception-mode ground_truth`：确定性投影，用于运行时、叠加和日志联调；
- `--perception-mode yolo`：加载已有本地 `.pt/.onnx/.engine`，不会隐式联网下载；
- `--vision-every` 控制采帧/推理频率；
- `--save-images`、`--save-sensors`、`--save-vision` 相互独立；
- 视觉采帧不依赖 GUI 是否打开；
- 检测器异常转为 `inference_error:*`，控制循环继续执行；
- 模型缺失转为 `model_unavailable`/`model_path_missing`，不产生伪造检测。

### 3.3 视觉模块

新增：

- `vision/ground_truth.py`：明确标注来源的真值投影适配器；
- `vision/tracker.py`：按类别、颜色和位置进行多帧关联并生成 `track_id`；
- `vision/attributes.py`：保守 RGB ROI 主色判断；
- `vision/rgbd_fusion.py`：保留归一化 bbox 的兼容性近似采样；
- `vision/rgbd_calibration.py`：读取 Genesis RGB/Depth 真实内参和同挂载外参，执行精确 bbox 重投影、深度统计和坐标融合；
- `Detection.track_id` 与叠加标签扩展；
- `vision/__init__.py` 导出运行时所需组件。

### 3.4 Genesis 兼容性修复

`genesis/utils/misc.py` 的 CPU 设备识别原先假设 `cpuinfo` 必有 `brand_raw/hardware_raw/vendor_id_raw`，当前机器返回空值时触发 `StopIteration`。现改为依次回退到 `platform.processor()` 和 `unknown-cpu`，使 `get_device(gs.cpu)` 在该环境下可以返回 CPU 设备。

这属于初始化兼容性修复，与视觉逻辑无关，但否则所有 Genesis 示例都无法进入场景构建阶段。

### 3.5 README 与测试

- README 增加视觉展示入口、ground-truth 联调命令、YOLO 命令和输出说明；
- 新增 `examples/mobile_robot/tests/test_vision_pipeline.py`，覆盖帧协议、禁用降级、真值投影、跟踪、颜色/深度辅助和 LiDAR 安全优先级。

## 4. 运行方式

### 4.1 控制基线

```bash
.venv/bin/python examples/mobile_robot/room_navigation_vision.py \
  --scenario vision_route_showcase \
  --perception-mode disabled \
  --steps 1200 \
  --output-dir out/mobile_robot_vision_baseline
```

### 4.2 确定性视觉联调

```bash
.venv/bin/python examples/mobile_robot/room_navigation_vision.py \
  --scenario vision_route_showcase \
  --perception-mode ground_truth \
  --save-images --save-sensors --save-vision \
  --output-dir out/mobile_robot_vision_gt
```

该模式的结果来源是场景注册元数据投影，不代表 YOLO 指标；日志和 `summary.json` 会写入 `perception_mode=ground_truth`。

### 4.3 真实 YOLO

```bash
.venv/bin/python examples/mobile_robot/room_navigation_vision.py \
  --scenario vision_route_showcase \
  --perception-mode yolo \
  --vision-model models/mobile_robot/car_obstacle.pt \
  --vision-device cpu \
  --save-vision \
  --output-dir out/mobile_robot_vision_yolo
```

当前仓库没有 `models/mobile_robot/car_obstacle.pt`，因此该命令在没有外部权重时会安全降级，不会自动下载模型。

## 5. 测试执行记录

### 5.1 已通过

| 测试                 | 命令/方式                                                                                           | 结果                       |
| -------------------- | --------------------------------------------------------------------------------------------------- | -------------------------- |
| Python 编译检查      | `.venv/bin/python -m compileall -q examples/mobile_robot genesis/utils/misc.py`                   | 通过                       |
| 模块导入和数据契约   | `FramePacket/Detection/VisionResult/ObjectTracker` 直接调用                                       | 通过                       |
| 视觉单元测试         | `.venv/bin/python -m pytest -q -o addopts='' examples/mobile_robot/tests/test_vision_pipeline.py` | 5 passed                   |
| GroundTruth 前向投影 | 前方物体保留、后方物体过滤                                                                          | 通过                       |
| 跟踪 ID              | 相邻帧同一世界位置保持`track-001`                                                                 | 通过                       |
| 颜色辅助             | 黄色 ROI 置信度判断                                                                                 | 通过                       |
| 近似深度辅助         | bbox 归一化采样返回 2.0 m                                                                           | 通过                       |
| LiDAR 安全层         | 近障碍时`linear_velocity=0` 且角速度为上限                                                        | 通过                       |
| YOLO 缺失降级        | 缺少模型路径转为`DisabledDetector(model_unavailable)`                                             | 通过                       |
| CLI 可用性           | `room_navigation_vision.py --help`                                                                | 通过                       |
| CPU 设备回退         | `get_device(gs.cpu)`                                                                              | 通过，返回`device='cpu'` |

测试工具与项目默认配置说明：

- pytest 已通过本机缓存离线安装；项目默认 `pyproject.toml` 的 `addopts` 还要求未安装的 `pytest-xdist/pytest-timeout`，因此使用 `-o addopts=''` 执行本测试文件；
- `.venv/bin/python -m ruff ...`：ruff 缓存缺失，无法离线安装；
- 未使用网络安装任何依赖。

因此当前视觉单元测试已用 pytest 通过，项目默认 pytest 全量命令和 ruff 检查仍待补齐开发依赖。

### 5.2 受限沙箱中的阻断

尝试运行：

```bash
.venv/bin/python examples/mobile_robot/room_navigation_vision.py \
  --steps 5 --scenario vision_route_showcase \
  --perception-mode disabled
```

以及：

```bash
.venv/bin/python examples/mobile_robot/room_navigation_vision.py \
  --steps 30 --scenario vision_route_showcase \
  --perception-mode ground_truth \
  --save-vision --save-sensors
```

均在受限沙箱的 Genesis 场景构建 OpenGL 初始化阶段失败：

```text
GenesisException: Failed to find an OpenGL 3.2+ core profile pixel format.
No OpenGL renderer is available on this machine.
```

这说明受限沙箱内无法验证：

- Genesis 场景是否完整构建；
- 小车是否按路线到达；
- RGB/Depth/LiDAR 的真实帧是否正常生成；
- ground-truth 或真实 YOLO 在 Genesis 图像上的运行结果；
- 端到端 `summary.json` 的 reached/collision 指标。

另一次直接运行旧的 `room_navigation_observable.py` 还暴露出 uv Python 安装目录下 Quadrants cache 不可写；新视觉入口已优先设置项目 `.quadrants_cache`，但这不替代宿主环境的 OpenGL 要求。

### 5.3 宿主环境完整 ground-truth 回归

在宿主 OpenGL 环境运行：

```bash
.venv/bin/python examples/mobile_robot/room_navigation_vision.py \
  --steps 1200 \
  --scenario vision_route_showcase \
  --perception-mode ground_truth \
  --save-images --save-sensors --save-vision \
  --vision-every 25 --image-every 50 --log-every 200 \
  --output-dir /tmp/mobile_robot_vision_gt_full_v2
```

结果：

| 指标            |                                                     结果 |
| --------------- | -------------------------------------------------------: |
| 仿真步数        |                                                     1126 |
| 仿真时间        |                                                  22.52 s |
| 到达            |                                                 `true` |
| 碰撞            |                                                `false` |
| 终止原因        |                                              `reached` |
| 视觉帧/推理次数 |                                                  45 / 45 |
| 检测总数        |                                                       25 |
| 稳定轨迹数      |                                                        3 |
| 类别覆盖        | `car` 4、`box_obstacle` 11、`cylinder_obstacle` 10 |
| 视觉异常        |                                                        0 |

| 终点位置        |                                     `(2.8001, 1.4241)` |
| 目标位置        |                                           `(2.8, 1.7)` |

三条轨迹均被稳定记录：黄色车最近距离约 1.53 m，红箱约 1.14 m，蓝圆柱约 1.08 m。完整输出位于 `/tmp/mobile_robot_vision_gt_full_v2/`，包含 45 条视觉结果、22 组图像序列、跟踪汇总、传感器 JSONL、telemetry CSV 和 summary JSON。

第一次完整回归曾因红箱体太靠近横向 waypoint 触发 LiDAR 安全层而在 `x≈-1.54,y≈1.38` 反复停车。已将红箱移至 `(0.4, 2.1)` 的安全走廊外并重新回归通过；该结果验证了场景设计和 LiDAR 安全优先级均能暴露问题并可修正。

### 5.4 宿主环境完整可视化回归

为验证“可视化执行”而不是仅保存离线图片，在宿主 OpenGL 环境同时打开 Genesis 总览 Viewer 与车载 RGB 窗口运行完整路线：

```bash
.venv/bin/python examples/mobile_robot/room_navigation_vision.py \
  --steps 1200 \
  --scenario vision_route_showcase \
  --perception-mode ground_truth \
  --vis --robot-view \
  --save-images --save-vision \
  --vision-every 25 --image-every 50 --log-every 200 \
  --output-dir /tmp/mobile_robot_vision_gui_full
```

执行结果：

| 指标 | 结果 |
| --- | ---: |
| Viewer / 车载 RGB 窗口 | 均成功启动 |
| 实际仿真步数 | 1126 |
| 到达 | `true` |
| 碰撞 | `false` |
| 终止原因 | `reached` |
| 视觉帧 / 推理次数 | 45 / 45 |
| 检测总数 | 25 |
| 稳定轨迹数 | 3 |
| 类别覆盖 | `car` 4、`box_obstacle` 11、`cylinder_obstacle` 10 |
| 视觉异常 | 0 |

该次运行确认 GUI 可视化链路和控制闭环均可用。终端出现的 Genesis 主线程交互提示以及 macOS AVFoundation 动态库重复提示不影响本次退出码（0）或上述结果。输出目录包含 `vision_results.jsonl`、标注图片和 `summary.json`；车载窗口显示附着相机的 RGB 视角。

### 5.5 关闭视觉的控制基线回归

为冻结视觉开关对导航行为的影响，在同一 `vision_route_showcase`、同一随机种子和 1200 步上运行 `perception-mode=disabled`：

```bash
.venv/bin/python examples/mobile_robot/room_navigation_vision.py \
  --steps 1200 \
  --scenario vision_route_showcase \
  --perception-mode disabled \
  --save-sensors --log-every 200 \
  --output-dir /tmp/mobile_robot_baseline_full
```

结果为 1126 步（22.52 s）到达、无碰撞、无超时，最终位置 `(2.8001, 1.4241)`，视觉帧/推理/检测均为 0；传感器与动作关联日志已写入 `sensor_observations.jsonl` 和 `telemetry.csv`。该结果与 ground-truth 视觉回归的到达步数和最终位置一致，说明当前视觉旁路未改变控制基线。

### 5.6 Genesis RGB-D 精确标定回归

按已确认的仿真口径，新增 `vision/rgbd_calibration.py`，直接读取 Genesis 运行时相机模型和挂载参数：

- RGB Camera：`256×192`，垂直 FOV `90°`，`fx=fy=96`、`cx=128`、`cy=96`；
- DepthCamera：`128×96`，水平 FOV `90°`，`fx=fy=64`、`cx=64`、`cy=48`；
- 两个传感器使用同一挂载位置 `(0.60, 0.00, 0.23)` 和朝向；运行时计算的 `depth_from_rgb` 平移残差约 `1.4e-7 m`，判定为同一光心；
- RGB bbox 通过两套 pinhole 内参投影到 Depth 像素，再用有效深度中位数、MAD、有效像素数计算距离置信度；同时输出相机、机器人和世界坐标；
- 每次运行保存 `camera_calibration.json`，视觉 JSONL 的 `depth_alignment` 在启用 `--calibrated-depth` 时标记为 `calibrated_pinhole`。原 `--approx-depth` 保留为兼容性降级路径。

更新后的完整 ground-truth 回归仍为 1126 步到达、0 碰撞、0 视觉异常；使用真实 RGB 内参后检测总数为 35（`car` 5、`box_obstacle` 14、`cylinder_obstacle` 16）。纯 Python 视觉测试更新为 `6 passed`。

回归命令：

```bash
.venv/bin/python examples/mobile_robot/room_navigation_vision.py \
  --steps 1200 --scenario vision_route_showcase \
  --perception-mode ground_truth --save-images --save-sensors --save-vision \
  --vision-every 25 --image-every 50 --log-every 200 \
  --output-dir /tmp/mobile_robot_vision_gt_calibrated
```

## 6. TODO 执行状态

| TASK 阶段            | 状态                           | 说明                                                                    |
| -------------------- | ------------------------------ | ----------------------------------------------------------------------- |
| VC0 基线冻结         | 已完成（模型资源待确认）       | 路线、类别、随机种子、运动学方案和关闭视觉基线已冻结；YOLO 权重/许可证仍待确认 |
| VC1 沿途多物体场景   | 已实现并通过 ground-truth 回归 | 3 类物体均被识别，路线到达且无碰撞                                      |
| VC2 离线模型可行性   | 阻塞                           | 仓库无本地 YOLO 权重，需提供权重或训练合成数据                          |
| VC3 实时边行驶边识别 | ground-truth 模式已通过        | 45 次推理、35 个检测、3 条轨迹；真实 YOLO 待权重                        |
| VC4 颜色/距离/跟踪   | 标定已实现，YOLO 待验证        | 颜色、精确 RGB-D 投影、距离置信度、机器人/世界坐标和跟踪已实现；真实 YOLO 仍待权重 |
| VC5 控制/安全/性能   | ground-truth 回归通过          | 1126 步到达、0 碰撞、0 视觉异常；实时性能指标仍需标准工具统计           |
| VC6 LLM 预留接口     | 已定义                         | `TaskSpec`/事件边界已写入两份 TASK，目标导航尚未接入                  |

## 7. 已知限制和风险

1. `ground_truth` 检测器是联调工具，不能作为视觉识别验收结果；
2. `--calibrated-depth` 已按 Genesis 已知内外参实现；`--approx-depth` 仅为兼容性降级路径，不能用于安全决策；
3. 真实 YOLO 需要项目专用权重。简化 Box/Cylinder 不一定被 COCO 预训练模型识别；
4. 当前 P0 视觉结果不直接改变导航动作，只保证 LiDAR 安全层；
5. 受限沙箱没有 OpenGL context；宿主环境已经完成 ground-truth 回归，后续仍需在团队标准机器固定测试；
6. URDF 四轮动力学仍应在运动学视觉闭环通过后单独校准；
7. `pytest-xdist/pytest-timeout/ruff` 尚未齐备，正式交付前应补齐开发依赖并执行项目标准测试命令。

## 8. 下一步交付顺序

1. 将宿主 OpenGL 回归命令固定到团队标准机器/CI，保留 ground-truth 结果作为场景回归基线；
2. 提供本地 YOLO 权重或用 Genesis segmentation 生成数据集并训练 `car/box_obstacle/cylinder_obstacle` 模型；
3. 使用同一场景运行 YOLO，统计类别召回率、误检、推理 P50/P95 和丢帧；
4. 使用本地 YOLO 权重运行 `--calibrated-depth`，验证检测框、深度和世界坐标的端到端误差；
5. 补齐 `pytest-xdist/pytest-timeout/ruff`，执行标准测试并把结果追加到本报告；
6. 视觉旁路稳定后，再评审视觉辅助限速/停车和 LLM `navigate_near_object` 集成。

## 9. 交付清单

- [X] 视觉/控制主任务文档拆分；
- [X] 沿途多物体场景代码；
- [X] 实时视觉入口；
- [X] disabled/ground-truth/yolo 三种模式；
- [X] 结果叠加、跟踪和 JSONL/CSV 输出；
- [X] 模型缺失/异常安全降级；
- [X] 纯 Python 单元级验证；
- [X] 有 OpenGL 机器上的 Genesis ground-truth 端到端运行报告；
- [ ] 真实 YOLO 权重上的识别指标；
- [X] Genesis RGB-D 标定和世界坐标融合实现/回归；真实 YOLO 下的误差报告待模型权重；
- [X] 视觉单元 pytest 报告（使用 `-o addopts=''`）；
- [ ] 项目默认 pytest + ruff 标准测试报告。
