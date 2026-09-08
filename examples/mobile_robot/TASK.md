# 移动小车视觉感知与控制任务

> 文档定位：视觉/小车控制主任务文档
>
> 当前验收故事：小车按照现有简易导航策略驶向既定目的地，并在行驶全过程持续识别沿途不同物体，实时展示并记录识别结果，最终安全到达。
>
> LLM 工作流：已拆分到 [TASK_LLM.md](TASK_LLM.md)，不作为当前视觉控制任务的前置条件。

## 1. 当前阶段目标

本阶段先实现一条不依赖 LLM 的稳定闭环：

```text
预设目的地 / 简易 waypoint 策略
                │
                ▼
      路径跟踪 + LiDAR 安全层
                │
                ▼
          Genesis 小车运动
                │
                ├──────────────► 到达目的地并停车
                │
                ▼
       车载 RGB / Depth / LiDAR
                │
                ▼
   YOLO 检测 + 颜色/距离 + 多帧跟踪
                │
                ▼
      实时叠加显示 + JSONL/视频记录
```

用户能够看到：

1. 小车按当前简易策略持续驶向既定目标点；
2. 第一视角中沿途物体被检测框标出；
3. 每个结果显示类别、置信度，并逐步补充颜色、距离和跟踪 ID；
4. 识别不会阻塞或破坏原有驾驶闭环；
5. 小车遇到近距离障碍仍由 LiDAR 安全层减速/停车；
6. 最终小车无碰撞到达目的地，运行日志可以按帧复盘“看到了什么、当时在哪里、执行了什么动作”。

本阶段的识别结果先作为**旁路观察能力**，不直接决定目的地或转向。这样能把视觉问题与导航问题隔离开：即使视觉模型漏检，车辆仍可依靠原有简易策略和 LiDAR 安全层完成路线。

## 2. 职责拆分

### 2.1 视觉与小车控制

负责：

- Genesis 视觉展示场景和沿途物体；
- 车载 RGB/Depth/LiDAR 的采集与时间同步；
- YOLO 或等价检测器的本地推理；
- 颜色、距离、坐标和多帧跟踪；
- 识别结果叠加、日志和性能统计；
- 现有简易导航策略的稳定运行；
- LiDAR 安全层、到达判定和异常停车；
- 对外提供稳定的观测、动作和语义对象接口。

### 2.2 自然语言任务理解

负责：

- 用户文本输入；
- LLM API 客户端和密钥配置；
- 自然语言到结构化 `TaskSpec` 的转换；
- Schema 校验、歧义澄清、超时和错误处理；
- 调用视觉控制团队暴露的高层技能；
- 向用户反馈任务状态。

详细 TODO 见 [TASK_LLM.md](TASK_LLM.md)。当前视觉控制验收不调用真实 LLM，可使用固定目标、CLI 参数或 Fake TaskSpec 启动。

### 2.3 集成边界

视觉控制不接收 LLM 生成的轮速或路径点。后续只接收受约束的高层任务：

```json
{
  "intent": "navigate_near_object",
  "target": {
    "category": "car",
    "attributes": {"color": "yellow"}
  },
  "constraints": {
    "near_distance_m": 0.8,
    "face_target": true
  }
}
```

视觉控制侧向 LLM/上层返回结构化事件：

```text
TASK_ACCEPTED
TARGET_ACQUIRED
PATH_PLANNED
EXECUTING
ARRIVED
TARGET_NOT_FOUND
PATH_NOT_FOUND
COLLISION_RISK
FAILED(reason=...)
```

此接口属于后续集成要求，不影响当前“既定目的地 + 沿途识别”的独立实现。

## 3. 当前代码基线

| 能力                 | 现有位置                                              | 当前状态                                       |
| -------------------- | ----------------------------------------------------- | ---------------------------------------------- |
| 稳定运动学小车       | `environment.py`、`room_navigation_observable.py` | 已有`reset/observe/step`，适合先完成视觉闭环 |
| 四轮 URDF 小车       | `room_navigation_urdf.py`                           | 已能执行轮速，轮胎/里程计仍需继续校准          |
| 简易导航             | `DifferentialDriveController`                       | 固定 waypoint + LiDAR 前向安全层               |
| 车载 RGB             | `robot_rgb_camera`                                  | 已挂载，但是否采帧仍与 render/保存条件耦合     |
| Depth/LiDAR/IMU/位姿 | 统一 observation                                      | 已有基础数据                                   |
| 视觉协议             | `vision/types.py`                                   | 已有`FramePacket/Detection/VisionResult`     |
| YOLO 适配            | `vision/yolo_detector.py`                           | 已有本地模型适配器，尚未进入实时控制循环       |
| 离线推理             | `vision/offline_infer.py`                           | 可对保存图片推理，仍需要本地模型权重           |
| 检测叠加             | `vision/overlay.py`                                 | 已有基础框和标签绘制                           |
| 视觉控制闭环         | 无                                                    | 尚未形成“边行驶边识别”的完整入口             |

### 3.1 当前关键缺口

1. 现有场景物体主要是 Box 障碍物，没有清晰、稳定的视觉类别集合和展示布局；
2. 仓库当前没有可直接验收的本地 YOLO 权重；
3. COCO 预训练模型不保证识别 Genesis 的简单 Box/Cylinder 或低纹理 URDF；
4. `room_navigation_urdf.py` 只创建 `vision_frame`，`observation["vision"]` 仍为 `None`；
5. 算法采帧与 GUI、保存图片条件耦合，headless 模式可能没有 RGB 输入；
6. RGB 为 `256×192`，DepthCamera 为 `128×96`，不能默认检测框和深度逐像素对齐；
7. 缺少运行时推理频率、延迟、结果年龄和丢帧管理；
8. 缺少颜色、距离、世界坐标和多帧跟踪；
9. 缺少贯穿整段路线的识别日志和端到端指标；
10. URDF 动力学的不稳定性可能掩盖视觉问题，应先在运动学环境完成视觉闭环。

## 4. 本阶段范围

### 4.1 P0 必须完成

- 小车使用既有简易 waypoint/LiDAR 策略驶向固定目的地；
- 场景沿路线布置至少 3 类可区分物体；
- 车载相机按独立频率持续产生 RGB 帧；
- 本地 YOLO/检测器在运行中持续推理；
- 第一视角显示检测框、类别、置信度、帧号和延迟；
- 原图、标注图、检测 JSONL、动作和位姿可按帧对应；
- 模型关闭、异常、超时或推理过慢时，小车仍能安全完成或停车；
- 到达目的地、发生碰撞和运行超时均有明确结果。

### 4.2 P1 建议完成

- 目标 ROI/mask 的颜色分类；
- 检测框与对齐深度融合，输出米制距离；
- 相机坐标、机器人坐标和世界坐标转换；
- 多帧目标跟踪和稳定 ID；
- YOLO-Seg 或自定义模型；
- 异步 latest-only 推理；
- 检测结果作为视觉辅助安全信号，但不替代 LiDAR。

### 4.3 当前不作为前置条件

- 真实 LLM API；
- 自然语言解析；
- 根据“黄色小车”动态改变目的地；
- 视觉结果直接输出轮速；
- 完整 SLAM；
- 动态目标跟随；
- ROS2 或真实车辆部署；
- Web/桌面交互界面。

## 5. 展示场景设计

### 5.1 推荐场景

新增一个独立场景，例如 `vision_route_showcase`：

- 使用现有房间边界；
- 使用一条可解释、不会紧贴墙体的 waypoint 路线；
- 路线终点继续使用固定绿色目标标记；
- 在路线左右两侧布置不同类别/颜色物体；
- 保证物体会进入车载相机视野，同时不全部堵住路线；
- 至少有一个远距离、一个近距离和一个部分遮挡物体；
- 保留一个只被 LiDAR 感知的普通障碍，用于证明安全层独立存在。

默认验收物体集合建议为：

| 语义类别                                  | 示例颜色    | 用途                         |
| ----------------------------------------- | ----------- | ---------------------------- |
| `car`                                   | yellow      | 对接后续“黄色小车”语言任务 |
| `box_obstacle`                          | red         | 验证自定义障碍物类别         |
| `cylinder_obstacle` 或 `traffic_cone` | blue/orange | 验证不同形状与类别           |

如果使用 COCO 预训练模型，应把场景资产替换为模型能识别的真实外形 mesh；如果继续使用 Genesis 基本几何体，则需要自定义训练集，不能把 COCO 未检测到 Box 视为代码故障。

### 5.2 场景语义注册

每个展示物体注册真值元数据：

```python
SceneObject(
    object_id="yellow_car_01",
    category="car",
    attributes={"color": "yellow", "role": "landmark"},
    entity_idx=...,
    footprint=...,
)
```

真值元数据只用于：

- 自动生成训练标注；
- 计算检测召回率、颜色准确率和定位误差；
- 调试场景；
- 端到端测试断言。

运行时 `YoloDetector` 不得读取真值类别或位置。日志必须标明 `perception_mode=yolo` 或 `perception_mode=ground_truth`，防止把真值演示误认为视觉模型效果。

## 6. 视觉处理链路

### 6.1 采帧

将以下三个概念拆开：

- `vision_enabled`：是否为算法采集 RGB-D；
- `robot_view`：是否显示第一视角窗口；
- `save_images`：是否保存图片。

即使 `robot_view=False` 且 `save_images=False`，只要 `vision_enabled=True`，系统就必须按 `vision_every` 产生视觉帧。

建议每个视觉输入使用已有 `FramePacket`：

```python
FramePacket(
    frame_id=step,
    sim_time=step * dt,
    image=rgb,
    camera_name="robot_rgb_camera",
    camera_pose=...,
    intrinsics=...,
)
```

Genesis Camera 的 `render()` 保留在仿真线程。若使用异步推理，工作线程只接收已复制/冻结的数组，不跨线程操作渲染器。

### 6.2 YOLO 检测

复用 `vision/yolo_detector.py`，运行时增加建议参数：

```text
--vision
--vision-model path/to/model.pt
--vision-device cpu/cuda/mps
--vision-every 5
--vision-conf 0.5
--vision-max-age 0.5
--save-vision
```

模型路径必须显式存在；程序不应在演示时隐式联网下载权重。启动时记录：

- 权重文件路径和 hash；
- Ultralytics/Torch 版本；
- 设备与输入尺寸；
- 置信度和 NMS 阈值；
- 类别表。

### 6.3 颜色、距离和坐标

普通 YOLO 只直接提供类别、置信度和二维框。扩展属性的数据来源应明确：

```text
YOLO / YOLO-Seg        → 类别、置信度、bbox/mask
HSV/Lab ROI 分析       → 颜色、颜色置信度
对齐 Depth / LiDAR      → 表面距离
相机内外参 + 机器人位姿  → 世界坐标
```

RGB 与 Depth 推荐使用同一成像模型和分辨率。若保留两个相机，必须保存内参和外参并做投影对齐；禁止只按分辨率比例缩放 bbox 后索引深度。

深度估计流程：

1. 取 bbox 中央区域或分割 mask；
2. 去除无效、零值和超量程深度；
3. 去除背景离群点；
4. 使用中位数或较低分位数估计目标表面距离；
5. 保存有效像素数、方差和距离置信度；
6. 必要时使用 LiDAR 投影或前方回波交叉验证。

### 6.4 多帧跟踪

“一路识别”不能只保存互不相关的单帧框。P1 建议增加轻量跟踪：

- 通过 IoU、类别和 3D 邻近关联相邻帧；
- 为同一物体分配稳定 `track_id`；
- 连续若干帧命中后才标记 `confirmed`；
- 记录首次出现、最后出现、可见帧数和最近距离；
- 结果超时后标记 stale；
- 不把旧检测结果无期限复制给后续控制周期。

## 7. 小车控制链路

### 7.1 P0 控制策略

保持现有简易策略：

```text
固定目的地
  ↓
预设 waypoint 依次跟踪
  ↓
LiDAR 前方安全检查
  ├──安全：继续 waypoint 跟踪
  └──近障碍：停车并朝空旷侧转向
  ↓
linear_velocity / angular_velocity
```

视觉推理首期在旁路运行，不改变上述动作。这样可以分别回答：

- 小车控制是否稳定到达？
- 相机是否在整个路线持续产生正确帧？
- 模型在不同距离和视角下识别到了什么？

### 7.2 安全要求

- LiDAR 安全层优先级高于 waypoint 控制和视觉模块；
- 速度上限沿用 `CarConfig` 并允许 CLI 进一步降低；
- 连续非有限 LiDAR、控制超时或定位跳变时输出零速度；
- `Ctrl+C`、异常退出和窗口关闭时执行停车；
- 视觉模型异常不得导致控制线程崩溃；
- 同步推理超过最大允许耗时，应记录告警并跳过后续帧；
- 异步推理仅保留最新帧，不能因队列积压使用数秒前结果；
- 视觉辅助控制启用后，视觉只能额外限速/停车，不能覆盖 LiDAR 的停车决定。

### 7.3 到达判定

到达目的地至少满足：

- 与最终 waypoint 距离小于阈值；
- 无碰撞；
- 线速度和角速度归零；
- 连续若干控制周期保持稳定；
- summary 中写入 `termination_reason=reached`。

沿途检测数量不是“到达”的前置条件，但属于视觉验收指标。路线成功和视觉成功应分别统计，避免用车辆到达掩盖模型未识别物体。

## 8. 运行时架构与时序

### 8.1 推荐频率

| 回路              |   建议频率 | 说明                        |
| ----------------- | ---------: | --------------------------- |
| Genesis 仿真      |      50 Hz | 沿用`dt=0.02`             |
| 控制和 LiDAR 安全 |   20~50 Hz | 每个控制周期执行            |
| RGB-D 采集        |    5~10 Hz | 由`vision_every` 控制     |
| YOLO 推理         | 取决于设备 | 同步 MVP 或异步 latest-only |
| 可视化/落盘       |    2~10 Hz | 不应拖慢控制回路            |

### 8.2 单步时序

建议明确观测与动作的物理时刻：

```text
observation(k)
  ├── 控制器计算 action(k)
  ├── 小车执行 action(k)
  ├── Genesis scene.step()
  └── 读取 observation(k+1)
         ├── LiDAR/Depth/IMU/pose
         └── 若到视觉周期：RGB frame(k+1) → detector
```

`VisionResult.frame_id` 必须引用其输入图像帧；日志中的 `action_step` 必须说明该动作产生的是哪一个后继观测。

### 8.3 同步与异步策略

第一步可采用同步推理验证正确性。性能不足时改为：

```text
仿真线程 ──► latest_frame_queue(maxsize=1)
                         │
                         ▼
                   推理工作线程
                         │
                         ▼
              latest_result（带时间戳）
```

队列满时替换旧帧而不是阻塞仿真。显示和日志可以使用最近结果，但任何未来参与控制的视觉结果都必须检查 `current_sim_time - result.sim_time <= max_age`。

## 9. 建议代码组织

```text
examples/mobile_robot/
├── TASK.md                         # 本文：视觉/控制主任务
├── TASK_LLM.md                     # LLM 独立任务
├── room_navigation_vision.py       # 当前团队最终演示入口
├── environment.py                  # 现有运动学标准环境
├── room_navigation_urdf.py         # 后续动力学版本
├── scenarios/
│   └── vision_route_showcase.py    # 沿途多物体场景
├── vision/
│   ├── types.py                    # 已有 FramePacket/Detection/VisionResult
│   ├── detector.py                 # 已有检测器协议
│   ├── yolo_detector.py            # 已有 YOLO 适配器
│   ├── overlay.py                  # 已有可视化基础
│   ├── runtime.py                  # 实时/异步推理调度
│   ├── attributes.py               # 颜色属性
│   ├── rgbd_fusion.py              # 深度与坐标转换
│   ├── tracker.py                  # 多帧跟踪
│   └── ground_truth.py             # 标注和评估后端
└── tests/
    ├── test_vision_runtime.py
    ├── test_attributes.py
    ├── test_rgbd_fusion.py
    ├── test_tracker.py
    └── test_room_navigation_vision.py
```

不要把 YOLO 加载、颜色判断、日志、导航控制全部堆进演示脚本；入口只负责装配各模块和运行循环。

## 10. 当前团队 TODO

以下 TODO 只属于视觉/小车控制工作流，不包含真实 LLM API 实现。

### VC0：冻结验收场景与基线

- [ ] `[P0]` 确定沿途需要识别的类别集合；默认 `car/box_obstacle/cylinder_or_cone`。
- [ ] `[P0]` 确定最终先使用运动学小车还是 URDF；建议运动学版本先验收。
- [ ] `[P0]` 确定可用本地模型权重、推理设备和模型许可证。
- [ ] 固化现有简易路线、目的地、随机种子和运行命令。
- [ ] 记录无视觉推理时的到达步数、碰撞、实时率和传感器输出。

**交付物：** 场景/类别清单、基线 `summary.json` 和演示截图。

### VC1：沿途多物体场景

- [ ] `[P0]` 新增 `vision_route_showcase` 场景。
- [ ] `[P0]` 沿路线布置至少 3 类物体，覆盖近/远、左右和部分遮挡视角。
- [ ] `[P0]` 保证物体不使预设路线变成不可达。
- [ ] 为物体增加稳定 ID、类别、颜色、footprint 和 entity ID 注册。
- [ ] 支持固定种子和可选位置随机化。
- [ ] 输出俯视场景图和预期可见区间。

**验收：** 不启用视觉时，小车仍能按简易策略无碰到达；整段路线中所有验收物体至少进入一次相机有效视野。

### VC2：离线模型可行性

- [ ] `[P0]` 保存一轮完整路线的车载 RGB 数据。
- [ ] `[P0]` 使用已有 `offline_infer.py` 对全量图片推理。
- [ ] 统计每类物体在不同距离、角度和遮挡条件下的检测效果。
- [ ] 判断预训练权重是否可用；不能识别时形成明确结论，不继续盲调阈值。
- [ ] 必要时用 Genesis segmentation 生成 bbox/mask 标注。
- [ ] 必要时训练/微调自定义 YOLO 或 YOLO-Seg 权重。
- [ ] 固定模型版本、类别映射、置信度和输入尺寸。

**验收：** 在冻结验证集上达到第 13 节门槛，标注图经人工抽查类别和框位置正确。

### VC3：实时“边行驶边识别”闭环

- [ ] `[P0]` 新增 `room_navigation_vision.py`，不修改稳定基线行为。
- [ ] `[P0]` 将算法采帧与 GUI/图片保存解耦。
- [ ] `[P0]` 在视觉周期创建 `FramePacket` 并调用 `VisionDetector.detect()`。
- [ ] `[P0]` 将 `VisionResult` 写入当前观测和独立 JSONL。
- [ ] `[P0]` 第一视角叠加检测框、类别、置信度、帧号和延迟。
- [ ] `[P0]` 保留原图，标注图输出到独立目录。
- [ ] `[P0]` 记录每帧检测数量、类别、bbox、推理耗时和结果年龄。
- [ ] `[P0]` 模型不可用或推理异常时降级为 `vision_status=unavailable/error`，控制循环继续安全运行。

**验收：** 小车完整行驶期间持续产生视觉结果，到达行为与无视觉基线一致，检测日志可关联到图像、机器人位姿和动作。

### VC4：颜色、距离与跟踪

- [ ] 标定黄色、红色、蓝色等 HSV/Lab 区间。
- [ ] 优先在 segmentation mask 内统计颜色；bbox 模式去除边界和背景。
- [ ] 统一 RGB-D 相机模型，或完成 RGB 到 Depth 的标定和重投影。
- [ ] 输出检测目标表面距离和距离置信度。
- [ ] 输出相机坐标、机器人坐标和世界坐标。
- [ ] 与 LiDAR 投影/最近回波交叉检查明显异常距离。
- [ ] 实现多帧关联、稳定 `track_id`、确认和过期状态。
- [ ] 在行程结束后生成“沿途物体清单”。

**验收：** 同一物体跨帧 ID 基本稳定，颜色和距离达到冻结指标，旧结果不会被误记为当前帧结果。

### VC5：控制、安全与性能

- [ ] `[P0]` 回归 waypoint + LiDAR 安全层，视觉开/关均能稳定运行。
- [ ] `[P0]` 确保异常、用户中断和正常到达都会输出零速度。
- [ ] 添加视觉处理预算和超时告警。
- [ ] 推理影响实时率时切换到 `maxsize=1` 的 latest-only 异步模式。
- [ ] 记录仿真 FPS、推理延迟 P50/P95、丢帧和结果年龄。
- [ ] 注入模型异常、慢推理和空检测，验证控制不崩溃。
- [ ] 可选实现视觉辅助限速/停车；启用时必须保留 LiDAR 最高优先级。
- [ ] 完成运动学版本后，再将同一视觉接口迁移到 URDF 版本。

**验收：** 视觉故障不导致碰撞或控制进程异常；端到端指标达到第 13 节要求。

### VC6：为 LLM/语义导航预留接口（后续集成）

- [ ] 输出稳定的 `SemanticObject[]`，包含类别、属性、位置、时间和数据源。
- [ ] 提供 `list_visible_objects()` 和 `get_task_status()` 查询接口。
- [ ] 接收经过校验的 `TaskSpec`，不接收自然语言和任意代码。
- [ ] 实现 `navigate_near_object()` 前先接 Fake TaskSpec 测试。
- [ ] 增加黄色目标搜索、目标附近停车位和 A*；这些不属于当前“固定目的地沿途识别”P0。
- [ ] 与 LLM 使用 mock 做契约测试，双方不依赖对方真实服务开发。

**验收：** LLM 接入时不需要改动相机、检测器、控制动作和日志基础协议。

## 11. 数据接口

### 11.1 视觉结果

复用并扩展现有 `VisionResult`：

```python
VisionResult(
    frame_id=120,
    sim_time=2.4,
    model_name="car_obstacle_yolo",
    latency_ms=18.4,
    detections=(
        Detection(
            class_id=0,
            label="car",
            confidence=0.94,
            bbox_xyxy=(82, 44, 148, 126),
            color="yellow",
            color_confidence=0.91,
            distance_m=2.15,
            position_world=(1.2, 0.7, 0.3),
        ),
    ),
)
```

### 11.2 沿途物体汇总

```python
@dataclass(frozen=True)
class TrackedObjectSummary:
    track_id: str
    category: str
    dominant_color: str | None
    max_confidence: float
    first_seen_sim_time: float
    last_seen_sim_time: float
    visible_frame_count: int
    nearest_distance_m: float | None
    last_position_world: tuple[float, float, float] | None
```

### 11.3 控制动作

继续沿用现有物理单位：

```python
action = {
    "linear_velocity": 0.3,  # m/s
    "angular_velocity": 0.0, # rad/s
}
```

## 12. 日志与可视化

每次运行创建独立目录：

```text
out/mobile_robot_vision/<run_id>/
├── config_resolved.json
├── summary.json
├── telemetry.csv
├── sensor_observations.jsonl
├── vision_results.jsonl
├── tracked_objects.json
├── rgb_robot/
├── rgb_robot_annotated/
├── depth/
└── demo.mp4
```

`summary.json` 至少包含：

- 场景、随机种子和车辆模式；
- 模型、权重 hash、设备和阈值；
- 是否到达、碰撞、超时和最终位置；
- 总视觉帧数、推理帧数、丢弃帧数；
- 每类检测次数和唯一轨迹数；
- 推理延迟 P50/P95；
- 最旧结果年龄；
- 视觉异常数量；
- 终止原因。

## 13. 测试与量化验收

### 13.1 测试场景

| 编号 | 场景                   | 预期结果                         |
| ---- | ---------------------- | -------------------------------- |
| V01  | 无视觉模型运行原路线   | 无碰到达，得到控制基线           |
| V02  | 开启 YOLO 运行原路线   | 到达结果与基线一致，持续输出识别 |
| V03  | 3 类物体分布在路线两侧 | 各类至少被稳定识别一次           |
| V04  | 物体远近和角度不同     | 日志能反映置信度/距离变化        |
| V05  | 一个物体部分遮挡       | 不因单帧漏检创建大量新轨迹       |
| V06  | 模型文件不存在         | 启动失败或安全降级，错误明确     |
| V07  | 推理异常/空结果        | 控制不崩溃，小车继续安全行驶     |
| V08  | 推理速度低于采帧速度   | 不积压旧帧，结果年龄受控         |
| V09  | headless 且不保存原图  | 仍持续产生视觉结果               |
| V10  | 前方存在近障碍         | LiDAR 安全层仍优先停车/转向      |
| V11  | 用户中断               | 立即停车并写出终止状态           |
| V12  | 四轮 URDF 版本         | 接口不变，单独记录动力学问题     |

### 13.2 建议验收指标

在冻结场景与至少 10 个固定随机种子上：

| 指标                           | P0 建议门槛 |
| ------------------------------ | ----------: |
| 导航到达成功率                 |  `>= 95%` |
| 几何碰撞率                     |      `0%` |
| 预期可见物体类别召回率         |  `>= 90%` |
| 检测类别准确率                 |  `>= 90%` |
| 视觉结果与图像 frame_id 匹配率 |    `100%` |
| 视觉异常导致控制崩溃           |    `0 次` |
| 过期结果被当作当前结果记录     |    `0 次` |
| 运行结束非零速度               |    `0 次` |

P1 建议门槛：

| 指标               |                   建议门槛 |
| ------------------ | -------------------------: |
| 颜色属性准确率     |                 `>= 95%` |
| 距离中位绝对误差   |              `<= 0.20 m` |
| 同一物体轨迹碎片数 |             根据验证集冻结 |
| 推理延迟 P95       | 根据目标机器和控制预算冻结 |

### 13.3 必需测试

- `FramePacket` 图像格式、帧号和时间戳；
- 模型禁用、模型不存在和检测为空；
- bbox 越界和异常模型输出；
- RGB-D 对齐与坐标变换；
- 颜色低置信度输出 `unknown`；
- 跟踪确认、丢失和 stale；
- 异步队列只保留最新帧；
- 视觉开/关时导航结果回归；
- 到达、碰撞、超时和用户中断停车；
- 日志中的图像、检测、动作和位姿关联。

## 14. 主要风险

| 风险                      | 后果           | 处理方式                           |
| ------------------------- | -------------- | ---------------------------------- |
| COCO 权重不识别合成几何体 | 沿途无检测     | 换真实外形资产或训练自定义模型     |
| RGB/Depth 不对齐          | 距离和坐标错误 | 同一 RGB-D Camera 或正式标定重投影 |
| 推理阻塞控制              | 小车卡顿       | 降频、异步 latest-only、延迟监控   |
| 检测帧和动作时间错位      | 无法复盘       | 强制 frame_id/sim_time/action_step |
| GUI 关闭后没有 RGB        | headless 失效  | 算法采帧与 GUI/保存开关解耦        |
| 视觉异常传播到主循环      | 控制中断       | 检测器边界捕获异常并安全降级       |
| 场景物体从未进入相机视野  | 指标失真       | 预先验证路线可见区间               |
| URDF 滑移导致路线失败     | 误判为视觉问题 | 先用运动学版本验收，再迁移 URDF    |
| 用真值结果冒充 YOLO       | 演示链路不真实 | 模式隔离、日志标识和真值泄漏测试   |
| 一开始让视觉参与驾驶      | 难以定位问题   | P0 旁路识别，视觉控制作为后续阶段  |

## 15. 当前需要确认的事项

1. 沿途具体需要识别哪些物体？默认采用黄色车、红色 Box、蓝色 Cylinder/Cone 三类。
2. 是否已有可用 YOLO 权重？若没有，需要选择真实外形资产或安排合成数据训练。
3. P0 是否只要求显示类别，还是必须同时显示颜色和距离？本文将类别列为 P0，颜色/距离列为 P1。
4. 最终演示先使用稳定运动学车还是必须使用四轮 URDF？默认先运动学、后 URDF。
5. 视觉是否首期参与避障？默认不参与，LiDAR 继续负责安全，视觉先旁路识别。
6. 目标运行平台是 CPU、CUDA 还是 Apple Metal？这会决定模型大小、输入分辨率和推理频率。

## 16. Definition of Done

当前视觉/控制任务只有同时满足以下条件才算完成：

- 小车无需 LLM 即可按简易策略启动并驶向既定目的地；
- 运行全过程按固定频率采集车载视觉，不依赖是否打开 GUI 或保存图片；
- 沿途约定的不同物体被真实检测器识别并在第一视角展示；
- 检测结果与图像、时间、机器人位姿和动作一一关联；
- 小车安全到达，视觉开关不会破坏控制基线；
- 模型不可用、推理异常、结果过期和用户中断都有安全行为；
- 原图、标注图、检测日志、物体汇总、控制日志和 summary 齐全；
- 自动化测试与固定场景回归达到冻结后的指标；
- 对外语义对象和任务状态接口稳定，可供 LLM 后续接入。

---

相关资料：

- [TASK_LLM.md](TASK_LLM.md)：LLM 独立任务；
- [README.md](README.md)：当前示例运行方式；
- [STRATEGY.md](STRATEGY.md)：现有简易 waypoint/LiDAR 策略；
- [_docs/YOLO_VISION_INTEGRATION_DESIGN_AND_PLAN.md](_docs/YOLO_VISION_INTEGRATION_DESIGN_AND_PLAN.md)：已有 YOLO 接入设计；
- [_docs/SIMULATION_ANALYSIS_AND_PLAN.md](_docs/SIMULATION_ANALYSIS_AND_PLAN.md)：仿真闭环分析；
- [_reports/EXECUTION_REPORT.md](_reports/EXECUTION_REPORT.md)：本次执行、测试和交付报告。
