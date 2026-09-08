# 自然语言驱动小车语义导航：需求、设计与实施任务

> 文档状态：方案基线（待评审）  
> 适用目录：`examples/mobile_robot/`  
> 核心演示命令：“行驶到黄色小车附近”  
> 最终目标：用户输入自然语言后，小车能够识别所指目标、规划无碰路径、行驶到目标附近并安全停车，同时完整展示和记录任务理解、视觉识别、规划、控制与执行结果。

## 1. 结论先行

本需求不是简单地让 LLM 输出一串左右轮速度，而是要构建以下闭环：

```text
自然语言
  ↓
LLM 任务理解（结构化 TaskSpec）
  ↓
语义目标解析（“黄色” + “小车”）
  ↓
目标搜索、视觉识别、RGB-D 定位与多帧跟踪
  ↓
目标附近的安全停车位生成
  ↓
全局路径规划 + 局部避障 + 底层差速控制
  ↓
Genesis 小车运动与传感器回传
  ↓
到达判定、过程可视化、日志和失败反馈
```

推荐的职责边界如下：

- **LLM 只理解用户意图并调用受约束的高层技能**，例如 `navigate_near_object`；不让 LLM 按控制周期输出轮速，也不让它替代 A*、局部规划器或安全停车逻辑。
- **YOLO 只负责“这可能是一辆车及其二维框”**。颜色由 ROI 颜色模块判断，距离由深度相机/LiDAR 提供，世界坐标由相机标定和机器人位姿变换得到。
- **导航算法负责真正的路径规划**。全局层使用栅格代价地图和 A*，局部层使用路径跟踪与动态避障；LiDAR/碰撞安全层可以无条件覆盖其他模块并停车。
- **Genesis 真值只用于开发调试、生成标注和评估**。最终“纯感知闭环”验收不得把黄色小车预置世界坐标直接传给导航器，否则只能证明导航成功，不能证明识别成功。

第一版应把范围控制在：单房间、单台受控小车、静态目标、静态障碍物、中文终端输入、单轮任务、已知或可构建二维地图。多轮对话、动态行人、真实机器人和开放世界识别放到后续阶段。

## 2. 需求边界与验收口径

### 2.1 必须实现的用户故事

用户在终端输入：

```text
行驶到黄色小车附近
```

系统应依次做到：

1. 调用配置好的 LLM API，将自然语言解析为合法、可验证的结构化任务；
2. 理解目标类别为 `car`，颜色属性为 `yellow`，动作意图为 `navigate_near`；
3. 从当前视野或搜索过程中找到黄色静态小车；
4. 估计黄色小车相对机器人和世界坐标系的位置；
5. 在目标周围选择一个可达、无碰撞、满足停靠距离的目标位姿，而不是驶入目标本体中心；
6. 根据环境障碍物规划路线并闭环执行；
7. 路径被阻挡时重新规划，感知失效或危险距离触发安全停车；
8. 到达后停稳，面向目标，并反馈“已到达黄色小车附近”；
9. 界面/终端能看到任务状态，观察窗口能看到检测框、目标点和规划路径，日志可以复盘全过程。

### 2.2 “附近”的明确含义

“附近”必须转成可计算条件。MVP 默认值建议为：

- 机器人车体边缘到目标物体边缘的期望间距：`0.6 ~ 1.0 m`；
- 机器人中心到目标中心的精确阈值由双方包围半径换算，不使用固定中心距离硬编码；
- 停车位必须在膨胀后代价地图的自由空间中；
- 最终朝向误差建议不超过 `20°`；
- 线速度和角速度连续若干帧低于阈值后才算“停稳”；
- 最终判定必须同时满足距离、无碰撞、停车和目标仍有效四个条件。

默认 `near_distance_m=0.8`，可由用户表达“离它一米左右”“靠近但别太近”覆盖，但必须被限制在安全配置范围内。

### 2.3 MVP 不包含的内容

- LLM 直接读取每帧原始图像并实时驾驶；
- LLM 生成每个仿真步的 `linear_velocity/angular_velocity`；
- 未知大场景中的完整视觉 SLAM；
- 动态目标跟随、动态行人预测；
- 多楼层导航、多机器人协作；
- 未经适配直接部署到真实车辆；
- 仅凭二维检测框宽度猜测实际距离。

## 3. 当前代码基线与缺口

### 3.1 已有能力

当前目录已经具备以下可复用基础：

| 能力 | 现有位置 | 当前状态 |
|---|---|---|
| 稳定运动学小车与标准环境 | `environment.py`、`room_navigation_observable.py` | 已有 `reset/observe/step`，适合先打通功能闭环 |
| 四轮 URDF 动力学小车 | `room_navigation_urdf.py` | 已能执行轮速，但轮胎/里程计参数仍需校准 |
| 车载传感器 | RGB、Depth、LiDAR、IMU、里程计、机器人位姿 | 已有，但 RGB/Depth 是否像素严格对齐需确认和改造 |
| 基线控制 | 固定航点 + LiDAR 前向安全层 | 能跑通，不能根据语义目标动态生成路径 |
| 算法动作接口 | `linear_velocity`、`angular_velocity` | 可复用 |
| 视觉数据协议 | `vision/FramePacket`、`Detection`、`VisionResult` | 基础协议已存在 |
| YOLO 适配 | `vision/yolo_detector.py`、`offline_infer.py` | 仅有本地离线推理入口；没有权重、实时融合和导航闭环 |
| 传感器与动作日志 | CSV、JSON/JSONL、图像 | 已有基础记录能力 |

### 3.2 关键缺口

1. 当前场景没有作为语义任务目标的“黄色静态小车”实体及其元数据；绿色圆盘只是固定目标点。
2. 当前路径由固定航点预先写死，没有占据栅格、动态目标点、A* 或局部规划器。
3. YOLO 检测结果尚未在仿真控制循环中实时产生。
4. 普通 YOLO 检测只能返回类别/检测框，不能直接输出颜色、距离和可靠世界坐标。
5. 当前 RGB 渲染相机为 `256×192`，DepthCamera 为 `128×96`；即使安装位姿接近，也不能假设检测框与深度图天然逐像素对应。
6. 目标可能不在初始相机视野内，缺少搜索/探索状态。
7. 缺少 LLM 客户端、结构化输出协议、超时/重试、非法输出校验和降级策略。
8. 缺少任务编排状态机和失败原因体系。
9. 当前运动学环境适合功能验证，URDF 环境更接近最终物理展示，但其轮胎、里程计和控制稳定性仍是独立风险。
10. 缺少可量化的端到端测试场景与成功率统计。
11. 当前 `MobileRobotEnv` 在 `render=False` 时不会产生车载 RGB；正式视觉采集不能与“是否打开 GUI/是否保存图片”使用同一个开关。
12. “小车”既可能指受控机器人自身，也可能指场景中的静态目标车，缺少角色语义和排除本车的规则。

## 4. 推荐总体架构

```text
┌──────────────┐
│ CLI / Web UI │
└──────┬───────┘
       │ user_text
       ▼
┌────────────────────┐       JSON Schema / Pydantic 校验
│ LLM Task Interpreter│ ──────────────────────────────┐
└─────────┬──────────┘                                │
          │ TaskSpec                                  │非法/歧义
          ▼                                           ▼
┌────────────────────┐                         澄清 / 拒绝 / 重试
│ Task Orchestrator  │
│ 状态机 + 技能调用   │
└───┬──────────┬─────┘
    │          │
    │          └──────────────┐
    ▼                         ▼
┌──────────────────┐    ┌───────────────────────┐
│ Semantic Perception│   │ Map & Localization    │
│ YOLO + 颜色 + RGB-D│   │ 位姿 + 占据/代价地图   │
│ 跟踪 + 目标解析     │   └──────────┬────────────┘
└─────────┬────────┘              │
          └──────────┬────────────┘
                     ▼
          ┌──────────────────────┐
          │ Goal Pose Selector   │
          │ 目标周围候选停车位    │
          └──────────┬───────────┘
                     ▼
          ┌──────────────────────┐
          │ Global Planner (A*)  │
          └──────────┬───────────┘
                     ▼
          ┌──────────────────────┐
          │ Local Planner/Tracker│
          │ + Independent Safety │
          └──────────┬───────────┘
                     ▼
          ┌──────────────────────┐
          │ MobileRobotEnv       │
          │ Genesis + Sensors    │
          └──────────────────────┘
```

建议按不同频率运行：

| 回路 | 建议频率 | 说明 |
|---|---:|---|
| 物理仿真 | 50 Hz（沿用 `dt=0.02`） | Genesis 推进 |
| 安全检查/底层控制 | 20~50 Hz | 不依赖 LLM；新鲜 LiDAR 优先 |
| 局部规划/路径跟踪 | 10~20 Hz | 输出速度指令 |
| RGB-D 感知 | 5~10 Hz | 可异步，只保留最新帧 |
| 全局重规划 | 事件触发或 1~2 Hz | 目标更新、路径阻塞、偏航过大时触发 |
| LLM | 每个用户任务一次，必要时事件触发 | 绝不进入实时控制回路 |

## 5. LLM 输入输出与任务协议

### 5.1 LLM 的正确职责

LLM 负责把开放式语言转换为有限、稳定的技能调用。例如：

```text
用户：行驶到黄色小车附近
                  ↓
LLM：navigate_near_object(category="car", color="yellow", near_distance_m=0.8)
```

LLM 不负责：

- 猜测地图坐标；
- 输出 A* 路径节点；
- 根据未经压缩的 LiDAR 数组逐帧判断转向；
- 输出任意 Python 代码；
- 绕过速度、安全距离和禁行区限制。

### 5.2 结构化 `TaskSpec`

建议使用 JSON Schema + Pydantic 双重校验：

```json
{
  "schema_version": "1.0",
  "intent": "navigate_near_object",
  "target": {
    "category": "car",
    "attributes": {
      "color": "yellow"
    },
    "spatial_qualifier": null
  },
  "constraints": {
    "near_distance_m": 0.8,
    "face_target": true,
    "max_speed_mps": 0.5
  },
  "on_ambiguity": "ask_user"
}
```

字段约束：

- `intent` 首期只允许 `navigate_near_object`、`stop`、`cancel`、`status`；
- `category` 和颜色先映射到受控词表，如“小车/汽车/车辆”统一为 `car`；
- 系统提示和场景词表应明确“本车/机器人”代表 ego robot，“黄色小车”默认查询场景目标 `role=landmark`，不得把受控机器人自身作为候选；
- 数值必须有限并限制到配置的安全范围；
- 未提及的距离使用系统默认值，不能由模型随意创造极端值；
- 输出中出现未知字段、未知动作、代码或自然语言尾巴时视为无效；
- 多目标但缺少区分属性时进入澄清，不默认随意选择；
- `stop/cancel` 应由本地规则直接处理，不必等待 LLM API。

### 5.3 提供给 LLM 的上下文

最小上下文包括：

- 当前允许的技能与参数 Schema；
- 支持的目标类别、颜色和关系词；
- 默认“附近”距离及速度限制；
- 可选的精简语义对象摘要，例如“视野中有 1 辆黄色车、1 个红色箱子”；
- 当前任务状态和上次失败的结构化原因。

不要向 LLM 发送：API 密钥、完整系统日志、无边界长度的传感器数组、每一帧图像或用户可注入的内部指令。

### 5.4 LLM 客户端要求

- 用统一 `TaskInterpreter` 协议隔离具体厂商 SDK；
- API Key 只从环境变量或密钥管理读取，不写入代码、配置示例或日志；
- 设置连接超时、总超时、有限重试和指数退避；
- 记录 request ID、模型名、耗时、token 用量和解析结果，日志中脱敏；
- 同一任务可缓存规范化结果，便于离线回归；
- API 失败时车辆保持静止，并向用户返回明确原因；
- 测试中使用 `FakeTaskInterpreter`，不依赖真实网络和计费 API。

### 5.5 LLM 与小车之间的对接

```python
class TaskInterpreter(Protocol):
    def parse(self, user_text: str, context: TaskContext) -> TaskSpec: ...

class RobotSkillExecutor(Protocol):
    def navigate_near_object(self, target: TargetQuery, constraints: NavConstraints) -> TaskHandle: ...
    def stop(self) -> None: ...
```

`TaskSpec` 进入任务编排器后，由本地代码执行。小车向上层返回的是状态事件，而不是把所有传感器原样塞回 LLM：

```text
TARGET_NOT_VISIBLE
TARGET_ACQUIRED
PATH_PLANNED
PATH_BLOCKED
PERCEPTION_STALE
ARRIVED
COLLISION_RISK
FAILED(reason=...)
```

## 6. “黄色小车”如何被识别和定位

### 6.1 YOLO 是否适合

结论：**适合做固定类别目标检测，但不是完整答案。**

对于本项目，建议拆成四步：

```text
YOLO / YOLO-Seg      → car 类别和二维区域
HSV/Lab ROI 分析     → yellow 属性和置信度
对齐深度 / LiDAR      → 目标距离及相机坐标
机器人位姿与外参       → 世界坐标
```

注意事项：

- COCO 预训练模型虽然包含 `car`，但不保证能识别 Genesis 中的简化 URDF、Box 拼成的车或低纹理合成图像；
- 如果黄色静态车使用有真实车外形的 mesh/URDF，可先用 COCO 权重做可行性测试；
- 如果召回率不足，使用 Genesis 合成数据训练项目专用 `car` 检测/分割模型；
- 颜色是属性，不建议把 `yellow_car/red_car/blue_car` 全部训练成独立类别；
- 如果未来必须理解大量未训练过的语言目标，再评估 Grounding DINO、OWL-ViT 等开放词汇检测器或视觉语言模型，但它们不是当前固定演示的首选依赖。

### 6.2 双适配器策略

实现统一 `SemanticPerceptor` 接口，至少提供两种后端：

1. `GenesisGroundTruthPerceptor`
   - 从场景实体注册表、实体位姿和 entity-level segmentation 获取确定结果；
   - 用于快速联调任务编排、目标位姿生成和导航；
   - 用于自动生成检测框/分割掩码和评估 YOLO；
   - 运行日志必须标注 `perception_mode=ground_truth`。
2. `RgbdYoloPerceptor`
   - 只从车载 RGB、对齐深度、机器人估计位姿和模型输出得到目标；
   - 用于最终“识别—定位—导航”闭环验收；
   - 不读取目标实体真实坐标；
   - 运行日志必须标注 `perception_mode=rgbd_yolo`。

这样既能快速定位导航问题，又不会把仿真真值伪装成视觉能力。

### 6.3 场景语义注册

新增目标实体时，场景层应保存内部元数据：

```python
SceneObject(
    object_id="car_yellow_01",
    category="car",
    attributes={"color": "yellow", "motion": "static", "role": "landmark"},
    entity_idx=...,
    footprint=...,
)
```

该注册表服务于真值后端、标注、可视化和评估。纯视觉后端只能用它计算离线指标，不能在控制期间读取其目标位姿。

受控机器人也必须注册为 `role=ego`。目标解析器默认排除 ego 的实体 ID、分割 ID 和轨迹 ID，避免“小车”被解析为自己；若以后支持“让蓝色小车跟随我”一类任务，再显式扩展角色语义。

### 6.4 RGB 与深度对齐

推荐将车载 RGB 和深度改成同一成像模型：

- 优先从同一个可渲染 Camera 同时获取 RGB、Depth，保证分辨率、FOV、内参和光轴一致；或
- 明确定义 RGB 与 Depth 的内参、畸变和外参，通过反投影/重投影对齐。

严禁简单地把 `256×192` RGB 检测框坐标除以 2 后直接索引 `128×96` 深度，除非已通过测试证明相机模型、裁剪和安装位姿完全一致。

视觉采集还应与界面渲染解耦：`sensor_rgb_enabled`/`vision_every` 决定算法是否采帧，`show_robot_view` 只决定是否显示窗口，`save_images` 只决定是否落盘。即使无 GUI、无图片保存，启用视觉导航时也必须按设定频率产生 RGB-D 帧。Genesis Camera 的渲染调用保留在仿真线程；异步工作线程只接收已经复制/冻结的 `FramePacket`，避免跨线程操作渲染器。

对每个检测目标：

1. 在检测框中心区域或分割掩码内采样深度；
2. 去除零值、非有限值和前后景离群点；
3. 使用中位数/较低分位数估计物体表面距离；
4. 用相机内参反投影到 `camera` 坐标；
5. 用 `T_world_base × T_base_camera` 转为世界坐标；
6. 与 LiDAR 投影或最近回波做容差检查；
7. 保存深度样本数、方差和最终定位置信度。

### 6.5 颜色判断

在 YOLO 框或分割掩码内：

- 去掉边界、轮胎、玻璃和高光区域；
- 转 HSV 或 Lab，按场景光照标定黄色范围；
- 计算黄色有效像素比例和颜色置信度；
- 多帧投票，避免单帧阴影导致颜色跳变；
- 颜色置信度不足时输出 `unknown`，不得强制归为黄色。

如果使用 YOLO-Seg，优先在 mask 内统计颜色；普通 bbox 中背景占比大，远距离时更容易误判。

### 6.6 多帧跟踪与目标解析

单帧检测不应立即成为导航目标。建议建立轻量目标轨迹：

- 以类别、颜色、3D 距离和 IoU/空间邻近关联检测；
- 连续 `N` 帧命中后才确认，例如 3/5 帧；
- 使用滑动平均或卡尔曼滤波平滑世界坐标；
- 每个轨迹携带 `last_seen_time`、`observation_count` 和置信度；
- 超时后标记 stale，不能继续无限期使用旧位置；
- 静态目标短时遮挡可保留，长时不可见时减速并重新观测。

`TargetResolver` 对 `TargetQuery(category="car", color="yellow")` 的选择顺序应固定：

1. 排除 `role=ego`、类别不符、属性置信度不足和已经过期的轨迹；
2. 类别与颜色均超过阈值才进入候选集，不以“最接近黄色”强行补足；
3. 用户给出“左边/最近/编号”等限定时，使用机器人接收命令时的参考坐标系进行过滤，并把该参考时刻写入任务；
4. 只有一个候选时锁定其 `track_id`；
5. 多个同等候选且用户未限定时返回 `AMBIGUOUS_TARGET` 并列出简短可区分信息；
6. 锁定后不因另一目标短暂置信度更高而切换，除非原轨迹失效并完成重新解析。

### 6.7 目标不在初始视野时

任务状态机必须支持 `SEARCH_TARGET`：

1. 原地分段旋转，覆盖约 360°，每段等待相机产生新帧；
2. 一旦稳定识别目标，立即停止搜索并锁定轨迹；
3. 旋转一周仍未发现时，在已知地图中前往预定义观察点或执行有限前沿搜索；
4. 达到搜索时间/距离预算仍无目标，则安全停止并反馈 `TARGET_NOT_FOUND`；
5. 搜索过程始终受 LiDAR 安全层约束。

## 7. 地图、规划和控制

### 7.1 定位与地图的分阶段方案

MVP 推荐先使用：

- 机器人定位：Genesis 机器人位姿（明确标注 `localization_mode=ground_truth`）；
- 地图：从房间边界和静态障碍物 footprint 构建二维占据栅格；
- 在线障碍物：LiDAR 更新局部代价地图。

第二阶段再替换为：

- 轮里程计 + IMU + LiDAR 定位/SLAM；
- 地图坐标系、里程计坐标系和 `base_link` 的显式变换；
- 定位协方差和失效检测。

不应把“视觉目标识别”“路径规划”“URDF 轮胎里程计校准”“完整 SLAM”四个高风险问题同时作为第一条闭环的前置条件。

### 7.2 代价地图

建议二维栅格分辨率 `0.05 m/cell`，包含：

- 房间墙体和静态障碍物占据层；
- LiDAR 实时障碍层；
- 机器人 footprint 膨胀层；
- 未知区域策略；
- 目标物体 footprint，防止规划穿过黄色小车本体；
- 边界安全余量。

膨胀半径至少为机器人外接半径加安全余量。地图更新、机器人位姿和传感器观测必须携带统一时间戳。

### 7.3 目标附近的停车位生成

视觉定位得到的是目标物体位置，不是机器人最终位姿。`GoalPoseSelector` 应：

1. 在目标 footprint 外围按期望间距采样一圈候选位姿；
2. 丢弃占据、未知、膨胀区内或越界候选；
3. 检查候选点到目标的可见性和最终朝向；
4. 对剩余候选分别计算路径代价；
5. 选择距离当前机器人较近、路径通畅、与目标保持安全距离的候选；
6. 所有候选都不可达时返回 `NO_SAFE_GOAL_POSE`，不强行靠近。

### 7.4 全局规划

MVP 使用 8 邻域 A*：

- 输入：当前栅格位姿、候选目标位姿、膨胀代价地图；
- 输出：世界坐标路径；
- 对路径做视线简化或样条平滑，但平滑后必须再次碰撞检查；
- 目标移动、路径被新障碍阻断或偏离路径超过阈值时重规划；
- 无路径时依次尝试其他候选停车位，全部失败才报错。

A* 足以应对当前静态房间。若后续地图规模扩大，再评估 D* Lite/Hybrid A*；当前差速车可原地旋转，不必第一版就引入复杂车辆曲率规划。

### 7.5 局部规划和路径跟踪

建议实施顺序：

1. 先使用 Pure Pursuit/航向比例控制跟踪 A* 路径；
2. LiDAR 安全层负责紧急停车和受限绕障；
3. 再引入 DWA 类局部轨迹采样，综合路径偏差、目标距离、速度和障碍间距打分；
4. 局部规划持续失败时通知全局层重规划，而不是原地无限振荡。

输出仍使用现有接口：

```python
action = {
    "linear_velocity": v_mps,
    "angular_velocity": w_radps,
}
```

### 7.6 独立安全层

安全层优先级高于 LLM、任务状态机和规划器：

- LiDAR 最小安全距离触发减速/停车；
- 根据当前速度设置动态制动距离；
- 传感器结果过期、非有限值过多、定位跳变时停车；
- 连续碰撞风险、规划震荡、控制超时进入故障状态；
- `Ctrl+C`、用户“停止”、窗口关闭均立即输出零速；
- 程序异常和退出路径必须执行 `car.stop()`/零动作；
- 安全停车后只有风险解除并满足恢复条件才能继续。

## 8. 任务编排状态机

```text
IDLE
  │ 用户命令
  ▼
PARSING ──无效/歧义──► WAITING_FOR_CLARIFICATION
  │ 合法 TaskSpec
  ▼
RESOLVING_TARGET
  ├──目标可见──► SELECTING_GOAL
  └──目标不可见► SEARCHING_TARGET
                     ├──找到──► SELECTING_GOAL
                     └──超时──► FAILED
SELECTING_GOAL
  ├──无安全停车位──► FAILED
  ▼
PLANNING
  ├──无路径/换候选/超预算──► FAILED
  ▼
EXECUTING
  ├──路径阻塞──► REPLANNING ──► EXECUTING
  ├──目标过期──► REACQUIRING ──► SELECTING_GOAL
  ├──危险──► SAFETY_STOP
  ├──取消──► CANCELLED
  └──满足到达条件──► ARRIVED
```

每个状态必须具有：进入时间、超时、最大重试次数、可取消处理和明确失败码。建议失败码至少包括：

```text
LLM_UNAVAILABLE
INVALID_TASK_SPEC
AMBIGUOUS_TARGET
TARGET_NOT_FOUND
PERCEPTION_STALE
LOCALIZATION_UNRELIABLE
NO_SAFE_GOAL_POSE
PATH_NOT_FOUND
PATH_BLOCKED
CONTROL_TIMEOUT
COLLISION_RISK
USER_CANCELLED
```

## 9. 建议代码组织

在保留现有基线脚本的前提下新增独立语义导航入口：

```text
examples/mobile_robot/
├── semantic_navigation.py           # 最终 CLI/演示入口
├── config/
│   └── semantic_navigation.yaml     # 速度、阈值、模型、地图配置（无密钥）
├── llm/
│   ├── types.py                     # TaskSpec / TargetQuery / Schema
│   ├── interpreter.py               # TaskInterpreter 协议
│   ├── api_interpreter.py           # 具体 LLM API 适配
│   └── fake_interpreter.py          # 离线测试
├── task/
│   ├── orchestrator.py              # 状态机
│   └── events.py                    # 状态与失败码
├── vision/                          # 复用现有目录
│   ├── types.py                     # 已有协议，按需扩展
│   ├── yolo_detector.py             # 已有适配器，接入实时推理
│   ├── attributes.py                # 颜色属性
│   ├── rgbd_fusion.py               # 深度和坐标变换
│   ├── tracker.py                   # 多帧目标轨迹
│   ├── semantic_perceptor.py        # 统一感知接口
│   └── ground_truth.py              # Genesis 真值后端
├── mapping/
│   ├── occupancy_grid.py
│   └── costmap.py
├── navigation/
│   ├── goal_selector.py
│   ├── global_planner.py            # A*
│   ├── local_planner.py             # 路径跟踪 / DWA
│   ├── controller.py
│   └── safety.py
├── scenarios/
│   └── semantic_target_scene.py     # 黄色静态车和障碍物场景
└── tests/
    ├── test_task_schema.py
    ├── test_target_resolver.py
    ├── test_rgbd_fusion.py
    ├── test_goal_selector.py
    ├── test_global_planner.py
    ├── test_safety.py
    └── test_semantic_navigation_e2e.py
```

如果项目暂不希望引入 YAML 依赖，可先使用 dataclass 配置；密钥仍只允许来自环境变量。

## 10. 核心数据接口

### 10.1 语义对象

```python
@dataclass(frozen=True)
class SemanticObject:
    track_id: str
    category: str
    attributes: dict[str, str]
    confidence: float
    position_world: tuple[float, float, float] | None
    footprint_radius_m: float | None
    last_seen_sim_time: float
    source: str  # rgbd_yolo / ground_truth
```

### 10.2 路径请求与结果

```python
@dataclass(frozen=True)
class PlanRequest:
    start_pose: tuple[float, float, float]
    goal_pose: tuple[float, float, float]
    map_version: int

@dataclass(frozen=True)
class PathPlan:
    plan_id: str
    world_points: tuple[tuple[float, float], ...]
    total_cost: float
    map_version: int
```

### 10.3 同步和新鲜度

所有观测和派生结果至少携带：

- `frame_id`；
- `sim_time`；
- 数据源；
- 处理耗时；
- 是否过期；
- 使用的机器人位姿时间。

异步视觉使用 `queue(maxsize=1)` 或等价 latest-only 缓冲，禁止积压旧帧。控制层只消费满足最大年龄阈值的结果。

## 11. 实施阶段与 TODO

以下顺序用于控制集成风险。带 `[P0]` 的项目是最终演示必需项。

### Phase 0：冻结演示口径和基线

- [ ] `[P0]` 确定最终演示使用运动学小车还是四轮 URDF；建议先用运动学小车完成闭环，URDF 作为最终替换/增强。
- [ ] `[P0]` 确定 LLM API 厂商、模型、结构化输出/工具调用方式、网络条件和预算。
- [ ] `[P0]` 确定“黄色小车”资产；优先真实车外形 mesh/URDF，而不是黄色 Box。
- [ ] `[P0]` 确定纯视觉验收是否禁止读取目标真值坐标；本文默认禁止。
- [ ] 固化当前基线运行命令、随机种子、Python/Genesis/Ultralytics 版本和输出样例。
- [ ] 为现有 `environment.py` 和 URDF 脚本各跑一次 smoke test，记录当前成功率、耗时与已知问题。

**验收物：** 评审后的场景截图、依赖清单、演示口径和基线报告。

### Phase 1：语义场景与真值导航闭环

- [ ] `[P0]` 新增黄色静态小车、至少一个干扰目标和多个障碍物。
- [ ] `[P0]` 建立 `SceneObject` 语义注册表和目标 footprint。
- [ ] `[P0]` 构建二维占据栅格和膨胀代价地图。
- [ ] `[P0]` 实现目标周围候选停车位生成。
- [ ] `[P0]` 实现 A*、路径简化和碰撞复查。
- [ ] `[P0]` 实现路径跟踪控制，并保留 LiDAR 安全覆盖。
- [ ] 实现路径、目标、footprint 和代价地图的俯视可视化。
- [ ] 使用真值目标坐标完成“动态目标点—规划—到达”端到端测试。

**验收物：** 不调用 LLM/YOLO 时，传入结构化黄色车查询即可在多个布局中无碰到达附近；日志标明 `perception_mode=ground_truth`。

### Phase 2：LLM 任务理解

- [ ] `[P0]` 定义并实现 `TaskSpec`、JSON Schema 和 Pydantic 校验。
- [ ] `[P0]` 实现 `TaskInterpreter` 与 Fake 后端。
- [ ] `[P0]` 接入指定 LLM API 的结构化输出/工具调用。
- [ ] `[P0]` 实现超时、重试、错误映射、脱敏日志和静止降级。
- [ ] 建立中文指令集，覆盖同义词、省略表达、距离修饰、停止、歧义和越权命令。
- [ ] 对同一测试集固定模型与采样参数做离线解析回归。

**验收物：** “行驶到黄色小车附近”等约定表达稳定生成合法 TaskSpec；无网络、超时和非法 JSON 时车辆不动且有明确提示。

### Phase 3：视觉识别与三维定位

- [ ] `[P0]` 统一/标定车载 RGB-D 相机，保存内参和 `T_base_camera`。
- [ ] `[P0]` 将算法采帧、GUI 显示和图片保存拆成独立开关，保证 headless 视觉导航仍有 RGB-D 输入。
- [ ] 使用 Genesis entity segmentation 自动生成合成训练/验证标注。
- [ ] 先测试本地 COCO YOLO 权重对目标资产的检测召回率。
- [ ] 若预训练模型不达标，随机化位置、距离、角度、光照、材质、遮挡和背景，训练自定义 car 模型。
- [ ] `[P0]` 将已有 `YoloDetector` 接入实时视觉循环。
- [ ] `[P0]` 实现黄色 ROI/mask 属性提取与置信度。
- [ ] `[P0]` 实现 RGB-D 反投影、世界坐标变换和 LiDAR 交叉检查。
- [ ] `[P0]` 实现多帧跟踪、结果年龄和目标稳定确认。
- [ ] `[P0]` 实现 ego 目标排除和确定性的 `TargetResolver` 多目标规则。
- [ ] 输出原图、标注图、深度、检测 JSONL 和真值误差统计。
- [ ] 实现 `RgbdYoloPerceptor` 与 `GenesisGroundTruthPerceptor` 可配置切换。
- [ ] 增加“真值泄漏”保护测试：纯视觉执行上下文不暴露目标 entity pose，评估器只在动作生成后旁路读取真值计算指标。

**验收物：** 静止与低速运动时能稳定输出黄色小车的类别、颜色、距离和世界位置；纯视觉后端不读取目标真值坐标。

### Phase 4：任务状态机与搜索闭环

- [ ] `[P0]` 实现 `IDLE→PARSING→RESOLVING→PLANNING→EXECUTING→ARRIVED` 主状态链。
- [ ] `[P0]` 实现目标不在视野时的 360° 分段搜索。
- [ ] `[P0]` 实现多目标歧义、无目标、感知过期、无停车位和无路径处理。
- [ ] `[P0]` 实现目标重定位、路径阻塞与有限次数重规划。
- [ ] `[P0]` 实现用户取消、紧急停车和程序异常零速收尾。
- [ ] 让终端持续展示当前状态、目标置信度、距离、规划进度和失败原因。

**验收物：** 用户只输入自然语言即可完成或得到可解释失败，不需要手工填写坐标和航点。

### Phase 5：最终演示入口与可视化

- [ ] `[P0]` 新增 `semantic_navigation.py`，统一加载场景、LLM、感知、规划和控制模块。
- [ ] `[P0]` 支持终端输入；Web/桌面 GUI 作为 P1。
- [ ] `[P0]` 第一视角叠加检测框、类别、颜色、距离、跟踪 ID 和结果年龄。
- [ ] `[P0]` 俯视视图叠加机器人位姿、目标位置、候选停车位、全局路径和当前局部目标。
- [ ] `[P0]` 生成 `summary.json`、`events.jsonl`、`telemetry.csv`、LLM 记录、视觉记录和图像/视频。
- [ ] 提供 `--llm-provider`、`--perception-mode`、`--vision-model`、`--scenario`、`--seed` 等 CLI 参数。
- [ ] 更新 `README.md`，给出离线测试、API 配置和最终演示命令。

**预期命令形态：**

```bash
export ROBOT_LLM_API_KEY="..."
python examples/mobile_robot/semantic_navigation.py \
  --scenario semantic_yellow_car \
  --llm-provider configured_provider \
  --perception-mode rgbd_yolo \
  --vision-model models/mobile_robot/car_detector.pt \
  --vis --robot-view --save-run
```

程序启动后提示：

```text
请输入任务> 行驶到黄色小车附近
```

### Phase 6：URDF、性能与工程化增强

- [ ] 校准四轮滑移转向模型、摩擦、轮速控制器和轮里程计。
- [ ] 将已验证的上层语义导航切换到 URDF 环境。
- [ ] 将视觉推理改为 latest-only 异步线程/进程，评估 GPU/CPU 性能。
- [ ] 加入 DWA 或等价局部规划器，处理新增临时障碍。
- [ ] 注入图像延迟、丢帧、LiDAR 噪声、里程计漂移和 LLM 故障。
- [ ] 为未来 ROS2/真实车实现传输适配层，保持 `TaskSpec/SemanticObject/Action` 语义不变。

**验收物：** 切换车辆或推理部署方式时，上层任务接口和测试用例无需重写。

## 12. 测试矩阵

### 12.1 单元测试

- 自然语言 Schema 校验、默认值、非法值和枚举拒绝；
- 颜色阈值和低置信度 `unknown`；
- 深度异常值、bbox 越界、坐标系变换；
- 目标关联、轨迹过期和多帧确认；
- 栅格坐标与世界坐标互转；
- footprint 膨胀、停车位生成和碰撞检查；
- A* 可达、不可达、起终点占据、窄通道；
- 安全距离、动态制动距离和 stale 观测停车。

### 12.2 集成场景

| 编号 | 场景 | 预期结果 |
|---|---|---|
| S01 | 黄色车初始可见、无遮挡 | 直接识别、规划、到达 |
| S02 | 黄色车在车后方 | 旋转搜索后到达 |
| S03 | 黄色车前有静态障碍 | A* 绕行后到达 |
| S04 | 两台车，仅一台黄色 | 正确选择黄色目标 |
| S05 | 两台黄色车，无额外限定 | 停车并请求澄清，不随机选择 |
| S06 | 场景没有黄色车 | 搜索超时并报告 `TARGET_NOT_FOUND` |
| S07 | 目标被部分遮挡 | 多帧确认/重获，不使用单帧误检 |
| S08 | 目标附近没有安全停车位 | 报告 `NO_SAFE_GOAL_POSE` |
| S09 | 路径执行中出现新障碍 | 安全停车并重规划 |
| S10 | YOLO 结果延迟或停止更新 | 感知过期，减速/停车 |
| S11 | LLM API 超时/返回非法结构 | 车辆保持静止并报告错误 |
| S12 | 用户执行中输入“停止” | 不等待 LLM，立即零速 |
| S13 | 相同颜色但类别不是 car | 不得误选 |
| S14 | 不同种子、光照、距离和视角 | 达到统计成功率目标 |
| S15 | 受控机器人自身也是 car 类别 | 目标解析必须排除 ego |
| S16 | headless 且不保存图片 | 视觉推理仍按频率获得 RGB-D 帧 |

### 12.3 回归模式

每个端到端场景都应支持：

- `FakeTaskInterpreter + GroundTruthPerceptor`：只测规划控制；
- `Real/Fake LLM + GroundTruthPerceptor`：测语言到导航；
- `FakeTaskInterpreter + RgbdYoloPerceptor`：测感知到导航；
- `Real LLM + RgbdYoloPerceptor`：最终完整闭环。

分层回归能快速判断失败属于 LLM、视觉、定位、规划还是动力学。

## 13. 量化验收标准

### 13.1 P0 功能验收

- 用户无需输入坐标或航点，只输入约定自然语言即可启动任务；
- LLM 输出 100% 经过 Schema 校验后才能执行；
- 目标类别、颜色和空间位置均有明确数据来源和置信度；
- 纯视觉模式中导航器不读取目标真值坐标；
- 受控机器人不会被语义目标解析器选为“黄色小车”；
- 路径不是针对黄色车位置预先硬编码；
- 到达后满足配置的安全间距、朝向和停稳条件；
- 任何异常退出、安全风险或用户停止都产生零速度；
- 一次运行可通过日志关联用户输入、LLM 结果、视觉帧、目标轨迹、路径、动作和最终状态。

### 13.2 建议指标

在约定的固定测试集与至少 20 个随机种子上：

| 指标 | MVP 建议门槛 |
|---|---:|
| 任务解析正确率 | `>= 95%`（约定中文指令集） |
| 黄色车检测召回率 | `>= 95%`（项目验证集） |
| 黄色属性准确率 | `>= 95%`（项目验证集） |
| 目标二维定位中位误差 | `<= 0.20 m` |
| 端到端到达成功率 | `>= 90%` |
| 碰撞率 | `0%`（验收测试集） |
| 最终安全间距合格率 | `>= 95%` |
| stale 视觉结果参与控制 | `0 次` |
| LLM API 调用次数 | 正常单任务 `1 次`，不进入控制循环 |

指标阈值应在 Phase 0 评审时根据目标资产大小、房间尺度和计算平台冻结。

## 14. 可观测性与输出文件

建议每次运行创建独立目录：

```text
out/mobile_robot_semantic/<run_id>/
├── config_resolved.json
├── summary.json
├── events.jsonl
├── telemetry.csv
├── llm_requests.jsonl          # 脱敏
├── vision_results.jsonl
├── plans.jsonl
├── rgb_robot/
├── rgb_robot_annotated/
├── depth/
├── map/
└── demo.mp4
```

`summary.json` 至少包含：任务原文、TaskSpec、模型和权重版本、随机种子、感知/定位模式、目标 ID、最终距离、到达状态、失败码、碰撞、耗时、重规划次数和各模块延迟分位数。

## 15. 主要风险与规避措施

| 风险 | 后果 | 规避措施 |
|---|---|---|
| 把 LLM 当低层控制器 | 高延迟、随机输出、不可验证 | LLM 只产生受约束技能调用 |
| 把 YOLO 当完整定位系统 | 有框但无距离/坐标 | 颜色后处理 + RGB-D + 坐标变换 + 跟踪 |
| 预训练 YOLO 不识别合成车 | 目标始终找不到 | 先换真实外形资产，再用仿真标注微调 |
| RGB/Depth 未对齐 | 世界坐标明显错误 | 同一 RGB-D 相机或严格标定重投影 |
| 用真值坐标冒充视觉定位 | 演示看似成功但技术链不成立 | 双模式、日志标识、纯视觉验收禁止读取真值 |
| 目标不在视野 | 一直等待或错误失败 | 360° 搜索 + 有限观察点探索 |
| 直接规划到目标中心 | 撞上黄色小车 | footprint 外候选停车位 |
| 固定航点无法适应目标变化 | 换位置就失败 | 代价地图 + A* 动态生成路径 |
| 视觉推理阻塞仿真 | 控制抖动/结果陈旧 | 降频、异步 latest-only、年龄检查 |
| URDF 滑移和里程计漂移 | 到达误差、路径跟踪失败 | 先在运动学环境验证，单独校准 URDF |
| 多个黄色目标 | 选错对象 | 目标排序规则或请求用户澄清 |
| API/模型版本漂移 | 回归不稳定 | 固定模型、参数、Schema 和回放测试 |
| 密钥或图像泄露 | 安全与合规风险 | 环境变量、脱敏日志、明确数据出域策略 |

## 16. 需要产品/领导确认的决策

这些问题不阻止按默认方案开始，但应在 Phase 0 冻结：

1. **最终演示是否必须证明视觉识别？** 默认：必须，最终使用 `rgbd_yolo`，真值仅做调试和指标计算。
2. **最终车辆模型选哪个？** 默认：先用稳定运动学环境完成功能闭环，功能稳定后切换 URDF；如果只看业务效果，可先以运动学版本交付。
3. **黄色静态车资产是什么？** 默认：使用具有明确车辆外形的静态 mesh/URDF，并单独记录 footprint；不使用普通黄色方块替代。
4. **指定哪个 LLM API/模型？** 需要给出兼容协议、模型名、网络环境、密钥注入方式和预算；代码保持厂商无关。
5. **是否允许图像上传云端？** 默认：不上传，YOLO 在本地推理，LLM 只接收文本和精简语义摘要。
6. **交互形式？** 默认：终端输入为 P0；Web/桌面界面为 P1。
7. **目标不唯一时如何处理？** 默认：安全停车并询问，不自动选择最近目标。
8. **“附近”的业务距离？** 默认：车体边缘到目标边缘约 `0.8 m`，配置范围内可由语言覆盖。
9. **演示是否要求目标一开始不可见？** 默认：测试集中包含不可见场景，以证明搜索链路。
10. **是否要求未来上真实车？** 若要求，应提前固定坐标系、消息协议和 ROS2 适配边界，但不把 ROS2 作为当前 MVP 前置条件。

## 17. 推荐交付顺序

最稳妥的里程碑顺序是：

1. `TaskSpec + 真值目标 + A* + 安全控制`，证明语言任务之外的导航骨架正确；
2. 接入 LLM，证明自然语言能可靠调用同一导航技能；
3. 接入 RGB-D YOLO 替换真值目标，证明目标由视觉识别和定位；
4. 加入目标搜索、歧义处理、重规划和故障降级；
5. 最后切换/校准 URDF 动力学小车并完善界面。

该顺序确保每个阶段都有独立可演示、可回归的结果，也能在最终效果不符合预期时准确定位到语言、感知、规划、控制或动力学中的具体一层。

## 18. Definition of Done

只有同时满足以下条件，才能认为领导描述的序号 1 已实现：

- 用户自然语言经过真实 LLM API 生成合法的结构化任务；
- 黄色静态小车由车载感知链路识别、颜色确认并定位，或运行记录明确声明使用真值调试模式；
- 小车目标位姿由目标当前位置和安全间距在线计算，不是固定坐标；
- 路径由当前地图在线规划，不是固定航点；
- 小车以闭环方式执行路径，具备局部避障、重规划和独立安全停车；
- 目标不存在、存在多个、路径不可达、模型/API 失败时均有安全且可解释的行为；
- 最终能看到第一视角识别结果、俯视规划路径和任务状态；
- 自动化测试和多种子回归达到冻结后的量化指标；
- README、配置、模型说明、运行命令和一次完整演示产物齐全。

---

相关现有文档：

- [README.md](README.md)：当前移动机器人示例运行方式；
- [STRATEGY.md](STRATEGY.md)：现有固定航点和 LiDAR 规则控制器；
- [_docs/SIMULATION_ANALYSIS_AND_PLAN.md](_docs/SIMULATION_ANALYSIS_AND_PLAN.md)：仿真闭环基础规划；
- [_docs/YOLO_VISION_INTEGRATION_DESIGN_AND_PLAN.md](_docs/YOLO_VISION_INTEGRATION_DESIGN_AND_PLAN.md)：现有视觉接口与 YOLO 分阶段接入设计。
