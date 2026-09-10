# Mobile robot MVP

这是基于 Genesis 的第一个移动机器人闭环实验，目标是先验证“房间场景 → 传感器 → 控制器 → 小车运动 → 日志/图像”的完整链路。

当前 MVP 包含：

- 参数化室内房间、墙体和障碍物；
- 简化差速小车几何体和运动学模型；
- 左右轮速度对应的差速运动学控制；
- LiDAR、IMU、里程计；
- RGB 相机和深度相机；
- 基于 waypoint、里程计和 LiDAR 前方安全层的规则导航；
- CSV telemetry、JSON summary 和可选 RGB/深度帧保存。

RGB 相机使用稳定的房间总览视角，便于观察小车、障碍物和目标点；深度相机保持车载前向视角，作为算法传感器输出。

注意：第一版为了稳定验证闭环，车体和车轮是固定 Genesis 几何体，由差速运动学推进；`assets/diff_drive_car.urdf` 是后续动态实验使用的四轮滑移转向模型。

## 运行

在项目根目录执行：

```powershell
conda activate genesis
python examples/mobile_robot/room_navigation.py --steps 300
```

如果 macOS 上激活项目 `.venv` 后出现 `ModuleNotFoundError: No module named 'numpy'`，说明虚拟环境只有 Python/部分基础包，尚未装入项目依赖。可使用本机 uv 缓存离线补全，不会重新下载：

```bash
uv pip install --offline --python .venv/bin/python -e . 'mujoco==3.10.0'
```

补全后重新执行 `source .venv/bin/activate`，再运行上面的案例命令即可。

使用 CUDA：

```powershell
python examples/mobile_robot/room_navigation.py --gpu --steps 1200 --save-images
```

CPU 完整导航回归：

```powershell
python examples/mobile_robot/room_navigation.py `
  --steps 1200 --save-images --image-every 300 `
  --output-dir out/mobile_robot_cpu_mvp
```

打开 Genesis Viewer，并同时打开 RGB 相机窗口：

```powershell
python examples/mobile_robot/room_navigation.py --gpu --vis --steps 1200 --save-images
```

输出默认位于 `out/mobile_robot/`：

```text
out/mobile_robot/
├── summary.json
├── telemetry.csv
├── rgb/       # 使用 --save-images 时生成
└── depth/     # 使用 --save-images 时生成
```

运行时会把 Quadrants 内核缓存放到当前 Conda 环境目录下的 `quadrants_cache/`，避免 Windows 上默认 `C:\quadrants_cache` 无写权限导致启动失败。也可以通过 `GENESIS_QUADRANTS_CACHE` 环境变量指定其他路径。

在固定种子和默认房间配置下，CPU 基线通常约 783 步到达目标，且不发生几何碰撞；GPU 后端可用 `--gpu` 做同样的短回归。

当前规则控制器是验证仿真闭环的基线，不是最终导航算法。后续应在保持 `reset`、观测、动作和日志接口稳定的前提下，增加批量环境和强化学习策略。

策略细节请参阅 [STRATEGY.md](STRATEGY.md)。

需求分析与实施计划请参阅 [_docs/SIMULATION_ANALYSIS_AND_PLAN.md](_docs/SIMULATION_ANALYSIS_AND_PLAN.md)。

## 可观测闭环实验副本

按照计划文档，当前新增了不修改原始基线的实验副本：

```bash
python examples/mobile_robot/room_navigation_observable.py \
  --vis --robot-view --save-sensors --scenario room_obstacle --steps 1200
```

该副本保留原有规则控制和运动学模型，并增加车载 RGB 第一视角、场景障碍物选项，以及同步的 RGB/深度帧、LiDAR、IMU、里程计、动作和位姿记录。使用 `--robot-view` 时，单独的 RGB 窗口显示车载第一视角；俯视图仍会保存到 `rgb/`，但不会覆盖第一视角窗口。原始 [room_navigation.py](room_navigation.py) 继续作为稳定回归基线。

`room_obstacle` 是展示场景：两个障碍物分布在蛇形航线两侧，小车会依次经过多个航点。`room_center_obstacle` 使用更直观的单向绕行路线：先向右前进，再向上绕过方块右侧，最后从前右方直达目标，避免绕过障碍后目标落到车后方。这两者都只是可解释的 MVP 实验，不等同于完整全局规划器。

展示场景的路线为：

```text
起点 → 左侧上行 → 中上方横行 → 中部下行 → 右侧横行 → 目标点
```

两个障碍物分别位于中左下方和中右上方，能够在俯视图中看出场景层次，也不会直接堵住小车的起步方向。

程序化算法接口位于 [environment.py](environment.py)：

```python
from examples.mobile_robot.environment import EnvironmentConfig, MobileRobotEnv

env = MobileRobotEnv(EnvironmentConfig(render=True))
observation = env.reset()
observation, reward, terminated, truncated, info = env.step(
    {"linear_velocity": 0.3, "angular_velocity": 0.0}
)
env.close()
```

其中 `observation` 包含机器人 RGB、深度、LiDAR、IMU、里程计和位姿；算法通过 `step(action)` 提供线速度和角速度。`render=False` 时仍会返回深度、LiDAR、IMU 和位姿，只是不渲染 RGB 数组。

如果要用副本中已有的规则控制器驱动环境，可使用同一个标准观测接口：

```python
observation = env.reset()
for _ in range(1200):
    action = env.rule_action(observation)
    observation, reward, terminated, truncated, info = env.step(action, render=False)
    if terminated or truncated:
        break
env.close()
```

`sensor_observations.jsonl` 中每条记录的 `observation` 是动作执行后的下一时刻观测，`action_step` 标明产生该观测的动作所在步；这样图像、传感器、位姿和动作不会被误认为来自同一物理时刻。CLI 运行时可用 `--save-sensors` 保存这些记录。

## 视觉控制展示：沿途识别物体

[room_navigation_vision.py](room_navigation_vision.py) 是当前视觉/控制主线入口：小车按简易 waypoint + LiDAR 策略驶向既定目的地，同时在车载 RGB 视角中持续产生检测结果。该入口不依赖 LLM；LLM 任务另见 [TASK_LLM.md](TASK_LLM.md)。完整基线需求和验收标准见 [TASK.md](TASK.md)；面向任意自然语言目标的后续主线已调整为开放词汇视觉指代，分阶段计划见 [TODO.md](TODO.md)。

使用确定性的 Genesis 真值感知后端联调运行时、叠加和日志：

```bash
.venv/bin/python examples/mobile_robot/room_navigation_vision.py \
  --scenario vision_route_showcase \
  --perception-mode ground_truth \
  --save-images --save-sensors --save-vision
```

`ground_truth` 仅用于开发联调，输出会明确标注 `perception_mode=ground_truth`。使用真实本地 YOLO 权重时：

```bash
.venv/bin/python examples/mobile_robot/room_navigation_vision.py \
  --scenario vision_route_showcase \
  --perception-mode yolo \
  --vision-model models/mobile_robot/training_full_v2/yolo11n_custom/weights/best.pt \
  --calibrated-depth \
  --save-vision --annotated-view
```

`--calibrated-depth` 使用 Genesis 运行时 RGB/Depth 内参和同挂载外参完成 bbox 重投影，输出距离置信度、相机/机器人/世界坐标，并保存 `camera_calibration.json`。程序不会联网下载权重；模型缺失或推理异常时会记录错误并降级，不替代 LiDAR 安全控制。输出默认位于 `out/mobile_robot_vision/`，包括 `vision_results.jsonl`、`tracked_objects.json`、标注图和 `summary.json`。

`--robot-view` 是 Genesis 原始车载 RGB 窗口，不包含 YOLO 框；`--annotated-view` 会另外打开一个 OpenCV 窗口，显示当前推理帧及类别、置信度、颜色、跟踪 ID 和距离标注。两个窗口可以同时开启。标注窗口只在新推理结果对应的帧上绘制框，不复用过期框，避免小车运动时框与目标错位。`--vision-every 25` 表示每 25 个仿真步刷新一次标注（默认 `dt=0.02` 时约 0.5 秒）；需要更连续的视觉刷新可降低为 `--vision-every 1`，代价是 CPU 推理开销增加。按 `q` 或 `Esc` 可关闭标注窗口，车辆控制仍会继续。

默认方案的自定义 YOLO 数据和权重位于 `datasets/mobile_robot_yolo_full_v2/` 与 `models/mobile_robot/training_full_v2/`。训练与独立 test：

```bash
.venv/bin/python -m examples.mobile_robot.vision.train_yolo \
  --data datasets/mobile_robot_yolo_full_v2/dataset.yaml \
  --output-dir models/mobile_robot/training_full_v2 \
  --epochs 20 --imgsz 256 --batch 16 --device cpu --workers 0
.venv/bin/python -m examples.mobile_robot.vision.evaluate_yolo \
  --model models/mobile_robot/training_full_v2/yolo11n_custom/weights/best.pt \
  --data datasets/mobile_robot_yolo_full_v2/dataset.yaml --split test \
  --imgsz 256 --batch 16 --device cpu \
  --output-dir models/mobile_robot/training_full_v2
```

对同一 seed 的 YOLO/ground-truth 运行日志，可用 `vision/evaluate_runtime.py` 做离线位置匹配、颜色/距离误差与延迟 P95 统计；真值不会进入运行时检测器。

### 开放词汇视觉候选评估（实验阶段）

任意自然语言目标不再通过增加闭集 YOLO 类别实现。当前提供本地 YOLO-World
候选器。OWLv2 暂不纳入当前验收路线；YOLO-World 只输出文本提示对应的候选框，不直接产生轮速、路径或导航目标。
模型必须提前保存在本地，运行时不会自动下载：

```bash
.venv/bin/python examples/mobile_robot/vision/evaluate_open_vocab.py \
  --backend yolo-world \
  --model models/mobile_robot/open_vocab/yolov8s-world.pt \
  --images out/mobile_robot_yolo_annotated_view/rgb_robot \
  --prompt "yellow car" --prompt "green pillar" --prompt platform \
  --prompt "red box" --prompt "blue cylinder" \
  --device auto --imgsz 640 --infer-conf 0.001 --decision-conf 0.05 \
  --annotate --output-dir out/open_vocab_benchmark
```

`--device auto` 遵循 MPS 优先、CPU 回退策略；当前机器若无可用 MPS 会在
`summary.json` 标记 `device_reason=auto_cpu`。评估器将每帧分为
`candidate`、`ambiguous`、`low_confidence` 和 `no_visual_match`，并写出
`candidates.jsonl` 和标注图。只有经过 OOD 数据集、多帧确认及 RGB-D 定位验收后，候选才允许进入规划。

也可以在 Genesis 车载相机上实时观察候选（不带 `--instruction` 时，此模式只记录/显示感知结果，仍使用原有 waypoint + LiDAR 控制）：

```bash
.venv/bin/python examples/mobile_robot/room_navigation_vision.py \
  --scenario vision_route_showcase --perception-mode open_vocab \
  --vision-model models/mobile_robot/open_vocab/yolov8s-world.pt \
  --vision-prompt "yellow car" --vision-prompt "green pillar" --vision-prompt platform \
  --open-vocab-device auto --open-vocab-infer-conf 0.001 \
  --open-vocab-decision-conf 0.05 --vision-imgsz 640 --vision-every 25 \
  --vis --robot-view --annotated-view --save-vision \
  --output-dir out/mobile_robot_open_vocab_live
```

`--robot-view` 仍是 Genesis 原始车载画面，`--annotated-view` 是独立的候选框窗口；
没有候选或深度无效时会在日志中显示对应状态，不会强行创建导航目标。

### 最小自然语言目标 Demo（视觉/控制侧）

为了先演示“指令 → 视觉目标 → RGB-D 坐标 → 目标附近航点”，可以直接输入一句中文指令：

```bash
.venv/bin/python examples/mobile_robot/room_navigation_vision.py \
  --scenario vision_route_showcase \
  --perception-mode open_vocab \
  --open-vocab-backend yolo-world \
  --vision-model models/mobile_robot/open_vocab/yolov8s-world.pt \
  --instruction "行驶到黄色小车附近" \
  --open-vocab-device auto \
  --open-vocab-infer-conf 0.001 \
  --open-vocab-decision-conf 0.05 \
  --target-lock-conf 0.001 \
  --vision-imgsz 640 --vision-every 25 \
  --vis --robot-view --annotated-view \
  --save-vision --output-dir out/mobile_robot_nl_demo
```

该 Demo 会把常见中文颜色/物体词转换成一个 YOLO-World prompt，连续确认目标后，使用已知
Genesis RGB-D 标定计算目标附近临时航点，并通过 `replace_waypoints()` 交给原有 waypoint + LiDAR
控制器。`summary.json` 中的 `navigation_mode` 会从 `searching` 变为 `target_locked`，
`active_waypoints` 会从搜索航点切换为单个目标航点，且 `command_target_x/y` 应与
`target_near_waypoint` 一致；这才表示车辆已经脱离固定搜索路线。
画面、`vision_results.jsonl` 和 `summary.json` 会保存在输出目录。它是视觉/控制侧的最小演示：
复杂空间关系、多语言改写和任意现实物体的完整语义解析仍由后续 LLM 适配器负责，LLM 不输出轮速或路径。

注意：`--open-vocab-decision-conf` 只控制日志中的 `candidate/low_confidence` 状态，
`--target-lock-conf` 才控制是否允许 RGB-D 候选接管导航。当前本地 YOLO-World 权重在 Genesis
小图上的原始分数通常约为 0.001～0.02，因此 Demo 默认锁定阈值为 0.001；正式验收仍需用
OOD 指标重新标定阈值。若搜索路线结束仍未获得有效世界坐标，程序会停止并写入
`termination_reason=target_not_found`，不会继续驶向旧固定终点。

另外，提示词中明确出现 `yellow/green/red/blue` 等颜色时，候选框还必须通过独立 RGB ROI
颜色校验；例如画面中只有黄色小车时输入“绿色平台”，会记录
`target_candidate_status=attribute_mismatch` 并停止为 `TARGET_NOT_FOUND`，不会把黄色小车
改名成绿色平台。未包含这些已知颜色属性的任意文本仍交给 YOLO-World，不会退化成固定物体类别表。

生成并评估 Genesis OOD 数据集（仅评测，不训练固定类别）：

```bash
.venv/bin/python examples/mobile_robot/vision/generate_open_vocab_ood.py \
  --layouts 12 --steps 1200 --sample-every 25 \
  --output-dir datasets/mobile_robot_open_vocab_ood

.venv/bin/python examples/mobile_robot/vision/evaluate_open_vocab_ood.py \
  --model models/mobile_robot/open_vocab/yolov8s-world.pt \
  --dataset datasets/mobile_robot_open_vocab_ood \
  --device cpu --imgsz 640 --infer-conf 0.001 \
  --decision-conf 0.05 --iou-threshold 0.5 --annotate \
  --output-dir out/mobile_robot_open_vocab_ood_eval
```

只在开发集试验等价自然语言模板时，使用 `--split dev` 和 prompt ensemble 文件；
模板去重后再评分，不能把 `ood_test` 用于反向调参：

```bash
.venv/bin/python examples/mobile_robot/vision/evaluate_open_vocab_ood.py \
  --model models/mobile_robot/open_vocab/yolov8s-world.pt \
  --dataset datasets/mobile_robot_open_vocab_ood --device cpu \
  --split dev --infer-conf 0.001 --decision-conf 0.05 \
  --prompt-variants-file examples/mobile_robot/vision/open_vocab_prompt_variants.dev.json \
  --output-dir out/mobile_robot_open_vocab_dev_prompt_ensemble
```

开发集选择后，可在冻结的 `ood_test` 复测同一文件。评估汇总还会按 split、原始 prompt、
资产变体以及可见/截断状态拆分，帮助定位渲染域失败，而不会把 OOD 真值传入运行时。
需要检查候选框内的弱颜色/形状证据时可额外加 `--roi-rerank`；它只改变已接受候选的排序，
不修改模型置信度，也不会替代多帧确认。

对已保存的评测结果可生成高置信度失败样本清单：

```bash
.venv/bin/python examples/mobile_robot/vision/analyze_open_vocab_failures.py \
  --dataset datasets/mobile_robot_open_vocab_ood \
  --results out/mobile_robot_open_vocab_dev_prompt_ensemble_v2/ood_results.jsonl \
  --output-dir out/mobile_robot_open_vocab_dev_prompt_ensemble_v2/failures \
  --min-confidence 0.05 --top-k 10
```

该数据集保存 Genesis 实例 mask、深度和 prompt 真值，用于计算 Recall、Top-1、
缺失目标误报率、歧义检出率及 RGB-D 误差；真值只在推理后参与评分，不会传入 YOLO-World。
当前结果见 [_reports/OPEN_VOCABULARY_EXECUTION_REPORT.md](_reports/OPEN_VOCABULARY_EXECUTION_REPORT.md)。

当前基准结果、实验推进矩阵和后续准入条件见 [_reports/OPEN_VOCABULARY_EXECUTION_REPORT.md](_reports/OPEN_VOCABULARY_EXECUTION_REPORT.md)。

## 四轮 URDF 动力学实验

新增的 [room_navigation_urdf.py](room_navigation_urdf.py) 使用四个连续轮关节：左侧前后轮同步、右侧前后轮同步，车体通过 Genesis 动力学和轮胎接触运动。四轮编码器会生成轮里程计，RGB、深度相机、LiDAR 和 IMU 仍安装在 URDF 的 `base_link` 上。

推荐先运行无障碍动力学回归：

```bash
python examples/mobile_robot/room_navigation_urdf.py --steps 1200
```

障碍物展示场景使用预先规划的安全绕行航点，并增加场景边界保护：

```bash
python examples/mobile_robot/room_navigation_urdf.py \
  --vis --robot-view --save-sensors \
  --scenario room_obstacle --steps 1400
```

由于四轮滑移转向的轮胎接触参数仍在校准，动态展示默认使用 Genesis 车体实际位姿进行高层航点控制，同时记录轮编码器里程计。使用 `--control-pose wheel_odom` 可以专门测试未经校准的轮里程计控制效果。
