# 开放词汇视觉阶段执行报告

> 执行日期：2026-09-09
>
> 对应计划：[TODO.md](../TODO.md)

## 1. 本阶段结论

已完成开放词汇视觉的第一阶段工程骨架、YOLO-World 离线基准和真实车载相机旁路，但尚未达到“自然语言目标可直接驱动导航”的准入条件。

当前结论：

- YOLO-World 可以在车载 RGB 帧上按文本提示运行，且输出多个候选框；
- 当前 Genesis 合成图像上的 zero-shot 结果不稳定，不能把候选直接交给路径规划；
- 当前路线暂不考虑 OWLv2；它不是 YOLO-World 运行或后续导航的必要依赖，后续仅作为可选替代模型保留；
- 原有闭集 YOLO 权重、`YoloDetector` 和 `room_navigation_vision.py --perception-mode yolo` 路径未改动，单帧回归通过。

本地开放词汇文件已固定路径和 SHA-256：

- `models/mobile_robot/open_vocab/yolov8s-world.pt`：`095f5266bb9b654bd5ad9e21e9cdeda78e0f2c8460f5d652eaf04bab7ee251cf`；
- `weights/clip/ViT-B-32.pt`：`40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`。

## 2. 已实现

### 2.1 统一开放词汇接口

新增 [open_vocab_grounder.py](../vision/open_vocab_grounder.py)：

- `OpenVocabularyGrounder` 协议；
- `GroundingCandidate` 数据契约，保留原始 prompt、候选框、置信度、帧号、仿真时间、模型和延迟；
- `YoloWorldGrounder` 本地权重适配器；
- `resolve_open_vocab_device("auto")` 遵循 MPS 优先、CPU 回退；实际设备和回退原因写入结果；
- MPS 算子运行时失败时自动回退 CPU；
- 模型路径必须是本地文件，推理阶段不隐式下载。

仓库中保留了未启用的 `owlv2_grounder.py` 实验适配器，但它不属于当前验收路线，
不会阻塞 YOLO-World 的 OOD 测试、目标确认或后续控制开发。

### 2.2 离线评估入口

新增 [evaluate_open_vocab.py](../vision/evaluate_open_vocab.py)：

- 当前实际评估后端为 `yolo-world`；CLI 保留后端扩展点，但不把 OWLv2 作为必选依赖；
- 支持重复 `--prompt` 或 UTF-8 prompts 文件；
- 输出逐帧 `candidates.jsonl`、`summary.json` 和可选标注图；
- 将观测分类为 `candidate`、`ambiguous`、`low_confidence`、`no_visual_match`；
- 记录候选数量、设备回退原因、P50/P95/平均/最大延迟；
- 不访问 Genesis 真值，不把结果直接接入控制。

### 2.3 Genesis 车载实时旁路

`room_navigation_vision.py` 新增 `--perception-mode open_vocab`：

- 可通过重复 `--vision-prompt` 将自然语言短语送入本地 grounder；
- `--open-vocab-device auto` 使用 MPS 优先、CPU 回退；
- `--annotated-view` 显示独立候选框窗口，`--robot-view` 保持原始相机窗口；
- 结果进入既有 `VisionResult`/tracker/JSONL 管线，但不改变 waypoint、轮速或 LiDAR 安全层；
- 模型不可用、提示缺失或无匹配时安全记录失败状态，不产生伪造导航目标。

宿主 OpenGL 环境 5 步真实车载回归已通过（无导航目标接入）：

```text
visual_frame_count=5, inference_count=5, vision_error_count=0
perception_mode=open_vocab, device=cpu(auto_cpu)
annotated_view=true, reached=false（仅运行 5 步）
```

输出目录：[out/open_vocab_live_gui_smoke](../../../out/open_vocab_live_gui_smoke/)。
Genesis 原始车载窗口和独立候选框窗口均成功启动；CPU 首帧/GUI 会产生明显 warm-up 延迟，
因此开放词汇结果当前只能旁路记录，不能放入实时控制回路。

示例候选框帧：[frame_00005.png](../../../out/open_vocab_live_gui_smoke/rgb_robot_annotated/frame_00005.png)。

### 2.4 回归保护

新增测试覆盖：

- MPS/CPU 设备解析；
- 开放词汇候选 JSON 契约和非法置信度；
- YOLO-World 缺少本地模型或 CLIP 权重时的明确失败；
- 原视觉测试共 12 项通过。

## 3. YOLO-World 真实基准

输入目录：`out/mobile_robot_yolo_annotated_view/rgb_robot`，共 45 帧。

执行命令：

```bash
.venv/bin/python examples/mobile_robot/vision/evaluate_open_vocab.py \
  --backend yolo-world \
  --model models/mobile_robot/open_vocab/yolov8s-world.pt \
  --images out/mobile_robot_yolo_annotated_view/rgb_robot \
  --prompt "yellow car" --prompt "green pillar" --prompt platform \
  --prompt "red box" --prompt "blue cylinder" \
  --device auto --imgsz 640 --infer-conf 0.001 --decision-conf 0.05 \
  --annotate --output-dir out/open_vocab_benchmark_full
```

结果文件：

- [summary.json](../../../out/open_vocab_benchmark_full/summary.json)
- [candidates.jsonl](../../../out/open_vocab_benchmark_full/candidates.jsonl)
- [annotated/](../../../out/open_vocab_benchmark_full/annotated/)

关键结果：

| 指标                   |                    结果 |
| ---------------------- | ----------------------: |
| 帧数                   |                      45 |
| 设备                   | `cpu`（`auto_cpu`） |
| P50 延迟               |                 96.3 ms |
| P95 延迟               |                115.3 ms |
| 最大延迟               |                270.6 ms |
| `candidate` 帧       |                       9 |
| `low_confidence` 帧  |                      30 |
| `no_visual_match` 帧 |                       6 |
| `ambiguous` 帧       |                       0 |

各 prompt 的最高置信度（仅表示模型分数，不表示正确率）：

| Prompt            | 最高分 |
| ----------------- | -----: |
| `yellow car`    |  0.009 |
| `green pillar`  |  0.063 |
| `platform`      |  0.034 |
| `red box`       |  0.266 |
| `blue cylinder` |  0.176 |

这组数据表明模型对当前 Genesis 渲染域存在明显差距；尤其“黄色小车”不能稳定被文本提示定位。平台候选数量较多但分数很低，存在背景/几何误检风险。故当前阶段不得接入主动搜索或路径规划。

## 4. 当前模型路线决策

本阶段冻结 YOLO-World 为唯一开放词汇视觉实验模型，暂不等待或下载 OWLv2：

- YOLO-World 已能完成文本 prompt → 候选框 → 车载相机可视化 → JSONL 记录的完整旁路；
- OWLv2 只会提供另一种模型对比，不是当前系统接口、RGB-D 标定或控制器的前置依赖；
- 当前 YOLO-World 结果尚未达到准入门槛，因此暂时不能把“已有模型”解释为“已经实现自然语言导航”；
- 若后续 YOLO-World 在 OOD 测试中不达标，再单独评估领域适配、提示策略或替代模型，不改变闭集 YOLO 基线。

## 5. 闭集 YOLO 保留验证

使用原权重 `models/mobile_robot/training_full_v2/yolo11n_custom/weights/best.pt` 对车载帧 `frame_00025.png` 做单帧 CPU 推理：

```text
status=ok, detections=2
car 0.953
cylinder_obstacle 0.918
```

`YoloDetector`、原权重和 `room_navigation_vision.py` 的闭集运行路径未修改。开放词汇模块仅新增旁路文件和评估入口。

## 6. 当前实验推进矩阵

| 实验阶段 | 当前状态 | 已有证据 | 尚未证明 |
| --- | --- | --- | --- |
| E0 本地模型加载 | 已完成 | YOLO-World 权重、CLIP 文本权重可离线加载；MPS→CPU 策略可运行 | MPS 长时间稳定性和正式内存基线 |
| E1 离线多 prompt 基准 | 已完成但未达标 | 45 帧、5 个 prompt、候选 JSONL、标注图和延迟统计 | 真实 Phrase Recall、Top-1 正确率 |
| E2 车载相机旁路 | 已完成 | Genesis 真实相机 5 帧、5 次推理、0 视觉异常、独立标注窗口 | 连续运行实时性、目标确认可靠性 |
| E3 闭集回归保护 | 已完成 | 原闭集 YOLO 单帧 `car=0.953`、`cylinder_obstacle=0.918` | 不代表开放词汇泛化 |
| E4 Genesis OOD 数据集 | 未开始 | 目前只有既有闭集展示场景帧 | 绿色柱子、平台、新材质、多目标、缺失目标、诱饵物体 |
| E5 开放词汇指标 | 未完成 | 当前只有候选状态和延迟统计 | Recall、Top-1、缺失误报率、歧义检出率、3D 误差 |
| E6 导航接入 | 明确禁止 | 控制旁路保持 waypoint + LiDAR | 多帧确认、目标坐标、附近姿态和路径规划 |

当前阶段出口是 **E4/E5 通过**，而不是模型能够成功返回任意一个候选框。

## 7. 下一步准入顺序

1. 冻结 YOLO-World 的模型、prompt、阈值、设备和输出协议，避免边测边改阈值；
2. 生成真正的 Genesis OOD 测试集：绿色柱子、平台、新材质、多目标、目标缺失、诱饵物体；按物体资产和语义组合划分，不能按相邻帧随机拆分；
3. 为每帧保存 RGB、Genesis 真值分割/实例框、物体资产 ID、语义 prompt、是否存在和遮挡信息；
4. 实现离线指标：Phrase Grounding Recall@IoU、Top-1 目标选择、目标缺失误报率、歧义检出率、候选框深度和世界坐标误差；
5. 在不控制车辆的前提下增加 mask/ROI 属性验证，明确 `NO_VISUAL_MATCH`、`AMBIGUOUS_TARGET` 和低置信度状态；
6. 增加多帧确认和 RGB-D 三维定位，只有稳定 target ID、有效深度和通过 LiDAR 交叉检查的目标才可进入规划；
7. 只有上述指标冻结并达标后，才实现主动搜索、目标附近姿态和路径规划；
8. LLM 仍保持独立：只输出完整开放指代表达和导航约束，不输出坐标、路径或轮速。

在第 2～6 项未通过前，任何“行驶到黄色小车/绿色柱子/平台附近”的演示都只能作为候选可视化，
不应宣称已经实现自然语言导航。
