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

新增 OOD 数据生成和评分工具：

- [generate_open_vocab_ood.py](../vision/generate_open_vocab_ood.py)：只生成 RGB、深度、Genesis 实例 mask、物体资产真值和 prompt 清单，不生成 YOLO `dataset.yaml`，不训练固定类别；
- [evaluate_open_vocab_ood.py](../vision/evaluate_open_vocab_ood.py)：推理完成后再读取真值计算 Recall、Top-1、缺失目标误报率、歧义检出率和 RGB-D 世界坐标误差。

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
| E4 Genesis OOD 数据集 | 已完成 | 542 帧、12 layout、RGB/深度/实例 mask/资产真值/prompt 清单，layout-level split | 扩大资产和光照覆盖仍可继续 |
| E5 开放词汇指标 | 已完成首轮但未达标 | Recall、Top-1、缺失误报率、歧义检出率、RGB-D 误差和延迟均已输出 | 需要提高泛化并重新验收 |
| E6 导航接入 | 明确禁止 | 控制旁路保持 waypoint + LiDAR | 多帧确认、目标坐标、附近姿态和路径规划 |

当前阶段出口是 **E4/E5 通过**，而不是模型能够成功返回任意一个候选框。

### 6.1 OOD 数据集与 YOLO-World 评测结果

正式数据集已生成：

- 数据目录：[datasets/mobile_robot_open_vocab_ood](../../../datasets/mobile_robot_open_vocab_ood/)；
- 生成配置：[generation_config.json](../../../datasets/mobile_robot_open_vocab_ood/generation_config.json)；
- prompt 真值：[prompts.jsonl](../../../datasets/mobile_robot_open_vocab_ood/prompts.jsonl)；
- 评测汇总：[summary.json](../../../out/mobile_robot_open_vocab_ood_eval/summary.json)；
- 逐帧结果：[ood_results.jsonl](../../../out/mobile_robot_open_vocab_ood_eval/ood_results.jsonl)。

数据规模：542 帧、12 个独立 layout，`dev=136`、`negative_test=136`、`ood_test=270`；
切分在 layout 级完成，未将相邻帧随机拆到不同 split。每帧包含 RGB、原始深度、实例 mask、
物体资产 ID、可见性、真值框和 prompt 目标 ID。生成配置明确标记 `closed_set_training=false`。

冻结参数：YOLO-World 本地权重、`imgsz=640`、推理阈值 `0.001`、决策阈值 `0.05`、IoU 阈值 `0.5`、CPU。

| 指标 | 结果 | 结论 |
| --- | ---: | --- |
| Phrase Grounding Recall@0.5 | 75/534 = **14.0%** | 未达标 |
| Top-1 目标选择 | 75/534 = **14.0%** | 未达标 |
| 目标不存在误报率 | 49/1626 = **3.0%** | 需要结合业务门槛判断 |
| 目标在场景但当前视野外误报率 | 2/1634 = **0.12%** | 较低 |
| 歧义检出率 | 0/15 = **0%** | 未达标 |
| RGB-D 世界坐标误差（已接受候选） | mean **0.537 m**，P95 **1.385 m** | 未达标 |
| 推理延迟 | P50 **94.1 ms**，P95 **153.0 ms** | 仅作旁路参考 |

按 prompt 的可见目标 Recall：`green pillar=37.2%`、`orange platform=16.9%`、
`yellow car=0%`、`purple sculpture=0%`。其中 `red traffic cone` 作为缺失目标诱饵的误报率为
8.5%，说明不能只看正例候选数量。

因此 E4 已完成，E5 已完成首轮测量但未通过准入；当前不能进入三维目标确认或路径规划。

### 6.2 本轮 dev prompt ensemble 实验

为避免用 `ood_test` 调参，本轮只在 `dev` split 比较了原始短语和等价自然语言模板。
配置文件：[open_vocab_prompt_variants.dev.json](../vision/open_vocab_prompt_variants.dev.json)。
等价模板的候选框先按 IoU `0.7` 去重，再按原始模型置信度决策；没有使用 Genesis 真值做运行时过滤。

执行命令：

```bash
.venv/bin/python examples/mobile_robot/vision/evaluate_open_vocab_ood.py \
  --model models/mobile_robot/open_vocab/yolov8s-world.pt \
  --dataset datasets/mobile_robot_open_vocab_ood \
  --device cpu --imgsz 640 --infer-conf 0.001 --decision-conf 0.05 \
  --iou-threshold 0.5 --split dev \
  --prompt-variants-file examples/mobile_robot/vision/open_vocab_prompt_variants.dev.json \
  --output-dir out/mobile_robot_open_vocab_dev_prompt_ensemble_v2
```

| dev 指标 | 原始短语 | prompt ensemble | 变化 |
| --- | ---: | ---: | ---: |
| Phrase Grounding Recall@0.5 | 20/126 = 15.9% | 28/126 = **22.2%** | +6.3 pp |
| 目标不存在误报率 | 14/408 = 3.43% | 13/408 = **3.19%** | -0.24 pp |
| 当前视野外误报率 | 0/418 = 0% | 3/418 = 0.72% | 变差 |
| 歧义检出率 | 0/3 = 0% | 1/3 = 33.3% | 仍不稳定 |

按 prompt 的 Recall 为：`green pillar=51.9%`、`orange platform=32.6%`、
`yellow car=0%`、`purple sculpture=0%`。因此 ensemble 只能作为候选生成改进，不能被解释为
已经解决开放词汇泛化。

阈值敏感性也在 `dev` 的已保存候选上做了离线检查：将决策阈值从 `0.05` 降到 `0.001`
时 Recall 可升至 54.0%，但视野外误报率升至 24.6%；故冻结阈值仍为 `0.05`，不使用低阈值
制造演示效果。

### 6.3 冻结 `ood_test` 的 ensemble 复测

在完成 dev 选择后，使用同一组模板、同一权重和阈值对未参与调参的 `ood_test` 运行：

- 结果：[summary.json](../../../out/mobile_robot_open_vocab_ood_eval_prompt_ensemble/summary.json)；
- 逐帧结果：[ood_results.jsonl](../../../out/mobile_robot_open_vocab_ood_eval_prompt_ensemble/ood_results.jsonl)。

| ood_test 指标 | 首轮原始短语 | 冻结 ensemble | 结论 |
| --- | ---: | ---: | --- |
| Phrase Grounding Recall@0.5 | 14.0% | **23.1%** (61/264) | 有改善但未达标 |
| 目标不存在误报率 | 3.0% | **2.22%** (18/810) | 有改善 |
| 当前视野外误报率 | 0.12% | 0.61% (5/816) | 仍需约束 |
| 歧义检出率 | 0% | 12.5% (1/8) | 未达标 |
| RGB-D 世界坐标误差 | mean 0.537 m | mean 0.705 m | 未达标 |

按 prompt 的 `ood_test` Recall：`green pillar=56.1%`、`orange platform=32.6%`、
`yellow car=0%`、`purple sculpture=0%`；`red traffic cone` 缺失误报率为 6.67%。
该结果确认 ensemble 不会把失败的黄色车辆和紫色物体问题隐藏起来，当前仍禁止进入三维定位、
主动搜索和路径规划。

### 6.4 ROI 证据验证层（实验，不改变置信度）

新增 [open_vocab_validation.py](../vision/open_vocab_validation.py)，从候选框中心 ROI 提取弱颜色和
粗形状证据，可选地只用于候选排序（`--roi-rerank`）。它不读取实例 mask、资产 ID 或世界坐标，
也不会把弱证据改写成模型置信度；因此不会把验证层变成闭集分类器。

在同一 `dev` ensemble 数据上打开 `--roi-rerank` 后，Recall、Top-1、误报率和歧义率均未变化，
说明当前主要瓶颈是 YOLO-World 对 `yellow car`/`purple sculpture` 的候选召回，而不是候选排序。
该层先保留为后续多帧确认和 mask/ROI 验证的接口，暂不作为导航准入依据。

### 6.5 失败样本诊断

新增 [analyze_open_vocab_failures.py](../vision/analyze_open_vocab_failures.py)，对已保存的 JSONL
结果按可见目标漏检、视野外误报、缺失目标误报和歧义漏检分类，并输出高置信度样本、prompt 和
资产变体。`dev` ensemble 诊断结果：[failure_analysis.json](../../../out/mobile_robot_open_vocab_dev_prompt_ensemble_v2/failures/failure_analysis.json)。

`min_confidence=0.05` 时共发现 116 个失败事件：

| 类型 | 数量 | 主要线索 |
| --- | ---: | --- |
| 可见目标漏检 | 98 | `purple sculpture` 45、`yellow car` 11，另有小/截断平台漏检 |
| 缺失目标误报 | 13 | `red traffic cone` 11，存在较高置信度诱饵框 |
| 视野外误报 | 3 | 均为 `orange platform` |
| 歧义漏检 | 2 | 两个绿色柱子同时可见时只保留一个候选 |

这说明下一轮应优先处理“模型未产生正确候选”和“同 prompt 多实例关联”，而不是继续调低
决策阈值；低阈值实验已经证明会显著放大视野外误报。

### 6.6 输入分辨率实验

在冻结 prompt ensemble 不变的前提下，仅将 `imgsz` 从 640 提高到 1280，在 `dev` 集复测：

| 指标 | `imgsz=640` | `imgsz=1280` | 结论 |
| --- | ---: | ---: | --- |
| Recall@0.5 | 22.2% | 23.0% | 无实质改善 |
| `yellow car` Recall | 0% | 0% | 未解决 |
| `purple sculpture` Recall | 0% | 4.4% | 仍不可用 |
| 视野外误报率 | 0.72% | 0.96% | 变差 |
| CPU P50 延迟 | 105.6 ms | 350.4 ms | 约 3.3 倍 |

因此不采用 1280 作为当前实时方案；失败主要不是输入缩放造成的，下一轮应转向渲染域/语义特征
适配和多实例关联，而不是继续堆高推理分辨率。

## 7. 下一步准入顺序

1. 保持当前 OOD test 不变，禁止用 test 结果反向调阈值；下一轮仍只在 `dev` split 做 prompt 模板、图像预处理和阈值实验；
2. 已完成 prompt ensemble 和失败样本诊断，但 `yellow car`/`purple sculpture` 仍无可靠召回；下一轮只在 `dev` 做渲染域预处理/提示策略实验，并用诊断清单复核诱饵误报和多目标歧义；
3. ROI 颜色/形状证据验证层已实现为可选排序旁路，但首轮没有改善 Recall；继续实现 mask/ROI 拒绝规则与多帧稳定性验证，验证层只能重排序/拒绝候选，不能读取 Genesis 真值；
4. 评分器已扩展为按 split、prompt、资产变体和可见/截断状态输出指标，后续用这些切片定位渲染域和小目标问题；
5. 在视觉候选达到门槛后，再增加 mask/ROI 属性验证、多帧确认和 RGB-D 三维定位；当前 0.537 m mean / 1.385 m P95 误差不能用于停车控制；
6. 只有 Recall、Top-1、缺失误报率、歧义检出率和三维误差同时达标后，才实现主动搜索、目标附近姿态和路径规划；
7. LLM 仍保持独立：只输出完整开放指代表达和导航约束，不输出坐标、路径或轮速。

在第 2～6 项未通过前，任何“行驶到黄色小车/绿色柱子/平台附近”的演示都只能作为候选可视化，
不应宣称已经实现自然语言导航。

## 8. 本轮交付验证

- `compileall examples/mobile_robot`：通过；
- 视觉单元测试：`17 passed`，包含新增目标属性一致性和动态航点契约测试；
- 原闭集 YOLO test 回归：151 张图，precision `0.950`、recall `0.898`、mAP50 `0.943`；
  原权重和 `perception-mode yolo` 路径未修改；
- 开放词汇 `ood_test` ensemble：已生成逐帧 JSONL 和按 prompt/资产/可见性拆分的 summary，
  但准入指标仍未通过，因此开放词汇旁路（不带 `--instruction`）不会改动任何导航目标、waypoint
  或轮速控制逻辑。

## 9. 最小自然语言 Demo 状态

已在 `room_navigation_vision.py` 增加一个显式 `--instruction` 入口，用于先演示视觉/控制侧的
最小链路。此前版本的实际运行证据显示：候选和 `position_world` 已产生，但默认状态阈值
`0.05` 高于本地 YOLO-World 在 Genesis 小图上的原始分数（约 `0.001`～`0.018`），导致
`target_lock=false`，车辆跑完旧固定路线。这不是坐标/航点算法未写，而是候选没有被准入，且
结束条件错误地允许指令模式继续跑到固定终点。

```text
用户指令 → 简单短语转换 → YOLO-World 候选 → 连续 RGB-D 确认
          → 目标附近临时航点 → 原 waypoint + LiDAR 控制器
```

当前已修正为：

- `--target-lock-conf` 与显示用的 `--open-vocab-decision-conf` 分离，默认 Demo 锁定阈值为 `0.001`；
- 只有带有限 `position_world` 的开放词汇候选，且连续确认达到 `--target-confirm-frames`，才调用
  控制器的 `replace_waypoints()`，重置旧航点进度和 LiDAR detour 状态；
- `summary.json`/`telemetry.csv` 写入 `navigation_mode`、`target_lock`、`target_near_waypoint`、
  `command_target_x/y`；目标确认后后两者应一致；
- 指令模式到达“视觉搜索路线”末端仍未锁定目标时，停止并报告 `termination_reason=target_not_found`，
  不再驶向旧固定终点。

该入口不会读取 Genesis 真值，也不会改变闭集 YOLO 路径。当前短语转换器只处理少量常见中文词，
未知英文词会原样保留；复杂关系（例如“柱子后面的平台”）、多语言改写和真正的任意现实物品
语义仍由独立 LLM 适配器负责。由于当前 OOD 指标未达标，这个 Demo 适合展示链路和可视化，
不应作为任意目标可靠到达的验收结果。

自动化环境中的无 GUI 冒烟运行受到宿主 OpenGL 限制（`Failed to find an OpenGL 3.2+ core profile`
而无法创建 Genesis 渲染器）；CLI 解析、17 项视觉单元测试和代码编译均通过。请在有 Genesis
WindowServer/OpenGL 的本机终端运行 README 中的可视化命令完成实际窗口演示。

## 10. 提示词切换误锁定修正

2026-09-10 的实际日志发现：同一场景先输入“黄色小车”、再输入“绿色平台”时，第二次
YOLO-World 仍在黄色小车的 ROI 上产生候选。候选标签虽然被模型写成 `green platform`，但
`enrich_colors` 给出的独立 RGB 属性为 `color=yellow`、`color_confidence≈0.99`，两次运行的
`target_world_position` 也几乎相同（约 `(-1.91, 0.10)`）。此前 `target-lock-conf=0.001`
只检查文本候选和世界坐标，因而错误接管了同一临时航点。

已增加开放词汇属性一致性门：

- 提示词包含可验证颜色时，候选颜色必须一致且 ROI 置信度至少为 `0.35`；
- 颜色不一致的候选只保留在视觉日志中，导航状态写为 `attribute_mismatch`；
- 当前 `vision_route_showcase` 没有绿色平台时，最终应为 `target_lock=false`、
  `termination_reason=target_not_found`，不会驶向黄色小车；
- 没有明确颜色的任意文本仍走开放词汇模型，不增加固定物体类别表。

对用户提供的第二次 `green platform` 日志离线复算：13 帧存在可用 RGB-D 候选，13 帧均被
颜色属性门拒绝，0 帧允许锁定。新增单元测试后视觉/控制测试为 `17 passed`。
