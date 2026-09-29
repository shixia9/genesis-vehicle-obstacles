# Genesis World：面向无人机与车联网研究的开放物理仿真平台

> 面向对象：无人机、车联网、智能交通、机器人与强化学习方向的学生和研究人员  
> 仓库基线：`genesis-world 1.3.3`，本地提交 `eb2606c`（2026-09-15）  
> 文档日期：2026-09-29  
> 说明：本文以当前仓库源码为准；“Genesis 原生能力”“本仓库扩展”“建议建设方案”会明确区分。

## 1. 核心结论

Genesis World 是一个面向 Physical AI 的开源仿真平台。它把多物理求解、机器人资产、渲染与传感器、并行环境、控制接口和可微分计算能力组织在统一的 Python 接口下，并通过 Quadrants 编译层适配 CUDA、ROCm、Apple Metal、Vulkan、x86 与 ARM64 等后端。

对无人机与车联网课题组而言，Genesis 最有价值的定位不是“替代全部仿真器”，而是承担以下角色：

1. **高吞吐物理世界与传感器底座**：无人机、车辆、障碍物、地形、相机、深度、LiDAR、IMU、接触等均处在同一个可编程场景中。
2. **控制与强化学习闭环**：动作写入实体，仿真推进，状态/传感器读出，再进入控制器或策略网络；并行环境适合 PPO 等 on-policy 训练。
3. **算法原型与数据生成平台**：可实现 PID、A*、采样规划、MPC、模仿学习、单/多智能体强化学习，以及视觉训练数据和场景随机化。
4. **V2X 联合仿真的物理侧**：Genesis 当前不是完整网络仿真器，不应宣称原生支持 5G/NR-V2X 协议栈、排队、调度、SINR/BER。更合理的方案是通过状态—动作桥与 ns-3、OMNeT++ 或自定义信道模型联合仿真。

一句话总结：**Genesis 可成为课题组“空—地智能体物理与感知仿真层”，网络系统和通信协议由专门工具补齐，二者共同形成闭环实验平台。**

## 2. 项目背景与工程定位

Genesis 最初于 2024 年 12 月以学术项目形式启动，现以 **Genesis World** 品牌发展，并由 Genesis AI 官方支持。当前仓库采用 Apache License 2.0，Python 要求为 3.10–3.13，包版本为 1.3.3。

项目解决的是 Physical AI 研发中常见的割裂问题：机器人接口、刚体与软体物理、渲染、传感器、批量训练和硬件后端往往来自不同工具；Genesis 尝试把它们放进同一场景、同一状态和同一 Python 工作流中。

### 2.1 四层技术栈

```text
研究应用
控制 / 规划 / RL / 模仿学习 / 数据生成 / 多智能体协同
                         │
┌────────────────────────▼────────────────────────┐
│ Simulation Interface                            │
│ Scene / Entity / Morph / Sensor / Camera / GUI │
├─────────────────────────────────────────────────┤
│ Physics                                         │
│ Rigid / FEM / MPM / PBD / SPH / Stable Fluid  │
│ + SAP / uIPC / 显式耦合                         │
├─────────────────────────────────────────────────┤
│ Render                                          │
│ Nyx / Luisa / Pyrender                         │
├─────────────────────────────────────────────────┤
│ Compiler: Quadrants                             │
│ CUDA / ROCm / Metal / Vulkan / x86 / ARM64     │
└────────────────────────┬────────────────────────┘
                         │
                       硬件
```

### 2.2 与常见仿真器的关系

Genesis 的突出特征是统一多物理、Python 可编程、高并行环境和跨后端编译。它不是通信协议栈，也不是道路交通流专用工具。研究中可按职责组合：

| 任务 | 更适合承担的工具 |
|---|---|
| 无人机/车辆动力学、碰撞、传感器、视觉 | Genesis |
| 大规模道路交通流、路网与交通信号 | SUMO 等交通工具 |
| 5G/C-V2X/802.11p、队列、调度、无线链路 | ns-3、OMNeT++/Veins 等 |
| 策略学习、控制与规划 | PyTorch、rsl-rl、自研算法，环境由 Genesis 提供 |

## 3. 源码架构与一次仿真如何运行

`gs.init()` 选择计算后端并初始化全局运行时；`gs.Scene` 聚合求解器、耦合器、可视化器与传感器管理器；`scene.add_entity()` 根据 material/morph 把实体分配给合适的求解器；`scene.build()` 完成资产解析、内存布局、环境复制和内核编译；`scene.step()` 依次推进物理、传感器和可视化。

```python
import genesis as gs

gs.init(backend=gs.gpu)
scene = gs.Scene(show_viewer=True)
scene.add_entity(gs.morphs.Plane())
drone = scene.add_entity(
    gs.morphs.Drone(file="urdf/drones/cf2x.urdf", pos=(0, 0, 1))
)
scene.build()

while True:
    drone.set_propellers_rpm([14468.4] * 4)
    scene.step()
```

### 3.1 求解器与耦合

`Simulator` 在一个场景中持有 Rigid、Kinematic、MPM、SPH、PBD、FEM、Stable Fluid 和 Tool 求解器，并选择 SAP、Legacy 或 IPC coupler。它的研究意义在于：同一实验中可以同时存在刚体无人机、柔性物体、颗粒/液体或复杂接触，而不必为每种物理现象单独维护场景状态。

### 3.2 资产与实体

接口支持 URDF、MJCF、OBJ、GLB、USD 等资产路径。Morph 描述“如何导入”，Material 决定“由哪个物理解算”，Surface 决定“如何显示”。实体建立后提供位姿、速度、关节、外力、控制目标等访问器。

### 3.3 渲染与传感器

Genesis 提供 Nyx、Luisa 和 Pyrender 三类渲染路径；相机被统一为传感器。当前源码中还包括深度相机、光线投射/LiDAR、IMU、接触力、关节力矩、表面距离、触觉与温度等传感器。

非相机传感器的通用选项支持：

- `history_length`：历史观测；
- `delay`：按仿真时间步离散的读取延迟；
- `jitter`：在 `[0, jitter)` 内采样的随机附加延迟，且不大于 `delay`；
- 附着到实体/连杆以及调试可视化。

这意味着通信研究可先把“过时观测”注入感知链路，但它仍不等价于分组网络、共享无线资源或协议栈。

## 4. 并行仿真、随机化与强化学习

### 4.1 并行环境

`scene.build(n_envs=B)` 会把同一场景布局批量复制，实体状态和动作以张量批处理。部分环境可以用 `envs_idx` 单独控制或重置。这种设计特别适合 on-policy 强化学习，因为每次策略更新可以同时收集大量轨迹。

### 4.2 异构环境与领域随机化

仓库示例支持：

- 同一个批次中为不同环境分配不同几何形状；
- 按环境随机化质量、质心、惯量和摩擦；
- 随机目标、初始姿态、扰动、传感器噪声与时延；
- 选择少量环境用于可视化，其余环境无窗口高速运行。

这些能力可用于 sim-to-real：训练时随机化无人机质量、旋翼系数、风扰、传感器噪声、通信时延与丢包，部署时降低模型对单一仿真参数的过拟合。

### 4.3 Genesis 与强化学习算法的边界

Genesis 提供环境动力学、批处理张量、reset/step、观测与动作接口；PPO 实现来自外部 `rsl-rl-lib>=5.0.0`。因此更准确的说法是“Genesis 对 RL 工作流友好并提供示例集成”，而不是“Genesis 内置了完整强化学习算法库”。SAC、TD3、MAPPO、QMIX 等可以通过 Gymnasium/PettingZoo 风格封装或直接张量接口接入。

## 5. 无人机支持：从资产到动力学

当前仓库提供 Crazyflie 2.X 四旋翼模型和四类可运行示例：键盘控制、预定义 RPM 序列、PID 航点飞行、PPO 悬停训练/评估。

### 5.1 Crazyflie 2.X URDF 建模

模型文件为 `genesis/assets/urdf/drones/cf2x.urdf`。关键参数如下：

| 参数 | 数值 | 含义 |
|---|---:|---|
| 质量 | 0.027 kg | `base_link` 质量 |
| 转动惯量 | Ixx=Iyy=1.4e-5, Izz=2.17e-5 kg·m² | 主惯量 |
| 臂长属性 | 0.0397 m | 模型属性 |
| 旋翼力臂 | `(±0.028, ±0.028, 0)` m | 四个旋翼连杆惯性原点 |
| `kf` | 3.16e-10 | 推力系数 |
| `km` | 7.94e-12 | 反扭矩系数 |
| 旋转方向 | `(-1,+1,-1,+1)` | CW/CCW 交替 |
| 碰撞近似 | 半径 0.06 m、长度 0.025 m 的圆柱 | `base_link` collision geometry |

四个旋翼是固定连接到机体的独立 link。旋翼 link 被保留，一方面用于施加局部外力和扭矩，另一方面用于旋翼视觉动画。

### 5.2 旋翼动力学

每个旋翼以自身局部坐标系施加：

\[
\mathbf{F}_i = [0,0,K_F\,\mathrm{RPM}_i^2]^T,
\qquad
\mathbf{M}_i = [0,0,s_iK_M\,\mathrm{RPM}_i^2]^T
\]

其中 `s_i` 是旋向。力作用点相对质心的偏置进一步产生滚转/俯仰力矩；交替旋向的反扭矩产生偏航控制。`set_propellers_rpm()` 每个仿真步只能调用一次，输入支持单环境 `(4,)` 或批量环境 `(B,4)`。

按模型参数估算悬停转速：

\[
\mathrm{RPM}_{hover} = \sqrt{\frac{mg}{4K_F}}
\approx \sqrt{\frac{0.027\times9.81}{4\times3.16\times10^{-10}}}
\approx 14469\ \mathrm{RPM}
\]

与示例使用的 `14468.429` 基本一致。

### 5.3 支持的模型模式与限制

`gs.morphs.Drone` 接受 URDF，并声明 `CF2X`、`CF2P`、`RACE` 三种 mixer/model 模式。当前仓库只附带并示范 `cf2x.urdf`。

必须注意源码文档中仍写有“Drone doesn't support collision checking for now”，而 Crazyflie URDF 又定义了碰撞几何。对避障、撞击和安全强化学习实验，不应仅凭模型可见或 URDF 有 collision 标签就认为碰撞链路已验证；应先做独立的墙面碰撞、接触力和反弹测试，并将结果写入实验基线。

另外，当前旋翼模型是平方转速的集中力/扭矩模型，不是 CFD：默认不包含桨叶气动细节、复杂下洗、地面效应和风场。它适合控制与学习原型，但高保真气动研究需要增加外力模型、数据驱动残差或与专用气动工具耦合。

## 6. 无人机案例详解与运行方式

以下命令均在仓库根目录执行。

### 6.1 键盘交互飞行

```bash
.venv/bin/python examples/drone/interactive_drone.py
```

方向键通过四个旋翼的差分 RPM 控制平移，Space/Left Shift 调整总推力，Esc 退出。它适合解释 mixer 与低层执行器接口，不适合作为精确位置控制器。

### 6.2 预定义旋翼轨迹

```bash
.venv/bin/python examples/drone/fly.py --vis
```

源文件内部已经嵌入 RPM 比例序列，不依赖 README 中提到的外部 `fly_traj.pkl`。这是当前文档与实现不一致的一处，应以源码为准。

### 6.3 PID 航点飞行

```bash
.venv/bin/python examples/drone/fly_route.py
```

无人机依次飞向 `(1,1,2)`、`(-1,2,1)`、`(0,0,0.5)`，输出 `out/fly_route.mp4`。控制器采用位置环 → 速度环 → 姿态环的级联 PID，然后通过四旋翼 mixer 生成四个 RPM，并限制在约 `0.9–1.5 × hover RPM`。

这个案例展示的是**点到点控制**，尚不包括占据栅格、三维避障、动态障碍预测或最优轨迹生成。

### 6.4 PPO 悬停/随机目标跟踪

安装训练依赖：

```bash
uv pip install --python .venv/bin/python 'rsl-rl-lib>=5.0.0' tensorboard
```

训练：

```bash
.venv/bin/python examples/drone/hover_train.py \
  --exp-name drone-hovering \
  --num-envs 8192 \
  --max-iterations 301
```

评估与录制：

```bash
.venv/bin/python examples/drone/hover_eval.py \
  --exp-name drone-hovering \
  --ckpt 300 \
  --record
```

注意：训练脚本会删除同名 `logs/<experiment>` 后重建，请为重要实验使用新名称或先备份。

#### 环境定义

- 仿真频率：100 Hz，刚体内部步长为 0.005 s；
- 动作：4 维，裁剪后映射为 `(1 + 0.8a_i) × hover_rpm`；
- 观测：目标相对位置 3、四元数 4、机体系线速度 3、角速度 3、上一动作 4，共 17 维；
- 目标：x/y 在 `[-1,1]` m，z 固定为 1 m；
- 终止：超时、离目标区域过远、接近地面或姿态越界；
- 奖励：靠近目标、动作平滑、偏航约束、角速度惩罚、坠毁惩罚；
- 策略：rsl-rl PPO，Actor/Critic 均为 `128×128` tanh MLP。

#### 训练数据流

```text
8192 个并行 Crazyflie
      │ 17维观测
      ▼
PPO Actor ──4维动作──> RPM 映射 ──> Genesis 刚体步进
      ▲                                      │
      └──── reward / done / next observation ┘
```

当前 `simulate_action_latency` 配置虽然为 True，但 `HoverEnv.step()` 实际直接执行当前动作，并未像 Go2 环境那样切换到 `last_actions`；若研究通信时延，应修正实现或在桥接层显式建立动作队列。

## 7. 从“能飞”扩展到无人机路径规划

Genesis 提供的是物理和感知底座，路径规划算法需要由研究者构建。推荐采用分层架构：

```text
任务层：任务分配 / 航点序列 / 语义目标
规划层：A* / RRT* / Kinodynamic planning / MPC / RL
局部避障：深度或 LiDAR → 局部地图 → 安全速度/轨迹
控制层：位置—速度—姿态控制器 → 四旋翼 mixer
执行层：set_propellers_rpm() → Genesis rigid solver
```

建议的技术路线：

1. 先以 PID 控制器为稳定内环，规划器输出时间参数化位置/速度航点；
2. 从二维 A* 过渡到三维 voxel A* 或 RRT*；
3. 加入动力学可达性、最小转弯/加速度/jerk 约束；
4. 用 LiDAR/深度更新局部占据图，对动态车辆执行重规划；
5. 最后再用 RL 学习局部避障、轨迹跟踪残差或通信调度，避免一开始就端到端学习全部问题。

## 8. 本仓库的移动机器人案例：可复用的规划范式

`examples/mobile_robot/` 是本分支上的扩展示例，不应作为 Genesis 官方上游能力宣传。它提供了一个可迁移到无人机的完整闭环：

```text
自然语言 instruction
      ↓
开放词汇检测（YOLO-World / OWLv2）+ 颜色属性校验
      ↓
RGB-D 标定与目标世界坐标
      ↓
障碍膨胀后的二维 A* 静态路径
      ↓
航点控制 + LiDAR 运行时安全层
      ↓
差速车执行、遥测与视觉结果落盘
```

规划器以 0.20 m 网格搜索，将障碍按车体半尺寸和安全余量膨胀，禁止斜向穿越障碍角点，并删除共线航点。其思想可迁移到无人机：把二维栅格改成三维占据体素，将车体包络替换为无人机安全球/椭球，并把转弯约束替换为速度、倾角和加速度约束。

这个案例还说明了一个重要工程原则：**语义感知只负责确定目标，几何规划与运行时安全层仍独立存在**。这样即使视觉模型暂时失效，也不会直接接管低层控制。

## 9. Genesis 对 V2X/通信研究的真实支持边界

### 9.1 能力矩阵

| 能力 | 当前状态 | 推荐做法 |
|---|---|---|
| 无人机/车辆三维位姿、速度、碰撞环境 | 原生 | Genesis |
| RGB、Depth、LiDAR、IMU 等观测 | 原生 | Genesis Sensor API |
| 观测 history、delay、jitter | 部分原生 | 可做基础时延实验 |
| 动作执行延迟 | 示例级 | 显式动作队列 |
| 包生成、队列、吞吐、拥塞 | 未发现原生支持 | ns-3/OMNeT++ 或自研模块 |
| 802.11p、C-V2X、NR-V2X 协议栈 | 未发现原生支持 | 专业网络仿真器 |
| 路损、遮挡、SINR、BER、MCS | 未发现完整原生实现 | 网络/信道模型 |
| 多智能体策略与资源分配 | 环境可承载，算法外接 | MAPPO/QMIX/MADDPG 等 |

### 9.2 推荐联合仿真架构

```text
┌──────────────── Genesis World ────────────────┐
│ UAV/车辆动力学 │ 场景/遮挡 │ 相机/LiDAR/IMU │
└───────────────┬───────────────────────────────┘
                │ 节点位姿、遮挡、业务包、动作
        时间同步/消息桥（Python/ZeroMQ/gRPC/共享内存）
                │ 延迟、丢包、吞吐、SINR、邻居表
┌───────────────▼───────────────────────────────┐
│ ns-3 / OMNeT++ / 自定义无线信道与队列模型     │
└───────────────┬───────────────────────────────┘
                │ 通信约束下的观测与奖励
┌───────────────▼───────────────────────────────┐
│ 路径规划 / 控制 / MARL / 边缘卸载 / 资源调度  │
└───────────────────────────────────────────────┘
```

### 9.3 联合仿真的时间与数据接口

建议采用离散同步：Genesis 每 `dt_phys=0.01 s` 推进一次；网络模拟器每 `N` 个物理步推进一次，例如 `dt_net=0.1 s`。桥接消息至少包含：

```text
PhysicalState:
  sim_time, agent_id, position, orientation, velocity,
  line_of_sight, obstacle/material tags

NetworkResult:
  message_id, sender, receiver, delivered_time,
  latency, dropped, rssi/sinr, throughput

PolicyInput:
  local_sensor_obs, received_messages, age_of_information,
  neighbor_table, queue_state
```

对强化学习而言，网络结果不是日志，而应进入状态、转移或奖励：例如收到的是 `t-d` 时刻的邻车状态；丢包时保持上次值并增加 AoI；资源分配动作影响下一周期吞吐与能耗。

## 10. 面向课题组的三类代表性实验

### 10.1 空—地协同感知与避障

无人机从高处观测道路，将障碍物或事故位置发送给车辆；车辆本地 LiDAR 与无人机消息融合后规划。变量包括上行时延、丢包率、目标定位误差和遮挡。评价指标包括成功率、碰撞率、路径长度、到达时间、AoI 和通信负载。

### 10.2 多无人机临时中继与轨迹—通信联合优化

多无人机作为移动车联网中继，策略同时决定下一航点和信道/功率/用户关联。Genesis 负责位置、动力学、障碍与能耗近似，网络模拟器计算链路质量和吞吐；MARL 奖励综合覆盖率、吞吐、能耗、安全距离和公平性。

### 10.3 边缘推理卸载与语义通信

车辆或无人机采集 RGB/Depth，只传目标框、特征或关键帧。研究“传什么、何时传、传给谁”，比较本地推理、边缘卸载和协同感知。Genesis 可生成可重复的视觉与运动真值，网络层施加带宽、时延、丢包和排队约束。

## 11. 强化学习环境的建议构建方式

### 11.1 状态、动作和奖励

以多无人机中继为例：

| 类别 | 建议内容 |
|---|---|
| 局部状态 | 自身位姿/速度、电量、LiDAR/深度摘要、目标相对位置 |
| 通信状态 | 邻居表、AoI、队列长度、SINR、最近消息掩码 |
| 动作 | 期望速度/航点增量、功率、信道、关联或卸载比例 |
| 安全约束 | 最小间距、禁飞区、最大倾角/速度/加速度 |
| 奖励 | 任务完成 + 吞吐/覆盖 − 碰撞 − 能耗 − AoI − 控制抖动 |

### 11.2 算法选择

- 单无人机连续控制：PPO、SAC、TD3；
- 多智能体集中训练分散执行：MAPPO、MADDPG；
- 离散资源分配：QMIX、DQN 系列；
- 规划与学习混合：全局 A*/RRT* + RL 局部策略；
- 安全要求高：控制屏障函数、安全过滤器或 MPC shield 包裹策略输出。

算法选择应由动作空间、可观测性与信用分配决定，而不是由仿真器决定。

## 12. 建议的落地路线图

### 阶段 0：基线复现（1–2 周）

- 固化 Python、PyTorch、Genesis、Quadrants 和 rsl-rl 版本；
- 跑通 `interactive_drone.py`、`fly_route.py` 和小规模 `hover_train.py`；
- 建立无人机悬停误差、航点误差、仿真吞吐和可重复性报告；
- 单独验证无人机碰撞与接触行为。

### 阶段 1：无人机三维导航（2–4 周）

- 挂载深度相机/LiDAR/IMU；
- 实现体素地图、全局 A* 或 RRT*、局部避障；
- 保留 PID 内环，先获得稳定、可解释的规划基线；
- 引入场景、质量、风扰和传感器随机化。

### 阶段 2：空—地多智能体（3–6 周）

- 在同一场景加入车辆与多无人机；
- 统一 agent ID、消息格式、时钟和日志；
- 先用可配置延迟/丢包的 Python channel model，再接入专业网络模拟器。

### 阶段 3：网络联合仿真与 MARL（6–12 周）

- 接入 ns-3 或 OMNeT++；
- 实现锁步同步、消息队列、失败恢复和确定性种子；
- 构建 MAPPO/QMIX 等基线，比较无通信、理想通信和真实网络三组实验；
- 输出碰撞率、任务成功率、吞吐、AoI、能耗与实时因子。

## 13. 风险、限制与验证清单

1. **平台年轻、API 演进快**：锁定提交与依赖，记录版本，避免只写“latest”。
2. **无人机气动简化**：对强风、下洗、近地效应和高速飞行，需扩展模型。
3. **碰撞支持表述存在源码矛盾**：先做测试再开展避障结论。
4. **RL 示例并不等于成熟任务**：悬停环境的动作时延开关当前未真正生效，奖励和终止条件也需按论文任务重构。
5. **GPU 与渲染后端有平台差异**：训练、批量传感器和高质量渲染分别做性能评估。
6. **V2X 不是 Genesis 原生模块**：必须设计联合时钟、数据桥和网络侧真值，避免把随机 sleep 当成通信仿真。
7. **评估应覆盖系统指标**：除累计奖励外，至少报告碰撞率、成功率、轨迹质量、通信指标、算力成本和随机种子置信区间。

## 14. 对课题组的直接价值

- 用一套 Python 场景同时表达无人机、车辆、障碍物、传感器和控制回路；
- 利用并行环境快速构造训练数据与强化学习 rollout；
- 将“通信质量如何影响运动决策”落到真实状态转移，而不只做离线链路曲线；
- 以同一场景比较传统规划、学习控制和通信—控制联合优化；
- 通过 URDF、传感器和自定义实体/控制器复用现有机器人资产与算法；
- 形成可复现的组内平台：场景、网络配置、算法、随机种子和评测指标统一管理。

推荐的总体定位是：**Genesis 作为课题组空—地协同研究的物理与感知数字试验场，以专业网络仿真器补齐通信真实性，以规划/RL 框架完成决策闭环。**

## 15. 源码依据与延伸阅读

### 项目与架构

- `README.md`：项目背景、四层架构、求解器/渲染器/示例目录、安装方式。
- `pyproject.toml`：版本、Python 范围、核心与可选依赖。
- `genesis/__init__.py`：后端选择、精度、设备与运行时初始化。
- `genesis/engine/scene.py`：Scene、实体、传感器、build/reset/step 接口。
- `genesis/engine/simulator.py`：求解器、coupler 与传感器管理器的组合。

### 无人机

- `genesis/assets/urdf/drones/cf2x.urdf`：质量、惯量、旋翼位置、`kf/km`、碰撞几何。
- `genesis/options/morphs.py`：`gs.morphs.Drone` 参数、模型模式与限制说明。
- `genesis/engine/entities/rigid_entity/drone_entity.py`：RPM 接口与批量输入。
- `genesis/engine/solvers/rigid/abd/accessor.py`：旋翼力/反扭矩内核。
- `examples/drone/`：交互、预定义轨迹、PID、PPO 环境和训练/评估。

### 并行、传感器与本地扩展

- `examples/tutorials/parallel_simulation.py`：并行环境和按环境索引控制。
- `examples/rigid/domain_randomization.py`：质量、质心、惯量、摩擦随机化。
- `examples/rigid/heterogeneous_simulation.py`：批量环境中的几何变体。
- `genesis/options/sensors/options.py`：传感器历史、延迟和抖动。
- `examples/sensors/`：深度、LiDAR、IMU、接触、触觉等示例。
- `examples/mobile_robot/room_navigation_vision.py`：本分支的自然语言目标定位、RGB-D 与 A* 路径规划。

### 外部组件

- Genesis World：https://github.com/Genesis-Embodied-AI/genesis-world
- 官方文档：https://genesis-world.readthedocs.io/
- Quadrants：https://github.com/Genesis-Embodied-AI/quadrants
- Nyx：https://github.com/Genesis-Embodied-AI/genesis-nyx
- rsl-rl：https://github.com/leggedrobotics/rsl_rl
- ns-3：https://www.nsnam.org/
- OMNeT++：https://omnetpp.org/

## 附录 A：汇报时建议现场演示的顺序

1. `interactive_drone.py`：解释四旋翼差分转速；
2. 播放 `out/fly_route.mp4`：展示 PID 航点闭环；
3. 打开 `hover_env.py`：解释状态、动作、奖励和 8192 并行环境；
4. 展示移动机器人自然语言导航：说明感知—定位—A*—安全控制的可扩展范式；
5. 最后展示 Genesis + 网络模拟器架构图，强调能力边界和课题组建设路线。

## 附录 B：建议的首个课题组基准任务

**任务名称：通信受限的空—地协同目标搜索与安全到达。**

- 场景：1 架无人机、2–8 辆车、静态建筑/障碍和随机目标；
- 无人机：高空搜索并广播目标坐标；
- 车辆：融合本地 LiDAR 与无人机消息到达目标；
- 通信：理想、固定时延/丢包、ns-3 三档；
- 基线：无通信、周期广播、事件触发、学习式调度；
- 路径：A* + 局部安全层作为传统基线，MAPPO 作为联合优化基线；
- 指标：成功率、碰撞率、完成时间、路径长度、AoI、吞吐、能耗、实时因子。

该任务能同时覆盖课题组的无人机、车联网、路径规划和强化学习方向，又能清楚区分各模块贡献，适合作为 Genesis 引入后的第一条可复现实验链路。
