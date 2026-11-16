# Genesis World 技术分享：从仿真内核到无人机智能决策

> 面向无人机深度强化学习路径规划、任务卸载与频谱分配课题组。对应同目录 PPTX。
> 本地基线：`pyproject.toml` 为 genesis-world 1.3.3、Python >=3.10,<3.14。上游 README 随版本变化，代码细节均指当前工作树。
> 资料：[Genesis AI 技术博客](https://www.genesis.ai/blog/the-role-of-simulation-in-scalable-robotics-genesis-world-10-and-the-path-forward)、[Genesis World GitHub](https://github.com/Genesis-Embodied-AI/genesis-world) 及本仓库文档。

## 1. 研究定位与能力边界

Genesis 官方博客强调闭环评估：策略动作改变环境，环境生成下一帧观测，再评价整个感知—决策—控制链。它比单次离线感知指标更适合发现失败模式。对课题组，Genesis 是可控的物理/感知实验底座，规划、卸载、无线资源算法与评价协议仍需自行实现。

下文用 **[原生]** 标识 Genesis 功能，**[仓库示例]** 标识附带代码，**[本地扩展]** 标识 `examples/mobile_robot`，**[拟建设]** 标识研究方案。Genesis 并无完整 5G/NR-V2X 协议、频谱调度、队列与边缘计算模拟。

## 2. 底层架构与一次仿真步

| 层 | 机制 | 对课题组的意义 |
|---|---|---|
| Simulation Interface | Scene、实体/资产、传感器、控制器、并行环境 | 构造空地场景，读观测、写动作、重置环境 |
| Physics | Rigid、FEM、MPM、PBD/SPH、求解器耦合 | 无人机与车辆主要使用刚体动力学；复杂物体可扩展 |
| Render | Nyx、Luisa、Pyrender | 图像输入与可视化；录制/高保真渲染增加开销 |
| Compiler | Quadrants 编译 Python kernel 到 CUDA/ROCm/Metal/Vulkan/CPU | 批量仿真的计算后端，性能须实测 |

```text
gs.init → Scene → add_entity/add_sensor → scene.build(n_envs=B)
策略读观测 → 计算动作 → 写实体控制 → scene.step()
solver 子步 → 传感器更新 → 奖励/终止 → envs_idx 局部重置
```

`genesis/engine/scene.py` 管理 `build/step`；`genesis/engine/simulator.py` 执行求解器子步并更新传感器。rigid-only 且无梯度时有专门的刚体步进路径。批量环境是独立状态的张量批，单次 `scene.step()` 推进全部环境；它并不保证单条轨迹加速 B 倍。应报告 env-steps/s、显存、传感器配置、渲染状态和硬件。

## 3. 无人机动力学与强化学习基线

`genesis/assets/urdf/drones/cf2x.urdf` 的 Crazyflie 2.X 机体质量约 0.027 kg，`K_F=3.16×10⁻¹⁰`、`K_M=7.94×10⁻¹²`。每个旋翼的局部力为 `F_i=[0,0,K_F RPM_i²]`，反扭矩为 `M_i=[0,0,s_i K_M RPM_i²]`，力臂贡献 `r_i×F_i`。悬停估算 `RPM≈√(mg/(4K_F))≈14,469`。`drone_entity.py` 的 `set_propellers_rpm` 支持批量输入且每仿真步只调用一次。此模型是集中力/力矩，不是 CFD、下洗或风场。

| 代码 | 已实现 | 不能推出 |
|---|---|---|
| `examples/drone/fly_route.py` | 级联 PID 航点飞行 | 三维避障或最优轨迹 |
| `examples/drone/hover_env.py` | 100 Hz、4 维 RPM 动作、17 维观测、奖励和终止 | 复杂任务规划 |
| `examples/drone/hover_train.py` | 外接 rsl-rl PPO，批量训练 | Genesis 内置完整 RL 算法库 |
| `examples/drone/hover_eval.py` | 评估与录制 | 真实无人机迁移 |

悬停环境的 17 维观测是相对目标 3、四元数 4、机体系线速度 3、角速度 3、上一动作 4；动作映射为 `RPM=(1+0.8a)×RPM_hover`。物理步长 0.01 s、刚体子步 0.005 s。`genesis/options/morphs.py` 对 Drone 标注当前不支持碰撞检查，因此避障研究须先用墙面撞击/接触/穿透实验验证，并以解析几何距离作为保守安全判据。

## 4. 课题 A：三维路径规划的 MDP

建议先由 PID/MPC 保持姿态内环，策略每 0.1 s 输出期望速度 `(v_x,v_y,v_z,ω_z)` 或短时航点，物理每 0.01 s 推进。这样可比较规划算法而不混入低层稳定性差异。

| 元素 | 最小实验定义 |
|---|---|
| 状态 | 相对目标、速度、航向、局部 LiDAR/深度摘要、上一动作；critic 可另用全局真值 |
| 动作 | 局部航点或期望速度，经稳定内环转为四旋翼 RPM |
| 奖励 | `w_p(d_{t-1}-d_t)+w_g 1_goal-w_c 1_collision-w_o proximity-w_e energy-w_j jerk` |
| 终止 | 到达、几何碰撞/近失、越界、超时、姿态失稳 |
| 基线 | 3D voxel A* / RRT* + PID，局部 MPC，PPO/SAC |

能耗可先用 `ΣRPM_i²Δt` 作代理，不能称为真实电池能耗。分四级实验：静态障碍；动态障碍与局部观测；质量/旋翼系数/风扰/传感器噪声随机化；多机共享信息。训练场景与 OOD 测试场景分离。评价成功率、碰撞和近失率、时间、路径长度、最小间距、能耗代理及样本效率；按障碍密度、噪声、动力偏差分别报告。

## 5. 课题 B：轨迹与任务卸载联合优化

Genesis 提供节点位置、运动和几何遮挡；任务到达、计算量、CPU/边缘队列、链路速率和能耗模型由课题组实现。任务 `j` 定义输入大小 `D_j`、计算周期 `C_j`、截止时间 `T_j`：

```text
T_local = Q_local + C_j/f_local
T_edge  = Q_tx + D_j/R_uplink + Q_edge + C_j/f_edge + T_return
miss_j  = 1[T_finish_j > T_j]
```

动作包括下一航点、卸载选择/比例、服务器选择。速率 `R_uplink` 必须由独立且可校准的信道/网络模型计算。奖励可为完成收益减去违约、飞行/通信/计算能耗与队列积压。基线为固定轨迹+贪心卸载、启发式轨迹+最短队列、联合 RL；消融位置、队列、遮挡信息。扫描任务到达率、带宽、边缘算力，报告完成率、95 分位时延、能耗和队列稳定性。

## 6. 课题 C：频谱分配及联合仿真

Genesis 每周期输出位姿与遮挡；无线模块计算 `g_{i,k}`。简化模型为：

```text
SINR_{i,k}=P_i g_{i,k}/(N₀B_k+Σ_{m≠i,m∈S_k}P_m g_{m,k})
R_{i,k}=B_k log₂(1+SINR_{i,k})·η_MCS
```

阴影、快衰落、波束、MCS/BLER 与协议开销需明确假设；标准协议研究应接 ns-3/OMNeT++。策略状态含位置、队列、上次 CSI、AoI；动作含信道、功率、用户关联/时隙；奖励组合吞吐、任务成功、干扰、功率、公平性。

建议从 `dt_phys=0.01 s`、`dt_net=dt_policy=0.1 s` 起步，属于**实验设计**而非平台限制。时刻 t 的策略只读已交付观测；Genesis 推进 10 步，网络层再根据同一时间戳推进发包、队列、干扰与 ACK，最后形成 t+Δt 观测/奖励。桥接消息至少含 `sim_time, agent_id, pose, velocity, visibility, task_id, queue_bytes, channel_id, tx_power, delivered_time`。禁止未来信息泄漏；物理、任务和网络随机种子分别记录。对照随机/轮询、最大 SINR、比例公平、优化器与 RL，报告总吞吐、5 分位速率、频谱效率、违约率和 Jain 公平指数。

## 7. 本地移动机器人 demo 与视频页

`examples/mobile_robot/` 是**本地扩展**。`room_navigation_vision.py` 演示中文指令 → YOLO-World/OWLv2 候选 → RGB-D 标定定位 → 静态二维 A* 路线 → 航点控制与 LiDAR 安全层 → 遥测和视觉落盘。视觉确定目标，控制与安全层独立。`room_navigation.py` 提供房间、差速车、LiDAR/深度/IMU、航点控制及 `summary.json`、`telemetry.csv`。

PPT 留有 16:9 视频框。当前 `examples/mobile_robot/` 中未发现可嵌入的录制视频；录制后在 PowerPoint 替换占位框即可。建议 60–90 秒：任务指令与场景 10 秒，检测/RGB-D 15 秒，A* 与 LiDAR 避障 25 秒，结果和日志 10 秒。最好同时展示俯视、机器人视角和时间戳。

迁移到无人机须把 2D 栅格换成 3D 体素、差速轮控制换成航点/速度到 RPM 内环、车体膨胀换成飞行器安全包络，并增加高度、姿态与动力学可达性。此 demo 证明闭环工程组织方式，不证明无人机避障或无线调度已实现。

## 8. 实施顺序与验收

| 阶段 | 交付与门槛 |
|---|---|
| 0 物理基线 | 复现 fly_route/hover；记录版本、种子、硬件；完成无人机撞墙/接触测试 |
| 1 三维导航 | 共用地图与基线；训练/测试场景分离；自动导出指标及失败轨迹 |
| 2 任务/队列 | 单任务时延可手算校验，任务与队列守恒 |
| 3 无线联合 | 同种子可复现，无未来信息泄漏，基线及消融齐全 |
| 4 真实性 | 标定参数、随机化、真实日志回放，报告 sim-to-real 差距 |

## 9. 从源码看状态推进、耦合与可微分

### 9.1 Scene 与 Simulator 的职责

`Scene` 负责对研究者暴露统一接口，保存实体、传感器、求解器和可视化设置。`Scene.build(n_envs=B)` 先调用环境并行化逻辑，再调用 `Simulator.build()`，最后重置场景状态。源码文档明确：`B>0` 时输入和返回状态的第一维是批量环境维；`env_spacing` 仅影响可视化排布，不改变仿真坐标。把 viewer 中环境间距当作实际智能体距离会得到错误的碰撞/通信结果。

`Scene.step()` 调用 `Simulator.step()`。当前源码在纯刚体且不求梯度时走刚体专用路径；一般路径调用 `process_input()`，然后逐子步执行 `coupler.preprocess → solver.substep_pre_coupling → coupler.couple → solver.substep_post_coupling`。最后清理外力并推进传感器管理器。**对于无人机实验，这意味着控制命令、物理步、传感器读数和奖励计算必须有明确先后顺序**。如果在 `scene.step()` 前后混读不同时间戳，会形成隐含一步超前或滞后。

### 9.2 编译、批量与可微分各解决什么问题

- Quadrants 将 Python kernel 编译到计算后端，解决数值内核的可移植与执行开销；它不自动并行化研究者写的 Python 任务队列或 ns-3 进程。
- `n_envs=B` 让物理状态按环境批量存储，适合 PPO 同时采样多条轨迹。若每步都将 LiDAR 张量转 NumPy、逐环境发网络请求，批量收益会被 Python 与数据传输抵消。
- 可微分仿真有助于参数/轨迹优化，但需要确认所用求解器、接触路径和操作支持梯度。本文三个课题的首版以常规前向仿真 + RL/优化器为主，避免把“平台有自动求导基础设施”误写成“任意全链路可微”。

### 9.3 训练与演示为什么分开

训练时通常关闭 GUI 与高频 RGB 录制，只保留策略真正使用的观测。演示时保留相机、俯视图和关键日志。报告吞吐时至少分三档测量：纯物理、物理+LiDAR/深度、物理+RGB/录制。比较算法时必须统一传感器与内环配置，否则训练速度和策略性能均不公平。

## 10. 悬停示例的精确状态机与可复用部分

当前 `HoverEnv.step(actions)` 顺序是：裁剪动作 → 映射 RPM → 设置目标显示实体 → `scene.step()` → 更新位姿/速度 → 检查到达与重新采样目标 → 终止/局部重置 → 奖励 → 新观测。目标位置来自批量张量，批量重置使用 `envs_idx`。这套代码可复用的是**环境生命周期与批量张量组织**；扩展路径规划时应新建观测、障碍、奖励和终止逻辑，而不是仅修改目标坐标范围。

`hover_train.py` 将 `simulate_action_latency=True` 放入配置，但 `HoverEnv.step` 当前直接执行本步 `actions`，未根据该标志切换上一动作。通信延迟或卸载返回延迟必须显式引入带时间戳的动作/消息队列；只设置该开关不会产生所需时延。一个可测试的动作队列应满足：

```text
issued_at = t
effective_at = t + sampled_delay
执行器只取 effective_at <= 当前仿真时刻的最新命令
丢包时保持上次命令，并累计 AoI；所有事件写入日志
```

若观测含邻机位置，还应区分真实状态 `x_t`、发送时刻状态 `x_{t-d}`、接收时刻 `t` 和策略实际读到的数据。训练 critic 可使用真值，但 actor 在训练与评估中只能读可交付信息，否则会高估分布式算法。

## 11. 三个课题可共用的实验接口

### 11.1 统一状态与动作格式

第一版可以在 Genesis 环境外建立一个联合仿真包装器，而不修改物理内核。建议每个智能体的观测至少包含：

```text
Observation:
  sim_time, agent_id
  proprioception: pose, body_velocity, battery_proxy
  perception: lidar/depth summary, target_relative_position
  communication: received_messages, message_age, neighbor_mask, last_csi
  compute: task_queue, deadline_histogram, edge_status

Action:
  motion: desired_velocity_or_waypoint
  offload: task_ids, destination, fraction
  radio: channel, slot, power, association
```

不是每个算法都必须联合输出三类动作。做单课题时固定其他两类动作的基线，可隔离研究变量。多智能体实验统一智能体 ID、时间戳和坐标系，并定义离场/重置后队列和未完成任务如何处理。

### 11.2 统一时钟的伪代码

```python
for decision_step in range(horizon):
    obs = compose_observation(physical_state, delivered_packets, task_queues)
    action = policy(obs)
    queue_new_commands(action, issued_at=sim_time)

    for _ in range(physics_steps_per_decision):
        execute_due_motion_commands(sim_time)
        scene.step()
        sim_time += dt_phys

    snapshot = read_pose_visibility_sensors(scene, sim_time)
    network_result = network.advance_until(sim_time, snapshot, action.radio)
    task_result = compute.advance_until(sim_time, network_result, action.offload)
    reward, done = evaluate(snapshot, network_result, task_result)
    log_transition(obs, action, reward, done, sim_time)
```

这是系统设计伪代码，`network.advance_until` 和 `compute.advance_until` 并非 Genesis API。若希望网络侧在物理子步之间处理事件，应缩小同步步长或实现事件屏障，不能把未来时刻的位置一次性提供给网络侧。

### 11.3 对照实验与消融矩阵

| 问题 | 固定条件 | 主要改变 | 归因目标 |
|---|---|---|---|
| 三维导航 | 动力学、地图、内环 | A* / RRT* / MPC / RL | 学习策略是否优于强规划基线 |
| 卸载 | 任务种子、边缘算力 | 固定轨迹/联合轨迹、不同卸载器 | 轨迹—链路耦合是否带来增益 |
| 频谱 | 节点轨迹、信道参数 | 随机/公平/优化/RL 调度 | 物理位置/遮挡信息是否有用 |
| 跨场景泛化 | 训练数据与预算 | 障碍、噪声、用户密度、信道偏差 | 算法是否依赖特定仿真设置 |

所有结果至少重复多个随机种子，报告均值与离散程度，并保存失败轨迹。对任务完成率等比例指标说明样本量和置信区间；对极端延迟报告 P95/P99 与截止违约率。若多智能体数量改变，明确策略是否共享参数、是否需要重新训练。

## 12. Demo 的现场讲解与录制检查

### 12.1 可直接运行的基线

在仓库根目录运行最小闭环：

```bash
python examples/mobile_robot/room_navigation.py --steps 1200 --save-images
```

结果默认写入 `out/mobile_robot/`，包括 `summary.json`、`telemetry.csv` 及可选 RGB/深度帧。带自然语言目标的 demo 还依赖本地 YOLO-World 权重、视觉参数和设备配置；完整命令以 `examples/mobile_robot/README.md` 的“最小自然语言目标 Demo”章节为准，录制前先确认模型文件存在。

### 12.2 视频页建议呈现的证据

视频画面左侧建议放俯视路线和机器人，右侧放机器人 RGB/检测框，下方只保留当前目标、最近 LiDAR 距离、当前航点与时间戳。讲解时指出：视觉目标锁定后，路线从搜索路线切换到目标附近路径；障碍安全层在视觉失误时仍独立生效。结束时打开 `summary.json` 的 `navigation_mode`、`active_waypoints`、`termination_reason`，避免只靠画面主观判断任务成功。

该 demo 的二维 A* 使用障碍膨胀、禁斜穿角点和航点化简，适合作为无人机三维占据规划的工程原型。无人机版必须另外验证垂直方向自由空间、机体包络、最大速度/加速度、旋翼响应和动态障碍预测。

## 13. 关键风险与实验有效性

1. **碰撞语义**：Drone morph 的说明与 URDF collision 几何并存，不能仅凭资产标签认定碰撞有效。做静态墙、斜撞、不同速度和不同子步长测试，并记录是否触发预期接触/终止。
2. **网络真实性**：简化 Shannon 速率只提供算法原型；若论文讨论 NR-V2X 结论，必须加入协议/调度/重传模型并标定参数。
3. **仿真到真实差距**：质量、旋翼推力、IMU 偏置、延迟、风扰、视觉外观均可能偏离真实系统。报告参数来源、随机化区间和外部分布测试。
4. **奖励漏洞**：只奖励到目标距离会诱导贴障或穿障；只奖励吞吐会牺牲边缘用户。用硬约束、近失统计和分布尾部指标共同检查。
5. **评价泄漏**：集中 critic 使用真值可以，但 actor 不能在评估时读取真实障碍图、未来链路或未交付消息。日志同时保存物理真值与策略观测，便于审计。

## 14. 资料与源码索引

- [Genesis AI 技术博客](https://www.genesis.ai/blog/the-role-of-simulation-in-scalable-robotics-genesis-world-10-and-the-path-forward)：闭环评估、Nyx、Quadrants、未来方向。
- [Genesis World GitHub README](https://github.com/Genesis-Embodied-AI/genesis-world)：四层架构、示例与安装。
- `genesis/engine/scene.py`、`genesis/engine/simulator.py`：场景构建、求解器与传感器步进。
- `genesis/engine/entities/rigid_entity/drone_entity.py`、`genesis/assets/urdf/drones/cf2x.urdf`：旋翼接口与物理参数。
- `examples/drone/hover_env.py`、`hover_train.py`、`fly_route.py`：PPO/PID 基线。
- `examples/mobile_robot/README.md`、`room_navigation.py`、`room_navigation_vision.py`：本地 demo。
