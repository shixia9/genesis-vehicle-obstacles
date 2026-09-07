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

`room_obstacle` 是展示场景：两个障碍物分布在蛇形航线两侧，小车会依次经过多个航点。若要专门演示中央障碍物触发的局部避障，可运行 `--scenario room_center_obstacle`；此时控制器执行“转向—横向离开—沿路线前进—恢复航点”的有限绕行动作。这两者都只是可解释的 MVP 实验，不等同于完整全局规划器。

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
