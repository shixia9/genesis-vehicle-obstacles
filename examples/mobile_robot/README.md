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

注意：第一版为了稳定验证闭环，车体和车轮是固定 Genesis 几何体，由差速运动学推进；`assets/diff_drive_car.urdf` 已保留，后续校准质量、轮轴、摩擦和接触后再切换到动态 URDF。

## 运行

在项目根目录执行：

```powershell
conda activate genesis
python examples/mobile_robot/room_navigation.py --steps 300
```

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
