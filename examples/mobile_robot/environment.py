"""Reusable reset/observe/step interface for the observable mobile-robot copy.

This module deliberately wraps room_navigation_observable instead of changing the
original room_navigation.py baseline. It is the first environment layer for
algorithm experiments; the underlying robot is still the deterministic kinematic
MVP vehicle.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

import genesis as gs
from genesis.utils.misc import set_random_seed

try:
    from .room_navigation_observable import (
        CarConfig,
        DifferentialDriveController,
        INITIAL_POSITION,
        Odometry,
        RoomConfig,
        build_scene,
        geometry_collision,
        read_sensor_observation,
        target_for_scenario,
        waypoints_for_scenario,
    )
except ImportError:
    from room_navigation_observable import (  # type: ignore[no-redef]
        CarConfig,
        DifferentialDriveController,
        INITIAL_POSITION,
        Odometry,
        RoomConfig,
        build_scene,
        geometry_collision,
        read_sensor_observation,
        target_for_scenario,
        waypoints_for_scenario,
    )


@dataclass(frozen=True)
class EnvironmentConfig:
    """Configuration for the programmatic simulation environment."""

    dt: float = 0.02
    seed: int = 7
    gpu: bool = False
    scenario: str = "default"
    max_steps: int = 1200
    render: bool = True


class MobileRobotEnv:
    """A small algorithm-facing environment around the observable MVP scene.

    The returned observation contains the robot RGB view, depth image, LiDAR,
    IMU, odometry, and robot pose. Actions use physical units:

        {"linear_velocity": meters_per_second,
         "angular_velocity": radians_per_second}

    The class intentionally does not depend on the hard-coded rule controller
    during step(). A controller, teleoperator, or learning policy can call
    reset(), read the observation, and supply its own action.
    """

    def __init__(self, config: EnvironmentConfig | None = None):
        self.config = config or EnvironmentConfig()
        if self.config.dt <= 0.0:
            raise ValueError("dt must be greater than zero")
        if self.config.max_steps <= 0:
            raise ValueError("max_steps must be greater than zero")

        scene_args = SimpleNamespace(
            gpu=self.config.gpu,
            dt=self.config.dt,
            seed=self.config.seed,
            vis=False,
            robot_view=False,
            scenario=self.config.scenario,
        )
        (
            self.scene,
            self.car,
            self.lidar,
            self.depth_camera,
            self.imu,
            self.overview_camera,
            self.robot_rgb_camera,
            self.obstacle_specs,
        ) = build_scene(scene_args)

        self.car_config = CarConfig()
        self.controller = DifferentialDriveController(
            self.car_config,
            waypoints=waypoints_for_scenario(self.config.scenario),
            enable_detour=self.config.scenario == "room_center_obstacle",
            dt=self.config.dt,
        )
        self.room_config = RoomConfig()
        self.odometry = Odometry(position=np.array(INITIAL_POSITION[:2], dtype=np.float32), yaw=0.0)
        self.step_index = 0
        self.closed = False
        self.reached = False
        self.collided = False
        self.last_action = {"linear_velocity": 0.0, "angular_velocity": 0.0}

        # Prime sensor caches before the first observation.
        self.scene.step()
        self.robot_rgb_camera.move_to_attach()

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        """Reset scene state and return the first observation."""
        self._ensure_open()
        if seed is not None:
            self.config = EnvironmentConfig(
                dt=self.config.dt,
                seed=seed,
                gpu=self.config.gpu,
                scenario=self.config.scenario,
                max_steps=self.config.max_steps,
                render=self.config.render,
            )
            # Keep NumPy, Torch and Genesis sensor noise on the same episode seed.
            set_random_seed(seed)

        self.scene.reset()
        self.car.reset()
        self.controller.reset()
        self.odometry = Odometry(position=np.array(INITIAL_POSITION[:2], dtype=np.float32), yaw=0.0)
        self.step_index = 0
        self.reached = False
        self.collided = False
        self.last_action = {"linear_velocity": 0.0, "angular_velocity": 0.0}
        self.scene.step()
        self.robot_rgb_camera.move_to_attach()
        return self.observe(render=self.config.render)

    def observe(self, render: bool | None = None) -> dict[str, Any]:
        """Read all robot sensors and return a synchronized observation."""
        self._ensure_open()
        if render is None:
            render = self.config.render
        return read_sensor_observation(
            step=self.step_index,
            dt=self.config.dt,
            car=self.car,
            lidar=self.lidar,
            depth_camera=self.depth_camera,
            imu=self.imu,
            odometry=self.odometry,
            overview_camera=self.overview_camera,
            robot_rgb_camera=self.robot_rgb_camera,
            render=render,
        )

    def rule_action(self, observation: Mapping[str, Any] | None = None) -> dict[str, float]:
        """Return one action from the bundled waypoint/LiDAR baseline controller."""
        if observation is None:
            observation = self.observe(render=False)
        linear_velocity, angular_velocity, _ = self.controller.command_from_observation(observation)
        return {
            "linear_velocity": float(linear_velocity),
            "angular_velocity": float(angular_velocity),
        }

    def step(
        self,
        action: Mapping[str, float] | Sequence[float],
        *,
        render: bool | None = None,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Apply one differential-drive action and return observation and diagnostics."""
        self._ensure_open()
        linear_velocity, angular_velocity = self._parse_action(action)
        previous_distance = self._distance_to_target()
        self.last_action = {
            "linear_velocity": linear_velocity,
            "angular_velocity": angular_velocity,
        }

        self.car.step(linear_velocity, angular_velocity, self.config.dt)
        self.odometry.integrate(linear_velocity, angular_velocity, self.config.dt)
        self.scene.step()
        self.robot_rgb_camera.move_to_attach()
        self.step_index += 1

        observation = self.observe(render=render)
        position = observation["robot_pose"][:3]
        yaw = float(observation["robot_pose"][3])
        current_distance = self._distance_to_target(position)
        self.reached = current_distance < 0.28
        self.collided = geometry_collision(position, self.room_config, self.obstacle_specs)

        terminated = bool(self.reached or self.collided)
        truncated = bool(self.step_index >= self.config.max_steps and not terminated)
        reward = float(previous_distance - current_distance)
        if self.reached:
            reward += 1.0
        if self.collided:
            reward -= 1.0

        termination_reason = (
            "reached"
            if self.reached
            else "collision"
            if self.collided
            else "timeout"
            if truncated
            else "running"
        )
        info = {
            "scenario": self.config.scenario,
            "step": self.step_index,
            "sim_time": self.step_index * self.config.dt,
            "position": position.copy(),
            "yaw": float(yaw),
            "distance_to_target": float(current_distance),
            "reached": self.reached,
            "collided": self.collided,
            "waypoint_idx": self.controller.waypoint_idx,
            "action": self.last_action.copy(),
            "timed_out": truncated,
            "termination_reason": termination_reason,
        }
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        """Release the Genesis scene and runtime resources."""
        if not self.closed:
            self.closed = True
            gs.destroy()

    def _parse_action(self, action: Mapping[str, float] | Sequence[float]) -> tuple[float, float]:
        if isinstance(action, Mapping):
            linear_velocity = float(action["linear_velocity"])
            angular_velocity = float(action["angular_velocity"])
        else:
            if len(action) != 2:
                raise ValueError("Sequence action must contain [linear_velocity, angular_velocity]")
            linear_velocity = float(action[0])
            angular_velocity = float(action[1])

        if not np.isfinite(linear_velocity) or not np.isfinite(angular_velocity):
            raise ValueError("Action values must be finite")
        linear_velocity = float(
            np.clip(linear_velocity, -self.car_config.max_linear_speed, self.car_config.max_linear_speed)
        )
        angular_velocity = float(
            np.clip(angular_velocity, -self.car_config.max_angular_speed, self.car_config.max_angular_speed)
        )
        return linear_velocity, angular_velocity

    def _distance_to_target(self, position: np.ndarray | None = None) -> float:
        if position is None:
            position = self.observe(render=False)["robot_pose"][:3]
        target = np.asarray(target_for_scenario(self.config.scenario), dtype=np.float32)
        return float(np.linalg.norm(target - position[:2]))

    def _ensure_open(self) -> None:
        if self.closed:
            raise RuntimeError("MobileRobotEnv is closed")


__all__ = ["EnvironmentConfig", "MobileRobotEnv"]
