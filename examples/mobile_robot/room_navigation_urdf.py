"""Minimal dynamic-URDF mobile robot experiment.

This file is deliberately separate from ``room_navigation.py`` and
``room_navigation_observable.py``.  It uses the custom differential-drive
vehicle in ``assets/diff_drive_car.urdf`` as a dynamic Genesis entity:

    observation -> waypoint controller -> wheel velocity targets -> physics
                   -> robot-mounted sensors -> next observation

The action interface remains linear/angular velocity so that the controller is
easy to compare with the kinematic MVP.  Before each physics step, the action
is converted to left/right wheel angular velocities.  The odometry in this
experiment is wheel-encoder odometry derived from the two URDF joint positions.

Headless smoke test::

    python examples/mobile_robot/room_navigation_urdf.py --steps 300

Sensor and first-person camera test::

    python examples/mobile_robot/room_navigation_urdf.py \
        --vis --robot-view --save-sensors --scenario room_obstacle --steps 1200

This is the smallest dynamic-URDF integration experiment.  It is not intended
to replace the stable kinematic baseline yet; the URDF's contact, friction and
caster parameters still need calibration for high-fidelity vehicle behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# Set this before importing the observable helper: it also initializes Genesis'
# cache environment variable at import time.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("QD_OFFLINE_CACHE_FILE_PATH", str(PROJECT_ROOT / ".quadrants_cache"))

import genesis as gs
from genesis.utils import misc as genesis_misc
from genesis.utils.geom import euler_to_quat, pos_lookat_up_to_T
from genesis.utils.misc import tensor_to_array

try:
    from .room_navigation_observable import (
        CarConfig,
        DifferentialDriveController,
        INITIAL_POSITION,
        RoomConfig,
        SCENARIO_OBSTACLE_SPECS,
        add_room,
        geometry_collision,
        obstacle_specs_for_scenario,
        read_sensor_observation,
        target_for_scenario,
        waypoints_for_scenario,
    )
except ImportError:  # Running this file directly from examples/mobile_robot.
    from room_navigation_observable import (
        CarConfig,
        DifferentialDriveController,
        INITIAL_POSITION,
        RoomConfig,
        SCENARIO_OBSTACLE_SPECS,
        add_room,
        geometry_collision,
        obstacle_specs_for_scenario,
        read_sensor_observation,
        target_for_scenario,
        waypoints_for_scenario,
    )


URDF_PATH = Path(__file__).resolve().parent / "assets" / "diff_drive_car.urdf"
INITIAL_URDF_POSITION = (INITIAL_POSITION[0], INITIAL_POSITION[1], 0.0)

# The dynamic body has small contact and odometry errors that the kinematic
# showcase route does not need to tolerate.  Keep the obstacles in the same
# room, but give the URDF vehicle a little more clearance around them.
URDF_SHOWCASE_WAYPOINTS = (
    INITIAL_POSITION[:2],
    (-2.8, 1.4),
    (0.1, 1.4),
    (0.1, -0.8),
    (2.8, -0.8),
    (2.8, 1.7),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1200, help="Number of simulation steps (default: 1200).")
    parser.add_argument("--dt", type=float, default=0.02, help="Simulation time step in seconds (default: 0.02).")
    parser.add_argument("--gpu", action="store_true", help="Use the Genesis GPU backend when available.")
    parser.add_argument("--vis", action="store_true", help="Open the Genesis third-person viewer.")
    parser.add_argument(
        "--robot-view",
        action="store_true",
        help="Open the RGB window attached to the URDF base_link.",
    )
    parser.add_argument("--save-images", action="store_true", help="Save overview, robot RGB and depth frames.")
    parser.add_argument(
        "--save-sensors",
        action="store_true",
        help="Save synchronized RGB/depth references and LiDAR/IMU/odometry records.",
    )
    parser.add_argument(
        "--scenario",
        choices=tuple(SCENARIO_OBSTACLE_SPECS),
        default="room_basic",
        help="Room obstacle preset (default: room_basic).",
    )
    parser.add_argument("--image-every", type=int, default=25, help="Capture one image every N steps.")
    parser.add_argument("--log-every", type=int, default=10, help="Print one telemetry row every N steps.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed used by Genesis and NumPy.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out/mobile_robot_urdf"),
        help="Directory for telemetry, summary and optional sensor frames.",
    )
    return parser.parse_args()


@dataclass
class DynamicDifferentialDriveCar:
    """Adapter from body-level actions to the two URDF wheel joints."""

    entity: Any
    base_link: Any
    wheel_radius: float
    wheel_base: float
    initial_position: np.ndarray

    def __post_init__(self) -> None:
        left_joint = self.entity.get_joint("left_wheel_joint")
        right_joint = self.entity.get_joint("right_wheel_joint")
        self.left_dof = int(left_joint.dofs_idx_local[0])
        self.right_dof = int(right_joint.dofs_idx_local[0])
        self.wheel_dofs = [self.left_dof, self.right_dof]
        self.initial_quat = np.asarray(euler_to_quat((0.0, 0.0, 0.0)), dtype=np.float32)

    def configure_velocity_control(self) -> None:
        """Configure conservative wheel-joint velocity control for the MVP."""
        self.entity.set_dofs_kv(8.0, dofs_idx_local=self.wheel_dofs)
        self.entity.set_dofs_force_range(
            [-20.0, -20.0],
            [20.0, 20.0],
            dofs_idx_local=self.wheel_dofs,
        )

    def wheel_targets(self, linear_velocity: float, angular_velocity: float) -> np.ndarray:
        """Convert chassis velocity to URDF joint angular velocities.

        The authored URDF uses a +Y wheel axis, and this model's contact
        convention makes a positive joint velocity roll toward +X.  The
        resulting sign is checked against the actual base-link motion in the
        smoke test below.
        """
        left_linear = linear_velocity - 0.5 * self.wheel_base * angular_velocity
        right_linear = linear_velocity + 0.5 * self.wheel_base * angular_velocity
        return np.asarray(
            [left_linear, right_linear],
            dtype=np.float32,
        ) / self.wheel_radius

    def apply_action(self, linear_velocity: float, angular_velocity: float) -> np.ndarray:
        targets = self.wheel_targets(linear_velocity, angular_velocity)
        self.entity.control_dofs_velocity(targets, dofs_idx_local=self.wheel_dofs)
        return targets

    def stop(self) -> None:
        self.entity.control_dofs_velocity(np.zeros(2, dtype=np.float32), dofs_idx_local=self.wheel_dofs)

    def reset(self) -> None:
        self.entity.set_pos(self.initial_position)
        self.entity.set_quat(self.initial_quat)
        self.entity.set_dofs_position(
            np.zeros(2, dtype=np.float32),
            dofs_idx_local=self.wheel_dofs,
            zero_velocity=True,
        )
        self.stop()

    def get_pos(self):
        return self.entity.get_pos()

    def get_quat(self):
        return self.entity.get_quat()

    def get_wheel_positions(self) -> np.ndarray:
        return np.asarray(tensor_to_array(self.entity.get_dofs_position(self.wheel_dofs))).reshape(2)

    def get_wheel_velocities(self) -> np.ndarray:
        return np.asarray(tensor_to_array(self.entity.get_dofs_velocity(self.wheel_dofs))).reshape(2)


@dataclass
class WheelOdometry:
    """Planar odometry integrated from the two continuous wheel joints."""

    car: DynamicDifferentialDriveCar
    position: np.ndarray
    yaw: float = 0.0

    def __post_init__(self) -> None:
        self.previous_wheel_positions = self.car.get_wheel_positions().copy()

    def reset(self) -> None:
        self.position[:] = self.car.get_pos()[:2]
        self.yaw = 0.0
        self.previous_wheel_positions = self.car.get_wheel_positions().copy()

    def update(self) -> None:
        current = self.car.get_wheel_positions()
        delta = current - self.previous_wheel_positions
        self.previous_wheel_positions = current.copy()

        # For this authored URDF, positive joint motion means forward +X.
        left_distance = float(delta[0]) * self.car.wheel_radius
        right_distance = float(delta[1]) * self.car.wheel_radius
        linear_distance = 0.5 * (left_distance + right_distance)
        angular_distance = (right_distance - left_distance) / self.car.wheel_base

        self.position[0] += linear_distance * math.cos(self.yaw + 0.5 * angular_distance)
        self.position[1] += linear_distance * math.sin(self.yaw + 0.5 * angular_distance)
        self.yaw = (self.yaw + angular_distance + math.pi) % (2.0 * math.pi) - math.pi


def build_scene(args: argparse.Namespace):
    backend = gs.gpu if args.gpu else gs.cpu
    try:
        gs.init(backend=backend, precision="32", logging_level="warning", seed=args.seed)
    except StopIteration:
        # Some macOS cpuinfo releases expose only arch/count and omit all three
        # names Genesis 1.3.3 checks.  Supply a local fallback for initialization
        # only; this does not install or alter any dependency.
        original_get_cpu_info = genesis_misc.cpuinfo.get_cpu_info
        cpu_info = dict(original_get_cpu_info())
        cpu_info.setdefault("brand_raw", platform.machine() or "unknown-cpu")
        genesis_misc.cpuinfo.get_cpu_info = lambda: cpu_info
        try:
            gs.init(backend=backend, precision="32", logging_level="warning", seed=args.seed)
        finally:
            genesis_misc.cpuinfo.get_cpu_info = original_get_cpu_info

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=args.dt, substeps=2),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(6.5, -7.0, 5.8),
            camera_lookat=(0.0, 0.0, 0.45),
            camera_fov=45,
        ),
        profiling_options=gs.options.ProfilingOptions(show_FPS=False),
        show_viewer=args.vis,
    )

    obstacle_specs = obstacle_specs_for_scenario(args.scenario)
    add_room(scene, RoomConfig(), obstacle_specs)

    target = target_for_scenario(args.scenario)
    scene.add_entity(
        gs.morphs.Cylinder(height=0.025, radius=0.22, pos=(*target, 0.013), fixed=True),
        surface=gs.surfaces.Emission(color=(0.1, 1.0, 0.2)),
    )

    robot = scene.add_entity(
        gs.morphs.URDF(
            file=str(URDF_PATH),
            pos=INITIAL_URDF_POSITION,
            euler=(0.0, 0.0, 0.0),
            fixed=False,
        ),
        material=gs.materials.Rigid(friction=1.0),
        name="custom_diff_drive_car",
    )
    base_link = robot.get_link("base_link")
    car = DynamicDifferentialDriveCar(
        entity=robot,
        base_link=base_link,
        wheel_radius=0.12,
        wheel_base=0.58,
        initial_position=np.asarray(INITIAL_URDF_POSITION, dtype=np.float32),
    )

    lidar_angles = np.linspace(-180.0, 180.0, 72, endpoint=False).tolist()
    lidar = scene.add_sensor(
        gs.sensors.Lidar(
            pattern=gs.sensors.SphericalPattern(angles=(lidar_angles, [0.0])),
            entity_idx=robot.idx,
            link_idx_local=base_link.idx_local,
            pos_offset=(0.60, 0.0, 0.23),
            max_range=6.0,
            return_points=False,
        )
    )
    depth_camera = scene.add_sensor(
        gs.sensors.DepthCamera(
            pattern=gs.sensors.DepthCameraPattern(res=(128, 96), fov_horizontal=90.0),
            entity_idx=robot.idx,
            link_idx_local=base_link.idx_local,
            pos_offset=(0.60, 0.0, 0.23),
            max_range=6.0,
            return_world_frame=False,
        )
    )
    imu = scene.add_sensor(
        gs.sensors.IMU(
            entity_idx=robot.idx,
            link_idx_local=base_link.idx_local,
            pos_offset=(0.0, 0.0, 0.23),
            acc_noise=(0.01, 0.01, 0.01),
            gyro_noise=(0.005, 0.005, 0.005),
            delay=args.dt,
        )
    )

    need_rgb_render = args.vis or args.robot_view or args.save_images or args.save_sensors
    overview_camera = None
    robot_rgb_camera = None
    if need_rgb_render:
        overview_camera = scene.add_camera(
            res=(256, 192),
            pos=(0.0, 0.0, 10.0),
            lookat=(0.0, 0.0, 0.0),
            up=(0.0, 1.0, 0.0),
            fov=50,
            GUI=args.vis and not args.robot_view,
        )
        robot_rgb_camera = scene.add_camera(
            res=(256, 192),
            pos=(-2.2, -1.8, 0.45),
            lookat=(-1.2, -1.8, 0.35),
            up=(0.0, 0.0, 1.0),
            fov=90,
            GUI=args.robot_view,
        )

    scene.build()
    car.configure_velocity_control()
    if robot_rgb_camera is not None:
        robot_camera_offset = pos_lookat_up_to_T(
            np.array((0.60, 0.0, 0.23), dtype=np.float32),
            np.array((1.60, 0.0, 0.23), dtype=np.float32),
            np.array((0.0, 0.0, 1.0), dtype=np.float32),
        )
        robot_rgb_camera.attach(base_link, robot_camera_offset)
        robot_rgb_camera.move_to_attach()
    return scene, car, lidar, depth_camera, imu, overview_camera, robot_rgb_camera, obstacle_specs


def navigation_waypoints(scenario: str) -> tuple[tuple[float, float], ...]:
    if scenario == "room_obstacle":
        return URDF_SHOWCASE_WAYPOINTS
    return waypoints_for_scenario(scenario)


def save_rgb(rgb: np.ndarray, path: Path) -> None:
    image = np.asarray(rgb)
    if image.ndim == 4:
        image = image[0]
    if image.dtype != np.uint8:
        image = np.clip(image * 255.0 if image.max() <= 1.0 else image, 0, 255).astype(np.uint8)
    Image.fromarray(image).save(path)


def serialize_observation(observation: dict[str, Any], frame_id: int | None) -> dict[str, Any]:
    return {
        "rgb_frame": frame_id,
        "robot_rgb_frame": frame_id,
        "depth_frame": frame_id,
        "rgb_path": f"rgb/frame_{frame_id:05d}.png" if frame_id is not None else None,
        "robot_rgb_path": f"rgb_robot/frame_{frame_id:05d}.png" if frame_id is not None else None,
        "depth_path": f"depth/frame_{frame_id:05d}.png" if frame_id is not None else None,
        "lidar": np.asarray(observation["lidar"]).tolist(),
        "imu_acc": np.asarray(observation["imu_acc"]).tolist(),
        "imu_gyro": np.asarray(observation["imu_gyro"]).tolist(),
        "odom_pose": np.asarray(observation["odom_pose"]).tolist(),
        "robot_pose": np.asarray(observation["robot_pose"]).tolist(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.steps <= 0 or args.dt <= 0.0:
        raise ValueError("--steps and --dt must be greater than zero")
    if args.image_every <= 0 or args.log_every <= 0:
        raise ValueError("--image-every and --log-every must be greater than zero")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_sensor_frames = args.save_images or args.save_sensors
    if save_sensor_frames:
        for directory in ("rgb", "rgb_robot", "depth"):
            (args.output_dir / directory).mkdir(exist_ok=True)

    scene, car, lidar, depth_camera, imu, overview_camera, robot_rgb_camera, obstacle_specs = build_scene(args)
    controller = DifferentialDriveController(
        CarConfig(),
        waypoints=navigation_waypoints(args.scenario),
        enable_detour=args.scenario in ("room_obstacle", "room_center_obstacle"),
        dt=args.dt,
    )
    odometry = WheelOdometry(
        car=car,
        position=np.asarray(INITIAL_POSITION[:2], dtype=np.float32),
    )
    room_config = RoomConfig()
    telemetry: list[dict[str, Any]] = []
    sensor_records: list[dict[str, Any]] = []
    reached = False
    collided = False
    timed_out = False
    last_lidar = np.full(72, 6.0, dtype=np.float32)
    observation_step = 0

    try:
        # Prime physics and sensor caches, then start encoder odometry at the settled pose.
        scene.step()
        odometry.reset()
        observation = read_sensor_observation(
            step=0,
            dt=args.dt,
            car=car,
            lidar=lidar,
            depth_camera=depth_camera,
            imu=imu,
            odometry=odometry,
        )

        for step in range(args.steps):
            linear_speed, angular_speed, reached = controller.command_from_observation(observation)
            action = {
                "linear_velocity": float(linear_speed),
                "angular_velocity": float(angular_speed),
            }
            wheel_targets = car.apply_action(linear_speed, angular_speed)
            scene.step()
            odometry.update()
            observation_step = step + 1
            capture_frame = (
                (args.save_images or args.save_sensors or args.vis or args.robot_view)
                and step % args.image_every == 0
            )
            observation = read_sensor_observation(
                step=observation_step,
                dt=args.dt,
                car=car,
                lidar=lidar,
                depth_camera=depth_camera,
                imu=imu,
                odometry=odometry,
                overview_camera=overview_camera,
                robot_rgb_camera=robot_rgb_camera,
                render=capture_frame,
            )
            last_lidar = np.asarray(observation["lidar"], dtype=np.float32)
            position = np.asarray(observation["robot_pose"][:3], dtype=np.float32)
            yaw = float(observation["robot_pose"][3])
            frame_id = observation_step if capture_frame else None

            if capture_frame and save_sensor_frames:
                save_rgb(observation["overview_rgb"], args.output_dir / "rgb" / f"frame_{frame_id:05d}.png")
                save_rgb(observation["rgb"], args.output_dir / "rgb_robot" / f"frame_{frame_id:05d}.png")
                depth_image = np.clip(np.asarray(observation["depth"]) / 6.0 * 255.0, 0, 255).astype(np.uint8)
                Image.fromarray(depth_image).save(args.output_dir / "depth" / f"frame_{frame_id:05d}.png")

            distance_to_target = float(np.linalg.norm(np.asarray(controller.target) - position[:2]))
            distance_to_waypoint = float(np.linalg.norm(np.asarray(controller.current_target) - position[:2]))
            collided = geometry_collision(position, room_config, obstacle_specs)
            wheel_positions = car.get_wheel_positions()
            wheel_velocities = car.get_wheel_velocities()

            if args.save_sensors:
                sensor_records.append(
                    {
                        "step": observation_step,
                        "sim_time": observation["sim_time"],
                        "observation": serialize_observation(observation, frame_id),
                        "action": action,
                        "wheel_targets": wheel_targets.tolist(),
                        "wheel_positions": wheel_positions.tolist(),
                        "wheel_velocities": wheel_velocities.tolist(),
                        "action_step": step,
                        "info": {
                            "scenario": args.scenario,
                            "waypoint_idx": controller.waypoint_idx,
                            "detour_phase": controller.detour_phase,
                            "distance_to_target": distance_to_target,
                            "distance_to_waypoint": distance_to_waypoint,
                            "reached": bool(reached),
                            "collided": bool(collided),
                        },
                    }
                )

            lidar_angles = np.linspace(-math.pi, math.pi, len(last_lidar), endpoint=False)
            front_mask = np.abs(lidar_angles) < math.radians(25.0)
            front_min_lidar = float(np.min(last_lidar[front_mask]))
            if step % args.log_every == 0 or reached or collided:
                telemetry.append(
                    {
                        "step": observation_step,
                        "time": observation["sim_time"],
                        "x": float(position[0]),
                        "y": float(position[1]),
                        "yaw": yaw,
                        "odom_x": float(odometry.position[0]),
                        "odom_y": float(odometry.position[1]),
                        "odom_yaw": float(odometry.yaw),
                        "distance_to_target": distance_to_target,
                        "distance_to_waypoint": distance_to_waypoint,
                        "waypoint_idx": controller.waypoint_idx,
                        "linear_command": linear_speed,
                        "angular_command": angular_speed,
                        "left_wheel_target": float(wheel_targets[0]),
                        "right_wheel_target": float(wheel_targets[1]),
                        "left_wheel_velocity": float(wheel_velocities[0]),
                        "right_wheel_velocity": float(wheel_velocities[1]),
                        "front_min_lidar": front_min_lidar,
                        "robot_rgb_frame": frame_id,
                        "depth_frame": frame_id,
                        "imu_acc_norm": float(np.linalg.norm(observation["imu_acc"])),
                        "imu_gyro_norm": float(np.linalg.norm(observation["imu_gyro"])),
                        "reached": bool(reached),
                        "collided": bool(collided),
                    }
                )
                print(
                    f"step={observation_step:04d} pos=({position[0]:+.2f},{position[1]:+.2f}) "
                    f"odom=({odometry.position[0]:+.2f},{odometry.position[1]:+.2f}) "
                    f"target_dist={distance_to_target:.2f} waypoint={controller.waypoint_idx} "
                    f"lidar_front_min={front_min_lidar:.2f} "
                    f"wheel=({wheel_velocities[0]:+.2f},{wheel_velocities[1]:+.2f})"
                )

            if reached or collided:
                break
        else:
            timed_out = True

        telemetry_path = args.output_dir / "telemetry.csv"
        if telemetry:
            with telemetry_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(telemetry[0]))
                writer.writeheader()
                writer.writerows(telemetry)

        sensor_log_path = args.output_dir / "sensor_observations.jsonl"
        if args.save_sensors:
            with sensor_log_path.open("w", encoding="utf-8") as file:
                for record in sensor_records:
                    file.write(json.dumps(record) + "\n")

        final_position = np.asarray(tensor_to_array(car.get_pos())).reshape(3)
        summary = {
            "steps": int(observation_step),
            "sim_time": float(observation_step * args.dt),
            "reached": bool(reached),
            "collided": bool(collided),
            "timed_out": bool(timed_out),
            "termination_reason": "reached" if reached else "collision" if collided else "timeout",
            "final_x": float(final_position[0]),
            "final_y": float(final_position[1]),
            "final_z": float(final_position[2]),
            "odom_x": float(odometry.position[0]),
            "odom_y": float(odometry.position[1]),
            "target_x": target_for_scenario(args.scenario)[0],
            "target_y": target_for_scenario(args.scenario)[1],
            "front_min_lidar": float(np.min(last_lidar[front_mask])),
            "wheel_dofs": car.wheel_dofs,
            "urdf": str(URDF_PATH),
            "backend": "gpu" if args.gpu else "cpu",
            "telemetry_file": str(telemetry_path),
            "sensor_log_file": str(sensor_log_path) if args.save_sensors else None,
            "robot_rgb_camera": "attached_to_base_link",
        }
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
    finally:
        car.stop()
        gs.destroy()


def main() -> None:
    args = parse_args()
    summary = run(args)
    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
