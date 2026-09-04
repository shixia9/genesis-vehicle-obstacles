"""MVP mobile-robot experiment: room, differential-drive car, sensors, and waypoint control.

The example intentionally keeps the vehicle model simple. It demonstrates the complete simulation loop before a
learning policy is introduced:

    scene -> physics -> sensors -> rule controller -> wheel velocity commands -> telemetry

Run without a viewer for a smoke test::

    python examples/mobile_robot/room_navigation.py --steps 300

Run with the CUDA backend and save RGB/depth frames::

    python examples/mobile_robot/room_navigation.py --gpu --steps 1200 --save-images

Add ``--vis`` to open the Genesis viewer. The RGB camera has its own optional GUI window when ``--vis`` is enabled.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

# Some Windows installations leave Quadrants' default C:\quadrants_cache read-only.
# Keep this experiment self-contained and reproducible by placing its kernel cache
# under the ignored project output directory.
os.environ.setdefault("QD_OFFLINE_CACHE_FILE_PATH", str(Path("out/quadrants_cache").resolve()))

import genesis as gs
from genesis.utils.geom import euler_to_quat
from genesis.utils.misc import tensor_to_array


@dataclass(frozen=True)
class RoomConfig:
    length: float = 8.0
    width: float = 6.0
    wall_height: float = 2.5
    wall_thickness: float = 0.12


@dataclass(frozen=True)
class CarConfig:
    wheel_radius: float = 0.12
    wheel_base: float = 0.58
    max_linear_speed: float = 0.75
    max_angular_speed: float = 1.5
    max_wheel_speed: float = 8.0


OBSTACLE_SPECS: tuple[tuple[tuple[float, float], tuple[float, float, float]], ...] = (
    ((-0.8, -0.5), (0.8, 0.8, 0.7)),
    ((0.9, 0.9), (0.65, 0.65, 0.9)),
    ((2.0, -1.2), (0.9, 0.55, 0.6)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1200, help="Number of simulation steps (default: 1200).")
    parser.add_argument("--dt", type=float, default=0.02, help="Simulation time step in seconds (default: 0.02).")
    parser.add_argument("--gpu", action="store_true", help="Use the Genesis CUDA backend.")
    parser.add_argument("--vis", action="store_true", help="Open the Genesis 3D viewer and RGB camera window.")
    parser.add_argument("--save-images", action="store_true", help="Save RGB and depth frames to the output folder.")
    parser.add_argument("--image-every", type=int, default=25, help="Save/render one image every N steps.")
    parser.add_argument("--log-every", type=int, default=10, help="Print and record telemetry every N steps.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed used by Genesis and NumPy.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out/mobile_robot"),
        help="Directory for telemetry, summary, and optional image frames.",
    )
    return parser.parse_args()


def add_room(scene: gs.Scene, config: RoomConfig) -> list[object]:
    """Add a parameterized room with a front doorway and fixed obstacles."""
    half_length = config.length / 2.0
    half_width = config.width / 2.0
    z = config.wall_height / 2.0
    wall = config.wall_thickness
    # Emission surfaces keep the CPU rasterizer's recorded overview frames readable
    # without relying on a backend-specific light configuration.
    room_surface = gs.surfaces.Emission(color=(0.62, 0.67, 0.75))
    obstacle_surface = gs.surfaces.Emission(color=(0.90, 0.25, 0.08))

    entities = [scene.add_entity(gs.morphs.Plane(), surface=gs.surfaces.Emission(color=(0.30, 0.34, 0.40)))]
    entities.extend(
        [
            scene.add_entity(
                gs.morphs.Box(size=(wall, config.width, config.wall_height), pos=(-half_length, 0, z), fixed=True),
                surface=room_surface,
            ),
            scene.add_entity(
                gs.morphs.Box(size=(wall, config.width, config.wall_height), pos=(half_length, 0, z), fixed=True),
                surface=room_surface,
            ),
            scene.add_entity(
                gs.morphs.Box(size=(config.length, wall, config.wall_height), pos=(0, -half_width, z), fixed=True),
                surface=room_surface,
            ),
        ]
    )

    # Leave a 1.6 m opening in the positive-x wall to make the room visually read as an indoor environment.
    doorway_half_width = 0.8
    side_segment = half_width - doorway_half_width
    for y in (-half_width + side_segment / 2.0, half_width - side_segment / 2.0):
        entities.append(
            scene.add_entity(
                gs.morphs.Box(size=(wall, side_segment, config.wall_height), pos=(half_length, y, z), fixed=True),
                surface=room_surface,
            )
        )

    for (x, y), size in OBSTACLE_SPECS:
        entities.append(
            scene.add_entity(
                gs.morphs.Box(size=size, pos=(x, y, size[2] / 2.0), fixed=True),
                surface=obstacle_surface,
            )
        )

    return entities


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def quat_to_yaw(quat: np.ndarray) -> float:
    w, x, y, z = quat
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


@dataclass
class Odometry:
    """Wheel-command odometry used as the controller's pose estimate."""

    position: np.ndarray
    yaw: float

    def integrate(self, linear_speed: float, angular_speed: float, dt: float) -> None:
        self.position[0] += linear_speed * math.cos(self.yaw) * dt
        self.position[1] += linear_speed * math.sin(self.yaw) * dt
        self.yaw = wrap_angle(self.yaw + angular_speed * dt)


def sanitize_lidar(distances: np.ndarray, max_range: float = 6.0) -> np.ndarray:
    values = np.asarray(distances, dtype=np.float32).reshape(-1)
    return np.where(np.isfinite(values) & (values > 0.0), values, max_range)


class DifferentialDriveController:
    """Waypoint controller with a small LiDAR safety layer.

    The waypoint term supplies the nominal navigation command.  If a close obstacle is
    detected in the forward scan, the car stops and turns toward the side with more
    free space.  This keeps the first controller explainable while making LiDAR part
    of the actual control loop instead of a logging-only signal.
    """

    def __init__(self, config: CarConfig, waypoints: tuple[tuple[float, float], ...]):
        self.config = config
        self.waypoints = tuple(np.asarray(waypoint, dtype=np.float32) for waypoint in waypoints)
        self.waypoint_idx = 0
        # The LiDAR is mounted 0.60 m in front of the body center, so a 0.55 m
        # return still leaves a conservative clearance for the 0.36 m half-length
        # chassis while avoiding false stops against the nearby room boundary.
        self.lidar_safety_distance = 0.55

    @property
    def target(self) -> np.ndarray:
        return self.waypoints[-1]

    @property
    def current_target(self) -> np.ndarray:
        return self.waypoints[self.waypoint_idx]

    def command(self, position: np.ndarray, yaw: float, lidar_distances: np.ndarray) -> tuple[float, float, bool]:
        lidar_distances = sanitize_lidar(lidar_distances)
        lidar_angles = np.linspace(-math.pi, math.pi, len(lidar_distances), endpoint=False)
        finite_lidar = np.where(np.isfinite(lidar_distances) & (lidar_distances > 0.0), lidar_distances, 6.0)
        front_mask = np.abs(lidar_angles) <= math.radians(32.0)
        if np.any(front_mask) and float(np.min(finite_lidar[front_mask])) < self.lidar_safety_distance:
            left_mask = (lidar_angles > math.radians(20.0)) & (lidar_angles < math.radians(115.0))
            right_mask = (lidar_angles < -math.radians(20.0)) & (lidar_angles > -math.radians(115.0))
            left_clearance = float(np.mean(finite_lidar[left_mask])) if np.any(left_mask) else 0.0
            right_clearance = float(np.mean(finite_lidar[right_mask])) if np.any(right_mask) else 0.0
            turn_direction = 1.0 if left_clearance >= right_clearance else -1.0
            return 0.0, turn_direction * self.config.max_angular_speed, False

        relative_target = self.current_target - position[:2]
        distance = float(np.linalg.norm(relative_target))
        while distance < 0.28 and self.waypoint_idx < len(self.waypoints) - 1:
            self.waypoint_idx += 1
            relative_target = self.current_target - position[:2]
            distance = float(np.linalg.norm(relative_target))
        if distance < 0.28 and self.waypoint_idx == len(self.waypoints) - 1:
            return 0.0, 0.0, True

        target_heading = math.atan2(float(relative_target[1]), float(relative_target[0]))
        heading_error = wrap_angle(target_heading - yaw)
        forward_speed = min(self.config.max_linear_speed, 0.9 * distance)
        forward_speed *= max(0.0, math.cos(heading_error))
        if abs(heading_error) > 0.5:
            forward_speed = 0.0
        angular_speed = np.clip(2.0 * heading_error, -self.config.max_angular_speed, self.config.max_angular_speed)

        return float(forward_speed), float(angular_speed), False


class KinematicDifferentialDriveCar:
    """Stable MVP vehicle body with differential-drive kinematics and visible wheels.

    The body and wheels are fixed Genesis rigid entities whose poses are advanced by the differential-drive model.
    This keeps the first experiment deterministic while preserving the same sensor and controller interfaces that a
    fully dynamic URDF vehicle will use in the next iteration.
    """

    def __init__(self, body, wheels, initial_pos: tuple[float, float, float], wheel_y: float):
        self.body = body
        self.wheels = wheels
        self.position = np.asarray(initial_pos, dtype=np.float32)
        self.yaw = 0.0
        self.wheel_y = wheel_y

    def get_pos(self):
        return self.body.get_pos()

    def get_quat(self):
        return self.body.get_quat()

    def step(self, linear_speed: float, angular_speed: float, dt: float) -> None:
        self.position[0] += linear_speed * math.cos(self.yaw) * dt
        self.position[1] += linear_speed * math.sin(self.yaw) * dt
        self.yaw = wrap_angle(self.yaw + angular_speed * dt)
        quat = euler_to_quat((0.0, 0.0, math.degrees(self.yaw)))
        self.body.set_pos(self.position)
        self.body.set_quat(quat)

        sin_yaw = math.sin(self.yaw)
        cos_yaw = math.cos(self.yaw)
        for wheel, side in zip(self.wheels, (1.0, -1.0)):
            wheel_offset = np.array((-sin_yaw * side * self.wheel_y, cos_yaw * side * self.wheel_y, -0.10))
            wheel.set_pos(self.position + wheel_offset)
            wheel.set_quat(quat)


def build_scene(args: argparse.Namespace):
    backend = gs.gpu if args.gpu else gs.cpu
    gs.init(backend=backend, precision="32", logging_level="warning", seed=args.seed)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=args.dt),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(6.5, -7.0, 5.8),
            camera_lookat=(0.0, 0.0, 0.45),
            camera_fov=45,
        ),
        profiling_options=gs.options.ProfilingOptions(show_FPS=False),
        show_viewer=args.vis,
    )
    add_room(scene, RoomConfig())

    target = (2.8, 1.7)
    scene.add_entity(
        gs.morphs.Cylinder(height=0.025, radius=0.22, pos=(*target, 0.013), fixed=True),
        surface=gs.surfaces.Emission(color=(0.1, 1.0, 0.2)),
    )

    body = scene.add_entity(
        gs.morphs.Box(size=(0.72, 0.50, 0.20), pos=(-2.8, -1.8, 0.22), fixed=True),
        surface=gs.surfaces.Emission(color=(0.05, 0.25, 0.95)),
    )
    wheels = [
        scene.add_entity(
            gs.morphs.Cylinder(
                radius=0.12,
                height=0.08,
                pos=(-2.8, -1.8 + 0.29, 0.12),
                euler=(90.0, 0.0, 0.0),
                fixed=True,
            ),
            surface=gs.surfaces.Emission(color=(0.015, 0.015, 0.02)),
        ),
        scene.add_entity(
            gs.morphs.Cylinder(
                radius=0.12,
                height=0.08,
                pos=(-2.8, -1.8 - 0.29, 0.12),
                euler=(90.0, 0.0, 0.0),
                fixed=True,
            ),
            surface=gs.surfaces.Emission(color=(0.015, 0.015, 0.02)),
        ),
    ]
    car = KinematicDifferentialDriveCar(body, wheels, (-2.8, -1.8, 0.22), wheel_y=0.29)

    lidar_angles = np.linspace(-180.0, 180.0, 72, endpoint=False).tolist()
    lidar = scene.add_sensor(
        gs.sensors.Lidar(
            pattern=gs.sensors.SphericalPattern(angles=(lidar_angles, [0.0])),
            entity_idx=body.idx,
            link_idx_local=0,
            pos_offset=(0.60, 0.0, 0.22),
            max_range=6.0,
            return_points=False,
        )
    )
    depth_camera = scene.add_sensor(
        gs.sensors.DepthCamera(
            pattern=gs.sensors.DepthCameraPattern(res=(128, 96), fov_horizontal=90.0),
            entity_idx=body.idx,
            link_idx_local=0,
            pos_offset=(0.60, 0.0, 0.23),
            max_range=6.0,
            return_world_frame=False,
        )
    )
    imu = scene.add_sensor(
        gs.sensors.IMU(
            entity_idx=body.idx,
            link_idx_local=0,
            pos_offset=(0.0, 0.0, 0.22),
            acc_noise=(0.01, 0.01, 0.01),
            gyro_noise=(0.005, 0.005, 0.005),
            delay=args.dt,
        )
    )

    rgb_camera = scene.add_camera(
        res=(256, 192),
        pos=(0.0, 0.0, 10.0),
        lookat=(0.0, 0.0, 0.0),
        up=(0.0, 1.0, 0.0),
        fov=50,
        GUI=args.vis,
    )
    # The RGB output is a stable overview camera so that the complete room, car,
    # obstacles, and target remain visible throughout the run.  The DepthCamera
    # above remains the forward-facing robot sensor used by algorithms.
    scene.build()
    return scene, car, lidar, depth_camera, imu, rgb_camera


def save_rgb(rgb: np.ndarray, path: Path) -> None:
    image = np.asarray(rgb)
    if image.ndim == 4:
        image = image[0]
    if image.dtype != np.uint8:
        image = np.clip(image * 255.0 if image.max() <= 1.0 else image, 0, 255).astype(np.uint8)
    Image.fromarray(image).save(path)


def geometry_collision(position: np.ndarray, room: RoomConfig) -> bool:
    """Approximate collision status for the kinematic MVP body.

    The next dynamic URDF vehicle will replace this conservative geometric check with Genesis contact data. Keeping the
    check here makes the current experiment's success/collision metric independent of LiDAR self-returns.
    """
    body_half_x, body_half_y = 0.36, 0.25
    if position[0] < -room.length / 2.0 + body_half_x or position[0] > room.length / 2.0 - body_half_x:
        return True
    if position[1] < -room.width / 2.0 + body_half_y or position[1] > room.width / 2.0 - body_half_y:
        return True

    for (obstacle_x, obstacle_y), (size_x, size_y, _) in OBSTACLE_SPECS:
        overlaps_x = abs(position[0] - obstacle_x) < size_x / 2.0 + body_half_x
        overlaps_y = abs(position[1] - obstacle_y) < size_y / 2.0 + body_half_y
        if overlaps_x and overlaps_y:
            return True
    return False


def run(args: argparse.Namespace) -> dict[str, float | int | bool]:
    if args.steps <= 0:
        raise ValueError("--steps must be greater than zero")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_images:
        (args.output_dir / "rgb").mkdir(exist_ok=True)
        (args.output_dir / "depth").mkdir(exist_ok=True)

    scene, car, lidar, depth_camera, imu, rgb_camera = build_scene(args)
    car_config = CarConfig()
    controller = DifferentialDriveController(
        car_config,
        waypoints=(
            (-2.8, -2.1),
            (-2.8, 1.7),
            (2.8, 1.7),
        ),
    )
    odometry = Odometry(position=np.array([-2.8, -1.8], dtype=np.float32), yaw=0.0)
    telemetry: list[dict[str, float | int | bool]] = []
    reached = False
    collided = False
    last_lidar = np.full(72, 6.0, dtype=np.float32)
    room_config = RoomConfig()

    # Sensor caches are populated by the simulation step. Prime them before the first control decision so the
    # controller does not interpret the zero-initialized cache as an obstacle at the origin.
    scene.step()

    for step in range(args.steps):
        odom_position = odometry.position.copy()
        odom_yaw = odometry.yaw
        lidar_data = lidar.read().distances
        last_lidar = sanitize_lidar(tensor_to_array(lidar_data))
        linear_speed, angular_speed, reached = controller.command(odom_position, odom_yaw, last_lidar)

        car.step(linear_speed, angular_speed, args.dt)
        odometry.integrate(linear_speed, angular_speed, args.dt)
        scene.step()

        position = tensor_to_array(car.get_pos()).reshape(3)
        quaternion = tensor_to_array(car.get_quat()).reshape(4)
        yaw = quat_to_yaw(quaternion)

        if args.save_images or args.vis:
            if step % args.image_every == 0:
                rgb = rgb_camera.render(rgb=True, force_render=True)[0]
                depth = tensor_to_array(depth_camera.read_image())
                if args.save_images:
                    save_rgb(rgb, args.output_dir / "rgb" / f"frame_{step:05d}.png")
                    depth_image = np.clip(depth / 6.0 * 255.0, 0, 255).astype(np.uint8)
                    Image.fromarray(depth_image).save(args.output_dir / "depth" / f"frame_{step:05d}.png")

        lidar_angles = np.linspace(-math.pi, math.pi, len(last_lidar), endpoint=False)
        front_lidar = np.abs(lidar_angles) < math.radians(25.0)
        front_min_lidar = float(np.min(last_lidar[front_lidar]))
        collided = geometry_collision(position, room_config)
        if step % args.log_every == 0 or reached or collided:
            imu_data = imu.read()
            distance_to_target = float(np.linalg.norm(controller.target - position[:2]))
            distance_to_waypoint = float(np.linalg.norm(controller.current_target - position[:2]))
            row = {
                "step": step,
                "time": (step + 1) * args.dt,
                "x": float(position[0]),
                "y": float(position[1]),
                "yaw": float(yaw),
                "odom_x": float(odometry.position[0]),
                "odom_y": float(odometry.position[1]),
                "odom_yaw": float(odometry.yaw),
                "distance_to_target": distance_to_target,
                "distance_to_waypoint": distance_to_waypoint,
                "waypoint_idx": controller.waypoint_idx,
                "linear_command": linear_speed,
                "angular_command": angular_speed,
                "front_min_lidar": front_min_lidar,
                "imu_acc_norm": float(np.linalg.norm(tensor_to_array(imu_data.lin_acc))),
                "imu_gyro_norm": float(np.linalg.norm(tensor_to_array(imu_data.ang_vel))),
                "reached": reached,
                "collided": collided,
            }
            telemetry.append(row)
            print(
                f"step={step:04d} pos=({position[0]:+.2f},{position[1]:+.2f}) "
                f"target_dist={distance_to_target:.2f} waypoint={controller.waypoint_idx} "
                f"lidar_front_min={front_min_lidar:.2f} "
                f"cmd=({linear_speed:+.2f},{angular_speed:+.2f})"
            )

        if reached or collided:
            break

    telemetry_path = args.output_dir / "telemetry.csv"
    if telemetry:
        with telemetry_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(telemetry[0]))
            writer.writeheader()
            writer.writerows(telemetry)

    final_position = tensor_to_array(car.get_pos()).reshape(3)
    summary = {
        "steps": int(step + 1),
        "sim_time": float((step + 1) * args.dt),
        "reached": bool(reached),
        "collided": bool(collided),
        "final_x": float(final_position[0]),
        "final_y": float(final_position[1]),
        "target_x": 2.8,
        "target_y": 1.7,
        "front_min_lidar": float(np.min(last_lidar[front_lidar])),
        "backend": "gpu" if args.gpu else "cpu",
        "telemetry_file": str(telemetry_path),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    summary = run(args)
    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
