"""Kinematic showcase: drive a simple route while continuously recognizing objects.

The vehicle policy remains the existing waypoint + LiDAR safety controller. Visual
perception runs as an explicitly observable side channel, so a missing or slow
model cannot silently replace the safety controller. The default ``disabled``
mode is useful for the control baseline; ``ground_truth`` is deterministic for
runtime/overlay/tracker tests; ``yolo`` requires an existing local weight file.

Examples::

    # Control baseline, no visual inference.
    .venv/bin/python examples/mobile_robot/room_navigation_vision.py \
        --scenario vision_route_showcase --steps 1200

    # Deterministic perception and complete visual logs without a GUI.
    .venv/bin/python examples/mobile_robot/room_navigation_vision.py \
        --scenario vision_route_showcase --perception-mode ground_truth \
        --save-images --save-sensors --save-vision

    # Real local YOLO model (the program never downloads weights).
    .venv/bin/python examples/mobile_robot/room_navigation_vision.py \
        --scenario vision_route_showcase --perception-mode yolo \
        --vision-model models/mobile_robot/car_obstacle.pt \
        --calibrated-depth \
        --save-vision
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image

# Keep the Genesis/Quadrants cache inside the project when this file is run
# directly from a clean checkout.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("QD_OFFLINE_CACHE_FILE_PATH", str(PROJECT_ROOT / ".quadrants_cache"))

import genesis as gs

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
        semantic_objects_for_scenario,
        target_for_scenario,
        waypoints_for_scenario,
    )
    from .vision import (
        DisabledDetector,
        FramePacket,
        GroundTruthDetector,
        ObjectTracker,
        build_genesis_rgbd_calibration,
        enrich_calibrated_depth,
        VisionResult,
        YoloDetector,
        annotate_rgb,
        enrich_approximate_depth,
        enrich_colors,
    )
except ImportError:  # pragma: no cover - direct script execution
    from room_navigation_observable import (  # type: ignore[no-redef]
        CarConfig,
        DifferentialDriveController,
        INITIAL_POSITION,
        Odometry,
        RoomConfig,
        build_scene,
        geometry_collision,
        read_sensor_observation,
        semantic_objects_for_scenario,
        target_for_scenario,
        waypoints_for_scenario,
    )
    from vision import (  # type: ignore[no-redef]
        DisabledDetector,
        FramePacket,
        GroundTruthDetector,
        ObjectTracker,
        build_genesis_rgbd_calibration,
        enrich_calibrated_depth,
        VisionResult,
        YoloDetector,
        annotate_rgb,
        enrich_approximate_depth,
        enrich_colors,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1200, help="Maximum simulation steps.")
    parser.add_argument("--dt", type=float, default=0.02, help="Simulation timestep in seconds.")
    parser.add_argument("--gpu", action="store_true", help="Use the Genesis GPU backend.")
    parser.add_argument("--vis", action="store_true", help="Open the Genesis overview viewer.")
    parser.add_argument("--robot-view", action="store_true", help="Show the attached robot RGB camera.")
    parser.add_argument("--save-images", action="store_true", help="Save overview, robot RGB and depth frames.")
    parser.add_argument("--save-sensors", action="store_true", help="Save synchronized sensor/action JSONL.")
    parser.add_argument("--save-vision", action="store_true", help="Save annotated RGB and vision JSONL.")
    parser.add_argument(
        "--perception-mode",
        choices=("disabled", "ground_truth", "yolo"),
        default="disabled",
        help="Visual backend; yolo requires an existing local model.",
    )
    parser.add_argument("--vision-model", type=Path, help="Local .pt/.onnx/.engine model path for yolo mode.")
    parser.add_argument("--vision-device", default="cpu", help="YOLO device, e.g. cpu, cuda:0 or mps.")
    parser.add_argument("--vision-conf", type=float, default=0.5, help="YOLO confidence threshold.")
    parser.add_argument("--vision-imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--vision-every", type=int, default=5, help="Run visual inference every N steps.")
    parser.add_argument("--vision-max-age", type=float, default=0.5, help="Maximum usable result age in seconds.")
    parser.add_argument(
        "--approx-depth",
        action="store_true",
        help="Deprecated normalized bbox depth sampling; use --calibrated-depth instead.",
    )
    parser.add_argument(
        "--calibrated-depth",
        action="store_true",
        help="Fuse YOLO boxes with the Genesis RGB-D pinhole calibration and output coordinates.",
    )
    parser.add_argument("--image-every", type=int, default=25, help="Save/display images every N steps.")
    parser.add_argument("--log-every", type=int, default=10, help="Print telemetry every N steps.")
    parser.add_argument("--seed", type=int, default=7, help="Genesis and NumPy random seed.")
    parser.add_argument(
        "--scenario",
        choices=("vision_route_showcase", "default", "room_basic", "room_obstacle", "room_center_obstacle"),
        default="vision_route_showcase",
        help="Navigation scene preset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out/mobile_robot_vision"),
        help="Run output directory.",
    )
    return parser.parse_args()


def _error_result(frame: FramePacket, status: str) -> VisionResult:
    return VisionResult(
        frame_id=frame.frame_id,
        sim_time=frame.sim_time,
        model_name="unavailable",
        latency_ms=0.0,
        detections=(),
        status=status,
        available=False,
    )


def _make_detector(args: argparse.Namespace, semantic_specs: tuple[Any, ...]):
    """Create a detector and return an initialization error without crashing control."""

    if args.perception_mode == "disabled":
        return DisabledDetector("disabled"), None
    if args.perception_mode == "ground_truth":
        return GroundTruthDetector(semantic_specs), None
    if args.vision_model is None:
        return DisabledDetector("model_path_missing"), "--vision-model is required in yolo mode"
    try:
        detector = YoloDetector(
            args.vision_model,
            device=args.vision_device,
            confidence=args.vision_conf,
            image_size=args.vision_imgsz,
        )
        return detector, None
    except Exception as exc:  # model import/format/device errors are safe degradation cases
        return DisabledDetector("model_unavailable"), f"{type(exc).__name__}: {exc}"


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _save_rgb(image: np.ndarray | None, path: Path) -> None:
    if image is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(image)
    if array.ndim == 4:
        array = array[0]
    if np.issubdtype(array.dtype, np.floating):
        if array.size and float(np.nanmax(array)) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0.0, 255.0)
    Image.fromarray(array.astype(np.uint8)).save(path)


def _front_min_lidar(lidar: np.ndarray) -> float:
    values = np.asarray(lidar, dtype=np.float32).reshape(-1)
    angles = np.linspace(-math.pi, math.pi, len(values), endpoint=False)
    mask = np.abs(angles) < math.radians(25.0)
    return float(np.min(values[mask])) if np.any(mask) else float(np.min(values))


def _nearest_detection(result: VisionResult) -> tuple[str | None, float | None]:
    candidates = [d for d in result.detections if d.distance_m is not None]
    if not candidates:
        return None, None
    detection = min(candidates, key=lambda item: float(item.distance_m))
    return detection.label, float(detection.distance_m)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.steps <= 0 or args.dt <= 0.0:
        raise ValueError("--steps and --dt must be positive")
    if args.vision_every <= 0 or args.image_every <= 0 or args.log_every <= 0:
        raise ValueError("--vision-every, --image-every and --log-every must be positive")
    if not 0.0 <= args.vision_conf <= 1.0:
        raise ValueError("--vision-conf must be in [0, 1]")
    if args.vision_max_age < 0.0:
        raise ValueError("--vision-max-age must be non-negative")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for directory in ("rgb", "rgb_robot", "rgb_robot_annotated", "depth"):
        if args.save_images or args.save_sensors or args.save_vision:
            (args.output_dir / directory).mkdir(exist_ok=True)

    scene_args = SimpleNamespace(
        gpu=args.gpu,
        dt=args.dt,
        seed=args.seed,
        vis=args.vis,
        robot_view=args.robot_view,
        scenario=args.scenario,
    )
    scene, car, lidar, depth_camera, imu, overview_camera, robot_rgb_camera, obstacle_specs = build_scene(scene_args)
    rgbd_calibration = build_genesis_rgbd_calibration(robot_rgb_camera, depth_camera, getattr(car, "body", None))
    calibration_path = args.output_dir / "camera_calibration.json"
    calibration_path.write_text(json.dumps(rgbd_calibration.to_dict(), indent=2), encoding="utf-8")
    semantic_specs = semantic_objects_for_scenario(args.scenario)
    detector, detector_init_error = _make_detector(args, semantic_specs)
    tracker = ObjectTracker()
    detector_records: list[dict[str, Any]] = []
    sensor_records: list[dict[str, Any]] = []
    telemetry: list[dict[str, Any]] = []
    reached = False
    collided = False
    timed_out = False
    visual_frame_count = 0
    inference_count = 0
    detection_count = 0
    vision_error_count = 0
    last_result: VisionResult | None = None
    last_vision_frame_id: int | None = None
    last_vision_result_age = None
    observation_step = 0
    last_lidar = np.full(72, 6.0, dtype=np.float32)
    telemetry_path = args.output_dir / "telemetry.csv"
    sensor_log_path = args.output_dir / "sensor_observations.jsonl"
    vision_log_path = args.output_dir / "vision_results.jsonl"

    controller = DifferentialDriveController(
        CarConfig(),
        waypoints=waypoints_for_scenario(args.scenario),
        enable_detour=args.scenario == "room_center_obstacle",
        dt=args.dt,
    )
    odometry = Odometry(position=np.array(INITIAL_POSITION[:2], dtype=np.float32), yaw=0.0)
    room_config = RoomConfig()

    try:
        scene.step()
        observation = read_sensor_observation(
            step=0,
            dt=args.dt,
            car=car,
            lidar=lidar,
            depth_camera=depth_camera,
            imu=imu,
            odometry=odometry,
            overview_camera=overview_camera,
            robot_rgb_camera=robot_rgb_camera,
            render=args.perception_mode != "disabled" or args.vis or args.robot_view,
        )

        for step in range(args.steps):
            linear_speed, angular_speed, reached = controller.command_from_observation(observation)
            action = {
                "linear_velocity": float(linear_speed),
                "angular_velocity": float(angular_speed),
            }
            car.step(linear_speed, angular_speed, args.dt)
            odometry.integrate(linear_speed, angular_speed, args.dt)
            scene.step()
            observation_step = step + 1
            vision_due = args.perception_mode != "disabled" and observation_step % args.vision_every == 0
            image_due = observation_step % args.image_every == 0
            capture_frame = vision_due or (
                image_due and (args.save_images or args.save_sensors or args.vis or args.robot_view)
            )
            observation = read_sensor_observation(
                step=observation_step,
                dt=args.dt,
                car=car,
                lidar=lidar,
                depth_camera=depth_camera,
                imu=imu,
                odometry=odometry,
                overview_camera=overview_camera if capture_frame else None,
                robot_rgb_camera=robot_rgb_camera if capture_frame else None,
                render=capture_frame,
            )

            frame: FramePacket | None = None
            result: VisionResult | None = None
            if vision_due and observation.get("rgb") is not None:
                rgb = np.asarray(observation["rgb"])
                frame = FramePacket(
                    frame_id=observation_step,
                    sim_time=float(observation["sim_time"]),
                    image=rgb,
                    camera_name="robot_rgb_camera",
                    camera_pose=tuple(float(value) for value in observation["robot_pose"]),
                    intrinsics=(
                        rgbd_calibration.rgb.fx,
                        rgbd_calibration.rgb.fy,
                        rgbd_calibration.rgb.cx,
                        rgbd_calibration.rgb.cy,
                    ),
                )
                visual_frame_count += 1
                started = time.perf_counter()
                try:
                    result = detector.detect(frame)
                    if args.perception_mode == "yolo":
                        result = enrich_colors(frame, result)
                        if args.calibrated_depth:
                            result = enrich_calibrated_depth(
                                result,
                                observation["depth"],
                                rgbd_calibration,
                                robot_pose=observation["robot_pose"],
                            )
                        elif args.approx_depth:
                            result = enrich_approximate_depth(
                                result,
                                observation["depth"],
                                rgb_shape=(frame.height, frame.width),
                            )
                    result = tracker.update(result)
                    inference_count += 1
                except Exception as exc:  # detector failures never own the control loop
                    result = _error_result(frame, f"inference_error:{type(exc).__name__}")
                if result.latency_ms == 0.0:
                    result = VisionResult(
                        frame_id=result.frame_id,
                        sim_time=result.sim_time,
                        model_name=result.model_name,
                        latency_ms=(time.perf_counter() - started) * 1000.0,
                        detections=result.detections,
                        status=result.status,
                        available=result.available,
                    )
                last_result = result
                last_vision_frame_id = frame.frame_id
                last_vision_result_age = 0.0
                detection_count += len(result.detections)
                if not result.available:
                    vision_error_count += 1
                if args.save_vision:
                    _save_rgb(rgb, args.output_dir / "rgb_robot" / f"frame_{frame.frame_id:05d}.png")
                    _save_rgb(
                        annotate_rgb(rgb, result),
                        args.output_dir / "rgb_robot_annotated" / f"frame_{frame.frame_id:05d}.png",
                    )
                detector_records.append(
                    {
                        "frame": frame.to_metadata(),
                        "result": result.to_dict(),
                        "perception_mode": args.perception_mode,
                        "depth_alignment": (
                            "calibrated_pinhole"
                            if args.calibrated_depth
                            else "normalized_approximation"
                            if args.approx_depth
                            else "not_fused"
                        ),
                    }
                )
            elif last_result is not None:
                last_vision_result_age = float(observation["sim_time"]) - last_result.sim_time

            if capture_frame and args.save_images:
                frame_id = observation_step
                _save_rgb(observation.get("overview_rgb"), args.output_dir / "rgb" / f"frame_{frame_id:05d}.png")
                _save_rgb(observation.get("rgb"), args.output_dir / "rgb_robot" / f"frame_{frame_id:05d}.png")
                depth = np.asarray(observation["depth"], dtype=np.float32)
                depth_image = np.clip(depth / 6.0 * 255.0, 0, 255).astype(np.uint8)
                Image.fromarray(depth_image).save(args.output_dir / "depth" / f"frame_{frame_id:05d}.png")

            position = np.asarray(observation["robot_pose"][:3], dtype=np.float32)
            yaw = float(observation["robot_pose"][3])
            last_lidar = np.asarray(observation["lidar"], dtype=np.float32)
            front_min_lidar = _front_min_lidar(last_lidar)
            collided = geometry_collision(position, room_config, obstacle_specs)
            distance_to_target = float(np.linalg.norm(np.asarray(controller.target) - position[:2]))
            distance_to_waypoint = float(np.linalg.norm(np.asarray(controller.current_target) - position[:2]))
            vision_status = last_result.status if last_result is not None else "disabled"
            nearest_label, nearest_distance = _nearest_detection(last_result) if last_result else (None, None)

            if args.save_sensors:
                sensor_records.append(
                    {
                        "step": observation_step,
                        "sim_time": float(observation["sim_time"]),
                        "observation": {
                            "robot_rgb_frame": last_vision_frame_id,
                            "depth_frame": observation_step if capture_frame else None,
                            "lidar": last_lidar.tolist(),
                            "imu_acc": np.asarray(observation["imu_acc"]).tolist(),
                            "imu_gyro": np.asarray(observation["imu_gyro"]).tolist(),
                            "odom_pose": np.asarray(observation["odom_pose"]).tolist(),
                            "robot_pose": np.asarray(observation["robot_pose"]).tolist(),
                        },
                        "vision": last_result.to_dict() if last_result is not None else None,
                        "action": action,
                        "action_step": step,
                        "action_sim_time": step * args.dt,
                    }
                )

            if step % args.log_every == 0 or reached or collided:
                row = {
                    "step": observation_step,
                    "sim_time": float(observation["sim_time"]),
                    "x": float(position[0]),
                    "y": float(position[1]),
                    "yaw": yaw,
                    "distance_to_target": distance_to_target,
                    "distance_to_waypoint": distance_to_waypoint,
                    "waypoint_idx": controller.waypoint_idx,
                    "linear_command": linear_speed,
                    "angular_command": angular_speed,
                    "front_min_lidar": front_min_lidar,
                    "vision_frame_id": last_vision_frame_id,
                    "vision_status": vision_status,
                    "vision_detection_count": len(last_result.detections) if last_result else 0,
                    "vision_latency_ms": last_result.latency_ms if last_result else None,
                    "vision_result_age_ms": (
                        last_vision_result_age * 1000.0 if last_vision_result_age is not None else None
                    ),
                    "nearest_vision_label": nearest_label,
                    "nearest_vision_distance_m": nearest_distance,
                    "reached": bool(reached),
                    "collided": bool(collided),
                }
                telemetry.append(row)
                print(
                    f"step={observation_step:04d} pos=({position[0]:+.2f},{position[1]:+.2f}) "
                    f"target_dist={distance_to_target:.2f} waypoint={controller.waypoint_idx} "
                    f"vision={vision_status}:{len(last_result.detections) if last_result else 0} "
                    f"lidar_front_min={front_min_lidar:.2f} cmd=({linear_speed:+.2f},{angular_speed:+.2f})"
                )

            if reached or collided:
                break
        else:
            timed_out = True

        if telemetry:
            with telemetry_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(telemetry[0]))
                writer.writeheader()
                writer.writerows(telemetry)
        if args.save_sensors:
            _write_jsonl(sensor_log_path, sensor_records)
        if args.save_vision:
            _write_jsonl(vision_log_path, detector_records)

        final_position = np.asarray(car.get_pos()).reshape(3)
        summary = {
            "scenario": args.scenario,
            "steps": int(observation_step),
            "sim_time": float(observation_step * args.dt),
            "reached": bool(reached),
            "collided": bool(collided),
            "timed_out": bool(timed_out),
            "termination_reason": "reached" if reached else "collision" if collided else "timeout",
            "final_x": float(final_position[0]),
            "final_y": float(final_position[1]),
            "target_x": target_for_scenario(args.scenario)[0],
            "target_y": target_for_scenario(args.scenario)[1],
            "perception_mode": args.perception_mode,
            "detector_model": (
                str(getattr(detector, "model_path"))
                if getattr(detector, "model_path", None)
                else getattr(detector, "reason", None)
            ),
            "detector_init_error": detector_init_error,
            "visual_frame_count": visual_frame_count,
            "inference_count": inference_count,
            "detection_count": detection_count,
            "vision_error_count": vision_error_count,
            "tracked_object_count": len(tracker.summaries()),
            "front_min_lidar": _front_min_lidar(last_lidar),
            "backend": "gpu" if args.gpu else "cpu",
            "telemetry_file": str(telemetry_path),
            "sensor_log_file": str(sensor_log_path) if args.save_sensors else None,
            "vision_log_file": str(vision_log_path) if args.save_vision else None,
            "tracked_objects_file": str(args.output_dir / "tracked_objects.json"),
            "robot_rgb_camera": "attached_forward_view",
            "camera_calibration_file": str(calibration_path),
            "depth_alignment": (
                "calibrated_pinhole"
                if args.calibrated_depth
                else "normalized_approximation"
                if args.approx_depth
                else "not_fused"
            ),
        }
        (args.output_dir / "tracked_objects.json").write_text(
            json.dumps(tracker.summaries(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return summary
    finally:
        gs.destroy()


def main() -> None:
    summary = run(parse_args())
    print("\nSummary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
