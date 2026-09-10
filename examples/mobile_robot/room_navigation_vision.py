"""Search with RGB-D vision and navigate to a text-described target.

The vehicle policy remains the existing waypoint + LiDAR safety controller. Without
``--instruction`` this program preserves the fixed-route baseline. With an
instruction, that route is only a bounded visual-search route; after a confirmed
RGB-D target, the controller is replanned to one temporary waypoint near that
target. A missing or slow model never invents a target and ends with
``TARGET_NOT_FOUND`` instead of driving to the baseline route's final point.

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
        --save-vision --annotated-view

    # Open-vocabulary camera side channel (does not alter vehicle control).
    .venv/bin/python examples/mobile_robot/room_navigation_vision.py \
        --scenario vision_route_showcase --perception-mode open_vocab \
        --vision-model models/mobile_robot/open_vocab/yolov8s-world.pt \
        --vision-prompt "yellow car" --vision-prompt "green pillar" \
        --open-vocab-device auto --annotated-view
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
import hashlib
import json
import math
import os
import re
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
        OpenVocabularyDetector,
        build_genesis_rgbd_calibration,
        enrich_calibrated_depth,
        AnnotatedRgbView,
        VisionResult,
        YoloDetector,
        annotate_rgb,
        enrich_approximate_depth,
        enrich_colors,
        instruction_to_visual_prompt,
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
        OpenVocabularyDetector,
        build_genesis_rgbd_calibration,
        enrich_calibrated_depth,
        AnnotatedRgbView,
        VisionResult,
        YoloDetector,
        annotate_rgb,
        enrich_approximate_depth,
        enrich_colors,
        instruction_to_visual_prompt,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1200, help="Maximum simulation steps.")
    parser.add_argument("--dt", type=float, default=0.02, help="Simulation timestep in seconds.")
    parser.add_argument("--gpu", action="store_true", help="Use the Genesis GPU backend.")
    parser.add_argument("--vis", action="store_true", help="Open the Genesis overview viewer.")
    parser.add_argument("--robot-view", action="store_true", help="Show the attached robot RGB camera.")
    parser.add_argument(
        "--annotated-view",
        action="store_true",
        help="Open a separate OpenCV window with per-frame RGB detection overlays.",
    )
    parser.add_argument("--save-images", action="store_true", help="Save overview, robot RGB and depth frames.")
    parser.add_argument("--save-sensors", action="store_true", help="Save synchronized sensor/action JSONL.")
    parser.add_argument("--save-vision", action="store_true", help="Save annotated RGB and vision JSONL.")
    parser.add_argument(
        "--perception-mode",
        choices=("disabled", "ground_truth", "yolo", "open_vocab"),
        default="disabled",
        help="Visual backend; yolo/open_vocab require explicit local weights.",
    )
    parser.add_argument(
        "--vision-model",
        type=Path,
        help="Local model path for yolo (.pt/.onnx/.engine) or open_vocab (backend directory/file).",
    )
    parser.add_argument("--vision-device", default="cpu", help="YOLO device, e.g. cpu, cuda:0 or mps.")
    parser.add_argument("--vision-conf", type=float, default=0.5, help="YOLO confidence threshold.")
    parser.add_argument("--vision-imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument(
        "--open-vocab-backend",
        choices=("yolo-world", "owlv2"),
        default="yolo-world",
        help="Backend used by open_vocab mode.",
    )
    parser.add_argument(
        "--vision-prompt",
        action="append",
        default=[],
        help="Open-vocabulary text prompt; repeat for multiple prompts.",
    )
    parser.add_argument(
        "--instruction",
        help=(
            "Minimal natural-language demo instruction, for example "
            "'行驶到黄色小车附近'; converted to one YOLO-World visual prompt."
        ),
    )
    parser.add_argument(
        "--target-near-distance",
        type=float,
        default=0.85,
        help="Demo stopping distance from the grounded target center, in metres.",
    )
    parser.add_argument(
        "--target-confirm-frames",
        type=int,
        default=2,
        help="Consecutive RGB-D observations required before target re-planning.",
    )
    parser.add_argument(
        "--open-vocab-decision-conf",
        type=float,
        default=0.05,
        help="Open-vocabulary confidence threshold used for the reported perception status.",
    )
    parser.add_argument(
        "--target-lock-conf",
        type=float,
        default=0.001,
        help=(
            "Minimum candidate confidence for instruction target locking. YOLO-World "
            "scores are calibrated separately from the status threshold."
        ),
    )
    parser.add_argument(
        "--open-vocab-infer-conf",
        type=float,
        default=0.001,
        help="Open-vocabulary model pre-filter confidence.",
    )
    parser.add_argument(
        "--open-vocab-device",
        default="auto",
        choices=("auto", "mps", "cpu", "cuda", "cuda:0"),
        help="Open-vocabulary device; auto is MPS first and CPU fallback.",
    )
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
        help="Fuse detections with the Genesis RGB-D pinhole calibration and output coordinates.",
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
    if args.perception_mode == "open_vocab":
        if args.vision_model is None:
            return DisabledDetector("model_path_missing"), "--vision-model is required in open_vocab mode"
        if not args.vision_prompt:
            return DisabledDetector("prompt_missing"), "--vision-prompt is required in open_vocab mode"
        try:
            detector = OpenVocabularyDetector(
                args.vision_model,
                args.vision_prompt,
                backend=args.open_vocab_backend,
                device=args.open_vocab_device,
                image_size=args.vision_imgsz,
                confidence=args.open_vocab_infer_conf,
                decision_confidence=args.open_vocab_decision_conf,
            )
            return detector, None
        except Exception as exc:  # optional model/dependency errors are safe degradation cases
            return DisabledDetector("model_unavailable"), f"{type(exc).__name__}: {exc}"
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


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _file_sha256(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nearby_waypoint(
    target_world: np.ndarray,
    robot_pose: np.ndarray,
    clearance_m: float,
) -> tuple[float, float]:
    """Return a simple point before the target, keeping the demo explainable."""

    target_xy = np.asarray(target_world[:2], dtype=np.float64)
    robot_xy = np.asarray(robot_pose[:2], dtype=np.float64)
    direction = target_xy - robot_xy
    distance = float(np.linalg.norm(direction))
    if distance <= 1e-6:
        return float(target_xy[0]), float(target_xy[1])
    point = target_xy - direction / distance * float(clearance_m)
    return float(point[0]), float(point[1])


_PROMPT_COLOR_TERMS = {
    "yellow": "yellow",
    "green": "green",
    "red": "red",
    "blue": "blue",
    "黄色": "yellow",
    "绿色": "green",
    "红色": "red",
    "蓝色": "blue",
}


def _prompt_expected_colors(prompt: str | None) -> tuple[str, ...]:
    """Extract RGB-verifiable colour attributes from an open prompt.

    This is an attribute-consistency guard, not an object category list. Terms
    outside the small RGB attribute vocabulary are left unconstrained so an
    arbitrary object phrase can still reach the open-vocabulary model.
    """

    if not prompt:
        return ()
    text = str(prompt).strip().lower()
    found: list[str] = []
    for term, canonical in _PROMPT_COLOR_TERMS.items():
        if term.isascii():
            present = re.search(rf"\b{re.escape(term)}\b", text) is not None
        else:
            present = term in text
        if present and canonical not in found:
            found.append(canonical)
    return tuple(found)


def _best_grounded_detection(
    result: VisionResult,
    *,
    confidence: float,
    prompt: str | None = None,
) -> Any | None:
    """Select the strongest open-vocabulary candidate usable for navigation.

    A candidate must have a finite RGB-D world position. When the prompt
    contains a colour, the independent RGB ROI attribute must agree. Selection
    is otherwise based on prompt-conditioned detections and evidence quality;
    no fixed object-class list is consulted, so arbitrary text prompts remain
    supported.
    """

    expected_colors = _prompt_expected_colors(prompt)
    grounded = []
    for detection in result.detections:
        if float(detection.confidence) < float(confidence):
            continue
        position = detection.position_world
        if position is None:
            continue
        values = np.asarray(position, dtype=np.float64).reshape(-1)
        if values.size < 3 or not np.all(np.isfinite(values[:3])):
            continue
        if detection.distance_confidence is not None and float(detection.distance_confidence) <= 0.0:
            continue
        if expected_colors:
            # A coloured referring expression must agree with the independent
            # RGB ROI attribute check. A text model's low-confidence box alone
            # cannot relabel a yellow object as a green one.
            if detection.color not in expected_colors:
                continue
            if detection.color_confidence is None or float(detection.color_confidence) < 0.35:
                continue
        grounded.append(detection)
    return max(grounded, key=lambda item: float(item.confidence), default=None)


def run(args: argparse.Namespace) -> dict[str, Any]:
    resolved_visual_prompt: str | None = None
    if args.instruction:
        if args.perception_mode != "open_vocab":
            raise ValueError("--instruction requires --perception-mode open_vocab")
        if args.vision_prompt:
            raise ValueError("use --instruction or --vision-prompt, not both")
        resolved_visual_prompt = instruction_to_visual_prompt(args.instruction)
        args.vision_prompt = [resolved_visual_prompt]
        # The demo needs a world point to create the temporary target waypoint.
        args.calibrated_depth = True
    if args.steps <= 0 or args.dt <= 0.0:
        raise ValueError("--steps and --dt must be positive")
    if args.vision_every <= 0 or args.image_every <= 0 or args.log_every <= 0:
        raise ValueError("--vision-every, --image-every and --log-every must be positive")
    if not 0.0 <= args.vision_conf <= 1.0:
        raise ValueError("--vision-conf must be in [0, 1]")
    if not 0.0 <= args.open_vocab_infer_conf <= 1.0:
        raise ValueError("--open-vocab-infer-conf must be in [0, 1]")
    if not 0.0 <= args.open_vocab_decision_conf <= 1.0:
        raise ValueError("--open-vocab-decision-conf must be in [0, 1]")
    if not 0.0 <= args.target_lock_conf <= 1.0:
        raise ValueError("--target-lock-conf must be in [0, 1]")
    if args.vision_max_age < 0.0:
        raise ValueError("--vision-max-age must be non-negative")
    if args.target_near_distance <= 0.0:
        raise ValueError("--target-near-distance must be positive")
    if args.target_confirm_frames <= 0:
        raise ValueError("--target-confirm-frames must be positive")

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
    annotated_view = AnnotatedRgbView() if args.annotated_view else None
    detector_records: list[dict[str, Any]] = []
    sensor_records: list[dict[str, Any]] = []
    telemetry: list[dict[str, Any]] = []
    reached = False
    collided = False
    timed_out = False
    visual_frame_count = 0
    inference_count = 0
    detection_count = 0
    detection_counts_by_label: Counter[str] = Counter()
    vision_latencies_ms: list[float] = []
    vision_error_count = 0
    last_result: VisionResult | None = None
    last_vision_frame_id: int | None = None
    last_vision_result_age = None
    target_lock = False
    target_confirm_count = 0
    target_world_position: tuple[float, float, float] | None = None
    target_near_waypoint: tuple[float, float] | None = None
    last_candidate_world: np.ndarray | None = None
    navigation_mode = "searching" if args.instruction else "fixed_waypoints"
    target_not_found = False
    target_candidate_status = "searching" if args.instruction else "inactive"
    target_attribute_mismatch_count = 0
    observation_step = 0
    last_lidar = np.full(72, 6.0, dtype=np.float32)
    telemetry_path = args.output_dir / "telemetry.csv"
    sensor_log_path = args.output_dir / "sensor_observations.jsonl"
    vision_log_path = args.output_dir / "vision_results.jsonl"

    search_waypoints = waypoints_for_scenario(args.scenario)
    controller = DifferentialDriveController(
        CarConfig(),
        waypoints=search_waypoints,
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
            render=args.perception_mode != "disabled" or args.vis or args.robot_view or args.annotated_view,
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
                image_due
                and (
                    args.save_images
                    or args.save_sensors
                    or args.vis
                    or args.robot_view
                    or args.annotated_view
                )
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
                    camera_pose=rgbd_calibration.camera_pose_from_robot_pose(observation["robot_pose"]),
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
                    if args.perception_mode in {"yolo", "open_vocab"}:
                        result = enrich_colors(frame, result)
                        if args.calibrated_depth:
                            result = enrich_calibrated_depth(
                                result,
                                observation["depth"],
                                rgbd_calibration,
                                robot_pose=observation["robot_pose"],
                            )
                        elif args.approx_depth and args.perception_mode == "yolo":
                            result = enrich_approximate_depth(
                                result,
                                observation["depth"],
                                rgb_shape=(frame.height, frame.width),
                            )
                    result = tracker.update(result)
                    if args.instruction and not target_lock:
                        unconstrained_candidate = _best_grounded_detection(
                            result,
                            confidence=args.target_lock_conf,
                        )
                        best_grounded = _best_grounded_detection(
                            result,
                            confidence=args.target_lock_conf,
                            prompt=args.vision_prompt[0] if args.vision_prompt else None,
                        )
                        if unconstrained_candidate is None:
                            target_candidate_status = "no_grounded_candidate"
                        elif best_grounded is None:
                            target_candidate_status = "attribute_mismatch"
                            target_attribute_mismatch_count += 1
                        else:
                            target_candidate_status = "candidate"
                        if best_grounded is None:
                            target_confirm_count = 0
                            last_candidate_world = None
                        else:
                            candidate_world = np.asarray(best_grounded.position_world, dtype=np.float64)
                            candidate_is_stable = (
                                last_candidate_world is not None
                                and float(np.linalg.norm(candidate_world - last_candidate_world)) < 1.25
                            )
                            target_confirm_count = target_confirm_count + 1 if candidate_is_stable else 1
                            last_candidate_world = candidate_world
                            if target_confirm_count >= args.target_confirm_frames:
                                target_world_position = tuple(float(value) for value in candidate_world)
                                target_near_waypoint = _nearby_waypoint(
                                    candidate_world,
                                    np.asarray(observation["robot_pose"], dtype=np.float64),
                                    args.target_near_distance,
                                )
                                # Replace the bounded search route. The
                                # existing controller and LiDAR safety layer
                                # remain responsible for every wheel command.
                                controller.replace_waypoints(
                                    (target_near_waypoint,),
                                    enable_detour=True,
                                )
                                target_lock = True
                                navigation_mode = "target_locked"
                                target_candidate_status = "locked"
                                reached = False
                                print(
                                    "target_confirmed "
                                    f"prompt={args.vision_prompt[0]!r} "
                                    f"world=({candidate_world[0]:+.2f},{candidate_world[1]:+.2f}) "
                                    f"near=({target_near_waypoint[0]:+.2f},{target_near_waypoint[1]:+.2f})"
                                )
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
                detection_counts_by_label.update(detection.label for detection in result.detections)
                vision_latencies_ms.append(float(result.latency_ms))
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
                        "open_vocab_prompts": (
                            list(args.vision_prompt) if args.perception_mode == "open_vocab" else None
                        ),
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

            if annotated_view is not None and capture_frame and observation.get("rgb") is not None:
                # Only draw a result on the frame it was inferred from.  Reusing
                # an older result would make boxes visibly lag behind the moving
                # camera; the next vision tick refreshes them with a new result.
                annotated_view.show(observation["rgb"], result)

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
            usable_result = last_result
            if (
                last_result is not None
                and last_vision_result_age is not None
                and last_vision_result_age > args.vision_max_age
            ):
                usable_result = None
            vision_status = (
                usable_result.status
                if usable_result is not None
                else "stale"
                if last_result is not None
                else "disabled"
            )
            nearest_label, nearest_distance = _nearest_detection(usable_result) if usable_result else (None, None)

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
                        "vision": usable_result.to_dict() if usable_result is not None else None,
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
                    "vision_detection_count": len(usable_result.detections) if usable_result else 0,
                    "vision_latency_ms": usable_result.latency_ms if usable_result else None,
                    "vision_result_age_ms": (
                        last_vision_result_age * 1000.0 if last_vision_result_age is not None else None
                    ),
                    "target_lock": bool(target_lock),
                    "navigation_mode": navigation_mode,
                    "target_candidate_status": target_candidate_status,
                    "target_attribute_mismatch_count": target_attribute_mismatch_count,
                    "target_world_x": None if target_world_position is None else target_world_position[0],
                    "target_world_y": None if target_world_position is None else target_world_position[1],
                    "nearest_vision_label": nearest_label,
                    "nearest_vision_distance_m": nearest_distance,
                    "reached": bool(reached),
                    "collided": bool(collided),
                }
                telemetry.append(row)
                print(
                    f"step={observation_step:04d} pos=({position[0]:+.2f},{position[1]:+.2f}) "
                    f"target_dist={distance_to_target:.2f} waypoint={controller.waypoint_idx} "
                    f"vision={vision_status}:{len(usable_result.detections) if usable_result else 0} "
                    f"lidar_front_min={front_min_lidar:.2f} cmd=({linear_speed:+.2f},{angular_speed:+.2f})"
                )

            # Reaching the end of the bounded search route is not success for
            # an instruction. Stop and make the missing-target state explicit;
            # never continue to the old fixed route's final point.
            if args.instruction and reached and not target_lock:
                target_not_found = True
                navigation_mode = "target_not_found"
                reached = False
                print("target_not_found search_route_complete")
                break

            demo_target_reached = bool(args.instruction and target_lock and reached)
            if collided or demo_target_reached or (reached and not args.instruction):
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
            "termination_reason": (
                "target_not_found"
                if target_not_found
                else "reached"
                if reached
                else "collision"
                if collided
                else "timeout"
            ),
            "final_x": float(final_position[0]),
            "final_y": float(final_position[1]),
            "target_x": target_for_scenario(args.scenario)[0],
            "target_y": target_for_scenario(args.scenario)[1],
            "search_waypoints": [list(map(float, waypoint)) for waypoint in search_waypoints],
            "active_waypoints": [list(map(float, waypoint)) for waypoint in controller.waypoints],
            "command_target_x": float(controller.target[0]),
            "command_target_y": float(controller.target[1]),
            "perception_mode": args.perception_mode,
            "open_vocab_backend": args.open_vocab_backend if args.perception_mode == "open_vocab" else None,
            "open_vocab_prompts": list(args.vision_prompt) if args.perception_mode == "open_vocab" else None,
            "user_instruction": args.instruction,
            "resolved_visual_prompt": resolved_visual_prompt,
            "target_lock": target_lock,
            "target_lock_conf": args.target_lock_conf if args.instruction else None,
            "target_confirm_frames": args.target_confirm_frames if args.instruction else None,
            "target_world_position": None if target_world_position is None else list(target_world_position),
            "target_near_waypoint": None if target_near_waypoint is None else list(target_near_waypoint),
            "navigation_mode": navigation_mode,
            "target_not_found": target_not_found,
            "target_candidate_status": target_candidate_status,
            "target_attribute_mismatch_count": target_attribute_mismatch_count,
            "detector_model": (
                str(getattr(detector, "model_path"))
                if getattr(detector, "model_path", None)
                else getattr(detector, "reason", None)
            ),
            "detector_model_sha256": _file_sha256(args.vision_model) if args.perception_mode == "yolo" else None,
            "detector_init_error": detector_init_error,
            "visual_frame_count": visual_frame_count,
            "inference_count": inference_count,
            "detection_count": detection_count,
            "detection_counts_by_label": dict(detection_counts_by_label),
            "vision_latency_ms": {
                "count": len(vision_latencies_ms),
                "mean": (
                    sum(vision_latencies_ms) / len(vision_latencies_ms) if vision_latencies_ms else None
                ),
                "p50": _percentile(vision_latencies_ms, 50.0),
                "p95": _percentile(vision_latencies_ms, 95.0),
                "max": max(vision_latencies_ms) if vision_latencies_ms else None,
            },
            "vision_error_count": vision_error_count,
            "tracked_object_count": len(tracker.summaries()),
            "front_min_lidar": _front_min_lidar(last_lidar),
            "backend": "gpu" if args.gpu else "cpu",
            "telemetry_file": str(telemetry_path),
            "sensor_log_file": str(sensor_log_path) if args.save_sensors else None,
            "vision_log_file": str(vision_log_path) if args.save_vision else None,
            "tracked_objects_file": str(args.output_dir / "tracked_objects.json"),
            "robot_rgb_camera": "attached_forward_view",
            "annotated_view": bool(args.annotated_view),
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
        if annotated_view is not None:
            annotated_view.close()
        gs.destroy()


def main() -> None:
    summary = run(parse_args())
    print("\nSummary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
