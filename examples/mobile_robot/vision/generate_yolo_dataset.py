"""Generate a local YOLO detection dataset from Genesis segmentation truth.

This tool is deliberately offline and dataset-only: segmentation is used to
produce labels, while the runtime detector receives RGB images only.  It
creates deterministic layout splits so adjacent frames from one route cannot
leak across train/validation/test.

Pilot example::

    .venv/bin/python -m examples.mobile_robot.vision.generate_yolo_dataset \
        --layouts 8 --output-dir datasets/mobile_robot_yolo_pilot
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image

import genesis as gs

if __package__:
    from ..room_navigation_observable import (
        CarConfig,
        DifferentialDriveController,
        INITIAL_POSITION,
        Odometry,
        RoomConfig,
        geometry_collision,
        read_sensor_observation,
        semantic_objects_for_scenario,
        waypoints_for_scenario,
        build_scene,
    )
    from .rgbd_calibration import build_genesis_rgbd_calibration
else:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from room_navigation_observable import (  # type: ignore[no-redef]
        CarConfig,
        DifferentialDriveController,
        INITIAL_POSITION,
        Odometry,
        RoomConfig,
        geometry_collision,
        read_sensor_observation,
        semantic_objects_for_scenario,
        waypoints_for_scenario,
        build_scene,
    )
    from vision.rgbd_calibration import build_genesis_rgbd_calibration  # type: ignore[no-redef]


CLASS_NAMES = ("car", "box_obstacle", "cylinder_obstacle")
CLASS_IDS = {name: index for index, name in enumerate(CLASS_NAMES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layouts", type=int, default=8, help="Number of independent randomized scene layouts.")
    parser.add_argument("--steps", type=int, default=1200, help="Maximum route steps per layout.")
    parser.add_argument("--sample-every", type=int, default=25, help="Save one frame every N simulation steps.")
    parser.add_argument("--dt", type=float, default=0.02, help="Genesis simulation timestep.")
    parser.add_argument("--seed", type=int, default=7, help="Master seed for layout and simulation randomness.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/mobile_robot_yolo_pilot"),
        help="Output directory containing YOLO images, labels, masks and metadata.",
    )
    parser.add_argument(
        "--scenario",
        choices=("vision_route_showcase",),
        default="vision_route_showcase",
        help="Only the semantic showcase scene currently has the required registry.",
    )
    return parser.parse_args()


def _randomized_specs(seed: int):
    """Keep all semantic objects in safe corridors while changing layouts."""

    base = semantic_objects_for_scenario("vision_route_showcase")
    rng = np.random.default_rng(seed)
    ranges = {
        # Keep the semantic footprints outside the waypoint corridor while
        # still changing their distance and lateral view across layouts.
        "yellow_car_01": ((-1.88, -1.68), (0.24, 0.46)),
        "red_box_01": ((0.30, 0.50), (2.08, 2.22)),
        "blue_cylinder_01": ((1.10, 1.30), (-1.55, -1.38)),
    }
    specs = []
    for spec in base:
        x_range, y_range = ranges[spec.object_id]
        position = (
            float(rng.uniform(*x_range)),
            float(rng.uniform(*y_range)),
            float(spec.position[2]),
        )
        specs.append(replace(spec, position=position))
    return tuple(specs)


def _entity_idx(value: Any) -> int | None:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, tuple) and value:
        return _entity_idx(value[0])
    if hasattr(value, "idx"):
        return int(value.idx)
    return None


def _segmentation_entity_map(scene: Any, registry: tuple[Any, ...]) -> dict[int, Any]:
    by_entity = {int(entity.idx): spec for spec, entity in registry}
    mapping: dict[int, Any] = {}
    for seg_id, value in scene.segmentation_idx_dict.items():
        entity_idx = _entity_idx(value)
        if entity_idx in by_entity:
            mapping[int(seg_id)] = by_entity[entity_idx]
    if len(mapping) != len(registry):
        missing = sorted(spec.object_id for spec, entity in registry if int(entity.idx) not in {
            _entity_idx(value) for value in scene.segmentation_idx_dict.values()
        })
        raise RuntimeError(
            f"could not map all semantic entities through Genesis segmentation: "
            f"mapped={len(mapping)} expected={len(registry)} missing={missing}"
        )
    return mapping


def _labels_from_segmentation(
    segmentation: np.ndarray,
    seg_to_spec: dict[int, Any],
    *,
    width: int,
    height: int,
) -> tuple[list[str], list[dict[str, Any]], np.ndarray]:
    values = np.asarray(segmentation)
    if values.ndim == 3:
        values = values[..., 0]
    if values.shape != (height, width):
        raise ValueError(f"unexpected segmentation shape {values.shape}; expected {(height, width)}")
    instance_mask = np.zeros((height, width), dtype=np.uint16)
    labels: list[str] = []
    records: list[dict[str, Any]] = []
    instance_id = 1
    for seg_id, spec in sorted(seg_to_spec.items()):
        ys, xs = np.where(values == seg_id)
        if xs.size == 0:
            continue
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        if x2 <= x1 or y2 <= y1:
            continue
        class_id = CLASS_IDS.get(str(spec.category))
        if class_id is None:
            continue
        labels.append(
            f"{class_id} {(x1 + x2) / (2.0 * width):.8f} {(y1 + y2) / (2.0 * height):.8f} "
            f"{(x2 - x1) / width:.8f} {(y2 - y1) / height:.8f}"
        )
        instance_mask[values == seg_id] = instance_id
        records.append(
            {
                "object_id": str(spec.object_id),
                "category": str(spec.category),
                "color": str(spec.color),
                "segmentation_id": int(seg_id),
                "instance_id": instance_id,
                "bbox_xyxy": [x1, y1, x2, y2],
                "visible_pixel_count": int(xs.size),
                "truncated": bool(x1 == 0 or y1 == 0 or x2 == width or y2 == height),
            }
        )
        instance_id += 1
    return labels, records, instance_mask


def _split_for_layout(layout_index: int, layout_order: list[int], total_layouts: int) -> str:
    ordered_index = layout_order.index(layout_index)
    train_count = max(1, int(round(total_layouts * 0.75)))
    val_count = max(1, int(round(total_layouts * 0.125))) if total_layouts >= 3 else 0
    if ordered_index < train_count:
        return "train"
    if ordered_index < train_count + val_count:
        return "val"
    return "test"


def _write_dataset_yaml(output_dir: Path) -> None:
    yaml_text = (
        f"path: {output_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "names:\n"
        "  0: car\n"
        "  1: box_obstacle\n"
        "  2: cylinder_obstacle\n"
    )
    (output_dir / "dataset.yaml").write_text(yaml_text, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.layouts <= 0 or args.steps <= 0 or args.sample_every <= 0 or args.dt <= 0.0:
        raise ValueError("layouts, steps, sample-every and dt must be positive")
    output_dir = args.output_dir
    for split in ("train", "val", "test"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    (output_dir / "masks").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    layout_order = [int(value) for value in rng.permutation(args.layouts)]
    metadata_path = output_dir / "metadata.jsonl"
    metadata_file = metadata_path.open("w", encoding="utf-8")
    total_images = 0
    empty_label_images = 0
    class_counts = {name: 0 for name in CLASS_NAMES}
    calibration_dict = None
    try:
        for layout_index in range(args.layouts):
            layout_seed = int(args.seed + layout_index * 1009)
            specs = _randomized_specs(layout_seed)
            scene_args = SimpleNamespace(
                gpu=False,
                dt=args.dt,
                seed=layout_seed,
                vis=False,
                robot_view=False,
                scenario=args.scenario,
                segmentation_level="entity",
                return_semantic_registry=True,
                semantic_specs_override=specs,
            )
            try:
                built = build_scene(scene_args)
                (
                    scene,
                    car,
                    lidar,
                    depth_camera,
                    imu,
                    _overview_camera,
                    robot_rgb_camera,
                    obstacle_specs,
                    registry,
                ) = built
                if calibration_dict is None:
                    calibration_dict = build_genesis_rgbd_calibration(
                        robot_rgb_camera,
                        depth_camera,
                        getattr(car, "body", None),
                    ).to_dict()
                seg_to_spec = _segmentation_entity_map(scene, registry)
                controller = DifferentialDriveController(
                    CarConfig(),
                    waypoints=waypoints_for_scenario(args.scenario),
                    dt=args.dt,
                )
                odometry = Odometry(position=np.array(INITIAL_POSITION[:2], dtype=np.float32), yaw=0.0)
                room_config = RoomConfig()
                scene.step()
                observation = read_sensor_observation(
                    step=0,
                    dt=args.dt,
                    car=car,
                    lidar=lidar,
                    depth_camera=depth_camera,
                    imu=imu,
                    odometry=odometry,
                    render=False,
                )
                for step in range(args.steps):
                    linear, angular, reached = controller.command_from_observation(observation)
                    car.step(linear, angular, args.dt)
                    odometry.integrate(linear, angular, args.dt)
                    scene.step()
                    observation_step = step + 1
                    observation = read_sensor_observation(
                        step=observation_step,
                        dt=args.dt,
                        car=car,
                        lidar=lidar,
                        depth_camera=depth_camera,
                        imu=imu,
                        odometry=odometry,
                        render=False,
                    )
                    if observation_step % args.sample_every != 0 and not reached:
                        if geometry_collision(
                            np.asarray(observation["robot_pose"][:3]), room_config, obstacle_specs
                        ):
                            raise RuntimeError(f"layout {layout_index} collided at step {observation_step}")
                        continue

                    rgb = np.asarray(robot_rgb_camera.render(rgb=True, force_render=True)[0])
                    segmentation = np.asarray(
                        robot_rgb_camera.render(rgb=False, segmentation=True, force_render=True)[2]
                    )
                    if rgb.ndim == 4:
                        rgb = rgb[0]
                    split = _split_for_layout(layout_index, layout_order, args.layouts)
                    stem = f"layout_{layout_index:03d}_frame_{observation_step:05d}"
                    image_rel = Path("images") / split / f"{stem}.png"
                    label_rel = Path("labels") / split / f"{stem}.txt"
                    mask_rel = Path("masks") / f"{stem}.png"
                    labels, object_records, instance_mask = _labels_from_segmentation(
                        segmentation,
                        seg_to_spec,
                        width=int(rgb.shape[1]),
                        height=int(rgb.shape[0]),
                    )
                    if not labels:
                        empty_label_images += 1
                    Image.fromarray(rgb.astype(np.uint8)).save(output_dir / image_rel)
                    (output_dir / label_rel).write_text("\n".join(labels) + "\n", encoding="utf-8")
                    Image.fromarray(instance_mask).save(output_dir / mask_rel)
                    for record in object_records:
                        class_counts[record["category"]] += 1
                    metadata_file.write(
                        json.dumps(
                            {
                                "layout_index": layout_index,
                                "layout_seed": layout_seed,
                                "split": split,
                                "step": observation_step,
                                "sim_time": observation_step * args.dt,
                                "image": str(image_rel),
                                "label": str(label_rel),
                                "mask": str(mask_rel),
                                "objects": object_records,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    total_images += 1
                    if total_images % 25 == 0:
                        print(f"generated images={total_images} layout={layout_index} split={split}")
                    if reached:
                        break
            finally:
                gs.destroy()
    finally:
        metadata_file.close()

    if calibration_dict is None:
        raise RuntimeError("no layout was generated")
    _write_dataset_yaml(output_dir)
    (output_dir / "camera_calibration.json").write_text(
        json.dumps(calibration_dict, indent=2), encoding="utf-8"
    )
    config = {
        "scenario": args.scenario,
        "layouts": args.layouts,
        "steps": args.steps,
        "sample_every": args.sample_every,
        "dt": args.dt,
        "master_seed": args.seed,
        "layout_order": layout_order,
        "classes": list(CLASS_NAMES),
        "images": total_images,
        "empty_label_images": empty_label_images,
        "class_instance_counts": class_counts,
        "metadata": str(metadata_path),
        "dataset_yaml": str(output_dir / "dataset.yaml"),
        "camera_calibration": str(output_dir / "camera_calibration.json"),
        "segmentation_source": "genesis_scene.segmentation_idx_dict",
        "runtime_truth_access": False,
    }
    (output_dir / "generation_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config


def main() -> None:
    try:
        summary = run(parse_args())
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
    print("\nDataset summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
